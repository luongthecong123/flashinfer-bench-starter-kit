"""Intra-phase profiling for kv_split_umma_v2.

Instruments the compute kernel with timer probes at EXISTING sync boundaries.
NO new syncs introduced — every range_start/range_stop is placed immediately
adjacent to a barrier/mbarrier_wait that already exists.

Per-iter (T_idx) phase boundaries inside the umma if-block:
  load    : cp.async issue → cp.async_wait_group(0) → barrier  (ends here)
  mma     : tcgen05 gemm    → mbarrier_wait                    (ends here)
  score   : tmem → smem_score → barrier                        (ends here)
  softmax : max + exp + sum + scatter + lse-write → barrier    (ends here)
  output  : 4 stage GEMV + reduce + write partial_out          (ends at last barrier)

Plus one "upfront" phase covering SMEM setup, prologue sparse load,
tmem/mbarrier alloc through to the sync_threads before the umma loop.

Reduce kernel: single "reduce_total" phase (one block per (T, head)).
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
import math, json, torch


# ── Timer helpers ─────────────────────────────────────────────────────────────
@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(MLIR_T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int64(t)

@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(MLIR_T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int32(t)

@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


PROBE_HEADER = 1
PROBE_ENTRY  = 4
# Compute: 1 upfront + LIMIT_REQUEST(8) * 5 phases = 41 → 48
MAX_ENTRIES_COMPUTE = 48
MAX_ENTRIES_REDUCE  = 4
PROBE_COLS_COMPUTE  = PROBE_HEADER + MAX_ENTRIES_COMPUTE * PROBE_ENTRY  # 193
PROBE_COLS_REDUCE   = PROBE_HEADER + MAX_ENTRIES_REDUCE  * PROBE_ENTRY  # 17

TAGS_COMPUTE = {
    "upfront": 0,
    "load":    2,
    "mma":     4,
    "score":   6,
    "softmax": 8,
    "output":  10,
}
TAG_NAMES_COMPUTE = {v: k for k, v in TAGS_COMPUTE.items()}
PHASE_ORDER_COMPUTE = ["upfront", "load", "mma", "score", "softmax", "output"]

TAGS_REDUCE = {"reduce_total": 0}
TAG_NAMES_REDUCE = {v: k for k, v in TAGS_REDUCE.items()}
PHASE_ORDER_REDUCE = ["reduce_total"]


def range_start(probe, row, cnt, sm_val, tag_val):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()

def range_stop(probe, row, cnt):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64() - probe[row, off + 2]
    return cnt + cutlass.Int32(1)

def range_finalize(probe, row, cnt):
    probe[row, 0] = cutlass.Int64(cnt)


# ── Dump helpers ──────────────────────────────────────────────────────────────
def _probe_events(probe_cpu, num_blocks, tag_names, pid_offset=0):
    events = []
    base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (base is None or s < base):
                base = s
    base = base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        if cnt == 0: continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off  = PROBE_HEADER + i * PROBE_ENTRY
            tag  = int(data[off + 1])
            t0   = int(data[off + 2])
            dur  = int(data[off + 3])
            if t0 == 0 and dur == 0: continue
            events.append(dict(
                name=tag_names.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0 - base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id + pid_offset, tid=bid))
    return events, base


def dump_compute(probe: torch.Tensor, num_blocks: int, num_splits: int):
    probe_cpu = probe.cpu().contiguous().tolist()

    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid

    data = probe_cpu[max_bid]; cnt = int(data[0])
    head_base = max_bid // num_splits
    split_old = max_bid %  num_splits
    print(f"\n--- Compute: Slowest block {max_bid} "
          f"(head_base={head_base}, split_old={split_old}, total={max_dur/1000:.1f}µs): "
          f"{cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES_COMPUTE.get(tag, f'tag_{tag}'):>10s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off  = PROBE_HEADER + i * PROBE_ENTRY
            tag  = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES_COMPUTE.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*65}")
    print(f"{'Phase (all blocks)':>20s} {'Total (ms)':>12s} {'Count':>6s}"
          f" {'Avg (µs)':>10s} {'%':>6s}")
    print(f"{'='*65}")
    grand = sum(tag_totals.values()) or 1
    for name in PHASE_ORDER_COMPUTE:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>20s} {tot/1e6:>12.3f} {cnt_:>6d}"
                  f" {tot/cnt_/1000:>10.1f} {100*tot/grand:>5.1f}%")
    print(f"{'TOTAL':>20s} {grand/1e6:>12.3f}")

    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_COMPUTE, pid_offset=0)


def dump_reduce(probe: torch.Tensor, num_blocks: int, num_heads: int):
    probe_cpu = probe.cpu().contiguous().tolist()
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid

    data = probe_cpu[max_bid]; cnt = int(data[0])
    T_slow = max_bid // num_heads
    H_slow = max_bid %  num_heads
    print(f"\n--- Reduce: Slowest block {max_bid} (T={T_slow}, head={H_slow}, "
          f"total={max_dur/1000:.1f}µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES_REDUCE.get(tag, f'tag_{tag}'):>14s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off  = PROBE_HEADER + i * PROBE_ENTRY
            tag  = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES_REDUCE.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*52}")
    for name in PHASE_ORDER_REDUCE:
        if name in tag_totals:
            n = tag_counts[name]
            print(f"  {name:>14}: avg={tag_totals[name]/n/1000:.1f}µs  "
                  f"total={tag_totals[name]/1e6:.3f}ms  n={n}")

    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_REDUCE, pid_offset=200)


def build_combined_trace(compute_events, compute_base, reduce_events, reduce_base) -> str:
    bases = [b for b in [compute_base, reduce_base] if b]
    shared_base = min(bases) if bases else 0
    all_events = []
    for ev in compute_events:
        all_events.append(dict(ev, ts=ev["ts"] + (compute_base - shared_base) / 1000.0))
    for ev in reduce_events:
        all_events.append(dict(ev, ts=ev["ts"] + (reduce_base - shared_base) / 1000.0))
    return json.dumps({"traceEvents": all_events})


# ── Kernel constants (mirror umma_v2) ────────────────────────────────────────
NUM_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
TOP_K = 2048
NUM_PAGES = 8462
PAGE_SIZE = 64
FLAT_CACHE = NUM_PAGES * PAGE_SIZE
LN2 = 0.6931471805599453
SM_SCALE: cutlass.Constexpr = 0.1352337788608801
LIMIT_REQUEST = 8
DIM_CHUNK = 8
NUM_SPLITS = 16
DIM_SPLIT = (TOP_K + NUM_SPLITS - 1) // NUM_SPLITS  # 128
HEADS_PER_SPLIT = 2


@cute.jit
def _panel_copy_layout(num_rows: int, k_packed: int, k_tiles: int):
    return cute.make_layout((num_rows, (k_packed, k_tiles)),
                            stride=(k_packed, (1, num_rows * k_packed)),)


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Dsa():
    def __init__(self):
        self.wsize = cute.arch.WARP_SIZE
        self.swz_rot_shift = 7
        self.sp_vec_size_i32 = 4
        self.out_stages = 4
        self.out_vec = HEAD_DIM_CKV // (self.out_stages * self.wsize)  # 4

        self.umma_threads = 256
        self.num_umma_warps = self.umma_threads // self.wsize
        self.umma_inst = (DIM_SPLIT, 8, 16)  # Opt4: N=8
        self.tmem_ld_rep = self.umma_inst[1]
        self.ab_dtype  = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.sgemm_threads = DIM_CHUNK * self.wsize
        self.umma_bar_id = 2
        self.umma_max_red_bar_id = 3

        self.reduce_threads = 256
        self.reduce_warps = self.reduce_threads // self.wsize
        self.vec_reduce = 2

        self.partial_out = torch.empty(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.float32, device="cuda")
        self.partial_lse = torch.empty(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2, dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
                 sm_scale: cutlass.Constexpr,
                 partial_out, partial_lse, output, lse,
                 probe_compute, probe_reduce, stream):
        T, _, _ = q_nope.shape
        ckv_flat = cute.make_tensor(ckv_cache.iterator,
            cute.make_layout((FLAT_CACHE, HEAD_DIM_CKV), stride=(HEAD_DIM_CKV, 1)))
        kpe_flat = cute.make_tensor(kpe_cache.iterator,
            cute.make_layout((FLAT_CACHE, HEAD_DIM_KPE), stride=(HEAD_DIM_KPE, 1)))

        op = tcgen05.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.umma_inst,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        @cute.struct
        class SharedStorage:
            umma_mbar_ptr:    cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.compute_kernel(
            tiled_mma, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
            partial_out, partial_lse, output, lse, probe_compute,
        ).launch(grid=[NUM_HEADS // HEADS_PER_SPLIT, NUM_SPLITS, 1],
                 block=[self.umma_threads + self.sgemm_threads, 1, 1], stream=stream)

        self.reduce_kernel(
            sparse_indices, partial_out, partial_lse, output, lse, probe_reduce,
        ).launch(grid=[T, NUM_HEADS, 1],
                 block=[self.reduce_threads, 1, 1], stream=stream)

    @staticmethod
    def _smem(allocator, dtype, shape, stride, byte_alignment=16, swizzle=None):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), byte_alignment, swizzle)

    @cute.kernel
    def compute_kernel(
        self, tiled_mma, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, probe_compute,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

        head_base_idx, split_idx_old, _ = cute.arch.block_idx()
        probe_row = head_base_idx * cutlass.Int32(NUM_SPLITS) + split_idx_old
        sm = cutlass.Int64(smid_u32())
        probe_cnt = cutlass.Int32(0)

        # ── Probe: open "upfront" right at kernel entry (tidx==0 only) ──
        if tidx == 0:
            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["upfront"])

        # ========= SMEM setup =========
        alloc = cutlass.utils.SmemAllocator()
        smem_sp_indices = self._smem(alloc, cutlass.Int32, (DIM_CHUNK, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_assign = self._smem(alloc, cutlass.Int32, (DIM_CHUNK, 2), (2, 1))
        smem_score = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_max          = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT,), (1,))
        smem_sum          = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT,), (1,))
        smem_logits_flat  = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))
        smem_partial_umma = self._smem(alloc, cutlass.Float32,
                              (self.num_umma_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
                              (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))

        swizzle    = cute.make_swizzle(3, 4, 3)
        _MK_PACK   = 4
        _MK_PACKED = 64
        _MK_TILES     = HEAD_DIM_CKV // _MK_PACKED   # 8
        _MK_TILES_PE  = HEAD_DIM_KPE  // _MK_PACKED  # 1
        _MK_TILES_FULL = _MK_TILES + _MK_TILES_PE    # 9
        _MMA_M = DIM_SPLIT
        _MMA_N = 8
        _MMA_K = 16
        _MMA_M_PACK, _MMA_N_PACK = 1, 1
        a_outer = cute.make_layout(
            ((_MMA_M, _MMA_K), _MMA_M_PACK, (_MK_PACK, _MK_TILES_FULL)),
            stride=((_MK_PACKED, 1), 0, (_MMA_K, _MMA_M * _MK_PACKED)))
        b_outer = cute.make_layout(
            ((_MMA_N, _MMA_K), _MMA_N_PACK, (_MK_PACK, _MK_TILES_FULL)),
            stride=((_MK_PACKED, 1), 0, (_MMA_K, _MMA_N * _MK_PACKED)))
        sA = alloc.allocate_tensor(cutlass.BFloat16, a_outer, byte_alignment=16, swizzle=swizzle)
        sB = alloc.allocate_tensor(cutlass.BFloat16, b_outer, byte_alignment=16, swizzle=swizzle)
        sA_ckv_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(_MMA_M, _MK_PACKED, _MK_TILES))
        sB_ckv_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(_MMA_N, _MK_PACKED, _MK_TILES))
        panel_stride_A = _MMA_M * _MK_PACKED * _MK_TILES
        panel_stride_B = _MMA_N * _MK_PACKED * _MK_TILES
        sA_kpe_copy = cute.make_tensor(sA.iterator + panel_stride_A, _panel_copy_layout(_MMA_M, _MK_PACKED, _MK_TILES_PE))
        sB_kpe_copy = cute.make_tensor(sB.iterator + panel_stride_B, _panel_copy_layout(_MMA_N, _MK_PACKED, _MK_TILES_PE))
        k_split_shape    = cute.make_layout(((_MK_PACKED, _MK_TILES),))
        k_split_shape_pe = cute.make_layout(((_MK_PACKED, _MK_TILES_PE),))
        atom_cpa   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy = cute.make_tiled_copy_tv(atom_cpa, thr_layout, val_layout)
        lane_copy  = tiled_copy.get_slice(lane_idx)
        atom_cpa_pe   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        val_layout_pe = cute.make_layout(((2, 1),), stride=((1, 0),))
        tiled_copy_pe = cute.make_tiled_copy_tv(atom_cpa_pe, thr_layout, val_layout_pe)
        lane_copy_pe  = tiled_copy_pe.get_slice(lane_idx)
        storage  = alloc.allocate(self.shared_storage)
        mma_mbar = storage.umma_mbar_ptr.data_ptr()

        T, _, _ = q_nope.shape

        # ── Prologue: sparse load (gated to sgemm warps) ──
        sparse_indices_  = cute.zipped_divide(sparse_indices, (1, self.sp_vec_size_i32))
        smem_sp_indices_ = cute.zipped_divide(smem_sp_indices, (1, self.sp_vec_size_i32))
        if DIM_CHUNK <= warp_idx < DIM_CHUNK + T:
            warp_idx_sgemm = warp_idx - DIM_CHUNK
            split_idx_new = (split_idx_old + warp_idx_sgemm * self.swz_rot_shift) % cutlass.Int32(NUM_SPLITS)
            split_vec_stride = DIM_SPLIT // self.sp_vec_size_i32
            si_vec = sparse_indices_[(0, None), (warp_idx_sgemm, split_idx_new * split_vec_stride + lane_idx)].load()
            num_valid_partial = 0
            for v in range(self.sp_vec_size_i32):
                val = si_vec[v]
                if 0 <= val < FLAT_CACHE:
                    num_valid_partial += 1
                else:
                    val = 0
                smem_sp_indices_[(0, v), (warp_idx_sgemm, lane_idx)] = val
            num_valid = warp_reduce(num_valid_partial, lambda a, b: a + b, width=self.wsize)
            if lane_idx == 0:
                smem_assign[warp_idx_sgemm, 0] = split_idx_new
                smem_assign[warp_idx_sgemm, 1] = num_valid

        cute.arch.sync_threads()

        # ── tmem / mbarrier setup ──
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C((_MMA_M, _MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)
        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # ── Probe: close "upfront" right after the second sync_threads ──
        if tidx == 0:
            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        M_acc           = cute.size(tCtAcc, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler       = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, cutlass.Float32)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, cutlass.Float32)

        smem_score_        = cute.zipped_divide(smem_score,        (1, DIM_SPLIT // self.wsize))
        smem_logits_flat_  = cute.zipped_divide(smem_logits_flat,  (HEADS_PER_SPLIT,))
        smem_partial_umma_ = cute.zipped_divide(smem_partial_umma, (1, 1, self.out_vec))
        ckv_flat_out       = cute.zipped_divide(ckv_flat,          (1, self.out_vec))
        sA_ckv_out         = cute.zipped_divide(sA_ckv_copy,       (1, self.out_vec))

        if warp_idx < self.num_umma_warps:
            umma_warp_idx = warp_idx
            umma_tidx     = tidx
            num_rounds    = DIM_SPLIT // self.num_umma_warps
            mma_phase = cutlass.Int32(0)

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]

                    if num_valid > 0:
                        # ── Probe: open "load" ──
                        if tidx == 0:
                            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["load"])

                        # Per-iter Q load: 2 heads → sB rows 0,1
                        if umma_warp_idx < HEADS_PER_SPLIT:
                            head_h = head_base_idx * HEADS_PER_SPLIT + umma_warp_idx
                            cute.copy(atom_cpa,
                                      lane_copy.partition_S(cute.composition(q_nope[T_idx, head_h, None], k_split_shape)),
                                      lane_copy.partition_D(sB_ckv_copy[umma_warp_idx, None]))
                            cute.copy(atom_cpa_pe,
                                      lane_copy_pe.partition_S(cute.composition(q_pe[T_idx, head_h, None], k_split_shape_pe)),
                                      lane_copy_pe.partition_D(sB_kpe_copy[umma_warp_idx, None]))

                        for round_idx in range(num_rounds):
                            sp_idx  = round_idx * self.num_umma_warps + umma_warp_idx
                            row_idx = smem_sp_indices[T_idx, sp_idx]
                            cute.copy(atom_cpa,
                                    lane_copy.partition_S(cute.composition(ckv_flat[row_idx, None], k_split_shape)),
                                    lane_copy.partition_D(sA_ckv_copy[sp_idx, None]))
                            cute.copy(atom_cpa_pe,
                                    lane_copy_pe.partition_S(cute.composition(kpe_flat[row_idx, None], k_split_shape_pe)),
                                    lane_copy_pe.partition_D(sA_kpe_copy[sp_idx, None]))

                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        cute.arch.barrier(barrier_id=self.umma_bar_id,
                                          number_of_threads=self.umma_threads)

                        # ── Probe: close "load", open "mma" ──
                        if tidx == 0:
                            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)
                            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["mma"])

                        tcgen05_fence()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        if umma_warp_idx == 0:
                            num_k_blocks = cute.size(tCrA, mode=[2])
                            for k_block_idx in range(num_k_blocks):
                                k_block_coord = (None, None, k_block_idx)
                                cute.gemm(tiled_mma, tCtAcc,
                                        tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            if umma_tidx == 0:
                                tcgen05.commit(mma_mbar)
                        cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                        mma_phase = mma_phase ^ cutlass.Int32(1)

                        # ── Probe: close "mma", open "score" ──
                        if tidx == 0:
                            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)
                            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["score"])

                        if tidx < DIM_SPLIT:
                            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
                            smem_score[0, tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                            smem_score[1, tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

                        cute.arch.barrier(barrier_id=self.umma_bar_id,
                                          number_of_threads=self.umma_threads)

                        # ── Probe: close "score", open "softmax" ──
                        if tidx == 0:
                            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)
                            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["softmax"])

                        if umma_warp_idx < HEADS_PER_SPLIT:
                            num_elems: cutlass.Constexpr = DIM_SPLIT // self.wsize
                            head_idx_global = head_base_idx * HEADS_PER_SPLIT + umma_warp_idx
                            vec = smem_score_[(0, None), (umma_warp_idx, lane_idx)].load()
                            vec_masked = cute.make_rmem_tensor(
                                cute.make_layout((num_elems,), stride=(1,)), cutlass.Float32)
                            for v_idx in range(num_elems):
                                vec_masked[v_idx] = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                if col_idx < num_valid:
                                    vec_masked[v_idx] = vec[v_idx]
                            row_max = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                row_max = cute.arch.fmax(row_max, vec_masked[v_idx])
                            row_max = warp_reduce(row_max, cute.arch.fmax)
                            row_sum = cutlass.Float32(0)
                            for v_idx in range(num_elems):
                                e = cute.math.exp(vec_masked[v_idx] - row_max)
                                vec_masked[v_idx] = e
                                row_sum += e
                            row_sum = warp_reduce(row_sum, lambda a, b: a + b)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                smem_logits_flat[col_idx * HEADS_PER_SPLIT + umma_warp_idx] = vec_masked[v_idx]
                            if lane_idx == 0:
                                partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                                partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum

                        cute.arch.barrier(barrier_id=self.umma_bar_id,
                                          number_of_threads=self.umma_threads)

                        # ── Probe: close "softmax", open "output" ──
                        if tidx == 0:
                            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)
                            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["output"])

                        num_rounds_out: cutlass.Constexpr = DIM_SPLIT // self.num_umma_warps
                        out0 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        for stage_idx in range(self.out_stages):
                            out0.fill(cutlass.Float32(0))
                            out1.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out):
                                k = round_idx * self.num_umma_warps + umma_warp_idx
                                if k < num_valid:
                                    gmem_ckv_vec = sA_ckv_out[(0, None), (k, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0[v_idx], out1[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0[v_idx], out1[v_idx]))
                            smem_partial_umma_[(0, 0, None), (umma_warp_idx, 0, lane_idx)].store(out0.load())
                            smem_partial_umma_[(0, 0, None), (umma_warp_idx, 1, lane_idx)].store(out1.load())
                            cute.arch.barrier(barrier_id=self.umma_bar_id, number_of_threads=self.umma_threads)
                            thr_group_idx  = tidx // DIM_SPLIT
                            thr_group_lane = tidx % DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_umma_warps):
                                    final_sum += smem_partial_umma[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum
                            cute.arch.barrier(barrier_id=self.umma_bar_id, number_of_threads=self.umma_threads)

                        # ── Probe: close "output" right after the last barrier ──
                        if tidx == 0:
                            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        if tidx == 0:
            range_finalize(probe_compute, probe_row, probe_cnt)

    @cute.kernel
    def reduce_kernel(
        self, sparse_indices, partial_out, partial_lse, output, lse, probe_reduce,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        T_idx, head_idx, _ = cute.arch.block_idx()

        probe_row = T_idx * cutlass.Int32(NUM_HEADS) + head_idx
        sm = cutlass.Int64(smid_u32())
        probe_cnt = cutlass.Int32(0)

        if tidx == 0:
            range_start(probe_reduce, probe_row, probe_cnt, sm, TAGS_REDUCE["reduce_total"])

        alloc = cutlass.utils.SmemAllocator()
        smem_red_i32 = self._smem(alloc, cutlass.Int32,   (32,),          (1,))
        smem_max_sum = self._smem(alloc, cutlass.Float32, (NUM_SPLITS, 2), (2, 1))

        partial_cnt = cutlass.Int32(0)
        for i in range(tidx, TOP_K, self.reduce_threads):
            idx = sparse_indices[T_idx, i]
            if idx >= cutlass.Int32(0):
                partial_cnt += cutlass.Int32(1)

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt_sum
        cute.arch.sync_threads()

        if warp_idx == 0:
            val = cutlass.Int32(0)
            if lane_idx < self.reduce_warps:
                val = smem_red_i32[lane_idx]
            val = warp_reduce(val, lambda a, b: a + b, width=self.reduce_warps)
            if lane_idx == 0:
                smem_red_i32[0] = val
        cute.arch.sync_threads()

        num_valid = smem_red_i32[0]
        num_active_splits = (num_valid + DIM_SPLIT - 1) // DIM_SPLIT

        if tidx < num_active_splits:
            smem_max_sum[tidx, 0] = partial_lse[T_idx, tidx, head_idx, 0]
            smem_max_sum[tidx, 1] = partial_lse[T_idx, tidx, head_idx, 1]
        cute.arch.sync_threads()

        partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, self.vec_reduce))
        output_v      = cute.zipped_divide(output,      (1, 1, self.vec_reduce))

        g_max = -cutlass.Float32(math.inf)
        for s in range(num_active_splits):
            local_max = smem_max_sum[s, 0]
            if local_max > g_max:
                g_max = local_max

        g_lse_sum = cutlass.Float32(0)
        acc_rmem = cute.make_rmem_tensor(cute.make_layout((self.vec_reduce,), stride=(1,)), cutlass.Float32)
        acc_rmem[0] = cutlass.Float32(0)
        acc_rmem[1] = cutlass.Float32(0)
        acc = acc_rmem.load()

        for s in range(num_active_splits):
            l_max = smem_max_sum[s, 0]
            l_sum = smem_max_sum[s, 1]
            scale = cute.math.exp(l_max - g_max)
            g_lse_sum += l_sum * scale
            a = partial_out_v[(0, 0, 0, None), (T_idx, s, head_idx, tidx)].load()
            acc = acc + scale * a

        if tidx == 0:
            lse[T_idx, head_idx] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)

        output_v[(0, 0, None), (T_idx, head_idx, tidx)].store((acc / g_lse_sum).to(cutlass.BFloat16))

        if tidx == 0:
            probe_cnt = range_stop(probe_reduce, probe_row, probe_cnt)
            range_finalize(probe_reduce, probe_row, probe_cnt)


def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kernel():
    T = cute.sym_int()
    q_nope         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K), (1, 0), 4)
    sm_scale       = SM_SCALE
    partial_out    = _fake(cute.Float32,  (LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, NUM_HEADS), (1, 0), 4)
    Bc = (NUM_HEADS // HEADS_PER_SPLIT) * NUM_SPLITS  # 8 * 16 = 128
    Br = LIMIT_REQUEST * NUM_HEADS                    # 8 * 16 = 128
    probe_compute  = _fake(cute.Int64,    (Bc, PROBE_COLS_COMPUTE), (1, 0), 8)
    probe_reduce   = _fake(cute.Int64,    (Br, PROBE_COLS_REDUCE),  (1, 0), 8)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    hybrid = Dsa()
    compiled = cute.compile(
        hybrid,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        probe_compute, probe_reduce, stream,
        options="--enable-tvm-ffi"
    )
    return hybrid, compiled


_hybrid, _compiled = compile_kernel()


def run_single(workload_idx: int) -> str:
    import os, json as _json
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    H, D_ckv = NUM_HEADS, HEAD_DIM_CKV
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [_json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]

    Bc = (NUM_HEADS // HEADS_PER_SPLIT) * NUM_SPLITS
    Br = T * NUM_HEADS
    print(f"\nWorkload {workload_idx + 1}: MaxValid={max_valid}  T={T}  "
          f"ComputeBlocks={Bc}  ReduceBlocks={Br}")

    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    output_t = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse_t    = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
    probe_compute = torch.zeros((Bc, PROBE_COLS_COMPUTE), dtype=torch.int64, device="cuda")
    probe_reduce  = torch.zeros((LIMIT_REQUEST * NUM_HEADS, PROBE_COLS_REDUCE), dtype=torch.int64, device="cuda")

    # Warmup
    for _ in range(3):
        output_t.zero_(); lse_t.fill_(-float("inf"))
        probe_compute.zero_(); probe_reduce.zero_()
        _compiled(q_nope, q_pe, ckv, kpe, si,
                  _hybrid.partial_out, _hybrid.partial_lse,
                  output_t, lse_t, probe_compute, probe_reduce)
        torch.cuda.synchronize()

    # Profile run
    probe_compute.zero_(); probe_reduce.zero_()
    output_t.zero_(); lse_t.fill_(-float("inf"))
    _compiled(q_nope, q_pe, ckv, kpe, si,
              _hybrid.partial_out, _hybrid.partial_lse,
              output_t, lse_t, probe_compute, probe_reduce)

    torch.cuda.synchronize()

    compute_events, compute_base = dump_compute(probe_compute, Bc, NUM_SPLITS)
    reduce_events,  reduce_base  = dump_reduce(probe_reduce, Br, NUM_HEADS)
    return build_combined_trace(compute_events, compute_base, reduce_events, reduce_base)
