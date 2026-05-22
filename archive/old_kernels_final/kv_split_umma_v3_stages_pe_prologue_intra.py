"""Intra-phase profiling for kv_split_umma_v3_stages_pe_prologue.

Trace layout (Chrome perfetto):
- pid = sm_id
- tid = bid * 4 + role
    role 0 → PRODUCER  (PE prologue + per-token pipeline + wait)
    role 1 → CONSUMER  (per-token wait + score + softmax + output)
    role 2 → SGEMM     (per-token score + softmax + output)
- Reduce kernel uses pid = 10000 + sm_id, tid = bid.
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils
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
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,)


PROBE_HEADER = 1
PROBE_ENTRY  = 4
# producer:  3 prologue + 8 tok × (1 outer + 4 stages×2 + 1 wait) = 3 + 80 = 83 → 96
MAX_ENTRIES_PROD   = 96
# consumer:  8 tok × (1 outer + wait + score + softmax + 4 out stages) = 64 → 80
MAX_ENTRIES_CONS   = 80
MAX_ENTRIES_SGEMM  = 48
MAX_ENTRIES_REDUCE = 16
PROBE_COLS_PROD   = PROBE_HEADER + MAX_ENTRIES_PROD   * PROBE_ENTRY
PROBE_COLS_CONS   = PROBE_HEADER + MAX_ENTRIES_CONS   * PROBE_ENTRY
PROBE_COLS_SGEMM  = PROBE_HEADER + MAX_ENTRIES_SGEMM  * PROBE_ENTRY
PROBE_COLS_REDUCE = PROBE_HEADER + MAX_ENTRIES_REDUCE * PROBE_ENTRY

TAGS_PROD = {
    "prologue_assign":  0,
    "prologue_pe_load": 2,
    "prologue_pe_umma": 4,
    "prod_token":       6,
    "prod_wait":        8,
    # per-stage chunk load/mma (chunk c ∈ 0..3)
    "ckv_load_s0":     10, "ckv_mma_s0": 12,
    "ckv_load_s1":     14, "ckv_mma_s1": 16,
    "ckv_load_s2":     18, "ckv_mma_s2": 20,
    "ckv_load_s3":     22, "ckv_mma_s3": 24,
}
TAGS_CONS = {
    "cons_token":   0,
    "cons_wait":    2,
    "cons_score":   4,
    "cons_softmax": 6,
    "cons_out_s0":  8,
    "cons_out_s1": 10,
    "cons_out_s2": 12,
    "cons_out_s3": 14,
}
TAGS_SGEMM = {
    "sgemm_token":   0,
    "sgemm_score":   2,
    "sgemm_softmax": 4,
    "sgemm_output":  6,
}
TAGS_REDUCE = {
    "reduce_total": 0,
}
TAG_NAMES_PROD   = {v: k for k, v in TAGS_PROD.items()}
TAG_NAMES_CONS   = {v: k for k, v in TAGS_CONS.items()}
TAG_NAMES_SGEMM  = {v: k for k, v in TAGS_SGEMM.items()}
TAG_NAMES_REDUCE = {v: k for k, v in TAGS_REDUCE.items()}

PHASE_ORDER_PROD   = [
    "prologue_assign", "prologue_pe_load", "prologue_pe_umma",
    "prod_token", "prod_wait",
    "ckv_load_s0", "ckv_mma_s0",
    "ckv_load_s1", "ckv_mma_s1",
    "ckv_load_s2", "ckv_mma_s2",
    "ckv_load_s3", "ckv_mma_s3",
]
PHASE_ORDER_CONS   = [
    "cons_token", "cons_wait", "cons_score", "cons_softmax",
    "cons_out_s0", "cons_out_s1", "cons_out_s2", "cons_out_s3",
]
PHASE_ORDER_SGEMM  = ["sgemm_token", "sgemm_score", "sgemm_softmax", "sgemm_output"]
PHASE_ORDER_REDUCE = ["reduce_total"]


def range_start(probe, row, cnt, sm_val, tag_val):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()

def range_stop(probe, row, cnt):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64() - probe[row, off + 2]

def range_finalize(probe, row, cnt):
    probe[row, 0] = cutlass.Int64(cnt)


# ── Dump helpers ──────────────────────────────────────────────────────────────
def _events(probe_cpu, num_blocks, tag_names, role, pid_offset=0):
    events = []
    base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (base is None or s < base): base = s
    base = base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id = int(data[off + 0])
            tag   = int(data[off + 1])
            t0    = int(data[off + 2])
            dur   = int(data[off + 3])
            if t0 == 0 and dur == 0: continue
            events.append(dict(
                name=tag_names.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0 - base) / 1000.0, dur=dur / 1000.0,
                pid=pid_offset + sm_id, tid=bid * 4 + role))
    return events, base


def _summary(probe_cpu, num_blocks, tag_names, phase_order, label):
    tag_totals = {}; tag_counts = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = tag_names.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    grand = sum(tag_totals.values()) or 1
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    print(f"{'Phase':>20s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>10s} {'%':>6s}")
    for name in phase_order:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>20s} {tot/1e6:>12.3f} {cnt_:>6d}"
                  f" {tot/cnt_/1000:>10.1f} {100*tot/grand:>5.1f}%")
    print(f"{'TOTAL':>20s} {grand/1e6:>12.3f}")


def dump_compute(probe_prod, probe_cons, probe_sgemm, num_blocks, num_splits):
    pp = probe_prod.cpu().contiguous().tolist()
    pc = probe_cons.cpu().contiguous().tolist()
    ps = probe_sgemm.cpu().contiguous().tolist()

    # Slowest producer block (by sum of prod_token outer durations)
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = pp[bid]; cnt = int(data[0])
        total = 0
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            if int(data[off + 1]) == TAGS_PROD["prod_token"]:
                total += int(data[off + 3])
        if total > max_dur:
            max_dur, max_bid = total, bid
    if max_dur > 0:
        data = pp[max_bid]; cnt = int(data[0])
        head_base = max_bid // num_splits; split_old = max_bid % num_splits
        print(f"\n--- Slowest PRODUCER block {max_bid} (head_base={head_base}, "
              f"split_old={split_old}, prod_token sum={max_dur/1000:.1f}µs): {cnt} entries ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES_PROD.get(tag, f"tag_{tag}")
            print(f"  sm={sm_id:>3} {name:>14s}  dur={dur:>10} ns  ({dur/1000:.2f} µs)")

    # Slowest consumer block
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = pc[bid]; cnt = int(data[0])
        total = 0
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            if int(data[off + 1]) == TAGS_CONS["cons_token"]:
                total += int(data[off + 3])
        if total > max_dur:
            max_dur, max_bid = total, bid
    if max_dur > 0:
        data = pc[max_bid]; cnt = int(data[0])
        head_base = max_bid // num_splits; split_old = max_bid % num_splits
        print(f"\n--- Slowest CONSUMER block {max_bid} (head_base={head_base}, "
              f"split_old={split_old}, cons_token sum={max_dur/1000:.1f}µs): {cnt} entries ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES_CONS.get(tag, f"tag_{tag}")
            print(f"  sm={sm_id:>3} {name:>14s}  dur={dur:>10} ns  ({dur/1000:.2f} µs)")

    # Slowest SGEMM block
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = ps[bid]; cnt = int(data[0])
        total = 0
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            if int(data[off + 1]) == TAGS_SGEMM["sgemm_token"]:
                total += int(data[off + 3])
        if total > max_dur:
            max_dur, max_bid = total, bid
    if max_dur > 0:
        data = ps[max_bid]; cnt = int(data[0])
        head_base = max_bid // num_splits; split_old = max_bid % num_splits
        print(f"\n--- Slowest SGEMM block {max_bid} (head_base={head_base}, "
              f"split_old={split_old}, sgemm_token sum={max_dur/1000:.1f}µs): {cnt} entries ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES_SGEMM.get(tag, f"tag_{tag}")
            print(f"  sm={sm_id:>3} {name:>14s}  dur={dur:>10} ns  ({dur/1000:.2f} µs)")

    _summary(pp, num_blocks, TAG_NAMES_PROD,  PHASE_ORDER_PROD,  "PRODUCER (PE prologue + UMMA pipeline)")
    _summary(pc, num_blocks, TAG_NAMES_CONS,  PHASE_ORDER_CONS,  "CONSUMER (tmem→softmax→output)")
    _summary(ps, num_blocks, TAG_NAMES_SGEMM, PHASE_ORDER_SGEMM, "SGEMM path (partial splits)")

    ep, bp = _events(pp, num_blocks, TAG_NAMES_PROD,  role=0)
    ec, bc = _events(pc, num_blocks, TAG_NAMES_CONS,  role=1)
    es, bs = _events(ps, num_blocks, TAG_NAMES_SGEMM, role=2)
    return ep, bp, ec, bc, es, bs


def dump_reduce(probe, num_blocks):
    probe_cpu = probe.cpu().contiguous().tolist()
    _summary(probe_cpu, num_blocks, TAG_NAMES_REDUCE, PHASE_ORDER_REDUCE, "Reduce kernel")
    events = []
    base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (base is None or s < base): base = s
    base = base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id = int(data[off + 0]); tag = int(data[off + 1])
            t0 = int(data[off + 2]); dur = int(data[off + 3])
            if t0 == 0 and dur == 0: continue
            events.append(dict(
                name=TAG_NAMES_REDUCE.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0 - base) / 1000.0, dur=dur / 1000.0,
                pid=10000 + sm_id, tid=bid))
    return events, base


def build_combined_trace(ep, bp, ec, bc, es, bs, er, br) -> str:
    bases = [b for b in [bp, bc, bs, br] if b]
    shared = min(bases) if bases else 0
    out = []
    for ev in ep: out.append(dict(ev, ts=ev["ts"] + (bp - shared) / 1000.0))
    for ev in ec: out.append(dict(ev, ts=ev["ts"] + (bc - shared) / 1000.0))
    for ev in es: out.append(dict(ev, ts=ev["ts"] + (bs - shared) / 1000.0))
    for ev in er: out.append(dict(ev, ts=ev["ts"] + (br - shared) / 1000.0))
    return json.dumps({"traceEvents": out})


# ── Constants (mirror kv_split_umma_v3_stages_pe_prologue.py) ─────────────────
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
DIM_SPLIT = (TOP_K + NUM_SPLITS - 1) // NUM_SPLITS
HEADS_PER_SPLIT = 2

_MMA_M, _MMA_N, _MMA_K = DIM_SPLIT, 8, 16
_MMA_K_PACK   = 4
_MMA_K_PACKED = _MMA_K * _MMA_K_PACK
_MMA_K_TILES  = HEAD_DIM_CKV // _MMA_K_PACKED
_MMA_K_TILES_FULL = _MMA_K_TILES

PANELS_PER_CHUNK: cutlass.Constexpr = 2
NUM_CKV_CHUNKS:   cutlass.Constexpr = _MMA_K_TILES // PANELS_PER_CHUNK
CHUNK_PACKED:     cutlass.Constexpr = _MMA_K_PACKED * PANELS_PER_CHUNK
CKV_KBLOCKS_PER_CHUNK: cutlass.Constexpr = _MMA_K_PACK * PANELS_PER_CHUNK
TMEM_COLS_PER_TOKEN = _MMA_N

PROLOGUE_BAR_ID    = 1
PROD_BAR_ID        = 2
CONS_BAR_ID        = 3
SGEMM_BAR_ID       = 4
UMMA_TOKEN_BAR_ID  = 5

# Per-stage tag tuples (indexed by constexpr chunk/stage index).
CKV_LOAD_TAGS = (TAGS_PROD["ckv_load_s0"], TAGS_PROD["ckv_load_s1"],
                 TAGS_PROD["ckv_load_s2"], TAGS_PROD["ckv_load_s3"])
CKV_MMA_TAGS  = (TAGS_PROD["ckv_mma_s0"],  TAGS_PROD["ckv_mma_s1"],
                 TAGS_PROD["ckv_mma_s2"],  TAGS_PROD["ckv_mma_s3"])
CONS_OUT_TAGS = (TAGS_CONS["cons_out_s0"], TAGS_CONS["cons_out_s1"],
                 TAGS_CONS["cons_out_s2"], TAGS_CONS["cons_out_s3"])


@cute.jit
def _panel_copy_layout(num_rows: int, k_packed: int, k_tiles: int):
    return cute.make_layout((num_rows, (k_packed, k_tiles)),
                            stride=(k_packed, (1, num_rows * k_packed)))


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
        self.out_vec = HEAD_DIM_CKV // (self.out_stages * self.wsize)

        self.umma_threads      = 512
        self.num_umma_warps    = self.umma_threads // self.wsize
        self.num_prod_warps    = 8
        self.num_cons_warps    = 8
        self.prod_threads      = self.num_prod_warps * self.wsize
        self.cons_threads      = self.num_cons_warps * self.wsize
        self.umma_inst         = (DIM_SPLIT, 8, 16)
        self.tmem_ld_rep       = HEADS_PER_SPLIT
        self.ab_dtype          = cutlass.BFloat16
        self.acc_dtype         = cutlass.Float32

        self.sgemm_threads   = 512
        self.num_sgemm_warps = self.sgemm_threads // self.wsize
        self.sgemm_ckv_vec   = 4
        self.sgemm_kpe_vec   = 2

        self.reduce_threads = 256
        self.reduce_warps   = self.reduce_threads // self.wsize
        self.vec_reduce     = 2

        self.partial_out = torch.zeros(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.float32, device="cuda")
        self.partial_lse = torch.zeros(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2,            dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
                 sm_scale: cutlass.Constexpr,
                 partial_out, partial_lse, output, lse,
                 probe_prod, probe_cons, probe_sgemm, probe_reduce, stream):
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
            score_mbars:      cute.struct.MemRange[cutlass.Int64, LIMIT_REQUEST]
            prologue_mbars:   cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.compute_kernel(
            tiled_mma, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
            partial_out, partial_lse, output, lse,
            probe_prod, probe_cons, probe_sgemm,
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
        partial_out, partial_lse, output, lse,
        probe_prod, probe_cons, probe_sgemm,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

        head_base_idx, split_idx_old, _ = cute.arch.block_idx()
        probe_row = head_base_idx * cutlass.Int32(NUM_SPLITS) + split_idx_old
        sm = cutlass.Int64(smid_u32())
        cnt_p = cutlass.Int32(0)   # producer probe cursor
        cnt_c = cutlass.Int32(0)   # consumer probe cursor
        cnt_s = cutlass.Int32(0)   # sgemm probe cursor

        # ========= SMEM setup =========
        alloc = cutlass.utils.SmemAllocator()

        smem_sp_indices = self._smem(alloc, cutlass.Int32,   (DIM_CHUNK, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_assign     = self._smem(alloc, cutlass.Int32,   (DIM_CHUNK, 2),         (2, 1))
        smem_score        = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_score_sgemm  = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_logits_flat       = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))
        smem_logits_flat_sgemm = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))

        smem_partial_umma = self._smem(alloc, cutlass.Float32,
            (self.num_cons_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
            (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))
        smem_partial_sgemm = self._smem(alloc, cutlass.Float32,
            (self.num_sgemm_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
            (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))

        swizzle = cute.make_swizzle(3, 4, 3)
        a_outer = cute.make_layout(
            ((_MMA_M, _MMA_K), 1, (_MMA_K_PACK, _MMA_K_TILES_FULL)),
            stride=((_MMA_K_PACKED, 1), 0, (_MMA_K, _MMA_M * _MMA_K_PACKED)))
        b_outer = cute.make_layout(
            ((_MMA_N, _MMA_K), 1, (_MMA_K_PACK, _MMA_K_TILES_FULL)),
            stride=((_MMA_K_PACKED, 1), 0, (_MMA_K, _MMA_N * _MMA_K_PACKED)))
        sA = alloc.allocate_tensor(cutlass.BFloat16, a_outer, byte_alignment=16, swizzle=swizzle)
        sB = alloc.allocate_tensor(cutlass.BFloat16, b_outer, byte_alignment=16, swizzle=swizzle)
        sA_ckv_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(_MMA_M, _MMA_K_PACKED, _MMA_K_TILES))
        sB_ckv_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(_MMA_N, _MMA_K_PACKED, _MMA_K_TILES))

        panel_stride_A: cutlass.Constexpr = _MMA_M * _MMA_K_PACKED
        panel_stride_B: cutlass.Constexpr = _MMA_N * _MMA_K_PACKED
        chunk_stride_A: cutlass.Constexpr = panel_stride_A * PANELS_PER_CHUNK
        chunk_stride_B: cutlass.Constexpr = panel_stride_B * PANELS_PER_CHUNK

        k_split_shape_chunk = cute.make_layout(((_MMA_K_PACKED, PANELS_PER_CHUNK),))
        k_split_shape_pe    = cute.make_layout(((_MMA_K_PACKED, 1),))

        atom_cpa_chunk   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=64)
        thr_layout       = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout_chunk = cute.make_layout(((4, 1),), stride=((1, 0),))
        tiled_copy_chunk = cute.make_tiled_copy_tv(atom_cpa_chunk, thr_layout, val_layout_chunk)
        lane_copy_chunk  = tiled_copy_chunk.get_slice(lane_idx)

        atom_cpa_pe   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        val_layout_pe = cute.make_layout(((2, 1),), stride=((1, 0),))
        tiled_copy_pe = cute.make_tiled_copy_tv(atom_cpa_pe, thr_layout, val_layout_pe)
        lane_copy_pe  = tiled_copy_pe.get_slice(lane_idx)

        # ---- 128-bit cp.async (16B per transaction) ----
        # CKV chunk row = 128 bf16 = 256 B → 16 threads/row, 8 vals each.
        atom_cpa_chunk128   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout_chunk128 = cute.make_layout(((16, 1),), stride=((1, 16),))
        val_layout_chunk128 = cute.make_layout(((8, 1),),  stride=((1, 0),))
        tiled_copy_chunk128 = cute.make_tiled_copy_tv(atom_cpa_chunk128, thr_layout_chunk128, val_layout_chunk128)
        lane_copy_chunk128  = tiled_copy_chunk128.get_slice(lane_idx % 16)

        # PE row (KPE = QPE = 64 bf16 = 128 B) → 8 threads/row, 8 vals each.
        atom_cpa_pe128   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout_pe128 = cute.make_layout(((8, 1),), stride=((1, 8),))
        val_layout_pe128 = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy_pe128 = cute.make_tiled_copy_tv(atom_cpa_pe128, thr_layout_pe128, val_layout_pe128)
        lane_copy_pe128  = tiled_copy_pe128.get_slice(lane_idx % 8)

        sA_ckv_out = cute.zipped_divide(sA_ckv_copy, (1, self.out_vec))

        storage             = alloc.allocate(self.shared_storage)
        score_mbar_base     = storage.score_mbars.data_ptr()
        prologue_mbar_base  = storage.prologue_mbars.data_ptr()

        head_base_idx, split_idx_old, _ = cute.arch.block_idx()
        T, _, _ = q_nope.shape

        # ── prologue_assign: time the work-assignment + mbarrier init phase ──
        if tidx == 0:
            range_start(probe_prod, probe_row, cnt_p, sm, TAGS_PROD["prologue_assign"])

        sparse_indices_  = cute.zipped_divide(sparse_indices, (1, self.sp_vec_size_i32))
        smem_sp_indices_ = cute.zipped_divide(smem_sp_indices, (1, self.sp_vec_size_i32))
        if DIM_CHUNK <= warp_idx < DIM_CHUNK + T:
            warp_idx_assign = warp_idx - DIM_CHUNK
            split_idx_new = (split_idx_old + warp_idx_assign * self.swz_rot_shift) % cutlass.Int32(NUM_SPLITS)
            split_vec_stride = DIM_SPLIT // self.sp_vec_size_i32
            si_vec = sparse_indices_[(0, None), (warp_idx_assign, split_idx_new * split_vec_stride + lane_idx)].load()
            num_valid_partial = 0
            for v in range(self.sp_vec_size_i32):
                val = si_vec[v]
                if 0 <= val < FLAT_CACHE:
                    num_valid_partial += 1
                else:
                    val = 0
                smem_sp_indices_[(0, v), (warp_idx_assign, lane_idx)] = val
            num_valid = warp_reduce(num_valid_partial, lambda a, b: a + b, width=self.wsize)
            if lane_idx == 0:
                smem_assign[warp_idx_assign, 0] = split_idx_new
                smem_assign[warp_idx_assign, 1] = num_valid

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C((_MMA_M, _MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        per_token_cols  = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(per_token_cols * LIMIT_REQUEST)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                for i in range(LIMIT_REQUEST):
                    cute.arch.mbarrier_init(score_mbar_base + i, cnt=1)
                # 8 warp_groups will each commit once; wait until all 8 retire.
                cute.arch.mbarrier_init(prologue_mbar_base, cnt=8)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if tidx == 0:
            range_stop(probe_prod, probe_row, cnt_p)
            cnt_p = cnt_p + cutlass.Int32(1)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)

        tCtAcc_base     = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        M_acc           = cute.size(tCtAcc_base, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler       = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi_base = cute.zipped_divide(tCtAcc_base, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, cutlass.Float32)
        cons_tidx       = tidx - self.prod_threads
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi_base[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(cons_tidx)
        tTR_tAcc_base   = tmem_thr_copy.partition_S(tCtAcc_epi_base)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc_base[None, None, 0].shape, cutlass.Float32)

        smem_score_              = cute.zipped_divide(smem_score,              (1, DIM_SPLIT // self.wsize))
        smem_score_sgemm_        = cute.zipped_divide(smem_score_sgemm,        (1, DIM_SPLIT // self.wsize))
        smem_logits_flat_        = cute.zipped_divide(smem_logits_flat,        (HEADS_PER_SPLIT,))
        smem_logits_flat_sgemm_  = cute.zipped_divide(smem_logits_flat_sgemm,  (HEADS_PER_SPLIT,))
        smem_partial_umma_  = cute.zipped_divide(smem_partial_umma,  (1, 1, self.out_vec))
        smem_partial_sgemm_ = cute.zipped_divide(smem_partial_sgemm, (1, 1, self.out_vec))
        ckv_flat_out        = cute.zipped_divide(ckv_flat,           (1, self.out_vec))
        q_nope_z   = cute.zipped_divide(q_nope,   (1, 1, self.sgemm_ckv_vec))
        q_pe_z     = cute.zipped_divide(q_pe,     (1, 1, self.sgemm_kpe_vec))
        ckv_flat_z = cute.zipped_divide(ckv_flat, (1, self.sgemm_ckv_vec))
        kpe_flat_z = cute.zipped_divide(kpe_flat, (1, self.sgemm_kpe_vec))

        is_producer = warp_idx < self.num_prod_warps
        is_consumer = (warp_idx >= self.num_prod_warps) & (warp_idx < self.num_umma_warps)

        # ============================================================
        # PE PROLOGUE — fully cooperative across all 32 warps (1024 thr).
        # 8 warp_groups × 4 warps each. warp_group g handles token g.
        # Per-token guard: only load+MMA when num_valid==DIM_SPLIT.
        # All 8 warp_groups commit prologue_mbar (cnt=8) unconditionally.
        # 128b cp.async: 8 thr/row. Within a warp_group (128 thr): 16 row groups
        # (= row groups per warp × 4 warps = 4 × 4) cover all 128 KPE rows in
        # one round (rows_per_group_pe = 1).
        # ============================================================
        warp_group_idx = warp_idx // 4         # 0..7
        lane_wg        = warp_idx %  4         # 0..3
        # Row-group within token's 4-warp pool: 4 warps × 4 row-groups/warp = 16.
        kpe_row_group_in_wg = lane_wg * 4 + (lane_idx // 8)   # 0..15
        kpe_rows_per_group: cutlass.Constexpr = _MMA_M // 16  # 8 rows/group/token

        if tidx == 0:
            range_start(probe_prod, probe_row, cnt_p, sm, TAGS_PROD["prologue_pe_load"])

        for i in cutlass.range_constexpr(LIMIT_REQUEST):
            if i < T:
                num_valid_pe = smem_assign[i, 1]
                if num_valid_pe == DIM_SPLIT:
                    sA_pe_i = cute.make_tensor(
                        sA.iterator + i * panel_stride_A,
                        _panel_copy_layout(_MMA_M, _MMA_K_PACKED, 1))
                    sB_pe_i = cute.make_tensor(
                        sB.iterator + i * panel_stride_B,
                        _panel_copy_layout(_MMA_N, _MMA_K_PACKED, 1))

                    if warp_group_idx == i:
                        # q_pe rows owned by lane_wg==0, lanes 0..15 (qpe_row in {0,1}).
                        if lane_wg == 0:
                            qpe_row = lane_idx // 8
                            if qpe_row < HEADS_PER_SPLIT:
                                head_h = head_base_idx * HEADS_PER_SPLIT + qpe_row
                                cute.copy(atom_cpa_pe128,
                                          lane_copy_pe128.partition_S(cute.composition(q_pe[i, head_h, None], k_split_shape_pe)),
                                          lane_copy_pe128.partition_D(sB_pe_i[qpe_row, None]))

                        # kpe rows: 16 row groups × 8 rows/group = 128 rows.
                        for r in range(kpe_rows_per_group):
                            row_idx  = r * 16 + kpe_row_group_in_wg
                            flat_row = smem_sp_indices[i, row_idx]
                            cute.copy(atom_cpa_pe128,
                                      lane_copy_pe128.partition_S(cute.composition(kpe_flat[flat_row, None], k_split_shape_pe)),
                                      lane_copy_pe128.partition_D(sA_pe_i[row_idx, None]))

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()   # 1024-thread fence

        if tidx == 0:
            range_stop(probe_prod, probe_row, cnt_p)
            range_start(probe_prod, probe_row, cnt_p + cutlass.Int32(1), sm, TAGS_PROD["prologue_pe_umma"])

        tcgen05_fence()
        # PE MMA: warp_group g fires for token g. Only the lead warp (lane_wg==0).
        if lane_wg == 0:
            for g in cutlass.range_constexpr(LIMIT_REQUEST):
                if warp_group_idx == g and g < T:
                    num_valid_g = smem_assign[g, 1]
                    if num_valid_g == DIM_SPLIT:
                        tCtAcc_g = cute.make_tensor(
                            tmem_ptr + cutlass.Int32(g * TMEM_COLS_PER_TOKEN),
                            tCtAcc_tmpl.layout)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in range(_MMA_K_PACK):
                            k_flat = g * _MMA_K_PACK + kb
                            coord  = (None, None, k_flat)
                            cute.gemm(tiled_mma, tCtAcc_g,
                                      tCrA[coord], tCrB[coord], tCtAcc_g)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            # Each warp_group lead lane commits once → 8 arrivals on cnt=8 mbar.
            if lane_idx == 0:
                tcgen05.commit(prologue_mbar_base)

        if tidx == 0:
            range_stop(probe_prod, probe_row, cnt_p + cutlass.Int32(1))
            cnt_p = cnt_p + cutlass.Int32(2)

        # ============================================================
        # PRODUCER (warps 0..7)
        # Per-token slot layout (10 entries when num_valid==DIM_SPLIT):
        #   +0=prod_token (outer)
        #   +1=ckv_load_s0 +2=ckv_mma_s0
        #   +3=ckv_load_s1 +4=ckv_mma_s1
        #   +5=ckv_load_s2 +6=ckv_mma_s2
        #   +7=ckv_load_s3 +8=ckv_mma_s3
        #   +9=prod_wait
        # ============================================================
        if is_producer:
            cute.arch.mbarrier_wait(prologue_mbar_base, cutlass.Int32(0))

            # 128b cp.async: 16 threads/row → 16 row groups across 256 prod threads.
            # row_group = warp_idx*2 + (lane_idx // 16). num_rounds = 128/16 = 8.
            prod_row_group = warp_idx * 2 + (lane_idx // 16)         # 0..15
            PROD_ROW_GROUPS: cutlass.Constexpr = 16
            num_rounds = DIM_SPLIT // PROD_ROW_GROUPS                 # 8

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    num_valid = smem_assign[T_idx, 1]
                    if num_valid == DIM_SPLIT:
                        if tidx == 0:
                            range_start(probe_prod, probe_row, cnt_p, sm, TAGS_PROD["prod_token"])

                        tCtAcc_i = cute.make_tensor(
                            tmem_ptr + cutlass.Int32(T_idx * TMEM_COLS_PER_TOKEN),
                            tCtAcc_tmpl.layout)

                        for c in cutlass.range_constexpr(NUM_CKV_CHUNKS):
                            if tidx == 0:
                                range_start(probe_prod, probe_row,
                                            cnt_p + cutlass.Int32(1 + 2 * c), sm,
                                            CKV_LOAD_TAGS[c])

                            sA_chunk = cute.make_tensor(
                                sA.iterator + c * chunk_stride_A,
                                _panel_copy_layout(_MMA_M, _MMA_K_PACKED, PANELS_PER_CHUNK))
                            sB_chunk = cute.make_tensor(
                                sB.iterator + c * chunk_stride_B,
                                _panel_copy_layout(_MMA_N, _MMA_K_PACKED, PANELS_PER_CHUNK))

                            if prod_row_group < HEADS_PER_SPLIT:
                                head_h = head_base_idx * HEADS_PER_SPLIT + prod_row_group
                                q_nope_chunk = cute.make_tensor(
                                    q_nope[T_idx, head_h, None].iterator + c * CHUNK_PACKED,
                                    cute.make_layout((CHUNK_PACKED,), stride=(1,)))
                                cute.copy(atom_cpa_chunk128,
                                          lane_copy_chunk128.partition_S(cute.composition(q_nope_chunk, k_split_shape_chunk)),
                                          lane_copy_chunk128.partition_D(sB_chunk[prod_row_group, None]))

                            for round_idx in range(num_rounds):
                                row_idx  = round_idx * PROD_ROW_GROUPS + prod_row_group
                                flat_row = smem_sp_indices[T_idx, row_idx]
                                ckv_chunk = cute.make_tensor(
                                    ckv_flat[flat_row, None].iterator + c * CHUNK_PACKED,
                                    cute.make_layout((CHUNK_PACKED,), stride=(1,)))
                                cute.copy(atom_cpa_chunk128,
                                          lane_copy_chunk128.partition_S(cute.composition(ckv_chunk, k_split_shape_chunk)),
                                          lane_copy_chunk128.partition_D(sA_chunk[row_idx, None]))

                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.fence_view_async_shared()
                            cute.arch.barrier(barrier_id=PROD_BAR_ID, number_of_threads=self.prod_threads)

                            if tidx == 0:
                                range_stop(probe_prod, probe_row, cnt_p + cutlass.Int32(1 + 2 * c))
                                range_start(probe_prod, probe_row,
                                            cnt_p + cutlass.Int32(2 + 2 * c), sm,
                                            CKV_MMA_TAGS[c])

                            tcgen05_fence()
                            if warp_idx == 0:
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                for kb in range(CKV_KBLOCKS_PER_CHUNK):
                                    k_flat = c * CKV_KBLOCKS_PER_CHUNK + kb
                                    coord  = (None, None, k_flat)
                                    cute.gemm(tiled_mma, tCtAcc_i,
                                              tCrA[coord], tCrB[coord], tCtAcc_i)

                            if tidx == 0:
                                range_stop(probe_prod, probe_row, cnt_p + cutlass.Int32(2 + 2 * c))

                        if warp_idx == 0 and lane_idx == 0:
                            tcgen05.commit(score_mbar_base + T_idx)

                        if tidx == 0:
                            range_start(probe_prod, probe_row, cnt_p + cutlass.Int32(9), sm, TAGS_PROD["prod_wait"])

                        cute.arch.mbarrier_wait(score_mbar_base + T_idx,
                                                cutlass.Int32(0))

                        if tidx == 0:
                            range_stop(probe_prod, probe_row, cnt_p + cutlass.Int32(9))
                            range_stop(probe_prod, probe_row, cnt_p)
                            cnt_p = cnt_p + cutlass.Int32(10)

                cute.arch.barrier(barrier_id=UMMA_TOKEN_BAR_ID,
                                  number_of_threads=self.umma_threads)

        # ============================================================
        # CONSUMER (warps 8..15)
        # Per iter slots: cnt_c=outer(token), +1=wait, +2=score, +3=softmax, +4=output
        # cons_tidx==0 → tidx==prod_threads (256)
        # ============================================================
        if is_consumer:
            cons_warp_idx = warp_idx - self.num_prod_warps

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]

                    if num_valid == DIM_SPLIT:
                        if cons_tidx == 0:
                            range_start(probe_cons, probe_row, cnt_c, sm, TAGS_CONS["cons_token"])
                            range_start(probe_cons, probe_row, cnt_c + cutlass.Int32(1), sm, TAGS_CONS["cons_wait"])

                        cute.arch.mbarrier_wait(score_mbar_base + T_idx, cutlass.Int32(0))

                        if cons_tidx == 0:
                            range_stop(probe_cons, probe_row, cnt_c + cutlass.Int32(1))
                            range_start(probe_cons, probe_row, cnt_c + cutlass.Int32(2), sm, TAGS_CONS["cons_score"])

                        tTR_tAcc_i = cute.make_tensor(
                            tTR_tAcc_base.iterator + cutlass.Int32(T_idx * TMEM_COLS_PER_TOKEN),
                            tTR_tAcc_base.layout)

                        if cons_tidx < DIM_SPLIT:
                            cute.copy(tmem_tiled_copy, tTR_tAcc_i[None, None, 0], tTR_rAcc)
                            smem_score[0, cons_tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                            smem_score[1, cons_tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

                        cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)

                        if cons_tidx == 0:
                            range_stop(probe_cons, probe_row, cnt_c + cutlass.Int32(2))
                            range_start(probe_cons, probe_row, cnt_c + cutlass.Int32(3), sm, TAGS_CONS["cons_softmax"])

                        if cons_warp_idx < HEADS_PER_SPLIT:
                            num_elems: cutlass.Constexpr = DIM_SPLIT // self.wsize
                            head_idx_global = head_base_idx * HEADS_PER_SPLIT + cons_warp_idx
                            vec = smem_score_[(0, None), (cons_warp_idx, lane_idx)].load()
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
                                smem_logits_flat[col_idx * HEADS_PER_SPLIT + cons_warp_idx] = vec_masked[v_idx]
                            if lane_idx == 0:
                                partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                                partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum

                        cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)

                        if cons_tidx == 0:
                            range_stop(probe_cons, probe_row, cnt_c + cutlass.Int32(3))

                        num_rounds_out: cutlass.Constexpr = DIM_SPLIT // self.num_cons_warps
                        out0 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        for stage_idx in cutlass.range_constexpr(self.out_stages):
                            if cons_tidx == 0:
                                range_start(probe_cons, probe_row,
                                            cnt_c + cutlass.Int32(4 + stage_idx), sm,
                                            CONS_OUT_TAGS[stage_idx])
                            out0.fill(cutlass.Float32(0))
                            out1.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out):
                                k = round_idx * self.num_cons_warps + cons_warp_idx
                                if k < num_valid:
                                    gmem_ckv_vec = sA_ckv_out[(0, None), (k, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0[v_idx], out1[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0[v_idx], out1[v_idx]))
                            smem_partial_umma_[(0, 0, None), (cons_warp_idx, 0, lane_idx)].store(out0.load())
                            smem_partial_umma_[(0, 0, None), (cons_warp_idx, 1, lane_idx)].store(out1.load())
                            cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)
                            thr_group_idx  = cons_tidx // DIM_SPLIT
                            thr_group_lane = cons_tidx %  DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_cons_warps):
                                    final_sum += smem_partial_umma[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum
                            cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)
                            if cons_tidx == 0:
                                range_stop(probe_cons, probe_row,
                                           cnt_c + cutlass.Int32(4 + stage_idx))

                        if cons_tidx == 0:
                            range_stop(probe_cons, probe_row, cnt_c)
                            cnt_c = cnt_c + cutlass.Int32(8)

                cute.arch.barrier(barrier_id=UMMA_TOKEN_BAR_ID,
                                  number_of_threads=self.umma_threads)

        # ============================================================
        # SGEMM (warps 16..31)
        # ============================================================
        if warp_idx >= self.num_umma_warps:
            sgemm_warp_idx = warp_idx - self.num_umma_warps
            sgemm_tidx     = tidx - self.umma_threads

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]

                    if 0 < num_valid < DIM_SPLIT:
                        head_idx0 = head_base_idx * HEADS_PER_SPLIT
                        head_idx1 = head_base_idx * HEADS_PER_SPLIT + 1

                        if sgemm_tidx == 0:
                            range_start(probe_sgemm, probe_row, cnt_s, sm, TAGS_SGEMM["sgemm_token"])
                            range_start(probe_sgemm, probe_row, cnt_s + cutlass.Int32(1), sm, TAGS_SGEMM["sgemm_score"])

                        num_rounds_score = (num_valid + self.num_sgemm_warps - 1) // self.num_sgemm_warps
                        for round_idx in range(num_rounds_score):
                            col_idx = round_idx * self.num_sgemm_warps + sgemm_warp_idx
                            if col_idx < num_valid:
                                flat_cache_idx = smem_sp_indices[T_idx, col_idx]
                                acc0 = cutlass.Float32(0)
                                acc1 = cutlass.Float32(0)
                                for i in range(HEAD_DIM_CKV // (self.sgemm_ckv_vec * self.wsize)):
                                    row_idx = i * self.wsize + lane_idx
                                    qn0_frag = q_nope_z[(0, 0, None), (T_idx, head_idx0, row_idx)].load().to(cutlass.Float32)
                                    qn1_frag = q_nope_z[(0, 0, None), (T_idx, head_idx1, row_idx)].load().to(cutlass.Float32)
                                    ckv_frag = ckv_flat_z[(0, None), (flat_cache_idx, row_idx)].load().to(cutlass.Float32)
                                    for v in range(self.sgemm_ckv_vec):
                                        acc0, acc1 = cute.arch.fma_packed_f32x2(
                                            (qn0_frag[v], qn1_frag[v]),
                                            (ckv_frag[v], ckv_frag[v]),
                                            (acc0, acc1))
                                for i in range(HEAD_DIM_KPE // (self.sgemm_kpe_vec * self.wsize)):
                                    row_idx = i * self.wsize + lane_idx
                                    qp0_frag = q_pe_z[(0, 0, None), (T_idx, head_idx0, row_idx)].load().to(cutlass.Float32)
                                    qp1_frag = q_pe_z[(0, 0, None), (T_idx, head_idx1, row_idx)].load().to(cutlass.Float32)
                                    kpe_frag = kpe_flat_z[(0, None), (flat_cache_idx, row_idx)].load().to(cutlass.Float32)
                                    for v in range(self.sgemm_kpe_vec):
                                        acc0, acc1 = cute.arch.fma_packed_f32x2(
                                            (qp0_frag[v], qp1_frag[v]),
                                            (kpe_frag[v], kpe_frag[v]),
                                            (acc0, acc1))
                                acc0 = warp_reduce(acc0, lambda a, b: a + b)
                                acc1 = warp_reduce(acc1, lambda a, b: a + b)
                                if lane_idx == 0:
                                    smem_score_sgemm[0, col_idx] = acc0 * cutlass.Float32(sm_scale)
                                    smem_score_sgemm[1, col_idx] = acc1 * cutlass.Float32(sm_scale)

                        cute.arch.barrier(barrier_id=SGEMM_BAR_ID, number_of_threads=self.sgemm_threads)

                        if sgemm_tidx == 0:
                            range_stop(probe_sgemm, probe_row, cnt_s + cutlass.Int32(1))
                            range_start(probe_sgemm, probe_row, cnt_s + cutlass.Int32(2), sm, TAGS_SGEMM["sgemm_softmax"])

                        if sgemm_warp_idx < HEADS_PER_SPLIT:
                            num_elems: cutlass.Constexpr = DIM_SPLIT // self.wsize
                            head_idx_global = head_base_idx * HEADS_PER_SPLIT + sgemm_warp_idx
                            vec = smem_score_sgemm_[(0, None), (sgemm_warp_idx, lane_idx)].load()
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
                                smem_logits_flat_sgemm[col_idx * HEADS_PER_SPLIT + sgemm_warp_idx] = vec_masked[v_idx]
                            if lane_idx == 0:
                                partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                                partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum

                        cute.arch.barrier(barrier_id=SGEMM_BAR_ID, number_of_threads=self.sgemm_threads)

                        if sgemm_tidx == 0:
                            range_stop(probe_sgemm, probe_row, cnt_s + cutlass.Int32(2))
                            range_start(probe_sgemm, probe_row, cnt_s + cutlass.Int32(3), sm, TAGS_SGEMM["sgemm_output"])

                        num_rounds_out_s = (num_valid + self.num_sgemm_warps - 1) // self.num_sgemm_warps
                        out0_s = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1_s = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)

                        for stage_idx in range(self.out_stages):
                            out0_s.fill(cutlass.Float32(0))
                            out1_s.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out_s):
                                k = round_idx * self.num_sgemm_warps + sgemm_warp_idx
                                if k < num_valid:
                                    flat_cache_idx = smem_sp_indices[T_idx, k]
                                    gmem_ckv_vec = ckv_flat_out[(0, None), (flat_cache_idx, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_sgemm_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0_s[v_idx], out1_s[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0_s[v_idx], out1_s[v_idx]))

                            smem_partial_sgemm_[(0, 0, None), (sgemm_warp_idx, 0, lane_idx)].store(out0_s.load())
                            smem_partial_sgemm_[(0, 0, None), (sgemm_warp_idx, 1, lane_idx)].store(out1_s.load())

                            cute.arch.barrier(barrier_id=SGEMM_BAR_ID, number_of_threads=self.sgemm_threads)

                            thr_group_idx  = sgemm_tidx // DIM_SPLIT
                            thr_group_lane = sgemm_tidx %  DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_sgemm_warps):
                                    final_sum += smem_partial_sgemm[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum

                            cute.arch.barrier(barrier_id=SGEMM_BAR_ID, number_of_threads=self.sgemm_threads)

                        if sgemm_tidx == 0:
                            range_stop(probe_sgemm, probe_row, cnt_s + cutlass.Int32(3))
                            range_stop(probe_sgemm, probe_row, cnt_s)
                            cnt_s = cnt_s + cutlass.Int32(4)

        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        if tidx == 0:
            range_finalize(probe_prod, probe_row, cnt_p)
        if tidx == self.prod_threads:
            range_finalize(probe_cons, probe_row, cnt_c)
        if tidx == self.umma_threads:
            range_finalize(probe_sgemm, probe_row, cnt_s)

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
        cnt_r = cutlass.Int32(0)
        if tidx == 0:
            range_start(probe_reduce, probe_row, cnt_r, sm, TAGS_REDUCE["reduce_total"])

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
            range_stop(probe_reduce, probe_row, cnt_r)
            range_finalize(probe_reduce, probe_row, cnt_r + cutlass.Int32(1))


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
    Bc = (NUM_HEADS // HEADS_PER_SPLIT) * NUM_SPLITS
    Br = LIMIT_REQUEST * NUM_HEADS
    probe_prod    = _fake(cute.Int64, (Bc, PROBE_COLS_PROD),    (1, 0), 8)
    probe_cons    = _fake(cute.Int64, (Bc, PROBE_COLS_CONS),    (1, 0), 8)
    probe_sgemm   = _fake(cute.Int64, (Bc, PROBE_COLS_SGEMM),   (1, 0), 8)
    probe_reduce  = _fake(cute.Int64, (Br, PROBE_COLS_REDUCE),  (1, 0), 8)
    stream        = make_fake_stream(use_tvm_ffi_env_stream=True)

    hybrid = Dsa()
    compiled = cute.compile(
        hybrid,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        probe_prod, probe_cons, probe_sgemm, probe_reduce, stream,
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
    probe_prod   = torch.zeros((Bc, PROBE_COLS_PROD),   dtype=torch.int64, device="cuda")
    probe_cons   = torch.zeros((Bc, PROBE_COLS_CONS),   dtype=torch.int64, device="cuda")
    probe_sgemm  = torch.zeros((Bc, PROBE_COLS_SGEMM),  dtype=torch.int64, device="cuda")
    probe_reduce = torch.zeros((LIMIT_REQUEST * NUM_HEADS, PROBE_COLS_REDUCE), dtype=torch.int64, device="cuda")

    for _ in range(3):
        output_t.zero_(); lse_t.fill_(-float("inf"))
        probe_prod.zero_(); probe_cons.zero_(); probe_sgemm.zero_(); probe_reduce.zero_()
        _compiled(q_nope, q_pe, ckv, kpe, si,
                  _hybrid.partial_out, _hybrid.partial_lse,
                  output_t, lse_t, probe_prod, probe_cons, probe_sgemm, probe_reduce)
        torch.cuda.synchronize()

    probe_prod.zero_(); probe_cons.zero_(); probe_sgemm.zero_(); probe_reduce.zero_()
    output_t.zero_(); lse_t.fill_(-float("inf"))
    _compiled(q_nope, q_pe, ckv, kpe, si,
              _hybrid.partial_out, _hybrid.partial_lse,
              output_t, lse_t, probe_prod, probe_cons, probe_sgemm, probe_reduce)
    torch.cuda.synchronize()

    ep, bp, ec, bc, es, bs = dump_compute(probe_prod, probe_cons, probe_sgemm, Bc, NUM_SPLITS)
    er, br = dump_reduce(probe_reduce, Br)
    return build_combined_trace(ep, bp, ec, bc, es, bs, er, br)
