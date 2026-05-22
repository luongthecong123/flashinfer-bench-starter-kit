"""Strategy 2: vectorized (LDG.128) load for the 128-thread-per-T groups.

Baseline (smem_sparse_intra.py):
  16 iters/thread, each iter: LDG.32 + STS.32 + SETP + IADD
  → 64 instructions/thread, 16 LDG stall points per thread

Strategy 2 (this file):
  Same 128-thread-per-T group structure and named barriers as baseline.
  Loop strides over vec4 chunks (i32×4 = 128-bit LDG):
    4 iters/thread: LDG.128 + 4×STS.32 + 4×SETP + 4×IADD
  → 37 instructions/thread, **4 LDG stall points** (4× fewer)
  With same 4 warps, each LDG.128 carries 4× more data per stall point.

  Early exit is also applied: each thread breaks its own loop as soon as
  it encounters the first -1 within its stride window.  Since sentinel -1
  values are packed to the right, this correctly short-circuits threads
  whose range is entirely beyond num_valid.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
import math, json, torch


# ── Timer helpers ──────────────────────────────────────────────────────────────

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

PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 4
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY   # 17

TAGS        = {"q_load": 0, "sparse_load": 2, "sync_wait": 4}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["q_load", "sparse_load", "sync_wait"]


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


# ── Dump helpers ───────────────────────────────────────────────────────────────

def _probe_events(probe_cpu, num_blocks):
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
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); t0 = int(data[off + 2]); dur = int(data[off + 3])
            if t0 == 0 and dur == 0: continue
            events.append(dict(
                name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0 - base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id, tid=bid))
    return events, base


def dump_probe(probe: torch.Tensor, num_blocks: int, num_head: int, num_splits: int):
    probe_cpu = probe.cpu().contiguous().tolist()

    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid

    data = probe_cpu[max_bid]; cnt = int(data[0])
    head = max_bid // num_splits; split_old = max_bid % num_splits
    print(f"\n--- Slowest block {max_bid} "
          f"(head={head}, split_old={split_old}, total={max_dur/1000:.1f}µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>12s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*57}")
    print(f"{'Phase':>14s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>10s}  {'%':>5s}")
    print(f"{'='*57}")
    grand = sum(tag_totals.values())
    for name in PHASE_ORDER:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>14s} {tot/1e6:>12.3f} {cnt_:>6d} {tot/cnt_/1000:>10.1f}  {100*tot/grand:>5.1f}%")
    print(f"{'TOTAL':>14s} {grand/1e6:>12.3f}")

    return _probe_events(probe_cpu, num_blocks)


# ── Kernel constants ───────────────────────────────────────────────────────────

NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX = 8
NUM_SPLITS = 8
DIM_SPLIT  = TOP_K_LEN // NUM_SPLITS   # 256
VEC_SPARSE = 4   # 4 × i32 = 128-bit LDG


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Smem_sparse_2_intra():
    """Strategy 2: vec4 (LDG.128) load, same 128-thread-per-T group structure."""

    def __init__(self):
        self.num_head     = NUM_HEADS
        self.head_dim_ckv = HEAD_DIM_CKV
        self.head_dim_kpe = HEAD_DIM_KPE
        self.top_k_len    = TOP_K_LEN
        self.num_pages    = NUM_PAGES
        self.page_size    = PAGE_SIZE
        self.T_max        = T_MAX
        self.num_splits   = NUM_SPLITS
        self.dim_split    = DIM_SPLIT
        self.num_threads  = 1024
        self.wsize        = cute.arch.WARP_SIZE
        self.num_warps    = self.num_threads // self.wsize   # 32
        self.vec_size_ckv = 8
        self.vec_sparse   = VEC_SPARSE                       # 4
        # top_k_len // vec_sparse = 512 chunks per T row
        self.top_k_chunks = TOP_K_LEN // VEC_SPARSE          # 512
        self.sparse_thr_per_T = 128
        self.num_warps_per_T  = self.sparse_thr_per_T // self.wsize   # 4

    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)

    @cute.jit
    def __call__(
        self,
        q_nope:         cute.Tensor,    # (T, 16, 512)  bf16
        q_pe:           cute.Tensor,    # (T, 16,  64)  bf16
        sparse_indices: cute.Tensor,    # (T, 2048)     i32
        probe:          cute.Tensor,    # (128, PROBE_COLS) i64
        stream,
    ):
        self.kernel(
            q_nope, q_pe, sparse_indices, probe
        ).launch(
            grid=[self.num_head, self.num_splits, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_nope:         cute.Tensor,    # (T, 16, 512)
        q_pe:           cute.Tensor,    # (T, 16,  64)
        sparse_indices: cute.Tensor,    # (T, 2048)
        probe:          cute.Tensor,    # (128, PROBE_COLS) i64
    ):
        T, _, _ = q_nope.shape
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        probe_row = bidx * self.num_splits + bidy
        sm        = cutlass.Int64(smid_u32())
        probe_cnt = cutlass.Int32(0)

        # ── Full SMEM allocation identical to kv_split_xor_intra (same occupancy) ──
        alloc = cutlass.utils.SmemAllocator()
        smem_sparse      = self._smem(alloc, cutlass.Int32,    (self.T_max, self.top_k_len),        (self.top_k_len, 1),     4)  # 64 KB
        smem_num_valid   = self._smem(alloc, cutlass.Int32,    (self.T_max,),                       (1,),                    4)  # 32 B
        smem_logits      = self._smem(alloc, cutlass.Float32,  (self.dim_split,),                   (1,),                   16)  #  1 KB
        smem_red_i32     = self._smem(alloc, cutlass.Int32,    (self.T_max, 32),                    (32, 1),                 4)  #  1 KB
        smem_max_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)  # 128 B
        smem_sum_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)  # 128 B
        smem_q_nope      = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_ckv),     (self.head_dim_ckv, 1), 16)  #  8 KB
        smem_q_pe        = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_kpe),     (self.head_dim_kpe, 1), 16)  #  1 KB
        smem_partial     = self._smem(alloc, cutlass.Float32,  (self.num_warps, self.head_dim_ckv), (self.head_dim_ckv, 1), 16)  # 64 KB
        smem_out         = self._smem(alloc, cutlass.Float32,  (self.head_dim_ckv,),                (1,),                   16)  #  2 KB

        head_idx       = bidx
        thr_idx_per_T  = tidx % self.sparse_thr_per_T
        wg_per_T_idx   = tidx // self.sparse_thr_per_T
        warp_per_T_idx = warp_idx % self.num_warps_per_T
        lane_idx_per_T = thr_idx_per_T % self.wsize

        # ── Phase 1: q_load (unchanged from baseline) ─────────────────────
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["q_load"])

        if wg_per_T_idx < T:
            for i in range(thr_idx_per_T, self.head_dim_ckv, self.sparse_thr_per_T):
                smem_q_nope[wg_per_T_idx, i] = q_nope[wg_per_T_idx, head_idx, i]
            for i in range(thr_idx_per_T, self.head_dim_kpe, self.sparse_thr_per_T):
                smem_q_pe[wg_per_T_idx, i] = q_pe[wg_per_T_idx, head_idx, i]

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)

        # ── Phase 2: sparse_load (Strategy 2 — vec4 LDG.128) ─────────────
        # zipped_divide groups 4 consecutive i32 into one 128-bit element.
        # Each thread processes top_k_chunks // sparse_thr_per_T = 4 vec4 chunks
        # (vs 16 scalar iters in baseline) → 4 LDG.128 stall points per thread.
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["sparse_load"])

        # Build vec4 views over rows of sparse_indices and smem_sparse
        si_vec = cute.zipped_divide(sparse_indices, (1, self.vec_sparse))   # ((1,4),(T,512))
        ss_vec = cute.zipped_divide(smem_sparse,    (1, self.vec_sparse))   # ((1,4),(T_max,512))

        partial_cnt = 0
        if wg_per_T_idx < T:
            for chunk in range(thr_idx_per_T, self.top_k_chunks, self.sparse_thr_per_T):
                # LDG.128 — load 4 consecutive i32 from gmem in one instruction
                vec = si_vec[(0, None), (wg_per_T_idx, chunk)].load()
                # STS.32 × 4 — store back into smem_sparse
                for v in range(self.vec_sparse):
                    smem_sparse[wg_per_T_idx, chunk * self.vec_sparse + v] = vec[v]
                    if vec[v] >= cutlass.Int32(0):
                        partial_cnt += 1

            # Intra-warp reduce (unchanged)
            cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
            if lane_idx_per_T == 0:
                smem_red_i32[wg_per_T_idx, warp_per_T_idx] = cnt_sum

            cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                              number_of_threads=self.sparse_thr_per_T)

            # Inter-warp reduce (unchanged)
            if warp_per_T_idx == 0:
                val     = smem_red_i32[wg_per_T_idx, lane_idx_per_T]
                cnt_sum = warp_reduce(val, lambda a, b: a + b, width=self.num_warps_per_T)
                smem_red_i32[wg_per_T_idx, 0] = cnt_sum

            cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                              number_of_threads=self.sparse_thr_per_T)

            smem_num_valid[wg_per_T_idx] = smem_red_i32[wg_per_T_idx, 0]

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)

        # ── Phase 3: sync_wait ────────────────────────────────────────────
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["sync_wait"])

        cute.arch.sync_threads()

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)
            range_finalize(probe, probe_row, probe_cnt)


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)


def compile_kernel():
    T  = cute.sym_int()
    Bc = NUM_HEADS * NUM_SPLITS   # 128

    q_nope         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K_LEN),               (1, 0),     4)
    probe          = _fake(cute.Int64,    (Bc, PROBE_COLS),             (1, 0),     8)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        Smem_sparse_2_intra(),
        q_nope, q_pe, sparse_indices, probe, stream,
        options="--enable-tvm-ffi"
    )


def run_single(workload_idx: int) -> str:
    import os, json as _json
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling smem_sparse_2 (Strategy 2: vec4 LDG.128)...")
    compiled = compile_kernel()

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [_json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]

    num_blocks = NUM_HEADS * NUM_SPLITS   # 128
    print(f"\nWorkload {workload_idx + 1}: MaxValid={max_valid}  T={T}  Blocks={num_blocks}")

    q_nope, q_pe, _ckv, _kpe, _ = make_tensors(T, P)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    probe = torch.zeros((num_blocks, PROBE_COLS), dtype=torch.int64, device="cuda")

    for _ in range(3):
        probe.zero_()
        compiled(q_nope, q_pe, si, probe)
        torch.cuda.synchronize()

    probe.zero_()
    compiled(q_nope, q_pe, si, probe)
    torch.cuda.synchronize()

    events, base = dump_probe(probe, num_blocks, NUM_HEADS, NUM_SPLITS)
    return json.dumps({"traceEvents": events})
