"""Standalone kernel profiling ONLY the upfront smem_sparse phase.

Reproduces the full SMEM allocation and thread layout of kv_split_xor_intra so
occupancy is identical.  Executes only the upfront work, then exits.

Sub-probes (thread-0 perspective):
  q_load:      gmem→smem for q_nope + q_pe  (~9 KB BF16 per block)
  sparse_load: gmem→smem for sparse_indices + intra-T count/reduce  (~64 KB per block)
  sync_wait:   final sync_threads — straggler overhead across T-groups

Grid:  [num_head=16, num_splits=8, 1]  =  128 blocks
Block: [1024, 1, 1]
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
MAX_ENTRIES  = 4                                      # q_load, sparse_load, sync_wait + 1 spare
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY   # 17

TAGS       = {"q_load": 0, "sparse_load": 2, "sync_wait": 4}
TAG_NAMES  = {v: k for k, v in TAGS.items()}
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
            tag  = int(data[off + 1])
            t0   = int(data[off + 2])
            dur  = int(data[off + 3])
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
    head      = max_bid // num_splits
    split_old = max_bid % num_splits
    print(f"\n--- Slowest block {max_bid} "
          f"(head={head}, split_old={split_old}, total={max_dur/1000:.1f}µs): {cnt} entries ---")
    for i in range(cnt):
        off   = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>12s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off  = PROBE_HEADER + i * PROBE_ENTRY
            tag  = int(data[off + 1]); dur = int(data[off + 3])
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


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Smem_sparse_intra():
    """Runs ONLY the upfront smem_sparse phase of kv_split_xor, with sub-probes."""

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
        self.sparse_thr_per_T  = 128
        self.num_warps_per_T   = self.sparse_thr_per_T // self.wsize   # 4

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

        # ── Full SMEM allocation matching kv_split_xor_intra (same occupancy) ──
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

        # ── Phase 1: q_load ─────────────────────────────────────────────────
        # Measures thread-0 time to load its T=0 slice of q_nope + q_pe.
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["q_load"])

        if wg_per_T_idx < T:
            for i in range(thr_idx_per_T, self.head_dim_ckv, self.sparse_thr_per_T):
                smem_q_nope[wg_per_T_idx, i] = q_nope[wg_per_T_idx, head_idx, i]
            for i in range(thr_idx_per_T, self.head_dim_kpe, self.sparse_thr_per_T):
                smem_q_pe[wg_per_T_idx, i] = q_pe[wg_per_T_idx, head_idx, i]

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)

        # ── Phase 2: sparse_load ─────────────────────────────────────────────
        # Inlined count_valid_indices for the T=0 group (thread-0 perspective):
        #   load sparse_indices → smem_sparse, count valid,
        #   per-T barriers + inter-warp reduce → smem_num_valid.
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["sparse_load"])

        partial_cnt = 0
        if wg_per_T_idx < T:
            for i in range(thr_idx_per_T, self.top_k_len, self.sparse_thr_per_T):
                idx = sparse_indices[wg_per_T_idx, i]
                smem_sparse[wg_per_T_idx, i] = idx
                if idx >= cutlass.Int32(0):
                    partial_cnt += 1

            cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
            if lane_idx_per_T == 0:
                smem_red_i32[wg_per_T_idx, warp_per_T_idx] = cnt_sum

            cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                              number_of_threads=self.sparse_thr_per_T)

            if warp_per_T_idx == 0:
                val     = smem_red_i32[wg_per_T_idx, lane_idx_per_T]
                cnt_sum = warp_reduce(val, lambda a, b: a + b, width=self.num_warps_per_T)
                smem_red_i32[wg_per_T_idx, 0] = cnt_sum

            cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                              number_of_threads=self.sparse_thr_per_T)

            smem_num_valid[wg_per_T_idx] = smem_red_i32[wg_per_T_idx, 0]

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)

        # ── Phase 3: sync_wait ───────────────────────────────────────────────
        # Time blocking at block-wide barrier waiting for all 8 T-groups.
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
        Smem_sparse_intra(),
        q_nope, q_pe, sparse_indices, probe, stream,
        options="--enable-tvm-ffi"
    )


def run_single(workload_idx: int) -> str:
    import os, json as _json
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    H = NUM_HEADS
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling smem_sparse_intra kernel...")
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

    # Warmup
    for _ in range(3):
        probe.zero_()
        compiled(q_nope, q_pe, si, probe)
        torch.cuda.synchronize()

    # Profile run
    probe.zero_()
    compiled(q_nope, q_pe, si, probe)
    torch.cuda.synchronize()

    events, base = dump_probe(probe, num_blocks, NUM_HEADS, NUM_SPLITS)
    return json.dumps({"traceEvents": events})
