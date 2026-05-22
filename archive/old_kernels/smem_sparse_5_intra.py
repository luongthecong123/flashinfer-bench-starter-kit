"""Strategy 5: cp.async gmem→smem for q_nope / q_pe, overlapped with sparse_load.

Baseline (S2): q_load(scalar,gmem→reg→smem) 1.0µs + sparse_load(vec4) 1.5µs = 2.6µs total
Strategy 5: fire cp.async for q_nope+q_pe (non-blocking) → commit_group →
            run sparse_load (S2 vec4) → cp_async_wait_group(0) → sync_threads.

cp.async bypasses registers entirely (gmem → smem directly), so the q data can
arrive in smem while sparse_load is in progress.  If the q transfer finishes
within 1.5µs of being issued, cp_async_wait costs ~0 and the total upfront window
collapses to just sparse_load ≈ 1.5µs.

Probes (top-to-bottom measurement):
  q_issue        — time to issue all cp.async + commit_group (fire-and-forget, ~50-100ns)
  sparse_load    — vec4 LDG.128 sparse_indices + warp-reduce (unchanged from S2, ~1.5µs)
  cp_async_wait  — stall at wait_group(0) (ideally ~0 if q overlapped; positive if not)

Grid:   [num_head=16, num_splits=8, 1] = 128 blocks
Block:  [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
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

TAGS        = {"q_issue": 0, "sparse_load": 2, "cp_async_wait": 4}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["q_issue", "sparse_load", "cp_async_wait"]


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
        print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>15s}"
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

    print(f"\n{'='*60}")
    print(f"{'Phase':>15s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>10s}  {'%':>5s}")
    print(f"{'='*60}")
    grand = sum(tag_totals.values())
    for name in PHASE_ORDER:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>15s} {tot/1e6:>12.3f} {cnt_:>6d} {tot/cnt_/1000:>10.1f}  {100*tot/grand:>5.1f}%")
    print(f"{'TOTAL':>15s} {grand/1e6:>12.3f}")

    return _probe_events(probe_cpu, num_blocks)


# ── Kernel constants ───────────────────────────────────────────────────────────

NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX = 8
NUM_SPLITS = 8
DIM_SPLIT  = TOP_K_LEN // NUM_SPLITS   # 256
VEC_SPARSE = 4    # 4 × i32 = 128-bit LDG for sparse_indices
VEC_Q      = 8    # 8 × BF16 = 128-bit cp.async for q_nope / q_pe


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Smem_sparse_5_intra():
    """Strategy 5: cp.async for q_nope/q_pe overlapped with vec4 sparse_load."""

    def __init__(self):
        self.num_head       = NUM_HEADS
        self.head_dim_ckv   = HEAD_DIM_CKV
        self.head_dim_kpe   = HEAD_DIM_KPE
        self.top_k_len      = TOP_K_LEN
        self.num_pages      = NUM_PAGES
        self.page_size      = PAGE_SIZE
        self.T_max          = T_MAX
        self.num_splits     = NUM_SPLITS
        self.dim_split      = DIM_SPLIT
        self.num_threads    = 1024
        self.wsize          = cute.arch.WARP_SIZE
        self.num_warps      = self.num_threads // self.wsize   # 32
        self.vec_q          = VEC_Q                             # 8 BF16 = 128-bit cp.async
        self.vec_sparse     = VEC_SPARSE                        # 4 i32  = 128-bit LDG
        self.q_nope_chunks  = HEAD_DIM_CKV // VEC_Q             # 64 chunks of 8 BF16 per T row
        self.q_pe_chunks    = HEAD_DIM_KPE // VEC_Q             # 8  chunks of 8 BF16 per T row
        self.top_k_chunks   = TOP_K_LEN   // VEC_SPARSE         # 512 chunks of 4 i32  per T row
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
        q_nope:         cute.Tensor,    # (T, 16, 512)  bf16
        q_pe:           cute.Tensor,    # (T, 16,  64)  bf16
        sparse_indices: cute.Tensor,    # (T, 2048)     i32
        probe:          cute.Tensor,    # (128, PROBE_COLS) i64
    ):
        T, _, _ = q_nope.shape
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _    = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        probe_row = bidx * self.num_splits + bidy
        sm        = cutlass.Int64(smid_u32())
        probe_cnt = cutlass.Int32(0)

        head_idx       = bidx
        thr_idx_per_T  = tidx % self.sparse_thr_per_T
        wg_per_T_idx   = tidx // self.sparse_thr_per_T
        warp_per_T_idx = warp_idx % self.num_warps_per_T
        lane_idx_per_T = thr_idx_per_T % self.wsize

        # ── Full SMEM allocation identical to kv_split_xor_intra ──────────────
        alloc            = cutlass.utils.SmemAllocator()
        smem_sparse      = self._smem(alloc, cutlass.Int32,    (self.T_max, self.top_k_len),        (self.top_k_len, 1),     4)   # 64 KB
        smem_num_valid   = self._smem(alloc, cutlass.Int32,    (self.T_max,),                       (1,),                    4)   # 32 B
        smem_logits      = self._smem(alloc, cutlass.Float32,  (self.dim_split,),                   (1,),                   16)   #  1 KB
        smem_red_i32     = self._smem(alloc, cutlass.Int32,    (self.T_max, 32),                    (32, 1),                 4)   #  1 KB
        smem_max_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)   # 128 B
        smem_sum_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)   # 128 B
        smem_q_nope      = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_ckv),     (self.head_dim_ckv, 1), 16)   #  8 KB
        smem_q_pe        = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_kpe),     (self.head_dim_kpe, 1), 16)   #  1 KB
        smem_partial     = self._smem(alloc, cutlass.Float32,  (self.num_warps, self.head_dim_ckv), (self.head_dim_ckv, 1), 16)   # 64 KB
        smem_out         = self._smem(alloc, cutlass.Float32,  (self.head_dim_ckv,),                (1,),                   16)   #  2 KB

        # ── cp.async copy atom: 8 × BF16 = 128 bits per transfer ──────────────
        copy_atom_q = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )

        # ── Vec views for cp.async ─────────────────────────────────────────────
        # q_nope (T, 16, 512) → group last dim by 8: ((1,1,8),(T,16,64))
        # q_pe   (T, 16,  64) → group last dim by 8: ((1,1,8),(T,16, 8))
        # smem_q_nope (T_max, 512) → group last dim by 8: ((1,8),(T_max,64))
        # smem_q_pe   (T_max,  64) → group last dim by 8: ((1,8),(T_max, 8))
        q_nope_vec      = cute.zipped_divide(q_nope,      (1, 1, self.vec_q))
        smem_q_nope_vec = cute.zipped_divide(smem_q_nope, (1, self.vec_q))
        q_pe_vec        = cute.zipped_divide(q_pe,        (1, 1, self.vec_q))
        smem_q_pe_vec   = cute.zipped_divide(smem_q_pe,   (1, self.vec_q))

        # ── Vec views for sparse_load (Strategy 2) ─────────────────────────────
        si_vec = cute.zipped_divide(sparse_indices, (1, self.vec_sparse))   # ((1,4),(T,512))

        # ══════════════════════════════════════════════════════════════════════
        # Phase 1: q_issue — fire cp.async for q_nope and q_pe (non-blocking).
        #
        # All 128 threads per T-group issue cp.async calls simultaneously:
        #   q_nope: 64 chunks × 8 BF16 = 512 BF16 per row → threads 0..63
        #   q_pe:    8 chunks × 8 BF16 =  64 BF16 per row → threads 0..7
        # Threads 64..127 skip q_nope (range empty), threads 8..127 skip q_pe.
        # cp_async_commit_group() closes the group immediately after.
        # The hardware memory subsystem will DMA the data to smem asynchronously
        # while the next phase (sparse_load) runs.
        # ══════════════════════════════════════════════════════════════════════
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["q_issue"])

        if wg_per_T_idx < T:
            # q_nope: chunks 0..63, 64 threads × 1 cp.async each
            for chunk in range(thr_idx_per_T, self.q_nope_chunks, self.sparse_thr_per_T):
                cute.copy(
                    copy_atom_q,
                    q_nope_vec[(0, 0, None), (wg_per_T_idx, head_idx, chunk)],
                    smem_q_nope_vec[(0, None), (wg_per_T_idx, chunk)],
                )
            # q_pe: chunks 0..7, 8 threads × 1 cp.async each
            for chunk in range(thr_idx_per_T, self.q_pe_chunks, self.sparse_thr_per_T):
                cute.copy(
                    copy_atom_q,
                    q_pe_vec[(0, 0, None), (wg_per_T_idx, head_idx, chunk)],
                    smem_q_pe_vec[(0, None), (wg_per_T_idx, chunk)],
                )

        # Commit: bundle all outstanding cp.async calls into one trackable group.
        # This is per-thread; threads with no pending calls produce an empty group.
        cute.arch.cp_async_commit_group()

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)

        # ══════════════════════════════════════════════════════════════════════
        # Phase 2: sparse_load — Strategy 2 vec4 LDG.128 (unchanged from S2).
        #
        # While sparse_load runs, the cp.async engine is fetching q_nope/q_pe
        # from L2/HBM to smem in the background.  Since sparse_load takes ~1.5µs
        # and a 9 KB BF16 load from L2 typically completes in < 1µs, we expect
        # smem_q_nope to be fully populated before we hit wait_group below.
        # ══════════════════════════════════════════════════════════════════════
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["sparse_load"])

        partial_cnt = 0
        if wg_per_T_idx < T:
            for chunk in range(thr_idx_per_T, self.top_k_chunks, self.sparse_thr_per_T):
                vec = si_vec[(0, None), (wg_per_T_idx, chunk)].load()
                for v in range(self.vec_sparse):
                    smem_sparse[wg_per_T_idx, chunk * self.vec_sparse + v] = vec[v]
                    if vec[v] >= cutlass.Int32(0):
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

        # ══════════════════════════════════════════════════════════════════════
        # Phase 3: cp_async_wait — stall until q_nope / q_pe land in smem.
        #
        # wait_group(0): wait until 0 groups remain pending (= all done).
        # If q transferred during sparse_load this is near-zero cost.
        # sync_threads() after ensures smem visibility across all 1024 threads.
        # ══════════════════════════════════════════════════════════════════════
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["cp_async_wait"])

        cute.arch.cp_async_wait_group(0)
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
        Smem_sparse_5_intra(),
        q_nope, q_pe, sparse_indices, probe, stream,
        options="--enable-tvm-ffi"
    )


def run_single(workload_idx: int) -> str:
    import os, json as _json
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling smem_sparse_5 (Strategy 5: cp.async q overlap + vec4 sparse)...")
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
