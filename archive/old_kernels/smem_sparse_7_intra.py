"""Strategy 7: round-robin warp-group task assignment for the upfront smem phase.

S6 assigns tasks 1:1 — WG t does q_nope[t] + q_pe[t] + sparse[t].
For T < 8 this leaves WGs T..7 COMPLETELY IDLE during the upfront window,
wasting the SM's load/store throughput.

S7 distributes the 3×T tasks across ALL 8 WGs with fixed shifts:
  WG i  →  q_nope[i]          (phase A, skip if i ≥ T)
         →  q_pe[(i+PE_SHIFT)%8]   (phase B, skip if (i+PE_SHIFT)%8 ≥ T)
         →  sparse[(i+SP_SHIFT)%8] (phase C, skip if (i+SP_SHIFT)%8 ≥ T)

Shifts PE_SHIFT=3, SP_SHIFT=6 are chosen so that for every T ∈ {1..8}
the three phases map to DISTINCT T-slots per WG.

Coverage verification for T=6, 7, 8 (no slot left unloaded):
  T=8: each WG has exactly 3 tasks — perfect balance.
  T=7: 1 WG has 2 tasks (no idle WGs, max imbalance 1).
  T=6: 2 WGs have 2 tasks (no idle WGs, max imbalance 1).

WG assignments for T=6 (MaxValid=[19,20,32,12,25,3]):
  WG0: q_nope[0], q_pe[3],    sparse[6→skip]
  WG1: q_nope[1], q_pe[4],    sparse[7→skip]
  WG2: q_nope[2], q_pe[5],    sparse[0]
  WG3: q_nope[3], q_pe[6→skip], sparse[1]
  WG4: q_nope[4], q_pe[7→skip], sparse[2]
  WG5: q_nope[5], q_pe[0],    sparse[3]
  WG6: q_nope[6→skip], q_pe[1], sparse[4]
  WG7: q_nope[7→skip], q_pe[2], sparse[5]

Phases (same probe structure as S6):
  q_issue       — fire cp.async for assigned q_nope + q_pe slots
  sparse_load   — vec4 LDG.128 with while+early-exit for assigned sparse slot
  cp_async_wait — wait_group(0) + sync_threads

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

# Round-robin shifts: WG i loads q_nope[i], q_pe[(i+PE_SHIFT)%8], sparse[(i+SP_SHIFT)%8]
# PE_SHIFT=3, SP_SHIFT=6 ensure no slot is unloaded for any T ∈ {1..8}.
PE_SHIFT = 3
SP_SHIFT = 6


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Smem_sparse_7_intra():
    """Strategy 7: round-robin WG task assignment (q_nope, q_pe, sparse distributed evenly)."""

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
        self.vec_q          = VEC_Q
        self.vec_sparse     = VEC_SPARSE
        self.q_nope_chunks  = HEAD_DIM_CKV // VEC_Q             # 64
        self.q_pe_chunks    = HEAD_DIM_KPE // VEC_Q             # 8
        self.top_k_chunks   = TOP_K_LEN   // VEC_SPARSE         # 512
        self.sparse_thr_per_T = 128
        self.num_warps_per_T  = self.sparse_thr_per_T // self.wsize   # 4
        self.pe_shift       = PE_SHIFT
        self.sp_shift       = SP_SHIFT

    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)

    @cute.jit
    def __call__(self, q_nope, q_pe, sparse_indices, probe, stream):
        self.kernel(q_nope, q_pe, sparse_indices, probe).launch(
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

        head_idx      = bidx
        # wg_idx: which of the 8 warp groups this thread belongs to (0..7)
        wg_idx        = tidx // self.sparse_thr_per_T
        thr_idx_per_T = tidx % self.sparse_thr_per_T    # 0..127 within the WG
        warp_per_T_idx = warp_idx % self.num_warps_per_T
        lane_idx_per_T = thr_idx_per_T % self.wsize

        # ── SMEM allocation ────────────────────────────────────────────────────
        alloc            = cutlass.utils.SmemAllocator()
        smem_sparse      = self._smem(alloc, cutlass.Int32,    (self.T_max, self.top_k_len),        (self.top_k_len, 1),     4)
        smem_num_valid   = self._smem(alloc, cutlass.Int32,    (self.T_max,),                       (1,),                    4)
        smem_logits      = self._smem(alloc, cutlass.Float32,  (self.dim_split,),                   (1,),                   16)
        smem_red_i32     = self._smem(alloc, cutlass.Int32,    (self.T_max, 32),                    (32, 1),                 4)
        smem_max_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)
        smem_sum_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)
        smem_q_nope      = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_ckv),     (self.head_dim_ckv, 1), 16)
        smem_q_pe        = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_kpe),     (self.head_dim_kpe, 1), 16)
        smem_partial     = self._smem(alloc, cutlass.Float32,  (self.num_warps, self.head_dim_ckv), (self.head_dim_ckv, 1), 16)
        smem_out         = self._smem(alloc, cutlass.Float32,  (self.head_dim_ckv,),                (1,),                   16)

        # ── cp.async copy atom ─────────────────────────────────────────────────
        copy_atom_q = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )

        # ── Vec views for cp.async (S5 pattern) ───────────────────────────────
        q_nope_vec      = cute.zipped_divide(q_nope,      (1, 1, self.vec_q))
        smem_q_nope_vec = cute.zipped_divide(smem_q_nope, (1, self.vec_q))
        q_pe_vec        = cute.zipped_divide(q_pe,        (1, 1, self.vec_q))
        smem_q_pe_vec   = cute.zipped_divide(smem_q_pe,   (1, self.vec_q))

        # ── Vec view for sparse scan ───────────────────────────────────────────
        si_vec = cute.zipped_divide(sparse_indices, (1, self.vec_sparse))

        # ── Round-robin T-slot assignments ────────────────────────────────────
        # WG i is responsible for:
        #   t_nope = i                        (q_nope slot)
        #   t_pe   = (i + PE_SHIFT) % T_MAX  (q_pe   slot)
        #   t_sp   = (i + SP_SHIFT) % T_MAX  (sparse  slot)
        # Each slot is handled by exactly one WG; uncovered slots (≥T) are skipped.
        t_nope = wg_idx
        t_pe   = (wg_idx + cutlass.Int32(self.pe_shift)) % cutlass.Int32(self.T_max)
        t_sp   = (wg_idx + cutlass.Int32(self.sp_shift)) % cutlass.Int32(self.T_max)

        # ══════════════════════════════════════════════════════════════════════
        # Phase 1: q_issue — fire cp.async for assigned q_nope and q_pe slots.
        #
        # cp.async is fire-and-forget; multiple WGs can issue simultaneously
        # because they touch disjoint smem rows (t_nope and t_pe differ per WG).
        # ══════════════════════════════════════════════════════════════════════
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["q_issue"])

        # Phase A: q_nope for T-slot t_nope
        if t_nope < T:
            for chunk in range(thr_idx_per_T, self.q_nope_chunks, self.sparse_thr_per_T):
                cute.copy(
                    copy_atom_q,
                    q_nope_vec[(0, 0, None), (t_nope, head_idx, chunk)],
                    smem_q_nope_vec[(0, None), (t_nope, chunk)],
                )

        # Phase B: q_pe for T-slot t_pe
        if t_pe < T:
            for chunk in range(thr_idx_per_T, self.q_pe_chunks, self.sparse_thr_per_T):
                cute.copy(
                    copy_atom_q,
                    q_pe_vec[(0, 0, None), (t_pe, head_idx, chunk)],
                    smem_q_pe_vec[(0, None), (t_pe, chunk)],
                )

        cute.arch.cp_async_commit_group()

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)

        # ══════════════════════════════════════════════════════════════════════
        # Phase 2: sparse_load — WG i scans sparse[t_sp] with early exit.
        #
        # Multiple WGs scan DIFFERENT T-slots simultaneously, spreading the
        # sparse-index load across all 8 WGs instead of only those with T < 8.
        # The intra-WG named barrier uses wg_idx (not t_sp) so that exactly
        # the 128 threads belonging to this WG synchronize together.
        # smem_num_valid[t_sp] is keyed by T-slot, not WG index.
        # ══════════════════════════════════════════════════════════════════════
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["sparse_load"])

        partial_cnt = 0
        if t_sp < T:
            chunk = cutlass.Int32(thr_idx_per_T)
            while chunk < cutlass.Int32(self.top_k_chunks):
                vec = si_vec[(0, None), (t_sp, chunk)].load()
                v0 = vec[0]   # extract before inner loop — dominates exit check
                for v in range(self.vec_sparse):
                    smem_sparse[t_sp, chunk * self.vec_sparse + v] = vec[v]
                    if vec[v] >= cutlass.Int32(0):
                        partial_cnt += 1
                if v0 < cutlass.Int32(0):
                    chunk = cutlass.Int32(self.top_k_chunks)   # exit while
                else:
                    chunk = chunk + cutlass.Int32(self.sparse_thr_per_T)

            # Intra-WG warp reduce — barrier keyed by wg_idx (not t_sp)
            # so exactly the 128 threads of this WG synchronize.
            cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
            if lane_idx_per_T == 0:
                smem_red_i32[wg_idx, warp_per_T_idx] = cnt_sum

            cute.arch.barrier(barrier_id=wg_idx + 1,
                              number_of_threads=self.sparse_thr_per_T)

            if warp_per_T_idx == 0:
                val     = smem_red_i32[wg_idx, lane_idx_per_T]
                cnt_sum = warp_reduce(val, lambda a, b: a + b, width=self.num_warps_per_T)
                smem_red_i32[wg_idx, 0] = cnt_sum

            cute.arch.barrier(barrier_id=wg_idx + 1,
                              number_of_threads=self.sparse_thr_per_T)

            # Write to the T-slot's smem_num_valid, not the WG index
            smem_num_valid[t_sp] = smem_red_i32[wg_idx, 0]

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)

        # ══════════════════════════════════════════════════════════════════════
        # Phase 3: cp_async_wait — stall until q lands in smem then sync.
        # sync_threads() provides block-wide visibility of all smem writes
        # (smem_sparse, smem_num_valid, smem_q_nope, smem_q_pe).
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
    Bc = NUM_HEADS * NUM_SPLITS

    q_nope         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K_LEN),               (1, 0),     4)
    probe          = _fake(cute.Int64,    (Bc, PROBE_COLS),             (1, 0),     8)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        Smem_sparse_7_intra(),
        q_nope, q_pe, sparse_indices, probe, stream,
        options="--enable-tvm-ffi"
    )


def run_single(workload_idx: int) -> str:
    import os, json as _json
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling smem_sparse_7 (Strategy 7: round-robin WG assignment)...")
    compiled = compile_kernel()

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [_json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]

    num_blocks = NUM_HEADS * NUM_SPLITS
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
