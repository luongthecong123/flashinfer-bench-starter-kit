"""opt4_output: Smem-cached GEMV output with warp-tiled accumulation.

Baseline problem: output phase does a GEMV (valid_count × 512) where each of
52 outer-loop iterations reads a full CKV row from global memory.

GMEM pattern per outer iteration:
  - Thread tidx reads ckv_cache[kv_idx, tidx] (tidx in [0, 512))
  - Threads 512..1023 are IDLE (50% wasted)
  - 52 iterations × L2/L1 latency per row

Optimization: preload all valid CKV rows into shared memory ONCE (coalesced),
then do the accumulation entirely from L1-smem.

Step 1 — Cooperative preload into smem_kv[MAX_VALID, head_dim_ckv]:
  - 1024 threads load MAX_VALID * 512 bfloat16 values cooperatively
  - Access pattern: thread t loads smem_kv[t // 512, t % 512] in strided fashion
  - For MAX_VALID=64: 64*512*2=65536 bytes = 64 KB from gmem (fully coalesced)

Step 2 — GEMV from smem (32 warps × 16 output dims):
  - Reorder loops: outer over output-dim tiles (per-warp), inner over valid_count
  - Each of 32 warps owns 16 contiguous output dimensions (512/32=16)
  - Each lane within a warp owns 1 output dim at a time: dim = warp_idx*16 + k
  - Iterate all valid_count tokens, accumulate attn[j]*smem_kv[j, dim]
  - Result: each lane has 1 complete output element → store directly

Memory budget (228 KB per SM):
  - smem_score_nope:    2048*4 =  8 KB
  - smem_score_pe:      2048*4 =  8 KB
  - smem_logits_scaled: 2048*4 =  8 KB
  - smem_sparse_idx:    2048*4 =  8 KB
  - smem_valid_count:           ~4  B
  - smem_kv:          64*512*2 = 64 KB  (MAX_VALID=64)
  - smem_output:        512*4  =  2 KB
  Total: ~98 KB — well within 228 KB budget.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm

import math
import json
import torch

# Maximum valid-count for smem preload (must be >= actual max_valid)
# WL3 has MaxValid=52; use 64 (power-of-2 headroom).
MAX_VALID = 64


# ── Inline PTX helpers ───────────────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(
        MLIR_T.i64(), [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(
        MLIR_T.i32(), [],
        "mov.u32 $0, %smid;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


# ── Probe constants ──────────────────────────────────────────────────────────

PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 12
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

TAGS = {
    "load_indices": 0,
    "score":        2,
    "valid_count":  3,
    "softmax":      4,
    "output":       6,
    "epilogue":     8,
}
TAG_NAMES = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["load_indices", "score", "valid_count", "softmax", "output", "epilogue"]


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


# ── Host-side dump ───────────────────────────────────────────────────────────

def dump_probe(probe: torch.Tensor, num_blocks: int) -> str:
    probe_cpu = probe.cpu().contiguous().tolist()

    for bid in range(min(num_blocks, 4)):
        data = probe_cpu[bid]
        cnt = int(data[0])
        print(f"\n--- Block {bid}: {cnt} entries ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id, tag = int(data[off]), int(data[off + 1])
            start, dur = int(data[off + 2]), int(data[off + 3])
            print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>15s}  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    tag_totals: dict = {}
    tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1])
            dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*60}")
    print(f"{'Phase':>15s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>12s} {'%':>8s}")
    print(f"{'='*60}")
    grand_total = sum(tag_totals.values())
    for name in PHASE_ORDER:
        if name in tag_totals:
            total_ns = tag_totals[name]
            count = tag_counts[name]
            pct = 100.0 * total_ns / grand_total if grand_total > 0 else 0
            print(f"{name:>15s} {total_ns/1e6:>12.3f} {count:>6d} {total_ns/count/1000:>12.1f} {pct:>7.1f}%")
    print(f"{'TOTAL':>15s} {grand_total/1e6:>12.3f}")

    events = []
    global_base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (global_base is None or s < global_base):
                global_base = s
    global_base = global_base or 0

    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        if cnt == 0:
            continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1])
            start = int(data[off + 2])
            dur = int(data[off + 3])
            if start == 0 and dur == 0:
                continue
            events.append(dict(
                name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - global_base) / 1000.0,
                dur=dur / 1000.0,
                pid=sm_id, tid=bid))

    num_sms = len({e["pid"] for e in events})
    print(f"\nTrace: {len(events)} events from {num_sms} SMs")
    return json.dumps({"traceEvents": events})


# ── Warp reduce ──────────────────────────────────────────────────────────────

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ── Profiled kernel (opt4: smem GEMV output) ────────────────────────────────

@cute.jit
def fused_dsa_opt4_profiled(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor,
    probe: cute.Tensor,
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape
    fused_dsa_kernel_opt4_profiled(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, probe
    ).launch(grid=[T, num_heads, 1], block=[1024, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_opt4_profiled(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor,
    probe: cute.Tensor):

    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = kpe_cache.shape[1]
    top_k_len = 2048

    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    num_threads = bdimx
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    wsize = cute.arch.WARP_SIZE
    num_warps = bdimx // wsize

    sm = smid_u32()
    probe_row = bidx * cutlass.Int32(num_heads) + bidy
    probe_cnt = cutlass.Int32(0)

    allocator = cutlass.utils.SmemAllocator()
    smem_score_nope    = allocator.allocate_tensor(cutlass.Float32,   cute.make_layout((top_k_len),              stride=(1)),           16, None)
    smem_score_pe      = allocator.allocate_tensor(cutlass.Float32,   cute.make_layout((top_k_len),              stride=(1)),           16, None)
    smem_logits_scaled = allocator.allocate_tensor(cutlass.Float32,   cute.make_layout((top_k_len),              stride=(1)),           16, None)
    smem_sparse_idx    = allocator.allocate_tensor(cutlass.Int32,     cute.make_layout((top_k_len),              stride=(1)),           4,  None)
    smem_valid_count   = allocator.allocate_tensor(cutlass.Int32,     cute.make_layout((1),                     stride=(1)),           4,  None)
    # OPT4: smem_kv[MAX_VALID, head_dim_ckv] — BFloat16 with stride=514 to eliminate bank conflicts
    # stride=514 (not 512) ensures bank(j,d) = (j + floor(d/2)) % 32 → no conflicts across lanes
    smem_kv            = allocator.allocate_tensor(cutlass.BFloat16,  cute.make_layout((MAX_VALID, head_dim_ckv), stride=(514, 1)),          16, None)
    smem_output        = allocator.allocate_tensor(cutlass.Float32,   cute.make_layout((head_dim_ckv),           stride=(1)),           16, None)

    # ── Phase 1: Load indices ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["load_indices"])
    for i in range(tidx, top_k_len, num_threads):
        smem_sparse_idx[i] = sparse_indices[bidx, i]
    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 2: Score (unchanged) ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["score"])
    q_nope_local = q_nope[bidx, bidy, None]
    q_pe_local   = q_pe[bidx, bidy, None]
    for round_idx in range(top_k_len // num_warps):
        sparse_idx = round_idx * num_warps + warp_idx
        cur_idx = smem_sparse_idx[sparse_idx]
        if cur_idx >= cutlass.Int32(0):
            lane_idx = cute.arch.lane_idx()
            sum_partial_nope = cutlass.Float32(0)
            for k_idx in range(head_dim_ckv // wsize):
                q_nope_val = cutlass.Float32(q_nope_local[k_idx * wsize + lane_idx])
                ckv_val    = cutlass.Float32(ckv_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial_nope += q_nope_val * ckv_val
            sum_nope = warp_reduce(sum_partial_nope, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_nope[sparse_idx] = sum_nope

            sum_partial_pe = cutlass.Float32(0)
            for k_idx in range(head_dim_kpe // wsize):
                q_pe_val = cutlass.Float32(q_pe_local[k_idx * wsize + lane_idx])
                kpe_val  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial_pe += q_pe_val * kpe_val
            sum_pe = warp_reduce(sum_partial_pe, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_pe[sparse_idx] = sum_pe
    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 3: Valid count (unchanged) ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["valid_count"])
        num_valid = cutlass.Int32(0)
        for i in range(top_k_len):
            if smem_sparse_idx[i] >= cutlass.Int32(0):
                num_valid = cutlass.Int32(i + 1)
        smem_valid_count[0] = num_valid
    cute.arch.sync_threads()
    valid_count = smem_valid_count[0]
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 4: Scale + Softmax (unchanged) ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["softmax"])
    for i in range(tidx, valid_count, num_threads):
        smem_logits_scaled[i] = sm_scale * (smem_score_nope[i] + smem_score_pe[i])
    cute.arch.sync_threads()
    if tidx == 0:
        row_max = smem_logits_scaled[0]
        for i in range(valid_count):
            if smem_logits_scaled[i] > row_max:
                row_max = smem_logits_scaled[i]
        row_sum = cutlass.Float32(0)
        for i in range(valid_count):
            row_sum += cute.math.exp(smem_logits_scaled[i] - row_max)
        lse[bidx, bidy] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(0.6931471805599453)
        for i in range(valid_count):
            smem_logits_scaled[i] = cute.math.exp(smem_logits_scaled[i] - row_max) / row_sum
    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 5: Output — OPT4: chunked smem preload + warp-tiled GEMV ──
    # valid_count can reach 2048, so we tile over MAX_VALID=64 chunks.
    # smem_kv[MAX_VALID, 512] is reused as a sliding window across chunks.
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["output"])

    lane_idx = cute.arch.lane_idx()
    dims_per_warp = head_dim_ckv // num_warps   # 16 for 32 warps × 512 dims

    # Compile-time constants derived from MAX_VALID=64, num_warps=32, wsize=32
    rows_per_warp = MAX_VALID // 32   # = 2 rows per warp per chunk
    jsteps        = MAX_VALID // 32   # = 2 j-steps per lane per chunk

    # Zero smem_output once; we accumulate += across chunks below.
    for i in range(tidx, head_dim_ckv, num_threads):
        smem_output[i] = cutlass.Float32(0)
    cute.arch.sync_threads()

    # Outer loop: slide over valid_count rows in windows of MAX_VALID.
    for chunk_start in range(0, valid_count, MAX_VALID):

        # Step A: cooperative preload of this chunk into smem_kv[0..MAX_VALID-1, :].
        # Warp k owns rows {k, k+32} within the chunk (rows_per_warp=2).
        # Guard partial last chunk with 'if actual_row < valid_count'.
        for r in range(rows_per_warp):
            kv_row = r * num_warps + warp_idx
            actual_row = chunk_start + kv_row
            if actual_row < valid_count:
                kv_idx = smem_sparse_idx[actual_row]
                for col in range(lane_idx, head_dim_ckv, wsize):
                    smem_kv[kv_row, col] = ckv_cache[kv_idx, col]

        cute.arch.sync_threads()

        # Step B: GEMV contribution of this chunk.
        # Each warp owns dims_per_warp=16 output dims; each lane accumulates 2 j-steps.
        for dim_offset in range(dims_per_warp):
            out_dim = warp_idx * dims_per_warp + dim_offset
            local_acc = cutlass.Float32(0)
            for js in range(jsteps):
                j = js * wsize + lane_idx
                actual_j = chunk_start + j
                if actual_j < valid_count:
                    local_acc += smem_logits_scaled[actual_j] * cutlass.Float32(smem_kv[j, out_dim])
            total_acc = warp_reduce(local_acc, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_output[out_dim] += total_acc

        # Barrier before next chunk reuses smem_kv.
        cute.arch.sync_threads()

    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 6: Epilogue ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["epilogue"])
    for i in range(tidx, head_dim_ckv, num_threads):
        output[bidx, bidy, i] = cutlass.BFloat16(smem_output[i])
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_finalize(probe, probe_row, probe_cnt)


# ── Compilation ──────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=assumed_align)


def compile_kernel():
    T = cute.sym_int()
    N = cute.sym_int()
    B = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048

    q_nope        = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe          = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache     = _fake(cute.BFloat16, (N, head_dim_ckv),            (1, 0),    16)
    kpe_cache     = _fake(cute.BFloat16, (N, head_dim_kpe),            (1, 0),    16)
    sparse_indices = _fake(cute.Int32,   (T, top_k_len),               (1, 0),    4)
    sm_scale      = 0.1352337788608801
    output        = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse           = _fake(cute.Float32,  (T, num_heads),               (1, 0),    4)
    probe         = _fake(cute.Int64,    (B, PROBE_COLS),              (1, 0),    8)
    stream        = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_dsa_opt4_profiled,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        output, lse, probe, stream,
        options="--enable-tvm-ffi"
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def run_single(workload_idx: int) -> str:
    """Run intra-kernel profiling for one workload. Returns trace JSON string."""
    import os, json as js
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    H, D_ckv = 16, 512

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling opt4_output (smem preload + warp-tiled GEMV) kernel...")
    compiled = compile_kernel()

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [js.loads(l) for l in open(JSONL)]

    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]

    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]
    num_blocks = T * H
    print(f"\nWorkload {workload_idx + 1}: MaxValid={max_valid}  T={T}  Blocks={num_blocks}")

    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
    Kc_all = ckv.reshape(-1, D_ckv)
    Kp_all = kpe.reshape(-1, 64)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    output = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse    = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
    probe  = torch.zeros((num_blocks, PROBE_COLS), dtype=torch.int64, device="cuda")

    for _ in range(3):
        output.zero_(); lse.fill_(-float("inf")); probe.zero_()
        compiled(q_nope, q_pe, Kc_all, Kp_all, si, output, lse, probe)
        torch.cuda.synchronize()

    probe.zero_(); output.zero_(); lse.fill_(-float("inf"))
    compiled(q_nope, q_pe, Kc_all, Kp_all, si, output, lse, probe)
    torch.cuda.synchronize()

    return dump_probe(probe, num_blocks=num_blocks)
