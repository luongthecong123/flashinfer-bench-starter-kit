"""opt1_score: Fused nope+pe score optimization.

Optimization: instead of two separate warp dot-products (nope: 512-dim, pe: 64-dim)
written to two smem arrays and then added+scaled in a later pass, we:
  1. Compute both dot-products into a SINGLE float32 accumulator per warp.
  2. Bake sm_scale into the score write-back — result goes directly to
     smem_logits_scaled, eliminating smem_score_nope and smem_score_pe entirely.
  3. Remove the separate "scale logits" loop that ran over valid_count entries.

Expected savings: ~8 KB smem + 1 fewer loop pass + the scale-logits pass latency.

Phase timeline (5 phases instead of 6):
  load_indices → score (fused) → valid_count → softmax → output → epilogue
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm

import math
import json
import torch


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
PROBE_ENTRY  = 4   # (sm_id, tag, start_time, duration)
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


# ── Profiled kernel (opt1: fused score) ─────────────────────────────────────

@cute.jit
def fused_dsa_opt1_profiled(
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
    fused_dsa_kernel_opt1_profiled(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, probe
    ).launch(grid=[T, num_heads, 1], block=[1024, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_opt1_profiled(
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
    # OPT1: single smem_logits_scaled — stores sm_scale*(nope+pe) directly from score phase
    # Eliminates smem_score_nope (8KB) and smem_score_pe (8KB) from baseline
    smem_logits_scaled = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_sparse_idx    = allocator.allocate_tensor(cutlass.Int32,   cute.make_layout((top_k_len), stride=(1)), 4,  None)
    smem_valid_count   = allocator.allocate_tensor(cutlass.Int32,   cute.make_layout((1),         stride=(1)), 4,  None)

    # ── Phase 1: Load indices ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["load_indices"])

    # Issue q-register loads before the sparse_indices loop.
    # load_indices takes ~2000+ cycles; L2 latency is ~130 cycles.
    # By the time sync_threads() below completes, all qn/qpe values are
    # already in registers — their latency is fully hidden behind the
    # sparse_indices load loop. Score phase sees zero q-load overhead.
    q_nope_local = q_nope[bidx, bidy, None]
    q_pe_local   = q_pe[bidx, bidy, None]
    lane_idx = cute.arch.lane_idx()
    qn0  = cutlass.Float32(q_nope_local[ 0 * wsize + lane_idx])
    qn1  = cutlass.Float32(q_nope_local[ 1 * wsize + lane_idx])
    qn2  = cutlass.Float32(q_nope_local[ 2 * wsize + lane_idx])
    qn3  = cutlass.Float32(q_nope_local[ 3 * wsize + lane_idx])
    qn4  = cutlass.Float32(q_nope_local[ 4 * wsize + lane_idx])
    qn5  = cutlass.Float32(q_nope_local[ 5 * wsize + lane_idx])
    qn6  = cutlass.Float32(q_nope_local[ 6 * wsize + lane_idx])
    qn7  = cutlass.Float32(q_nope_local[ 7 * wsize + lane_idx])
    qn8  = cutlass.Float32(q_nope_local[ 8 * wsize + lane_idx])
    qn9  = cutlass.Float32(q_nope_local[ 9 * wsize + lane_idx])
    qn10 = cutlass.Float32(q_nope_local[10 * wsize + lane_idx])
    qn11 = cutlass.Float32(q_nope_local[11 * wsize + lane_idx])
    qn12 = cutlass.Float32(q_nope_local[12 * wsize + lane_idx])
    qn13 = cutlass.Float32(q_nope_local[13 * wsize + lane_idx])
    qn14 = cutlass.Float32(q_nope_local[14 * wsize + lane_idx])
    qn15 = cutlass.Float32(q_nope_local[15 * wsize + lane_idx])
    qpe0 = cutlass.Float32(q_pe_local[0 * wsize + lane_idx])
    qpe1 = cutlass.Float32(q_pe_local[1 * wsize + lane_idx])

    for i in range(tidx, top_k_len, num_threads):
        smem_sparse_idx[i] = sparse_indices[bidx, i]
    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 2: Score — q values already in registers from phase 1 ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["score"])

    # OPT: early-exit flag — sparse indices are left-packed.
    still_valid = cutlass.Int32(1)
    for round_idx in range(top_k_len // num_warps):
        if still_valid:
            if smem_sparse_idx[round_idx * num_warps] < cutlass.Int32(0):
                still_valid = cutlass.Int32(0)
        if still_valid:
            sparse_idx = round_idx * num_warps + warp_idx
            cur_idx = smem_sparse_idx[sparse_idx]
            if cur_idx >= cutlass.Int32(0):
                # 4 independent accumulators → 4 parallel FMA chains (depth-4 each)
                # vs one serial chain of depth-18 in the original.
                acc0 = cutlass.Float32(0)
                acc1 = cutlass.Float32(0)
                acc2 = cutlass.Float32(0)
                acc3 = cutlass.Float32(0)

                # nope: 16 loads interleaved across 4 accumulators
                acc0 += qn0  * cutlass.Float32(ckv_cache[cur_idx,  0 * wsize + lane_idx])
                acc1 += qn1  * cutlass.Float32(ckv_cache[cur_idx,  1 * wsize + lane_idx])
                acc2 += qn2  * cutlass.Float32(ckv_cache[cur_idx,  2 * wsize + lane_idx])
                acc3 += qn3  * cutlass.Float32(ckv_cache[cur_idx,  3 * wsize + lane_idx])
                acc0 += qn4  * cutlass.Float32(ckv_cache[cur_idx,  4 * wsize + lane_idx])
                acc1 += qn5  * cutlass.Float32(ckv_cache[cur_idx,  5 * wsize + lane_idx])
                acc2 += qn6  * cutlass.Float32(ckv_cache[cur_idx,  6 * wsize + lane_idx])
                acc3 += qn7  * cutlass.Float32(ckv_cache[cur_idx,  7 * wsize + lane_idx])
                acc0 += qn8  * cutlass.Float32(ckv_cache[cur_idx,  8 * wsize + lane_idx])
                acc1 += qn9  * cutlass.Float32(ckv_cache[cur_idx,  9 * wsize + lane_idx])
                acc2 += qn10 * cutlass.Float32(ckv_cache[cur_idx, 10 * wsize + lane_idx])
                acc3 += qn11 * cutlass.Float32(ckv_cache[cur_idx, 11 * wsize + lane_idx])
                acc0 += qn12 * cutlass.Float32(ckv_cache[cur_idx, 12 * wsize + lane_idx])
                acc1 += qn13 * cutlass.Float32(ckv_cache[cur_idx, 13 * wsize + lane_idx])
                acc2 += qn14 * cutlass.Float32(ckv_cache[cur_idx, 14 * wsize + lane_idx])
                acc3 += qn15 * cutlass.Float32(ckv_cache[cur_idx, 15 * wsize + lane_idx])

                # pe: 2 loads split across acc0/acc1
                acc0 += qpe0 * cutlass.Float32(kpe_cache[cur_idx, 0 * wsize + lane_idx])
                acc1 += qpe1 * cutlass.Float32(kpe_cache[cur_idx, 1 * wsize + lane_idx])

                sum_total = (acc0 + acc1) + (acc2 + acc3)

                fused_score = warp_reduce(sum_total, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_logits_scaled[sparse_idx] = sm_scale * fused_score

    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 3: Compute valid count (unchanged from baseline) ──
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

    # ── Phase 4: Softmax — OPT1: no separate scale pass needed ──
    # smem_logits_scaled already holds sm_scale*(nope+pe) scores
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["softmax"])
    # No scaling loop needed — sm_scale already baked in during score phase

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

    # ── Phase 5: Output accumulation ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["output"])
    smem_output = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((head_dim_ckv), stride=(1)), 16, None)
    for i in range(tidx, head_dim_ckv, num_threads):
        smem_output[i] = cutlass.Float32(0)
    cute.arch.sync_threads()
    for j in range(valid_count):
        kv_idx = smem_sparse_idx[j]
        attn_weight = smem_logits_scaled[j]
        for i in range(tidx, head_dim_ckv, num_threads):
            smem_output[i] += attn_weight * cutlass.Float32(ckv_cache[kv_idx, i])
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
        fused_dsa_opt1_profiled,
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
    print("Compiling opt1_score (fused nope+pe) kernel...")
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
