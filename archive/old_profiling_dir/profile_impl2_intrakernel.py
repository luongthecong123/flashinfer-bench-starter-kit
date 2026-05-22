"""
Intra-kernel profiling for impl2 (32-warp parallel-keys letmecook).
Uses globaltimer PTX to measure per-phase durations inside the kernel.

Phases:
  - load_indices: cooperative load of sparse_indices into smem
  - score:        32-warp parallel dot products (nope + pe)
  - valid_count:  serial scan for last valid index
  - softmax:      scale + max + exp + normalize (serial on thread 0)
  - output:       weighted accumulation into output buffer
  - epilogue:     write output to gmem
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm

from typing import Tuple
import math
import json
import torch


# ── Inline PTX helpers ──────────────────────────────────────────────

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


# ── Profiler constants ──────────────────────────────────────────────
PROBE_HEADER = 1
PROBE_ENTRY  = 4   # (sm_id, tag, start_time, duration)

TAGS = {
    "load_indices": 0,
    "score":        2,
    "valid_count":  3,
    "softmax":      4,
    "output":       6,
    "epilogue":     8,
}
TAG_NAMES = {v: k for k, v in TAGS.items()}


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


# ── Host-side dump ──────────────────────────────────────────────────

def dump_probe(probe: torch.Tensor, num_blocks: int,
               out_path: str = "impl2_intrakernel_trace.json"):
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

    tag_totals = {}
    tag_counts = {}
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
    for name in ["load_indices", "score", "valid_count", "softmax", "output", "epilogue"]:
        if name in tag_totals:
            total_ns = tag_totals[name]
            count = tag_counts[name]
            pct = 100.0 * total_ns / grand_total if grand_total > 0 else 0
            print(f"{name:>15s} {total_ns/1e6:>12.3f} {count:>6d} {total_ns/count/1000:>12.1f} {pct:>7.1f}%")
    print(f"{'TOTAL':>15s} {grand_total/1e6:>12.3f}")

    events, global_base = [], None
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

    with open(out_path, "w") as f:
        json.dump({"traceEvents": events}, f)
    num_sms = len({e["pid"] for e in events})
    print(f"\nTrace: {len(events)} events from {num_sms} SMs -> {out_path}")


# ── Profiled impl2 kernel ──────────────────────────────────────────

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def fused_dsa_v2_profiled(
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

    fused_dsa_kernel_v2_profiled(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, probe
    ).launch(grid=[T, num_heads, 1], block=[1024, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_v2_profiled(
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
    num_warps = 32

    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    num_threads = bdimx
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    wsize = cute.arch.WARP_SIZE

    sm = smid_u32()
    probe_row = bidx * cutlass.Int32(num_heads) + bidy
    probe_cnt = cutlass.Int32(0)

    allocator = cutlass.utils.SmemAllocator()
    smem_score_nope = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_score_pe = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_logits_scaled = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_sparse_idx = allocator.allocate_tensor(cutlass.Int32, cute.make_layout((top_k_len), stride=(1)), 4, None)
    smem_valid_count = allocator.allocate_tensor(cutlass.Int32, cute.make_layout((1), stride=(1)), 4, None)

    # ── Phase 1: Load indices ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["load_indices"])

    for i in range(tidx, top_k_len, num_threads):
        smem_sparse_idx[i] = sparse_indices[bidx, i]

    cute.arch.sync_threads()

    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 2: Score (32-warp parallel) ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["score"])

    q_nope_local = q_nope[bidx, bidy, None]
    q_pe_local = q_pe[bidx, bidy, None]

    for round_idx in range(top_k_len // num_warps):
        sparse_idx = round_idx * num_warps + warp_idx
        cur_idx = smem_sparse_idx[sparse_idx]

        if cur_idx >= cutlass.Int32(0):
            lane_idx = cute.arch.lane_idx()

            sum_partial_nope = cutlass.Float32(0)
            for k_idx in range(head_dim_ckv // wsize):
                q_nope_val = cutlass.Float32(q_nope_local[k_idx * wsize + lane_idx])
                ckv_val = cutlass.Float32(ckv_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial_nope += q_nope_val * ckv_val
            sum_nope = warp_reduce(sum_partial_nope, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_nope[sparse_idx] = sum_nope

            sum_partial_pe = cutlass.Float32(0)
            for k_idx in range(head_dim_kpe // wsize):
                q_pe_val = cutlass.Float32(q_pe_local[k_idx * wsize + lane_idx])
                kpe_val = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial_pe += q_pe_val * kpe_val
            sum_pe = warp_reduce(sum_partial_pe, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_pe[sparse_idx] = sum_pe

    cute.arch.sync_threads()

    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Phase 3: Compute valid count ──
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

    # ── Phase 4: Softmax ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["softmax"])

    for i in range(tidx, valid_count, num_threads):
        logits_scaled = sm_scale * (smem_score_nope[i] + smem_score_pe[i])
        smem_logits_scaled[i] = logits_scaled
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


# ── Compilation ─────────────────────────────────────────────────────

def fake_wrapper(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)


def compile_profiled_impl2():
    T = cute.sym_int()
    N = cute.sym_int()
    B = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048

    q_nope = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache = fake_wrapper(cute.BFloat16, (N, head_dim_ckv), (1, 0), 16)
    kpe_cache = fake_wrapper(cute.BFloat16, (N, head_dim_kpe), (1, 0), 16)
    sparse_indices = fake_wrapper(cute.Int32, (T, top_k_len), (1, 0), 4)
    sm_scale = 0.1352337788608801
    output = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse = fake_wrapper(cute.Float32, (T, num_heads), (1, 0), 4)

    max_entries = 12
    probe_cols = PROBE_HEADER + max_entries * PROBE_ENTRY
    probe = fake_wrapper(cute.Int64, (B, probe_cols), (1, 0), 8)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_dsa_v2_profiled,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse,
        probe, stream,
        options="--enable-tvm-ffi"
    )


# ── Main: run profiling ────────────────────────────────────────────

def run_profiling():
    import sys, json as js
    from pathlib import Path
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/dev")
    from safetensors.torch import load_file
    from cook import make_tensors

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    H = 16
    D_ckv = 512
    D_kpe = 64

    print("Compiling profiled impl2 kernel...")
    compiled = compile_profiled_impl2()

    max_entries = 12
    probe_cols = PROBE_HEADER + max_entries * PROBE_ENTRY

    CONTEST = Path("/app").parent / "flashinfer26dsa" / "mlsys26-contest"
    JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [js.loads(l) for l in open(JSONL)]

    TARGET_IDS = [1, 2, 6, 9, 16]
    traces = {}

    for wid in TARGET_IDS:
        w = workloads[wid - 1]
        ax = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        T, P = ax["num_tokens"], ax["num_pages"]
        uuid = w["workload"]["uuid"][:8]

        q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
        sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
        si = sf[inp["sparse_indices"]["tensor_key"]].cuda()
        max_valid_list = [(si[t] != -1).sum().item() for t in range(T)]

        Kc_all = ckv.reshape(-1, D_ckv)
        Kp_all = kpe.reshape(-1, D_kpe)

        num_blocks = T * H
        probe = torch.zeros((num_blocks, probe_cols), dtype=torch.int64, device="cuda")
        output = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
        lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

        for _ in range(3):
            output.zero_(); lse.fill_(-float("inf")); probe.zero_()
            compiled(q_nope, q_pe, Kc_all, Kp_all, si, output, lse, probe)
            torch.cuda.synchronize()

        probe.zero_(); output.zero_(); lse.fill_(-float("inf"))
        compiled(q_nope, q_pe, Kc_all, Kp_all, si, output, lse, probe)
        torch.cuda.synchronize()

        trace_path = f"/tmp/impl2_wl{wid}_trace.json"
        print(f"\n{'='*70}")
        print(f"Workload {wid} ({uuid}): T={T}, P={P}, MaxValid={max_valid_list}, Blocks={num_blocks}")
        print(f"{'='*70}")
        dump_probe(probe, num_blocks=num_blocks, out_path=trace_path)
        traces[str(wid)] = open(trace_path).read()

    return js.dumps(traces)
