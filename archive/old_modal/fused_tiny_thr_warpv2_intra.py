"""Intra-kernel profiling for fused_tiny_thr_warpv2.
Phases:
  load / valid_count / score / softmax_max / softmax_exp_sum / output / reduce
Compared to tiny5v2:
  - score uses vectorized LDG.128
  - softmax_exp_sum fuses exp+writeback (no separate normalize)
  - output uses vectorized LDG.128
  - reduce writes directly to global (no smem_output / epilogue)
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
import math, json, torch

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
MAX_ENTRIES  = 14
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

TAGS = {"load": 0, "valid_count": 2, "score": 4,
        "softmax_max": 6, "softmax_exp_sum": 8, "output": 10, "reduce": 12}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["load", "valid_count", "score", "softmax_max", "softmax_exp_sum", "output", "reduce"]

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

def dump_probe(probe: torch.Tensor, num_blocks: int) -> str:
    probe_cpu = probe.cpu().contiguous().tolist()
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid
    for bid in [max_bid]:
        data = probe_cpu[bid]
        cnt = int(data[0])
        print(f"\n--- Block {bid} (longest, total={max_dur/1000:.1f}µs): {cnt} entries ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id, tag = int(data[off]), int(data[off + 1])
            dur = int(data[off + 3])
            print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>15s}  dur={dur:>10} ns  ({dur/1000:.1f} µs)")
    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1
    print(f"\n{'='*60}")
    print(f"{'Phase':>15s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>12s} {'%':>8s}")
    print(f"{'='*60}")
    grand_total = sum(tag_totals.values())
    for name in PHASE_ORDER:
        if name in tag_totals:
            total_ns = tag_totals[name]; count = tag_counts[name]
            pct = 100.0 * total_ns / grand_total if grand_total > 0 else 0
            print(f"{name:>15s} {total_ns/1e6:>12.3f} {count:>6d} {total_ns/count/1000:>12.1f} {pct:>7.1f}%")
    print(f"{'TOTAL':>15s} {grand_total/1e6:>12.3f}")
    events = []; global_base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (global_base is None or s < global_base):
                global_base = s
    global_base = global_base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        if cnt == 0: continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); start = int(data[off + 2]); dur = int(data[off + 3])
            if start == 0 and dur == 0: continue
            events.append(dict(name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - global_base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id, tid=bid))
    return json.dumps({"traceEvents": events})

# ── Kernel constants ──────────────────────────────────────────────────────────
BLOCK_SIZE = 1024
NUM_WARPS  = BLOCK_SIZE // 32
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32  # 16
NUM_VEC      : cutlass.Constexpr = 8
ITERS_PER_LANE: cutlass.Constexpr = (512 // 32) // 8  # 2
LN2 = 0.6931471805599453

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.jit
def fused_dsa_thr_warpv2_profiled(
    q_nope: cute.Tensor, q_pe: cute.Tensor,
    ckv_cache: cute.Tensor, kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor, sm_scale: cutlass.Constexpr,
    output: cute.Tensor, lse: cute.Tensor,
    probe: cute.Tensor, stream):
    T, num_heads, head_dim_ckv = q_nope.shape
    fused_dsa_kernel_thr_warpv2_profiled(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, probe
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE, 1, 1], stream=stream)

@cute.kernel
def fused_dsa_kernel_thr_warpv2_profiled(
    q_nope: cute.Tensor, q_pe: cute.Tensor,
    ckv_cache: cute.Tensor, kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor, sm_scale: cutlass.Constexpr,
    output: cute.Tensor, lse: cute.Tensor,
    probe: cute.Tensor):

    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = kpe_cache.shape[1]
    top_k_len    = 2048
    dims_per_lane: cutlass.Constexpr = DIMS_PER_LANE
    num_vec: cutlass.Constexpr = NUM_VEC
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE

    bidx, bidy, _ = cute.arch.block_idx()
    num_threads: cutlass.Constexpr = BLOCK_SIZE
    num_warps:   cutlass.Constexpr = NUM_WARPS
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    probe_row = bidx * num_heads + bidy
    sm = cutlass.Int64(smid_u32())
    probe_cnt = cutlass.Int32(0)

    allocator = cutlass.utils.SmemAllocator()
    smem_logits  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((top_k_len,),    stride=(1,)), 16, None)
    smem_sparse  = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((top_k_len,),    stride=(1,)),  4, None)
    smem_red_i32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),           stride=(1,)),  4, None)
    smem_red_f32 = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),           stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)

    # ── Phase 1: Load ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["load"])

    partial_cnt_valid = 0
    for i in range(tidx, top_k_len, num_threads):
        idx = sparse_indices[bidx, i]
        smem_sparse[i] = idx
        if idx >= cutlass.Int32(0):
            partial_cnt_valid += 1
    for i in range(tidx, head_dim_ckv, num_threads):
        smem_q_nope[i] = q_nope[bidx, bidy, i]
    for i in range(tidx, head_dim_kpe, num_threads):
        smem_q_pe[i] = q_pe[bidx, bidy, i]

    # ── Phase 2: Valid count ──
    sum_valid = warp_reduce(partial_cnt_valid, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_red_i32[warp_idx] = sum_valid
    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["valid_count"])

    if warp_idx == 0:
        val = smem_red_i32[lane_idx]
        sum_valid = warp_reduce(val, lambda a, b: a + b, width=num_warps)
        smem_red_i32[0] = sum_valid
    cute.arch.sync_threads()

    valid_count = smem_red_i32[0]
    num_rounds  = (valid_count + num_warps - 1) // num_warps

    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["score"])

    # ── Phase 3: Score (vectorized LDG.128) ──
    q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec,))

    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < valid_count:
            cur_idx = smem_sparse[sparse_idx]

            ckv_row = ckv_cache[cur_idx, None]
            ckv_z   = cute.zipped_divide(ckv_row, (num_vec,))

            sum_partial = cutlass.Float32(0)
            for it in range(iters_per_lane):
                group  = it * wsize + lane_idx
                q_frag = q_nope_z[(None, (group,))].load()
                K_frag = ckv_z[(None, (group,))].load()
                sumSSA = q_frag * K_frag
                partial = cutlass.Float32(
                    sumSSA.reduce(cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0)
                )
                sum_partial = sum_partial + partial

            for k_idx in range(head_dim_kpe // wsize):
                q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial += q_p * kv

            s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_logits[sparse_idx] = s * sm_scale
    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["softmax_max"])

    # ── Phase 4: Softmax max ──
    partial_max = -cutlass.Float32(math.inf)
    for idx in range(tidx, valid_count, num_threads):
        v = smem_logits[idx]
        if v > partial_max:
            partial_max = v
    max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
    if lane_idx == 0:
        smem_red_f32[warp_idx] = max_val
    cute.arch.sync_threads()
    if warp_idx == 0:
        val = smem_red_f32[lane_idx]
        max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
        smem_red_f32[0] = max_val
    cute.arch.sync_threads()
    row_max = smem_red_f32[0]
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["softmax_exp_sum"])

    # ── Phase 5: Softmax exp+sum+writeback (fused — no separate normalize) ──
    partial_sum = cutlass.Float32(0)
    for idx in range(tidx, valid_count, num_threads):
        e = cute.math.exp(smem_logits[idx] - row_max)
        smem_logits[idx] = e
        partial_sum += e
    sum_val = warp_reduce(partial_sum, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_red_f32[warp_idx] = sum_val
    cute.arch.sync_threads()
    if warp_idx == 0:
        val = smem_red_f32[lane_idx]
        sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
        smem_red_f32[0] = sum_val
    cute.arch.sync_threads()
    row_sum = smem_red_f32[0]
    if tidx == 0:
        lse[bidx, bidy] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
    # No separate normalize pass — divide inline in output loop
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["output"])

    # ── Phase 6: Output (vectorized LDG.128) ──
    out_regs = cute.make_rmem_tensor(
        cute.make_layout((dims_per_lane,), stride=(1,)),
        cutlass.Float32,
    )
    for k in range(dims_per_lane):
        out_regs[k] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        j = round_idx * num_warps + warp_idx
        if j < valid_count:
            kv_idx = smem_sparse[j]
            weight = smem_logits[j] / row_sum

            V_row = ckv_cache[kv_idx, None]
            V_z   = cute.zipped_divide(V_row, (num_vec,))

            for it in range(iters_per_lane):
                group = it * wsize + lane_idx
                frag  = V_z[(None, (group,))].load()
                for v in range(num_vec):
                    out_regs[it * num_vec + v] += weight * cutlass.Float32(frag[v])

    # Write to smem_partial
    for it in range(iters_per_lane):
        for v in range(num_vec):
            smem_partial[warp_idx, (it * wsize + lane_idx) * num_vec + v] = out_regs[it * num_vec + v]
    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["reduce"])

    # ── Phase 7: Cross-warp reduce → global output (no smem_output) ──
    for i in range(tidx, head_dim_ckv, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        output[bidx, bidy, i] = cutlass.BFloat16(acc)
    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_finalize(probe, probe_row, probe_cnt)


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)

def compile_kernel():
    T = cute.sym_int(); N = cute.sym_int(); B = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048
    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (N, head_dim_ckv),            (1, 0),    16)
    kpe_cache      = _fake(cute.BFloat16, (N, head_dim_kpe),            (1, 0),    16)
    sparse_indices = _fake(cute.Int32,    (T, top_k_len),               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    probe          = _fake(cute.Int64,    (B, PROBE_COLS),              (1, 0),     8)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        fused_dsa_thr_warpv2_profiled,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        output, lse, probe, stream,
        options="--enable-tvm-ffi"
    )

def run_single(workload_idx: int) -> str:
    import os
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors
    H, D_ckv = 16, 512
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling profiled fused_tiny_thr_warpv2 kernel...")
    compiled = compile_kernel()
    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [json.loads(l) for l in open(JSONL)]
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
