"""
mla_reduction.py — Dense MLA attention using FastGEMV with BF16 HFMA2.

Follows the algorithm from kv_split_xor_pdl_v3_pro_v2_bf16.py adapted for
dense (non-sparse) KV cache with SEQ_LEN=64.

Grid: [NUM_HEADS, 1, T].  Block: 1024 threads (32 warps).

Score phase: FastGEMV with BF16 HFMA2 inner CKV dot + FP32 KPE dot.
  ROWS_PER_WARP=2, 32 warps × 2 rows = 64 = SEQ_LEN per round.
Softmax: Block-wide max → exp → sum.
Output: Warp-parallel weighted CKV accumulation + block-wide reduction.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

NUM_HEADS    = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
SEQ_LEN      = 64

LN2 = 0.6931471805599453

NUM_THREADS = 1024
NUM_WARPS   = NUM_THREADS // 32

VEC_SIZE_CKV = 8
VEC_SIZE_KPE = 2
VEC_SIZE_OUT = 16
ITERS_PER_LANE_CKV = HEAD_DIM_CKV // (32 * VEC_SIZE_CKV)  # 2

# FastGEMV score constants
ROWS_PER_WARP = 2
ROWS_PER_ROUND_SCORE = NUM_WARPS * ROWS_PER_WARP  # 64

# BF16 accumulation chunk size
NUM_VEC_BF16 = 8


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Host JIT
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def mla_reduction(
    q_nope:   cute.Tensor,    # [T, H, D]  bf16
    q_pe:     cute.Tensor,    # [T, H, Dp] bf16
    kc:       cute.Tensor,    # [T, S, D]  bf16
    kp:       cute.Tensor,    # [T, S, Dp] bf16
    sm_scale: cute.Tensor,    # [1]        float32
    output:   cute.Tensor,    # [T, H, D]  bf16
    lse:      cute.Tensor,    # [T, H]     float32
    stream,
):
    T, num_heads, head_dim_ckv = q_nope.shape

    mla_compute_kernel(
        q_nope, q_pe, kc, kp, sm_scale, output, lse,
    ).launch(
        grid=[NUM_HEADS, 1, T],
        block=[NUM_THREADS, 1, 1],
        stream=stream,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel: FastGEMV score + softmax + output accumulation
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def mla_compute_kernel(
    q_nope:   cute.Tensor,    # [T, H, D]  bf16
    q_pe:     cute.Tensor,    # [T, H, Dp] bf16
    kc:       cute.Tensor,    # [T, S, D]  bf16
    kp:       cute.Tensor,    # [T, S, Dp] bf16
    sm_scale: cute.Tensor,    # [1]        float32
    output:   cute.Tensor,    # [T, H, D]  bf16
    lse:      cute.Tensor,    # [T, H]     float32
):
    _, S, _ = kc.shape
    _, _, dkc = q_nope.shape
    _, _, dkp = q_pe.shape

    head_dim_ckv:   cutlass.Constexpr = HEAD_DIM_CKV
    head_dim_kpe:   cutlass.Constexpr = HEAD_DIM_KPE
    seq_len:        cutlass.Constexpr = SEQ_LEN
    num_threads:    cutlass.Constexpr = NUM_THREADS
    num_warps:      cutlass.Constexpr = NUM_WARPS
    vec_size_ckv:   cutlass.Constexpr = VEC_SIZE_CKV
    vec_size_kpe:   cutlass.Constexpr = VEC_SIZE_KPE
    vec_size_out:   cutlass.Constexpr = VEC_SIZE_OUT
    iters_per_lane_ckv: cutlass.Constexpr = ITERS_PER_LANE_CKV
    rows_per_warp:  cutlass.Constexpr = ROWS_PER_WARP
    rows_per_round_score: cutlass.Constexpr = ROWS_PER_ROUND_SCORE
    num_vec_bf16:   cutlass.Constexpr = NUM_VEC_BF16

    head_idx, _, batch_idx = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize = cute.arch.WARP_SIZE

    # ── SMEM allocation ──────────────────────────────────────────────────────
    alloc = cutlass.utils.SmemAllocator()
    smem_logits      = _smem(alloc, cutlass.Float32,  (seq_len,),                    (1,),              16)
    smem_max_red_f32 = _smem(alloc, cutlass.Float32,  (32,),                         (1,),              16)
    smem_sum_red_f32 = _smem(alloc, cutlass.Float32,  (32,),                         (1,),              16)
    smem_q_nope      = _smem(alloc, cutlass.BFloat16, (1, head_dim_ckv),             (head_dim_ckv, 1), 16)
    smem_q_pe        = _smem(alloc, cutlass.BFloat16, (1, head_dim_kpe),             (head_dim_kpe, 1), 16)
    smem_partial     = _smem(alloc, cutlass.Float32,  (num_warps, head_dim_ckv),     (head_dim_ckv, 1), 16)
    smem_out         = _smem(alloc, cutlass.Float32,  (head_dim_ckv,),               (1,),              16)

    # ── Load Q into smem ─────────────────────────────────────────────────────
    for i in range(tidx, head_dim_ckv, num_threads):
        smem_q_nope[0, i] = q_nope[batch_idx, head_idx, i]
    for i in range(tidx, head_dim_kpe, num_threads):
        smem_q_pe[0, i] = q_pe[batch_idx, head_idx, i]
    cute.arch.sync_threads()

    # ── GMEM views for this batch ────────────────────────────────────────────
    kc_batch = kc[batch_idx, None, None]    # (S, D)
    kp_batch = kp[batch_idx, None, None]    # (S, Dp)

    # ── Vectorized views for BF16 score ──────────────────────────────────────
    smem_q_nope_ = cute.zipped_divide(smem_q_nope, (1, num_vec_bf16))
    kc_batch_    = cute.zipped_divide(kc_batch,    (1, num_vec_bf16))
    kp_batch_    = cute.zipped_divide(kp_batch,    (1, vec_size_kpe))
    smem_q_pe_   = cute.zipped_divide(smem_q_pe,   (1, vec_size_kpe))

    # Output phase uses vec_size_ckv tiling
    kc_batch_out_ = cute.zipped_divide(kc_batch, (1, vec_size_ckv))

    iters_per_lane_bf16: cutlass.Constexpr = HEAD_DIM_CKV // (32 * NUM_VEC_BF16)

    # ══════════════════════════════════════════════════════════════════════════
    # Score (FastGEMV: 2-row interleaved, BF16 HFMA2 accum)
    # 32 warps × 2 rows = 64 = SEQ_LEN → single round
    # ══════════════════════════════════════════════════════════════════════════
    num_rounds_score: cutlass.Constexpr = (SEQ_LEN + ROWS_PER_ROUND_SCORE - 1) // ROWS_PER_ROUND_SCORE

    for round_idx in range(num_rounds_score):
        base_row = round_idx * rows_per_round_score + warp_idx * rows_per_warp

        row0 = base_row + 0
        row1 = base_row + 1

        ckv_row0 = kc_batch_[(0, None), (row0, None)]
        ckv_row1 = kc_batch_[(0, None), (row1, None)]

        kpe_row0 = kp_batch_[(0, None), (row0, None)]
        kpe_row1 = kp_batch_[(0, None), (row1, None)]

        # FP32 accumulators — promoted from BF16 chunks
        sums = cute.make_rmem_tensor(
            cute.make_layout((rows_per_warp,), stride=(1,)),
            cutlass.Float32,
        )
        for r in range(rows_per_warp):
            sums[r] = cutlass.Float32(0)

        # CKV dot products — BF16 HFMA2 inner loop with periodic FP32 promotion
        for it in range(iters_per_lane_bf16):
            rest_idx = it * wsize + lane_idx
            qn_frag = smem_q_nope_[(0, None), (0, rest_idx)].load()

            ckv_f0 = ckv_row0[None, rest_idx].load()
            ckv_f1 = ckv_row1[None, rest_idx].load()

            # BF16 partial sums within this chunk
            p0 = cutlass.BFloat16(0)
            p1 = cutlass.BFloat16(0)

            for v in range(num_vec_bf16):
                qv = qn_frag[v]
                p0 = p0 + qv * ckv_f0[v]
                p1 = p1 + qv * ckv_f1[v]

            # Promote to FP32 after each chunk
            sums[0] = sums[0] + cutlass.Float32(p0)
            sums[1] = sums[1] + cutlass.Float32(p1)

        # KPE dot products — interleaved (FP32 accum, small dim)
        qp_frag = smem_q_pe_[(0, None), (0, lane_idx)].load()
        kpe_f0 = kpe_row0[None, lane_idx].load()
        kpe_f1 = kpe_row1[None, lane_idx].load()
        for v in range(vec_size_kpe):
            qv = cutlass.Float32(qp_frag[v])
            sums[0] = sums[0] + qv * cutlass.Float32(kpe_f0[v])
            sums[1] = sums[1] + qv * cutlass.Float32(kpe_f1[v])

        # Batched warp reduction (FP32)
        for r in range(rows_per_warp):
            sums[r] = warp_reduce(sums[r], lambda a, b: a + b, width=32)
        if lane_idx == 0:
            for r in range(rows_per_warp):
                if base_row + r < seq_len:
                    smem_logits[base_row + r] = sums[r] * sm_scale[0]

    cute.arch.sync_threads()

    # ══════════════════════════════════════════════════════════════════════════
    # Softmax: max
    # ══════════════════════════════════════════════════════════════════════════
    partial_max = -cutlass.Float32(math.inf)
    for idx in range(tidx, seq_len, num_threads):
        v = smem_logits[idx]
        if v > partial_max:
            partial_max = v

    max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
    if lane_idx == 0:
        smem_max_red_f32[warp_idx] = max_val
    cute.arch.sync_threads()
    if warp_idx == 0:
        val = smem_max_red_f32[lane_idx]
        max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
        smem_max_red_f32[0] = max_val
    cute.arch.sync_threads()

    row_max = smem_max_red_f32[0]

    # ══════════════════════════════════════════════════════════════════════════
    # Softmax: exp + sum
    # ══════════════════════════════════════════════════════════════════════════
    local_sum = cutlass.Float32(0)
    for idx in range(tidx, seq_len, num_threads):
        e = cute.math.exp(smem_logits[idx] - row_max)
        smem_logits[idx] = e
        local_sum += e

    sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_sum_red_f32[warp_idx] = sum_val
    cute.arch.sync_threads()
    if warp_idx == 0:
        val = smem_sum_red_f32[lane_idx]
        sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
        smem_sum_red_f32[0] = sum_val
    cute.arch.sync_threads()

    row_sum = smem_sum_red_f32[0]

    # Write LSE
    if tidx == 0:
        lse[batch_idx, head_idx] = (
            (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Output: warp-parallel weighted CKV accumulation
    # 32 warps, 64 rows → 2 rounds of 32
    # ══════════════════════════════════════════════════════════════════════════
    num_rounds: cutlass.Constexpr = (SEQ_LEN + NUM_WARPS - 1) // NUM_WARPS  # 2

    out_regs = cute.make_rmem_tensor(
        cute.make_layout((vec_size_out,), stride=(1,)), cutlass.Float32)
    for i in range(vec_size_out):
        out_regs[i] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        seq_idx = round_idx * num_warps + warp_idx
        if seq_idx < seq_len:
            ckv_row_ = kc_batch_out_[(0, None), (seq_idx, None)]
            e = smem_logits[seq_idx]

            for it in range(iters_per_lane_ckv):
                rest_idx = it * wsize + lane_idx
                ckv_vec = ckv_row_[None, rest_idx].load()
                for i in range(vec_size_ckv):
                    out_regs[it * vec_size_ckv + i] += e * cutlass.Float32(ckv_vec[i])

    # Write per-warp partials to smem
    for it in range(iters_per_lane_ckv):
        for v in range(vec_size_ckv):
            smem_partial[warp_idx, (it * wsize + lane_idx) * vec_size_ckv + v] = out_regs[it * vec_size_ckv + v]

    cute.arch.sync_threads()

    # Block-wide reduction across warps
    for i in range(tidx, head_dim_ckv, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        smem_out[i] = acc
    cute.arch.sync_threads()

    # Normalize and write output
    for i in range(tidx, head_dim_ckv, num_threads):
        output[batch_idx, head_idx, i] = cutlass.BFloat16(smem_out[i] / row_sum)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape,
        stride_order=stride_order, assumed_align=assumed_align)


def _compile():
    T = cute.sym_int()

    q_nope   = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    q_pe     = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
    kc       = _fake(cute.BFloat16, (T, SEQ_LEN, HEAD_DIM_CKV),   (2, 1, 0), 16)
    kp       = _fake(cute.BFloat16, (T, SEQ_LEN, HEAD_DIM_KPE),   (2, 1, 0), 16)
    sm_scale = _fake(cute.Float32,  (1,),                         (0,),       4)
    output   = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    lse      = _fake(cute.Float32,  (T, NUM_HEADS),               (1, 0),     4)
    stream   = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        mla_reduction,
        q_nope, q_pe, kc, kp, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


_compiled = _compile()


# ═══════════════════════════════════════════════════════════════════════════════
# Host wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def run(q_nope, q_pe, kc, kp, sm_scale, output, lse):
    sm_scale_t = torch.tensor([sm_scale], dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, kc, kp, sm_scale_t, output, lse)


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch reference
# ═══════════════════════════════════════════════════════════════════════════════

def ref_run(q_nope, q_pe, kc, kp, sm_scale):
    """Numerically stable reference for dense MLA attention.

    q_nope : [T, H, D]   bf16
    q_pe   : [T, H, Dp]  bf16
    kc     : [T, S, D]   bf16  (per-request KV cache, also used as values)
    kp     : [T, S, Dp]  bf16

    Returns (output [T, H, D] bf16, lse [T, H] float32).
    """
    q_n = q_nope.float()   # [T, H,  D]
    q_p = q_pe.float()     # [T, H, Dp]
    k_c = kc.float()       # [T, S,  D]
    k_p = kp.float()       # [T, S, Dp]

    # scores [T, H, S]
    scores = (
        torch.einsum("thd,tsd->ths", q_n, k_c)
        + torch.einsum("thp,tsp->ths", q_p, k_p)
    ) * sm_scale

    attn = torch.softmax(scores, dim=-1)          # [T, H, S]
    out  = torch.einsum("ths,tsd->thd", attn, k_c).to(torch.bfloat16)
    lse  = torch.logsumexp(scores, dim=-1)        # [T, H]  (log base e)
    return out, lse
