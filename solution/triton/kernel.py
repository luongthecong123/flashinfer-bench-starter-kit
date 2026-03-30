import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream


# ── Shared warp reduction ─────────────────────────────────────────────────────

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ── CuTe DSL Gather Kernel ────────────────────────────────────────────────────

class Gather():
    def __init__(self):
        self.BM = 2
        self.num_threads = 256
        self.warp_size = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        ckv_cache: cute.Tensor,
        kpe_cache: cute.Tensor,
        sparse_indices: cute.Tensor,
        kc: cute.Tensor,
        Kp: cute.Tensor,
        max_valid: cute.Tensor,
        stream,
    ):
        T, topk = sparse_indices.shape
        self.kernel(ckv_cache, kpe_cache, sparse_indices, kc, Kp, max_valid).launch(
            grid=[T, topk // self.BM, 1], block=[self.num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        ckv_cache: cute.Tensor,
        kpe_cache: cute.Tensor,
        sparse_indices: cute.Tensor,
        kc: cute.Tensor,
        Kp: cute.Tensor,
        max_valid: cute.Tensor,
    ):
        N, dkc = ckv_cache.shape
        N2, dkp = kpe_cache.shape
        T, topk = sparse_indices.shape

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        for m in range(self.BM):
            row = bidy * self.BM + m
            idx = sparse_indices[bidx, row]

            if idx >= cutlass.Int32(0):
                for d in range(tidx, dkc, self.num_threads):
                    kc[bidx, row, d] = ckv_cache[idx, d]
                for d in range(tidx, dkp, self.num_threads):
                    Kp[bidx, row, d] = kpe_cache[idx, d]
            else:
                for d in range(tidx, dkc, self.num_threads):
                    kc[bidx, row, d] = cutlass.BFloat16(0)
                for d in range(tidx, dkp, self.num_threads):
                    Kp[bidx, row, d] = cutlass.BFloat16(0)

        if bidy == 0:
            allocator = cutlass.utils.SmemAllocator()
            smem_counts = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout((self.warp_size,), stride=(1,)), 4, None
            )

            local_count = cutlass.Int32(0)
            for i in range(topk // self.num_threads):
                if sparse_indices[bidx, tidx * (topk // self.num_threads) + i] >= cutlass.Int32(0):
                    local_count += cutlass.Int32(1)

            warp_sum = warp_reduce(local_count, lambda a, b: a + b)

            if lane_idx == 0:
                smem_counts[warp_idx] = warp_sum

            cute.arch.sync_threads()

            if warp_idx == 0:
                partial = smem_counts[lane_idx]
                total = warp_reduce(partial, lambda a, b: a + b, width=self.num_threads // self.warp_size)
                if lane_idx == 0:
                    max_valid[bidx] = total


def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def _compile_gather():
    T = cute.sym_int()
    N = cute.sym_int()
    return cute.compile(
        Gather(),
        _fake(cute.BFloat16, (N, 512), (1, 0), 16),
        _fake(cute.BFloat16, (N, 64), (1, 0), 16),
        _fake(cute.Int32, (T, 2048), (1, 0), 4),
        _fake(cute.BFloat16, (T, 2048, 512), (2, 1, 0), 16),
        _fake(cute.BFloat16, (T, 2048, 64), (2, 1, 0), 16),
        _fake(cute.Int32, (T,), (0,), 4),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi"
    )


gather_compiled = _compile_gather()


# ── CuTe DSL Fused Tiny Kernel (32-warp, for small T) ────────────────────────
# Grid: [T, H, 1]  Block: 1024 threads = 32 warps
# Each warp scores 1 key independently; 64 rounds cover 2048 keys.

@cute.jit
def fused_dsa_v2(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor,
    stream,
):
    T, num_heads, head_dim_ckv = q_nope.shape
    fused_dsa_kernel_v2(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(grid=[T, num_heads, 1], block=[1024, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_v2(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor,
):
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

    allocator = cutlass.utils.SmemAllocator()
    smem_score_nope   = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_score_pe     = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_logits_scaled = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_sparse_idx   = allocator.allocate_tensor(cutlass.Int32,   cute.make_layout((top_k_len), stride=(1)), 4,  None)
    smem_valid_count  = allocator.allocate_tensor(cutlass.Int32,   cute.make_layout((1),         stride=(1)), 4,  None)

    for i in range(tidx, top_k_len, num_threads):
        smem_sparse_idx[i] = sparse_indices[bidx, i]
    cute.arch.sync_threads()

    q_nope_local = q_nope[bidx, bidy, None]
    q_pe_local   = q_pe[bidx, bidy, None]

    for round_idx in range(top_k_len // num_warps):
        sparse_idx = round_idx * num_warps + warp_idx
        cur_idx = smem_sparse_idx[sparse_idx]

        if cur_idx >= cutlass.Int32(0):
            lane_idx = cute.arch.lane_idx()

            sum_partial_nope = cutlass.Float32(0)
            for k_idx in range(head_dim_ckv // wsize):
                sum_partial_nope += cutlass.Float32(q_nope_local[k_idx * wsize + lane_idx]) * \
                                    cutlass.Float32(ckv_cache[cur_idx, k_idx * wsize + lane_idx])
            sum_nope = warp_reduce(sum_partial_nope, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_nope[sparse_idx] = sum_nope

            sum_partial_pe = cutlass.Float32(0)
            for k_idx in range(head_dim_kpe // wsize):
                sum_partial_pe += cutlass.Float32(q_pe_local[k_idx * wsize + lane_idx]) * \
                                  cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
            sum_pe = warp_reduce(sum_partial_pe, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_pe[sparse_idx] = sum_pe

    cute.arch.sync_threads()

    if tidx == 0:
        num_valid = cutlass.Int32(0)
        for i in range(top_k_len):
            if smem_sparse_idx[i] >= cutlass.Int32(0):
                num_valid = cutlass.Int32(i + 1)
        smem_valid_count[0] = num_valid
    cute.arch.sync_threads()
    valid_count = smem_valid_count[0]

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

    for i in range(tidx, head_dim_ckv, num_threads):
        output[bidx, bidy, i] = cutlass.BFloat16(smem_output[i])


def _compile_fused_tiny():
    T = cute.sym_int()
    N = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048
    return cute.compile(
        fused_dsa_v2,
        _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16),
        _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16),
        _fake(cute.BFloat16, (N, head_dim_ckv),            (1, 0),    16),
        _fake(cute.BFloat16, (N, head_dim_kpe),            (1, 0),    16),
        _fake(cute.Int32,    (T, top_k_len),               (1, 0),    4),
        0.1352337788608801,
        _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16),
        _fake(cute.Float32,  (T, num_heads),               (1, 0),    4),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi"
    )


fused_tiny_compiled = _compile_fused_tiny()


# ── Torch Compile Attention ────────────────────────────────────────────────────

@torch.compile
def compute_attention_batched(qn, qp, Kc, Kp, mask, sm_scale, output, lse):
    logits = torch.bmm(qn, Kc.transpose(1, 2), out_dtype=torch.float32) + \
             torch.bmm(qp, Kp.transpose(1, 2), out_dtype=torch.float32)
    logits.masked_fill_(mask.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale
    lse.copy_(torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0))
    attn = torch.softmax(logits_scaled, dim=-1)
    output.copy_(torch.bmm(attn.float(), Kc.float()).to(torch.bfloat16))


# ── Entry Point ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    T = q_nope.shape[0]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    topk = sparse_indices.shape[-1]

    ckv_flat = ckv_cache.reshape(-1, head_dim_ckv)
    kpe_flat = kpe_cache.reshape(-1, head_dim_kpe)

    if T < 3:
        # Small batch: single fused per-token kernel (32-warp, no gather overhead)
        fused_tiny_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)
    else:
        # Large batch: parallel gather + batched torch.compile attention
        Kc = torch.empty(T, topk, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
        Kp = torch.empty(T, topk, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
        max_valid = torch.empty(T, dtype=torch.int32, device="cuda")
        gather_compiled(ckv_flat, kpe_flat, sparse_indices, Kc, Kp, max_valid)
        mask = sparse_indices == -1
        compute_attention_batched(q_nope, q_pe, Kc, Kp, mask, sm_scale, output, lse)
