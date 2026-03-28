import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream


# ── CuTe DSL Gather Kernel ────────────────────────────────────────────────────

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


_GATHER_BM = 2
_GATHER_THREADS = 256
_GATHER_WARP_SIZE = cute.arch.WARP_SIZE


class Gather():
    def __init__(self):
        self.BM = _GATHER_BM
        self.num_threads = _GATHER_THREADS
        self.warp_size = _GATHER_WARP_SIZE

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
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]
    T = num_tokens

    # Flatten paged KV cache: [num_pages, 64, D] → [N, D]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)

    # CuTe DSL gather kernel
    Kc = torch.empty(T, topk, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
    Kp = torch.empty(T, topk, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
    max_valid = torch.empty(T, dtype=torch.int32, device="cuda")
    gather_compiled(Kc_all, Kp_all, sparse_indices, Kc, Kp, max_valid)

    # Build mask and run batched attention
    mask = sparse_indices == -1
    compute_attention_batched(q_nope, q_pe, Kc, Kp, mask, sm_scale, output, lse)
