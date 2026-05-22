"""
test_grid_sync_pdl.py — Grid-wide softmax via PDL (Programmatic Dependent Launch).

PDL pattern (from flashinfer mla_decode_fp16.py):
  Kernel 1: partial max+sum per split
            → ends with cute.arch.griddepcontrol_launch_dependents()
  Kernel 2: global reduce
            → starts with cute.arch.griddepcontrol_wait()
  Both launched with use_pdl=True in .launch()

Benefit over naive two-kernel: kernel 2 blocks can start occupying SM resources
and waiting at griddepcontrol_wait() while kernel 1 is still finishing, reducing
kernel-boundary dead time.

Correctness checked against torch.logsumexp.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, from_dlpack
import torch
import math

N          = 2048
NUM_SPLITS = 8
SPLIT_SIZE = N // NUM_SPLITS   # 256
BLOCK_THR  = 256
NUM_WARPS  = BLOCK_THR // 32   # 8


# ── helpers ───────────────────────────────────────────────────────────────────

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ── Kernel 1: partial max + sum per split, signals PDL at end ────────────────

@cute.kernel
def partial_softmax_kernel_pdl(
    x:           cute.Tensor,   # (2048,)   f32
    partial_lse: cute.Tensor,   # (8, 2)    f32  [:, 0]=max  [:, 1]=sum
):
    num_warps: cutlass.Constexpr = NUM_WARPS
    split_size: cutlass.Constexpr = SPLIT_SIZE
    block_thr: cutlass.Constexpr = BLOCK_THR

    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane_idx    = cute.arch.lane_idx()
    warp_idx    = cute.arch.warp_idx()
    warp_idx    = cute.arch.make_warp_uniform(warp_idx)
    split_start = bidx * split_size

    alloc    = cutlass.utils.SmemAllocator()
    smem_red = alloc.allocate_tensor(
        cutlass.Float32, cute.make_layout((num_warps,), stride=(1,)), 16, None)

    # ── block-wide max ────────────────────────────────────────────────────────
    local_max = -cutlass.Float32(math.inf)
    for i in range(tidx, split_size, block_thr):
        v = cutlass.Float32(x[split_start + i])
        if v > local_max:
            local_max = v

    wmax = warp_reduce(local_max, lambda a, b: a if a > b else b)
    if lane_idx == 0:
        smem_red[warp_idx] = wmax
    cute.arch.sync_threads()
    if warp_idx == 0:
        wmax = warp_reduce(smem_red[lane_idx], lambda a, b: a if a > b else b, width=num_warps)
        smem_red[0] = wmax
    cute.arch.sync_threads()
    block_max = smem_red[0]

    # ── block-wide sum(exp) ───────────────────────────────────────────────────
    local_sum = cutlass.Float32(0)
    for i in range(tidx, split_size, block_thr):
        local_sum += cute.math.exp(cutlass.Float32(x[split_start + i]) - block_max)

    wsum = warp_reduce(local_sum, lambda a, b: a + b)
    if lane_idx == 0:
        smem_red[warp_idx] = wsum
    cute.arch.sync_threads()
    if warp_idx == 0:
        wsum = warp_reduce(smem_red[lane_idx], lambda a, b: a + b, width=num_warps)
        smem_red[0] = wsum
    cute.arch.sync_threads()
    block_sum = smem_red[0]

    if tidx == 0:
        partial_lse[bidx, 0] = block_max
        partial_lse[bidx, 1] = block_sum

    # ── PDL: signal dependent kernel it is safe to read our outputs ──────────
    cute.arch.griddepcontrol_launch_dependents()


# ── Kernel 2: global reduce — waits for PDL signal before reading ─────────────

@cute.kernel
def reduce_lse_kernel_pdl(
    partial_lse: cute.Tensor,   # (8, 2)    f32
    lse_out:     cute.Tensor,   # (1,)      f32
):
    num_splits: cutlass.Constexpr = NUM_SPLITS
    tidx, _, _ = cute.arch.thread_idx()

    # ── PDL: wait until kernel 1 has written partial_lse ─────────────────────
    cute.arch.griddepcontrol_wait()

    if tidx == 0:
        g_max = -cutlass.Float32(math.inf)
        for s in range(num_splits):
            m = partial_lse[s, 0]
            if m > g_max:
                g_max = m

        g_sum = cutlass.Float32(0)
        for s in range(num_splits):
            g_sum += partial_lse[s, 1] * cute.math.exp(partial_lse[s, 0] - g_max)

        lse_out[0] = g_max + cute.math.log(g_sum)


# ── JIT entry point ───────────────────────────────────────────────────────────

@cute.jit
def run_softmax_pdl(
    x:           cute.Tensor,
    partial_lse: cute.Tensor,
    lse_out:     cute.Tensor,
    stream,
):
    partial_softmax_kernel_pdl(x, partial_lse).launch(
        grid=[NUM_SPLITS, 1, 1], block=[BLOCK_THR, 1, 1],
        stream=stream, use_pdl=True)
    reduce_lse_kernel_pdl(partial_lse, lse_out).launch(
        grid=[1, 1, 1], block=[1, 1, 1],
        stream=stream, use_pdl=True)


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)

def compile_softmax_pdl():
    x           = _fake(cute.Float32, (N,),            (0,),    4)
    partial_lse = _fake(cute.Float32, (NUM_SPLITS, 2), (1, 0),  4)
    lse_out     = _fake(cute.Float32, (1,),            (0,),    4)
    stream      = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(run_softmax_pdl, x, partial_lse, lse_out, stream,
                        options="--enable-tvm-ffi")

_compiled = compile_softmax_pdl()


# ── Correctness check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    x_t           = torch.randn(N, device="cuda", dtype=torch.float32)
    partial_lse_t = torch.empty(NUM_SPLITS, 2, device="cuda", dtype=torch.float32)
    lse_out_t     = torch.empty(1, device="cuda", dtype=torch.float32)

    x_c           = from_dlpack(x_t,           assumed_align=4, enable_tvm_ffi=True)
    partial_lse_c = from_dlpack(partial_lse_t,  assumed_align=4, enable_tvm_ffi=True)
    lse_out_c     = from_dlpack(lse_out_t,      assumed_align=4, enable_tvm_ffi=True)

    _compiled(x_c, partial_lse_c, lse_out_c)
    torch.cuda.synchronize()

    ref = torch.logsumexp(x_t, dim=0)

    print(f"our lse : {lse_out_t.item():.6f}")
    print(f"ref lse : {ref.item():.6f}")
    print(f"abs err : {abs(lse_out_t.item() - ref.item()):.2e}")
    print("CORRECTNESS PASS" if abs(lse_out_t.item() - ref.item()) < 1e-4 else "MISMATCH!")
