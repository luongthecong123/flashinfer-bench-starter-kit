import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch
import math

# ── redux ──────────────────────────────────────────────────────────────────────

# @cute.jit
# def test_redux(input, output):
#     reduction_kernel_redux(input, output).launch(grid=[1], block=[32, 1, 1])

# @cute.kernel
# def reduction_kernel_redux(input: cute.Tensor, output: cute.Tensor):
#     lane_idx = cute.arch.lane_idx()
#     result = cute.arch.warp_redux_sync(value=input[lane_idx], kind="fmax")
#     if lane_idx == 0:
#         output[lane_idx] = result

# ── shuffle ────────────────────────────────────────────────────────────────────

@cute.jit
def test_shuffle(input, output):
    reduction_kernel_shuffle(input, output).launch(grid=[1], block=[32, 1, 1])

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        # cute.arch.shuffle_sync_bfly will read from another thread's registers
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val
    
# @cute.kernel
# def reduction_kernel_shuffle(input: cute.Tensor, output: cute.Tensor):
#     lane_idx = cute.arch.lane_idx()
#     result = warp_reduce(input[lane_idx], cute.arch.fmax, width=32)
#     if lane_idx == 0:
#         output[lane_idx] = result

@cute.kernel
def reduction_kernel_shuffle(input: cute.Tensor, output: cute.Tensor):
    lane_idx = cute.arch.lane_idx()
    result = warp_reduce(input[lane_idx], lambda a, b: a + b, width=32)
    if lane_idx == 0:
        output[lane_idx] = result

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    a = torch.randn(32, device='cuda', dtype=torch.float32)
    out = torch.zeros(32, device='cuda', dtype=torch.float32)
    a_   = from_dlpack(a,   assumed_align=16)
    out_ = from_dlpack(out, assumed_align=16)

    print("input:    ", a)
    print("torch max:", a.max().item())
    print("torch sum:", a.sum().item())

    # # redux
    # compiled = cute.compile(test_redux, a_, out_)
    # compiled(a_, out_)
    # print("redux result:", out[0].item())
    # time = benchmark(compiled, kernel_arguments=JitArguments(a_, out_))
    # print(f"redux    : {time:>5.4f} µs")

    # shuffle
    compiled = cute.compile(test_shuffle, a_, out_)
    compiled(a_, out_)
    print("shuffle result:", out[0].item())
    time = benchmark(compiled, kernel_arguments=JitArguments(a_, out_))
    print(f"shuffle  : {time:>5.4f} µs")

if __name__ == "__main__":
    main()