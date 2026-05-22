"""
Score kernel test: score = q_nope @ ckv^T + q_pe @ kpe^T

Shapes:
  q_nope: (T=8, 2, HEAD_DIM_CKV=512)   bf16
  q_pe:   (T=8, 2, HEAD_DIM_KPE=64)    bf16
  ckv:    (NUM_KV=128, HEAD_DIM_CKV=512) bf16
  kpe:    (NUM_KV=128, HEAD_DIM_KPE=64)  bf16
  output: (T=8, 2, NUM_KV=128)           f32

Thread block: NUM_WARPS=16 (512 threads).

Prologue:
  Each warp (0..15) owns row warp_idx of smem_qn and smem_qr.
  Warp loads via cp.async:
    - smem_qn row: vec_size=8 (ldg.128), 2 iters per lane
    - smem_qr row: vec_size=2 (ldg.32),  1 iter  per lane

Score loop:
  for T_idx in range(T):
    num_rounds = NUM_KV // NUM_WARPS = 8
    for round_idx in range_constexpr(num_rounds):
      col_idx = round_idx * NUM_WARPS + warp_idx
      acc0, acc1 = dot(smem_qn[T_idx*2+0,:], ckv[col_idx,:])
                 + dot(smem_qn[T_idx*2+1,:], ckv[col_idx,:])   (packed via fma_packed_f32x2)
      + same for KPE
      → warp-reduce → store to output[T_idx, 0/1, col_idx]
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

# ── constants ─────────────────────────────────────────────────────────────────
T:            cutlass.Constexpr = 8    # number of tokens
NUM_HEADS:    cutlass.Constexpr = 2    # heads per token (HEADS_PER_SPLIT)
NUM_ROWS:     cutlass.Constexpr = 16   # T * NUM_HEADS
HEAD_DIM_CKV: cutlass.Constexpr = 512
HEAD_DIM_KPE: cutlass.Constexpr = 64
NUM_KV:       cutlass.Constexpr = 128  # KV rows to score

VEC_SIZE_QN:  cutlass.Constexpr = 8   # bf16×8 = 128-bit ldg
VEC_SIZE_QR:  cutlass.Constexpr = 2   # bf16×2 = 32-bit ldg

NUM_WARPS:    cutlass.Constexpr = 16   # T * NUM_HEADS — one warp per row
NUM_THREADS:  cutlass.Constexpr = 512  # NUM_WARPS * 32

# chunks per lane when 1 warp loads 1 row
ITERS_QN_LOAD: cutlass.Constexpr = 2  # HEAD_DIM_CKV // (VEC_SIZE_QN * 32) = 512 // 256 = 2
# ITERS_QR_LOAD = HEAD_DIM_KPE // (VEC_SIZE_QR * 32) = 64 // 64 = 1  (no loop needed)

# score dot-product iters per lane
ITERS_CKV: cutlass.Constexpr = 2  # same as ITERS_QN_LOAD
# ITERS_KPE = 1  (no loop needed)

NUM_ROUNDS: cutlass.Constexpr = 8  # NUM_KV // NUM_WARPS


# ── warp-reduce helper ────────────────────────────────────────────────────────

@cute.jit
def warp_reduce_f32x2_add(
    val0: cutlass.Float32,
    val1: cutlass.Float32,
    width: cutlass.Constexpr = 32,
):
    for i in range(int(math.log2(width))):
        s0 = cute.arch.shuffle_sync_bfly(val0, offset=1 << i)
        s1 = cute.arch.shuffle_sync_bfly(val1, offset=1 << i)
        val0, val1 = cute.arch.add_packed_f32x2((val0, val1), (s0, s1))
    return val0, val1


# ── kernel ────────────────────────────────────────────────────────────────────

class TestScore:

    @cute.kernel
    def _kernel(
        self,
        q_nope: cute.Tensor,   # (T, 2, HEAD_DIM_CKV) bf16
        q_pe:   cute.Tensor,   # (T, 2, HEAD_DIM_KPE) bf16
        ckv:    cute.Tensor,   # (NUM_KV, HEAD_DIM_CKV) bf16
        kpe:    cute.Tensor,   # (NUM_KV, HEAD_DIM_KPE) bf16
        output: cute.Tensor,   # (T, 2, NUM_KV) f32
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        # ── smem ──────────────────────────────────────────────────────────────
        alloc   = cutlass.utils.SmemAllocator()
        smem_qn = alloc.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((T, NUM_HEADS, HEAD_DIM_CKV), stride=(NUM_HEADS * HEAD_DIM_CKV, HEAD_DIM_CKV, 1)),
            16, None,
        )
        smem_qr = alloc.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((T, NUM_HEADS, HEAD_DIM_KPE), stride=(NUM_HEADS * HEAD_DIM_KPE, HEAD_DIM_KPE, 1)),
            16, None,
        )

        # ── cp.async tiled copies ────────────────────────────────────────────
        # QN: 128-bit loads, 8×bf16.  thr_layout (1,32) → thread t maps to
        # column t in the flat row; val_layout (1,8) → 8 consecutive elements.
        # Thread t's base = t*8 bf16 = t*16 bytes — always 128-bit aligned.
        # 32 threads × 8 vals = 256 per tile; 2 tiles cover 512 dims.
        atom_qn = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        tiled_copy_qn = cute.make_tiled_copy_tv(atom_qn,
            cute.make_layout((32,), stride=(1,)),
            cute.make_layout((VEC_SIZE_QN,), stride=(1,)))
        lane_copy_qn  = tiled_copy_qn.get_slice(lane_idx)

        # QR: 32-bit loads, 2×bf16.  32 threads × 2 vals = 64 dims in 1 tile.
        atom_qr = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        tiled_copy_qr = cute.make_tiled_copy_tv(atom_qr,
            cute.make_layout((32,), stride=(1,)),
            cute.make_layout((VEC_SIZE_QR,), stride=(1,)))
        lane_copy_qr  = tiled_copy_qr.get_slice(lane_idx)

        # ── prologue: warps 0..T-1 each copy both heads of their T-row ──────
        # Must use 1D per-head slices: the 1D thr_layout (32,) + val (8,) maps
        # val-stride to the tensor's row-stride (512) when given a 2D (2,512)
        # slice, producing OOB accesses (8 "rows" but only 2 exist).
        # Looping over heads keeps slices 1D and the mapping correct.
        # T_row_idx = warp_idx % T  # safe index for all warps
        if warp_idx < T:
            T_row_idx = warp_idx
            for h in cutlass.range_constexpr(NUM_HEADS):
                cute.copy(atom_qn,
                          lane_copy_qn.partition_S(q_nope[T_row_idx, h, None]),
                          lane_copy_qn.partition_D(smem_qn[T_row_idx, h, None]))
                cute.copy(atom_qr,
                          lane_copy_qr.partition_S(q_pe[T_row_idx, h, None]),
                          lane_copy_qr.partition_D(smem_qr[T_row_idx, h, None]))

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # ── score computation ─────────────────────────────────────────────────
        smem_qn_z = cute.zipped_divide(smem_qn, (1, 1, VEC_SIZE_QN))  # tile=(1,1,8), rest=(T,2,64)
        smem_qr_z = cute.zipped_divide(smem_qr, (1, 1, VEC_SIZE_QR))  # tile=(1,1,2), rest=(T,2,32)
        ckv_z     = cute.zipped_divide(ckv,     (1, VEC_SIZE_QN))     # tile=(1,8),   rest=(128,64)
        kpe_z     = cute.zipped_divide(kpe,     (1, VEC_SIZE_QR))     # tile=(1,2),   rest=(128,32)

        for T_idx in range(T):
            for round_idx in cutlass.range_constexpr(NUM_ROUNDS):  # 8
                col_idx = round_idx * NUM_WARPS + warp_idx

                acc0 = cutlass.Float32(0.0)
                acc1 = cutlass.Float32(0.0)

                # CKV part: HEAD_DIM_CKV=512, 2 iters × 32 lanes × 8 vec = 512 ✓
                for it in cutlass.range_constexpr(ITERS_CKV):
                    chunk = it * 32 + lane_idx
                    qn0_frag = smem_qn_z[(0, 0, None), (T_idx, 0, chunk)].load()
                    qn1_frag = smem_qn_z[(0, 0, None), (T_idx, 1, chunk)].load()
                    ckv_frag = ckv_z    [(0, None),    (col_idx, chunk)].load()
                    for v in cutlass.range_constexpr(VEC_SIZE_QN):
                        a0_v = cutlass.Float32(qn0_frag[v])
                        a1_v = cutlass.Float32(qn1_frag[v])
                        b_v  = cutlass.Float32(ckv_frag[v])
                        acc0, acc1 = cute.arch.fma_packed_f32x2(
                            (a0_v, a1_v), (b_v, b_v), (acc0, acc1)
                        )

                # KPE part: HEAD_DIM_KPE=64, 1 iter × 32 lanes × 2 vec = 64 ✓
                qr0_frag = smem_qr_z[(0, 0, None), (T_idx, 0, lane_idx)].load()
                qr1_frag = smem_qr_z[(0, 0, None), (T_idx, 1, lane_idx)].load()
                kpe_frag = kpe_z    [(0, None),    (col_idx, lane_idx)].load()
                for v in cutlass.range_constexpr(VEC_SIZE_QR):
                    a0_v = cutlass.Float32(qr0_frag[v])
                    a1_v = cutlass.Float32(qr1_frag[v])
                    b_v  = cutlass.Float32(kpe_frag[v])
                    acc0, acc1 = cute.arch.fma_packed_f32x2(
                        (a0_v, a1_v), (b_v, b_v), (acc0, acc1)
                    )

                acc0, acc1 = warp_reduce_f32x2_add(acc0, acc1, width=32)

                if lane_idx == 0:
                    output[T_idx, 0, col_idx] = acc0
                    output[T_idx, 1, col_idx] = acc1

    @cute.jit
    def __call__(self, q_nope, q_pe, ckv, kpe, output, stream):
        self._kernel(q_nope, q_pe, ckv, kpe, output).launch(
            grid=[1, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream
        )


# ── compile ───────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align=16):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align,
    )


def compile_test():
    q_nope = _fake(cute.BFloat16, (T, 2, HEAD_DIM_CKV), (2, 1, 0))
    q_pe   = _fake(cute.BFloat16, (T, 2, HEAD_DIM_KPE), (2, 1, 0))
    ckv    = _fake(cute.BFloat16, (NUM_KV, HEAD_DIM_CKV), (1, 0))
    kpe    = _fake(cute.BFloat16, (NUM_KV, HEAD_DIM_KPE), (1, 0))
    output = _fake(cute.Float32,  (T, 2, NUM_KV), (2, 1, 0))
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    test = TestScore()
    compiled = cute.compile(
        test, q_nope, q_pe, ckv, kpe, output, stream,
        options="--enable-tvm-ffi",
    )
    return test, compiled


_test, _compiled = compile_test()


# ── run + correctness check ───────────────────────────────────────────────────

def run():
    T_v  = int(T)
    NKV  = int(NUM_KV)
    DCKV = int(HEAD_DIM_CKV)
    DKPE = int(HEAD_DIM_KPE)

    q_nope = torch.randn(T_v, 2, DCKV, dtype=torch.bfloat16, device="cuda")
    q_pe   = torch.randn(T_v, 2, DKPE, dtype=torch.bfloat16, device="cuda")
    ckv    = torch.randn(NKV, DCKV,    dtype=torch.bfloat16, device="cuda")
    kpe    = torch.randn(NKV, DKPE,    dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(T_v, 2, NKV,  dtype=torch.float32,  device="cuda")

    _compiled(q_nope, q_pe, ckv, kpe, output)
    torch.cuda.synchronize()

    # reference: (T*2, 512) @ (512, 128) + (T*2, 64) @ (64, 128)
    qn_f = q_nope.float().view(T_v * 2, DCKV)   # (16, 512)
    qr_f = q_pe.float().view(T_v * 2, DKPE)     # (16, 64)
    ref  = (qn_f @ ckv.float().T + qr_f @ kpe.float().T).view(T_v, 2, NKV)

    out     = output.cpu().float()
    ref_cpu = ref.cpu()

    max_err = (out - ref_cpu).abs().max().item()
    rel_err = (out - ref_cpu).abs().max() / ref_cpu.abs().max()

    print(f"\n=== test_score T={T_v}, NUM_KV={NKV}, CKV={DCKV}, KPE={DKPE} ===")
    print(f"max_abs_err = {max_err:.4f}   rel_err = {rel_err:.4f}")

    ok = max_err < 2.0
    print(f"PASS={ok}")
    return ok
