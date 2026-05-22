"""
Explore cute.zipped_divide layouts for smem/flat tensors.

Run with:
    python src/kernels/explore_zipped_divide.py
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import torch

T_MAX        = 8
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
NUM_PAGES    = 8462
PAGE_SIZE    = 64
N            = NUM_PAGES * PAGE_SIZE   # 541568
VEC_CKV: cutlass.Constexpr = 8
VEC_KPE: cutlass.Constexpr = 2


@cute.jit
def explore(
    smem_q_nope: cute.Tensor,   # (T_max, 512)  bf16
    smem_q_pe:   cute.Tensor,   # (T_max,  64)  bf16
    ckv_flat:    cute.Tensor,   # (541568, 512) bf16
    kpe_flat:    cute.Tensor,   # (541568,  64) bf16
    stream,
):
    print("=== Original layouts ===")
    print("smem_q_nope :", smem_q_nope)
    print("smem_q_pe   :", smem_q_pe)
    print("ckv_flat    :", ckv_flat)
    print("kpe_flat    :", kpe_flat)

    print("\n=== zipped_divide with (1, vec_size) ===")
    smem_q_nope_a = cute.zipped_divide(smem_q_nope, (1, VEC_CKV))
    smem_q_pe_a   = cute.zipped_divide(smem_q_pe,   (1, VEC_KPE))
    ckv_flat_a    = cute.zipped_divide(ckv_flat,    (1, VEC_CKV))
    kpe_flat_a    = cute.zipped_divide(kpe_flat,    (1, VEC_KPE))
    print("smem_q_nope_ (1,8) :", smem_q_nope_a)
    print("smem_q_pe_   (1,2) :", smem_q_pe_a)
    print("ckv_flat_    (1,8) :", ckv_flat_a)
    print("kpe_flat_    (1,2) :", kpe_flat_a)

    print("\n=== zipped_divide with (vec_size,) — no leading 1 ===")
    smem_q_nope_b = cute.zipped_divide(smem_q_nope, (VEC_CKV,))
    smem_q_pe_b   = cute.zipped_divide(smem_q_pe,   (VEC_KPE,))
    ckv_flat_b    = cute.zipped_divide(ckv_flat,    (VEC_CKV,))
    kpe_flat_b    = cute.zipped_divide(kpe_flat,    (VEC_KPE,))
    print("smem_q_nope_ (8,)  :", smem_q_nope_b)
    print("smem_q_pe_   (2,)  :", smem_q_pe_b)
    print("ckv_flat_    (8,)  :", ckv_flat_b)
    print("kpe_flat_    (2,)  :", kpe_flat_b)


@cute.jit
def test_index_styles(smem_q_pe: cute.Tensor, kpe_flat: cute.Tensor, stream):
    """Check whether loaded vec elements need [None, i] or [i]."""
    tidx, _, _ = cute.arch.thread_idx()
    lane_idx = cute.arch.lane_idx()

    q_pe_  = cute.zipped_divide(smem_q_pe, (1, VEC_KPE))  # ((1,2),(T_max,32))
    kpe_   = cute.zipped_divide(kpe_flat,  (1, VEC_KPE))  # ((1,2),(N,32))

    T_idx  = 0
    kv_idx = 0

    qp_vec  = q_pe_ [(None, None), (T_idx,  lane_idx)].load()
    kpe_vec = kpe_  [(None, None), (kv_idx, lane_idx)].load()

    print("loaded qp_vec  :", qp_vec)
    print("qp_vec [None, 0] :", qp_vec[None, 0])

    # Try indexing with (0, None) to squeeze the leading 1 dim
    kv_idx = 0
    ckv_row_a = cute.zipped_divide(kpe_flat, (1, VEC_KPE))[(None, None), (kv_idx, None)]
    ckv_row_b = cute.zipped_divide(kpe_flat, (1, VEC_KPE))[(0,    None), (kv_idx, None)]
    print("\n=== (None, None) vs (0, None) for row slice ===")
    print("kpe_flat_ [(None,None),(kv_idx,None)] :", ckv_row_a)
    print("kpe_flat_ [(0,   None),(kv_idx,None)] :", ckv_row_b)
    vec_a = ckv_row_a[None, None, lane_idx].load()
    vec_b = ckv_row_b[None,       lane_idx].load()
    print("vec from (None,None) row, [None,None,lane_idx].load() :", vec_a)
    print("vec from (0,   None) row, [None,      lane_idx].load() :", vec_b)
    print("vec_a[None,0] :", vec_a[None, 0])
    print("vec_b[None,0] :", vec_b[None, 0])


smem_q_pe_fake  = make_fake_compact_tensor(dtype=cute.BFloat16, shape=(T_MAX, HEAD_DIM_KPE), stride_order=(1, 0), assumed_align=16)
kpe_flat_fake   = make_fake_compact_tensor(dtype=cute.BFloat16, shape=(N, HEAD_DIM_KPE),     stride_order=(1, 0), assumed_align=16)

stream2 = make_fake_stream(use_tvm_ffi_env_stream=True)
compiled2 = cute.compile(test_index_styles, smem_q_pe_fake, kpe_flat_fake, stream2, options="--enable-tvm-ffi")

smem_q_pe_gpu2 = torch.ones(T_MAX, HEAD_DIM_KPE, dtype=torch.bfloat16, device='cuda')
kpe_flat_gpu2  = torch.ones(N, HEAD_DIM_KPE,     dtype=torch.bfloat16, device='cuda') * 2

compiled2(smem_q_pe_gpu2, kpe_flat_gpu2)
torch.cuda.synchronize()

smem_q_pe   = make_fake_compact_tensor(dtype=cute.BFloat16, shape=(T_MAX, HEAD_DIM_KPE), stride_order=(1, 0), assumed_align=16)
smem_q_nope = make_fake_compact_tensor(dtype=cute.BFloat16, shape=(T_MAX, HEAD_DIM_CKV), stride_order=(1, 0), assumed_align=16)
ckv_flat    = make_fake_compact_tensor(dtype=cute.BFloat16, shape=(N, HEAD_DIM_CKV),     stride_order=(1, 0), assumed_align=16)
kpe_flat    = make_fake_compact_tensor(dtype=cute.BFloat16, shape=(N, HEAD_DIM_KPE),     stride_order=(1, 0), assumed_align=16)

stream = make_fake_stream(use_tvm_ffi_env_stream=True)
compiled = cute.compile(explore, smem_q_nope, smem_q_pe, ckv_flat, kpe_flat, stream,
                        options="--enable-tvm-ffi")

smem_q_nope_gpu = torch.zeros(T_MAX, HEAD_DIM_CKV, dtype=torch.bfloat16, device='cuda')
smem_q_pe_gpu   = torch.zeros(T_MAX, HEAD_DIM_KPE, dtype=torch.bfloat16, device='cuda')
ckv_flat_gpu    = torch.zeros(N, HEAD_DIM_CKV,     dtype=torch.bfloat16, device='cuda')
kpe_flat_gpu    = torch.zeros(N, HEAD_DIM_KPE,     dtype=torch.bfloat16, device='cuda')

compiled(smem_q_nope_gpu, smem_q_pe_gpu, ckv_flat_gpu, kpe_flat_gpu)
torch.cuda.synchronize()
