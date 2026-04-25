"""TVM-FFI wrapper variant with kv_split_umma-style module interface.

This keeps the same compile/run naming convention used by kv_split_umma.py
so submit.py can load this module directly via IMPL_MODULE.
"""

import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from src.kernels.kv_split_umma import (
    Dsa,
    NUM_HEADS,
    HEAD_DIM_CKV,
    HEAD_DIM_KPE,
    TOP_K,
    NUM_PAGES,
    PAGE_SIZE,
    NUM_SPLITS,
    LIMIT_REQUEST,
    SM_SCALE,
)


def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype,
        shape=shape,
        stride_order=stride_order,
        assumed_align=align,
    )


def compile_hybrid():
    T = cute.sym_int()
    q_nope = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    q_pe = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
    ckv_cache = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV), (2, 1, 0), 16)
    kpe_cache = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32, (T, TOP_K), (1, 0), 4)
    sm_scale = SM_SCALE
    partial_out = _fake(cute.Float32, (LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV), (3, 2, 1, 0), 16)
    partial_lse = _fake(cute.Float32, (LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2), (3, 2, 1, 0), 16)
    output = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    lse = _fake(cute.Float32, (T, NUM_HEADS), (1, 0), 4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    hybrid = Dsa()

    compiled = cute.compile(
        hybrid,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
        partial_out,
        partial_lse,
        output,
        lse,
        stream,
        options="--enable-tvm-ffi",
    )
    return hybrid, compiled


_hybrid, _compiled = compile_hybrid()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    _compiled(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        _hybrid.partial_out,
        _hybrid.partial_lse,
        output,
        lse,
    )
