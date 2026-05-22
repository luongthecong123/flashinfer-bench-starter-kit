"""tcgen05_cpasync_compose.py — explores cute.composition for layout construction.

Differences from tcgen05_cpasync.py
────────────────────────────────────
The key insight about cute.composition in the Python DSL:
    - rank-1 lhs  ✓  works at trace time (pure Python, no MLIR op)
    - rank-2+ lhs ✗  generates an MLIR op that the lowering rejects

gmem tensors (ckv, q_nope):
    Instead of make_tensor(x.iterator, layout), apply cute.composition directly on
    rank-1 row slices at each copy site:

        ckv[row_idx, None]                       # rank-1: (HEAD_DIM_CKV,):(1,)
        cute.composition(that, k_split_shape)     # → ((MMA_K_PACKED, MMA_K_TILES),):(1, MMA_K_PACKED)

    The rank-1 restriction is satisfied because slicing out the row/batch dims
    leaves a flat K tensor.  No make_tensor(iterator, layout) call is needed.

smem tensors (sA_copy, sB_copy) — _panel_copy_layout() helper:
    Panel strides have no algebraic relation to any flat K layout, so
    composition cannot derive them.  Uses explicit strides instead.
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

# ── constants ─────────────────────────────────────────────────────────────────
T = 8
NUM_HEADS = 2
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
DIM_SPLIT = 128

NUM_WARPS = 16
NUM_THREADS = 512

UMMA_TILER = (128, 8, 16)
MMA_M, MMA_N, MMA_K = UMMA_TILER
MMA_M_PACK, MMA_N_PACK, MMA_K_PACK = 1, 1, 4
MMA_K_PACKED = MMA_K * MMA_K_PACK   # = 64  (contiguous K elements per panel row)
CTA_TILER = (128, 8, 512)
MMA_K_TILES = CTA_TILER[2] // MMA_K_PACKED  # = 8  (number of panels)


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _panel_copy_layout(num_rows: int, k_packed: int, k_tiles: int):
    """Copy-friendly (num_rows, (k_packed, k_tiles)) view of a panel-packed smem buffer.

    Panel-packed smem layout:
        Panel p occupies offsets [p*num_rows*k_packed, (p+1)*num_rows*k_packed).
        Row r of panel p starts at offset p*num_rows*k_packed + r*k_packed.

    Resulting strides:
        row   → k_packed                    (rows are k_packed elements apart)
        kp    → 1                           (contiguous within a panel row)
        kt    → num_rows * k_packed         (jump to next panel)
    """
    return cute.make_layout(
        (num_rows, (k_packed, k_tiles)),
        stride=(k_packed, (1, num_rows * k_packed)),
    )


# ── kernel ────────────────────────────────────────────────────────────────────

class TestScore:

    @cute.jit
    def __call__(self, q_nope, q_pe, ckv, kpe, output, stream):
        T, _, _ = q_nope.shape
        self.num_warps = NUM_THREADS // 32
        op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, UMMA_TILER,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        self.tmem_ld_rep = 2

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self._kernel(tiled_mma, q_nope, q_pe, ckv, kpe, output).launch(
            grid=[T, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def _kernel(
        self,
        tiled_mma,
        q_nope: cute.Tensor,   # (T, 2, HEAD_DIM_CKV) bf16
        q_pe:   cute.Tensor,   # (T, 2, HEAD_DIM_KPE) bf16
        ckv:    cute.Tensor,   # (DIM_SPLIT, HEAD_DIM_CKV) bf16
        kpe:    cute.Tensor,   # (DIM_SPLIT, HEAD_DIM_KPE) bf16
        output: cute.Tensor,   # (T, 2, DIM_SPLIT) f32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        # ── smem ──────────────────────────────────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()
        # ── tcgen05 / tmem setup ───────────────────────────────────────────────
        storage = alloc.allocate(self.shared_storage)
        mma_mbar = storage.mma_mbar_ptr.data_ptr()
        
        a_outer = cute.make_layout(
            ((MMA_M, MMA_K), MMA_M_PACK, (MMA_K_PACK, MMA_K_TILES)),
            stride=((MMA_K_PACKED, 1), 0, (MMA_K, MMA_M * MMA_K_PACKED))
        )
        b_outer = cute.make_layout(
            ((MMA_N, MMA_K), MMA_N_PACK, (MMA_K_PACK, MMA_K_TILES)),
            stride=((MMA_K_PACKED, 1), 0, (MMA_K, MMA_N * MMA_K_PACKED))
        )
        swizzle = cute.make_swizzle(3, 4, 3)

        sA = alloc.allocate_tensor(
            element_type=cutlass.BFloat16, layout=a_outer,
            byte_alignment=16, swizzle=swizzle,
        )
        sB = alloc.allocate_tensor(
            element_type=cutlass.BFloat16, layout=b_outer,
            byte_alignment=16, swizzle=swizzle,
        )

        # smem copy views: same as above.
        sA_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(MMA_M, MMA_K_PACKED, MMA_K_TILES))
        sB_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(MMA_N, MMA_K_PACKED, MMA_K_TILES))

        # k_split_shape: the K axis of any rank-1 row slice is reshaped by
        # composition from (HEAD_DIM_CKV,) to ((MMA_K_PACKED, MMA_K_TILES),).
        # The rank-1 lhs ensures composition runs as pure Python at trace time.
        k_split_shape = cute.make_layout(((MMA_K_PACKED, MMA_K_TILES),))

        atom_cpa = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        # Rank-1 thr/val tiler: all 32 lanes cover K only.
        thr_layout = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy_ckv = cute.make_tiled_copy_tv(atom_cpa, thr_layout, val_layout)
        lane_copy = tiled_copy_ckv.get_slice(lane_idx)



        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape       = tiled_mma.partition_shape_C(CTA_TILER[:2])
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)

        M_acc           = cute.size(tCtAcc, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler       = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, cutlass.Float32)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, cutlass.Float32)

        # ── loads ─────────────────────────────────────────────────────────────
        num_rounds = DIM_SPLIT // self.num_warps

        # q_nope → sB: warp 0 loads head 0, warp 1 loads head 1.
        # q_nope[bidx, warp_idx, None] is a rank-1 (HEAD_DIM_CKV,):(1,) slice;
        # composition reshapes its K axis to ((MMA_K_PACKED, MMA_K_TILES),).
        if warp_idx < 2:
            tQB_src = lane_copy.partition_S(
                cute.composition(q_nope[bidx, warp_idx, None], k_split_shape)
            )
            tQB_dst = lane_copy.partition_D(sB_copy[warp_idx, None])
            cute.copy(atom_cpa, tQB_src, tQB_dst)
            cute.arch.cp_async_commit_group()

        # ckv → sA: ckv[row_idx, None] is rank-1; composition reshapes K.
        for round_idx in range(num_rounds):
            row_idx = round_idx * self.num_warps + warp_idx
            cute.copy(
                atom_cpa,
                lane_copy.partition_S(cute.composition(ckv[row_idx, None], k_split_shape)),
                lane_copy.partition_D(sA_copy[row_idx, None]),
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # ── MMA ───────────────────────────────────────────────────────────────
        tcgen05_fence()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        mma_phase = 0
        if warp_idx == 0:
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx)
                cute.gemm(tiled_mma, tCtAcc,
                          tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if tidx == 0:
                tcgen05.commit(mma_mbar)
        cute.arch.mbarrier_wait(mma_mbar, mma_phase)

        # ── epilogue ──────────────────────────────────────────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        if tidx < DIM_SPLIT:
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
            for reg_idx in range(NUM_HEADS):
                output[bidx, reg_idx, tidx] = tTR_rAcc[reg_idx]

        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── compile ───────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align=16):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align,
    )


def compile_test():
    T = cute.sym_int()
    q_nope = _fake(cute.BFloat16, (T, 2, HEAD_DIM_CKV), (2, 1, 0))
    q_pe   = _fake(cute.BFloat16, (T, 2, HEAD_DIM_KPE), (2, 1, 0))
    ckv    = _fake(cute.BFloat16, (DIM_SPLIT, HEAD_DIM_CKV), (1, 0))
    kpe    = _fake(cute.BFloat16, (DIM_SPLIT, HEAD_DIM_KPE), (1, 0))
    output = _fake(cute.Float32,  (T, 2, DIM_SPLIT), (2, 1, 0))
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
    T_v = 8
    q_nope = torch.randn(T_v, 2, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
    q_pe   = torch.randn(T_v, 2, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda")
    ckv    = torch.randn(DIM_SPLIT, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
    kpe    = torch.randn(DIM_SPLIT, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(T_v, 2, DIM_SPLIT, dtype=torch.float32, device="cuda")

    _compiled(q_nope, q_pe, ckv, kpe, output)
    torch.cuda.synchronize()

    ref_out = q_nope.float() @ ckv.float().T
    max_err = (output.cpu().float() - ref_out.cpu().float()).abs().max().item()
    rel_err = (output.cpu().float() - ref_out.cpu().float()).abs().max() / ref_out.cpu().float().abs().max()

    print(f"max_abs_err = {max_err:.4f}   rel_err = {rel_err:.4f}")
    ok = max_err < 2.0
    print(f"PASS={ok}")
    return ok
