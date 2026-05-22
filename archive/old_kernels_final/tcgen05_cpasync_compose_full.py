"""tcgen05_cpasync_compose_full.py — full MLA score kernel (nope + pe).

Extends tcgen05_cpasync_compose.py to fuse two GEMMs into one accumulator:
    score = Q_nope × CKV^T   (K = HEAD_DIM_CKV = 512)
          + Q_pe  × KPE^T    (K = HEAD_DIM_KPE  = 64)

Single smem allocation with MMA_K_TILES_FULL = 9 panels:
    sA: (MMA_M, MMA_K_PACKED, 9)   panels 0-7 = CKV,  panel 8 = KPE
    sB: (MMA_N, MMA_K_PACKED, 9)   panels 0-7 = Q_nope, panel 8 = Q_pe

Two copy-view aliases per smem buffer:
    sA_ckv_copy  → base iterator,   8-panel layout  (panels 0-7)
    sA_kpe_copy  → base + 8*panel,  1-panel layout  (panel  8)

make_fragment_A(sA) naturally produces 9 k-blocks.
A single MMA loop over all 9 k-blocks accumulates both GEMMs into the same
tCtAcc.  One tcgen05.commit + mbarrier_wait covers the combined result.
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
MMA_K_PACKED     = MMA_K * MMA_K_PACK              # = 64
CTA_TILER        = (128, 8, 512)
MMA_K_TILES      = CTA_TILER[2] // MMA_K_PACKED    # = 8  (CKV panels)
MMA_K_TILES_PE   = HEAD_DIM_KPE  // MMA_K_PACKED   # = 1  (KPE panel)
MMA_K_TILES_FULL = MMA_K_TILES + MMA_K_TILES_PE    # = 9  (total panels)


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _panel_copy_layout(num_rows: int, k_packed: int, k_tiles: int):
    return cute.make_layout((num_rows, (k_packed, k_tiles)), 
                            stride=(k_packed, (1, num_rows * k_packed)),)


# ── kernel ────────────────────────────────────────────────────────────────────

class TestScoreFull:

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

        # ── smem — single 9-panel allocation ──────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()
        storage = alloc.allocate(self.shared_storage)
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        swizzle = cute.make_swizzle(3, 4, 3)

        # 9-panel outer layout: panels 0-7 = CKV/Q_nope, panel 8 = KPE/Q_pe
        a_outer = cute.make_layout(
            ((MMA_M, MMA_K), MMA_M_PACK, (MMA_K_PACK, MMA_K_TILES_FULL)),
            stride=((MMA_K_PACKED, 1), 0, (MMA_K, MMA_M * MMA_K_PACKED)),
        )
        b_outer = cute.make_layout(
            ((MMA_N, MMA_K), MMA_N_PACK, (MMA_K_PACK, MMA_K_TILES_FULL)),
            stride=((MMA_K_PACKED, 1), 0, (MMA_K, MMA_N * MMA_K_PACKED)),
        )
        sA = alloc.allocate_tensor(
            element_type=cutlass.BFloat16, layout=a_outer,
            byte_alignment=16, swizzle=swizzle,
        )
        sB = alloc.allocate_tensor(
            element_type=cutlass.BFloat16, layout=b_outer,
            byte_alignment=16, swizzle=swizzle,
        )

        # Copy views: alias panels 0-7 (base pointer, 8-panel layout)
        sA_ckv_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(MMA_M, MMA_K_PACKED, MMA_K_TILES))
        sB_ckv_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(MMA_N, MMA_K_PACKED, MMA_K_TILES))
        # Copy views: alias panel 8 (pointer offset by 8 panels)
        panel_stride_A = MMA_M * MMA_K_PACKED * MMA_K_TILES   # = 128 * 64 * 8 = 65536 elements
        panel_stride_B = MMA_N * MMA_K_PACKED * MMA_K_TILES   # = 8   * 64 * 8 = 4096  elements
        sA_kpe_copy = cute.make_tensor(sA.iterator + panel_stride_A, _panel_copy_layout(MMA_M, MMA_K_PACKED, MMA_K_TILES_PE))
        sB_kpe_copy = cute.make_tensor(sB.iterator + panel_stride_B, _panel_copy_layout(MMA_N, MMA_K_PACKED, MMA_K_TILES_PE))

        # composition shapes for rank-1 row slices
        k_split_shape    = cute.make_layout(((MMA_K_PACKED, MMA_K_TILES),))
        k_split_shape_pe = cute.make_layout(((MMA_K_PACKED, MMA_K_TILES_PE),))

        # cp.async tiled copy for CKV/Q_nope — 128-bit per thread, 32×8 = 256 elems/step
        atom_cpa = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy = cute.make_tiled_copy_tv(atom_cpa, thr_layout, val_layout)
        lane_copy  = tiled_copy.get_slice(lane_idx)

        # cp.async tiled copy for KPE/Q_pe — 32-bit per thread, 32×2 = 64 elems/step (= K_PE)
        atom_cpa_pe = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        val_layout_pe = cute.make_layout(((2, 1),), stride=((1, 0),))
        tiled_copy_pe = cute.make_tiled_copy_tv(atom_cpa_pe, thr_layout, val_layout_pe)
        lane_copy_pe  = tiled_copy_pe.get_slice(lane_idx)

        # ── tcgen05 / tmem setup ───────────────────────────────────────────────
        # make_fragment_A(sA) sees the full 9-panel smem → 9 k-blocks
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
        num_rounds = DIM_SPLIT // self.num_warps   # = 8 rows per warp

        # q_nope → sB panels 0-7,  q_pe → sB panel 8
        if warp_idx < 2:
            cute.copy(atom_cpa,
                      lane_copy.partition_S(cute.composition(q_nope[bidx, warp_idx, None], k_split_shape)),
                      lane_copy.partition_D(sB_ckv_copy[warp_idx, None]))
            cute.copy(atom_cpa_pe,
                      lane_copy_pe.partition_S(cute.composition(q_pe[bidx, warp_idx, None], k_split_shape_pe)),
                      lane_copy_pe.partition_D(sB_kpe_copy[warp_idx, None]))

        # ckv → sA panels 0-7,  kpe → sA panel 8
        for round_idx in range(num_rounds):
            row_idx = round_idx * self.num_warps + warp_idx
            cute.copy(atom_cpa,
                      lane_copy.partition_S(cute.composition(ckv[row_idx, None], k_split_shape)),
                      lane_copy.partition_D(sA_ckv_copy[row_idx, None]))
            cute.copy(atom_cpa_pe,
                      lane_copy_pe.partition_S(cute.composition(kpe[row_idx, None], k_split_shape_pe)),
                      lane_copy_pe.partition_D(sA_kpe_copy[row_idx, None]))

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # ── MMA: single loop over all 9 k-blocks ──────────────────────────────
        tcgen05_fence()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        mma_phase = 0
        if warp_idx == 0:
            num_k_blocks = cute.size(tCrA, mode=[2])   # = 9
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
    test = TestScoreFull()
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

    ref_out = q_nope.float() @ ckv.float().T + q_pe.float() @ kpe.float().T
    max_err = (output.cpu().float() - ref_out.cpu().float()).abs().max().item()
    rel_err = (output.cpu().float() - ref_out.cpu().float()).abs().max() / ref_out.cpu().float().abs().max()

    print(f"max_abs_err = {max_err:.4f}   rel_err = {rel_err:.4f}")
    ok = max_err < 2.0
    print(f"PASS={ok}")
    return ok
