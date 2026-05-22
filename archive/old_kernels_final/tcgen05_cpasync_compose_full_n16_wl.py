"""tcgen05_cpasync_compose_full_n16_wl.py — full MLA score with flat KV cache + sparse indices.

Extends tcgen05_cpasync_compose_full_n16.py to use a flat page-table KV cache layout
and gather rows via sparse_indices stored in smem:

    score = Q_nope × CKV_gathered^T   (K = HEAD_DIM_CKV = 512)
          + Q_pe   × KPE_gathered^T   (K = HEAD_DIM_KPE  = 64)

Inputs:
    ckv_flat:       (FLAT_CACHE, HEAD_DIM_CKV) bf16  — flat (page×page_size, dim) KV cache
    kpe_flat:       (FLAT_CACHE, HEAD_DIM_KPE) bf16  — flat PE cache
    sparse_indices: (DIM_SPLIT,) int32               — which flat-cache rows to gather

At kernel start, sparse_indices is loaded into smem_sp_indices (DIM_SPLIT threads, one
each).  The existing sync_threads() after alloc_tmem fences those writes.  The cp.async
gather loop then reads smem_sp_indices[row_idx] to pick the flat-cache row for each sA
panel row.
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
DIM_CHUNK = 8

HEADS_PER_SPLIT = 2
LIMIT_REQUEST = 8

NUM_WARPS = 16
NUM_THREADS = 512

# Flat KV cache dimensions (page-table layout)
NUM_PAGES  = 8462
PAGE_SIZE  = 64
FLAT_CACHE = NUM_PAGES * PAGE_SIZE  # = 541568

UMMA_INST = (DIM_SPLIT, HEADS_PER_SPLIT * DIM_CHUNK, 16)  # (128, 16, 16)
MMA_M, MMA_N, MMA_K = UMMA_INST
MMA_M_PACK, MMA_N_PACK, MMA_K_PACK = 1, 1, 4
MMA_K_PACKED     = MMA_K * MMA_K_PACK              # = 64
MMA_K_TILES      = HEAD_DIM_CKV  // MMA_K_PACKED   # = 8  (CKV panels)
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

class TestScoreFullWL:

    @cute.jit
    def __call__(self, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, stream):
        T, _, _ = q_nope.shape
        self.num_warps = NUM_THREADS // 32
        op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, UMMA_INST,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        self.tmem_ld_rep = HEADS_PER_SPLIT * DIM_CHUNK

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self._kernel(tiled_mma, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output).launch(
            grid=[T, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def _kernel(
        self,
        tiled_mma,
        q_nope:         cute.Tensor,  # (T, 2, HEAD_DIM_CKV) bf16
        q_pe:           cute.Tensor,  # (T, 2, HEAD_DIM_KPE) bf16
        ckv_flat:       cute.Tensor,  # (FLAT_CACHE, HEAD_DIM_CKV) bf16
        kpe_flat:       cute.Tensor,  # (FLAT_CACHE, HEAD_DIM_KPE) bf16
        sparse_indices: cute.Tensor,  # (DIM_SPLIT,) int32
        output:         cute.Tensor,  # (T, 2, DIM_SPLIT) f32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        # ── smem allocation ────────────────────────────────────────────────────
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
        # Sparse indices buffer: DIM_SPLIT int32 values
        smem_sp_indices = alloc.allocate_tensor(
            cutlass.Int32, cute.make_layout((DIM_SPLIT,), stride=(1,)), 4
        )

        # Copy views: alias panels 0-7 (base pointer, 8-panel layout)
        sA_ckv_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(MMA_M, MMA_K_PACKED, MMA_K_TILES))
        sB_ckv_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(MMA_N, MMA_K_PACKED, MMA_K_TILES))
        # Copy views: alias panel 8 (pointer offset by 8 panels)
        panel_stride_A = MMA_M * MMA_K_PACKED * MMA_K_TILES   # = 65536 elements
        panel_stride_B = MMA_N * MMA_K_PACKED * MMA_K_TILES   # = 8192  elements
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

        # cp.async tiled copy for KPE/Q_pe — 32-bit per thread, 32×2 = 64 elems/step
        atom_cpa_pe = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        val_layout_pe = cute.make_layout(((2, 1),), stride=((1, 0),))
        tiled_copy_pe = cute.make_tiled_copy_tv(atom_cpa_pe, thr_layout, val_layout_pe)
        lane_copy_pe  = tiled_copy_pe.get_slice(lane_idx)

        # ── tcgen05 / tmem setup ───────────────────────────────────────────────
        # make_fragment_A(sA) sees the full 9-panel smem → 9 k-blocks
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape       = tiled_mma.partition_shape_C((MMA_M, MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        # Load sparse indices into smem: threads 0..DIM_SPLIT-1 each take one entry.
        # The sync_threads() after alloc_tmem below also fences these scalar writes.
        if tidx < DIM_SPLIT:
            smem_sp_indices[tidx] = sparse_indices[tidx]

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()  # also fences smem_sp_indices writes above

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
        num_rounds = DIM_SPLIT // self.num_warps  # = 8 rows per warp

        # q_nope → sB panels 0-7,  q_pe → sB panel 8
        if warp_idx < 2:
            cute.copy(atom_cpa,
                      lane_copy.partition_S(cute.composition(q_nope[bidx, warp_idx, None], k_split_shape)),
                      lane_copy.partition_D(sB_ckv_copy[warp_idx, None]))
            cute.copy(atom_cpa_pe,
                      lane_copy_pe.partition_S(cute.composition(q_pe[bidx, warp_idx, None], k_split_shape_pe)),
                      lane_copy_pe.partition_D(sB_kpe_copy[warp_idx, None]))

        # ckv_flat / kpe_flat → sA panels 0-7 / 8 (sparse gather via smem_sp_indices)
        for round_idx in range(num_rounds):
            row_idx  = round_idx * self.num_warps + warp_idx  # smem dest row (0..DIM_SPLIT-1)
            flat_row = smem_sp_indices[row_idx]               # global flat-cache row lookup
            cute.copy(atom_cpa,
                      lane_copy.partition_S(cute.composition(ckv_flat[flat_row, None], k_split_shape)),
                      lane_copy.partition_D(sA_ckv_copy[row_idx, None]))
            cute.copy(atom_cpa_pe,
                      lane_copy_pe.partition_S(cute.composition(kpe_flat[flat_row, None], k_split_shape_pe)),
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
    T              = cute.sym_int()
    q_nope         = _fake(cute.BFloat16, (T, 2, HEAD_DIM_CKV), (2, 1, 0))
    q_pe           = _fake(cute.BFloat16, (T, 2, HEAD_DIM_KPE), (2, 1, 0))
    ckv_flat       = _fake(cute.BFloat16, (FLAT_CACHE, HEAD_DIM_CKV), (1, 0))
    kpe_flat       = _fake(cute.BFloat16, (FLAT_CACHE, HEAD_DIM_KPE), (1, 0))
    sparse_indices = _fake(cute.Int32,    (DIM_SPLIT,), (0,), align=4)
    output         = _fake(cute.Float32,  (T, 2, DIM_SPLIT), (2, 1, 0))
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    test = TestScoreFullWL()
    compiled = cute.compile(
        test, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, stream,
        options="--enable-tvm-ffi",
    )
    return test, compiled


_test, _compiled = compile_test()


# ── run + correctness check ───────────────────────────────────────────────────

def run():
    T_v = 8
    q_nope         = torch.randn(T_v, 2, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
    q_pe           = torch.randn(T_v, 2, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda")
    ckv_flat       = torch.randn(FLAT_CACHE, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
    kpe_flat       = torch.randn(FLAT_CACHE, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda")
    sparse_indices = torch.randint(0, FLAT_CACHE, (DIM_SPLIT,), dtype=torch.int32, device="cuda")
    output         = torch.zeros(T_v, 2, DIM_SPLIT, dtype=torch.float32, device="cuda")

    _compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output)
    torch.cuda.synchronize()

    # Reference: gather rows via sparse_indices, then matmul
    idx     = sparse_indices.long()
    ref_ckv = ckv_flat[idx, :]   # (DIM_SPLIT, HEAD_DIM_CKV)
    ref_kpe = kpe_flat[idx, :]   # (DIM_SPLIT, HEAD_DIM_KPE)
    ref_out = q_nope.float() @ ref_ckv.float().T + q_pe.float() @ ref_kpe.float().T
    # ref_out: (T, 2, DIM_SPLIT)

    max_err = (output.cpu() - ref_out.cpu()).abs().max().item()
    rel_err = (output.cpu() - ref_out.cpu()).abs().max() / ref_out.cpu().abs().max()

    print(f"max_abs_err = {max_err:.4f}   rel_err = {rel_err:.4f}")
    ok = max_err < 2.0
    print(f"PASS={ok}")
    return ok
