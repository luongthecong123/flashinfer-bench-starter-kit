import json
import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

# ── constants ─────────────────────────────────────────────────────────────────
T = 8    # number of tokens
NUM_HEADS = 2    # heads per token (HEADS_PER_SPLIT)
NUM_ROWS = 16   # T * NUM_HEADS
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
DIM_SPLIT = 128  # KV rows to score

VEC_SIZE_QN = 8   # bf16×8 = 128-bit ldg
VEC_SIZE_QR = 2   # bf16×2 = 32-bit ldg

NUM_WARPS = 16   # T * NUM_HEADS — one warp per row
NUM_THREADS = 512  # NUM_WARPS * 32

# chunks per lane when 1 warp loads 1 row
ITERS_QN_LOAD = 2  # HEAD_DIM_CKV // (VEC_SIZE_QN * 32) = 512 // 256 = 2
# ITERS_QR_LOAD = HEAD_DIM_KPE // (VEC_SIZE_QR * 32) = 64 // 64 = 1  (no loop needed)

# score dot-product iters per lane
ITERS_CKV = 2  # same as ITERS_QN_LOAD
# ITERS_KPE = 1  (no loop needed)

NUM_ROUNDS = 8  # DIM_SPLIT // NUM_WARPS
UMMA_TILER = (128, 8, 16)
MMA_M, MMA_N, MMA_K = UMMA_TILER
MMA_M_PACK, MMA_N_PACK, MMA_K_PACK = 1, 1, 4
MMA_K_PACKED = MMA_K * MMA_K_PACK
CTA_TILER = (128, 8, 512)
MMA_K_TILES = CTA_TILER[2] // MMA_K_PACKED


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
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
        alloc   = cutlass.utils.SmemAllocator()

        """
        sA ckv: 128x512 gmem -> 8x(128x64) smem
        Panel 0 (k=0..63):     offsets     0 ..  8191   ← 128 rows x 64 cols, row stride = 64
        Panel 1 (k=64..127):   offsets  8192 .. 16383
        Panel 2 (k=128..191):  offsets 16384 .. 24575
        ...
        Panel 7 (k=448..511):  offsets 57344 .. 65535        
        -> 1 row in gmem is the top left 8x64 tile in smem

        sA ckv + kpe: 128x576 gmem -> 9x(128x64) smem

        sB: 8x512 gmem -> 8x(8x64) smem
        Panel 0 (k=0..63):     offsets    0 ..  511   ← 8 rows x 64 cols, row stride = 64
        Panel 1 (k=64..127):   offsets  512 .. 1023
        Panel 2 (k=128..191):  offsets 1024 .. 1535
        ...
        Panel 7 (k=448..511):  offsets 3584 .. 4095
        -> 1 row in gmem is the top left 8x64 tile in smem 
        
        sA ckv + kpe: 8x576 gmem -> 9x(8x64) smem
        """
        
        a_outer = cute.make_layout(
            ((MMA_M, MMA_K), MMA_M_PACK, (MMA_K_PACK, MMA_K_TILES)), # ((128,16), 1, (4,8))
            stride=((MMA_K_PACKED, 1), 0, (MMA_K, MMA_M * MMA_K_PACKED)) # ((64,1), 0, (16,8192))
        )
        b_outer = cute.make_layout(
            ((MMA_N, MMA_K), MMA_N_PACK, (MMA_K_PACK, MMA_K_TILES)), # ((8,16), 1, (4,8))
            stride=((MMA_K_PACKED, 1), 0, (MMA_K, MMA_N * MMA_K_PACKED)) # ((64,1), 0, (16,512))
        )
        swizzle = cute.make_swizzle(3, 4, 3)
        
        sA = alloc.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=a_outer,
            byte_alignment=1024,
            swizzle=swizzle
        )
        
        sB = alloc.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_outer,
            byte_alignment=1024,
            swizzle=swizzle
        )

        # Row slices [row, None] → rank-1 ((K_PACK,K_TILES),); rank-1 thr/val tiler satisfies
        # partition_S requirement rank(input) >= rank(tiler).
        sA_copy = cute.make_tensor(
            sA.iterator,
            cute.make_layout(
                (MMA_M, (MMA_K_PACKED, MMA_K_TILES)),
                stride=(MMA_K_PACKED, (1, MMA_M * MMA_K_PACKED)),
            ),
        )
        sB_copy = cute.make_tensor(
            sB.iterator,
            cute.make_layout(
                (MMA_N, (MMA_K_PACKED, MMA_K_TILES)),
                stride=(MMA_K_PACKED, (1, MMA_N * MMA_K_PACKED)),
            ),
        )
        # (T, H, (K_PACK, K_TILES)) gmem view of q_nope
        q_nope_full = cute.make_tensor(
            q_nope.iterator,
            cute.make_layout(
                (q_nope.shape[0], NUM_HEADS, (MMA_K_PACKED, MMA_K_TILES)),
                stride=(NUM_HEADS * HEAD_DIM_CKV, HEAD_DIM_CKV, (1, MMA_K_PACKED)),
            ),
        )
        # (DIM_SPLIT, (K_PACK, K_TILES)) gmem view of ckv
        ckv_full = cute.make_tensor(
            ckv.iterator,
            cute.make_layout(
                (DIM_SPLIT, (MMA_K_PACKED, MMA_K_TILES)),
                stride=(HEAD_DIM_CKV, (1, MMA_K_PACKED)),
            ),
        )

        atom_cpa = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)

        # 32 lanes cover K only; rank-1 tiler → partition_S needs rank(input) >= 1.
        thr_layout = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout = cute.make_layout(((8, 1),), stride=((1, 0),))
        
        tiled_copy_ckv = cute.make_tiled_copy_tv(
            atom_cpa, thr_layout, val_layout
        )
        
        lane_copy = tiled_copy_ckv.get_slice(lane_idx)
        
        # ================== tcgen05 stuffs ===================

        storage = alloc.allocate(self.shared_storage)
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        
        print("tCrA: ", tCrA)
        print("tCrB: ", tCrB)
        
        acc_shape       = tiled_mma.partition_shape_C(CTA_TILER[:2])
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
        cute.arch.sync_threads()      # full CTA — bar 0 reserved
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()      # full CTA

        M_acc          = cute.size(tCtAcc, mode=[0, 0])
        ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler      = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r  = cute.make_copy_atom(ld_op, cutlass.Float32)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy  = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc       = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc       = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, cutlass.Float32)        
        # ===================================================        

        
        num_rounds = DIM_SPLIT // self.num_warps

        # q_nope → sB: warp 0 loads head 0, warp 1 loads head 1
        if warp_idx < 2:
            tQB_src = lane_copy.partition_S(q_nope_full[bidx, warp_idx, None])
            tQB_dst = lane_copy.partition_D(sB_copy[warp_idx, None])
            cute.copy(atom_cpa, tQB_src, tQB_dst)
            cute.arch.cp_async_commit_group()

        # ckv → sA: rank-2 row slices [None, row, None] → (1,(K_PACK,K_TILES))
        for round_idx in range(num_rounds):
            row_idx = round_idx * self.num_warps + warp_idx
            cute.copy(
                atom_cpa,
                lane_copy.partition_S(ckv_full[row_idx, None]),
                lane_copy.partition_D(sA_copy[row_idx, None])
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()  # ensure all warps' smem writes are visible before MMA

        tcgen05_fence()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        mma_phase = 0
        if warp_idx == 0:
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx)
                cute.gemm(tiled_mma, tCtAcc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if tidx == 0:
                tcgen05.commit(mma_mbar)
        # All warps wait for MMA to complete before reading tmem.
        cute.arch.mbarrier_wait(mma_mbar, mma_phase)

        if tidx < DIM_SPLIT:
            cute.copy(
                tmem_tiled_copy,
                tTR_tAcc[None, None, 0], tTR_rAcc
            )
            for reg_idx in range(NUM_HEADS):
                output[bidx, reg_idx, tidx] = tTR_rAcc[reg_idx]

        # Epilogue — full CTA must converge before tmem dealloc.
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
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
    T_v  = 8

    q_nope = torch.randn(T_v, 2, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
    q_pe   = torch.randn(T_v, 2, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda")
    ckv    = torch.randn(DIM_SPLIT, HEAD_DIM_CKV,    dtype=torch.bfloat16, device="cuda")
    kpe    = torch.randn(DIM_SPLIT, HEAD_DIM_KPE,    dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(T_v, 2, DIM_SPLIT,  dtype=torch.float32,  device="cuda")

    _compiled(q_nope, q_pe, ckv, kpe, output)
    torch.cuda.synchronize()

    # reference: (8,2,512) @ (512,128) → (8,2,128)
    ref_out = q_nope.float() @ ckv.float().T
    ref_cpu     = ref_out.cpu().float()
    out = output.cpu().float()
    
    max_err = (out - ref_cpu).abs().max().item()
    rel_err = (out - ref_cpu).abs().max() / ref_cpu.abs().max()

    print(f"max_abs_err = {max_err:.4f}   rel_err = {rel_err:.4f}")

    ok = max_err < 2.0
    print(f"PASS={ok}")
    return ok
