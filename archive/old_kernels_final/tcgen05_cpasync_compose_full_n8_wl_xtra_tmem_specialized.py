"""tcgen05_cpasync_compose_full_n8_wl_xtra_tmem_specialized.py

Specialized variant — 2-stage XOR ping-pong, manual mbarriers (NO PipelineAsyncUmma).

Layout:
    * 128 threads / block = 4 warps
        - warps 0..2 (96 threads) — cp.async PRODUCER
        - warp 3      (32 threads) — UMMA CONSUMER
    * 2-stage panel ring. `stage` flips via XOR each panel (left/right halves of sA/sB).
    * Manual mbarriers:
        - ab_full[2]   cnt=96  : producer→consumer per stage (96 cpa thr arrive once)
        - ab_empty[2]  cnt=1   : consumer→producer per stage (tcgen05.commit by mma)
        - mma_full[T]  cnt=1   : per-token signal to drain epilogue (tcgen05.commit)
    * Phase tracking: each stage's full/empty mbar gets its own XOR-toggled phase
      scalar. Missing this XOR = deadlock (the second use of a stage stalls
      forever waiting for a phase that has already passed).
    * Per-token tmem-slot isolation (T=8 × 8 cols = 64 cols).
    * Drain epilogue writes raw `score * sm_scale` to gmem (no softmax).
"""

import math
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
HEADS_PER_SPLIT = NUM_HEADS

NUM_CPASYNC_WARPS = 3
NUM_MMA_WARPS     = 1
NUM_WARPS         = NUM_CPASYNC_WARPS + NUM_MMA_WARPS   # 4
NUM_THREADS       = NUM_WARPS * 32                      # 128
CPASYNC_THREADS   = NUM_CPASYNC_WARPS * 32              # 96
MMA_THREADS       = NUM_MMA_WARPS * 32                  # 32

NUM_PAGES  = 8462
PAGE_SIZE  = 64
FLAT_CACHE = NUM_PAGES * PAGE_SIZE                       # 541568

UMMA_INST = (DIM_SPLIT, 8, 16)
MMA_M, MMA_N, MMA_K = UMMA_INST
TMEM_COLS_PER_TOKEN = MMA_N                              # 8
TMEM_LD_REP         = HEADS_PER_SPLIT                    # 2

_MK_PACK     = 4
_MK_PACKED   = MMA_K * _MK_PACK                          # 64
_MK_TILES_CKV  = HEAD_DIM_CKV // _MK_PACKED              # 8
_MK_TILES_PE   = HEAD_DIM_KPE // _MK_PACKED              # 1
_MK_TILES_FULL = _MK_TILES_CKV + _MK_TILES_PE            # 9

NUM_STAGES = 2


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


# ── kernel ────────────────────────────────────────────────────────────────────

class TestScoreSpecialized:

    @cute.jit
    def __call__(self, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                 sm_scale: cutlass.Constexpr, output, stream):
        op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, UMMA_INST,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        @cute.struct
        class SharedStorage:
            ab_full_mbars:    cute.struct.MemRange[cutlass.Int64, NUM_STAGES]
            ab_empty_mbars:   cute.struct.MemRange[cutlass.Int64, NUM_STAGES]
            mma_full_mbars:   cute.struct.MemRange[cutlass.Int64, T]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self._kernel(
            tiled_mma, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            sm_scale, output,
        ).launch(grid=[1, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def _kernel(
        self,
        tiled_mma,
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        output:         cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        is_cpasync = warp_idx < NUM_CPASYNC_WARPS
        is_mma     = warp_idx == NUM_CPASYNC_WARPS

        # ── smem allocation ───────────────────────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()

        # 2-stage panel ring: stage = 0 / 1 (XOR flip) → left/right half of sA/sB
        swizzle = cute.make_swizzle(3, 4, 3)
        S = NUM_STAGES
        a_outer = cute.make_layout(
            ((MMA_M, MMA_K), 1, (_MK_PACK,), S),
            stride=((_MK_PACKED, 1), 0, (MMA_K,), MMA_M * _MK_PACKED))
        b_outer = cute.make_layout(
            ((MMA_N, MMA_K), 1, (_MK_PACK,), S),
            stride=((_MK_PACKED, 1), 0, (MMA_K,), MMA_N * _MK_PACKED))
        sA = alloc.allocate_tensor(cutlass.BFloat16, a_outer, byte_alignment=16, swizzle=swizzle)
        sB = alloc.allocate_tensor(cutlass.BFloat16, b_outer, byte_alignment=16, swizzle=swizzle)
        sA_iter = sA.iterator
        sB_iter = sB.iterator
        SA_STAGE_STRIDE: cutlass.Constexpr = MMA_M * _MK_PACKED   # 8192
        SB_STAGE_STRIDE: cutlass.Constexpr = MMA_N * _MK_PACKED   # 512

        smem_sp_indices = alloc.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((T, DIM_SPLIT), stride=(DIM_SPLIT, 1)),
            4,
        )

        storage = alloc.allocate(self.shared_storage)
        ab_full_mbars  = storage.ab_full_mbars.data_ptr()
        ab_empty_mbars = storage.ab_empty_mbars.data_ptr()
        mma_full_mbars = storage.mma_full_mbars.data_ptr()

        # Sparse indices preload (all 128 threads → 8 i32 each)
        for k in range(8):
            flat_idx = k * NUM_THREADS + tidx
            row = flat_idx // DIM_SPLIT
            col = flat_idx %  DIM_SPLIT
            if row < T:
                smem_sp_indices[row, col] = sparse_indices[row, col]

        # ── tmem alloc + mbar init ────────────────────────────────────────────
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C((MMA_M, MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        per_token_cols  = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(per_token_cols * T)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                for s in range(NUM_STAGES):
                    cute.arch.mbarrier_init(ab_full_mbars  + s, cnt=CPASYNC_THREADS)
                    cute.arch.mbarrier_init(ab_empty_mbars + s, cnt=1)
                for i in range(T):
                    cute.arch.mbarrier_init(mma_full_mbars + i, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )

        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        M_acc           = cute.size(tCtAcc_base, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(TMEM_LD_REP))
        epi_tiler       = ((M_acc, TMEM_LD_REP),)
        tCtAcc_epi_base = cute.zipped_divide(tCtAcc_base, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, cutlass.Float32)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi_base[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc_base   = tmem_thr_copy.partition_S(tCtAcc_epi_base)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc_base[None, None, 0].shape, cutlass.Float32)

        atom_cpa = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)

        # ════════════════════════════════════════════════════════════════════
        # CP.ASYNC PRODUCER (warps 0..2, 96 threads)
        # ════════════════════════════════════════════════════════════════════
        if is_cpasync:
            cpa_warp_idx = warp_idx
            cpa_tid      = cpa_warp_idx * 32 + lane_idx

            VEC_OPS_PER_PANEL: cutlass.Constexpr = MMA_M * (_MK_PACKED // 8)            # 1024
            ROUNDS_PER_PANEL : cutlass.Constexpr = (VEC_OPS_PER_PANEL + 96 - 1) // 96   # 11

            # Per-stage XOR phase trackers for ab_empty waits.
            empty_phase0 = cutlass.Int32(0)
            empty_phase1 = cutlass.Int32(0)
            stage  = cutlass.Int32(0)
            pcycle = cutlass.Int32(0)

            for T_idx in cutlass.range_constexpr(T):
                for panel_idx in cutlass.range_constexpr(_MK_TILES_FULL):
                    # Wait empty[stage] (skip first NUM_STAGES iters; stages start free).
                    if pcycle >= cutlass.Int32(NUM_STAGES):
                        if stage == cutlass.Int32(0):
                            cute.arch.mbarrier_wait(ab_empty_mbars + 0, empty_phase0)
                            empty_phase0 = empty_phase0 ^ cutlass.Int32(1)
                        else:
                            cute.arch.mbarrier_wait(ab_empty_mbars + 1, empty_phase1)
                            empty_phase1 = empty_phase1 ^ cutlass.Int32(1)

                    sa_stage_off = stage * cutlass.Int32(SA_STAGE_STRIDE)
                    sb_stage_off = stage * cutlass.Int32(SB_STAGE_STRIDE)

                    if panel_idx < _MK_TILES_CKV:
                        ckv_panel_k_base: cutlass.Constexpr = panel_idx * _MK_PACKED
                        if cpa_warp_idx == 0 and lane_idx < HEADS_PER_SPLIT * 8:
                            head_local = lane_idx // 8
                            k_vec      = lane_idx %  8
                            src_off = (T_idx * NUM_HEADS * HEAD_DIM_CKV
                                       + head_local * HEAD_DIM_CKV
                                       + ckv_panel_k_base + k_vec * 8)
                            src_p = cute.make_ptr(cutlass.BFloat16,
                                (q_nope.iterator + src_off).toint(),
                                mem_space=cute.AddressSpace.gmem, assumed_align=16)
                            dst_p = cute.make_ptr(cutlass.BFloat16,
                                (sB_iter + (sb_stage_off + head_local * _MK_PACKED + k_vec * 8)).toint(),
                                mem_space=cute.AddressSpace.smem, assumed_align=16)
                            src_vec = cute.make_tensor(src_p, cute.make_layout((8,), stride=(1,)))
                            dst_vec = cute.make_tensor(dst_p, cute.make_layout((8,), stride=(1,)))
                            cute.copy(atom_cpa, src_vec, dst_vec)
                        for round_idx in cutlass.range_constexpr(ROUNDS_PER_PANEL):
                            pos = round_idx * 96 + cpa_tid
                            if pos < VEC_OPS_PER_PANEL:
                                row   = pos // 8
                                k_vec = pos %  8
                                row_global = smem_sp_indices[T_idx, row]
                                src_p = cute.make_ptr(cutlass.BFloat16,
                                    (ckv_flat.iterator + (row_global * HEAD_DIM_CKV
                                                          + ckv_panel_k_base + k_vec * 8)).toint(),
                                    mem_space=cute.AddressSpace.gmem, assumed_align=16)
                                dst_p = cute.make_ptr(cutlass.BFloat16,
                                    (sA_iter + (sa_stage_off + row * _MK_PACKED + k_vec * 8)).toint(),
                                    mem_space=cute.AddressSpace.smem, assumed_align=16)
                                src_vec = cute.make_tensor(src_p, cute.make_layout((8,), stride=(1,)))
                                dst_vec = cute.make_tensor(dst_p, cute.make_layout((8,), stride=(1,)))
                                cute.copy(atom_cpa, src_vec, dst_vec)
                    else:
                        if cpa_warp_idx == 0 and lane_idx < HEADS_PER_SPLIT * 8:
                            head_local = lane_idx // 8
                            k_vec      = lane_idx %  8
                            src_p = cute.make_ptr(cutlass.BFloat16,
                                (q_pe.iterator + (T_idx * NUM_HEADS * HEAD_DIM_KPE
                                                  + head_local * HEAD_DIM_KPE + k_vec * 8)).toint(),
                                mem_space=cute.AddressSpace.gmem, assumed_align=16)
                            dst_p = cute.make_ptr(cutlass.BFloat16,
                                (sB_iter + (sb_stage_off + head_local * _MK_PACKED + k_vec * 8)).toint(),
                                mem_space=cute.AddressSpace.smem, assumed_align=16)
                            src_vec = cute.make_tensor(src_p, cute.make_layout((8,), stride=(1,)))
                            dst_vec = cute.make_tensor(dst_p, cute.make_layout((8,), stride=(1,)))
                            cute.copy(atom_cpa, src_vec, dst_vec)
                        for round_idx in cutlass.range_constexpr(ROUNDS_PER_PANEL):
                            pos = round_idx * 96 + cpa_tid
                            if pos < VEC_OPS_PER_PANEL:
                                row   = pos // 8
                                k_vec = pos %  8
                                row_global = smem_sp_indices[T_idx, row]
                                src_p = cute.make_ptr(cutlass.BFloat16,
                                    (kpe_flat.iterator + (row_global * HEAD_DIM_KPE + k_vec * 8)).toint(),
                                    mem_space=cute.AddressSpace.gmem, assumed_align=16)
                                dst_p = cute.make_ptr(cutlass.BFloat16,
                                    (sA_iter + (sa_stage_off + row * _MK_PACKED + k_vec * 8)).toint(),
                                    mem_space=cute.AddressSpace.smem, assumed_align=16)
                                src_vec = cute.make_tensor(src_p, cute.make_layout((8,), stride=(1,)))
                                dst_vec = cute.make_tensor(dst_p, cute.make_layout((8,), stride=(1,)))
                                cute.copy(atom_cpa, src_vec, dst_vec)

                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(ab_full_mbars + stage)

                    stage  = stage  ^ cutlass.Int32(1)
                    pcycle = pcycle + cutlass.Int32(1)

        # ════════════════════════════════════════════════════════════════════
        # UMMA CONSUMER (warp 3, 32 threads)
        # ════════════════════════════════════════════════════════════════════
        if is_mma:
            full_phase0 = cutlass.Int32(0)
            full_phase1 = cutlass.Int32(0)
            stage = cutlass.Int32(0)

            for T_idx in cutlass.range_constexpr(T):
                tmem_slot_offset = cutlass.Int32(T_idx * TMEM_COLS_PER_TOKEN)
                tCtAcc_i = cute.make_tensor(tmem_ptr + tmem_slot_offset, tCtAcc_tmpl.layout)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for panel_idx in cutlass.range_constexpr(_MK_TILES_FULL):
                    if stage == cutlass.Int32(0):
                        cute.arch.mbarrier_wait(ab_full_mbars + 0, full_phase0)
                        full_phase0 = full_phase0 ^ cutlass.Int32(1)
                    else:
                        cute.arch.mbarrier_wait(ab_full_mbars + 1, full_phase1)
                        full_phase1 = full_phase1 ^ cutlass.Int32(1)

                    tcgen05_fence()
                    if lane_idx == 0:
                        for kb in cutlass.range_constexpr(_MK_PACK):
                            coord = (None, None, kb, stage)
                            cute.gemm(tiled_mma, tCtAcc_i,
                                      tCrA[coord], tCrB[coord], tCtAcc_i)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        # tcgen05.commit fires after MMA's smem reads complete → safe release.
                        tcgen05.commit(ab_empty_mbars + stage)

                    stage = stage ^ cutlass.Int32(1)

                if lane_idx == 0:
                    tcgen05.commit(mma_full_mbars + T_idx)

        # ════════════════════════════════════════════════════════════════════
        # DRAIN EPILOGUE — all 128 threads cooperatively read tmem → gmem
        # ════════════════════════════════════════════════════════════════════
        cute.arch.sync_threads()

        for T_idx in cutlass.range_constexpr(T):
            cute.arch.mbarrier_wait(mma_full_mbars + T_idx, cutlass.Int32(0))

            tmem_slot_offset = cutlass.Int32(T_idx * TMEM_COLS_PER_TOKEN)
            tTR_tAcc_i = cute.make_tensor(
                tTR_tAcc_base.iterator + tmem_slot_offset,
                tTR_tAcc_base.layout,
            )

            if tidx < DIM_SPLIT:
                cute.copy(tmem_tiled_copy, tTR_tAcc_i[None, None, 0], tTR_rAcc)
                output[T_idx, 0, tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                output[T_idx, 1, tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

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
    q_nope         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0))
    q_pe           = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0))
    ckv_flat       = _fake(cute.BFloat16, (FLAT_CACHE, HEAD_DIM_CKV), (1, 0))
    kpe_flat       = _fake(cute.BFloat16, (FLAT_CACHE, HEAD_DIM_KPE), (1, 0))
    sparse_indices = _fake(cute.Int32,    (T, DIM_SPLIT), (1, 0), align=4)
    sm_scale       = 0.1352337788608801
    output         = _fake(cute.Float32,  (T, NUM_HEADS, DIM_SPLIT), (2, 1, 0))
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    test = TestScoreSpecialized()
    compiled = cute.compile(
        test, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        output, stream, options="--enable-tvm-ffi",
    )
    return test, compiled


_test, _compiled = compile_test()


# ── run + correctness check ───────────────────────────────────────────────────

def run():
    sm_scale = 0.1352337788608801
    torch.manual_seed(0)

    q_nope         = torch.randn(T, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
    q_pe           = torch.randn(T, NUM_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda")
    ckv_flat       = torch.randn(FLAT_CACHE, HEAD_DIM_CKV,   dtype=torch.bfloat16, device="cuda")
    kpe_flat       = torch.randn(FLAT_CACHE, HEAD_DIM_KPE,   dtype=torch.bfloat16, device="cuda")
    sparse_indices = torch.randint(0, FLAT_CACHE, (T, DIM_SPLIT), dtype=torch.int32, device="cuda")
    output         = torch.zeros(T, NUM_HEADS, DIM_SPLIT, dtype=torch.float32, device="cuda")

    _compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output)
    torch.cuda.synchronize()

    idx = sparse_indices.long()
    ref = torch.empty_like(output)
    for t in range(T):
        gck = ckv_flat[idx[t], :].float()
        gkp = kpe_flat[idx[t], :].float()
        score = q_nope[t].float() @ gck.T + q_pe[t].float() @ gkp.T
        ref[t] = score * sm_scale

    diff = (output.cpu() - ref.cpu()).abs()
    print(f"max_abs_err = {diff.max().item():.6f}   mean_abs_err = {diff.mean().item():.6f}")
    for t in range(T):
        td = (output[t].cpu() - ref[t].cpu()).abs().max().item()
        print(f"  token {t}: max_abs_err = {td:.6f}")

    ok = diff.max().item() < 5e-2
    print(f"PASS={ok}")
    return ok


if __name__ == "__main__":
    run()
