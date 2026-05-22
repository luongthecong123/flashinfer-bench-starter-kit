import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from typing import Tuple
import math
import torch

NUM_HEADS    = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
SEQ_LEN      = 64     # dense KV sequence length (all positions valid)

LN2 = 0.6931471805599453


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class MLA_WMMA:
    """WMMA-based MLA attention with tensor-core output.

    Grid: [1, 1, T].  Block: 1024 threads (32 warps).

    Score phase (tid < 128 only):
      WMMA logits [16, SEQ_LEN] via Q[16,D] @ K[SEQ_LEN,D]^T
      atom_layout_mnk=(1,4,1) -> 128 threads.
      Loads Q_nope/Q_pe and K_nope/K_pe in Bdkc/Bdkp tiles.

    Softmax (all 1024 threads):
      Block-wide max -> exp -> sum -> normalize -> bf16 into sA_out[16, SEQ_LEN].

    Output phase (all 1024 threads):
      WMMA [16, 512, SEQ_LEN]:  sA_out[16, SEQ_LEN] @ V[SEQ_LEN, 512]
      atom_layout_mnk=(1,32,1) -> 1024 threads.
      V is transposed-loaded into sB_out[512, SEQ_LEN].
    """

    def __init__(self):
        self.BM   = 16
        self.BN   = SEQ_LEN       # 64 -- full sequence in one tile
        self.Bdkc = 64            # K-tile for score nope
        self.Bdkp = 64            # K-tile for score pe

        # Score MMA: 128 threads (4 warps)
        self.score_mma_inst    = (16, 8, 16)
        self.score_atom_layout = (1, 4, 1)
        self.score_num_threads = 128

        # Output MMA: 1024 threads (32 warps)
        self.out_mma_inst     = (16, 8, 16)
        self.out_atom_layout  = (1, 32, 1)
        self.num_threads      = 1024
        self.warp_size        = cute.arch.WARP_SIZE

        self.out_BN = HEAD_DIM_CKV  # 512

    @cute.jit
    def __call__(
        self,
        q_nope:   cute.Tensor,    # [T, H, D]  bf16
        q_pe:     cute.Tensor,    # [T, H, Dp] bf16
        kc:       cute.Tensor,    # [T, S, D]  bf16
        kp:       cute.Tensor,    # [T, S, Dp] bf16
        sm_scale: cute.Tensor,    # [1]        float32
        output:   cute.Tensor,    # [T, H, D]  bf16
        lse:      cute.Tensor,    # [T, H]     float32
        stream,
    ):
        T, num_heads, dkc = q_nope.shape

        # --- Score tiled_mma (128 threads) ---
        score_mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32,
            shape_mnk=self.score_mma_inst)
        score_perm = (
            self.score_atom_layout[0] * self.score_mma_inst[0],
            self.score_atom_layout[1] * self.score_mma_inst[1] * 2,
            self.score_atom_layout[2] * self.score_mma_inst[2],
        )
        score_tiled_mma = cute.make_tiled_mma(
            op_or_atom=score_mma_op,
            atom_layout_mnk=self.score_atom_layout,
            permutation_mnk=score_perm)

        # --- Output tiled_mma (1024 threads) ---
        out_mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32,
            shape_mnk=self.out_mma_inst)
        out_perm = (
            self.out_atom_layout[0] * self.out_mma_inst[0],
            self.out_atom_layout[1] * self.out_mma_inst[1] * 2,
            self.out_atom_layout[2] * self.out_mma_inst[2],
        )
        out_tiled_mma = cute.make_tiled_mma(
            op_or_atom=out_mma_op,
            atom_layout_mnk=self.out_atom_layout,
            permutation_mnk=out_perm)

        self.kernel(
            q_nope, q_pe, kc, kp, sm_scale, output, lse,
            score_tiled_mma, out_tiled_mma,
        ).launch(
            grid=[1, 1, T],
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_nope:   cute.Tensor,    # [T, H, D]  bf16
        q_pe:     cute.Tensor,    # [T, H, Dp] bf16
        kc:       cute.Tensor,    # [T, S, D]  bf16
        kp:       cute.Tensor,    # [T, S, Dp] bf16
        sm_scale: cute.Tensor,    # [1]        float32
        output:   cute.Tensor,    # [T, H, D]  bf16
        lse:      cute.Tensor,    # [T, H]     float32
        score_tiled_mma: cute.TiledMma,
        out_tiled_mma: cute.TiledMma,
    ):
        _, S, _    = kc.shape
        _, _, dkc  = q_nope.shape
        _, _, dkp  = q_pe.shape
        _, _, dv   = output.shape

        _, _, batch_idx = cute.arch.block_idx()
        tid, _, _  = cute.arch.thread_idx()
        warp_idx   = cute.arch.warp_idx()
        warp_idx   = cute.arch.make_warp_uniform(warp_idx)
        lane_idx   = cute.arch.lane_idx()

        num_threads: cutlass.Constexpr = 1024
        num_warps:   cutlass.Constexpr = 32

        # ===== Smem allocation =====
        allocator = cutlass.utils.SmemAllocator()

        # Score phase smem: sQ [16, 64], sK [64, 64] -- for K-loop tiles
        sQ_layout = cute.make_layout((self.BM, self.Bdkc), stride=(self.Bdkc, 1))
        sK_layout = cute.make_layout((self.BN, self.Bdkc), stride=(self.Bdkc, 1))
        sQ = allocator.allocate_tensor(cutlass.Float16, sQ_layout, 16, None)
        sK = allocator.allocate_tensor(cutlass.Float16, sK_layout, 16, None)

        # Logits smem: [16, S=64] f32 -- score TC writes here
        sL = allocator.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((self.BM, S), stride=(S, 1)), 16, None)

        sLSE = allocator.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((self.BM,), stride=(1,)), 4, None)

        # Output phase smem:
        # sA_out [16, 64] bf16 -- normalized softmax weights
        sA_out_layout = cute.make_layout((self.BM, self.BN), stride=(self.BN, 1))
        sA_out = allocator.allocate_tensor(cutlass.Float16, sA_out_layout, 16, None)

        # sB_out [512, 64+8] bf16 -- V values (transposed), padded for bank conflicts
        sB_out_layout = cute.make_layout(
            (self.out_BN, self.BN),
            stride=(self.BN + 8, 1))
        sB_out = allocator.allocate_tensor(cutlass.Float16, sB_out_layout, 16, None)

        smem_red_f32 = allocator.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((num_warps,), stride=(1,)), 16, None)

        # (Score MMA setup moved inside if tid < 128 block below)
        tv_layout_C_s = score_tiled_mma.tv_layout_C_tiled
        sL_tile_shape = cute.make_layout((self.BM, self.BN), stride=(self.BN, 1)).shape

        # ===== Output MMA setup (all 1024 threads) =====
        thr_mma_o  = out_tiled_mma.get_slice(tid)
        tCsA_o     = thr_mma_o.partition_A(sA_out)
        tCsB_o     = thr_mma_o.partition_B(sB_out)

        atom_s2r_A_o = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16)
        atom_s2r_B_o = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16)

        tc_s2r_A_o = cute.make_tiled_copy_A(atom_s2r_A_o, out_tiled_mma)
        tc_s2r_B_o = cute.make_tiled_copy_B(atom_s2r_B_o, out_tiled_mma)

        thr_cp_A_o   = tc_s2r_A_o.get_slice(tid)
        thr_cp_B_o   = tc_s2r_B_o.get_slice(tid)
        tCsA_o_cv    = thr_cp_A_o.partition_S(sA_out)
        tCsB_o_cv    = thr_cp_B_o.partition_S(sB_out)

        acc_shape_o   = thr_mma_o.partition_shape_C((self.BM, self.out_BN))
        tv_layout_C_o = out_tiled_mma.tv_layout_C_tiled
        out_shape     = cute.make_layout(
            (self.BM, self.out_BN), stride=(self.out_BN, 1)).shape

        # ===== GMEM views =====
        qn = q_nope[batch_idx, None, None]                         # (H, D)
        gQn_ = cute.zipped_divide(qn, (self.BM, self.Bdkc))       # ((BM, Bdkc), (H//BM, D//Bdkc))

        qp = q_pe[batch_idx, None, None]                           # (H, Dp)
        gQp_ = cute.zipped_divide(qp, (self.BM, self.Bdkp))       # ((BM, Bdkp), (H//BM, Dp//Bdkp))

        kc_batch = kc[batch_idx, None, None]                       # (S, D)
        kp_batch = kp[batch_idx, None, None]                       # (S, Dp)
        gKc_ = cute.zipped_divide(kc_batch, (self.BN, self.Bdkc))  # ((BN, Bdkc), (S//BN, D//Bdkc))
        gKp_ = cute.zipped_divide(kp_batch, (self.BN, self.Bdkp))  # ((BN, Bdkp), (S//BN, Dp//Bdkp))

        gOut_ = cute.zipped_divide(output[batch_idx, None, None], (self.BM, self.out_BN))

        num_heads_tiles = NUM_HEADS // self.BM  # 1

        # ===== Main loop over head groups =====
        for hidx in range(num_heads_tiles):
            gQn = gQn_[(None, None), (hidx, None)]   # (BM, Bdkc, D//Bdkc)
            gQp = gQp_[(None, None), (hidx, None)]   # (BM, Bdkp, Dp//Bdkp)

            sLSE.fill(0.0)

            # -- Score phase: only tid < 128 ------------------------------------
            if tid < 128:
                score_tid = tid
                thr_mma_s  = score_tiled_mma.get_slice(score_tid)
                tCsA_s     = thr_mma_s.partition_A(sQ)
                tCsB_s     = thr_mma_s.partition_B(sK)

                atom_s2r_A_s = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    q_nope.element_type)
                atom_s2r_B_s = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    kc.element_type)

                tc_s2r_A_s = cute.make_tiled_copy_A(atom_s2r_A_s, score_tiled_mma)
                tc_s2r_B_s = cute.make_tiled_copy_B(atom_s2r_B_s, score_tiled_mma)

                thr_cp_A_s   = tc_s2r_A_s.get_slice(score_tid)
                thr_cp_B_s   = tc_s2r_B_s.get_slice(score_tid)
                tCsA_s_cv    = thr_cp_A_s.partition_S(sQ)
                tCsB_s_cv    = thr_cp_B_s.partition_S(sK)

                acc_shape_s  = thr_mma_s.partition_shape_C((self.BM, self.BN))

                tCrA_s = score_tiled_mma.make_fragment_A(tCsA_s)
                tCrB_s = score_tiled_mma.make_fragment_B(tCsB_s)
                tCrC_s = score_tiled_mma.make_fragment_C(acc_shape_s)
                tCrA_s_cv = thr_cp_A_s.retile(tCrA_s)
                tCrB_s_cv = thr_cp_B_s.retile(tCrB_s)

                tCrC_s.fill(0.0)

                gKc = gKc_[(None, None), (0, None)]   # (BN, Bdkc, D//Bdkc)

                # Q_nope @ Kc^T  (K-loop: dkc/Bdkc = 512/64 = 8 tiles)
                for kidx in range(dkc // self.Bdkc):
                    cute.autovec_copy(gQn[None, None, kidx], sQ)
                    cute.autovec_copy(gKc[None, None, kidx], sK)
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    cute.copy(atom=tc_s2r_A_s, src=tCsA_s_cv, dst=tCrA_s_cv)
                    cute.copy(atom=tc_s2r_B_s, src=tCsB_s_cv, dst=tCrB_s_cv)
                    cute.gemm(atom=score_tiled_mma, d=tCrC_s, a=tCrA_s, b=tCrB_s, c=tCrC_s)
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)

                gKp = gKp_[(None, None), (0, None)]   # (BN, Bdkp, Dp//Bdkp)

                # Q_pe @ Kp^T (added into same accumulator)
                for kidx in range(dkp // self.Bdkp):
                    cute.autovec_copy(gQp[None, None, kidx], sQ)
                    cute.autovec_copy(gKp[None, None, kidx], sK)
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    cute.copy(atom=tc_s2r_A_s, src=tCsA_s_cv, dst=tCrA_s_cv)
                    cute.copy(atom=tc_s2r_B_s, src=tCsB_s_cv, dst=tCrB_s_cv)
                    cute.gemm(atom=score_tiled_mma, d=tCrC_s, a=tCrA_s, b=tCrB_s, c=tCrC_s)
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)

                # Write raw logits * sm_scale to sL[16, 64]
                for reg_idx in range(cute.size(tCrC_s)):
                    coord   = cute.idx2crd((score_tid, reg_idx), tv_layout_C_s.shape)
                    mn_flat = cute.crd2idx(coord, tv_layout_C_s)
                    m, n    = cute.idx2crd(mn_flat, sL_tile_shape)
                    sL[m, n] = tCrC_s[reg_idx] * sm_scale[0]

            cute.arch.sync_threads()

            # -- Softmax: block-wide max (all 1024 threads) --------------------
            total_elems = self.BM * S
            partial_max = -cutlass.Float32(math.inf)
            for idx in range(tid, total_elems, num_threads):
                row = idx // S
                col = idx % S
                v = sL[row, col]
                if v > partial_max:
                    partial_max = v

            max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
            if lane_idx == 0:
                smem_red_f32[warp_idx] = max_val
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_red_f32[lane_idx]
                max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
                smem_red_f32[0] = max_val
            cute.arch.sync_threads()
            row_max = smem_red_f32[0]

            # exp + sum
            partial_sum = cutlass.Float32(0)
            for idx in range(tid, total_elems, num_threads):
                row = idx // S
                col = idx % S
                e = cute.math.exp(sL[row, col] - row_max)
                sL[row, col] = e
                partial_sum += e

            sum_val = warp_reduce(partial_sum, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_red_f32[warp_idx] = sum_val
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_red_f32[lane_idx]
                sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
                smem_red_f32[0] = sum_val
            cute.arch.sync_threads()
            row_sum = smem_red_f32[0]

            # Write per-row LSE
            if tid < self.BM:
                lse[batch_idx, hidx * self.BM + tid] = (
                    (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
                )

            # Normalize -> bf16 into sA_out [16, 64]
            for idx in range(tid, total_elems, num_threads):
                row = idx // S
                col = idx % S
                sA_out[row, col] = cutlass.BFloat16(sL[row, col] / row_sum)
            cute.arch.sync_threads()

            # -- Output phase: WMMA [16, 512, 64] -----------------------------
            # Load V into sB_out[512, 64+8]: sB_out[n, k] = kc_batch[k, n]
            num_loads_V = self.BN * self.out_BN   # 64 * 512 = 32768
            for i in range(tid, num_loads_V, num_threads):
                k = i // self.out_BN
                n = i % self.out_BN
                sB_out[n, k] = kc_batch[k, n]
            cute.arch.sync_threads()

            # Output MMA
            tCrA_o = out_tiled_mma.make_fragment_A(tCsA_o)
            tCrB_o = out_tiled_mma.make_fragment_B(tCsB_o)
            tCrC_o = out_tiled_mma.make_fragment_C(acc_shape_o)
            tCrA_o_cv = thr_cp_A_o.retile(tCrA_o)
            tCrB_o_cv = thr_cp_B_o.retile(tCrB_o)

            tCrC_o.fill(0.0)

            cute.copy(atom=tc_s2r_A_o, src=tCsA_o_cv, dst=tCrA_o_cv)
            cute.copy(atom=tc_s2r_B_o, src=tCsB_o_cv, dst=tCrB_o_cv)
            cute.gemm(atom=out_tiled_mma, d=tCrC_o, a=tCrA_o, b=tCrB_o, c=tCrC_o)

            # Store output to global
            gOut = gOut_[(None, None), (hidx, 0)]   # (BM, out_BN)

            for reg_idx in range(cute.size(tCrC_o)):
                coord   = cute.idx2crd((tid, reg_idx), tv_layout_C_o.shape)
                mn_flat = cute.crd2idx(coord, tv_layout_C_o)
                m, n    = cute.idx2crd(mn_flat, out_shape)
                gOut[m, n] = cutlass.BFloat16(tCrC_o[reg_idx])


# --- Compile ------------------------------------------------------------------

def _fake(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape,
        stride_order=stride_order, assumed_align=assumed_align)


def _compile():
    T = cute.sym_int()

    q_nope   = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    q_pe     = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
    kc       = _fake(cute.BFloat16, (T, SEQ_LEN, HEAD_DIM_CKV),   (2, 1, 0), 16)
    kp       = _fake(cute.BFloat16, (T, SEQ_LEN, HEAD_DIM_KPE),   (2, 1, 0), 16)
    sm_scale = _fake(cute.Float32,  (1,),                         (0,),       4)
    output   = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    lse      = _fake(cute.Float32,  (T, NUM_HEADS),               (1, 0),     4)
    stream   = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        MLA_WMMA(),
        q_nope, q_pe, kc, kp, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


_compiled = _compile()


# --- Host wrapper -------------------------------------------------------------

def run(q_nope, q_pe, kc, kp, sm_scale, output, lse):
    sm_scale_t = torch.tensor([sm_scale], dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, kc, kp, sm_scale_t, output, lse)


# --- PyTorch reference --------------------------------------------------------

def ref_run(q_nope, q_pe, kc, kp, sm_scale):
    """Numerically stable reference for dense MLA attention.

    q_nope : [T, H, D]   bf16
    q_pe   : [T, H, Dp]  bf16
    kc     : [T, S, D]   bf16  (per-request KV cache, also used as values)
    kp     : [T, S, Dp]  bf16

    Returns (output [T, H, D] bf16, lse [T, H] float32).
    """
    q_n = q_nope.float()   # [T, H,  D]
    q_p = q_pe.float()     # [T, H, Dp]
    k_c = kc.float()       # [T, S,  D]
    k_p = kp.float()       # [T, S, Dp]

    # scores [T, H, S]
    scores = (
        torch.einsum("thd,tsd->ths", q_n, k_c)
        + torch.einsum("thp,tsp->ths", q_p, k_p)
    ) * sm_scale

    attn = torch.softmax(scores, dim=-1)          # [T, H, S]
    out  = torch.einsum("ths,tsd->thd", attn, k_c).to(torch.bfloat16)
    lse  = torch.logsumexp(scores, dim=-1)        # [T, H]  (log base e)
    return out, lse
