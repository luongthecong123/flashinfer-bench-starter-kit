"""tcgen05_cpasync_compose_full_n8_wl_xtra_tmem_stages.py

Variant of `tcgen05_cpasync_compose_full_n8_wl_xtra_tmem.py` where the per-token
loads are split into stages and run sequentially:

    load PE              → wait → MMA(PE)
    load CKV stage 0     → wait → MMA(stage 0)
    load CKV stage 1     → wait → MMA(stage 1)
    load CKV stage 2     → wait → MMA(stage 2)
    load CKV stage 3     → wait → MMA(stage 3)
    tcgen05.commit(mbar[i])

Same block geometry as the parent kernel: 16 warps / 512 threads, 8 producer
warps + 8 consumer warps, per-token tmem-slot isolation, 9-panel sA/sB
allocation.
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

NUM_PROD_WARPS = 8
NUM_CONS_WARPS = 8
NUM_WARPS      = NUM_PROD_WARPS + NUM_CONS_WARPS         # 16
NUM_THREADS    = NUM_WARPS * 32                          # 512
PROD_THREADS   = NUM_PROD_WARPS * 32                     # 256
CONS_THREADS   = NUM_CONS_WARPS * 32                     # 256

NUM_PAGES  = 8462
PAGE_SIZE  = 64
FLAT_CACHE = NUM_PAGES * PAGE_SIZE                       # 541568

UMMA_INST = (DIM_SPLIT, 8, 16)
MMA_M, MMA_N, MMA_K = UMMA_INST
TMEM_COLS_PER_TOKEN = MMA_N                              # 8
TOTAL_TMEM_COLS     = T * TMEM_COLS_PER_TOKEN            # 64
TMEM_LD_REP         = HEADS_PER_SPLIT                    # 2

MMA_M_PACK, MMA_N_PACK, MMA_K_PACK = 1, 1, 4
MMA_K_PACKED     = MMA_K * MMA_K_PACK                    # 64
MMA_K_TILES      = HEAD_DIM_CKV  // MMA_K_PACKED         # 8 CKV panels
MMA_K_TILES_PE   = HEAD_DIM_KPE  // MMA_K_PACKED         # 1 PE panel
MMA_K_TILES_FULL = MMA_K_TILES + MMA_K_TILES_PE          # 9

# ── chunking (CKV split into 4 stages of 2 panels each) ─────────────────────
PANELS_PER_CHUNK: cutlass.Constexpr = 2
NUM_CKV_CHUNKS:   cutlass.Constexpr = MMA_K_TILES // PANELS_PER_CHUNK   # 4
CHUNK_PACKED:     cutlass.Constexpr = MMA_K_PACKED * PANELS_PER_CHUNK   # 128

# Flat k_block index into tCrA / tCrB. The K mode is (MMA_K_PACK=4, MMA_K_TILES_FULL=9)
# laid out as k_flat = pack + 4*panel. Panels 0..7 = CKV, panel 8 = PE.
CKV_KBLOCKS_PER_CHUNK: cutlass.Constexpr = MMA_K_PACK * PANELS_PER_CHUNK  # 8

PROD_BAR_ID = 1
CONS_BAR_ID = 2


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _panel_copy_layout(num_rows: int, k_packed: int, k_tiles: int):
    return cute.make_layout(
        (num_rows, (k_packed, k_tiles)),
        stride=(k_packed, (1, num_rows * k_packed)),
    )


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ── kernel ────────────────────────────────────────────────────────────────────

class TestScoreXtraTmemStages:

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
            mma_mbars:        cute.struct.MemRange[cutlass.Int64, T]
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

        is_producer = warp_idx < NUM_PROD_WARPS
        is_consumer = warp_idx >= NUM_PROD_WARPS

        # ── smem allocation ───────────────────────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()
        storage = alloc.allocate(self.shared_storage)
        mbar_base = storage.mma_mbars.data_ptr()

        swizzle = cute.make_swizzle(3, 4, 3)

        # 9-panel outer layout (unchanged from parent): panels 0-7 = CKV, panel 8 = KPE.
        a_outer = cute.make_layout(
            ((MMA_M, MMA_K), MMA_M_PACK, (MMA_K_PACK, MMA_K_TILES_FULL)),
            stride=((MMA_K_PACKED, 1), 0, (MMA_K, MMA_M * MMA_K_PACKED)),
        )
        b_outer = cute.make_layout(
            ((MMA_N, MMA_K), MMA_N_PACK, (MMA_K_PACK, MMA_K_TILES_FULL)),
            stride=((MMA_K_PACKED, 1), 0, (MMA_K, MMA_N * MMA_K_PACKED)),
        )
        sA = alloc.allocate_tensor(cutlass.BFloat16, a_outer, byte_alignment=16, swizzle=swizzle)
        sB = alloc.allocate_tensor(cutlass.BFloat16, b_outer, byte_alignment=16, swizzle=swizzle)

        smem_sp_indices = alloc.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((T, DIM_SPLIT), stride=(DIM_SPLIT, 1)),
            4,
        )
        smem_score = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((HEADS_PER_SPLIT, DIM_SPLIT), stride=(DIM_SPLIT, 1)),
            16,
        )

        # ── SMEM copy views ───────────────────────────────────────────────────
        panel_stride_A: cutlass.Constexpr = MMA_M * MMA_K_PACKED
        panel_stride_B: cutlass.Constexpr = MMA_N * MMA_K_PACKED
        chunk_stride_A: cutlass.Constexpr = panel_stride_A * PANELS_PER_CHUNK
        chunk_stride_B: cutlass.Constexpr = panel_stride_B * PANELS_PER_CHUNK

        # PE-panel views (panel 8 lives at offset 8*panel_stride).
        sA_kpe_copy = cute.make_tensor(
            sA.iterator + MMA_K_TILES * panel_stride_A,
            _panel_copy_layout(MMA_M, MMA_K_PACKED, MMA_K_TILES_PE),
        )
        sB_kpe_copy = cute.make_tensor(
            sB.iterator + MMA_K_TILES * panel_stride_B,
            _panel_copy_layout(MMA_N, MMA_K_PACKED, MMA_K_TILES_PE),
        )

        k_split_shape_chunk = cute.make_layout(((MMA_K_PACKED, PANELS_PER_CHUNK),))
        k_split_shape_pe    = cute.make_layout(((MMA_K_PACKED, MMA_K_TILES_PE),))

        # ── cp.async tiled copies ─────────────────────────────────────────────
        # Chunked CKV copy: 128 bf16 per row = 32 threads × 4 bf16 (64-bit copies).
        atom_cpa_chunk   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=64)
        thr_layout       = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout_chunk = cute.make_layout(((4, 1),), stride=((1, 0),))
        tiled_copy_chunk = cute.make_tiled_copy_tv(atom_cpa_chunk, thr_layout, val_layout_chunk)
        lane_copy_chunk  = tiled_copy_chunk.get_slice(lane_idx)

        # PE copy: 64 bf16 per row = 32 threads × 2 bf16 (32-bit copies).
        atom_cpa_pe = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        val_layout_pe = cute.make_layout(((2, 1),), stride=((1, 0),))
        tiled_copy_pe = cute.make_tiled_copy_tv(atom_cpa_pe, thr_layout, val_layout_pe)
        lane_copy_pe  = tiled_copy_pe.get_slice(lane_idx)

        # ── tmem alloc + mbar init ────────────────────────────────────────────
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape       = tiled_mma.partition_shape_C((MMA_M, MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        per_token_cols  = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(per_token_cols * T)

        for k in range(2):
            flat_idx = k * NUM_THREADS + tidx
            row = flat_idx // DIM_SPLIT
            col = flat_idx %  DIM_SPLIT
            if row < T:
                smem_sp_indices[row, col] = sparse_indices[row, col]

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                for i in range(T):
                    cute.arch.mbarrier_init(mbar_base + i, cnt=1)
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
        cons_tidx       = tidx - PROD_THREADS
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi_base[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(cons_tidx)
        tTR_tAcc_base   = tmem_thr_copy.partition_S(tCtAcc_epi_base)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc_base[None, None, 0].shape, cutlass.Float32)

        # ════════════════════════════════════════════════════════════════════
        # PRODUCER pool — warps 0..7
        # ════════════════════════════════════════════════════════════════════
        if is_producer:
            num_rounds = DIM_SPLIT // NUM_PROD_WARPS         # 16 rows per producer warp

            for i in cutlass.range_constexpr(T):
                if i > 0:
                    cute.arch.mbarrier_wait(mbar_base + (i - 1), cutlass.Int32(0))

                tCtAcc_i = cute.make_tensor(
                    tmem_ptr + cutlass.Int32(i * TMEM_COLS_PER_TOKEN),
                    tCtAcc_tmpl.layout,
                )

                # ── stage PE: load q_pe + kpe → wait → MMA(PE) ────────────────
                if warp_idx < HEADS_PER_SPLIT:
                    cute.copy(atom_cpa_pe,
                              lane_copy_pe.partition_S(cute.composition(q_pe[i, warp_idx, None], k_split_shape_pe)),
                              lane_copy_pe.partition_D(sB_kpe_copy[warp_idx, None]))
                for round_idx in range(num_rounds):
                    row_idx  = round_idx * NUM_PROD_WARPS + warp_idx
                    flat_row = smem_sp_indices[i, row_idx]
                    cute.copy(atom_cpa_pe,
                              lane_copy_pe.partition_S(cute.composition(kpe_flat[flat_row, None], k_split_shape_pe)),
                              lane_copy_pe.partition_D(sA_kpe_copy[row_idx, None]))
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=PROD_BAR_ID, number_of_threads=PROD_THREADS)

                tcgen05_fence()
                if warp_idx == 0:
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in range(MMA_K_PACK):
                        k_flat = MMA_K_TILES * MMA_K_PACK + kb       # PE panel = panel 8
                        coord = (None, None, k_flat)
                        cute.gemm(tiled_mma, tCtAcc_i,
                                  tCrA[coord], tCrB[coord], tCtAcc_i)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # ── stages 0..3: load CKV chunk c → wait → MMA(chunk c) ───────
                for c in cutlass.range_constexpr(NUM_CKV_CHUNKS):
                    sA_chunk = cute.make_tensor(
                        sA.iterator + c * chunk_stride_A,
                        _panel_copy_layout(MMA_M, MMA_K_PACKED, PANELS_PER_CHUNK),
                    )
                    sB_chunk = cute.make_tensor(
                        sB.iterator + c * chunk_stride_B,
                        _panel_copy_layout(MMA_N, MMA_K_PACKED, PANELS_PER_CHUNK),
                    )

                    if warp_idx < HEADS_PER_SPLIT:
                        q_nope_chunk = cute.make_tensor(
                            q_nope[i, warp_idx, None].iterator + c * CHUNK_PACKED,
                            cute.make_layout((CHUNK_PACKED,), stride=(1,)),
                        )
                        cute.copy(atom_cpa_chunk,
                                  lane_copy_chunk.partition_S(cute.composition(q_nope_chunk, k_split_shape_chunk)),
                                  lane_copy_chunk.partition_D(sB_chunk[warp_idx, None]))

                    for round_idx in range(num_rounds):
                        row_idx  = round_idx * NUM_PROD_WARPS + warp_idx
                        flat_row = smem_sp_indices[i, row_idx]
                        ckv_chunk = cute.make_tensor(
                            ckv_flat[flat_row, None].iterator + c * CHUNK_PACKED,
                            cute.make_layout((CHUNK_PACKED,), stride=(1,)),
                        )
                        cute.copy(atom_cpa_chunk,
                                  lane_copy_chunk.partition_S(cute.composition(ckv_chunk, k_split_shape_chunk)),
                                  lane_copy_chunk.partition_D(sA_chunk[row_idx, None]))

                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=PROD_BAR_ID, number_of_threads=PROD_THREADS)

                    tcgen05_fence()
                    if warp_idx == 0:
                        for kb in range(CKV_KBLOCKS_PER_CHUNK):
                            k_flat = c * CKV_KBLOCKS_PER_CHUNK + kb
                            coord = (None, None, k_flat)
                            cute.gemm(tiled_mma, tCtAcc_i,
                                      tCrA[coord], tCrB[coord], tCtAcc_i)

                if warp_idx == 0 and lane_idx == 0:
                    tcgen05.commit(mbar_base + i)

            cute.arch.mbarrier_wait(mbar_base + (T - 1), cutlass.Int32(0))

        # ════════════════════════════════════════════════════════════════════
        # CONSUMER pool — warps 8..15  (unchanged from parent)
        # ════════════════════════════════════════════════════════════════════
        if is_consumer:
            cons_warp_idx = warp_idx - NUM_PROD_WARPS

            for i in cutlass.range_constexpr(T):
                cute.arch.mbarrier_wait(mbar_base + i, cutlass.Int32(0))

                tTR_tAcc_i = cute.make_tensor(
                    tTR_tAcc_base.iterator + cutlass.Int32(i * TMEM_COLS_PER_TOKEN),
                    tTR_tAcc_base.layout,
                )

                if cons_tidx < DIM_SPLIT:
                    cute.copy(tmem_tiled_copy, tTR_tAcc_i[None, None, 0], tTR_rAcc)
                    smem_score[0, cons_tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                    smem_score[1, cons_tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

                cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=CONS_THREADS)

                if cons_warp_idx < HEADS_PER_SPLIT:
                    NUM_ELEMS: cutlass.Constexpr = DIM_SPLIT // 32
                    smem_score_warp = cute.zipped_divide(smem_score, (1, NUM_ELEMS))
                    vec = smem_score_warp[(0, None), (cons_warp_idx, lane_idx)].load()

                    vec_buf = cute.make_rmem_tensor(
                        cute.make_layout((NUM_ELEMS,), stride=(1,)), cutlass.Float32)
                    for v_idx in range(NUM_ELEMS):
                        vec_buf[v_idx] = vec[v_idx]

                    row_max = -cutlass.Float32(math.inf)
                    for v_idx in range(NUM_ELEMS):
                        row_max = cute.arch.fmax(row_max, vec_buf[v_idx])
                    row_max = warp_reduce(row_max, cute.arch.fmax)

                    row_sum = cutlass.Float32(0)
                    for v_idx in range(NUM_ELEMS):
                        e = cute.math.exp(vec_buf[v_idx] - row_max)
                        vec_buf[v_idx] = e
                        row_sum += e
                    row_sum = warp_reduce(row_sum, lambda a, b: a + b)
                    inv_sum = cutlass.Float32(1.0) / row_sum

                    for v_idx in range(NUM_ELEMS):
                        col_idx = lane_idx * NUM_ELEMS + v_idx
                        output[i, cons_warp_idx, col_idx] = vec_buf[v_idx] * inv_sum

                cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=CONS_THREADS)

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

    test = TestScoreXtraTmemStages()
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
        score = score * sm_scale
        ref[t] = torch.softmax(score, dim=-1)

    diff = (output.cpu() - ref.cpu()).abs()
    print(f"max_abs_err = {diff.max().item():.6f}   mean_abs_err = {diff.mean().item():.6f}")
    for t in range(T):
        td = (output[t].cpu() - ref[t].cpu()).abs().max().item()
        print(f"  token {t}: max_abs_err = {td:.6f}")

    ok = diff.max().item() < 5e-3
    print(f"PASS={ok}")
    return ok


if __name__ == "__main__":
    run()
