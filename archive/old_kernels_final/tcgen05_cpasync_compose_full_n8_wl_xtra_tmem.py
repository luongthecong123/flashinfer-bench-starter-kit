"""tcgen05_cpasync_compose_full_n8_wl_xtra_tmem.py — per-token tmem-slot isolation.

Isolation test for the FMHA-style trick where each token gets its own dedicated
tmem column slot so producer & consumer can pipeline:

  * Producer pool (warps 0..7, 256 threads) loops over T=8 tokens. For each
    token i it (a) gathers Q+K into shared sA/sB, (b) issues UMMA into the
    8-column tmem slot starting at column i*8, (c) tcgen05.commit(mbar[i]).
    Producer waits mbar[i] before iter i+1 so it can safely reuse sA.

  * Consumer pool (warps 8..15, 256 threads) also loops over T=8 tokens.
    For each token i it (a) waits on mbar[i], (b) loads from tmem at offset
    i*8 via tcgen05.Ld32x32b(rep=2), (c) softmaxes per head, (d) writes
    softmax(score) to gmem.

  * Total tmem allocation: T * 8 = 64 columns.

  * No output (V) phase — this is purely a score+softmax isolation test.

Synthetic data: fully random Q_nope/Q_pe/CKV/KPE; sparse_indices is random in
[0, FLAT_CACHE) per token (T=8 distinct rows-per-token sets), so every token
hits 128 valid sparse rows. Mirrors the geometry of a workload with
max_valid==2048 but doesn't depend on a real workload file.
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
NUM_HEADS = 2                 # heads per split (matches HEADS_PER_SPLIT in v5)
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

# Flat KV cache (page-table layout)
NUM_PAGES  = 8462
PAGE_SIZE  = 64
FLAT_CACHE = NUM_PAGES * PAGE_SIZE                       # 541568

# UMMA inst — per-token: M=128 sparse rows, N=8 (2 useful head rows + 6 unused),
# K=16. 8 tmem columns per token; T*8 = 64 cols total.
UMMA_INST = (DIM_SPLIT, 8, 16)
MMA_M, MMA_N, MMA_K = UMMA_INST
TMEM_COLS_PER_TOKEN = MMA_N                              # 8
TOTAL_TMEM_COLS     = T * TMEM_COLS_PER_TOKEN            # 64
TMEM_LD_REP         = HEADS_PER_SPLIT                    # 2 (one col per head)

MMA_M_PACK, MMA_N_PACK, MMA_K_PACK = 1, 1, 4
MMA_K_PACKED     = MMA_K * MMA_K_PACK                    # 64
MMA_K_TILES      = HEAD_DIM_CKV  // MMA_K_PACKED         # 8
MMA_K_TILES_PE   = HEAD_DIM_KPE  // MMA_K_PACKED         # 1
MMA_K_TILES_FULL = MMA_K_TILES + MMA_K_TILES_PE          # 9

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

class TestScoreXtraTmem:

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

        # 1 CTA total — pure isolation test
        self._kernel(
            tiled_mma, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            sm_scale, output,
        ).launch(grid=[1, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def _kernel(
        self,
        tiled_mma,
        q_nope:         cute.Tensor,  # (T, NUM_HEADS, HEAD_DIM_CKV) bf16
        q_pe:           cute.Tensor,  # (T, NUM_HEADS, HEAD_DIM_KPE) bf16
        ckv_flat:       cute.Tensor,  # (FLAT_CACHE, HEAD_DIM_CKV) bf16
        kpe_flat:       cute.Tensor,  # (FLAT_CACHE, HEAD_DIM_KPE) bf16
        sparse_indices: cute.Tensor,  # (T, DIM_SPLIT) int32
        sm_scale:       cutlass.Constexpr,
        output:         cute.Tensor,  # (T, NUM_HEADS, DIM_SPLIT) f32 — softmax(score)
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

        # 9-panel outer layout: panels 0-7 = CKV/Q_nope, panel 8 = KPE/Q_pe
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

        # Sparse indices: (T, DIM_SPLIT) int32, loaded once at start by all threads.
        smem_sp_indices = alloc.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((T, DIM_SPLIT), stride=(DIM_SPLIT, 1)),
            4,
        )

        # Score smem: (HEADS_PER_SPLIT, DIM_SPLIT) f32 — single-buffered, drained per-token.
        smem_score = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((HEADS_PER_SPLIT, DIM_SPLIT), stride=(DIM_SPLIT, 1)),
            16,
        )

        # SMEM copy views (panel aliasing)
        sA_ckv_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(MMA_M, MMA_K_PACKED, MMA_K_TILES))
        sB_ckv_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(MMA_N, MMA_K_PACKED, MMA_K_TILES))
        panel_stride_A = MMA_M * MMA_K_PACKED * MMA_K_TILES
        panel_stride_B = MMA_N * MMA_K_PACKED * MMA_K_TILES
        sA_kpe_copy = cute.make_tensor(sA.iterator + panel_stride_A, _panel_copy_layout(MMA_M, MMA_K_PACKED, MMA_K_TILES_PE))
        sB_kpe_copy = cute.make_tensor(sB.iterator + panel_stride_B, _panel_copy_layout(MMA_N, MMA_K_PACKED, MMA_K_TILES_PE))

        k_split_shape    = cute.make_layout(((MMA_K_PACKED, MMA_K_TILES),))
        k_split_shape_pe = cute.make_layout(((MMA_K_PACKED, MMA_K_TILES_PE),))

        # cp.async tiled copies
        atom_cpa = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy = cute.make_tiled_copy_tv(atom_cpa, thr_layout, val_layout)
        lane_copy  = tiled_copy.get_slice(lane_idx)

        atom_cpa_pe = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        val_layout_pe = cute.make_layout(((2, 1),), stride=((1, 0),))
        tiled_copy_pe = cute.make_tiled_copy_tv(atom_cpa_pe, thr_layout, val_layout_pe)
        lane_copy_pe  = tiled_copy_pe.get_slice(lane_idx)

        # ── tmem alloc + mbar init ────────────────────────────────────────────
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape       = tiled_mma.partition_shape_C((MMA_M, MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        # Allocate enough columns for ALL T tokens packed side-by-side.
        per_token_cols  = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        # Sanity in constexpr — should equal MMA_N == TMEM_COLS_PER_TOKEN (=8).
        tmem_alloc_cols = cutlass.Int32(per_token_cols * T)

        # All threads load sparse_indices into smem (T*DIM_SPLIT = 1024 i32s, 512 thr → 2 each).
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

        # Build a reusable tmem-load tiled-copy from the per-token acc layout.
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        M_acc           = cute.size(tCtAcc_base, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(TMEM_LD_REP))
        epi_tiler       = ((M_acc, TMEM_LD_REP),)
        tCtAcc_epi_base = cute.zipped_divide(tCtAcc_base, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, cutlass.Float32)
        # Consumers do tmem ld; partition by consumer-relative tidx (0..255).
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
                # Wait MMA(i-1) before reusing sA — phase=0 (each mbar single-use).
                if i > 0:
                    cute.arch.mbarrier_wait(mbar_base + (i - 1), cutlass.Int32(0))

                # ── Q_nope[i], Q_pe[i] → sB rows 0..1 (warps 0,1 active) ──
                if warp_idx < HEADS_PER_SPLIT:
                    cute.copy(atom_cpa,
                              lane_copy.partition_S(cute.composition(q_nope[i, warp_idx, None], k_split_shape)),
                              lane_copy.partition_D(sB_ckv_copy[warp_idx, None]))
                    cute.copy(atom_cpa_pe,
                              lane_copy_pe.partition_S(cute.composition(q_pe[i, warp_idx, None], k_split_shape_pe)),
                              lane_copy_pe.partition_D(sB_kpe_copy[warp_idx, None]))

                # ── gathered K → sA (8 producer warps × 16 rounds = 128 rows) ──
                for round_idx in range(num_rounds):
                    row_idx  = round_idx * NUM_PROD_WARPS + warp_idx
                    flat_row = smem_sp_indices[i, row_idx]
                    cute.copy(atom_cpa,
                              lane_copy.partition_S(cute.composition(ckv_flat[flat_row, None], k_split_shape)),
                              lane_copy.partition_D(sA_ckv_copy[row_idx, None]))
                    cute.copy(atom_cpa_pe,
                              lane_copy_pe.partition_S(cute.composition(kpe_flat[flat_row, None], k_split_shape_pe)),
                              lane_copy_pe.partition_D(sA_kpe_copy[row_idx, None]))

                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=PROD_BAR_ID, number_of_threads=PROD_THREADS)

                # ── UMMA into tmem slot i (offset = i * TMEM_COLS_PER_TOKEN) ──
                tcgen05_fence()
                if warp_idx == 0:
                    tCtAcc_i = cute.make_tensor(
                        tmem_ptr + cutlass.Int32(i * TMEM_COLS_PER_TOKEN),
                        tCtAcc_tmpl.layout,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    num_k_blocks = cute.size(tCrA, mode=[2])      # = 9
                    for k_block_idx in range(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx)
                        cute.gemm(tiled_mma, tCtAcc_i,
                                  tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc_i)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    if lane_idx == 0:
                        tcgen05.commit(mbar_base + i)

            # Final wait so producer doesn't exit before MMA(T-1) drains.
            cute.arch.mbarrier_wait(mbar_base + (T - 1), cutlass.Int32(0))

        # ════════════════════════════════════════════════════════════════════
        # CONSUMER pool — warps 8..15
        # ════════════════════════════════════════════════════════════════════
        if is_consumer:
            cons_warp_idx = warp_idx - NUM_PROD_WARPS         # 0..7

            for i in cutlass.range_constexpr(T):
                cute.arch.mbarrier_wait(mbar_base + i, cutlass.Int32(0))

                # ── Per-token tmem view via iterator offset (FMHA trick) ──
                tTR_tAcc_i = cute.make_tensor(
                    tTR_tAcc_base.iterator + cutlass.Int32(i * TMEM_COLS_PER_TOKEN),
                    tTR_tAcc_base.layout,
                )

                # 128 cons threads load (DIM_SPLIT × HEADS_PER_SPLIT) f32 → 2 elems/thread.
                if cons_tidx < DIM_SPLIT:
                    cute.copy(tmem_tiled_copy, tTR_tAcc_i[None, None, 0], tTR_rAcc)
                    smem_score[0, cons_tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                    smem_score[1, cons_tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

                cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=CONS_THREADS)

                # ── Per-head softmax: 1 warp per head (warps 0..1 in cons-relative) ──
                if cons_warp_idx < HEADS_PER_SPLIT:
                    NUM_ELEMS: cutlass.Constexpr = DIM_SPLIT // 32       # 4
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

    test = TestScoreXtraTmem()
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

    # Reference: gather + matmul + scaled softmax per token
    idx = sparse_indices.long()                                          # (T, 128)
    ref = torch.empty_like(output)
    for t in range(T):
        gck = ckv_flat[idx[t], :].float()                                # (128, 512)
        gkp = kpe_flat[idx[t], :].float()                                # (128, 64)
        score = q_nope[t].float() @ gck.T + q_pe[t].float() @ gkp.T      # (2, 128)
        score = score * sm_scale
        ref[t] = torch.softmax(score, dim=-1)

    diff = (output.cpu() - ref.cpu()).abs()
    print(f"max_abs_err = {diff.max().item():.6f}   mean_abs_err = {diff.mean().item():.6f}")

    # Per-token max-err breakdown so we can see which slot fails
    for t in range(T):
        td = (output[t].cpu() - ref[t].cpu()).abs().max().item()
        print(f"  token {t}: max_abs_err = {td:.6f}")

    ok = diff.max().item() < 5e-3
    print(f"PASS={ok}")
    return ok


if __name__ == "__main__":
    run()
