"""score_scale_full_bt_ws_cpasync — cp.async A-load into UMMA-swizzled sA.

Combines src/kernels/score_scale_full_bt_ws.py with the v2 cp.async A-load
strategy (from src/kernels/load_a_v2.py). Adds intra-CTA probing.

Sync ordering (key request):
    fire cp.async A                ─┐
    cp_async_commit_group           │  in flight together
    fire TMA B + arrive_expect_tx  ─┘
    mbarrier_wait(tma_mbar)         <-- TMA sync first
    cp_async_wait_group(0)          <-- cp.async sync RIGHT AFTER TMA sync
    sync_threads
    tcgen05_fence
    MMA / commit / wait
    epilogue

A-load destination is the SAME UMMA-swizzled sA used by v1 / score_scale_full_bt_ws.
The fp8 gA is built from an explicit `make_ptr(Int32, ..., assumed_align=4)` and
recast back to fp8, so alignment metadata propagates through partition_A and the
final per-thread Int32 recast satisfies the 32-bit cp.async alignment requirement.
"""
import math, json, torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05

from src.kernels.score_scale_bt_v0_intra import (
    globaltimer_u64, smid_u32, range_start, range_stop, range_finalize,
    PROBE_HEADER, PROBE_ENTRY, MAX_ENTRIES, PROBE_COLS,
)

TAGS = {"total": 0, "fire": 2, "wait_tma": 4, "wait_cpa": 6, "mma": 8, "epilogue": 10}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["total", "fire", "wait_tma", "wait_cpa", "mma", "epilogue"]


def dump_probe(probe: torch.Tensor, num_blocks: int) -> str:
    probe_cpu = probe.cpu().contiguous().tolist()
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            if int(data[off + 1]) == TAGS.get("total", -1):
                t = int(data[off + 3])
                if t > max_dur:
                    max_dur, max_bid = t, bid
                break
    data = probe_cpu[max_bid]; cnt = int(data[0])
    print(f"\n--- Block {max_bid} (longest total={max_dur/1000:.2f} µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id, tag = int(data[off]), int(data[off + 1])
        dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>15s}  dur={dur:>10} ns  ({dur/1000:.2f} µs)")
    tag_totals, tag_counts = {}, {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1
    print(f"\n{'='*64}")
    print(f"{'Phase':>16s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>12s} {'% of total':>12s}")
    print(f"{'='*64}")
    total_ref = tag_totals.get("total", 0)
    for name in PHASE_ORDER:
        if name in tag_totals:
            total_ns = tag_totals[name]; count = tag_counts[name]
            pct = 100.0 * total_ns / total_ref if total_ref > 0 else 0
            print(f"{name:>16s} {total_ns/1e6:>12.3f} {count:>6d} {total_ns/count/1000:>12.2f} {pct:>11.1f}%")
    events, gb = [], None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (gb is None or s < gb): gb = s
    gb = gb or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        if cnt == 0: continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag, start, dur = int(data[off + 1]), int(data[off + 2]), int(data[off + 3])
            if start == 0 and dur == 0: continue
            events.append(dict(name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - gb)/1000.0, dur=dur/1000.0, pid=sm_id, tid=bid))
    return json.dumps({"traceEvents": events})


# ── Constants ────────────────────────────────────────────────────────────────
PAGE_SIZE  = 64
N          = 64
HEAD_DIM   = 128
ROW_STRIDE = 132
PAGES_PER_TILE = 2

MMA_INST_MNK = (128, 64, 32)
BM, BN, BK   = 128, N, HEAD_DIM

THREADS_PER_CTA   = 512        # CTA size: 512 threads load A via cp.async
COMPUTE_THREADS   = 128        # threads that do TMA/MMA/epilogue
TMEM_LD_REP     = N
# Named-barrier IDs (avoid deadlock when 384 threads exit early).
INIT_BAR_ID    = 1   # 512-wide: tmem alloc + mbarrier init visible to all
EPI_BAR_ID     = 2   # 128-wide: epilogue → dealloc_tmem fence (compute-only)

WS_ROWS = 128
WS_COLS = 640000

ROW_STRIDE_I32 = ROW_STRIDE // 4   # 33
HEAD_DIM_I32   = HEAD_DIM   // 4   # 32


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(None, [],
        "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


class ScoreScaleFullBTWSCpAsyncProfiled:
    def __init__(self):
        self.threads_per_cta    = THREADS_PER_CTA
        self.num_stages         = 1
        self.tmem_ld_rep        = TMEM_LD_REP
        self.cta_tile_mnk       = (BM, BN, BK)
        self.mma_inst_shape_mnk = MMA_INST_MNK
        self.workspace = torch.empty(WS_ROWS, WS_COLS, dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(self, kv_pool, block_table, q, w, workspace, probe, stream):
        self.fp8_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32

        num_pg  = cute.size(block_table, mode=[0])
        grid_m  = num_pg // PAGES_PER_TILE

        op = tcgen05.MmaFP8Op(
            self.fp8_dtype, self.acc_dtype, self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.cta_tile_mnk, self.fp8_dtype, self.num_stages)
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.cta_tile_mnk, q.element_type, self.num_stages)

        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])
        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, q, b_smem_layout_one_stage, self.cta_tile_mnk, self.tiled_mma)

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma, kv_pool, block_table,
            tma_atom_b, tma_tensor_b, w, workspace, probe,
            self.a_smem_layout, self.b_smem_layout,
        ).launch(grid=(grid_m, 1, 1), block=(self.threads_per_cta, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self, tiled_mma, kv_pool, block_table, tma_atom_b, mB_tma_tensor,
        w, workspace, probe, a_smem_layout, b_smem_layout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)
        bidx, _, _  = cute.arch.block_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_b)

        smem    = cutlass.utils.SmemAllocator()
        # CRITICAL: sA MUST be allocated at SMEM offset 0. The cp.async path
        # binds Sw<3,2,3>∘row_major to an i32 view of sA. Swizzle composition
        # via recast_ptr does NOT properly subtract the SMEM base offset, so
        # if `storage` is allocated first sA sits at offset>0 and Sw<3,2,3>
        # XORs against absolute address bits → wrong byte placement.
        # (Confirmed by load_a_v2_swizzled_cmp_faithful2.py: PASS at offset 0,
        # FAIL with 128-byte pad before sA.)
        sA = smem.allocate_tensor(
            element_type=self.fp8_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        # i32 view of sA for cp.async writes — uses sA's iterator (aliases sA
        # at the IR level so MMA reads see cp.async writes).
        sA_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BM, HEAD_DIM_I32),
                             stride=(HEAD_DIM_I32, 1)),
        )
        sA_i32_ptr = cute.recast_ptr(sA.iterator, dtype=cutlass.Int32)
        sA_load = cute.make_tensor(sA_i32_ptr, sA_load_layout)
        storage = smem.allocate(self.shared_storage)
        sB = smem.allocate_tensor(self.fp8_dtype, b_smem_layout.outer, 128, b_smem_layout.inner)
        sScales  = smem.allocate_tensor(cutlass.Float32,
                       cute.make_layout(self.threads_per_cta), 16, None)
        sWeights = smem.allocate_tensor(cutlass.Float32,
                       cute.make_layout(N), 16, None)

        # ── Per-CTA pages ────────────────────────────────────────────
        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])
        page_stride_b   = PAGE_SIZE * ROW_STRIDE
        page_stride_i32 = page_stride_b // 4
        page0_off_i32   = page0_id * page_stride_i32
        jump_i32        = (page1_id - page0_id) * page_stride_i32
        # also fp8/byte versions for partition_A
        page0_off_b = page0_id * page_stride_b
        jump_b      = (page1_id - page0_id) * page_stride_b

        # ── GMEM A view (page-base addresses are 16-aligned) ──────────
        u8_off_ptr = kv_pool.iterator + page0_off_b
        fp8_base_aligned = cute.make_ptr(
            self.fp8_dtype, u8_off_ptr.toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=16,
        )
        gA = cute.make_tensor(fp8_base_aligned, cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
            stride=((ROW_STRIDE, jump_b), 1),
        ))

        gB = cute.local_tile(mB_tma_tensor, self.cta_tile_mnk, (bidx, 0, None), proj=(None, 1, 1))

        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA)
        tCgB    = thr_mma.partition_B(gB)
        tCrA    = tiled_mma.make_fragment_A(sA)
        tCrB    = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(self.cta_tile_mnk[:2])
        tCtAcc    = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()
        tma_transaction_bytes = cute.size_in_bytes(
            self.fp8_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(tma_mbar, cnt=1)
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        # 512-wide barrier: tmem alloc + mbarrier init visible to all threads.
        cute.arch.barrier(barrier_id=INIT_BAR_ID, number_of_threads=self.threads_per_cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # TMEM epilogue setup
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler  = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        probe_row = bidx
        sm = cutlass.Int64(smid_u32())
        probe_cnt = cutlass.Int32(0)

        # ════════════════════════════════════════════════════════════
        # total
        # ════════════════════════════════════════════════════════════
        if tidx == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["total"])
            probe_cnt = probe_cnt + cutlass.Int32(1)
            range_start(probe, probe_row, probe_cnt, sm, TAGS["fire"])

        # ════════════════════════════════════════════════════════════
        # FIRE: A-load — 512-thread cp.async via TV-layout PISL view
        # ════════════════════════════════════════════════════════════
        i32_base = cute.make_ptr(
            cutlass.Int32,
            (cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Int32) + page0_off_i32).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=4,
        )
        gA_i32 = cute.make_tensor(i32_base, cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM_I32),
            stride=((ROW_STRIDE_I32, jump_i32), 1),
        ))
        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
        )
        N_PER_THREAD_I32 = (BM * HEAD_DIM_I32) // THREADS_PER_CTA  # 8
        thr_layout_load  = cute.make_layout((16, HEAD_DIM_I32),
                                            stride=(HEAD_DIM_I32, 1))
        val_layout_load  = cute.make_layout((N_PER_THREAD_I32, 1),
                                            stride=(1, 1))
        tiled_copy_a = cute.make_tiled_copy_tv(atom_cpa, thr_layout_load, val_layout_load)
        thr_copy_a   = tiled_copy_a.get_slice(tidx)
        tAgA = thr_copy_a.partition_S(gA_i32)
        tAsA = thr_copy_a.partition_D(sA_load)
        cute.copy(atom_cpa, tAgA, tAsA)
        cute.arch.cp_async_commit_group()

        # TMA B fire + scale/weight loads happen on the 128-thread compute group.
        # The 384 extra threads only participate in the cp.async A load above.
        if tidx < COMPUTE_THREADS:
            if warp_idx == 0:
                cute.copy(tma_atom_b, tBgB[None, 0], tBsB[None, 0], tma_bar_ptr=tma_mbar)
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_transaction_bytes)

            SCALE_ROW_STRIDE_F32 = ROW_STRIDE // 4
            page_stride_f32      = PAGE_SIZE * SCALE_ROW_STRIDE_F32
            page0_off_f32        = page0_id * page_stride_f32
            jump_f32             = (page1_id - page0_id) * page_stride_f32

            fp32_base = cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Float32) + page0_off_f32
            scale_ptr = fp32_base + (HEAD_DIM // 4)
            scale_layout = cute.make_layout(((PAGE_SIZE, PAGES_PER_TILE),),
                                            stride=((SCALE_ROW_STRIDE_F32, jump_f32),))
            gScale = cute.make_tensor(scale_ptr, scale_layout)
            sScales[tidx] = gScale[tidx]
            if tidx < N:
                sWeights[tidx] = w[tidx]

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)
            range_start(probe, probe_row, probe_cnt, sm, TAGS["wait_tma"])

        # ════════════════════════════════════════════════════════════
        # WAIT TMA (compute group only) → WAIT cp.async (all 512 sync)
        # ════════════════════════════════════════════════════════════
        if tidx < COMPUTE_THREADS:
            cute.arch.mbarrier_wait(tma_mbar, 0)

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)
            range_start(probe, probe_row, probe_cnt, sm, TAGS["wait_cpa"])

        # All 512 threads issued cp.async → must all sync to make sA visible.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # cp.async writes go through the GENERIC proxy; UMMA reads SMEM via
        # the ASYNC proxy. Without this fence UMMA may see stale data.
        cute.arch.fence_view_async_shared()

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)
            range_start(probe, probe_row, probe_cnt, sm, TAGS["mma"])

        # Compute group (128 threads) does MMA + epilogue. The 384 extra
        # threads skip everything below and fall through to kernel exit.
        if tidx < COMPUTE_THREADS:
            tcgen05_fence()

            # ════════════════════════════════════════════════════════════
            # MMA + commit + wait
            # ════════════════════════════════════════════════════════════
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            num_k_blocks = cute.size(tCrA, mode=[2])

            if warp_idx == 0:
                for k_block_idx in range(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, 0)
                    cute.gemm(tiled_mma, tCtAcc,
                              tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                if tidx == 0:
                    tcgen05.commit(mma_mbar)

            cute.arch.mbarrier_wait(mma_mbar, 0)

            if tidx == 0:
                probe_cnt = range_stop(probe, probe_row, probe_cnt)
                range_start(probe, probe_row, probe_cnt, sm, TAGS["epilogue"])

            # ════════════════════════════════════════════════════════════
            # epilogue
            # ════════════════════════════════════════════════════════════
            if warp_idx == 0:
                cute.arch.relinquish_tmem_alloc_permit()

            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

            scale   = sScales[tidx]
            out_val = cutlass.Float32(0)
            for n_idx in cutlass.range_constexpr(N):
                val      = tTR_rAcc[n_idx] * scale
                out_val  = out_val + max(val, cutlass.Float32(0)) * sWeights[n_idx]

            m_out = bidx * BM + tidx
            workspace[0, m_out] = out_val

            # 128-wide barrier — only compute group is alive here.
            cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=COMPUTE_THREADS)

            if warp_idx == 0:
                cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        if tidx == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)
            off = PROBE_HEADER + 0 * PROBE_ENTRY
            probe[probe_row, off + 3] = globaltimer_u64() - probe[probe_row, off + 2]
            range_finalize(probe, probe_row, probe_cnt)


# ── tvm-ffi compile ──────────────────────────────────────────────────────────
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)

def compile_kernel():
    NUM_PG = cute.sym_int(divisibility=2)
    NUM_PP = cute.sym_int()
    NUM_BL = cute.sym_int()

    kv_pool     = _fake(cute.Uint8,  (NUM_PP, PAGE_SIZE, ROW_STRIDE), (2, 1, 0), 16)
    block_table = _fake(cute.Int32,  (NUM_PG,),                       (0,),      4)
    q           = _fake(cute.Float8E4M3FN, (N, HEAD_DIM),             (1, 0),    16)
    w           = _fake(cute.Float32,(N,),                            (0,),      16)
    workspace   = _fake(cute.Float32,(WS_ROWS, WS_COLS),              (1, 0),    16)
    probe       = _fake(cute.Int64,  (NUM_BL, PROBE_COLS),            (1, 0),     8)
    stream      = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = ScoreScaleFullBTWSCpAsyncProfiled()
    compiled = cute.compile(
        kernel, kv_pool, block_table, q, w, workspace, probe, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


WORKLOAD_CASES = [
    ("WL 14 contig pg=34",          2161, list(range(3, 37))),
    ("WL 21 1-gap pg=35",           2177, list(range(3, 37)) + [38]),
    ("WL 25 2-gap pg=36",           2241, list(range(3, 37)) + [38, 42]),
    ("WL 64 backwards-jump pg=82",  5194, list(range(44, 65)) + [25, 18] + list(range(65, 95)) + [42, 33] + list(range(95, 122))),
    ("WL 70 long-tail pg=89",       5679, [7] + list(range(65, 153))),
]
NUM_PAGES_POOL = 11923


def run_single(workload_idx: int) -> str:
    label, seq_len, bt_list = WORKLOAD_CASES[workload_idx]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n── {label}  (seq_len={seq_len}) ──")
    print("Compiling cpasync intra kernel...")
    kernel, compiled = compile_kernel()
    workspace = kernel.workspace

    device = "cuda"
    torch.manual_seed(len(bt_list))
    num_pg_real = len(bt_list)
    num_pg = num_pg_real if num_pg_real % 2 == 0 else num_pg_real + 1
    bt_padded = bt_list + ([0] if num_pg != num_pg_real else [])
    M_real = num_pg_real * PAGE_SIZE
    grid_m = num_pg // PAGES_PER_TILE

    K_fp8_used    = torch.randn(num_pg_real, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
    K_scales_used = torch.rand(num_pg_real, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5

    kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
    for i, pid in enumerate(bt_list):
        kv_pool[pid, :, :HEAD_DIM] = K_fp8_used[i].view(torch.uint8)
        kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
            K_scales_used[i].view(torch.uint8).reshape(PAGE_SIZE, 4))

    block_table = torch.tensor(bt_padded, dtype=torch.int32, device=device)
    q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
    w     = torch.randn(N, device=device, dtype=torch.float32)
    probe = torch.zeros((grid_m, PROBE_COLS), dtype=torch.int64, device=device)

    for _ in range(3):
        probe.zero_()
        compiled(kv_pool, block_table, q_fp8, w, workspace, probe)
        torch.cuda.synchronize()

    K_ref  = K_fp8_used.reshape(M_real, HEAD_DIM).float()
    K_sc   = K_scales_used.reshape(M_real)
    scores = (K_ref @ q_fp8.float().T) * K_sc[:, None]
    ref    = torch.relu(scores) @ w
    c_view = workspace[0, :M_real]
    match  = torch.allclose(c_view, ref, atol=1.0, rtol=0.5)
    max_err = (c_view - ref).abs().max().item()
    print(f"  CORRECTNESS {'PASS' if match else 'FAIL'}  max_err={max_err:.4f}")

    probe.zero_()
    compiled(kv_pool, block_table, q_fp8, w, workspace, probe)
    torch.cuda.synchronize()

    return dump_probe(probe, num_blocks=grid_m)
