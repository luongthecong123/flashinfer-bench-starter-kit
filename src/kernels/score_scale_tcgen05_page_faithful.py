"""score_scale_tcgen05_page_faithful.py — tcgen05 fp8 GEMM with paged KV cache + per-token scale.

Faithful paged interface with block_table indirection (pages are NOT contiguous).

  kv_pool     : (num_pool_pages, PAGE_SIZE, 1, ROW_STRIDE) int8  — global page pool
  block_table : (num_tiles * PAGES_PER_TILE,) int32              — page-ID indirection
  q_fp8       : (N, HEAD_DIM) Float8E4M3FN
  c_out       : (M,) Float32

Page layout (flat, PAGE_BYTES = 8448 bytes per page):
  first FP8_REGION=8192 bytes : fp8 data  — token t, dim d = flat byte t*HEAD_DIM+d
  last  256 bytes             : float32 scales — token t scale at FP8_REGION + t*4

The page-jump stride between the two pages is computed at runtime from
block_table[bidx*2] and block_table[bidx*2+1], allowing non-contiguous pages.

C[m, n] = (sum_k fp8_A[m,k] * fp8_B[n,k]) * scale[m]
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.testing import benchmark, JitArguments

# ── Dimensions ────────────────────────────────────────────────────────────────
M              = 2048
N              = 64
HEAD_DIM       = 128
PAGE_SIZE      = 64
ROW_STRIDE     = HEAD_DIM + 4            # 132 bytes/token
PAGE_BYTES     = PAGE_SIZE * ROW_STRIDE  # 8448
FP8_REGION     = PAGE_SIZE * HEAD_DIM   # 8192 — fp8 bytes per page
PAGES_PER_TILE = 2
BM             = PAGE_SIZE * PAGES_PER_TILE  # 128

MMA_INST_MNK    = (128, N, 32)
CTA_TILE_MNK    = (BM, N, HEAD_DIM)

THREADS_PER_CTA = 128


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


class ScoreScaleTcgen05Page:
    """
    tcgen05 fp8 GEMM with non-contiguous paged KV cache via block_table.

    Each CTA tile handles PAGES_PER_TILE=2 pages whose indices are read from
    block_table[bidx*2] and block_table[bidx*2+1].  The page-jump stride is
    computed at runtime so a single autovec_copy covers both arbitrary pages.
    """

    def __init__(self):
        self.num_stages  = 1
        self.tmem_ld_rep = N   # load all N columns in one shot

    @cute.jit
    def __call__(
        self,
        kv_pool:     cute.Tensor,  # (num_pool_pages, PAGE_SIZE, 1, ROW_STRIDE) int8
        block_table: cute.Tensor,  # (num_tiles * PAGES_PER_TILE,) int32
        q:           cute.Tensor,  # (N, HEAD_DIM) Float8E4M3FN
        weights:     cute.Tensor,  # (N,) Float32  — per-head reduction weights
        c_out:       cute.Tensor,  # (M,) Float32  — flat output
    ):
        self.ab_dtype  = cutlass.Float8E4M3FN
        self.c_dtype   = c_out.element_type
        self.acc_dtype = cutlass.Float32

        op = tcgen05.MmaFP8Op(
            self.ab_dtype,
            self.acc_dtype,
            MMA_INST_MNK,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)

        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, self.ab_dtype, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, q.element_type, self.num_stages,
        )

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
            weights_smem:     cute.struct.MemRange[cutlass.Float32, N]

        self.shared_storage = SharedStorage

        num_tiles = (block_table.shape[0] + PAGES_PER_TILE - 1) // PAGES_PER_TILE

        self.kernel(
            self.tiled_mma,
            kv_pool,
            block_table,
            q,
            weights,
            c_out,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=(num_tiles, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        kv_pool:       cute.Tensor,          # (num_pool_pages, PAGE_SIZE, 1, ROW_STRIDE) int8
        block_table:   cute.Tensor,          # (num_tiles * PAGES_PER_TILE,) int32
        mB:            cute.Tensor,          # (N, HEAD_DIM) Float8E4M3FN
        mWeights:      cute.Tensor,          # (N,) Float32
        mC:            cute.Tensor,          # (M,) Float32  — flat output
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _  = cute.arch.block_idx()

        # ── Block-table indirection ──────────────────────────────────
        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])

        # ── SMEM allocation ──────────────────────────────────────────
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ── sWeights: load N weights to smem for thread broadcast ────
        # First N threads each load one weight; sync_threads() below
        # covers this load before the MMA + epilogue read.
        sWeights_ptr = cute.make_ptr(
            cutlass.Float32,
            storage.weights_smem.data_ptr().toint(),
            mem_space=cute.AddressSpace.smem,
            assumed_align=4,
        )
        sWeights = cute.make_tensor(sWeights_ptr, cute.make_layout((N,), stride=(1,)))
        mW = cute.make_tensor(mWeights.iterator, cute.make_layout((N,), stride=(1,)))
        if tidx < cutlass.Int32(N):
            sWeights[tidx] = mW[tidx]

        # ── MMA fragments ────────────────────────────────────────────
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCrA    = tiled_mma.make_fragment_A(sA)
        tCrB    = tiled_mma.make_fragment_B(sB)

        acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc          = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        # ── TMEM alloc + mbarrier init ───────────────────────────────
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        tmem_barrier_id = 1
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # ── autovec G→S for sA: runtime page-jump stride ─────────────
        # page0_id and page1_id come from block_table (non-contiguous pages).
        # jump_bytes = (page1_id - page0_id) * PAGE_BYTES is a runtime value;
        # CUTE layout supports runtime strides so we embed it as the outer stride
        # of the page dimension in gA_paged.
        thr_layout = cute.make_layout(THREADS_PER_CTA)

        fp8_base   = cute.recast_ptr(kv_pool.iterator, dtype=self.ab_dtype)
        page0_off  = page0_id * cutlass.Int32(PAGE_BYTES)
        jump_bytes = (page1_id - page0_id) * cutlass.Int32(PAGE_BYTES)

        fp8_ptr = cute.make_ptr(
            cutlass.Float8E4M3FN,
            (fp8_base + page0_off).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=1,
        )
        # Layout: ((PAGE_SIZE=64, PAGES_PER_TILE=2), HEAD_DIM=128)
        #   inner row stride = HEAD_DIM (flat: fp8 bytes packed, no scale gaps between tokens)
        #   page stride      = jump_bytes (runtime jump to next page's fp8 base)
        #   dim stride       = 1
        gA_paged = cute.make_tensor(
            fp8_ptr,
            cute.make_layout(
                ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
                stride=((HEAD_DIM, jump_bytes), 1),
            ),
        )
        tCgA   = thr_mma.partition_A(gA_paged)
        sA_thr = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr = cute.local_partition(tCgA, thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        # ── autovec G→S for sB ───────────────────────────────────────
        gB_2d  = cute.make_tensor(mB.iterator,
                                  cute.make_layout((N, HEAD_DIM), stride=(HEAD_DIM, 1)))
        tCgB   = thr_mma.partition_B(gB_2d)
        sB_thr = cute.local_partition(sB[None, None, None, 0], thr_layout, tidx)
        gB_thr = cute.local_partition(tCgB, thr_layout, tidx)
        cute.autovec_copy(gB_thr, sB_thr)

        # autovec_copy issues cp.async — must commit + wait before MMA
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # ── Epilogue setup: TMEM → RMEM ──────────────────────────────
        M_acc          = cute.size(tCtAcc, mode=[0, 0])
        ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler      = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)

        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # ── tcgen05 MMA (warp 0) ─────────────────────────────────────
        tcgen05_fence()

        mma_phase    = 0
        num_k_blocks = cute.size(tCrA, mode=[2])
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        if warp_idx == 0:
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(tiled_mma, tCtAcc,
                          tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if tidx == 0:
                tcgen05.commit(mma_mbar)

        cute.arch.mbarrier_wait(mma_mbar, mma_phase)
        mma_phase ^= 1

        # ── Epilogue: TMEM → RMEM → GMEM with per-token scale ────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        # Each thread owns one token row (tidx) within the 128-token CTA tile.
        # Load per-token scale via block_table → flat layout (scales after fp8 region).
        page_sel      = tidx // cutlass.Int32(PAGE_SIZE)         # 0 or 1
        token_in_page = tidx - page_sel * cutlass.Int32(PAGE_SIZE)
        page_id_t     = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + page_sel])
        scale_f32_off = (page_id_t * cutlass.Int32(PAGE_BYTES // 4)
                         + cutlass.Int32(FP8_REGION // 4)
                         + token_in_page)
        fp32_base  = cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Float32)
        scale_ptr  = cute.make_ptr(
            cutlass.Float32,
            (fp32_base + scale_f32_off).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=1,
        )
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,), stride=(1,)))[0]

        # Weighted reduction: out[m] = scale[m] * sum_n(acc[n] * weight[n])
        acc = cutlass.Float32(0.0)
        for n_idx in cutlass.range_constexpr(N):
            acc = acc + tTR_rAcc[n_idx] * sWeights[n_idx]

        m_out = bidx * cutlass.Int32(BM) + tidx
        mC[m_out] = (acc * scale).to(self.c_dtype)

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Python wrapper ────────────────────────────────────────────────────────────
def run_gemm(
    kv_pool:     torch.Tensor,  # (num_pool_pages, PAGE_SIZE, 1, ROW_STRIDE) int8
    block_table: torch.Tensor,  # (num_tiles * PAGES_PER_TILE,) int32
    q_fp8:       torch.Tensor,  # (N, HEAD_DIM) Float8E4M3FN
    weights:     torch.Tensor,  # (N,) Float32
    c_out:       torch.Tensor,  # (M,) Float32
):
    c_out.zero_()
    kv_ = from_dlpack(kv_pool,     assumed_align=16)
    bt_ = from_dlpack(block_table, assumed_align=4)
    q_  = from_dlpack(q_fp8,       assumed_align=16)
    w_  = from_dlpack(weights,     assumed_align=4)
    c_  = from_dlpack(c_out,       assumed_align=16)
    gemm     = ScoreScaleTcgen05Page()
    compiled = cute.compile(gemm, kv_, bt_, q_, w_, c_)
    compiled(kv_, bt_, q_, w_, c_)


# ── Reference dequant via block_table ───────────────────────────────────────
def dequant_flat_with_bt(kv_pool: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """Extract fp8 + scale via block_table indirection (flat layout)."""
    k_u8   = kv_pool.view(torch.uint8)                            # (num_pool, 64, 1, 132)
    k_flat = k_u8.reshape(k_u8.shape[0], PAGE_BYTES)              # (num_pool, 8448)
    pages  = block_table.cpu().tolist()
    fp8_rows, scale_rows = [], []
    for pid in pages:
        page_flat   = k_flat[int(pid)]
        fp8_rows.append(page_flat[:FP8_REGION].view(torch.float8_e4m3fn).reshape(PAGE_SIZE, HEAD_DIM))
        scale_rows.append(page_flat[FP8_REGION:].view(torch.float32))   # (64,)
    fp8_tensor = torch.stack(fp8_rows, dim=0).to(torch.float32)   # (num_pages, 64, 128)
    scales     = torch.stack(scale_rows, dim=0)                    # (num_pages, 64)
    return fp8_tensor * scales.unsqueeze(-1)                       # (num_pages, 64, 128)


# ── helpers ──────────────────────────────────────────────────────────────────
def _build_k_cache(pool_size: int, device: str, seed: int = 42) -> torch.Tensor:
    """Synthetic (pool_size, PAGE_SIZE, 1, ROW_STRIDE) int8 with flat fp8 + f32 scales."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    # Flat layout: (pool_size, PAGE_BYTES) — first FP8_REGION bytes fp8, then 256 bytes scales
    k_u8   = torch.zeros(pool_size, PAGE_BYTES, dtype=torch.uint8, device=device)
    fp8_vals = (torch.randn(pool_size * PAGE_SIZE * HEAD_DIM, device=device, generator=gen)
                .clamp(-4, 4).to(torch.float8_e4m3fn))
    k_u8[:, :FP8_REGION].copy_(fp8_vals.view(torch.uint8).reshape(pool_size, FP8_REGION))
    scales_f32 = torch.rand(pool_size * PAGE_SIZE, device=device, generator=gen) * 0.1 + 0.01
    k_u8[:, FP8_REGION:].copy_(scales_f32.view(torch.uint8).reshape(pool_size, PAGE_SIZE * 4))
    return k_u8.reshape(pool_size, PAGE_SIZE, 1, ROW_STRIDE).view(torch.int8)


def _test_one(k_cache, flat_bt_full, seq_len, q_fp8, weights, c_out, label):
    """Run kernel + reference on one (k_cache, flat_bt, seq_len) triple.
    flat_bt_full already padded to even length."""
    import math
    run_gemm(k_cache, flat_bt_full, q_fp8, weights, c_out)
    torch.cuda.synchronize()

    num_pages_valid = math.ceil(seq_len / PAGE_SIZE)
    K_scaled = dequant_flat_with_bt(k_cache, flat_bt_full[:num_pages_valid])
    K_flat   = K_scaled.reshape(num_pages_valid * PAGE_SIZE, HEAD_DIM).to(k_cache.device)
    ref_c    = (K_flat @ q_fp8.float().T * weights).sum(dim=1)

    out_v  = c_out[:seq_len]
    ref_v  = ref_c[:seq_len]
    diff   = (out_v - ref_v).abs()
    mean_e = diff.mean().item()
    rel_e  = (diff / (ref_v.abs() + 1e-6)).mean().item()
    max_e  = diff.max().item()
    flag   = "  <<< HIGH ERROR" if (mean_e == mean_e and mean_e > 0.01) else ""
    print(f"  {label}  seq={seq_len:5d}  mean={mean_e:.6f}  rel={rel_e:.4f}  max={max_e:.4f}{flag}")
    return mean_e, rel_e


# ── Main: repeat each long-seq case 10× to classify race vs deterministic ────
def main():
    import json
    from pathlib import Path

    device = "cuda"

    JSONL_PATH = Path("/data/workloads/dsa_paged/"
                      "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl")

    if not JSONL_PATH.exists():
        print("[Synthetic fallback — no /data available]")
        pool_size = 65
        num_pages = 33
        seq_len   = num_pages * PAGE_SIZE
        flat_bt   = torch.randperm(pool_size, device=device)[:num_pages].to(torch.int32)
        flat_bt   = torch.cat([flat_bt, flat_bt.new_zeros(1)])
        q_fp8     = torch.randn(N, HEAD_DIM, device=device).clamp(-4, 4).to(torch.float8_e4m3fn)
        weights   = torch.randn(N, device=device)
        k_cache   = _build_k_cache(pool_size, device)
        c_out     = torch.zeros(320000, device=device)
        _test_one(k_cache, flat_bt, seq_len, q_fp8, weights, c_out, "synth")
        return

    from safetensors.torch import load_file as st_load

    all_workloads = [json.loads(l) for l in open(JSONL_PATH)]
    print(f"Loaded {len(all_workloads)} workloads.  Scanning for seq_len > 2048 ...")

    SEED    = 42
    REPEATS = 10
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(SEED)
    q_fp8   = torch.randn(N, HEAD_DIM, generator=gen_cpu).clamp(-4, 4).to(torch.float8_e4m3fn).cuda()
    weights = torch.randn(N, generator=gen_cpu).cuda()
    c_out   = torch.zeros(320000, device=device)

    # Results: label -> list of mean_err across REPEATS runs
    results = {}   # label -> [mean_e, ...]

    for wi, w in enumerate(all_workloads):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        sf       = st_load(str(Path("/data") / inp["seq_lens"]["path"]))
        seq_lens = sf[inp["seq_lens"]["tensor_key"]].cuda()
        bt_full  = sf[inp["block_table"]["tensor_key"]].cuda()
        long_idx = (seq_lens > 2048).nonzero(as_tuple=True)[0]
        if long_idx.numel() == 0:
            continue
        long_idx = long_idx[seq_lens[long_idx].argsort(descending=True)]

        pool_size = ax["num_pages"]
        k_cache   = _build_k_cache(pool_size, device, seed=SEED)

        for b in long_idx:
            b       = int(b.item())
            seq_len = int(seq_lens[b].item())
            flat_bt = bt_full[b, :].contiguous().to(torch.int32)
            if flat_bt.shape[0] % PAGES_PER_TILE != 0:
                flat_bt = torch.cat([flat_bt, flat_bt.new_zeros(1)])

            label = f"W{wi+1:03d}/b{b:02d}(seq={seq_len})"
            errs  = []
            for _ in range(REPEATS):
                me, _ = _test_one(k_cache, flat_bt, seq_len, q_fp8, weights, c_out, label)
                errs.append(me)
            results[label] = errs

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY — {REPEATS} repeats per case, HIGH ERROR threshold = mean > 0.01")
    print(f"{'='*70}")
    print(f"{'Label':<30}  {'fails':>5}  {'min':>10}  {'max':>10}  category")
    print(f"{'-'*70}")

    categories = {"clean": 0, "race": 0, "deterministic": 0}
    for label, errs in results.items():
        valid = [e for e in errs if e == e]  # drop NaN
        if not valid:
            continue
        n_high = sum(1 for e in valid if e > 0.01)
        mn, mx = min(valid), max(valid)
        if n_high == 0:
            cat = "CLEAN"
            categories["clean"] += 1
        elif n_high == len(valid):
            cat = "DETERMINISTIC BUG"
            categories["deterministic"] += 1
        else:
            cat = f"RACE ({n_high}/{len(valid)} fail)"
            categories["race"] += 1
        print(f"  {label:<28}  {n_high:>5}  {mn:>10.6f}  {mx:>10.6f}  {cat}")

    print(f"\n  clean={categories['clean']}  race={categories['race']}  deterministic={categories['deterministic']}")

    # Timing
    kc_ = from_dlpack(k_cache,  assumed_align=16)
    bt_ = from_dlpack(flat_bt,  assumed_align=4)
    q_  = from_dlpack(q_fp8,    assumed_align=16)
    w_  = from_dlpack(weights,  assumed_align=4)
    c_  = from_dlpack(c_out,    assumed_align=16)
    gemm     = ScoreScaleTcgen05Page()
    compiled = cute.compile(gemm, kc_, bt_, q_, w_, c_)
    t = benchmark(compiled, kernel_arguments=JitArguments(kc_, bt_, q_, w_, c_))
    print(f"\nDURATION (last case): {t:.4f} µs")


if __name__ == "__main__":
    main()

