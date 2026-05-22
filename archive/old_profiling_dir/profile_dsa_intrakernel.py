"""
Intra-kernel profiling for the Fused_DSA kernel.
Uses globaltimer PTX to measure per-phase durations inside the kernel,
then dumps a Chrome trace JSON for visualization.

Usage: modal run zen/modal_profile_dsa_intrakernel.py
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm

from typing import Tuple
import math
import json
import torch

from zen.gather import gather_compiled

# ── Inline PTX helpers ──────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(
        MLIR_T.i64(), [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(
        MLIR_T.i32(), [],
        "mov.u32 $0, %smid;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


# ── Profiler constants ──────────────────────────────────────────────
PROBE_HEADER = 1   # first element = count of entries
PROBE_ENTRY  = 4   # (sm_id, tag, start_time, duration)

TAGS = {
    "score_nope":   0,
    "score_pe":     2,
    "softmax":      4,
    "output_load":  6,
    "output_gemm":  8,
    "epilogue":    10,
}
TAG_NAMES = {v: k for k, v in TAGS.items()}


# ── Profiler helpers (used inside kernel) ───────────────────────────

def range_start(probe, row, cnt, sm_val, tag_val):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()


def range_stop(probe, row, cnt):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64() - probe[row, off + 2]
    return cnt + cutlass.Int32(1)


def range_finalize(probe, row, cnt):
    probe[row, 0] = cutlass.Int64(cnt)


# ── Host-side dump ──────────────────────────────────────────────────

def dump_probe(probe: torch.Tensor, num_blocks: int,
               out_path: str = "dsa_intrakernel_trace.json"):
    probe_cpu = probe.cpu().contiguous().tolist()

    # Print per-block detail (first few blocks)
    for bid in range(min(num_blocks, 4)):
        data = probe_cpu[bid]
        cnt = int(data[0])
        print(f"\n--- Block {bid}: {cnt} entries ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id, tag = int(data[off]), int(data[off + 1])
            start, dur = int(data[off + 2]), int(data[off + 3])
            print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>15s}  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    # Aggregate: sum duration per tag across all blocks
    tag_totals = {}
    tag_counts = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1])
            dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*60}")
    print(f"{'Phase':>15s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>12s} {'%':>8s}")
    print(f"{'='*60}")
    grand_total = sum(tag_totals.values())
    for name in ["score_nope", "score_pe", "softmax", "output_load", "output_gemm", "epilogue"]:
        if name in tag_totals:
            total_ns = tag_totals[name]
            count = tag_counts[name]
            pct = 100.0 * total_ns / grand_total if grand_total > 0 else 0
            print(f"{name:>15s} {total_ns/1e6:>12.3f} {count:>6d} {total_ns/count/1000:>12.1f} {pct:>7.1f}%")
    print(f"{'TOTAL':>15s} {grand_total/1e6:>12.3f}")

    # Chrome trace JSON
    events, global_base = [], None
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (global_base is None or s < global_base):
                global_base = s
    global_base = global_base or 0

    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        if cnt == 0:
            continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1])
            start = int(data[off + 2])
            dur = int(data[off + 3])
            if start == 0 and dur == 0:
                continue
            events.append(dict(
                name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - global_base) / 1000.0,
                dur=dur / 1000.0,
                pid=sm_id, tid=bid))

    with open(out_path, "w") as f:
        json.dump({"traceEvents": events}, f)
    num_sms = len({e["pid"] for e in events})
    print(f"\nTrace: {len(events)} events from {num_sms} SMs -> {out_path}")
    print("Open with chrome://tracing or https://ui.perfetto.dev")


# ── Profiled Fused_DSA kernel ───────────────────────────────────────

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Fused_DSA_Profiled:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int, int, int] = (16, 64, 64, 64, 8)
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.Bdkc, self.Bdkp, self.Bdv = self.tile_shape_mnk
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (1, 4, 1)
        self.num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self.warp_size = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        kc: cute.Tensor,
        kp: cute.Tensor,
        sparse_indices: cute.Tensor,
        max_valid: cute.Tensor,
        sm_scale: cute.Tensor,
        output: cute.Tensor,
        lse: cute.Tensor,
        probe: cute.Tensor,
        stream,
    ):
        T, num_heads, dkc = q_nope.shape
        T, num_heads, dv = output.shape

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma_logits = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)

        # Launch with REDUCED grid: only 1 token (batch_idx=0), 1 head tile, 1 output tile
        self.kernel(
            q_nope, q_pe, kc, kp, sparse_indices, max_valid, sm_scale, output, lse,
            tiled_mma_logits, probe
        ).launch(
            grid=[1, 1, 1],
            block=(self.num_threads, 1, 1),
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        kc: cute.Tensor,
        kp: cute.Tensor,
        sparse_indices: cute.Tensor,
        max_valid: cute.Tensor,
        sm_scale: cute.Tensor,
        output: cute.Tensor,
        lse: cute.Tensor,
        tiled_mma_logits: cute.TiledMma,
        probe: cute.Tensor,
    ):
        T, topk = sparse_indices.shape
        _, num_heads, dkc = q_nope.shape
        _, _, dkp = q_pe.shape
        _, _, dv = output.shape

        bidx, bidy, batch_idx = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        sm = smid_u32()
        probe_row = cutlass.Int32(0)  # single block -> row 0
        probe_cnt = cutlass.Int32(0)

        # ===== Smem allocation ======
        allocator = cutlass.utils.SmemAllocator()

        sQn_layout = cute.make_layout((self.BM, self.Bdkc), stride=(self.Bdkc, 1))
        sK1_layout = cute.make_layout((self.BN, self.Bdkc), stride=(self.Bdkc, 1))
        sQn = allocator.allocate_tensor(cutlass.Float16, sQn_layout, 16, None)
        sK1 = allocator.allocate_tensor(cutlass.Float16, sK1_layout, 16, None)

        sL = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((self.BM, self.BN), stride=(self.BN, 1)), 16, None)
        sK2_layout = cute.make_layout((self.Bdv, self.BN), stride=(self.BN + 4, 1))
        sK2 = allocator.allocate_tensor(cutlass.Float16, sK2_layout, 4, None)
        sLSE = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((self.BM,), stride=(1,)), 4, None)
        sLSE.fill(0.0)

        # ============================== GMEM partitioning ===============================
        qn = q_nope[batch_idx, None, None]
        gQn_ = cute.zipped_divide(qn, (self.BM, self.Bdkc))
        gQn = gQn_[(None, None), (bidx, None)]
        kc_batch = kc[batch_idx, None, None]
        gKc1_ = cute.zipped_divide(kc_batch, (self.BN, self.Bdkc))

        qp = q_pe[batch_idx, None, None]
        gQp_ = cute.zipped_divide(qp, (self.BM, self.Bdkp))
        gQp = gQp_[(None, None), (bidx, None)]
        kp_batch = kp[batch_idx, None, None]
        gKp_ = cute.zipped_divide(kp_batch, (self.BN, self.Bdkp))

        gKc2__ = cute.zipped_divide(kc_batch, (self.BN, self.Bdv))
        gKc2_ = gKc2__[(None, None), (None, bidy)]

        # ============================== Logits MMA setup ===============================
        thr_mma = tiled_mma_logits.get_slice(tid)
        tCsA = thr_mma.partition_A(sQn)
        tCsB = thr_mma.partition_B(sK1)

        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            q_nope.element_type)
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            kc.element_type)

        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma_logits)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma_logits)

        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tid)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tid)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sQn)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sK1)

        acc_shape = thr_mma.partition_shape_C((self.BM, self.BN))

        accum_out = cutlass.Float32(0)
        local_max_valid = max_valid[batch_idx]
        num_BN_tiles = (local_max_valid + cutlass.Int32(self.BN - 1)) // cutlass.Int32(self.BN)

        for nidx in range(num_BN_tiles):
            # ── Step 1: Score nope (WMMA) ──
            gKc1 = gKc1_[(None, None), (nidx, None)]

            tCrA = tiled_mma_logits.make_fragment_A(tCsA)
            tCrB = tiled_mma_logits.make_fragment_B(tCsB)
            tCrC = tiled_mma_logits.make_fragment_C(acc_shape)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
            tCrC.fill(0.0)
            sL.fill(0.0)

            if tid == 0:
                range_start(probe, probe_row, probe_cnt, sm, TAGS["score_nope"])

            for kidx in range(dkc // self.Bdkc):
                cute.autovec_copy(gQn[None, None, kidx], sQn)
                cute.autovec_copy(gKc1[None, None, kidx], sK1)
                cute.arch.sync_threads()
                cute.copy(atom=tiled_copy_s2r_A, src=tCsA_copy_view, dst=tCrA_copy_view)
                cute.copy(atom=tiled_copy_s2r_B, src=tCsB_copy_view, dst=tCrB_copy_view)
                cute.gemm(atom=tiled_mma_logits, d=tCrC, a=tCrA, b=tCrB, c=tCrC)
                cute.arch.sync_threads()

            if tid == 0:
                probe_cnt = range_stop(probe, probe_row, probe_cnt)

            # ── Step 2: Score pe (WMMA) ──
            gKp = gKp_[(None, None), (nidx, None)]

            if tid == 0:
                range_start(probe, probe_row, probe_cnt, sm, TAGS["score_pe"])

            for kidx in range(dkp // self.Bdkp):
                cute.autovec_copy(gQp[None, None, kidx], sQn)
                cute.autovec_copy(gKp[None, None, kidx], sK1)
                cute.arch.sync_threads()
                cute.copy(atom=tiled_copy_s2r_A, src=tCsA_copy_view, dst=tCrA_copy_view)
                cute.copy(atom=tiled_copy_s2r_B, src=tCsB_copy_view, dst=tCrB_copy_view)
                cute.gemm(atom=tiled_mma_logits, d=tCrC, a=tCrA, b=tCrB, c=tCrC)
                cute.arch.sync_threads()

            if tid == 0:
                probe_cnt = range_stop(probe, probe_row, probe_cnt)

            # ── Step 3: Softmax ──
            if tid == 0:
                range_start(probe, probe_row, probe_cnt, sm, TAGS["softmax"])

            tv_layout_C = tiled_mma_logits.tv_layout_C_tiled
            sL_shape = cute.make_layout((self.BM, self.BN), stride=(self.BN, 1)).shape
            for reg_idx in range(cute.size(tCrC)):
                coord = cute.idx2crd((tid, reg_idx), tv_layout_C.shape)
                mn_flat = cute.crd2idx(coord, tv_layout_C)
                m, n = cute.idx2crd(mn_flat, sL_shape)
                global_n = nidx * self.BN + n
                if global_n < local_max_valid:
                    sL[m, n] = cute.math.exp(tCrC[reg_idx] * sm_scale[0])
            cute.arch.sync_threads()

            lane_idx = cute.arch.lane_idx()
            for row_idx in range(warp_idx, self.BM, self.num_threads // self.warp_size):
                local_sum = cutlass.Float32(0.0)
                for i in range(self.BN // self.warp_size):
                    local_sum += sL[row_idx, lane_idx + i * self.warp_size]
                total_sum = warp_reduce(local_sum, lambda a, b: a + b)
                if lane_idx == 0:
                    sLSE[row_idx] += total_sum

            if tid == 0:
                probe_cnt = range_stop(probe, probe_row, probe_cnt)

            # ── Step 4a: Output load (transpose K to smem) ──
            if tid == 0:
                range_start(probe, probe_row, probe_cnt, sm, TAGS["output_load"])

            gKc2 = gKc2_[None, None, nidx]
            num_loads_B = self.BN * self.Bdv
            for i in range(tid, num_loads_B, self.num_threads):
                k = i // self.Bdv
                n = i % self.Bdv
                if k < local_max_valid:
                    sK2[n, k] = gKc2[k, n]
            cute.arch.sync_threads()

            if tid == 0:
                probe_cnt = range_stop(probe, probe_row, probe_cnt)

            # ── Step 4b: Output gemm (scalar accumulation) ──
            if tid == 0:
                range_start(probe, probe_row, probe_cnt, sm, TAGS["output_gemm"])

            tidx = tid % self.Bdv
            tidy = tid // self.Bdv

            for mmak in range(self.BN):
                if mmak < local_max_valid:
                    accum_out += sL[tidy, mmak] * cutlass.Float32(sK2[tidx, mmak])

            if tid == 0:
                probe_cnt = range_stop(probe, probe_row, probe_cnt)

        # ── Epilogue ──
        if tid == 0:
            range_start(probe, probe_row, probe_cnt, sm, TAGS["epilogue"])

        if bidy == 0:
            if tid < self.BM:
                lse[batch_idx, tid] = cute.math.log(sLSE[tid]) / cutlass.Float32(0.6931471805599453)

        tidx = tid % self.Bdv
        tidy = tid // self.Bdv

        gOut_ = cute.zipped_divide(output[batch_idx, None, None], (self.BM, self.Bdv))
        gOut = gOut_[(None, None), (bidx, bidy)]
        gOut[tidy, tidx] = cutlass.BFloat16(accum_out / sLSE[tidy])

        if tid == 0:
            probe_cnt = range_stop(probe, probe_row, probe_cnt)
            range_finalize(probe, probe_row, probe_cnt)


# ── Compilation ─────────────────────────────────────────────────────

def fake_wrapper(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)


def compile_profiled_dsa():
    T = cute.sym_int()
    num_heads, dkc, dkp, topk = 16, 512, 64, 2048

    q_nope = fake_wrapper(cute.BFloat16, (T, num_heads, dkc), (2, 1, 0), 16)
    q_pe = fake_wrapper(cute.BFloat16, (T, num_heads, dkp), (2, 1, 0), 16)
    kc = fake_wrapper(cute.BFloat16, (T, topk, dkc), (2, 1, 0), 16)
    kp = fake_wrapper(cute.BFloat16, (T, topk, dkp), (2, 1, 0), 16)
    sparse_indices = fake_wrapper(cute.Int32, (T, topk), (1, 0), 4)
    max_valid = fake_wrapper(cute.Int32, (T,), (0,), 4)
    sm_scale = fake_wrapper(cute.Float32, (1,), (0,), 4)
    output = fake_wrapper(cute.BFloat16, (T, num_heads, dkc), (2, 1, 0), 16)
    lse = fake_wrapper(cute.Float32, (T, num_heads), (1, 0), 4)

    # Probe tensor: 1 block, generous room for entries
    # Per tile: 5 phases (score_nope, score_pe, softmax, output_load, output_gemm) + 1 epilogue = 6
    # topk=2048, BN=64 => 32 tiles => 32*5 + 1 = 161 entries max
    max_entries = 200
    probe_cols = PROBE_HEADER + max_entries * PROBE_ENTRY
    probe = fake_wrapper(cute.Int64, (1, probe_cols), (1, 0), 8)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    fused_dsa = Fused_DSA_Profiled()
    return cute.compile(
        fused_dsa,
        q_nope, q_pe, kc, kp, sparse_indices, max_valid, sm_scale, output, lse,
        probe, stream,
        options="--enable-tvm-ffi"
    )


# ── Main: run locally or on Modal ──────────────────────────────────

def run_profiling():
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/zen")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    T = 1   # single token for profiling
    P = 512
    H = 16
    D_ckv = 512
    D_kpe = 64
    PAGE_SIZE = 64
    TOPK = 2048
    sm_scale_val = 1.0 / math.sqrt(D_ckv + D_kpe)

    q_nope = torch.randn(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn(T, H, D_kpe, dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn(P, PAGE_SIZE, D_ckv, dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn(P, PAGE_SIZE, D_kpe, dtype=torch.bfloat16, device="cuda")
    total_kv = P * PAGE_SIZE
    sparse_indices = torch.randint(0, total_kv, (T, TOPK), dtype=torch.int32, device="cuda")

    # Pre-gather
    Kc_all = ckv_cache.reshape(-1, D_ckv)
    Kp_all = kpe_cache.reshape(-1, D_kpe)
    Kc = torch.empty(T, TOPK, D_ckv, dtype=torch.bfloat16, device="cuda")
    Kp = torch.empty(T, TOPK, D_kpe, dtype=torch.bfloat16, device="cuda")
    max_valid_t = torch.empty(T, dtype=torch.int32, device="cuda")
    gather_compiled(Kc_all, Kp_all, sparse_indices, Kc, Kp, max_valid_t)
    torch.cuda.synchronize()

    output = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
    sm_scale_tensor = torch.tensor([sm_scale_val], dtype=torch.float32, device="cuda")

    # Probe: 1 block only
    max_entries = 200
    probe_cols = PROBE_HEADER + max_entries * PROBE_ENTRY
    probe = torch.zeros((1, probe_cols), dtype=torch.int64, device="cuda")

    print("Compiling profiled DSA kernel...")
    compiled = compile_profiled_dsa()

    # Warmup
    for _ in range(3):
        output.zero_()
        lse.fill_(-float("inf"))
        probe.zero_()
        compiled(q_nope, q_pe, Kc, Kp, sparse_indices, max_valid_t, sm_scale_tensor, output, lse, probe)
        torch.cuda.synchronize()

    # Profiled run
    probe.zero_()
    output.zero_()
    lse.fill_(-float("inf"))
    compiled(q_nope, q_pe, Kc, Kp, sparse_indices, max_valid_t, sm_scale_tensor, output, lse, probe)
    torch.cuda.synchronize()

    dump_probe(probe, num_blocks=1, out_path="/tmp/dsa_intrakernel_trace.json")

    return probe.cpu(), open("/tmp/dsa_intrakernel_trace.json").read()
