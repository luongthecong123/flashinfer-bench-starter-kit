"""Intra-phase profiling: TMA S2G reduce vs thread-level reduction.

Two approaches timed with globaltimer probes:
  A) Thread-level reduction: 1024 threads each reduce 32 smem values → 1 gmem value
  B) TMA S2G reduce: warp 0 issues 32 bulk tensor reductions via TMA unit

Setup:
  smem_partial:  (NUM_WARPS=32, HEAD_DIM=512) f32  — each warp row = warp_idx + 1
  gmem expected: output[i] = sum(w+1 for w=0..31) = 32*33/2 = 528.0  for all i

Grid: [1, 1, 1] — single block for each approach.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
import json, torch


# ── Timer helpers ─────────────────────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(MLIR_T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int64(t)

@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(MLIR_T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int32(t)


# ── Probe layout ──────────────────────────────────────────────────────────────

PROBE_HEADER = 1
PROBE_ENTRY  = 4          # sm_id, tag, start_ns, duration_ns
MAX_ENTRIES  = 4           # fill, reduce (plenty of headroom)
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY   # 17
NUM_BLOCKS   = 2           # block 0 = thread-level, block 1 = TMA

TAGS       = {"fill": 0, "reduce": 2}
TAG_NAMES  = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["fill", "reduce"]


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


# ── Constants ─────────────────────────────────────────────────────────────────

HEAD_DIM = 512
NUM_WARPS = 32
NUM_THREADS = NUM_WARPS * 32  # 1024


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Approach A: Thread-level reduction
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def thread_reduce_host(output: cute.Tensor, probe: cute.Tensor, stream):
    head_dim: cutlass.Constexpr = HEAD_DIM
    num_warps: cutlass.Constexpr = NUM_WARPS

    thread_reduce_kernel(output, probe).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)


@cute.kernel
def thread_reduce_kernel(output: cute.Tensor, probe: cute.Tensor):
    head_dim:  cutlass.Constexpr = HEAD_DIM
    num_warps: cutlass.Constexpr = NUM_WARPS
    num_threads: cutlass.Constexpr = NUM_THREADS

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    alloc = cutlass.utils.SmemAllocator()
    smem_partial = _smem(alloc, cutlass.Float32, (num_warps, head_dim), (head_dim, 1), 128)

    probe_row = cutlass.Int32(0)  # block 0 = thread-level
    sm = cutlass.Int64(smid_u32())
    probe_cnt = cutlass.Int32(0)

    # ── Fill ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["fill"])

    for i in range(lane_idx, head_dim, 32):
        smem_partial[warp_idx, i] = cutlass.Float32(warp_idx + 1)
    cute.arch.sync_threads()

    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # ── Reduce ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["reduce"])

    for i in range(tidx, head_dim, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        output[0, i] = acc
    cute.arch.sync_threads()

    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_finalize(probe, probe_row, probe_cnt)


# ═══════════════════════════════════════════════════════════════════════════════
# Approach B: TMA S2G reduce
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def tma_reduce_host(output: cute.Tensor, probe: cute.Tensor, stream):
    head_dim:   cutlass.Constexpr = HEAD_DIM
    num_warps:  cutlass.Constexpr = NUM_WARPS

    tma_op = cpasync.CopyReduceBulkTensorTileS2GOp(
        reduction_kind=cute.ReductionOp.ADD
    )
    smem_tile_layout = cute.make_layout((head_dim,), stride=(1,))
    gmem_row = cute.make_tensor(
        output.iterator,
        cute.make_layout((head_dim,), stride=(1,))
    )
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        tma_op, gmem_row, smem_tile_layout, (head_dim,)
    )

    tma_reduce_kernel(output, probe, tma_atom, tma_tensor).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)


@cute.kernel
def tma_reduce_kernel(
    output:     cute.Tensor,
    probe:      cute.Tensor,
    tma_atom:   cute.CopyAtom,
    tma_tensor: cute.Tensor,
):
    head_dim:   cutlass.Constexpr = HEAD_DIM
    num_warps:  cutlass.Constexpr = NUM_WARPS
    num_threads: cutlass.Constexpr = NUM_THREADS

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    alloc = cutlass.utils.SmemAllocator()
    smem_data = _smem(alloc, cutlass.Float32, (num_warps, head_dim), (head_dim, 1), 128)
    smem_tile = _smem(alloc, cutlass.Float32, (head_dim,), (1,), 128)

    probe_row = cutlass.Int32(1)  # block 1 = TMA
    sm = cutlass.Int64(smid_u32())
    probe_cnt = cutlass.Int32(0)

    # ── Fill ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["fill"])

    for i in range(lane_idx, head_dim, 32):
        smem_data[warp_idx, i] = cutlass.Float32(warp_idx + 1)
    cute.arch.sync_threads()

    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)

    # TMA partition
    tS, tG = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1), smem_tile, tma_tensor,
    )

    # ── Reduce (TMA S2G) ──
    if tidx == 0:
        range_start(probe, probe_row, probe_cnt, sm, TAGS["reduce"])

    for w in range(num_warps):
        for i in range(tidx, head_dim, num_threads):
            smem_tile[i] = smem_data[w, i]
        cute.arch.sync_threads()

        cute.arch.fence_proxy("async.shared", space="cta")

        if warp_idx == 0:
            cute.copy(tma_atom, tS[None], tG[None])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)

        cute.arch.sync_threads()

    if tidx == 0:
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_finalize(probe, probe_row, probe_cnt)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_both():
    output_fake = _fake(cute.Float32, (1, HEAD_DIM), (1, 0), 16)
    probe_fake  = _fake(cute.Int64, (NUM_BLOCKS, PROBE_COLS), (1, 0), 8)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_thread = cute.compile(
        thread_reduce_host, output_fake, probe_fake, stream, options="--enable-tvm-ffi")
    compiled_tma = cute.compile(
        tma_reduce_host, output_fake, probe_fake, stream, options="--enable-tvm-ffi")
    return compiled_thread, compiled_tma


compiled_thread, compiled_tma = compile_both()
EXPECTED_SUM = float(NUM_WARPS * (NUM_WARPS + 1) // 2)  # 528.0


# ═══════════════════════════════════════════════════════════════════════════════
# Probe dump
# ═══════════════════════════════════════════════════════════════════════════════

def _probe_events(probe_cpu, num_blocks):
    events = []
    base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (base is None or s < base):
                base = s
    base = base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        if cnt == 0:
            continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); t0 = int(data[off + 2]); dur = int(data[off + 3])
            if t0 == 0 and dur == 0:
                continue
            events.append(dict(
                name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0 - base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id, tid=bid))
    return events, base


APPROACH_NAMES = {0: "Thread-level", 1: "TMA S2G reduce"}


def dump_probe(probe: torch.Tensor):
    probe_cpu = probe.cpu().contiguous().tolist()

    print(f"\n{'='*60}")
    for bid in range(NUM_BLOCKS):
        data = probe_cpu[bid]; cnt = int(data[0])
        name = APPROACH_NAMES.get(bid, f"block_{bid}")
        print(f"\n--- {name} ({cnt} entries) ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
            phase = TAG_NAMES.get(tag, f"tag_{tag}")
            print(f"  sm={sm_id:>3} {phase:>10s}  dur={dur:>8} ns  ({dur/1000:.1f} µs)")

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"{'Approach':>20s}  {'fill (µs)':>10s}  {'reduce (µs)':>12s}")
    print(f"{'-'*60}")
    for bid in range(NUM_BLOCKS):
        data = probe_cpu[bid]; cnt = int(data[0])
        name = APPROACH_NAMES.get(bid, f"block_{bid}")
        phases = {}
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            phase = TAG_NAMES.get(tag, f"tag_{tag}")
            phases[phase] = dur / 1000.0
        print(f"{name:>20s}  {phases.get('fill', 0):>10.1f}  {phases.get('reduce', 0):>12.1f}")
    print(f"{'='*60}")

    return _probe_events(probe_cpu, NUM_BLOCKS)


# ═══════════════════════════════════════════════════════════════════════════════
# run_single — called by modal runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(workload_idx: int = 0) -> str:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Expected sum per element: {EXPECTED_SUM}")

    probe = torch.zeros((NUM_BLOCKS, PROBE_COLS), dtype=torch.int64, device="cuda")

    # ── Approach A: thread-level ──
    output_a = torch.zeros(1, HEAD_DIM, dtype=torch.float32, device="cuda")
    for _ in range(3):
        probe.zero_(); output_a.zero_()
        compiled_thread(output_a, probe)
        torch.cuda.synchronize()

    probe.zero_(); output_a.zero_()
    compiled_thread(output_a, probe)
    torch.cuda.synchronize()

    max_err_a = (output_a - EXPECTED_SUM).abs().max().item()
    print(f"\n[Thread-level] max_err={max_err_a:.6f}  {'PASS' if max_err_a < 1e-3 else 'FAIL'}")
    print(f"  output[0:8] = {output_a[0, :8].tolist()}")

    # Save probe row 0
    probe_a = probe[0].clone()

    # ── Approach B: TMA S2G reduce ──
    output_b = torch.zeros(1, HEAD_DIM, dtype=torch.float32, device="cuda")
    for _ in range(3):
        probe.zero_(); output_b.zero_()
        compiled_tma(output_b, probe)
        torch.cuda.synchronize()

    probe.zero_(); output_b.zero_()
    compiled_tma(output_b, probe)
    torch.cuda.synchronize()

    max_err_b = (output_b - EXPECTED_SUM).abs().max().item()
    print(f"\n[TMA S2G reduce] max_err={max_err_b:.6f}  {'PASS' if max_err_b < 1e-3 else 'FAIL'}")
    print(f"  output[0:8] = {output_b[0, :8].tolist()}")

    # Merge both probe rows into one tensor for dump
    probe_merged = torch.zeros((NUM_BLOCKS, PROBE_COLS), dtype=torch.int64, device="cpu")
    probe_merged[0] = probe_a.cpu()
    probe_merged[1] = probe[1].cpu()
    # Put back on "cuda" for dump_probe
    probe_merged = probe_merged.cuda()

    events, base = dump_probe(probe_merged)
    return json.dumps({"traceEvents": events})
