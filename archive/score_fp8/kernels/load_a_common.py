"""Common runner for load-A microbenchmarks (v1/v2/v3).

Each impl module must export:
  - compile_kernel()       -> (kernel_obj, compiled_callable)
  - TAGS, TAG_NAMES, PHASE_ORDER (for probe dump)
  - PROBE_HEADER, PROBE_ENTRY, PROBE_COLS  (re-exported from this module)
  - NUM_THREADS            (for printing)
"""
import json, torch
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm

# ── Probe primitives (shared) ────────────────────────────────────────────────
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

PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 8
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

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


# ── Workload constants ──────────────────────────────────────────────────────
PAGE_SIZE  = 64
N          = 64
HEAD_DIM   = 128
ROW_STRIDE = 132
PAGES_PER_TILE = 2
BM, BN, BK = 128, N, HEAD_DIM
NUM_PAGES_POOL = 11923

WORKLOAD_CASES = [
    ("WL 14 contig pg=34",          2161, list(range(3, 37))),
    ("WL 21 1-gap pg=35",           2177, list(range(3, 37)) + [38]),
    ("WL 25 2-gap pg=36",           2241, list(range(3, 37)) + [38, 42]),
    ("WL 64 backwards-jump pg=82",  5194,
        list(range(44, 65)) + [25, 18] + list(range(65, 95)) + [42, 33] + list(range(95, 122))),
    ("WL 70 long-tail pg=89",       5679, [7] + list(range(65, 153))),
]


def dump_probe(probe: torch.Tensor, num_blocks: int,
               TAGS: dict, TAG_NAMES: dict, PHASE_ORDER: list) -> str:
    probe_cpu = probe.cpu().contiguous().tolist()
    max_dur, max_bid = -1, 0
    total_tag = TAGS.get("total", 0)
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            if int(data[off + 1]) == total_tag:
                bt = int(data[off + 3])
                if bt > max_dur:
                    max_dur, max_bid = bt, bid
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
            tn = tag_totals[name]; cnt = tag_counts[name]
            pct = 100.0 * tn / total_ref if total_ref > 0 else 0
            print(f"{name:>16s} {tn/1e6:>12.3f} {cnt:>6d} {tn/cnt/1000:>12.2f} {pct:>11.1f}%")

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
                ts=(start - gb) / 1000.0, dur=dur / 1000.0, pid=sm_id, tid=bid))
    return json.dumps({"traceEvents": events})


def run_single(impl_module, workload_idx: int) -> str:
    label, seq_len, bt_list = WORKLOAD_CASES[workload_idx]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n── {label}  (seq_len={seq_len}, threads={impl_module.NUM_THREADS}) ──")
    print(f"Compiling load-A kernel from {impl_module.__name__}...")
    kernel, compiled = impl_module.compile_kernel()

    device = "cuda"
    torch.manual_seed(len(bt_list))
    num_pg_real = len(bt_list)
    num_pg = num_pg_real if num_pg_real % 2 == 0 else num_pg_real + 1
    bt_padded = bt_list + ([0] if num_pg != num_pg_real else [])
    grid_m = num_pg // PAGES_PER_TILE

    K_fp8_used = torch.randn(num_pg_real, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
    K_scales_used = torch.rand(num_pg_real, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5

    kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
    for i, pid in enumerate(bt_list):
        kv_pool[pid, :, :HEAD_DIM] = K_fp8_used[i].view(torch.uint8)
        kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
            K_scales_used[i].view(torch.uint8).reshape(PAGE_SIZE, 4))

    block_table = torch.tensor(bt_padded, dtype=torch.int32, device=device)
    probe = torch.zeros((grid_m, PROBE_COLS), dtype=torch.int64, device=device)
    sink  = torch.zeros(grid_m, dtype=torch.int32, device=device)

    for _ in range(3):
        probe.zero_(); sink.zero_()
        compiled(kv_pool, block_table, sink, probe)
        torch.cuda.synchronize()

    print(f"  sink[0]={sink[0].item()}  (proves loads were not DCE'd)")

    probe.zero_(); sink.zero_()
    compiled(kv_pool, block_table, sink, probe)
    torch.cuda.synchronize()

    return dump_probe(probe, grid_m,
        impl_module.TAGS, impl_module.TAG_NAMES, impl_module.PHASE_ORDER)
