"""Modal runner: persistent_v3_shell — real CUTLASS scheduler work-distribution diagnostic.

Prints two tables:
  1. Per-workload detailed 148-row table (controlled by PROBE_WL)
  2. Grand summary table at end

Usage:
    modal run src/modal/persistent_v3_shell.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

# ── Which workloads get the detailed 148-row table? ──────────────────────────
# None  → summary table only (no detailed per-SM output)
# int   → single workload, e.g. 17
# list  → multiple, e.g. [17, 21, 22]
# "all" → every workload (warning: very long output)
PROBE_WL = None


def _parse_probe(probe_cpu, count_cpu, gvc_cpu, T_val, num_ctas, num_splits, num_heads):
    """Return (cta_waves, vsums, stats)."""
    n_sh = num_splits * num_heads
    cta_waves = {}
    max_waves = 0
    for cta in range(num_ctas):
        n = count_cpu[cta].item()
        tasks = []
        for t in range(n):
            row = probe_cpu[cta, t]
            tasks.append(dict(
                sm=row[0].item(), flat=row[1].item(), tok=row[2].item(),
                split=row[3].item(), head=row[4].item(), lv=row[5].item(),
            ))
        cta_waves[cta] = tasks
        max_waves = max(max_waves, len(tasks))

    per_tok = gvc_cpu[:T_val].tolist()
    vsums = [sum(t["lv"] for t in cta_waves[c]) for c in range(num_ctas)]
    reals = [sum(1 for t in cta_waves[c] if t["lv"] > 0) for c in range(num_ctas)]
    active_vsums = [v for v in vsums if v > 0]

    stats = dict(
        per_tok=per_tok, max_waves=max_waves,
        total_tasks=sum(len(cta_waves[c]) for c in range(num_ctas)),
        total_real=sum(reals),
        total_oob=sum(len(cta_waves[c]) for c in range(num_ctas)) - sum(reals),
        vsum_max=max(vsums) if vsums else 0,
        vsum_min=min(active_vsums) if active_vsums else 0,
        active=len(active_vsums),
        idle=num_ctas - len(active_vsums),
    )
    return cta_waves, vsums, stats


def _fmt_valid(per_tok):
    if len(per_tok) <= 4:
        return "[" + ",".join(str(v) for v in per_tok) + "]"
    shown = ",".join(str(v) for v in per_tok[:3])
    return f"[{shown},...+{len(per_tok)-3}]"


def _cell(task):
    if task is None:
        return ""
    if task["lv"] == 0:
        return f"t{task['tok']}s{task['split']}h{task['head']}=0"
    return f"t{task['tok']}s{task['split']}h{task['head']}={task['lv']}"


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def run_shell_remote():
    import sys, json
    sys.path.insert(0, "/app")

    import torch
    from pathlib import Path
    from safetensors.torch import load_file

    from src.kernels.fused_persistent_v3_shell import (
        run_shell, MAX_ACTIVE_CLUSTERS, DIM_SPLIT, NUM_SPLITS, NUM_HEADS,
    )

    from src import utils
    utils.CONTEST = Path("/data")
    utils.JSONL = utils.CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    workloads = [json.loads(l) for l in open(utils.JSONL)]
    N_CTA = MAX_ACTIVE_CLUSTERS  # 148

    print(f"GPU: {torch.cuda.get_device_name(0)}  |  SMs: {N_CTA}  |  "
          f"Scheduler: StaticPersistentTileScheduler (round-robin)")
    print(f"H={NUM_HEADS}  S={NUM_SPLITS}  DIM_SPLIT={DIM_SPLIT}  "
          f"tasks/workload = T*{NUM_SPLITS}*{NUM_HEADS} = T*128")
    print()

    summary_rows = []

    for i_w, w in enumerate(workloads):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        T_val = ax["num_tokens"]

        sf = load_file(str(utils.CONTEST / inp["sparse_indices"]["path"]))
        si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

        probe, probe_count, gvc = run_shell(si, T_val)
        cta_waves, vsums, stats = _parse_probe(
            probe.cpu(), probe_count.cpu(), gvc.cpu(),
            T_val, N_CTA, NUM_SPLITS, NUM_HEADS)

        oob_pct = 100 * stats["total_oob"] / max(stats["total_tasks"], 1)
        imbal = stats["vsum_max"] / max(stats["vsum_min"], 1)

        summary_rows.append(dict(
            wl=i_w+1, T=T_val, valid=stats["per_tok"],
            real=stats["total_real"], oob_pct=oob_pct,
            max_work=stats["vsum_max"], idle=stats["idle"], ratio=imbal,
        ))

        # ── Detailed table: 148 rows x waves (only for selected WLs) ────
        wl_num = i_w + 1
        show_detail = (
            PROBE_WL == "all"
            or (isinstance(PROBE_WL, int) and PROBE_WL == wl_num)
            or (isinstance(PROBE_WL, (list, tuple)) and wl_num in PROBE_WL)
        )
        if show_detail:
            nw = stats["max_waves"]
            CW = 16
            print(f"{'─'*60}")
            print(f"WL{wl_num}  T={T_val}  valid={_fmt_valid(stats['per_tok'])}  "
                  f"real={stats['total_real']}  OOB={oob_pct:.0f}%  "
                  f"maxW={stats['vsum_max']}  idle={stats['idle']}  ratio={imbal:.1f}x")
            print(f"{'─'*60}")
            hdr = f"{'CTA':>4} {'SM':>4} {'work':>6}"
            for wi in range(nw):
                hdr += f"  {'wave'+str(wi):>{CW}}"
            print(hdr)
            print(f"{'─'*4} {'─'*4} {'─'*6}" + f"  {'─'*CW}" * nw)
            for cta in range(N_CTA):
                tasks = cta_waves[cta]
                sm = tasks[0]["sm"] if tasks else cta
                vsum = vsums[cta]
                row = f"{cta:>4} {sm:>4} {vsum:>6}"
                for wi in range(nw):
                    t = tasks[wi] if wi < len(tasks) else None
                    c = _cell(t)
                    row += f"  {c:>{CW}}"
                print(row)
            print()

    # ── Grand summary table ───────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print(f"SUMMARY — {len(workloads)} workloads")
    print(f"{'═'*100}")
    print(f"{'WL':>3}  {'T':>2}  {'valid':<22}  {'real':>5}  {'OOB%':>5}  "
          f"{'maxW/SM':>7}  {'idle':>4}  {'ratio':>6}")
    print(f"{'─'*3}  {'─'*2}  {'─'*22}  {'─'*5}  {'─'*5}  "
          f"{'─'*7}  {'─'*4}  {'─'*6}")
    for r in summary_rows:
        vstr = _fmt_valid(r["valid"])
        print(f"{r['wl']:>3}  {r['T']:>2}  {vstr:<22}  {r['real']:>5}  "
              f"{r['oob_pct']:>4.0f}%  {r['max_work']:>7}  "
              f"{r['idle']:>4}  {r['ratio']:>5.1f}x")
    print(f"{'═'*100}\n")

    return "done"


@app.local_entrypoint()
def main():
    run_shell_remote.remote()
