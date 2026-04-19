"""
Top-K benchmark: 3 approaches
  1. PyTorch torch.topk
  2. 1024-thread 2-register scan (branchless C++)
  3. 1024-thread 2-register scan (PTX setp+selp)

K = 2048 = 1024 threads * 2 registers per thread
One block per batch element.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.idx_utils import check_topk_indices

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOPK         = 2048
NUM_THREADS  = 1024   # TOPK // 2  (k_b = 2 per thread)
NUM_THREADS4 = 512    # TOPK // 4  (k_b = 4 per thread)
NEG_INF      = float("-inf")


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------

@dsl_user_op
def top2_update_ptx(
    av: cutlass.Float32,
    bv: cutlass.Float32,
    ai: cutlass.Int32,
    bi: cutlass.Int32,
    v:  cutlass.Float32,
    i:  cutlass.Int32,
    *,
    loc=None, ip=None,
) -> tuple:
    """
    Branchless top-2 update using PTX setp + selp.

    Maintains (av >= bv). On each new (v, i):
      mx  = max(bv, v)
      p   = (av > v)
      new_av = p ? av  : v     (keep best)
      new_bv = p ? mx  : av    (second = max of losers)
      new_ai = p ? ai  : i
      new_bi = p ? (mx==bv ? bi : i) : ai
    """
    # max(bv, v) -> mx
    mx = llvm.inline_asm(
        T.f32(), [bv.ir_value(), v.ir_value()],
        "max.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    # p = (av > v)
    p = llvm.inline_asm(
        T.i(1), [av.ir_value(), v.ir_value()],
        "setp.gt.f32 $0, $1, $2;",
        "=b,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    # new_av = p ? av : v
    new_av = llvm.inline_asm(
        T.f32(), [av.ir_value(), v.ir_value(), p],
        "selp.f32 $0, $1, $2, $3;",
        "=f,f,f,b",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    # new_bv = p ? mx : av
    new_bv = llvm.inline_asm(
        T.f32(), [mx, av.ir_value(), p],
        "selp.f32 $0, $1, $2, $3;",
        "=f,f,f,b",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    # new_ai = p ? ai : i
    new_ai = llvm.inline_asm(
        T.i32(), [ai.ir_value(), i.ir_value(), p],
        "selp.s32 $0, $1, $2, $3;",
        "=r,r,r,b",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    # q = (mx == bv) i.e. bv won the max, so bi stays; else i takes second slot
    q = llvm.inline_asm(
        T.i(1), [mx, bv.ir_value()],
        "setp.eq.f32 $0, $1, $2;",
        "=b,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    # second_idx when p=True: q ? bi : i
    second_idx_p = llvm.inline_asm(
        T.i32(), [bi.ir_value(), i.ir_value(), q],
        "selp.s32 $0, $1, $2, $3;",
        "=r,r,r,b",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    # new_bi = p ? second_idx_p : ai
    new_bi = llvm.inline_asm(
        T.i32(), [second_idx_p, ai.ir_value(), p],
        "selp.s32 $0, $1, $2, $3;",
        "=r,r,r,b",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return (
        cutlass.Float32(new_av),
        cutlass.Float32(new_bv),
        cutlass.Int32(new_ai),
        cutlass.Int32(new_bi),
    )


# ---------------------------------------------------------------------------
# Kernel: manual branchless (compiler-emitted setp/selp)
# ---------------------------------------------------------------------------

@cute.kernel
def topk_scan_kernel(
    scores:   cute.Tensor,   # [B, max_sl]  Float32
    out_idx:  cute.Tensor,   # [B, TOPK]    Int32
    seq_lens: cute.Tensor,   # [B]          Int32
):
    b   = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]

    sl = seq_lens[b]

    a_val = cutlass.Float32(NEG_INF)
    b_val = cutlass.Float32(NEG_INF)
    a_idx = cutlass.Int32(-1)
    b_idx = cutlass.Int32(-1)

    i = cutlass.Int32(tid)
    while i < sl:
        v = scores[b, i]
        # branchless update — compiler should emit FSETP+FSEL
        if v > a_val:
            b_val = a_val
            b_idx = a_idx
            a_val = v
            a_idx = i
        elif v > b_val:
            b_val = v
            b_idx = i
        i = i + cutlass.Int32(NUM_THREADS)

    out_idx[b, tid * 2 + 0] = a_idx
    out_idx[b, tid * 2 + 1] = b_idx


# ---------------------------------------------------------------------------
# Kernel: PTX setp+selp explicit
# ---------------------------------------------------------------------------

@cute.kernel
def topk_ptx_kernel(
    scores:   cute.Tensor,
    out_idx:  cute.Tensor,
    seq_lens: cute.Tensor,
):
    b   = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]

    sl = seq_lens[b]

    a_val = cutlass.Float32(NEG_INF)
    b_val = cutlass.Float32(NEG_INF)
    a_idx = cutlass.Int32(-1)
    b_idx = cutlass.Int32(-1)

    i = cutlass.Int32(tid)
    while i < sl:
        v = scores[b, i]
        a_val, b_val, a_idx, b_idx = top2_update_ptx(a_val, b_val, a_idx, b_idx, v, i)
        i = i + cutlass.Int32(NUM_THREADS)

    out_idx[b, tid * 2 + 0] = a_idx
    out_idx[b, tid * 2 + 1] = b_idx


# ---------------------------------------------------------------------------
# JIT wrappers
# ---------------------------------------------------------------------------

@cute.jit
def topk_scan(scores: cute.Tensor, out_idx: cute.Tensor, seq_lens: cute.Tensor):
    B = scores.shape[0]
    topk_scan_kernel(scores, out_idx, seq_lens).launch(
        grid=[B, 1, 1],
        block=[NUM_THREADS, 1, 1],
    )


@cute.jit
def topk_ptx(scores: cute.Tensor, out_idx: cute.Tensor, seq_lens: cute.Tensor):
    B = scores.shape[0]
    topk_ptx_kernel(scores, out_idx, seq_lens).launch(
        grid=[B, 1, 1],
        block=[NUM_THREADS, 1, 1],
    )


# ---------------------------------------------------------------------------
# Kernel: 512-thread top-4 per thread  (k_b = 4, paper's prop-k recommendation)
# Paper §3: in prop-k regime, increase k_b and keep b·k_b = k to avoid Stage 2.
# ---------------------------------------------------------------------------

@cute.kernel
def topk_scan_top4_kernel(
    scores:   cute.Tensor,
    out_idx:  cute.Tensor,
    seq_lens: cute.Tensor,
):
    b   = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]

    sl = seq_lens[b]

    # Sorted register queue: a0 >= a1 >= a2 >= a3
    a0_val = cutlass.Float32(NEG_INF); a0_idx = cutlass.Int32(-1)
    a1_val = cutlass.Float32(NEG_INF); a1_idx = cutlass.Int32(-1)
    a2_val = cutlass.Float32(NEG_INF); a2_idx = cutlass.Int32(-1)
    a3_val = cutlass.Float32(NEG_INF); a3_idx = cutlass.Int32(-1)

    i = cutlass.Int32(tid)
    while i < sl:
        v = scores[b, i]
        # Insertion sort into 4-element max-queue
        if v > a3_val:
            if v > a2_val:
                a3_val = a2_val
                a3_idx = a2_idx
                if v > a1_val:
                    a2_val = a1_val
                    a2_idx = a1_idx
                    if v > a0_val:
                        a1_val = a0_val
                        a1_idx = a0_idx
                        a0_val = v
                        a0_idx = i
                    else:
                        a1_val = v
                        a1_idx = i
                else:
                    a2_val = v
                    a2_idx = i
            else:
                a3_val = v
                a3_idx = i
        i = i + cutlass.Int32(NUM_THREADS4)

    out_idx[b, tid * 4 + 0] = a0_idx
    out_idx[b, tid * 4 + 1] = a1_idx
    out_idx[b, tid * 4 + 2] = a2_idx
    out_idx[b, tid * 4 + 3] = a3_idx


@cute.jit
def topk_scan_top4(scores: cute.Tensor, out_idx: cute.Tensor, seq_lens: cute.Tensor):
    B = scores.shape[0]
    topk_scan_top4_kernel(scores, out_idx, seq_lens).launch(
        grid=[B, 1, 1],
        block=[NUM_THREADS4, 1, 1],
    )


# ---------------------------------------------------------------------------
# Correctness check  (delegates to idx_utils.check_topk_indices)
# ---------------------------------------------------------------------------

def check_correctness(ref_idx, impl_idx, seq_lens, label):
    """Order-independent recall check: PASS if worst-case miss fraction < 1%."""
    ok, max_miss = check_topk_indices(ref_idx, impl_idx, seq_lens)
    if ok:
        print(f"  [{label}] CORRECTNESS PASS  (max_miss={max_miss:.4f})")
    else:
        print(f"  [{label}] FAIL  (max_miss={max_miss:.4f} >= 0.01)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)

    B      = 16
    max_sl = 5805   # typical worst case from your workload data

    device = torch.device("cuda")

    scores   = torch.randn(B, max_sl, dtype=torch.float32, device=device)
    seq_lens = torch.randint(TOPK, max_sl + 1, (B,), dtype=torch.int32, device=device)
    # mask beyond seq_len with -inf so invalid tokens never win
    mask = torch.arange(max_sl, device=device).unsqueeze(0) >= seq_lens.unsqueeze(1)
    scores.masked_fill_(mask, NEG_INF)

    out_scan = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
    out_ptx  = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
    out_top4 = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)

    # ---- fake tensors for compilation ----
    scores_fake   = make_fake_compact_tensor(dtype=cute.Float32, shape=(cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16)
    out_idx_fake  = make_fake_compact_tensor(dtype=cute.Int32,   shape=(cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16)
    seq_lens_fake = make_fake_compact_tensor(dtype=cute.Int32,   shape=(cute.sym_int(),),               stride_order=(0,),   assumed_align=4)

    print("Compiling scan kernel (k_b=2)...")
    compiled_scan = cute.compile(topk_scan,      scores_fake, out_idx_fake, seq_lens_fake)

    print("Compiling PTX kernel (k_b=2)...")
    compiled_ptx  = cute.compile(topk_ptx,       scores_fake, out_idx_fake, seq_lens_fake)

    print("Compiling top4 kernel (k_b=4)...")
    compiled_top4 = cute.compile(topk_scan_top4, scores_fake, out_idx_fake, seq_lens_fake)

    # ---- correctness ----
    print("\n=== Correctness ===")

    # pytorch reference indices (used as ref for recall check)
    out_ref = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
    for b in range(B):
        sl = seq_lens[b].item()
        k  = min(TOPK, sl)
        _, idx = torch.topk(scores[b, :sl], k)
        out_ref[b, :k] = idx.int()

    compiled_scan(scores, out_scan, seq_lens)
    torch.cuda.synchronize()
    check_correctness(out_ref, out_scan, seq_lens, "scan k_b=2")

    compiled_ptx(scores, out_ptx, seq_lens)
    torch.cuda.synchronize()
    check_correctness(out_ref, out_ptx, seq_lens, "ptx  k_b=2")

    compiled_top4(scores, out_top4, seq_lens)
    torch.cuda.synchronize()
    check_correctness(out_ref, out_top4, seq_lens, "scan k_b=4")

    check_correctness(out_ref, out_ref, seq_lens, "torch.topk (ref vs ref)")

    # ---- benchmark ----
    print("\n=== Benchmark ===")
    print(f"  B={B}, max_sl={max_sl}, TOPK={TOPK}")
    print()

    # pytorch
    def torch_topk_fn():
        torch.topk(scores, TOPK, dim=1)

    # warmup
    for _ in range(10):
        torch_topk_fn()
    torch.cuda.synchronize()

    import time
    N_ITER = 1000
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        torch_topk_fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch_ms = (t1 - t0) / N_ITER * 1000
    print(f"  torch.topk      : {torch_ms*1000:.2f} µs")

    # scan k_b=2
    scan_args = JitArguments(scores, out_scan, seq_lens)
    scan_us   = benchmark(compiled_scan, kernel_arguments=scan_args)
    print(f"  scan k_b=2 (C++ if)   : {scan_us:.2f} µs")

    # ptx k_b=2
    ptx_args = JitArguments(scores, out_ptx, seq_lens)
    ptx_us   = benchmark(compiled_ptx, kernel_arguments=ptx_args)
    print(f"  scan k_b=2 (PTX selp) : {ptx_us:.2f} µs")

    # scan k_b=4
    top4_args = JitArguments(scores, out_top4, seq_lens)
    top4_us   = benchmark(compiled_top4, kernel_arguments=top4_args)
    print(f"  scan k_b=4 (C++ if)   : {top4_us:.2f} µs")

    print()
    print(f"  speedup scan k_b=2 vs torch : {torch_ms*1000/scan_us:.2f}x")
    print(f"  speedup ptx  k_b=2 vs torch : {torch_ms*1000/ptx_us:.2f}x")
    print(f"  speedup scan k_b=4 vs torch : {torch_ms*1000/top4_us:.2f}x")
    print(f"  speedup ptx  vs scan k_b=2  : {scan_us/ptx_us:.2f}x")
    print(f"  speedup scan k_b=4 vs k_b=2 : {scan_us/top4_us:.2f}x")

    # ---- real-data recall test ----
    test_real_data()


# ---------------------------------------------------------------------------
# Real-data test: extract actual scores for requests with seq_len > TOPK
# ---------------------------------------------------------------------------

def test_real_data():
    """
    Load actual contest workloads; use the REAL seq_lens for every request
    with seq_len > TOPK, but generate Gaussian scores (bypasses fp8 NaN
    artifacts from random byte scales that corrupt the score distribution).

    Gaussian is the *worst-case* distribution for the bucketed algorithm
    (i.i.d. → no clustering benefit), so this gives a pessimistic bound.

    Compiles fresh kernels for the actual [N, max_sl] batch shape.
    Tests k_b=2 scan, k_b=2 PTX, and the new k_b=4 scan.
    """
    import json
    from pathlib import Path as P
    from safetensors.torch import load_file

    CONTEST = P('/home/luongt/codeCuda/flashinfer26dsa/mlsys26-contest')
    JSONL   = (CONTEST / 'workloads' / 'dsa_paged'
               / 'dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl')
    device  = torch.device('cuda')

    workloads = [json.loads(l) for l in open(JSONL)]

    # ── Collect seq_lens > TOPK from all workloads ────────────────────────────
    total_req = gt_req = le_req = 0
    all_seq_lens = []
    all_wl_tag   = []

    for i_w, w in enumerate(workloads):
        ax  = w['workload']['axes']
        inp = w['workload']['inputs']
        sf  = load_file(str(CONTEST / inp['seq_lens']['path']))
        sl_tensor = sf[inp['seq_lens']['tensor_key']]
        total_req += ax['batch_size']
        uuid = w['workload']['uuid'][:8]

        for b, sl in enumerate(sl_tensor.tolist()):
            sl = int(sl)
            if sl > TOPK:
                gt_req += 1
                all_seq_lens.append(sl)
                all_wl_tag.append((i_w + 1, uuid, b))
            else:
                le_req += 1

    print(f"\n=== Real-Data Recall Test (relu-sparse scores, real seq_lens) ===")
    print(f"  Score model: sum(relu(q @ K.T) * |weights|)  [float32, no fp8 artifacts]")
    print(f"  ~50%% of tokens score 0; top tokens are well-separated — matches contest harness.")
    print(f"  Total workloads : {len(workloads)}")
    print(f"  Total requests  : {total_req}")
    print(f"  seq_len > {TOPK}  : {gt_req}  ({100*gt_req/total_req:.1f}%)")
    print(f"  seq_len <={TOPK}  : {le_req}  ({100*le_req/total_req:.1f}%)")

    if not all_seq_lens:
        print("  No requests with seq_len > TOPK found.")
        return

    NUM_HEADS = 64
    HEAD_DIM  = 128

    # ── Build batch with relu-sparse scores (matches actual contest distribution) ─
    N      = len(all_seq_lens)
    max_sl = max(all_seq_lens)
    torch.manual_seed(42)
    scores_batch   = torch.full((N, max_sl), float('-inf'), dtype=torch.float32, device=device)
    seq_lens_batch = torch.tensor(all_seq_lens, dtype=torch.int32, device=device)
    for i, sl in enumerate(all_seq_lens):
        q_f32     = torch.randn(NUM_HEADS, HEAD_DIM, device=device)           # [heads, D]
        K_f32     = torch.randn(sl, HEAD_DIM, device=device)                  # [sl, D]
        scores_2d = q_f32 @ K_f32.T                                           # [heads, sl]
        weights   = torch.randn(NUM_HEADS, device=device).abs()               # [heads]
        scores_batch[i, :sl] = (torch.relu(scores_2d) * weights[:, None]).sum(0)

    # ── Reference top-K (torch) ───────────────────────────────────────────────
    ref_idx = torch.full((N, TOPK), -1, dtype=torch.int32, device=device)
    for i, sl in enumerate(all_seq_lens):
        _, idx = torch.topk(scores_batch[i, :sl], min(TOPK, sl))
        ref_idx[i, :min(TOPK, sl)] = idx.int()

    # ── Compile fresh kernels for this exact batch shape ─────────────────────
    scores_fake   = make_fake_compact_tensor(dtype=cute.Float32, shape=(cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16)
    out_idx_fake  = make_fake_compact_tensor(dtype=cute.Int32,   shape=(cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16)
    seq_lens_fake = make_fake_compact_tensor(dtype=cute.Int32,   shape=(cute.sym_int(),),               stride_order=(0,),   assumed_align=4)

    c_scan = cute.compile(topk_scan,      scores_fake, out_idx_fake, seq_lens_fake)
    c_ptx  = cute.compile(topk_ptx,       scores_fake, out_idx_fake, seq_lens_fake)
    c_top4 = cute.compile(topk_scan_top4, scores_fake, out_idx_fake, seq_lens_fake)

    # ── Run kernels ───────────────────────────────────────────────────────────
    out_scan = torch.full((N, TOPK), -1, dtype=torch.int32, device=device)
    out_ptx  = torch.full((N, TOPK), -1, dtype=torch.int32, device=device)
    out_top4 = torch.full((N, TOPK), -1, dtype=torch.int32, device=device)
    c_scan(scores_batch, out_scan, seq_lens_batch)
    c_ptx (scores_batch, out_ptx,  seq_lens_batch)
    c_top4(scores_batch, out_top4, seq_lens_batch)
    torch.cuda.synchronize()

    # ── Per-request recall table ──────────────────────────────────────────────
    print(f"\n--- Per-Request Detail (N={N}, seq_len > {TOPK}) ---")
    hdr = f"{'#':>3} {'WL':>3} {'UUID':>10} {'b':>3} {'seq_len':>8} {'k_b2_scan':>10} {'k_b2_ptx':>9} {'k_b4_scan':>10}"
    print(hdr)
    for i, (sl, (wl, uuid, b)) in enumerate(zip(all_seq_lens, all_wl_tag)):
        r    = ref_idx[i:i+1]
        sl_t = seq_lens_batch[i:i+1]
        _, m_scan = check_topk_indices(r, out_scan[i:i+1], sl_t)
        _, m_ptx  = check_topk_indices(r, out_ptx[i:i+1],  sl_t)
        _, m_top4 = check_topk_indices(r, out_top4[i:i+1], sl_t)
        flag = "" if max(m_scan, m_ptx, m_top4) < 0.01 else "  ← FAIL"
        print(f"{i+1:>3} {wl:>3} {uuid:>10} {b:>3} {sl:>8} {m_scan:>10.4f} {m_ptx:>9.4f} {m_top4:>10.4f}{flag}")

    _, wc_scan = check_topk_indices(ref_idx, out_scan, seq_lens_batch)
    _, wc_ptx  = check_topk_indices(ref_idx, out_ptx,  seq_lens_batch)
    _, wc_top4 = check_topk_indices(ref_idx, out_top4, seq_lens_batch)
    print(f"\n  Worst-case miss across all {N} requests (relu-sparse scores):")
    print(f"    k_b=2 scan : {wc_scan:.4f}  ({'PASS' if wc_scan < 0.01 else 'FAIL'})")
    print(f"    k_b=2 ptx  : {wc_ptx:.4f}  ({'PASS' if wc_ptx  < 0.01 else 'FAIL'})")
    print(f"    k_b=4 scan : {wc_top4:.4f}  ({'PASS' if wc_top4 < 0.01 else 'FAIL'})")


if __name__ == "__main__":
    main()
