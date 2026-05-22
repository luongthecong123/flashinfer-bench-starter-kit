"""Independent Modal script: install FA4 wheel and run smoke tests on B200.

FA4 (FlashAttention-4) is written in CuTeDSL and supports Hopper/Blackwell GPUs.
Wheel: flash_attn_4-4.0.0b4-py3-none-any.whl (pure-Python, contains JIT kernels)

Tests:
  1. Non-causal forward pass — compare against PyTorch reference
  2. GQA non-causal forward pass
  3. Latency sweep: randn inputs vs zeroed inputs

Usage:
    modal run src/modal/fa4_test.py
"""
import modal

FA4_WHL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "fa4-v4.0.0.beta4/flash_attn_4-4.0.0b4-py3-none-any.whl"
)

app = modal.App("fa4-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "triton",
        "packaging",
        "ninja",
        "numpy",
        "nvidia-cutlass-dsl",
    )
    .pip_install(FA4_WHL)
)


# ── helpers ───────────────────────────────────────────────────────────────────

_TEST_SCRIPT = """
import torch
import math
import time

def ref_attn(q, k, v):
    q32, k32, v32 = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(q32.shape[-1])
    attn = torch.softmax(torch.einsum("bshd,bthd->bsht", q32, k32) * scale, dim=-1)
    return torch.einsum("bsht,bthd->bshd", attn, v32).to(q.dtype)


from flash_attn.cute import flash_attn_func
import flash_attn
import importlib.metadata

print("flash_attn.cute imported OK")
print("  __file__  :", flash_attn.__file__)
try:
    print("  version   :", importlib.metadata.version("flash-attn-4"))
except importlib.metadata.PackageNotFoundError:
    print("  version   :", importlib.metadata.version("flash_attn_4"))
print()

DEVICE = "cuda"
DTYPE  = torch.bfloat16
TOL    = dict(atol=5e-2, rtol=1e-2)

# ── Test 1: non-causal ───────────────────────────────────────────
print("=" * 55)
print("Test 1: non-causal  (B=2, S=512, H=16, D=64)")
B, S, H, D = 2, 512, 16, 64
q = torch.randn(B, S, H, D, dtype=DTYPE, device=DEVICE)
k = torch.randn(B, S, H, D, dtype=DTYPE, device=DEVICE)
v = torch.randn(B, S, H, D, dtype=DTYPE, device=DEVICE)
out, _lse = flash_attn_func(q, k, v, causal=False)
ref = ref_attn(q, k, v)
match = torch.allclose(out, ref, **TOL)
max_err = (out.float() - ref.float()).abs().max().item()
print(f"  max_err={max_err:.4f}  pass={match}")

# ── Test 2: GQA non-causal ──────────────────────────────────────
print()
print("=" * 55)
print("Test 2: GQA  (B=2, S=512, Hq=16, Hkv=4, D=64)")
Hq, Hkv = 16, 4
q_gqa = torch.randn(B, S, Hq,  D, dtype=DTYPE, device=DEVICE)
k_gqa = torch.randn(B, S, Hkv, D, dtype=DTYPE, device=DEVICE)
v_gqa = torch.randn(B, S, Hkv, D, dtype=DTYPE, device=DEVICE)
out_gqa, _ = flash_attn_func(q_gqa, k_gqa, v_gqa, causal=False)
k_exp = k_gqa.repeat_interleave(Hq // Hkv, dim=2)
v_exp = v_gqa.repeat_interleave(Hq // Hkv, dim=2)
ref_gqa = ref_attn(q_gqa, k_exp, v_exp)
match_gqa = torch.allclose(out_gqa, ref_gqa, **TOL)
max_err_gqa = (out_gqa.float() - ref_gqa.float()).abs().max().item()
print(f"  max_err={max_err_gqa:.4f}  pass={match_gqa}")

# ── Test 3: latency sweep ───────────────────────────────────────
print()
print("=" * 55)
print("Test 3: latency sweep  (non-causal, BF16)")
print("  Paper config: total_tokens=32768, hidden=2048")

configs = [
    (1024,   64),
    (2048,   64),
    (4096,   64),
    (8192,   64),
    (16384,  64),
    (32768,  64),
    (1024,  128),
    (2048,  128),
    (4096,  128),
    (8192,  128),
    (16384, 128),
    (32768, 128),
]
TOTAL_TOKENS = 32768
HIDDEN = 2048
REPS = 20
HDR = "  %8s  %8s  %6s  %6s  %8s  %8s"
ROW = "  %8d  %8d  %6d  %6d  %8.3f  %8.2f"

def bench(label, zeroed):
    print()
    print("  -- inputs: %s --" % label)
    print(HDR % ("seqlen", "headdim", "batch", "heads", "ms", "TFLOPS"))
    for S_, D_ in configs:
        B_ = max(1, TOTAL_TOKENS // S_)
        H_ = HIDDEN // D_
        mk = torch.zeros if zeroed else torch.randn
        q_ = mk(B_, S_, H_, D_, dtype=DTYPE, device=DEVICE)
        k_ = mk(B_, S_, H_, D_, dtype=DTYPE, device=DEVICE)
        v_ = mk(B_, S_, H_, D_, dtype=DTYPE, device=DEVICE)
        for _ in range(5):
            flash_attn_func(q_, k_, v_, causal=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            flash_attn_func(q_, k_, v_, causal=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3 / REPS
        flops = 4 * B_ * H_ * S_ * S_ * D_
        tflops = flops / (ms * 1e-3) / 1e12
        print(ROW % (S_, D_, B_, H_, ms, tflops))

bench("randn ", zeroed=False)
bench("zeroed", zeroed=True)

print()
print("All tests done.")
"""


@app.function(image=image, gpu="B200:1", timeout=600)
def run_fa4_tests() -> str:
    import subprocess, sys, textwrap
    result = subprocess.run(
        [sys.executable, "-c", _TEST_SCRIPT],
        capture_output=True, text=True, timeout=540,
    )
    output = result.stdout + result.stderr
    return output


@app.local_entrypoint()
def main():
    print("\n" + "=" * 60)
    print("Running FA4 smoke tests on B200 ...")
    print("=" * 60 + "\n")
    output = run_fa4_tests.remote()
    print(output)
