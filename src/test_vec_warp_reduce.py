"""
Test script: vectorized warp reduction for score dot-product.

Three approaches compared:
  1. SCALAR: 16 iters, ld.global.b16 per element (current fused_tiny5)
  2. LOAD:   zipped_divide + .load() — notebook pattern for vectorized loads
  3. TV:     TV layout + cute.composition + .load() — full TV layout approach

PTX dump:  compiled.__ptx__
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


NUM_VEC: cutlass.Constexpr = 4   # BF16 elements per vectorized load
HEAD_DIM: cutlass.Constexpr = 512
WARP_SIZE: cutlass.Constexpr = 32
ITERS_PER_LANE: cutlass.Constexpr = HEAD_DIM // WARP_SIZE // NUM_VEC  # 4


# ── 1) Scalar baseline (current fused_tiny5 pattern) ─────────────────────────
@cute.kernel
def dot_scalar_kernel(
    q_vec: cute.Tensor,    # (512,) BF16
    kv_vec: cute.Tensor,   # (512,) BF16
    result: cute.Tensor,   # (1,) FP32
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_idx = cute.arch.lane_idx()
    wsize = cute.arch.WARP_SIZE

    if tidx < wsize:
        acc = cutlass.Float32(0)
        for k in range(HEAD_DIM // wsize):          # 16 iters
            q_val = cutlass.Float32(q_vec[k * wsize + lane_idx])
            k_val = cutlass.Float32(kv_vec[k * wsize + lane_idx])
            acc += q_val * k_val

        s = warp_reduce(acc, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            result[0] = s


# ── 2) Vectorized via zipped_divide + .load() ────────────────────────────────
# From elementwise_add.ipynb: tile tensor, slice sub-tensor, call .load()
# .load() on a contiguous sub-tensor triggers vectorized LDG
@cute.kernel
def dot_load_kernel(
    q_vec: cute.Tensor,    # (512,) BF16
    kv_vec: cute.Tensor,   # (512,) BF16
    result: cute.Tensor,   # (1,) FP32
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_idx = cute.arch.lane_idx()

    if tidx < WARP_SIZE:
        # Tile into groups of NUM_VEC consecutive elements
        # (512,) → ((4,), (128,))  — 128 groups of 4 contiguous BF16
        q_t  = cute.zipped_divide(q_vec,  (NUM_VEC,))
        kv_t = cute.zipped_divide(kv_vec, (NUM_VEC,))

        acc = cutlass.Float32(0)
        for it in range(ITERS_PER_LANE):                    # 4 iters
            group = it * WARP_SIZE + lane_idx
            # Slice: None = full inner tile (4 elems), group = which tile
            # .load() triggers vectorized load (LDG.64 for 4×BF16)
            q_frag  = q_t[(None, (group,))].load()
            kv_frag = kv_t[(None, (group,))].load()

            # Element-wise FP32 dot product on loaded fragment
            for v in cutlass.range_constexpr(NUM_VEC):
                acc += cutlass.Float32(q_frag[(v,)]) * cutlass.Float32(kv_frag[(v,)])

        s = warp_reduce(acc, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            result[0] = s


# ── 3) Vectorized via TV layout + composition ────────────────────────────────
# From elementwise_add.ipynb TV Layout section:
#   thr_layout × val_layout → tiler + tv_layout
#   zipped_divide by tiler, then composition with tv_layout inside kernel
@cute.kernel
def dot_tv_kernel(
    q_tiled: cute.Tensor,     # ((TILE,), (ITERS,)) BF16 — pre-tiled
    kv_tiled: cute.Tensor,    # ((TILE,), (ITERS,)) BF16 — pre-tiled
    result: cute.Tensor,      # (1,) FP32
    tv_layout: cute.Layout,   # (T, V) → logical offset in tile
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_idx = cute.arch.lane_idx()

    if tidx < WARP_SIZE:
        acc = cutlass.Float32(0)
        for it in range(ITERS_PER_LANE):
            # Get tile for this iteration: (TILE,) sub-tensor
            blk_q  = q_tiled[(None, it)]
            blk_kv = kv_tiled[(None, it)]

            # Compose: (tid, vid) → physical address
            frg_q  = cute.composition(blk_q,  tv_layout)
            frg_kv = cute.composition(blk_kv, tv_layout)

            # Slice for this thread: (V,) → address
            thr_q  = frg_q[(tidx, None)]
            thr_kv = frg_kv[(tidx, None)]

            # Vectorized load
            q_vals  = thr_q.load()
            kv_vals = thr_kv.load()

            # Accumulate
            for v in cutlass.range_constexpr(NUM_VEC):
                acc += cutlass.Float32(q_vals[(v,)]) * cutlass.Float32(kv_vals[(v,)])

        s = warp_reduce(acc, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            result[0] = s


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_scalar():
    q   = _fake(cute.BFloat16, (HEAD_DIM,), (0,), 16)
    kv  = _fake(cute.BFloat16, (HEAD_DIM,), (0,), 16)
    res = _fake(cute.Float32,  (1,),        (0,),  4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    @cute.jit
    def launcher(q_vec: cute.Tensor, kv_vec: cute.Tensor, result: cute.Tensor, stream):
        dot_scalar_kernel(q_vec, kv_vec, result).launch(
            grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    return cute.compile(launcher, q, kv, res, stream, options="--enable-tvm-ffi")


def compile_load():
    q   = _fake(cute.BFloat16, (HEAD_DIM,), (0,), 16)
    kv  = _fake(cute.BFloat16, (HEAD_DIM,), (0,), 16)
    res = _fake(cute.Float32,  (1,),        (0,),  4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    @cute.jit
    def launcher(q_vec: cute.Tensor, kv_vec: cute.Tensor, result: cute.Tensor, stream):
        dot_load_kernel(q_vec, kv_vec, result).launch(
            grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    return cute.compile(launcher, q, kv, res, stream, options="--enable-tvm-ffi")


def compile_tv():
    q   = _fake(cute.BFloat16, (HEAD_DIM,), (0,), 16)
    kv  = _fake(cute.BFloat16, (HEAD_DIM,), (0,), 16)
    res = _fake(cute.Float32,  (1,),        (0,),  4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    TILE_SIZE = WARP_SIZE * NUM_VEC  # 32 * 4 = 128

    @cute.jit
    def launcher(q_vec: cute.Tensor, kv_vec: cute.Tensor, result: cute.Tensor, stream):
        # TV layout: (thread=32, value=4) → offset in [0, 128)
        # Each thread owns 4 contiguous elements: tid*4+vid
        tv_layout = cute.make_layout((32, 4), stride=(4, 1))

        # 1D tiler: split (512,) → ((128,), (4,))
        q_tiled  = cute.zipped_divide(q_vec,  (TILE_SIZE,))
        kv_tiled = cute.zipped_divide(kv_vec, (TILE_SIZE,))
        dot_tv_kernel(q_tiled, kv_tiled, result, tv_layout).launch(
            grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    return cute.compile(launcher, q, kv, res, stream, options="--enable-tvm-ffi")


# ── Compile all ───────────────────────────────────────────────────────────────
print("=== Compiling SCALAR ===")
scalar_compiled = compile_scalar()
print("=== Compiling LOAD (.load() approach) ===")
load_compiled   = compile_load()
print("=== Compiling TV (TV layout approach) ===")
tv_compiled     = compile_tv()

def print_ptx_loads(ptx, label):
    if not ptx:
        print(f"{label}: PTX not available")
        return
    print(f"\n=== {label} — key instructions ===")
    for line in ptx.split('\n'):
        ll = line.strip()
        if any(kw in ll for kw in ['ld.', 'st.', 'fma.', 'add.f', 'shfl.', 'cvt.']):
            print(f"  {ll}")

def count_pattern(ptx, pattern):
    if not ptx: return 0
    return sum(1 for line in ptx.split('\n') if pattern in line)

def print_counts(ptx, label):
    if not ptx:
        print(f"{label}: PTX not available")
        return
    print(f"\n--- {label} instruction counts ---")
    for pat in ['ld.global.b16', 'ld.global.b32', 'ld.global.b64', 'ld.global.b128',
                'ld.global.v2', 'ld.global.v4',
                'ld.param', 'st.global', 'fma.rn', 'shfl.sync', 'cvt.']:
        c = count_pattern(ptx, pat)
        if c: print(f"  {pat}: {c}")


# ── Correctness test + PTX extraction ────────────────────────────────────────
def test():
    q  = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    res_scalar = torch.zeros(1, dtype=torch.float32, device="cuda")
    res_load   = torch.zeros(1, dtype=torch.float32, device="cuda")
    res_tv     = torch.zeros(1, dtype=torch.float32, device="cuda")

    # Execute first to trigger JIT compilation
    scalar_compiled(q, kv, res_scalar)
    load_compiled(q, kv, res_load)
    tv_compiled(q, kv, res_tv)

    ref = (q.float() * kv.float()).sum().item()
    print(f"\nReference:  {ref:.6f}")
    print(f"Scalar:     {res_scalar.item():.6f}  err={abs(res_scalar.item() - ref):.2e}")
    print(f"Load:       {res_load.item():.6f}  err={abs(res_load.item() - ref):.2e}")
    print(f"TV:         {res_tv.item():.6f}  err={abs(res_tv.item() - ref):.2e}")

    ok = all(abs(r.item() - ref) < 1.0 for r in [res_scalar, res_load, res_tv])
    print("PASS ✓" if ok else "FAIL ✗")

    # Extract PTX after execution
    ptx_scalar = getattr(scalar_compiled, '__ptx__', None)
    ptx_load   = getattr(load_compiled, '__ptx__', None)
    ptx_tv     = getattr(tv_compiled, '__ptx__', None)

    for ptx, label in [(ptx_scalar, "SCALAR"), (ptx_load, "LOAD"), (ptx_tv, "TV")]:
        print_ptx_loads(ptx, label)
        print_counts(ptx, label)

test()
