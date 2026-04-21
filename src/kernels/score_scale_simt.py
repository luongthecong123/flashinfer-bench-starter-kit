"""
score_scale_simt.py — SIMT FP8 GEMM with per-token scale, correct FLAT byte layout.

Memory layout per page (8448 bytes total = 64 tokens × 132 bytes/token):
  The tensor is (num_pages, 64, 1, 132) int8, C-contiguous.
  Physical bytes: [row0(132B), row1(132B), ..., row63(132B)]

Reference dequant (idxer_tc.py) does FLAT extraction:
  kv_flat = view(num_pages, 8448)
  fp8  = kv_flat[:, :8192].reshape(64, 128)   → token t, dim d = flat byte t*128+d
  scale = kv_flat[:, 8192:].reshape(64, 4).view(float32)  → token t scale = flat bytes 8192+t*4

So fp8 for token t lives at flat byte offset t*128 (NOT t*132).
Scale for token t lives at flat byte offset 8192 + t*4.

Kernel: SIMT dot product, 128 threads per block, one block per 2-page (128-token) tile.
  C[m, n] = sum_k(fp8_A[m,k] * fp8_B[n,k]) * scale[m]
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.testing import benchmark, JitArguments

# ── Dimensions ───────────────────────────────────────────────────────────────
M           = 2048   # total KV tokens
N           = 64     # query heads
HEAD_DIM    = 128    # fp8 head dim (K)
PAGE_SIZE   = 64     # tokens per page
ROW_STRIDE  = HEAD_DIM + 4   # 132 bytes per physical row
PAGE_BYTES  = PAGE_SIZE * ROW_STRIDE   # 8448 bytes per page
FP8_REGION  = PAGE_SIZE * HEAD_DIM     # 8192 bytes of fp8 per page
PAGES_PER_TILE = 2                     # 128 tokens = 2 pages per block
BM          = PAGE_SIZE * PAGES_PER_TILE   # 128 tokens per block

# SIMT constants
NUM_VEC     = 4
K_ITERS     = HEAD_DIM // NUM_VEC      # 32


class ScoreScaleSIMT:
    """SIMT FP8 GEMM + scale with correct flat byte layout."""

    def __init__(self):
        self.threads = BM   # 128 threads, one per token row

    @cute.jit
    def __call__(
        self,
        k_index_cache_fp8,   # (num_pages, PAGE_SIZE, 1, ROW_STRIDE) int8
        q_fp8,               # (N, HEAD_DIM) float8_e4m3fn
        c_out,               # (M, N) float32
        stream,
    ):
        num_pages = k_index_cache_fp8.shape[0]
        grid_m = num_pages // PAGES_PER_TILE   # M/128

        self.kernel(
            k_index_cache_fp8, q_fp8, c_out, num_pages,
        ).launch(
            grid=[grid_m, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        k_index_cache_fp8,   # (num_pages, PAGE_SIZE, 1, ROW_STRIDE) int8
        q_fp8,               # (N, HEAD_DIM) float8_e4m3fn
        c_out,               # (M, N) float32
        num_pages,           # int
    ):
        num_vec: cutlass.Constexpr = NUM_VEC
        k_iters: cutlass.Constexpr = K_ITERS

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # This thread handles token m_global
        m_global = bidx * cutlass.Int32(BM) + tidx

        # Which page and token within page (for flat byte addressing)
        page_sel      = tidx // cutlass.Int32(PAGE_SIZE)      # 0 or 1
        token_in_page = tidx - page_sel * cutlass.Int32(PAGE_SIZE)  # 0..63
        page_id       = bidx * cutlass.Int32(PAGES_PER_TILE) + page_sel

        # ── FP8 data: flat byte offset = page_id * PAGE_BYTES + token_in_page * HEAD_DIM ──
        # (Within each page, the reference reads contiguous bytes 0..8191 as 64×128 fp8)
        fp8_byte_off = page_id * cutlass.Int32(PAGE_BYTES) + token_in_page * cutlass.Int32(HEAD_DIM)

        a_fp8_ptr = cute.make_ptr(
            cutlass.Float8E4M3FN,
            (cute.recast_ptr(k_index_cache_fp8.iterator, dtype=cutlass.Float8E4M3FN) + fp8_byte_off).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=1,
        )
        a_row = cute.make_tensor(a_fp8_ptr, cute.make_layout((HEAD_DIM,), stride=(1,)))
        a_z = cute.zipped_divide(a_row, (num_vec,))

        # ── Scale: flat byte offset = page_id * PAGE_BYTES + FP8_REGION + token_in_page * 4 ──
        scale_byte_off = page_id * cutlass.Int32(PAGE_BYTES) + cutlass.Int32(FP8_REGION) + token_in_page * cutlass.Int32(4)
        scale_f32_off = scale_byte_off // cutlass.Int32(4)
        scale_ptr = cute.make_ptr(
            cutlass.Float32,
            (cute.recast_ptr(k_index_cache_fp8.iterator, dtype=cutlass.Float32) + scale_f32_off).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=1,
        )
        scale_tensor = cute.make_tensor(scale_ptr, cute.make_layout((1,), stride=(1,)))
        scale = scale_tensor[0]

        # ── Q tensor: (N, HEAD_DIM) contiguous fp8 ──
        gQ = cute.make_tensor(
            q_fp8.iterator,
            cute.make_layout((N, HEAD_DIM), stride=(HEAD_DIM, 1)),
        )

        # ── SIMT dot product for each head ──
        for n_idx in range(N):
            b_row = gQ[n_idx, None]
            b_z   = cute.zipped_divide(b_row, (num_vec,))

            acc = cutlass.Float32(0)
            for k4 in range(k_iters):
                a_frag = a_z[(None, (k4,))].load()
                b_frag = b_z[(None, (k4,))].load()
                a_f32  = a_frag.to(cutlass.Float32)
                b_f32  = b_frag.to(cutlass.Float32)
                for v in cutlass.range_constexpr(num_vec):
                    acc += a_f32[v] * b_f32[v]

            # Apply scale and write
            c_out[m_global, n_idx] = acc * scale


# ── Compile ──────────────────────────────────────────────────────────────────
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)

def compile_kernel():
    num_pages = cute.sym_int()

    k_cache = _fake(cute.Int8, (num_pages, PAGE_SIZE, 1, ROW_STRIDE), (3, 2, 1, 0), 16)
    q       = _fake(cute.Float8E4M3FN, (N, HEAD_DIM), (1, 0), 16)
    c_out   = _fake(cute.Float32, (M, N), (1, 0), 16)
    stream  = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = ScoreScaleSIMT()
    compiled = cute.compile(kernel, k_cache, q, c_out, stream, options="--enable-tvm-ffi")
    return kernel, compiled


_kernel, _compiled = compile_kernel()


# ── Reference: match idxer_tc.py dequant exactly ────────────────────────────
def dequant_flat(k_index_cache_fp8):
    """Flat contiguous dequant — same as idxer_tc.py."""
    k_u8 = k_index_cache_fp8.view(torch.uint8)
    np_, ps_, _, hdsf_ = k_u8.shape
    hd_ = hdsf_ - 4

    kv_flat = k_u8.view(np_, ps_ * hdsf_)
    fp8_bytes = kv_flat[:, :ps_ * hd_].contiguous()
    fp8_tensor = fp8_bytes.view(np_, ps_, hd_).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, ps_ * hd_:].contiguous()
    scale = scale_bytes.view(np_, ps_, 4).view(torch.float32)

    return fp8_float * scale   # [np, 64, 128] already scaled


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda"
    num_pages = M // PAGE_SIZE   # 32

    # Generate data EXACTLY like idx_utils.make_tensors
    q_fp8 = torch.randn(N, HEAD_DIM, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    k_index_cache_fp8 = torch.randint(
        0, 256, (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
        dtype=torch.uint8, device=device,
    ).view(torch.int8)

    c_out = torch.zeros((M, N), device=device, dtype=torch.float32)

    # ── Run kernel ──
    _compiled(k_index_cache_fp8, q_fp8, c_out)
    torch.cuda.synchronize()

    # ── Reference (idxer_tc.py flat dequant) ──
    K_all = dequant_flat(k_index_cache_fp8)      # [np, 64, 128] already scaled
    K_flat = K_all.reshape(M, HEAD_DIM)           # [M, 128]
    ref_c = K_flat @ q_fp8.float().T              # [M, N]

    # ── Diagnostics ──
    c_nan = c_out.isnan().sum().item()
    ref_nan = ref_c.isnan().sum().item()
    total = c_out.numel()
    print(f"\nNaN: kernel={c_nan}/{total} ({100*c_nan/total:.1f}%)  ref={ref_nan}/{total} ({100*ref_nan/total:.1f}%)")

    # Check NaN agreement per-row (a whole row should be NaN or not)
    kern_row_nan = c_out.isnan().any(dim=1)   # [M]
    ref_row_nan = ref_c.isnan().any(dim=1)    # [M]
    both_nan_rows = (kern_row_nan & ref_row_nan).sum().item()
    kern_only_nan = (kern_row_nan & ~ref_row_nan).sum().item()
    ref_only_nan = (~kern_row_nan & ref_row_nan).sum().item()
    print(f"Row-level NaN: both={both_nan_rows}  kern_only={kern_only_nan}  ref_only={ref_only_nan}")

    # Diagnose ref_only_nan rows: WHY does ref have NaN but kernel doesn't?
    ref_only_mask = ~kern_row_nan & ref_row_nan
    if ref_only_mask.any():
        ref_only_indices = ref_only_mask.nonzero(as_tuple=True)[0]
        print(f"\n--- Diagnosing {ref_only_indices.shape[0]} ref_only_nan rows ---")
        for idx in ref_only_indices[:5]:
            m = idx.item()
            page_id = m // PAGE_SIZE
            tok = m % PAGE_SIZE

            # Check the dequanted K row (pre-multiplied)
            K_row = K_flat[m]  # [128] fp8*scale
            n_nan = K_row.isnan().sum().item()
            n_inf = K_row.isinf().sum().item()
            n_fin = K_row.isfinite().sum().item()

            # Check raw fp8 and scale
            k_u8 = k_index_cache_fp8.view(torch.uint8)
            kv_flat_raw = k_u8.view(num_pages, PAGE_SIZE * (HEAD_DIM + 4))
            fp8_raw = kv_flat_raw[page_id, tok*HEAD_DIM:(tok+1)*HEAD_DIM].view(torch.float8_e4m3fn).float()
            scale_off = PAGE_SIZE * HEAD_DIM + tok * 4
            scale_raw = kv_flat_raw[page_id, scale_off:scale_off+4].contiguous().view(torch.float32).item()

            fp8_nan_cnt = fp8_raw.isnan().sum().item()
            fp8_inf_cnt = fp8_raw.isinf().sum().item()

            # Kernel output for this row
            kern_row = c_out[m]
            kern_nan_cnt = kern_row.isnan().sum().item()
            kern_inf_cnt = kern_row.isinf().sum().item()

            print(f"  row {m}: scale={scale_raw}  fp8_nan={fp8_nan_cnt}  fp8_inf={fp8_inf_cnt}")
            print(f"    K_row(fp8*scale): nan={n_nan} inf={n_inf} fin={n_fin}")
            print(f"    ref output: nan={ref_c[m].isnan().sum().item()} inf={ref_c[m].isinf().sum().item()}")
            print(f"    kern output: nan={kern_nan_cnt} inf={kern_inf_cnt}")
            # Check if Inf * 0 = NaN scenario
            q_row = q_fp8.float()  # [N, 128]
            for h in range(min(3, N)):
                dot_pre = (K_row * q_row[h]).sum().item()  # ref way
                dot_post_raw = (fp8_raw * q_row[h]).sum().item()
                dot_post = dot_post_raw * scale_raw
                print(f"    head {h}: pre_dot={dot_pre}  post_raw={dot_post_raw}  post={dot_post}")

    # Compare only rows where BOTH are fully finite
    both_row_fin = ~kern_row_nan & ~ref_row_nan   # [M]
    fin_rows = both_row_fin.sum().item()
    if fin_rows > 0:
        c_fin = c_out[both_row_fin]    # [fin_rows, N]
        r_fin = ref_c[both_row_fin]
        diff = (c_fin - r_fin).abs()
        # Filter to rows where values aren't astronomically large (garbage random scales)
        sane_mask = r_fin.abs().max(dim=1).values < 1e20
        sane_rows = sane_mask.sum().item()

        print(f"\nFully-finite rows: {fin_rows}/{M}  (sane scale: {sane_rows})")

        if sane_rows > 0:
            c_sane = c_fin[sane_mask]
            r_sane = r_fin[sane_mask]
            d_sane = (c_sane - r_sane).abs()
            rel_sane = d_sane / (r_sane.abs().clamp(min=1e-6))
            print(f"  Sane rows: abs max_err={d_sane.max().item():.6f}  mean={d_sane.mean().item():.6f}")
            print(f"  Sane rows: rel max_err={rel_sane.max().item():.6f}  mean={rel_sane.mean().item():.6f}")

        # Show some finite rows
        fin_indices = both_row_fin.nonzero(as_tuple=True)[0]
        for i in range(min(8, fin_rows)):
            m = fin_indices[i].item()
            k_v = c_out[m, 0].item()
            r_v = ref_c[m, 0].item()
            d = abs(k_v - r_v)
            print(f"  row {m:4d}: kern={k_v:16.4f}  ref={r_v:16.4f}  diff={d:.6f}")
    else:
        print("No fully-finite rows!")
        sane_rows = 0

    # Overall correctness
    if sane_rows > 0:
        sane_rel_max = rel_sane.max().item()
        if sane_rel_max < 0.01:
            print(f"\nCORRECTNESS: PASS (sane rel_max={sane_rel_max:.8f})")
        else:
            print(f"\nCORRECTNESS: FAIL (sane rel_max={sane_rel_max:.6f})")
    else:
        print("\nCORRECTNESS: FAIL (no sane finite rows)")


if __name__ == "__main__":
    main()
