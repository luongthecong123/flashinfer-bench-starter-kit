"""
fp8_gemm_reduce: fused FP8 GEMM + ReLU + head-reduction kernel.

Problem:
  q_fp8    : [64, 128]   float8_e4m3fn   (query, 64 heads × 128 dims)
  K_fp8    : [2048, 128] float8_e4m3fn   (keys,  2048 tokens × 128 dims)
  K_scales : [2048]      float32         (per-token dequant scale)
  weights  : [64]        float32         (per-head weight for reduction)

Output:
  scores   : [2048]      float32

Math per output token t:
  scores[t] = sum_h( relu( dot(q_fp8[h], K_fp8[t]) * K_scales[t] ) * weights[h] )

Grid  : [NUM_TOKENS // TOKEN_TILE, 1, 1]      (one block per tile of output tokens)
Block : [BLOCK_THREADS, 1, 1]
"""
import cutlass
import cutlass.cute as cute
import math
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_HEADS   = 64
HEAD_DIM    = 128
NUM_TOKENS  = 2048          # static seq_len
TOKEN_TILE  = 32            # tokens processed per block
BLOCK_THREADS = 256
NUM_WARPS   = BLOCK_THREADS // 32   # 8


# ── Helper: warp-level reduction ──────────────────────────────────────────────
@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ══════════════════════════════════════════════════════════════════════════════
# Kernel class
# ══════════════════════════════════════════════════════════════════════════════

class FP8GemmReduce:
    def __init__(self):
        self.num_heads    = NUM_HEADS
        self.head_dim     = HEAD_DIM
        self.num_tokens   = NUM_TOKENS
        self.token_tile   = TOKEN_TILE
        self.block_threads = BLOCK_THREADS
        self.num_warps    = NUM_WARPS

    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(
            dtype, cute.make_layout(shape, stride=stride), align, None)

    # ── Top-level launcher ────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        q_fp8:    cute.Tensor,   # [64, 128]   float8_e4m3fn
        K_fp8:    cute.Tensor,   # [2048, 128] float8_e4m3fn
        K_scales: cute.Tensor,   # [2048]      float32
        weights:  cute.Tensor,   # [64]        float32
        scores:   cute.Tensor,   # [2048]      float32  (output)
        stream,
    ):
        num_blocks = self.num_tokens // self.token_tile
        self.score_kernel(
            q_fp8, K_fp8, K_scales, weights, scores,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Main kernel: one block handles TOKEN_TILE output tokens
    #
    # For each token t in [block_start, block_start + TOKEN_TILE):
    #   1. Load K_fp8[t]  and K_scales[t] from gmem
    #   2. Load q_fp8[h]  for all h         from gmem  (or smem cache)
    #   3. dot  = sum_d( float(q[h,d]) * float(K[t,d]) )  -- GEMM row
    #   4. contrib = relu(dot * K_scales[t]) * weights[h]
    #   5. scores[t] = sum_h( contrib[h] )               -- head reduction
    # ══════════════════════════════════════════════════════════════════════════

    @cute.kernel
    def score_kernel(
        self,
        q_fp8:    cute.Tensor,
        K_fp8:    cute.Tensor,
        K_scales: cute.Tensor,
        weights:  cute.Tensor,
        scores:   cute.Tensor,
    ):
        num_heads:     cutlass.Constexpr = self.num_heads      # 64
        head_dim:      cutlass.Constexpr = self.head_dim       # 128
        token_tile:    cutlass.Constexpr = self.token_tile     # 32
        num_threads:   cutlass.Constexpr = self.block_threads  # 256
        num_warps:     cutlass.Constexpr = self.num_warps      # 8

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        wsize    = cute.arch.WARP_SIZE

        # Global token range this block owns
        tok_start = bidx * token_tile

        # ── SMEM allocation ──────────────────────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()

        # Cache q in smem: [num_heads, head_dim] float32
        smem_q = self._smem(alloc, cutlass.Float32,
                            (num_heads, head_dim), (head_dim, 1), 16)

        # Cache weights in smem: [num_heads] float32
        smem_w = self._smem(alloc, cutlass.Float32,
                            (num_heads,), (1,), 16)

        # Per-token partial score accumulator in smem: [token_tile] float32
        smem_scores = self._smem(alloc, cutlass.Float32,
                                 (token_tile,), (1,), 16)

        # Warp reduction scratch: [num_warps] float32
        smem_red = self._smem(alloc, cutlass.Float32,
                              (num_warps,), (1,), 16)

        # ── Phase 1: load q_fp8 → smem_q (fp8 → fp32) ───────────────────────
        # Each thread loads elements strided across the [num_heads * head_dim] flat space
        #
        # TODO: load q_fp8[h, d] → smem_q[h, d] (fp8-to-fp32 cast per element)
        #       tiling: threads stride over num_heads * head_dim elements
        #
        # for i in range(tidx, num_heads * head_dim, num_threads):
        #     h = i // head_dim
        #     d = i %  head_dim
        #     smem_q[h, d] = cutlass.Float32(q_fp8[h, d])

        # ── Phase 2: load weights → smem_w ───────────────────────────────────
        #
        # TODO: load weights[h] → smem_w[h]
        #       tiling: threads stride over num_heads elements
        #
        # for i in range(tidx, num_heads, num_threads):
        #     smem_w[i] = weights[i]

        cute.arch.sync_threads()

        # ── Phase 3: init smem_scores to 0 ───────────────────────────────────
        #
        # TODO: zero out smem_scores[0..token_tile)
        #
        # for i in range(tidx, token_tile, num_threads):
        #     smem_scores[i] = cutlass.Float32(0)

        cute.arch.sync_threads()

        # ── Phase 4: per-token GEMM + ReLU + weighted head-reduction ─────────
        #
        # Assign one token per warp (token_tile / num_warps rounds if token_tile > num_warps).
        # Each warp's lanes cover the head_dim dot-product in parallel.
        #
        # TODO: implement the main compute loop
        #
        # for round_idx in range(token_tile // num_warps):
        #     tok_local = round_idx * num_warps + warp_idx   # token within tile
        #     tok_global = tok_start + tok_local
        #     scale = K_scales[tok_global]
        #
        #     acc_score = cutlass.Float32(0)
        #     for h in range(num_heads):
        #         # dot product over head_dim, each lane covers a stripe
        #         dot = cutlass.Float32(0)
        #         for d in range(lane_idx, head_dim, wsize):
        #             dot += smem_q[h, d] * cutlass.Float32(K_fp8[tok_global, d])
        #
        #         # warp-reduce dot across lanes
        #         dot = warp_reduce(dot, lambda a, b: a + b, width=32)
        #
        #         # ReLU, scale, weight — only lane 0 has the final sum
        #         if lane_idx == 0:
        #             acc_score += cute.math.max(dot * scale, cutlass.Float32(0)) * smem_w[h]
        #
        #     # lane 0 writes the token score
        #     if lane_idx == 0:
        #         smem_scores[tok_local] = acc_score

        cute.arch.sync_threads()

        # ── Phase 5: write smem_scores → global scores ───────────────────────
        #
        # TODO: store smem_scores[i] → scores[tok_start + i]
        #
        # for i in range(tidx, token_tile, num_threads):
        #     scores[tok_start + i] = smem_scores[i]


# ── Python wrapper (matches idxer_gemm interface) ─────────────────────────────

_instance = None

def compute_scores(
    q_fp8:    torch.Tensor,   # [64, 128]   float8_e4m3fn
    K_fp8:    torch.Tensor,   # [2048, 128] float8_e4m3fn
    K_scales: torch.Tensor,   # [2048]      float32
    weights:  torch.Tensor,   # [64]        float32
) -> torch.Tensor:
    global _instance
    if _instance is None:
        _instance = FP8GemmReduce()

    scores = torch.empty(NUM_TOKENS, dtype=torch.float32, device=q_fp8.device)

    from cutlass.cute.runtime import make_fake_stream
    stream = make_fake_stream(torch.cuda.current_stream().cuda_stream)

    q_ct    = cute.from_dlpack(q_fp8)
    K_ct    = cute.from_dlpack(K_fp8)
    Ks_ct   = cute.from_dlpack(K_scales)
    w_ct    = cute.from_dlpack(weights)
    out_ct  = cute.from_dlpack(scores)

    _instance(q_ct, K_ct, Ks_ct, w_ct, out_ct, stream)
    return scores
