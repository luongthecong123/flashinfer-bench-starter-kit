"""gemv_kvsplit_v1: GEMV with K-split across cluster blocks + DSMEM pull reduce.

Problem:
  logit_scaled [K]     fp32   (attention weights, post-softmax, 1D)
  V            [K, D]  bf16
  output       [D]     fp32

  output[d] = sum_{k=0}^{K-1} logit_scaled[k] * V[k, d]

Parallelism (4-level reduction hierarchy):
  Level 1 — Register : each thread accumulates partial dot over K_PER_SPLIT K-values
  Level 2 — Warp     : (none needed — each thread fully owns its output dim)
  Level 3 — Block    : (none needed — each thread fully owns its output dim)
  Level 4 — Cluster  : DSMEM pull: block 0 reads peer smem after cluster_arrive/wait

Grid:    [CLUSTER_N, 1, 1]   — one block per K-split
Cluster: [CLUSTER_N, 1, 1]   — all K-split blocks co-scheduled on same GPC
Block:   [D, 1, 1]            — 1 thread per output dim (D=512 threads)

Thread t owns output dim t, accumulates over k in [split_id*K_PER_SPLIT, (split_id+1)*K_PER_SPLIT).

After all blocks write their partial to smem_out:
  cluster_arrive()  — fence + signal
  cluster_wait()    — wait for all cluster peers
  block 0 pulls peers' smem via mapa(smem_out_ptr, peer) + load(ss='cluster')
  block 0 writes final output[d]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import torch

K         = 2048
D         = 512
CLUSTER_N = 4
K_PER_SPLIT = K // CLUSTER_N   # 512  — each block processes this many K rows
BLOCK_SIZE  = D                 # 512  — one thread per output dim


@cute.jit
def gemv_kvsplit(
    logit:  cute.Tensor,   # [K]     fp32
    V:      cute.Tensor,   # [K, D]  bf16
    output: cute.Tensor,   # [D]     fp32
    stream,
):
    gemv_kvsplit_kernel(logit, V, output).launch(
        grid=[CLUSTER_N, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        cluster=[CLUSTER_N, 1, 1],
        stream=stream,
    )


@cute.kernel
def gemv_kvsplit_kernel(
    logit:  cute.Tensor,   # [K]     fp32
    V:      cute.Tensor,   # [K, D]  bf16
    output: cute.Tensor,   # [D]     fp32
):
    cluster_n:   cutlass.Constexpr = CLUSTER_N
    k_per_split: cutlass.Constexpr = K_PER_SPLIT
    block_size:  cutlass.Constexpr = BLOCK_SIZE

    tidx, _, _  = cute.arch.thread_idx()
    split_id    = cute.arch.block_idx_in_cluster()   # Int32 in [0, cluster_n)

    # ── Smem allocation ────────────────────────────────────────────────────
    # Each block owns smem_out_ptr[D] fp32.
    # All cluster blocks allocate at the same relative smem offset.
    # After cluster sync, block 0 reads peer[i]'s smem_out via mapa(smem_out_ptr, i).
    allocator    = cutlass.utils.SmemAllocator()
    smem_out_ptr = allocator.allocate_array(cutlass.Float32, num_elems=block_size)

    # ── Level 1: Register accumulation ────────────────────────────────────
    # Thread tidx computes:
    #   acc = sum_{k_off in [0, k_per_split)} logit[split_id*k_per_split + k_off] * V[...k..., tidx]
    k_start = split_id * k_per_split   # runtime Int32
    acc = cutlass.Float32(0)
    for k_off in range(k_per_split):   # compile-time loop, k_per_split=512
        k   = k_start + k_off
        acc = acc + logit[k] * cutlass.Float32(V[k, tidx])

    # ── Level 2/3: Write partial to smem (each thread owns 1 dim) ─────────
    # No warp/block reduction needed — thread t is the sole owner of dim t.
    cute.arch.store(smem_out_ptr + tidx, acc, ss='cta')

    # ── Level 4: Cluster DSMEM reduce ─────────────────────────────────────
    cute.arch.sync_threads()      # ensure all threads in this block wrote smem_out
    cute.arch.cluster_arrive()    # fence (release) — make smem writes visible to cluster
    cute.arch.cluster_wait()      # wait for all cluster peers to arrive

    # Only block 0 performs the final cross-split summation.
    if split_id == cutlass.Int32(0):
        # Re-read own partial from smem
        final = cute.arch.load(smem_out_ptr + tidx, cutlass.Float32, ss='cta')

        # Pull from each peer block via DSMEM.
        # Key: pass the ALREADY-ADVANCED cute.ptr into mapa, so no pointer arithmetic
        # is needed on the ir.Value that mapa returns.
        # mapa(smem_out_ptr + tidx, peer) → ir.Value pointing to peer's smem[tidx].
        # load(ir_value, dtype, ss='cluster') reads directly from that DSMEM address.
        for peer in range(1, cluster_n):   # compile-time unrolled (cluster_n=4)
            peer_element_ptr = cute.arch.mapa(smem_out_ptr + tidx, peer)
            peer_val         = cute.arch.load(peer_element_ptr, cutlass.Float32, ss='cluster')
            final            = final + peer_val

        output[tidx] = final

    # ── 2nd cluster sync: keep blocks 1..3 alive until block 0 is done reading ─
    # Without this, blocks 1..3 exit and their smem is freed before block 0
    # finishes the DSMEM pull above — use-after-free crash.
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_gemv_kvsplit():
    logit  = _fake(cute.Float32,   (K,),    (0,),    16)
    V      = _fake(cute.BFloat16,  (K, D),  (1, 0),  16)
    output = _fake(cute.Float32,   (D,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        gemv_kvsplit,
        logit, V, output, stream,
        options="--enable-tvm-ffi",
    )


gemv_kvsplit_compiled = compile_gemv_kvsplit()


def run(logit: torch.Tensor, V: torch.Tensor, output: torch.Tensor):
    """logit: [K] fp32, V: [K, D] bf16, output: [D] fp32 (pre-allocated)."""
    gemv_kvsplit_compiled(logit, V, output)
