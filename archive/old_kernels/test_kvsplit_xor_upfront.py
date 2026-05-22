"""Standalone correctness test for the 'upfront' phase of kv_split_xor.

The upfront phase (Load sparse_indices and calculate OOB tiles up front) does:
  1. Every group of `sparse_thr_per_T` threads cooperatively loads
     sparse_indices[wg_per_T_idx] into SMEM and counts non-negative entries.
  2. A two-level warp-reduce accumulates the per-warp counts.
  3. The total is written to smem_num_valid[wg_per_T_idx].

This file compiles a stripped-down kernel that ONLY performs this phase and
writes smem_num_valid back to global memory, so we can compare against the
CPU reference:  (sparse_indices >= 0).sum(dim=1)

XID-13 root cause & fix
-----------------------
`cute.arch.barrier(barrier_id=wg_per_T_idx, ...)` with wg_per_T_idx ∈ {0,1,2,3}
hits barrier ID 0, which is **reserved** for `sync_threads()` (bar.sync 0).
Using it as a named barrier corrupts the implied thread-count contract and
triggers "Illegal Instruction Parameter".

Fix: use `NamedBarrier` objects with compile-time IDs 1-4 (one per thread-group)
dispatched via a Python-level if-chain so each branch contains a constant ID.

Usage (local, no GPU required for compilation):
    python src/kernels/test_kvsplit_xor_upfront.py

Usage (GPU):
    modal run src/modal/test_kvsplit_xor_upfront.py
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

# ── Constants (must match kv_split_xor.py) ────────────────────────────────────
NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX    = 8
NUM_SPLITS = 8


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def count_valid_indices(
    sparse_indices:  cute.Tensor,       # (T, top_k_len) i32  — global
    smem_sparse:     cute.Tensor,       # (T_max, top_k_len) i32 — smem cache
    smem_red_i32:    cute.Tensor,       # (T_max, 32) i32     — smem scratch
    smem_num_valid:  cute.Tensor,       # (T_max,) i32        — smem output
    T:               cute.Numeric,
    tidx:            cute.Numeric,      # flat thread index (threadIdx.x)
    warp_idx:        cute.Numeric,      # warp index (warp-uniform)
    top_k_len:       cutlass.Constexpr,
    sparse_thr_per_T: cutlass.Constexpr,
    num_warps_per_T: cutlass.Constexpr,
) -> None:
    """Load sparse_indices into smem_sparse and count non-negative entries
    in the same loop (count uses local register, no extra barrier needed),
    then two-level warp-reduce stores the total in smem_num_valid[wg_per_T_idx].

    Uses barrier_id = wg_per_T_idx + 1 so ID 0 stays reserved for sync_threads.
    """
    thr_idx_per_T  = tidx % sparse_thr_per_T
    lane_idx_per_T = thr_idx_per_T % cute.arch.WARP_SIZE
    wg_per_T_idx   = tidx // sparse_thr_per_T
    warp_per_T_idx = warp_idx % num_warps_per_T

    partial_cnt = 0
    if wg_per_T_idx < T:
        # Load into smem and count using the local register — no race
        for i in range(thr_idx_per_T, top_k_len, sparse_thr_per_T):
            idx = sparse_indices[wg_per_T_idx, i]
            smem_sparse[wg_per_T_idx, i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt += 1

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx_per_T == 0:
            smem_red_i32[wg_per_T_idx, warp_per_T_idx] = cnt_sum

        # Barrier 1: all warps in this group have written their partial sum
        cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                          number_of_threads=sparse_thr_per_T)

        # Level-2: warp 0 of each group reduces the per-warp sums
        if warp_per_T_idx == 0:
            val     = smem_red_i32[wg_per_T_idx, lane_idx_per_T]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps_per_T)
            smem_red_i32[wg_per_T_idx, 0] = cnt_sum

        # Barrier 2: warp 0 has committed the group total
        cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                          number_of_threads=sparse_thr_per_T)

        # All threads read the same value — any one of them can write back
        smem_num_valid[wg_per_T_idx] = smem_red_i32[wg_per_T_idx, 0]


# ═══════════════════════════════════════════════════════════════════════════════
# Stripped kernel: upfront phase only
# ═══════════════════════════════════════════════════════════════════════════════

class UpfrontTest:
    def __init__(self):
        self.top_k_len        = TOP_K_LEN
        self.T_max            = T_MAX
        self.num_threads      = 1024
        self.wsize            = cute.arch.WARP_SIZE
        self.sparse_thr_per_T = 128                                   # same as kv_split_xor
        self.num_warps_per_T  = self.sparse_thr_per_T // self.wsize   # 4

    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)

    @cute.jit
    def __call__(
        self,
        sparse_indices: cute.Tensor,   # (T, TOP_K_LEN)  i32
        num_valid_out:  cute.Tensor,   # (T_MAX,)         i32  — receives smem_num_valid
        stream,
    ):
        self.kernel(sparse_indices, num_valid_out).launch(
            grid=[1, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        sparse_indices: cute.Tensor,   # (T, TOP_K_LEN)  i32
        num_valid_out:  cute.Tensor,   # (T_MAX,)         i32
    ):
        T, _ = sparse_indices.shape

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.warp_idx()
        warp_idx   = cute.arch.make_warp_uniform(warp_idx)

        alloc          = cutlass.utils.SmemAllocator()
        smem_sparse    = self._smem(alloc, cutlass.Int32,
                                    (self.T_max, self.top_k_len), (self.top_k_len, 1), 4)
        smem_red_i32   = self._smem(alloc, cutlass.Int32,
                                    (self.T_max, 32), (32, 1), 4)
        smem_num_valid = self._smem(alloc, cutlass.Int32,
                                    (self.T_max,),    (1,),     4)

        count_valid_indices(
            sparse_indices, smem_sparse, smem_red_i32, smem_num_valid,
            T, tidx, warp_idx,
            self.top_k_len, self.sparse_thr_per_T, self.num_warps_per_T,
        )

        cute.arch.sync_threads()

        wg_per_T_idx  = tidx // self.sparse_thr_per_T
        thr_idx_per_T = tidx % self.sparse_thr_per_T
        if wg_per_T_idx < T and thr_idx_per_T == 0:
            num_valid_out[wg_per_T_idx] = smem_num_valid[wg_per_T_idx]


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_upfront_test():
    T = cute.sym_int()
    sparse_indices = _fake(cute.Int32, (T, TOP_K_LEN), (1, 0), 4)
    num_valid_out  = _fake(cute.Int32, (T_MAX,),        (0,),   4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    test_obj = UpfrontTest()
    return cute.compile(
        test_obj,
        sparse_indices, num_valid_out, stream,
        options="--enable-tvm-ffi",
    )


_compiled = compile_upfront_test()


# ── Run helper ─────────────────────────────────────────────────────────────────

def run_upfront_test(sparse_indices: torch.Tensor) -> torch.Tensor:
    """Run the GPU upfront phase and return num_valid tensor (CPU).

    Args:
        sparse_indices: (T, TOP_K_LEN) int32 on CUDA.  Non-negative values are
                        valid token indices; -1 (or any negative) marks padding.

    Returns:
        num_valid: (T,) int32 on CPU with the GPU-computed valid counts.
    """
    T = sparse_indices.shape[0]
    assert T <= T_MAX, f"T={T} exceeds T_MAX={T_MAX}"
    num_valid_out = torch.zeros(T_MAX, dtype=torch.int32, device=sparse_indices.device)
    _compiled(sparse_indices, num_valid_out)
    torch.cuda.synchronize()
    return num_valid_out[:T].cpu()


# ── Self-contained correctness check ──────────────────────────────────────────

def check(sparse_indices_cpu: torch.Tensor, label: str = "") -> bool:
    """Compare GPU upfront result against CPU reference for one workload.

    Returns True if all counts match exactly.
    """
    si_gpu   = sparse_indices_cpu.cuda()
    T        = si_gpu.shape[0]
    gpu_out  = run_upfront_test(si_gpu)                        # shape (T,)
    cpu_ref  = (sparse_indices_cpu >= 0).sum(dim=1).int()     # shape (T,)

    match    = (gpu_out == cpu_ref).all().item()
    tag      = label or "workload"
    if match:
        print(f"  PASS  {tag}  gpu={gpu_out.tolist()}  ref={cpu_ref.tolist()}")
    else:
        print(f"  FAIL  {tag}")
        print(f"    GPU : {gpu_out.tolist()}")
        print(f"    CPU : {cpu_ref.tolist()}")
        diff = (gpu_out - cpu_ref).abs()
        print(f"    diff: {diff.tolist()}")
    return match


def run_all_checks():
    """Run several synthetic workloads to exercise the upfront logic."""
    print("=" * 60)
    print("UpfrontTest: smem_num_valid correctness check")
    print("=" * 60)
    all_pass = True

    # ── Case 1: all valid (T=4) ────────────────────────────────────────────
    T = 4
    si = torch.randint(0, 8462 * 64, (T, TOP_K_LEN), dtype=torch.int32)
    all_pass &= check(si, "all-valid T=4")

    # ── Case 2: all padding (-1) ───────────────────────────────────────────
    si_pad = torch.full((T, TOP_K_LEN), -1, dtype=torch.int32)
    all_pass &= check(si_pad, "all-padding T=4")

    # ── Case 3: half valid (alternating) ──────────────────────────────────
    si_half = torch.full((T, TOP_K_LEN), -1, dtype=torch.int32)
    si_half[:, ::2] = torch.randint(0, 8462 * 64, (T, TOP_K_LEN // 2))
    all_pass &= check(si_half, "half-valid T=4")

    # ── Case 4: T=T_MAX (stress all wg_per_T_idx groups) ─────────────────
    T = T_MAX
    si_full = torch.randint(-1, 8462 * 64, (T, TOP_K_LEN), dtype=torch.int32)
    all_pass &= check(si_full, f"random T={T}")

    # ── Case 5: T=1 (single token) ────────────────────────────────────────
    si_one = torch.randint(-1, 8462 * 64, (1, TOP_K_LEN), dtype=torch.int32)
    all_pass &= check(si_one, "T=1")

    print("=" * 60)
    print("Overall:", "ALL PASS" if all_pass else "SOME FAILED")
    return all_pass


if __name__ == "__main__":
    ok = run_all_checks()
    raise SystemExit(0 if ok else 1)
