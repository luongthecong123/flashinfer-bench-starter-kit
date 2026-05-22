"""fused_persistent_v3_shell.py — diagnostic shell using the real CUTLASS tile scheduler.

Uses the actual utils.StaticPersistentTileScheduler API (same as v3b).
Each CTA logs: (sm_id, flat_idx, tok, split, head, local_valid) per task.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T as MLIR_T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cutlass.utils as utils

import math
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_HEADS  = 16
D_CKV      = 512
D_KPE      = 64
DIM_SPLIT  = 256
TOP_K      = 2048
NUM_SPLITS = TOP_K // DIM_SPLIT   # 8
T_MAX      = 8
N_SH       = NUM_SPLITS * NUM_HEADS  # 128
TOTAL_TASKS: cutlass.Constexpr = T_MAX * NUM_SPLITS * NUM_HEADS  # 1024

MAX_ACTIVE_CLUSTERS = 148
NUM_CTAS            = 148
BLOCK_SIZE_SHELL    = 32   # tiny — just recording

MAX_TASKS_PER_CTA: cutlass.Constexpr = 8   # ceil(1024/148) = 7, +1 buffer
PROBE_COLS: cutlass.Constexpr = 6          # sm_id, flat_idx, tok, split, head, local_valid

N_PAGES_FLAT: cutlass.Constexpr = 8462 * 64

# ── PTX: read hardware SM ID ─────────────────────────────────────────────────
@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(MLIR_T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int32(t)


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-pass: count valid entries per token (identical to v3b)
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.kernel
def valid_count_kernel_shell(
    sparse_indices: cute.Tensor,
    global_valid_count: cute.Tensor,
):
    tok = cute.arch.block_idx()[0]
    tidx, _, _ = cute.arch.thread_idx()
    num_threads_k: cutlass.Constexpr = 1024
    num_warps_k:   cutlass.Constexpr = 32
    top_k:         cutlass.Constexpr = TOP_K
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    allocator = cutlass.utils.SmemAllocator()
    smem_red = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((32,), stride=(1,)), 4, None)

    cnt = 0
    for i in range(tidx, top_k, num_threads_k):
        if sparse_indices[tok, i] >= cutlass.Int32(0):
            cnt += 1

    s = warp_reduce(cnt, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_red[warp_idx] = s
    cute.arch.sync_threads()

    if warp_idx == 0:
        val = smem_red[lane_idx]
        s = warp_reduce(val, lambda a, b: a + b, width=num_warps_k)
        if lane_idx == 0:
            global_valid_count[tok] = s


# ═══════════════════════════════════════════════════════════════════════════════
# Shell kernel: real CUTLASS StaticPersistentTileScheduler, just logs work
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def persistent_shell_kernel(
    global_valid_count: cute.Tensor,
    probe: cute.Tensor,
    probe_count: cute.Tensor,
    actual_T: cutlass.Int32,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    dim_split:  cutlass.Constexpr = DIM_SPLIT
    num_splits: cutlass.Constexpr = NUM_SPLITS
    n_sh:       cutlass.Constexpr = NUM_SPLITS * NUM_HEADS
    max_tasks:  cutlass.Constexpr = MAX_TASKS_PER_CTA

    tidx, _, _ = cute.arch.thread_idx()
    bidx_z = cute.arch.block_idx()[2]
    sm_id  = smid_u32()

    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    task_cnt = cutlass.Int32(0)

    while work_tile.is_valid_tile:
        flat_idx = work_tile.tile_idx[0]

        tok   =  flat_idx // n_sh
        split = (flat_idx // NUM_HEADS) % num_splits
        head  =  flat_idx % NUM_HEADS
        split_start = split * dim_split

        local_valid = cutlass.Int32(0)
        if tok < actual_T:
            valid_cnt   = global_valid_count[tok]
            local_valid = valid_cnt - split_start
            if local_valid > dim_split:
                local_valid = dim_split
            if local_valid < cutlass.Int32(0):
                local_valid = cutlass.Int32(0)

        if tidx == 0:
            if task_cnt < max_tasks:
                probe[bidx_z, task_cnt, 0] = sm_id
                probe[bidx_z, task_cnt, 1] = flat_idx
                probe[bidx_z, task_cnt, 2] = tok
                probe[bidx_z, task_cnt, 3] = split
                probe[bidx_z, task_cnt, 4] = head
                probe[bidx_z, task_cnt, 5] = local_valid
                task_cnt = task_cnt + 1

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()

    if tidx == 0:
        probe_count[bidx_z] = task_cnt


# ═══════════════════════════════════════════════════════════════════════════════
# JIT launcher
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def shell_launcher(
    sparse_indices: cute.Tensor,
    global_valid_count: cute.Tensor,
    probe: cute.Tensor,
    probe_count: cute.Tensor,
    actual_T: cutlass.Int32,
    max_active_clusters: cutlass.Constexpr,
    stream,
):
    T_dim = sparse_indices.shape[0]

    valid_count_kernel_shell(sparse_indices, global_valid_count).launch(
        grid=[T_dim, 1, 1], block=[1024, 1, 1], stream=stream)

    total_tasks: cutlass.Constexpr = TOTAL_TASKS
    cluster_shape_mnl = (1, 1, 1)
    num_ctas_mnl = (total_tasks, 1, 1)

    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl, cluster_shape_mnl, swizzle_size=1, raster_along_m=True,
    )
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    persistent_shell_kernel(
        global_valid_count, probe, probe_count, actual_T,
        tile_sched_params,
    ).launch(grid=grid, block=[BLOCK_SIZE_SHELL, 1, 1], stream=stream)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_shell():
    T_sym = cute.sym_int()

    sparse_indices     = _fake(cute.Int32, (T_sym, TOP_K), (1, 0), 4)
    global_valid_count = _fake(cute.Int32, (T_MAX,), (0,), 4)
    probe              = _fake(cute.Int32, (MAX_ACTIVE_CLUSTERS, MAX_TASKS_PER_CTA, PROBE_COLS),
                               (2, 1, 0), 4)
    probe_count        = _fake(cute.Int32, (MAX_ACTIVE_CLUSTERS,), (0,), 4)
    actual_T           = cutlass.Int32(0)
    stream             = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        shell_launcher,
        sparse_indices, global_valid_count, probe, probe_count, actual_T,
        MAX_ACTIVE_CLUSTERS, stream,
        options="--enable-tvm-ffi",
    )


_compiled = compile_shell()


def run_shell(sparse_indices, actual_T):
    """Run shell and return (probe, probe_count, global_valid_count)."""
    device = sparse_indices.device
    global_valid_count = torch.empty(T_MAX, dtype=torch.int32, device=device)
    probe = torch.zeros(MAX_ACTIVE_CLUSTERS, MAX_TASKS_PER_CTA, PROBE_COLS,
                        dtype=torch.int32, device=device)
    probe_count = torch.zeros(MAX_ACTIVE_CLUSTERS, dtype=torch.int32, device=device)
    _compiled(sparse_indices, global_valid_count, probe, probe_count, actual_T)
    torch.cuda.synchronize()
    return probe, probe_count, global_valid_count
