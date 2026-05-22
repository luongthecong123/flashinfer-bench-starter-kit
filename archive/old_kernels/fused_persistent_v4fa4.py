"""fused_persistent_v4fa4.py — persistent kernel with FA4-style inlined tile scheduler.

Inlines the Flash Attention 4 StaticPersistentTileScheduler (Tri Dao) directly,
replacing cutlass.utils.PersistentTileSchedulerParams with:
  - FastDivmod for fast integer division (precomputed multiplier + shift)
  - SM-count-based grid sizing via HardwareInfo
  - Direct tile_idx += grid_dim advancement

Otherwise identical compute logic to fused_persistent_v3.

Grid (pre-pass):  [T, 1, 1]              Block: 1024 threads
Grid (compute):   persistent, min(SM_COUNT, total_tasks) CTAs, Block: 1024 threads
Grid (reduce):    [T, H, 1]              Block: 512 threads
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass import Int32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cutlass.utils as utils

import math
import torch
from dataclasses import dataclass, fields

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_HEADS  = 16
D_CKV      = 512
D_KPE      = 64
DIM_SPLIT  = 256
TOP_K      = 2048
NUM_SPLITS = TOP_K // DIM_SPLIT   # 8
T_MAX      = 8
TOTAL_TASKS: cutlass.Constexpr = T_MAX * NUM_SPLITS * NUM_HEADS  # 1024

BLOCK_SIZE_COMPUTE = 1024
NUM_WARPS_COMPUTE  = BLOCK_SIZE_COMPUTE // 32   # 32
DIMS_PER_LANE: cutlass.Constexpr = D_CKV // 32  # 16
NUM_VEC:       cutlass.Constexpr = 8
ITERS_PER_LANE: cutlass.Constexpr = (D_CKV // 32) // 8  # 2

BLOCK_SIZE_REDUCE = 512

LN2 = 0.6931471805599453
SENTINEL_SKIP = float("inf")

N_PAGES_FLAT: cutlass.Constexpr = 8462 * 64


# ═══════════════════════════════════════════════════════════════════════════════
# Inlined FA4 FastDivmod (from flash_attn/cute/fast_math.py)
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def _clz(x: Int32) -> Int32:
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res


@cute.jit
def _find_log2(x: Int32) -> Int32:
    a: Int32 = Int32(31 - _clz(x))
    return a + ((x & (x - 1)) != 0)


@dsl_user_op
def _umulhi(a: Int32, b: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class FastDivmod:
    def __init__(self, divisor: Int32, multiplier: Uint32, shift_right: Uint32, *, loc=None, ip=None):
        self.divisor = divisor
        self.multiplier = multiplier
        self.shift_right = shift_right
        self._loc = loc

    @staticmethod
    def create(divisor: Int32, *, loc=None, ip=None) -> "FastDivmod":
        p = Uint32(31 + _find_log2(divisor))
        divisor_u32 = Uint32(divisor)
        multiplier = Uint32(((cutlass.Uint64(1) << p) + divisor_u32 - 1) // divisor_u32)
        shift_right = Uint32(p - 32)
        return FastDivmod(divisor, multiplier, shift_right, loc=loc, ip=ip)

    @cute.jit
    def div(self, dividend: Int32) -> Int32:
        return (
            Int32(_umulhi(dividend, self.multiplier) >> self.shift_right)
            if self.divisor != 1
            else dividend
        )

    def divmod(self, dividend: Int32):
        quotient = self.div(dividend)
        remainder = dividend - quotient * self.divisor
        return quotient, remainder

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.divisor, self.multiplier, self.shift_right]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.divisor, self.multiplier, self.shift_right], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FastDivmod(*(tuple(obj_list)), loc=self._loc)


# ═══════════════════════════════════════════════════════════════════════════════
# Inlined FA4 StaticPersistentTileScheduler
# (from flash_attn/cute/tile_scheduler.py)
# ═══════════════════════════════════════════════════════════════════════════════

class _ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class FA4SchedParams(_ParamsBase):
    num_splits_heads_divmod: FastDivmod   # divides by (NUM_SPLITS * NUM_HEADS)
    total_blocks: Int32

    @staticmethod
    def create(total_tasks: int, n_sh: int, *, loc=None, ip=None) -> "FA4SchedParams":
        return FA4SchedParams(
            num_splits_heads_divmod=FastDivmod.create(Int32(n_sh)),
            total_blocks=Int32(total_tasks),
        )


class FA4StaticPersistentTileScheduler:
    def __init__(self, params: FA4SchedParams, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def create(params: FA4SchedParams, *, loc=None, ip=None) -> "FA4StaticPersistentTileScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return FA4StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: FA4SchedParams, *, loc=None, ip=None):
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (cutlass.min(sm_count, params.total_blocks), Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        is_valid = self._tile_idx < self.params.total_blocks
        return cutlass.utils.WorkTileInfo(
            (Int32(self._tile_idx), Int32(0), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._tile_idx += cute.arch.grid_dim()[0]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FA4StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-pass: count valid (non-(-1)) sparse_indices per token.
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def valid_count_kernel_v4fa4(
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
# Persistent compute kernel with inlined FA4 tile scheduler
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def persistent_compute_kernel_v4fa4(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    global_valid_count: cute.Tensor,
    sched_params: FA4SchedParams,
):
    dim_split:     cutlass.Constexpr = DIM_SPLIT
    num_splits:    cutlass.Constexpr = NUM_SPLITS
    top_k:         cutlass.Constexpr = TOP_K
    num_vec:       cutlass.Constexpr = NUM_VEC
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
    dims_per_lane:  cutlass.Constexpr = DIMS_PER_LANE
    num_threads:   cutlass.Constexpr = BLOCK_SIZE_COMPUTE
    num_warps:     cutlass.Constexpr = NUM_WARPS_COMPUTE
    n_sh:          cutlass.Constexpr = NUM_SPLITS * NUM_HEADS   # 128

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    actual_T = q_nope.shape[0]

    # ── Smem (reused across tiles in the persistent loop) ────────────────────
    allocator  = cutlass.utils.SmemAllocator()
    smem_sparse  = allocator.allocate_tensor(
        cutlass.Int32,    cute.make_layout((top_k,),     stride=(1,)),  4, None)
    smem_logits  = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((dim_split,), stride=(1,)), 16, None)
    smem_red_f32 = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((32,),        stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((D_CKV,),     stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((D_KPE,),     stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((num_warps, D_CKV), stride=(D_CKV, 1)), 16, None)

    # ── FA4-style persistent tile scheduler ───────────────────────────────────
    tile_sched = FA4StaticPersistentTileScheduler.create(sched_params)
    work_tile = tile_sched.initial_work_tile_info()

    while work_tile.is_valid_tile:
        flat_idx = work_tile.tile_idx[0]

        # T>S>H decode: flat_idx = tok*(S*H) + split*H + head
        tok   =  flat_idx // n_sh
        split = (flat_idx // NUM_HEADS) % num_splits
        head  =  flat_idx % NUM_HEADS
        split_start = split * dim_split

        if tok < actual_T:
            valid_cnt   = global_valid_count[tok]
            local_valid = valid_cnt - split_start
            if local_valid > dim_split:
                local_valid = dim_split
            if local_valid < cutlass.Int32(0):
                local_valid = cutlass.Int32(0)

            active_splits = (valid_cnt + dim_split - 1) // dim_split

            if local_valid == cutlass.Int32(0):
                # OOB split: write sentinel partials for reduce kernel
                for i in range(tidx, D_CKV, num_threads):
                    partial_out[tok, head, split, i] = cutlass.Float32(0)
                if tidx == 0:
                    partial_lse[tok, head, split, 0] = -cutlass.Float32(math.inf)
                    partial_lse[tok, head, split, 1] = cutlass.Float32(0)
            else:
                # ── Load sparse indices + q into smem ────────────────────────
                for i in range(tidx, top_k, num_threads):
                    smem_sparse[i] = sparse_indices[tok, i]

                for i in range(tidx, D_CKV, num_threads):
                    smem_q_nope[i] = q_nope[tok, head, i]
                for i in range(tidx, D_KPE, num_threads):
                    smem_q_pe[i] = q_pe[tok, head, i]

                cute.arch.sync_threads()

                # ── Score: LDG.128 + fp32 multiply ───────────────────────────
                q_nope_z   = cute.zipped_divide(smem_q_nope, (num_vec,))
                num_rounds = (local_valid + num_warps - 1) // num_warps

                for round_idx in range(num_rounds):
                    sparse_idx = round_idx * num_warps + warp_idx
                    if sparse_idx < local_valid:
                        cur_idx = smem_sparse[split_start + sparse_idx]

                        ckv_row = ckv_cache[cur_idx, None]
                        ckv_z   = cute.zipped_divide(ckv_row, (num_vec,))

                        sum_partial = cutlass.Float32(0)
                        for it in range(iters_per_lane):
                            group  = it * wsize + lane_idx
                            q_frag = q_nope_z[(None, (group,))].load()
                            K_frag = ckv_z[(None, (group,))].load()
                            for v in range(num_vec):
                                sum_partial += (cutlass.Float32(q_frag[v])
                                                * cutlass.Float32(K_frag[v]))

                        for k_idx in range(D_KPE // wsize):
                            q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                            kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                            sum_partial += q_p * kv

                        s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                        if lane_idx == 0:
                            smem_logits[sparse_idx] = s * sm_scale

                cute.arch.sync_threads()

                # ── Softmax max ───────────────────────────────────────────────
                partial_max = -cutlass.Float32(math.inf)
                for idx in range(tidx, local_valid, num_threads):
                    v = smem_logits[idx]
                    if v > partial_max:
                        partial_max = v

                max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
                if lane_idx == 0:
                    smem_red_f32[warp_idx] = max_val
                cute.arch.sync_threads()
                if warp_idx == 0:
                    val = smem_red_f32[lane_idx]
                    max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
                    smem_red_f32[0] = max_val
                cute.arch.sync_threads()
                row_max = smem_red_f32[0]

                # ── Softmax exp + sum ─────────────────────────────────────────
                partial_sum = cutlass.Float32(0)
                for idx in range(tidx, local_valid, num_threads):
                    e = cute.math.exp(smem_logits[idx] - row_max)
                    smem_logits[idx] = e
                    partial_sum += e

                sum_val = warp_reduce(partial_sum, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_red_f32[warp_idx] = sum_val
                cute.arch.sync_threads()
                if warp_idx == 0:
                    val = smem_red_f32[lane_idx]
                    sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
                    smem_red_f32[0] = sum_val
                cute.arch.sync_threads()
                row_sum = smem_red_f32[0]

                # ── Output: vectorized LDG.128 reads (unnormalised) ──────────
                out_regs = cute.make_rmem_tensor(
                    cute.make_layout((dims_per_lane,), stride=(1,)),
                    cutlass.Float32,
                )
                for k in range(dims_per_lane):
                    out_regs[k] = cutlass.Float32(0)

                for round_idx in range(num_rounds):
                    j = round_idx * num_warps + warp_idx
                    if j < local_valid:
                        kv_idx = smem_sparse[split_start + j]
                        weight = smem_logits[j]

                        V_row = ckv_cache[kv_idx, None]
                        V_z   = cute.zipped_divide(V_row, (num_vec,))

                        for it in range(iters_per_lane):
                            group = it * wsize + lane_idx
                            frag  = V_z[(None, (group,))].load()
                            for v in range(num_vec):
                                out_regs[it * num_vec + v] += (
                                    weight * cutlass.Float32(frag[v]))

                # Write per-warp accumulators to smem
                for it in range(iters_per_lane):
                    for v in range(num_vec):
                        smem_partial[warp_idx, (it * wsize + lane_idx) * num_vec + v] = (
                            out_regs[it * num_vec + v])

                cute.arch.sync_threads()

                # ── Write results (sentinel or partial) ───────────────────────
                if active_splits == cutlass.Int32(1):
                    if split == cutlass.Int32(0):
                        # Single-split fast path: write directly + sentinel
                        for i in range(tidx, D_CKV, num_threads):
                            acc = cutlass.Float32(0)
                            for w in range(num_warps):
                                acc += smem_partial[w, i]
                            output[tok, head, i] = cutlass.BFloat16(acc / row_sum)
                        if tidx == 0:
                            lse[tok, head] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
                            partial_lse[tok, head, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                else:
                    # Multi-split: write to partial buffer (unnormalised)
                    for i in range(tidx, D_CKV, num_threads):
                        acc = cutlass.Float32(0)
                        for w in range(num_warps):
                            acc += smem_partial[w, i]
                        partial_out[tok, head, split, i] = acc
                    if tidx == 0:
                        partial_lse[tok, head, split, 0] = row_max
                        partial_lse[tok, head, split, 1] = row_sum

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()


# ═══════════════════════════════════════════════════════════════════════════════
# Reduce kernel — matches sentinel pattern exactly
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def persistent_reduce_kernel_v4fa4(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
):
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()   # tok, head
    tidx, _, _    = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    smem_sentinel = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

    if tidx == 0:
        smem_sentinel[0] = partial_lse[bidx, bidy, 0, 0]
    cute.arch.sync_threads()

    sentinel_val = smem_sentinel[0]

    if sentinel_val < cutlass.Float32(1e30):
        # Multi-split path: merge all splits
        allocator2 = cutlass.utils.SmemAllocator()
        smem_g_max   = allocator2.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_g_denom = allocator2.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max:
                    g_max = local_max
            smem_g_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                g_denom += local_denom * cute.math.exp(local_max - g_max)
            smem_g_denom[0] = g_denom
        cute.arch.sync_threads()

        g_max   = smem_g_max[0]
        g_denom = smem_g_denom[0]

        if tidx == 0:
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        if tidx < D_CKV:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                scale = cute.math.exp(local_max - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ═══════════════════════════════════════════════════════════════════════════════
# JIT launcher: pre-pass → persistent compute → reduce
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def fused_persistent_v4fa4_launcher(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    global_valid_count: cute.Tensor,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    stream,
):
    T, num_heads, _ = q_nope.shape

    ckv_flat = cute.make_tensor(
        ckv_cache.iterator,
        cute.make_layout((N_PAGES_FLAT, D_CKV), stride=(D_CKV, 1)))
    kpe_flat = cute.make_tensor(
        kpe_cache.iterator,
        cute.make_layout((N_PAGES_FLAT, D_KPE), stride=(D_KPE, 1)))

    # ── 0. Pre-pass: count valid KV entries per token ─────────────────────────
    valid_count_kernel_v4fa4(sparse_indices, global_valid_count).launch(
        grid=[T, 1, 1], block=[1024, 1, 1], stream=stream)

    # ── 1. Persistent compute with FA4-inlined scheduler ─────────────────────
    total_tasks: cutlass.Constexpr = TOTAL_TASKS
    n_sh: cutlass.Constexpr = NUM_SPLITS * NUM_HEADS

    sched_params = FA4SchedParams.create(total_tasks, n_sh)
    grid = FA4StaticPersistentTileScheduler.get_grid_shape(sched_params)

    persistent_compute_kernel_v4fa4(
        q_nope, q_pe, ckv_flat, kpe_flat,
        sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        global_valid_count,
        sched_params,
    ).launch(grid=grid, block=[BLOCK_SIZE_COMPUTE, 1, 1], stream=stream)

    # ── 2. Reduce: merge S splits ─────────────────────────────────────────────
    persistent_reduce_kernel_v4fa4(partial_out, partial_lse, output, lse).launch(
        grid=[T, num_heads, 1], block=[BLOCK_SIZE_REDUCE, 1, 1], stream=stream,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_persistent_v4fa4():
    T = cute.sym_int()
    num_pages, page_size = 8462, 64

    q_nope             = _fake(cute.BFloat16, (T, NUM_HEADS, D_CKV),                  (2, 1, 0), 16)
    q_pe               = _fake(cute.BFloat16, (T, NUM_HEADS, D_KPE),                  (2, 1, 0), 16)
    ckv_cache          = _fake(cute.BFloat16, (num_pages, page_size, D_CKV),          (2, 1, 0), 16)
    kpe_cache          = _fake(cute.BFloat16, (num_pages, page_size, D_KPE),          (2, 1, 0), 16)
    sparse_indices     = _fake(cute.Int32,    (T, TOP_K),                             (1, 0),     4)
    sm_scale           = 0.1352337788608801
    global_valid_count = _fake(cute.Int32,    (T_MAX,),                               (0,),       4)
    partial_out        = _fake(cute.Float32,  (T_MAX, NUM_HEADS, NUM_SPLITS, D_CKV),  (3, 2, 1, 0), 16)
    partial_lse        = _fake(cute.Float32,  (T_MAX, NUM_HEADS, NUM_SPLITS, 2),      (3, 2, 1, 0), 16)
    output             = _fake(cute.BFloat16, (T, NUM_HEADS, D_CKV),                  (2, 1, 0), 16)
    lse                = _fake(cute.Float32,  (T, NUM_HEADS),                         (1, 0),      4)
    stream             = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_persistent_v4fa4_launcher,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        global_valid_count, partial_out, partial_lse, output, lse,
        stream,
        options="--enable-tvm-ffi",
    )


_compiled = compile_fused_persistent_v4fa4()

_global_valid_count = None
_partial_out = None
_partial_lse = None


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    global _global_valid_count, _partial_out, _partial_lse
    if _global_valid_count is None:
        _global_valid_count = torch.empty(T_MAX, dtype=torch.int32, device=output.device)
        _partial_out = torch.empty(
            T_MAX, NUM_HEADS, NUM_SPLITS, D_CKV, dtype=torch.float32, device=output.device)
        _partial_lse = torch.empty(
            T_MAX, NUM_HEADS, NUM_SPLITS, 2, dtype=torch.float32, device=output.device)
    _compiled(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        _global_valid_count, _partial_out, _partial_lse, output, lse,
    )
