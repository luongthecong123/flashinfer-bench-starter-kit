"""Intra-kernel profiling for kv_split_xor_sentinel.

sentinel trick: compute writes SENTINEL_SKIP (= +inf) to partial_lse[T_idx, h, 0, 0]
for single-split tokens. Reduce reads this first; if sentinel → skip entire reduction.

Extra probe tag: "sentinel_skip" fires when the reduce block takes the fast path.
The reduction in count_valid+reduce time across all blocks reveals the savings.

Phases:
  compute: upfront / score / softmax_max / softmax_exp_sum / output / write
  reduce:  sentinel_skip  (fast path: single-split token, output already written)
           count_valid    (slow path: multi-split token, must count valid)
           reduce         (slow path cont'd: cross-split merge)
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
import math, json, torch

# ── Timer helpers ─────────────────────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(MLIR_T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int64(t)

@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(MLIR_T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int32(t)

PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES_COMPUTE = 48
MAX_ENTRIES_REDUCE  = 8   # more entries: sentinel_skip OR (count_valid + reduce)
PROBE_COLS_COMPUTE = PROBE_HEADER + MAX_ENTRIES_COMPUTE * PROBE_ENTRY
PROBE_COLS_REDUCE  = PROBE_HEADER + MAX_ENTRIES_REDUCE  * PROBE_ENTRY

TAGS_COMPUTE = {
    "upfront": 0, "score": 2, "softmax_max": 4, "softmax_exp_sum": 6, "output": 8, "write": 10,
}
TAG_NAMES_COMPUTE = {v: k for k, v in TAGS_COMPUTE.items()}
PHASE_ORDER_COMPUTE = ["upfront", "score", "softmax_max", "softmax_exp_sum", "output", "write"]

TAGS_REDUCE = {"count_valid": 0, "reduce": 2, "sentinel_skip": 4}
TAG_NAMES_REDUCE = {v: k for k, v in TAGS_REDUCE.items()}
PHASE_ORDER_REDUCE = ["sentinel_skip", "count_valid", "reduce"]


def range_start(probe, row, cnt, sm_val, tag_val):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()

def range_stop(probe, row, cnt):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64() - probe[row, off + 2]
    return cnt + cutlass.Int32(1)

def range_finalize(probe, row, cnt):
    probe[row, 0] = cutlass.Int64(cnt)


def _probe_events(probe_cpu, num_blocks, tag_names, pid_offset=0):
    events = []; base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (base is None or s < base): base = s
    base = base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        if cnt == 0: continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off+1]); t0 = int(data[off+2]); dur = int(data[off+3])
            if t0 == 0 and dur == 0: continue
            events.append(dict(name=tag_names.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0-base)/1000.0, dur=dur/1000.0, pid=sm_id+pid_offset, tid=bid))
    return events, base


def dump_compute(probe: torch.Tensor, num_blocks: int, num_head: int, num_splits: int):
    probe_cpu = probe.cpu().contiguous().tolist()
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur: max_dur, max_bid = total, bid
    data = probe_cpu[max_bid]; cnt = int(data[0])
    print(f"\n--- Compute: Slowest block {max_bid} (head={max_bid//num_splits}, split_old={max_bid%num_splits}, total={max_dur/1000:.1f}µs) ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        print(f"  sm={int(data[off]):>3} {TAG_NAMES_COMPUTE.get(int(data[off+1]), f'tag_{int(data[off+1])}'):>16s}  dur={int(data[off+3]):>10} ns  ({int(data[off+3])/1000:.1f} µs)")
    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            name = TAG_NAMES_COMPUTE.get(int(data[off+1]), f"tag_{int(data[off+1])}")
            tag_totals[name] = tag_totals.get(name, 0) + int(data[off+3])
            tag_counts[name] = tag_counts.get(name, 0) + 1
    print(f"\n{'='*65}")
    print(f"{'Phase (all blocks)':>24s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>10s} {'%':>6s}")
    print(f"{'='*65}")
    grand = sum(tag_totals.values())
    for name in PHASE_ORDER_COMPUTE:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>24s} {tot/1e6:>12.3f} {cnt_:>6d} {tot/cnt_/1000:>10.1f} {100*tot/grand:>5.1f}%")
    print(f"{'TOTAL':>24s} {grand/1e6:>12.3f}")
    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_COMPUTE, pid_offset=0)


def dump_reduce(probe: torch.Tensor, num_blocks: int):
    probe_cpu = probe.cpu().contiguous().tolist()
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur: max_dur, max_bid = total, bid
    data = probe_cpu[max_bid]; cnt = int(data[0])
    print(f"\n--- Reduce: Slowest block {max_bid} (total={max_dur/1000:.1f}µs) ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        print(f"  sm={int(data[off]):>3} {TAG_NAMES_REDUCE.get(int(data[off+1]), f'tag_{int(data[off+1])}'):>12s}  dur={int(data[off+3]):>10} ns  ({int(data[off+3])/1000:.1f} µs)")
    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            name = TAG_NAMES_REDUCE.get(int(data[off+1]), f"tag_{int(data[off+1])}")
            tag_totals[name] = tag_totals.get(name, 0) + int(data[off+3])
            tag_counts[name] = tag_counts.get(name, 0) + 1
    print(f"\n{'='*55}")
    print(f"{'Phase':>14s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>10s}")
    print(f"{'='*55}")
    for name in PHASE_ORDER_REDUCE:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>14s} {tot/1e6:>12.3f} {cnt_:>6d} {tot/cnt_/1000:>10.1f}")
    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_REDUCE, pid_offset=200)


def build_combined_trace(compute_events, compute_base, reduce_events, reduce_base) -> str:
    shared_base = min(b for b in [compute_base, reduce_base] if b)
    all_events = []
    for ev in compute_events:
        all_events.append(dict(ev, ts=ev["ts"] + (compute_base - shared_base) / 1000.0))
    for ev in reduce_events:
        all_events.append(dict(ev, ts=ev["ts"] + (reduce_base - shared_base) / 1000.0))
    return json.dumps({"traceEvents": all_events})


# ── Kernel constants ──────────────────────────────────────────────────────────

NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX = 8
NUM_SPLITS = 8
DIM_SPLIT = TOP_K_LEN // NUM_SPLITS
LN2 = 0.6931471805599453
SENTINEL_SKIP = float("inf")


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def count_valid_indices(
    sparse_indices: cute.Tensor, smem_sparse: cute.Tensor,
    smem_red_i32: cute.Tensor, smem_num_valid: cute.Tensor,
    T: cute.Numeric, tidx: cute.Numeric, warp_idx: cute.Numeric,
    top_k_len: cutlass.Constexpr, sparse_thr_per_T: cutlass.Constexpr,
    num_warps_per_T: cutlass.Constexpr,
) -> None:
    thr_idx_per_T  = tidx % sparse_thr_per_T
    lane_idx_per_T = thr_idx_per_T % cute.arch.WARP_SIZE
    wg_per_T_idx   = tidx // sparse_thr_per_T
    warp_per_T_idx = warp_idx % num_warps_per_T
    partial_cnt = 0
    if wg_per_T_idx < T:
        for i in range(thr_idx_per_T, top_k_len, sparse_thr_per_T):
            idx = sparse_indices[wg_per_T_idx, i]
            smem_sparse[wg_per_T_idx, i] = idx
            if idx >= cutlass.Int32(0): partial_cnt += 1
        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx_per_T == 0: smem_red_i32[wg_per_T_idx, warp_per_T_idx] = cnt_sum
        cute.arch.barrier(barrier_id=wg_per_T_idx + 1, number_of_threads=sparse_thr_per_T)
        if warp_per_T_idx == 0:
            val = smem_red_i32[wg_per_T_idx, lane_idx_per_T]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps_per_T)
            smem_red_i32[wg_per_T_idx, 0] = cnt_sum
        cute.arch.barrier(barrier_id=wg_per_T_idx + 1, number_of_threads=sparse_thr_per_T)
        smem_num_valid[wg_per_T_idx] = smem_red_i32[wg_per_T_idx, 0]


class Kv_split_xor_sentinel_intra():
    def __init__(self):
        self.num_head, self.head_dim_ckv, self.head_dim_kpe, self.top_k_len = NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN
        self.num_pages, self.page_size  = NUM_PAGES, PAGE_SIZE
        self.T_max = T_MAX
        self.num_splits = NUM_SPLITS
        self.dim_split = self.top_k_len // self.num_splits
        self.num_threads = 1024
        self.wsize = cute.arch.WARP_SIZE
        self.num_warps = self.num_threads // self.wsize
        self.vec_size_ckv = 8
        self.vec_size_kpe = 2
        self.vec_size_out = 16
        self.iters_per_lane_ckv = self.head_dim_ckv // (self.wsize * self.vec_size_ckv)
        self.sparse_thr_per_T = 128
        self.num_warps_per_T = self.sparse_thr_per_T // self.wsize

    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)

    @cute.jit
    def __call__(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
                 sm_scale: cutlass.Constexpr,
                 partial_out, partial_lse, output, lse, probe_compute, probe_reduce, stream):
        T, _, _ = q_nope.shape
        N = self.num_pages * self.page_size
        ckv_flat = cute.make_tensor(ckv_cache.iterator,
            cute.make_layout((N, self.head_dim_ckv), stride=(self.head_dim_ckv, 1)))
        kpe_flat = cute.make_tensor(kpe_cache.iterator,
            cute.make_layout((N, self.head_dim_kpe), stride=(self.head_dim_kpe, 1)))
        self.compute_kernel(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
            partial_out, partial_lse, output, lse, probe_compute
        ).launch(grid=[self.num_head, self.num_splits, 1], block=[self.num_threads, 1, 1], stream=stream)
        self.reduce_kernel(sparse_indices, partial_out, partial_lse, output, lse, probe_reduce
        ).launch(grid=[T, self.num_head, 1], block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def reduce_kernel(self, sparse_indices, partial_out, partial_lse, output, lse, probe_reduce):
        T, _, _ = output.shape
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _    = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx(); warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        reduce_row = bidx * self.num_head + bidy
        sm = cutlass.Int64(smid_u32()); probe_cnt = cutlass.Int32(0)

        sentinel_val = partial_lse[bidx, bidy, 0, 0]
        if sentinel_val >= cutlass.Float32(1e30):
            # Fast path: single-split token, output already written by compute kernel
            if tidx == 0:
                range_start(probe_reduce, reduce_row, probe_cnt, sm, TAGS_REDUCE["sentinel_skip"])
                probe_cnt = range_stop(probe_reduce, reduce_row, probe_cnt)
                range_finalize(probe_reduce, reduce_row, probe_cnt)
        else:
            # Slow path: multi-split token, count valid + merge partial results
            if tidx == 0:
                range_start(probe_reduce, reduce_row, probe_cnt, sm, TAGS_REDUCE["count_valid"])

            alloc = cutlass.utils.SmemAllocator()
            smem_red_i32 = self._smem(alloc, cutlass.Int32,   (32,),                (1,),    4)
            smem_max_sum = self._smem(alloc, cutlass.Float32, (self.num_splits, 2), (2, 1),  4)

            if tidx < self.num_splits:
                smem_max_sum[tidx, 0] = partial_lse[bidx, bidy, tidx, 0]
                smem_max_sum[tidx, 1] = partial_lse[bidx, bidy, tidx, 1]

            partial_cnt = 0
            for i in range(tidx, self.top_k_len, self.num_threads):
                idx = sparse_indices[bidx, i]
                if idx >= cutlass.Int32(0): partial_cnt += 1
            cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
            if lane_idx == 0: smem_red_i32[warp_idx] = cnt_sum
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_red_i32[lane_idx]
                cnt_sum = warp_reduce(val, lambda a, b: a + b, width=self.num_warps)
                smem_red_i32[0] = cnt_sum
            if tidx == 0: probe_cnt = range_stop(probe_reduce, reduce_row, probe_cnt)
            cute.arch.sync_threads()

            num_valid = smem_red_i32[0]
            # We know it's multi-split (sentinel check handled single-split)
            if tidx == 0:
                range_start(probe_reduce, reduce_row, probe_cnt, sm, TAGS_REDUCE["reduce"])
            num_active_splits = (num_valid + self.dim_split - 1) // self.dim_split
            g_max = -cutlass.Float32(math.inf)
            for split_idx in range(num_active_splits):
                l_max = smem_max_sum[split_idx, 0]
                if l_max > g_max: g_max = l_max
            if tidx < self.head_dim_ckv:
                g_lse_sum = cutlass.Float32(0); acc = cutlass.Float32(0)
                for split_idx in range(num_active_splits):
                    l_max = smem_max_sum[split_idx, 0]; l_sum = smem_max_sum[split_idx, 1]
                    g_lse_sum += l_sum * cute.math.exp(l_max - g_max)
                    acc += cute.math.exp(l_max - g_max) * partial_out[bidx, bidy, split_idx, tidx]
                output[bidx, bidy, tidx] = cutlass.BFloat16(acc / g_lse_sum)
                if tidx == 0:
                    lse[bidx, bidy] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)
            if tidx == 0: probe_cnt = range_stop(probe_reduce, reduce_row, probe_cnt)
            if tidx == 0: range_finalize(probe_reduce, reduce_row, probe_cnt)

    @cute.kernel
    def compute_kernel(self, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                       sm_scale: cutlass.Constexpr,
                       partial_out, partial_lse, output, lse, probe_compute):
        T, _, _ = q_nope.shape
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _    = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx(); warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        probe_row = bidx * self.num_splits + bidy
        sm = cutlass.Int64(smid_u32()); probe_cnt = cutlass.Int32(0)

        if tidx == 0:
            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["upfront"])

        head_idx      = bidx
        thr_idx_per_T = tidx % self.sparse_thr_per_T
        wg_per_T_idx  = tidx // self.sparse_thr_per_T

        alloc = cutlass.utils.SmemAllocator()
        smem_sparse      = self._smem(alloc, cutlass.Int32,    (self.T_max, self.top_k_len),        (self.top_k_len, 1),     4)
        smem_num_valid   = self._smem(alloc, cutlass.Int32,    (self.T_max,),                       (1,),                    4)
        smem_logits      = self._smem(alloc, cutlass.Float32,  (self.dim_split,),                   (1,),                   16)
        smem_red_i32     = self._smem(alloc, cutlass.Int32,    (self.T_max, 32),                    (32, 1),                 4)
        smem_max_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)
        smem_sum_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)
        smem_q_nope      = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_ckv),     (self.head_dim_ckv, 1), 16)
        smem_q_pe        = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_kpe),     (self.head_dim_kpe, 1), 16)
        smem_partial     = self._smem(alloc, cutlass.Float32,  (self.num_warps, self.head_dim_ckv), (self.head_dim_ckv, 1), 16)
        smem_out         = self._smem(alloc, cutlass.Float32,  (self.head_dim_ckv,),                (1,),                   16)

        if wg_per_T_idx < T:
            for i in range(thr_idx_per_T, self.head_dim_ckv, self.sparse_thr_per_T):
                smem_q_nope[wg_per_T_idx, i] = q_nope[wg_per_T_idx, head_idx, i]
            for i in range(thr_idx_per_T, self.head_dim_kpe, self.sparse_thr_per_T):
                smem_q_pe[wg_per_T_idx, i] = q_pe[wg_per_T_idx, head_idx, i]

        count_valid_indices(sparse_indices, smem_sparse, smem_red_i32, smem_num_valid,
            T, tidx, warp_idx, self.top_k_len, self.sparse_thr_per_T, self.num_warps_per_T)
        cute.arch.sync_threads()
        if tidx == 0: probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

        split_idx_old = bidy
        smem_q_nope_ = cute.zipped_divide(smem_q_nope, (1, self.vec_size_ckv))
        ckv_flat_    = cute.zipped_divide(ckv_flat,    (1, self.vec_size_ckv))
        kpe_flat_    = cute.zipped_divide(kpe_flat,    (1, self.vec_size_kpe))
        smem_q_pe_   = cute.zipped_divide(smem_q_pe,   (1, self.vec_size_kpe))

        for T_idx in range(T):
            split_idx_new = T_idx ^ split_idx_old
            num_valid_T   = smem_num_valid[T_idx]
            split_start   = split_idx_new * self.dim_split
            is_OOB        = split_start >= num_valid_T

            if not is_OOB:
                local_valid = min(num_valid_T - split_start, self.dim_split)
                num_rounds  = (local_valid + self.num_warps - 1) // self.num_warps

                # ── Score ────────────────────────────────────────────────────
                if tidx == 0: range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["score"])
                for round_idx in range(num_rounds):
                    sparse_idx = round_idx * self.num_warps + warp_idx
                    if sparse_idx < local_valid:
                        cur_idx  = smem_sparse[T_idx, split_start + sparse_idx]
                        ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)]
                        kpe_row_ = kpe_flat_[(0, None), (cur_idx, None)]
                        sum_partial = cutlass.Float32(0)
                        for it in range(self.iters_per_lane_ckv):
                            rest_idx = it * self.wsize + lane_idx
                            qn_vec   = smem_q_nope_[(0, None), (T_idx, rest_idx)].load()
                            ckv_vec  = ckv_row_[None, rest_idx].load()
                            for i in range(self.vec_size_ckv):
                                sum_partial += cutlass.Float32(qn_vec[i]) * cutlass.Float32(ckv_vec[i])
                        qp_vec  = smem_q_pe_[(0, None), (T_idx, lane_idx)].load()
                        kpe_vec = kpe_row_[None, lane_idx].load()
                        for i in range(self.vec_size_kpe):
                            sum_partial += cutlass.Float32(qp_vec[i]) * cutlass.Float32(kpe_vec[i])
                        s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                        if lane_idx == 0: smem_logits[sparse_idx] = s * sm_scale
                cute.arch.sync_threads()
                if tidx == 0: probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

                # ── Softmax max ───────────────────────────────────────────────
                if tidx == 0: range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["softmax_max"])
                partial_max = -cutlass.Float32(math.inf)
                for idx in range(tidx, local_valid, self.num_threads):
                    v = smem_logits[idx]
                    if v > partial_max: partial_max = v
                max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
                if lane_idx == 0: smem_max_red_f32[warp_idx] = max_val
                cute.arch.sync_threads()
                if warp_idx == 0:
                    val = smem_max_red_f32[lane_idx]
                    max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=self.num_warps)
                    smem_max_red_f32[0] = max_val
                cute.arch.sync_threads()
                row_max = smem_max_red_f32[0]
                if tidx == 0: probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

                # ── Softmax exp+sum ───────────────────────────────────────────
                if tidx == 0: range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["softmax_exp_sum"])
                local_sum = cutlass.Float32(0)
                for idx in range(tidx, local_valid, self.num_threads):
                    e = cute.math.exp(smem_logits[idx] - row_max)
                    smem_logits[idx] = e
                    local_sum += e
                sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
                if lane_idx == 0: smem_sum_red_f32[warp_idx] = sum_val
                cute.arch.sync_threads()
                if warp_idx == 0:
                    val = smem_sum_red_f32[lane_idx]
                    sum_val = warp_reduce(val, lambda a, b: a + b, width=self.num_warps)
                    smem_sum_red_f32[0] = sum_val
                cute.arch.sync_threads()
                row_sum = smem_sum_red_f32[0]
                if tidx == 0: probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

                # ── Output GEMV ───────────────────────────────────────────────
                if tidx == 0: range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["output"])
                out_regs = cute.make_rmem_tensor(cute.make_layout((self.vec_size_out,), stride=(1,)), cutlass.Float32)
                for i in range(self.vec_size_out): out_regs[i] = cutlass.Float32(0)
                for round_idx in range(num_rounds):
                    sparse_idx = round_idx * self.num_warps + warp_idx
                    if sparse_idx < local_valid:
                        cur_idx  = smem_sparse[T_idx, split_start + sparse_idx]
                        ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)]
                        e        = smem_logits[sparse_idx]
                        for it in range(self.iters_per_lane_ckv):
                            rest_idx = it * self.wsize + lane_idx
                            ckv_vec  = ckv_row_[None, rest_idx].load()
                            for i in range(self.vec_size_ckv):
                                out_regs[it * self.vec_size_ckv + i] += e * cutlass.Float32(ckv_vec[i])
                if warp_idx < local_valid:
                    for it in range(self.iters_per_lane_ckv):
                        for v in range(self.vec_size_ckv):
                            smem_partial[warp_idx, (it * self.wsize + lane_idx) * self.vec_size_ckv + v] = out_regs[it * self.vec_size_ckv + v]
                cute.arch.sync_threads()
                num_active_warps = local_valid if local_valid < self.num_warps else self.num_warps
                for i in range(tidx, self.head_dim_ckv, self.num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_active_warps): acc += smem_partial[w, i]
                    smem_out[i] = acc
                cute.arch.sync_threads()
                if tidx == 0: probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

                # ── Write ─────────────────────────────────────────────────────
                if tidx == 0: range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["write"])
                is_single_split_request = num_valid_T < self.dim_split
                if is_single_split_request and split_idx_new == 0:
                    for i in range(tidx, self.head_dim_ckv, self.num_threads):
                        output[T_idx, head_idx, i] = cutlass.BFloat16(smem_out[i] / row_sum)
                    if tidx == 0:
                        partial_lse[T_idx, head_idx, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                        lse[T_idx, head_idx] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
                else:
                    for i in range(tidx, self.head_dim_ckv, self.num_threads):
                        partial_out[T_idx, head_idx, split_idx_new, i] = smem_out[i]
                    if tidx == 0:
                        partial_lse[T_idx, head_idx, split_idx_new, 0] = row_max
                        partial_lse[T_idx, head_idx, split_idx_new, 1] = row_sum
                cute.arch.sync_threads()
                if tidx == 0: probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

        if tidx == 0: range_finalize(probe_compute, probe_row, probe_cnt)


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kernel():
    T = cute.sym_int(); Br = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN
    num_pages, page_size = NUM_PAGES, PAGE_SIZE
    Bc = num_heads * NUM_SPLITS

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv),                 (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe),                 (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv),         (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe),         (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, top_k_len),                               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (T_MAX, num_heads, NUM_SPLITS, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T_MAX, num_heads, NUM_SPLITS, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv),                 (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),                               (1, 0),     4)
    probe_compute  = _fake(cute.Int64,    (Bc, PROBE_COLS_COMPUTE),                     (1, 0),     8)
    probe_reduce   = _fake(cute.Int64,    (Br, PROBE_COLS_REDUCE),                      (1, 0),     8)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(Kv_split_xor_sentinel_intra(),
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, probe_compute, probe_reduce, stream,
        options="--enable-tvm-ffi")


def run_single(workload_idx: int) -> str:
    import os, json as _json
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors
    H, D_ckv = NUM_HEADS, HEAD_DIM_CKV
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling kv_split_xor_sentinel kernel...")
    compiled = compile_kernel()
    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [_json.loads(l) for l in open(JSONL)]
    w = workloads[workload_idx]; ax = w["workload"]["axes"]; inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]
    Bc = NUM_HEADS * NUM_SPLITS; num_reduce_blocks = T * H
    print(f"\nWorkload {workload_idx+1}: MaxValid={max_valid}  T={T}  ComputeBlocks={Bc}  ReduceBlocks={num_reduce_blocks}")
    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()
    output_t    = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse_t       = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
    partial_out = torch.empty(T_MAX, H, NUM_SPLITS, D_ckv, dtype=torch.float32, device="cuda")
    partial_lse = torch.full((T_MAX, H, NUM_SPLITS, 2), -float("inf"), dtype=torch.float32, device="cuda")
    probe_compute = torch.zeros((Bc, PROBE_COLS_COMPUTE), dtype=torch.int64, device="cuda")
    probe_reduce  = torch.zeros((num_reduce_blocks, PROBE_COLS_REDUCE), dtype=torch.int64, device="cuda")
    for _ in range(3):
        output_t.zero_(); lse_t.fill_(-float("inf"))
        partial_lse.fill_(-float("inf"))
        probe_compute.zero_(); probe_reduce.zero_()
        compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse, output_t, lse_t, probe_compute, probe_reduce)
        torch.cuda.synchronize()
    probe_compute.zero_(); probe_reduce.zero_()
    output_t.zero_(); lse_t.fill_(-float("inf")); partial_lse.fill_(-float("inf"))
    compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse, output_t, lse_t, probe_compute, probe_reduce)
    torch.cuda.synchronize()
    compute_events, compute_base = dump_compute(probe_compute, Bc, NUM_HEADS, NUM_SPLITS)
    reduce_events,  reduce_base  = dump_reduce(probe_reduce, num_reduce_blocks)
    return build_combined_trace(compute_events, compute_base, reduce_events, reduce_base)
