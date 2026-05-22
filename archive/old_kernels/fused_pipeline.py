"""
fused_pipeline.py: Pipeline warp specialization for fused DSA kernel.

V1: 16 score warps (producer) + 16 output warps (consumer).
Score warps compute q@K scores and pipe them via PipelineAsync.
Output warps use online softmax and accumulate output = weights@V.
Overlaps score GMEM reads (K) with output GMEM reads (V).

Grid: [T, 16, 1]  Block: 1024 threads = 32 warps
  warps 0-15:  score producer
  warps 16-31: output consumer (online softmax)
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.pipeline import PipelineAsync, CooperativeGroup, Agent

import math
import torch


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


BLOCK_SIZE = 1024
NUM_WARPS  = BLOCK_SIZE // 32   # 32

SCORE_WARPS:  cutlass.Constexpr = 16
OUTPUT_WARPS: cutlass.Constexpr = 16
SCORE_THREADS  = SCORE_WARPS * 32   # 512
OUTPUT_THREADS = OUTPUT_WARPS * 32  # 512
CHUNK_SIZE: cutlass.Constexpr = 16  # == SCORE_WARPS == OUTPUT_WARPS

DIMS_PER_LANE: cutlass.Constexpr = 512 // 32   # 16
NUM_VEC:       cutlass.Constexpr = 8
ITERS_PER_LANE: cutlass.Constexpr = (512 // 32) // 8   # 2

NUM_STAGES = 2
LN2 = 0.6931471805599453


class FusedPipelineV1:
    def __init__(self):
        self.num_stages = NUM_STAGES
        self.block_size = BLOCK_SIZE
        self.shared_storage = None

    @cute.jit
    def __call__(
        self,
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        ckv_cache: cute.Tensor,
        kpe_cache: cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale: cutlass.Constexpr,
        output: cute.Tensor,
        lse: cute.Tensor,
        stream,
    ):
        T, num_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[2]

        N: cutlass.Constexpr = 8462 * 64
        ckv_flat = cute.make_tensor(
            ckv_cache.iterator,
            cute.make_layout((N, head_dim_ckv), stride=(head_dim_ckv, 1)))
        kpe_flat = cute.make_tensor(
            kpe_cache.iterator,
            cute.make_layout((N, head_dim_kpe), stride=(head_dim_kpe, 1)))

        @cute.struct
        class SharedStorage:
            pipeline_mbarrier_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            q_nope, q_pe, ckv_flat, kpe_flat,
            sparse_indices, sm_scale, output, lse,
        ).launch(
            grid=[T, num_heads, 1],
            block=[self.block_size, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        ckv_cache: cute.Tensor,
        kpe_cache: cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale: cutlass.Constexpr,
        output: cute.Tensor,
        lse: cute.Tensor,
    ):
        T, num_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = kpe_cache.shape[1]
        top_k_len    = 2048
        dims_per_lane: cutlass.Constexpr = DIMS_PER_LANE
        num_vec: cutlass.Constexpr = NUM_VEC
        iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
        score_warps: cutlass.Constexpr = SCORE_WARPS
        output_warps: cutlass.Constexpr = OUTPUT_WARPS
        chunk_size: cutlass.Constexpr = CHUNK_SIZE

        bidx, bidy, _ = cute.arch.block_idx()
        num_threads: cutlass.Constexpr = BLOCK_SIZE
        num_warps:   cutlass.Constexpr = NUM_WARPS
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        wsize    = cute.arch.WARP_SIZE

        # ── Smem allocation ──────────────────────────────────────────────────
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        smem_sparse  = smem.allocate_tensor(cutlass.Int32,
            cute.make_layout((top_k_len,), stride=(1,)), 4, None)
        smem_q_nope  = smem.allocate_tensor(cutlass.BFloat16,
            cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
        smem_q_pe    = smem.allocate_tensor(cutlass.BFloat16,
            cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
        smem_red_i32 = smem.allocate_tensor(cutlass.Int32,
            cute.make_layout((32,), stride=(1,)), 4, None)
        smem_red_f32 = smem.allocate_tensor(cutlass.Float32,
            cute.make_layout((32,), stride=(1,)), 16, None)
        smem_partial = smem.allocate_tensor(cutlass.Float32,
            cute.make_layout((output_warps, head_dim_ckv),
                             stride=(head_dim_ckv, 1)), 16, None)
        smem_warp_max = smem.allocate_tensor(cutlass.Float32,
            cute.make_layout((output_warps,), stride=(1,)), 16, None)
        smem_warp_sum = smem.allocate_tensor(cutlass.Float32,
            cute.make_layout((output_warps,), stride=(1,)), 16, None)
        smem_pipe_scores = smem.allocate_tensor(cutlass.Float32,
            cute.make_layout((self.num_stages, chunk_size),
                             stride=(chunk_size, 1)), 16, None)

        # ── Pipeline setup ───────────────────────────────────────────────────
        mainloop_pipeline = PipelineAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, SCORE_THREADS),
            consumer_group=CooperativeGroup(Agent.Thread, OUTPUT_THREADS),
            barrier_storage=storage.pipeline_mbarrier_ptr.data_ptr(),
        )
        producer, consumer = mainloop_pipeline.make_participants()

        # ── Load phase (ALL 1024 threads) ────────────────────────────────────
        partial_cnt_valid = 0
        for i in range(tidx, top_k_len, num_threads):
            idx = sparse_indices[bidx, i]
            smem_sparse[i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt_valid += 1

        for i in range(tidx, head_dim_ckv, num_threads):
            smem_q_nope[i] = q_nope[bidx, bidy, i]
        for i in range(tidx, head_dim_kpe, num_threads):
            smem_q_pe[i] = q_pe[bidx, bidy, i]

        sum_valid = warp_reduce(partial_cnt_valid, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = sum_valid
        cute.arch.sync_threads()

        if warp_idx == 0:
            val = smem_red_i32[lane_idx]
            sum_valid = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_i32[0] = sum_valid
        cute.arch.sync_threads()

        valid_count = smem_red_i32[0]
        num_chunks  = (valid_count + chunk_size - 1) // chunk_size

        # ── Pipeline phase ───────────────────────────────────────────────────
        q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec,))

        # Score producer (warps 0-15)
        if warp_idx < score_warps:
            score_warp_idx = warp_idx

            for chunk_idx in range(num_chunks):
                handle = producer.acquire_and_advance()
                token_idx = chunk_idx * score_warps + score_warp_idx

                if token_idx < valid_count:
                    cur_idx = smem_sparse[token_idx]

                    ckv_row = ckv_cache[cur_idx, None]
                    ckv_z   = cute.zipped_divide(ckv_row, (num_vec,))

                    sum_partial = cutlass.Float32(0)
                    for it in range(iters_per_lane):
                        group  = it * wsize + lane_idx
                        q_frag = q_nope_z[(None, (group,))].load()
                        K_frag = ckv_z[(None, (group,))].load()
                        for v in range(num_vec):
                            sum_partial += cutlass.Float32(q_frag[v]) * cutlass.Float32(K_frag[v])

                    for k_idx in range(head_dim_kpe // wsize):
                        q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                        kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                        sum_partial += q_p * kv

                    s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                    if lane_idx == 0:
                        smem_pipe_scores[handle.index, score_warp_idx] = s * sm_scale
                else:
                    if lane_idx == 0:
                        smem_pipe_scores[handle.index, score_warp_idx] = -cutlass.Float32(math.inf)

                handle.commit()

            producer.tail()

        # Output consumer (warps 16-31)
        if warp_idx >= score_warps:
            out_warp_idx = warp_idx - score_warps

            out_regs = cute.make_rmem_tensor(
                cute.make_layout((dims_per_lane,), stride=(1,)),
                cutlass.Float32,
            )
            for k in range(dims_per_lane):
                out_regs[k] = cutlass.Float32(0)

            my_max = -cutlass.Float32(math.inf)
            my_sum = cutlass.Float32(0)

            for chunk_idx in range(num_chunks):
                handle = consumer.wait_and_advance()
                token_idx = chunk_idx * output_warps + out_warp_idx

                if token_idx < valid_count:
                    s = smem_pipe_scores[handle.index, out_warp_idx]

                    # Online softmax update
                    old_max = my_max
                    if s > my_max:
                        my_max = s
                    rescale = cute.math.exp(old_max - my_max)
                    my_sum = my_sum * rescale + cute.math.exp(s - my_max)
                    weight = cute.math.exp(s - my_max)

                    # Rescale existing output
                    for k in range(dims_per_lane):
                        out_regs[k] = out_regs[k] * rescale

                    # Load V and accumulate
                    kv_idx = smem_sparse[token_idx]
                    V_row = ckv_cache[kv_idx, None]
                    V_z   = cute.zipped_divide(V_row, (num_vec,))

                    for it in range(iters_per_lane):
                        group = it * wsize + lane_idx
                        frag  = V_z[(None, (group,))].load()
                        for v in range(num_vec):
                            out_regs[it * num_vec + v] += weight * cutlass.Float32(frag[v])

                handle.release()

            # Write partial results to smem
            for it in range(iters_per_lane):
                for v in range(num_vec):
                    smem_partial[out_warp_idx, (it * wsize + lane_idx) * num_vec + v] = out_regs[it * num_vec + v]

            if lane_idx == 0:
                smem_warp_max[out_warp_idx] = my_max
                smem_warp_sum[out_warp_idx] = my_sum

        cute.arch.sync_threads()

        # ── Merge phase (ALL threads) ────────────────────────────────────────
        if valid_count > 0:
            # Warp 0: find global max/sum, precompute corrections
            if warp_idx == 0:
                w_max = -cutlass.Float32(math.inf)
                if lane_idx < output_warps:
                    w_max = smem_warp_max[lane_idx]
                global_max = warp_reduce(w_max, lambda a, b: a if a > b else b, width=32)
                smem_red_f32[0] = global_max

                w_sum = cutlass.Float32(0)
                if lane_idx < output_warps:
                    w_sum = smem_warp_sum[lane_idx] * cute.math.exp(smem_warp_max[lane_idx] - global_max)
                global_sum = warp_reduce(w_sum, lambda a, b: a + b, width=32)
                smem_red_f32[1] = global_sum

                # Per-warp correction: exp(max_w - global_max) / global_sum
                if lane_idx < output_warps:
                    smem_red_f32[2 + lane_idx] = cute.math.exp(smem_warp_max[lane_idx] - global_max) / global_sum

            cute.arch.sync_threads()

            global_max = smem_red_f32[0]
            global_sum = smem_red_f32[1]

            # Write LSE
            if tidx == 0:
                lse[bidx, bidy] = (global_max + cute.math.log(global_sum)) / cutlass.Float32(LN2)

            # Cross-warp reduce → global output
            for i in range(tidx, head_dim_ckv, num_threads):
                acc = cutlass.Float32(0)
                for w in range(output_warps):
                    acc += smem_partial[w, i] * smem_red_f32[2 + w]
                output[bidx, bidy, i] = cutlass.BFloat16(acc)


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_pipeline_v1():
    T = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048
    num_pages, page_size = 8462, 64

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, top_k_len),               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = FusedPipelineV1()
    return cute.compile(
        kernel,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


_compiled = compile_fused_pipeline_v1()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, output, lse)
