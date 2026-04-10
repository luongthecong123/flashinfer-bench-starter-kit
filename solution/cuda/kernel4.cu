/*
 * kernel4.cu — Hybrid dispatch: fused single-block (T<3) + KV-split (T>=3).
 *
 * Fused path  (kernel2): Grid [T, 16, 1]  Block 1024  — processes all 2048 keys in one block
 * KV-split path (kernel3): Grid [T, 16, 8] + [T, 16, 1] compute+reduce
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

// ── Constants ───────────────────────────────────────────────────────────────
static constexpr int NUM_HEADS       = 16;
static constexpr int HEAD_DIM_CKV    = 512;
static constexpr int HEAD_DIM_KPE    = 64;
static constexpr int DIM_SPLIT       = 256;
static constexpr int TOP_K           = 2048;
static constexpr int NUM_SPLITS      = 8;
static constexpr int T_MAX           = 8;

static constexpr int BLOCK_COMPUTE   = 1024;
static constexpr int NUM_WARPS       = 32;
static constexpr int BLOCK_REDUCE    = 512;

static constexpr float SM_SCALE      = 0.1352337788608801f;
static constexpr float LN2_INV       = 1.0f / 0.6931471805599453f;

static constexpr int VEC             = 8;
static constexpr int DIMS_PER_LANE   = HEAD_DIM_CKV / 32;   // 16
static constexpr int ITERS_PER_LANE  = DIMS_PER_LANE / VEC;  // 2

// ── Warp reductions ─────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, o));
    return v;
}

__device__ __forceinline__ int warp_reduce_sum_int(int v) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}

// ═════════════════════════════════════════════════════════════════════════════
// Shared memory layouts
// ═════════════════════════════════════════════════════════════════════════════

// Fused kernel (kernel2 path): logits[2048] since we process ALL keys in one block
struct FusedSmem {
    int    sparse[TOP_K];                        // 8192 B
    float  logits[TOP_K];                        // 8192 B
    int    red_i32[32];                          // 128 B
    float  red_f32[32];                          // 128 B
    __nv_bfloat16 q_nope[HEAD_DIM_CKV];         // 1024 B
    __nv_bfloat16 q_pe[HEAD_DIM_KPE];           // 128 B
    float  partial[NUM_WARPS][HEAD_DIM_CKV];    // 65536 B
};  // ~83 KB

// KV-split kernel (kernel3 path): logits[256] since each split handles DIM_SPLIT keys
struct ComputeSmem {
    int    sparse[TOP_K];                        // 8192 B
    float  logits[DIM_SPLIT];                    // 1024 B
    int    red_i32[32];                          // 128 B
    float  red_f32[32];                          // 128 B
    __nv_bfloat16 q_nope[HEAD_DIM_CKV];         // 1024 B
    __nv_bfloat16 q_pe[HEAD_DIM_KPE];           // 128 B
    float  partial[NUM_WARPS][HEAD_DIM_CKV];    // 65536 B
};  // ~76 KB


// ═════════════════════════════════════════════════════════════════════════════
// Kernel A: Fused single-block (kernel2 path) — Grid [T, 16, 1]
// ═════════════════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(BLOCK_COMPUTE)
fused_dsa_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int*           __restrict__ sparse_indices,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse)
{
    extern __shared__ char smem_raw[];
    FusedSmem& sm = *reinterpret_cast<FusedSmem*>(smem_raw);

    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 32;
    const int lane_idx = tidx & 31;

    // ── Load sparse_indices + Q → smem, count valid ─────────────────────
    int partial_cnt = 0;
    for (int i = tidx; i < TOP_K; i += BLOCK_COMPUTE) {
        int idx = sparse_indices[tok * TOP_K + i];
        sm.sparse[i] = idx;
        if (idx >= 0) ++partial_cnt;
    }

    const __nv_bfloat16* q_nope_row = q_nope + (tok * NUM_HEADS + head) * HEAD_DIM_CKV;
    const __nv_bfloat16* q_pe_row   = q_pe   + (tok * NUM_HEADS + head) * HEAD_DIM_KPE;
    for (int i = tidx; i < HEAD_DIM_CKV; i += BLOCK_COMPUTE)
        sm.q_nope[i] = q_nope_row[i];
    for (int i = tidx; i < HEAD_DIM_KPE; i += BLOCK_COMPUTE)
        sm.q_pe[i] = q_pe_row[i];

    int cnt_sum = warp_reduce_sum_int(partial_cnt);
    if (lane_idx == 0) sm.red_i32[warp_idx] = cnt_sum;
    __syncthreads();
    if (warp_idx == 0) {
        int val = sm.red_i32[lane_idx];
        val = warp_reduce_sum_int(val);
        sm.red_i32[0] = val;
    }
    __syncthreads();

    const int valid_count = sm.red_i32[0];
    const int num_rounds = (valid_count + NUM_WARPS - 1) / NUM_WARPS;

    // ── Score phase: LDG.128 vectorized ─────────────────────────────────
    for (int round = 0; round < num_rounds; ++round) {
        int sparse_idx = round * NUM_WARPS + warp_idx;
        if (sparse_idx < valid_count) {
            int cur_idx = sm.sparse[sparse_idx];
            const __nv_bfloat16* ckv_row = ckv_cache + (long long)cur_idx * HEAD_DIM_CKV;

            float sum_partial = 0.0f;
            #pragma unroll
            for (int it = 0; it < ITERS_PER_LANE; ++it) {
                int group = it * 32 + lane_idx;
                int base  = group * VEC;
                const uint4* q_vec = reinterpret_cast<const uint4*>(&sm.q_nope[base]);
                const uint4* k_vec = reinterpret_cast<const uint4*>(&ckv_row[base]);
                uint4 qv = *q_vec;
                uint4 kv = *k_vec;
                const __nv_bfloat16* q8 = reinterpret_cast<const __nv_bfloat16*>(&qv);
                const __nv_bfloat16* k8 = reinterpret_cast<const __nv_bfloat16*>(&kv);
                #pragma unroll
                for (int v = 0; v < VEC; ++v)
                    sum_partial += __bfloat162float(q8[v]) * __bfloat162float(k8[v]);
            }

            const __nv_bfloat16* kpe_row = kpe_cache + (long long)cur_idx * HEAD_DIM_KPE;
            #pragma unroll
            for (int k = 0; k < HEAD_DIM_KPE / 32; ++k) {
                int off = k * 32 + lane_idx;
                sum_partial += __bfloat162float(sm.q_pe[off]) * __bfloat162float(kpe_row[off]);
            }

            float s = warp_reduce_sum(sum_partial);
            if (lane_idx == 0) sm.logits[sparse_idx] = s * SM_SCALE;
        }
    }
    __syncthreads();

    // ── Softmax: max ────────────────────────────────────────────────────
    float partial_max = -INFINITY;
    for (int i = tidx; i < valid_count; i += BLOCK_COMPUTE) {
        float v = sm.logits[i];
        if (v > partial_max) partial_max = v;
    }
    float max_val = warp_reduce_max(partial_max);
    if (lane_idx == 0) sm.red_f32[warp_idx] = max_val;
    __syncthreads();
    if (warp_idx == 0) {
        float val = sm.red_f32[lane_idx];
        val = warp_reduce_max(val);
        sm.red_f32[0] = val;
    }
    __syncthreads();
    const float row_max = sm.red_f32[0];

    // ── Softmax: exp + sum ──────────────────────────────────────────────
    float local_sum = 0.0f;
    for (int i = tidx; i < valid_count; i += BLOCK_COMPUTE) {
        float e = expf(sm.logits[i] - row_max);
        sm.logits[i] = e;
        local_sum += e;
    }
    float sum_val = warp_reduce_sum(local_sum);
    if (lane_idx == 0) sm.red_f32[warp_idx] = sum_val;
    __syncthreads();
    if (warp_idx == 0) {
        float val = sm.red_f32[lane_idx];
        val = warp_reduce_sum(val);
        sm.red_f32[0] = val;
    }
    __syncthreads();
    const float row_sum = sm.red_f32[0];

    // Write LSE
    if (tidx == 0)
        lse[tok * NUM_HEADS + head] = (row_max + logf(row_sum)) * LN2_INV;

    // ── Output GEMV: weight/row_sum inline ──────────────────────────────
    float out_regs[DIMS_PER_LANE];
    #pragma unroll
    for (int k = 0; k < DIMS_PER_LANE; ++k) out_regs[k] = 0.0f;

    for (int round = 0; round < num_rounds; ++round) {
        int j = round * NUM_WARPS + warp_idx;
        if (j < valid_count) {
            int kv_idx = sm.sparse[j];
            float weight = sm.logits[j] / row_sum;
            const __nv_bfloat16* V_row = ckv_cache + (long long)kv_idx * HEAD_DIM_CKV;

            #pragma unroll
            for (int it = 0; it < ITERS_PER_LANE; ++it) {
                int group = it * 32 + lane_idx;
                int base  = group * VEC;
                const uint4* v_vec = reinterpret_cast<const uint4*>(&V_row[base]);
                uint4 vv = *v_vec;
                const __nv_bfloat16* v8 = reinterpret_cast<const __nv_bfloat16*>(&vv);
                #pragma unroll
                for (int v = 0; v < VEC; ++v)
                    out_regs[it * VEC + v] += weight * __bfloat162float(v8[v]);
            }
        }
    }

    // Write per-warp partials to smem
    #pragma unroll
    for (int it = 0; it < ITERS_PER_LANE; ++it) {
        #pragma unroll
        for (int v = 0; v < VEC; ++v)
            sm.partial[warp_idx][(it * 32 + lane_idx) * VEC + v] = out_regs[it * VEC + v];
    }
    __syncthreads();

    // ── Cross-warp reduce → global output ───────────────────────────────
    for (int i = tidx; i < HEAD_DIM_CKV; i += BLOCK_COMPUTE) {
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) acc += sm.partial[w][i];
        output[(tok * NUM_HEADS + head) * HEAD_DIM_CKV + i] = __float2bfloat16(acc);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Kernel B: KV-split compute (kernel3 path) — Grid [T, 16, 8]
// ═════════════════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(BLOCK_COMPUTE)
kvsplit_compute_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int*           __restrict__ sparse_indices,
    float*               __restrict__ partial_out,
    float*               __restrict__ partial_lse,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse)
{
    extern __shared__ char smem_raw[];
    ComputeSmem& sm = *reinterpret_cast<ComputeSmem*>(smem_raw);

    const int tok   = blockIdx.x;
    const int head  = blockIdx.y;
    const int split = blockIdx.z;
    const int tidx  = threadIdx.x;
    const int warp_idx = tidx / 32;
    const int lane_idx = tidx & 31;

    // ── Phase 1: load sparse_indices + Q, count valid ───────────────────
    int partial_cnt = 0;
    for (int i = tidx; i < TOP_K; i += BLOCK_COMPUTE) {
        int idx = sparse_indices[tok * TOP_K + i];
        sm.sparse[i] = idx;
        if (idx >= 0) ++partial_cnt;
    }

    const __nv_bfloat16* q_nope_row = q_nope + (tok * NUM_HEADS + head) * HEAD_DIM_CKV;
    const __nv_bfloat16* q_pe_row   = q_pe   + (tok * NUM_HEADS + head) * HEAD_DIM_KPE;
    for (int i = tidx; i < HEAD_DIM_CKV; i += BLOCK_COMPUTE)
        sm.q_nope[i] = q_nope_row[i];
    for (int i = tidx; i < HEAD_DIM_KPE; i += BLOCK_COMPUTE)
        sm.q_pe[i] = q_pe_row[i];

    int cnt_sum = warp_reduce_sum_int(partial_cnt);
    if (lane_idx == 0) sm.red_i32[warp_idx] = cnt_sum;
    __syncthreads();
    if (warp_idx == 0) {
        int val = sm.red_i32[lane_idx];
        val = warp_reduce_sum_int(val);
        sm.red_i32[0] = val;
    }
    __syncthreads();

    const int global_num_valid = sm.red_i32[0];
    const int split_start = split * DIM_SPLIT;
    int local_valid = global_num_valid - split_start;
    if (local_valid > DIM_SPLIT) local_valid = DIM_SPLIT;
    if (local_valid < 0)         local_valid = 0;
    const int active_splits = (global_num_valid + DIM_SPLIT - 1) / DIM_SPLIT;

    // ── Phase 2: compute ────────────────────────────────────────────────
    if (local_valid == 0) {
        for (int i = tidx; i < HEAD_DIM_CKV; i += BLOCK_COMPUTE)
            partial_out[((tok * NUM_HEADS + head) * NUM_SPLITS + split) * HEAD_DIM_CKV + i] = 0.0f;
        if (tidx == 0) {
            int base = ((tok * NUM_HEADS + head) * NUM_SPLITS + split) * 2;
            partial_lse[base + 0] = -INFINITY;
            partial_lse[base + 1] = 0.0f;
        }
        return;
    }

    const int num_rounds = (local_valid + NUM_WARPS - 1) / NUM_WARPS;

    // ── Score phase ─────────────────────────────────────────────────────
    for (int round = 0; round < num_rounds; ++round) {
        int sparse_idx = round * NUM_WARPS + warp_idx;
        if (sparse_idx < local_valid) {
            int cur_idx = sm.sparse[split_start + sparse_idx];
            const __nv_bfloat16* ckv_row = ckv_cache + (long long)cur_idx * HEAD_DIM_CKV;

            float sum_partial = 0.0f;
            #pragma unroll
            for (int it = 0; it < ITERS_PER_LANE; ++it) {
                int group = it * 32 + lane_idx;
                int base  = group * VEC;
                const uint4* q_vec = reinterpret_cast<const uint4*>(&sm.q_nope[base]);
                const uint4* k_vec = reinterpret_cast<const uint4*>(&ckv_row[base]);
                uint4 qv = *q_vec;
                uint4 kv = *k_vec;
                const __nv_bfloat16* q8 = reinterpret_cast<const __nv_bfloat16*>(&qv);
                const __nv_bfloat16* k8 = reinterpret_cast<const __nv_bfloat16*>(&kv);
                #pragma unroll
                for (int v = 0; v < VEC; ++v)
                    sum_partial += __bfloat162float(q8[v]) * __bfloat162float(k8[v]);
            }

            const __nv_bfloat16* kpe_row = kpe_cache + (long long)cur_idx * HEAD_DIM_KPE;
            #pragma unroll
            for (int k = 0; k < HEAD_DIM_KPE / 32; ++k) {
                int off = k * 32 + lane_idx;
                sum_partial += __bfloat162float(sm.q_pe[off]) * __bfloat162float(kpe_row[off]);
            }

            float s = warp_reduce_sum(sum_partial);
            if (lane_idx == 0) sm.logits[sparse_idx] = s * SM_SCALE;
        }
    }
    __syncthreads();

    // ── Softmax: max ────────────────────────────────────────────────────
    float partial_max = -INFINITY;
    for (int i = tidx; i < local_valid; i += BLOCK_COMPUTE) {
        float v = sm.logits[i];
        if (v > partial_max) partial_max = v;
    }
    float max_val = warp_reduce_max(partial_max);
    if (lane_idx == 0) sm.red_f32[warp_idx] = max_val;
    __syncthreads();
    if (warp_idx == 0) {
        float val = sm.red_f32[lane_idx];
        val = warp_reduce_max(val);
        sm.red_f32[0] = val;
    }
    __syncthreads();
    const float row_max = sm.red_f32[0];

    // ── Softmax: exp + sum ──────────────────────────────────────────────
    float local_sum = 0.0f;
    for (int i = tidx; i < local_valid; i += BLOCK_COMPUTE) {
        float e = expf(sm.logits[i] - row_max);
        sm.logits[i] = e;
        local_sum += e;
    }
    float sum_val = warp_reduce_sum(local_sum);
    if (lane_idx == 0) sm.red_f32[warp_idx] = sum_val;
    __syncthreads();
    if (warp_idx == 0) {
        float val = sm.red_f32[lane_idx];
        val = warp_reduce_sum(val);
        sm.red_f32[0] = val;
    }
    __syncthreads();
    const float row_sum = sm.red_f32[0];

    // Single-split fast path
    if (active_splits == 1 && split == 0) {
        for (int i = tidx; i < local_valid; i += BLOCK_COMPUTE)
            sm.logits[i] = sm.logits[i] / row_sum;
        __syncthreads();
    }

    // ── Output GEMV ─────────────────────────────────────────────────────
    float out_regs[DIMS_PER_LANE];
    #pragma unroll
    for (int k = 0; k < DIMS_PER_LANE; ++k) out_regs[k] = 0.0f;

    for (int round = 0; round < num_rounds; ++round) {
        int j = round * NUM_WARPS + warp_idx;
        if (j < local_valid) {
            int kv_idx = sm.sparse[split_start + j];
            float weight = sm.logits[j];
            const __nv_bfloat16* V_row = ckv_cache + (long long)kv_idx * HEAD_DIM_CKV;

            #pragma unroll
            for (int it = 0; it < ITERS_PER_LANE; ++it) {
                int group = it * 32 + lane_idx;
                int base  = group * VEC;
                const uint4* v_vec = reinterpret_cast<const uint4*>(&V_row[base]);
                uint4 vv = *v_vec;
                const __nv_bfloat16* v8 = reinterpret_cast<const __nv_bfloat16*>(&vv);
                #pragma unroll
                for (int v = 0; v < VEC; ++v)
                    out_regs[it * VEC + v] += weight * __bfloat162float(v8[v]);
            }
        }
    }

    #pragma unroll
    for (int it = 0; it < ITERS_PER_LANE; ++it) {
        #pragma unroll
        for (int v = 0; v < VEC; ++v)
            sm.partial[warp_idx][(it * 32 + lane_idx) * VEC + v] = out_regs[it * VEC + v];
    }
    __syncthreads();

    // ── Write results ───────────────────────────────────────────────────
    if (active_splits == 1 && split == 0) {
        for (int i = tidx; i < HEAD_DIM_CKV; i += BLOCK_COMPUTE) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) acc += sm.partial[w][i];
            output[(tok * NUM_HEADS + head) * HEAD_DIM_CKV + i] = __float2bfloat16(acc);
        }
        if (tidx == 0) {
            int base = ((tok * NUM_HEADS + head) * NUM_SPLITS + 0) * 2;
            partial_lse[base + 0] = INFINITY;  // sentinel
            lse[tok * NUM_HEADS + head] = (row_max + logf(row_sum)) * LN2_INV;
        }
    } else {
        for (int i = tidx; i < HEAD_DIM_CKV; i += BLOCK_COMPUTE) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) acc += sm.partial[w][i];
            partial_out[((tok * NUM_HEADS + head) * NUM_SPLITS + split) * HEAD_DIM_CKV + i] = acc;
        }
        if (tidx == 0) {
            int base = ((tok * NUM_HEADS + head) * NUM_SPLITS + split) * 2;
            partial_lse[base + 0] = row_max;
            partial_lse[base + 1] = row_sum;
        }
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Kernel C: Reduce splits — Grid [T, 16, 1]
// ═════════════════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(BLOCK_REDUCE)
kvsplit_reduce_kernel(
    const float*         __restrict__ partial_out,
    const float*         __restrict__ partial_lse,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse)
{
    __shared__ float smem_sentinel;
    __shared__ float smem_g_max;
    __shared__ float smem_g_denom;

    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int tidx = threadIdx.x;

    if (tidx == 0) {
        int base0 = ((tok * NUM_HEADS + head) * NUM_SPLITS + 0) * 2;
        smem_sentinel = partial_lse[base0 + 0];
    }
    __syncthreads();

    if (smem_sentinel >= 1e30f) return;

    if (tidx == 0) {
        float g_max = -INFINITY;
        for (int s = 0; s < NUM_SPLITS; ++s) {
            float lm = partial_lse[((tok * NUM_HEADS + head) * NUM_SPLITS + s) * 2 + 0];
            if (lm > g_max) g_max = lm;
        }
        smem_g_max = g_max;

        float g_denom = 0.0f;
        for (int s = 0; s < NUM_SPLITS; ++s) {
            float lm = partial_lse[((tok * NUM_HEADS + head) * NUM_SPLITS + s) * 2 + 0];
            float ld = partial_lse[((tok * NUM_HEADS + head) * NUM_SPLITS + s) * 2 + 1];
            g_denom += ld * expf(lm - g_max);
        }
        smem_g_denom = g_denom;
    }
    __syncthreads();

    float g_max   = smem_g_max;
    float g_denom = smem_g_denom;

    if (tidx == 0)
        lse[tok * NUM_HEADS + head] = (g_max + logf(g_denom)) * LN2_INV;

    if (tidx < HEAD_DIM_CKV) {
        float acc = 0.0f;
        for (int s = 0; s < NUM_SPLITS; ++s) {
            float lm = partial_lse[((tok * NUM_HEADS + head) * NUM_SPLITS + s) * 2 + 0];
            float ld = partial_lse[((tok * NUM_HEADS + head) * NUM_SPLITS + s) * 2 + 1];
            float scale = expf(lm - g_max) / g_denom;
            acc += partial_out[((tok * NUM_HEADS + head) * NUM_SPLITS + s) * HEAD_DIM_CKV + tidx] * scale;
        }
        output[(tok * NUM_HEADS + head) * HEAD_DIM_CKV + tidx] = __float2bfloat16(acc);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Host entry: hybrid dispatch
// ═════════════════════════════════════════════════════════════════════════════

static float* d_partial_out = nullptr;
static float* d_partial_lse = nullptr;

static constexpr size_t PARTIAL_OUT_BYTES = T_MAX * NUM_HEADS * NUM_SPLITS * HEAD_DIM_CKV * sizeof(float);
static constexpr size_t PARTIAL_LSE_BYTES = T_MAX * NUM_HEADS * NUM_SPLITS * 2 * sizeof(float);

void kernel_impl(
    ffi::TensorView q_nope,
    ffi::TensorView q_pe,
    ffi::TensorView ckv_cache,
    ffi::TensorView kpe_cache,
    ffi::TensorView sparse_indices,
    double sm_scale_unused,
    ffi::TensorView output_t,
    ffi::TensorView lse_t)
{
    const int T = static_cast<int>(q_nope.shape()[0]);

    DLDevice dev = q_nope.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    auto* q_nope_ptr  = static_cast<const __nv_bfloat16*>(q_nope.data_ptr());
    auto* q_pe_ptr    = static_cast<const __nv_bfloat16*>(q_pe.data_ptr());
    auto* ckv_ptr     = static_cast<const __nv_bfloat16*>(ckv_cache.data_ptr());
    auto* kpe_ptr     = static_cast<const __nv_bfloat16*>(kpe_cache.data_ptr());
    auto* sparse_ptr  = static_cast<const int*>(sparse_indices.data_ptr());
    auto* output_ptr  = static_cast<__nv_bfloat16*>(output_t.data_ptr());
    auto* lse_ptr     = static_cast<float*>(lse_t.data_ptr());

    if (T < 3) {
        // ── Fused path: single block per (token, head) ──────────────────
        static bool fused_smem_configured = false;
        size_t fused_smem = sizeof(FusedSmem);
        if (!fused_smem_configured) {
            cudaFuncSetAttribute(fused_dsa_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(fused_smem));
            fused_smem_configured = true;
        }

        dim3 grid(T, NUM_HEADS, 1);
        dim3 block(BLOCK_COMPUTE);
        fused_dsa_kernel<<<grid, block, fused_smem, stream>>>(
            q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, sparse_ptr,
            output_ptr, lse_ptr);
    } else {
        // ── KV-split path: 8 splits + reduce ───────────────────────────
        if (!d_partial_out) {
            cudaMalloc(&d_partial_out, PARTIAL_OUT_BYTES);
            cudaMalloc(&d_partial_lse, PARTIAL_LSE_BYTES);
        }

        static bool split_smem_configured = false;
        size_t split_smem = sizeof(ComputeSmem);
        if (!split_smem_configured) {
            cudaFuncSetAttribute(kvsplit_compute_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(split_smem));
            split_smem_configured = true;
        }

        dim3 grid_c(T, NUM_HEADS, NUM_SPLITS);
        dim3 block_c(BLOCK_COMPUTE);
        kvsplit_compute_kernel<<<grid_c, block_c, split_smem, stream>>>(
            q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, sparse_ptr,
            d_partial_out, d_partial_lse, output_ptr, lse_ptr);

        dim3 grid_r(T, NUM_HEADS, 1);
        dim3 block_r(BLOCK_REDUCE);
        kvsplit_reduce_kernel<<<grid_r, block_r, 0, stream>>>(
            d_partial_out, d_partial_lse, output_ptr, lse_ptr);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel_impl);
