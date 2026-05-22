# Benchmark Results — Kernel Variants (ALL PASS)

B200, Modal, L2 flush, arg clone, 100 reps, 3 warmup, DPS-style (output pre-allocated)

## Summary

| # | Variant | Geo Speedup | Geo Impl ms | Arith Speedup |
|---|---|---|---|---|
| 1 | thr_warpv3 | 60.39x | 0.036 | 61.29x |
| 2 | xor_pdl_v3_pro | 64.39x | 0.032 | 65.73x |
| 3 | rot_pdl_v3_pro | 62.38x | 0.034 | 63.81x |
| 4 | xor | 51.79x | 0.040 | 52.75x |
| 5 | xor_pdl_v3 | 56.41x | 0.036 | 57.45x |
| 6 | SSA_v2 | 56.45x | 0.035 | 57.44x |
| 7 | SSA_TMA_static | 54.93x | 0.036 | 55.89x |
| 8 | SSA_TMA | 56.93x | 0.036 | 57.99x |
| 9 | xor_skew | 53.18x | 0.040 | 54.38x |

## Per-Workload Speedup Comparison

| # | T | thr_warpv3 | xor_pdl_v3_pro | rot_pdl_v3_pro | xor | xor_pdl_v3 | SSA_v2 | SSA_TMA_static | SSA_TMA | xor_skew |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 46.30x | 52.93x | 47.51x | 44.95x | 49.52x | 49.69x | 46.92x | 51.10x | 42.32x |
| 2 | 2 | 57.98x | 65.10x | 56.15x | 51.46x | 56.71x | 57.36x | 53.94x | 57.86x | 34.28x |
| 3 | 2 | 59.89x | 64.60x | 53.73x | 49.17x | 52.79x | 54.45x | 51.78x | 54.19x | 50.62x |
| 4 | 2 | 59.93x | 65.16x | 58.27x | 51.17x | 53.74x | 56.92x | 51.62x | 55.66x | 48.23x |
| 5 | 2 | 43.07x | 48.82x | 46.90x | 38.53x | 40.95x | 40.47x | 41.07x | 41.10x | 40.82x |
| 6 | 2 | 57.48x | 64.86x | 57.76x | 49.87x | 58.22x | 55.98x | 54.33x | 58.59x | 50.10x |
| 7 | 2 | 59.96x | 66.56x | 58.18x | 48.52x | 51.32x | 53.09x | 49.01x | 52.02x | 45.49x |
| 8 | 2 | 43.51x | 48.24x | 48.35x | 40.21x | 41.65x | 41.53x | 41.56x | 41.95x | 40.31x |
| 9 | 2 | 56.24x | 64.38x | 58.50x | 50.66x | 56.27x | 55.90x | 52.59x | 56.76x | 49.52x |
| 10 | 8 | 75.52x | 80.13x | 81.12x | 64.70x | 70.53x | 69.07x | 69.77x | 70.03x | 68.40x |
| 11 | 8 | 77.51x | 85.62x | 90.14x | 72.05x | 74.65x | 74.62x | 72.51x | 73.49x | 71.97x |
| 12 | 7 | 67.90x | 63.83x | 65.35x | 40.13x | 56.78x | 57.00x | 56.41x | 56.41x | 55.75x |
| 13 | 8 | 64.81x | 70.00x | 70.31x | 58.89x | 61.71x | 63.49x | 60.63x | 62.61x | 61.80x |
| 14 | 6 | 64.08x | 66.33x | 71.83x | 57.46x | 61.45x | 59.96x | 60.69x | 61.81x | 60.63x |
| 15 | 8 | 61.89x | 65.33x | 65.91x | 55.24x | 59.70x | 58.42x | 58.38x | 60.11x | 57.94x |
| 16 | 6 | 86.66x | 113.44x | 100.01x | 79.85x | 88.93x | 87.80x | 84.60x | 91.80x | 82.62x |
| 17 | 8 | 53.79x | 53.76x | 51.84x | 44.97x | 47.23x | 47.53x | 46.28x | 47.75x | 50.84x |
| 18 | 7 | 59.00x | 67.98x | 71.25x | 56.13x | 61.54x | 60.81x | 58.72x | 61.54x | 58.92x |
| 19 | 8 | 76.42x | 69.92x | 70.06x | 56.72x | 61.27x | 60.25x | 60.41x | 62.75x | 60.98x |
| 20 | 8 | 70.70x | 78.31x | 85.22x | 68.75x | 74.12x | 74.17x | 73.07x | 74.96x | 73.64x |
| 21 | 8 | 56.57x | 52.87x | 56.85x | 46.80x | 50.99x | 51.01x | 50.29x | 50.13x | 52.73x |
| 22 | 6 | 54.66x | 45.68x | 46.90x | 39.87x | 41.66x | 42.19x | 41.81x | 41.60x | 43.18x |
| 23 | 7 | 55.82x | 58.02x | 55.49x | 47.23x | 49.65x | 49.32x | 48.99x | 49.56x | 49.54x |
| **Geo** |  | **60.39x** | **64.39x** | **62.38x** | **51.79x** | **56.41x** | **56.45x** | **54.93x** | **56.93x** | **53.18x** |
| **Arith** |  | **61.29x** | **65.73x** | **63.81x** | **52.75x** | **57.45x** | **57.44x** | **55.89x** | **57.99x** | **54.38x** |

## Per-Workload Impl Time (ms)

| # | T | thr_warpv3 | xor_pdl_v3_pro | rot_pdl_v3_pro | xor | xor_pdl_v3 | SSA_v2 | SSA_TMA_static | SSA_TMA | xor_skew |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.024 | 0.021 | 0.023 | 0.025 | 0.022 | 0.022 | 0.023 | 0.022 | 0.028 |
| 2 | 2 | 0.024 | 0.021 | 0.024 | 0.027 | 0.024 | 0.023 | 0.024 | 0.023 | 0.041 |
| 3 | 2 | 0.024 | 0.021 | 0.026 | 0.028 | 0.026 | 0.025 | 0.026 | 0.025 | 0.039 |
| 4 | 2 | 0.024 | 0.022 | 0.024 | 0.027 | 0.025 | 0.023 | 0.026 | 0.024 | 0.029 |
| 5 | 2 | 0.033 | 0.028 | 0.030 | 0.037 | 0.033 | 0.034 | 0.033 | 0.033 | 0.034 |
| 6 | 2 | 0.024 | 0.021 | 0.024 | 0.027 | 0.023 | 0.023 | 0.024 | 0.023 | 0.027 |
| 7 | 2 | 0.024 | 0.021 | 0.024 | 0.029 | 0.027 | 0.025 | 0.027 | 0.026 | 0.031 |
| 8 | 2 | 0.034 | 0.029 | 0.029 | 0.035 | 0.033 | 0.032 | 0.032 | 0.033 | 0.035 |
| 9 | 2 | 0.025 | 0.021 | 0.023 | 0.027 | 0.024 | 0.024 | 0.025 | 0.024 | 0.028 |
| 10 | 8 | 0.039 | 0.036 | 0.036 | 0.044 | 0.040 | 0.039 | 0.039 | 0.040 | 0.043 |
| 11 | 8 | 0.038 | 0.033 | 0.033 | 0.040 | 0.037 | 0.037 | 0.037 | 0.037 | 0.040 |
| 12 | 7 | 0.041 | 0.042 | 0.041 | 0.066 | 0.045 | 0.045 | 0.045 | 0.045 | 0.048 |
| 13 | 8 | 0.048 | 0.041 | 0.042 | 0.049 | 0.045 | 0.045 | 0.045 | 0.045 | 0.049 |
| 14 | 6 | 0.040 | 0.036 | 0.034 | 0.041 | 0.038 | 0.037 | 0.038 | 0.038 | 0.040 |
| 15 | 8 | 0.049 | 0.044 | 0.044 | 0.051 | 0.047 | 0.047 | 0.047 | 0.047 | 0.052 |
| 16 | 6 | 0.029 | 0.021 | 0.024 | 0.029 | 0.025 | 0.025 | 0.026 | 0.025 | 0.029 |
| 17 | 8 | 0.058 | 0.054 | 0.056 | 0.064 | 0.060 | 0.060 | 0.060 | 0.060 | 0.060 |
| 18 | 7 | 0.047 | 0.038 | 0.038 | 0.046 | 0.042 | 0.042 | 0.042 | 0.042 | 0.045 |
| 19 | 8 | 0.040 | 0.042 | 0.042 | 0.050 | 0.046 | 0.045 | 0.045 | 0.047 | 0.048 |
| 20 | 8 | 0.044 | 0.036 | 0.034 | 0.041 | 0.038 | 0.037 | 0.037 | 0.038 | 0.040 |
| 21 | 8 | 0.056 | 0.054 | 0.053 | 0.060 | 0.056 | 0.056 | 0.055 | 0.055 | 0.057 |
| 22 | 6 | 0.047 | 0.053 | 0.052 | 0.061 | 0.057 | 0.056 | 0.055 | 0.056 | 0.056 |
| 23 | 7 | 0.050 | 0.045 | 0.050 | 0.057 | 0.053 | 0.053 | 0.052 | 0.053 | 0.055 |
| **Geo** |  | **0.036** | **0.032** | **0.034** | **0.040** | **0.036** | **0.035** | **0.036** | **0.036** | **0.040** |
| **Arith** |  | **0.037** | **0.034** | **0.035** | **0.042** | **0.038** | **0.037** | **0.038** | **0.037** | **0.041** |

## Speedup — T > 2 only (multi-split workloads)

| # | T | thr_warpv3 | xor_pdl_v3_pro | rot_pdl_v3_pro | xor | xor_pdl_v3 | SSA_v2 | SSA_TMA_static | SSA_TMA | xor_skew |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 8 | 75.52x | 80.13x | 81.12x | 64.70x | 70.53x | 69.07x | 69.77x | 70.03x | 68.40x |
| 11 | 8 | 77.51x | 85.62x | 90.14x | 72.05x | 74.65x | 74.62x | 72.51x | 73.49x | 71.97x |
| 12 | 7 | 67.90x | 63.83x | 65.35x | 40.13x | 56.78x | 57.00x | 56.41x | 56.41x | 55.75x |
| 13 | 8 | 64.81x | 70.00x | 70.31x | 58.89x | 61.71x | 63.49x | 60.63x | 62.61x | 61.80x |
| 14 | 6 | 64.08x | 66.33x | 71.83x | 57.46x | 61.45x | 59.96x | 60.69x | 61.81x | 60.63x |
| 15 | 8 | 61.89x | 65.33x | 65.91x | 55.24x | 59.70x | 58.42x | 58.38x | 60.11x | 57.94x |
| 16 | 6 | 86.66x | 113.44x | 100.01x | 79.85x | 88.93x | 87.80x | 84.60x | 91.80x | 82.62x |
| 17 | 8 | 53.79x | 53.76x | 51.84x | 44.97x | 47.23x | 47.53x | 46.28x | 47.75x | 50.84x |
| 18 | 7 | 59.00x | 67.98x | 71.25x | 56.13x | 61.54x | 60.81x | 58.72x | 61.54x | 58.92x |
| 19 | 8 | 76.42x | 69.92x | 70.06x | 56.72x | 61.27x | 60.25x | 60.41x | 62.75x | 60.98x |
| 20 | 8 | 70.70x | 78.31x | 85.22x | 68.75x | 74.12x | 74.17x | 73.07x | 74.96x | 73.64x |
| 21 | 8 | 56.57x | 52.87x | 56.85x | 46.80x | 50.99x | 51.01x | 50.29x | 50.13x | 52.73x |
| 22 | 6 | 54.66x | 45.68x | 46.90x | 39.87x | 41.66x | 42.19x | 41.81x | 41.60x | 43.18x |
| 23 | 7 | 55.82x | 58.02x | 55.49x | 47.23x | 49.65x | 49.32x | 48.99x | 49.56x | 49.54x |
| **Geo** |  | **65.40x** | **67.67x** | **68.68x** | **55.17x** | **60.29x** | **60.02x** | **59.14x** | **60.53x** | **59.79x** |
| **Arith** |  | **66.09x** | **69.37x** | **70.16x** | **56.34x** | **61.44x** | **61.12x** | **60.18x** | **61.75x** | **60.64x** |

## Impl Time (ms) — T > 2 only

| # | T | thr_warpv3 | xor_pdl_v3_pro | rot_pdl_v3_pro | xor | xor_pdl_v3 | SSA_v2 | SSA_TMA_static | SSA_TMA | xor_skew |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 8 | 0.039 | 0.036 | 0.036 | 0.044 | 0.040 | 0.039 | 0.039 | 0.040 | 0.043 |
| 11 | 8 | 0.038 | 0.033 | 0.033 | 0.040 | 0.037 | 0.037 | 0.037 | 0.037 | 0.040 |
| 12 | 7 | 0.041 | 0.042 | 0.041 | 0.066 | 0.045 | 0.045 | 0.045 | 0.045 | 0.048 |
| 13 | 8 | 0.048 | 0.041 | 0.042 | 0.049 | 0.045 | 0.045 | 0.045 | 0.045 | 0.049 |
| 14 | 6 | 0.040 | 0.036 | 0.034 | 0.041 | 0.038 | 0.037 | 0.038 | 0.038 | 0.040 |
| 15 | 8 | 0.049 | 0.044 | 0.044 | 0.051 | 0.047 | 0.047 | 0.047 | 0.047 | 0.052 |
| 16 | 6 | 0.029 | 0.021 | 0.024 | 0.029 | 0.025 | 0.025 | 0.026 | 0.025 | 0.029 |
| 17 | 8 | 0.058 | 0.054 | 0.056 | 0.064 | 0.060 | 0.060 | 0.060 | 0.060 | 0.060 |
| 18 | 7 | 0.047 | 0.038 | 0.038 | 0.046 | 0.042 | 0.042 | 0.042 | 0.042 | 0.045 |
| 19 | 8 | 0.040 | 0.042 | 0.042 | 0.050 | 0.046 | 0.045 | 0.045 | 0.047 | 0.048 |
| 20 | 8 | 0.044 | 0.036 | 0.034 | 0.041 | 0.038 | 0.037 | 0.037 | 0.038 | 0.040 |
| 21 | 8 | 0.056 | 0.054 | 0.053 | 0.060 | 0.056 | 0.056 | 0.055 | 0.055 | 0.057 |
| 22 | 6 | 0.047 | 0.053 | 0.052 | 0.061 | 0.057 | 0.056 | 0.055 | 0.056 | 0.056 |
| 23 | 7 | 0.050 | 0.045 | 0.050 | 0.057 | 0.053 | 0.053 | 0.052 | 0.053 | 0.055 |
| **Geo** |  | **0.044** | **0.040** | **0.040** | **0.049** | **0.044** | **0.044** | **0.044** | **0.044** | **0.047** |
| **Arith** |  | **0.045** | **0.041** | **0.041** | **0.050** | **0.045** | **0.045** | **0.044** | **0.045** | **0.047** |

---

## Submission Kernels — Full Comparison (solution/triton/)

B200, Modal, 100 reps, 5 trials, all 23 workloads, ALL PASS

| Kernel | Description | Geo Speedup | Geo Lat (ms) | Arith Speedup | Arith Lat (ms) |
|---|---|---:|---:|---:|---:|
| k1 | Fused single-block | **42.06x** | 0.0502 | 48.28x | 0.0665 |
| k2 | Fused single-block + thr_warpv3 | **61.40x** | 0.0346 | 70.39x | 0.0460 |
| k3 | KV-split XOR persistent | **60.76x** | 0.0357 | 61.92x | 0.0374 |
| k4 | Hybrid fused (T<3) + KV-split XOR PDL | **80.44x** | 0.0289 | 81.91x | 0.0316 |
| k5 | Hybrid fused (T<3) + XOR-persistent PDL v3 pro | **82.35x** | 0.0258 | 84.59x | 0.0288 |
| k5cl | k5c class + no sentinel (4-run avg) | **87.18x** | 0.0243 | 89.18x | 0.0266 |
| k5cb | k5c bare-function style | **86.72x** | 0.0244 | 88.66x | 0.0269 |
| kcn | k5cb without SENTINEL_SKIP | **86.58x** | 0.0237 | 88.69x | 0.0260 |

> **Note:** k5 and k5cl values are 4-run means; k5cb and kcn values are 4-run means. k1–k4 values are single-run.

### Per-Workload: Latency (ms) and Speedup

| # | Ref (ms) | k1 Lat | k1 Spd | k2 Lat | k2 Spd | k3 Lat | k3 Spd | k4 Lat | k4 Spd | k5 Lat | k5 Spd | k5cl Lat | k5cl Spd | k5cb Lat | k5cb Spd | kcn Lat | kcn Spd |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.131 | 0.0186 | 60.84x | 0.0119 | 94.14x | 0.0220 | 50.79x | 0.0157 | 81.91x | 0.0130 | 87.50x | 0.0130 | 90.84x | 0.0127 | 89.07x | 0.0123 | 94.61x |
| 2 | 1.371 | 0.0177 | 77.57x | 0.0121 | 111.31x | 0.0217 | 62.77x | 0.0138 | 108.19x | 0.0132 | 104.49x | 0.0130 | 111.06x | 0.0127 | 110.06x | 0.0123 | 112.86x |
| 3 | 1.375 | 0.0180 | 76.47x | 0.0123 | 111.86x | 0.0368 | 39.60x | 0.0168 | 96.05x | 0.0138 | 101.89x | 0.0130 | 103.62x | 0.0135 | 106.35x | 0.0130 | 104.14x |
| 4 | 1.387 | 0.0190 | 72.91x | 0.0134 | 104.21x | 0.0236 | 60.91x | 0.0171 | 102.53x | 0.0140 | 102.09x | 0.0140 | 100.69x | 0.0135 | 103.22x | 0.0138 | 101.13x |
| 5 | 1.391 | 0.0315 | 44.16x | 0.0217 | 63.70x | 0.0319 | 44.74x | 0.0226 | 67.18x | 0.0240 | 58.52x | 0.0240 | 57.75x | 0.0238 | 58.52x | 0.0240 | 58.05x |
| 6 | 1.393 | 0.0208 | 67.10x | 0.0145 | 95.50x | 0.0242 | 57.74x | 0.0170 | 91.30x | 0.0132 | 105.08x | 0.0130 | 108.36x | 0.0127 | 107.87x | 0.0123 | 112.00x |
| 7 | 1.395 | 0.0186 | 75.14x | 0.0129 | 106.80x | 0.0225 | 63.18x | 0.0171 | 89.80x | 0.0152 | 91.86x | 0.0150 | 92.84x | 0.0150 | 94.15x | 0.0147 | 93.08x |
| 8 | 1.434 | 0.0347 | 41.38x | 0.0237 | 59.02x | 0.0339 | 42.65x | 0.0266 | 60.59x | 0.0220 | 63.90x | 0.0220 | 63.25x | 0.0217 | 63.69x | 0.0217 | 63.41x |
| 9 | 1.449 | 0.0188 | 76.98x | 0.0123 | 111.12x | 0.0235 | 58.61x | 0.0170 | 92.14x | 0.0128 | 107.53x | 0.0130 | 110.58x | 0.0127 | 108.33x | 0.0128 | 106.30x |
| 10 | 2.375 | 0.0181 | 131.21x | 0.0119 | 197.51x | 0.0265 | 93.47x | 0.0219 | 107.58x | 0.0320 | 92.80x | 0.0290 | 102.38x | 0.0297 | 99.94x | 0.0288 | 99.70x |
| 11 | 2.440 | 0.1185 | 20.60x | 0.0823 | 30.86x | 0.0465 | 56.21x | 0.0426 | 61.42x | 0.0290 | 101.17x | 0.0280 | 106.05x | 0.0285 | 104.13x | 0.0275 | 103.47x |
| 12 | 2.594 | 0.0997 | 26.03x | 0.0695 | 35.24x | 0.0390 | 64.95x | 0.0354 | 74.91x | 0.0377 | 72.82x | 0.0350 | 77.76x | 0.0355 | 76.63x | 0.0343 | 76.12x |
| 13 | 2.665 | 0.0722 | 36.93x | 0.0501 | 55.14x | 0.0388 | 72.53x | 0.0355 | 81.91x | 0.0367 | 80.74x | 0.0330 | 90.75x | 0.0335 | 89.27x | 0.0320 | 88.80x |
| 14 | 2.671 | 0.1192 | 22.41x | 0.0825 | 32.43x | 0.0463 | 61.05x | 0.0413 | 71.88x | 0.0318 | 77.30x | 0.0290 | 85.78x | 0.0292 | 83.81x | 0.0283 | 83.29x |
| 15 | 2.695 | 0.1175 | 22.94x | 0.0820 | 37.60x | 0.0491 | 58.32x | 0.0455 | 63.92x | 0.0398 | 74.52x | 0.0370 | 80.77x | 0.0372 | 79.77x | 0.0360 | 78.38x |
| 16 | 2.881 | 0.0685 | 42.04x | 0.0480 | 61.99x | 0.0391 | 78.11x | 0.0375 | 93.75x | 0.0178 | 135.08x | 0.0180 | 132.97x | 0.0185 | 130.56x | 0.0170 | 136.26x |
| 17 | 2.905 | 0.1187 | 24.47x | 0.0823 | 35.30x | 0.0468 | 65.21x | 0.0437 | 72.10x | 0.0500 | 60.80x | 0.0420 | 70.90x | 0.0427 | 70.20x | 0.0415 | 69.42x |
| 18 | 2.938 | 0.1183 | 24.84x | 0.0825 | 35.69x | 0.0486 | 64.22x | 0.0434 | 73.31x | 0.0343 | 79.22x | 0.0320 | 83.71x | 0.0323 | 84.31x | 0.0310 | 82.97x |
| 19 | 2.965 | 0.0733 | 40.48x | 0.0506 | 57.98x | 0.0414 | 73.90x | 0.0361 | 99.34x | 0.0380 | 78.94x | 0.0360 | 82.51x | 0.0360 | 82.78x | 0.0348 | 81.82x |
| 20 | 2.971 | 0.1006 | 29.54x | 0.0696 | 46.12x | 0.0432 | 71.50x | 0.0384 | 81.75x | 0.0330 | 89.54x | 0.0290 | 99.90x | 0.0297 | 98.94x | 0.0290 | 98.45x |
| 21 | 2.991 | 0.1191 | 25.12x | 0.0822 | 38.65x | 0.0582 | 53.66x | 0.0530 | 59.56x | 0.0505 | 60.12x | 0.0440 | 67.19x | 0.0455 | 66.38x | 0.0440 | 64.96x |
| 22 | 2.994 | 0.1172 | 25.55x | 0.0816 | 37.26x | 0.0546 | 54.82x | 0.0513 | 60.72x | 0.0488 | 53.18x | 0.0410 | 60.63x | 0.0418 | 60.29x | 0.0400 | 60.10x |
| 23 | 3.198 | 0.0701 | 45.65x | 0.0487 | 59.47x | 0.0410 | 75.17x | 0.0374 | 91.95x | 0.0415 | 66.61x | 0.0380 | 70.90x | 0.0395 | 70.97x | 0.0377 | 70.50x |
| **Geo** | | **0.0503** | **42.06x** | **0.0346** | **61.40x** | **0.0357** | **60.76x** | **0.0289** | **80.44x** | **0.0258** | **82.35x** | **0.0243** | **87.18x** | **0.0244** | **86.72x** | **0.0237** | **86.58x** |
| **Arith** | | **0.0665** | **48.28x** | **0.0460** | **70.39x** | **0.0374** | **61.92x** | **0.0316** | **81.90x** | **0.0288** | **84.60x** | **0.0266** | **89.18x** | **0.0269** | **88.66x** | **0.0260** | **88.69x** |
| **Geo T>2** | | |  | |  | |  | |  | **0.0361** | **78.05x** | **0.0329** | **84.82x** | **0.0335** | **83.89x** | **0.0322** | **83.43x** |
| **Arith T>2** | | |  | |  | |  | |  | **0.0372** | **80.20x** | **0.0336** | **86.59x** | **0.0343** | **85.57x** | **0.0330** | **85.30x** |

> **Adjusted (T>2):** 14 workloads where the KV-split compute kernel differs between k5, k5cl, k5cb, and kcn. WL 1–9 (T≤2) use the same fused single-block kernel. kcn gains **+6.9%** geo speedup over k5 on T>2 workloads.

---

## Final Submission Results — kernel5c_ws.py (B200, 23 workloads, ALL PASS)

`T` = number of requests (batch size). `seq_lens` = KV cache size per request (valid non-sentinel token counts).

| # | UUID | T | abs_err | Ref ms | Impl ms | Speedup | GFLOPS | seq_lens (tokens/req) |
|---:|---|---:|:---:|---:|---:|---:|---:|---|
|  1 | 0c23b10c | 1 | 6.10e-05 | 1.119 | 0.021 | 52.11x | 3.25 | [2] |
|  2 | 9d4a5f21 | 2 | 4.88e-04 | 1.364 | 0.022 | 62.30x | 46.22 | [18, 11] |
|  3 | b7668cfd | 2 | 1.95e-03 | 1.410 | 0.023 | 61.76x | 129.93 | [33, 52] |
|  4 | 0a63b87b | 2 | 4.88e-04 | 1.394 | 0.022 | 64.19x | 115.67 | [63, 9] |
|  5 | 05f6de65 | 2 | 6.10e-05 | 1.410 | 0.027 | 51.73x | 439.25 | [6, 337] |
|  6 | fc85411e | 2 | 3.05e-05 | 1.379 | 0.023 | 61.05x | 46.33 | [17, 13] |
|  7 | e6b849f2 | 2 | 1.95e-03 | 1.403 | 0.022 | 63.40x | 220.74 | [92, 48] |
|  8 | 9f3f891b | 2 | 1.22e-04 | 1.416 | 0.027 | 51.74x | 372.36 | [288, 4] |
|  9 | f77df5ce | 2 | 1.22e-04 | 1.380 | 0.023 | 61.16x | 57.22 | [18, 19] |
| 10 | 385742b2 | 8 | 3.91e-03 | 2.948 | 0.034 | 86.04x | 1693.88 | [92, 48, 1044, 14, 411, 30, 16, 8] |
| 11 | 4c46a94b | 8 | 3.91e-03 | 2.922 | 0.032 | 90.28x | 1534.37 | [18, 19, 1002, 31, 11, 316, 24, 2] |
| 12 | 38389961 | 7 | 3.91e-03 | 2.726 | 0.040 | 68.68x | 1478.66 | [33, 52, 72, 17, 18, 401, 1089] |
| 13 | 02d6ae9c | 8 | 3.91e-03 | 2.968 | 0.037 | 79.73x | 2272.40 | [63, 9, 2048, 212, 11, 25, 6, 50] |
| 14 | ddfa9e34 | 6 | 9.77e-04 | 2.470 | 0.034 | 73.45x | 1814.09 | [6, 9, 9, 14, 1639, 71] |
| 15 | 78b2e11c | 8 | 7.81e-03 | 2.983 | 0.041 | 72.40x | 2225.48 | [18, 11, 2048, 20, 25, 45, 135, 326] |
| 16 | 68d6817d | 6 | 3.91e-03 | 2.354 | 0.022 | 105.11x | 172.92 | [19, 20, 32, 12, 25, 3] |
| 17 | 564007ac | 8 | 3.91e-03 | 3.058 | 0.047 | 65.02x | 3530.64 | [288, 4, 1884, 21, 136, 2048, 42, 335] |
| 18 | ae4219a9 | 7 | 3.91e-03 | 2.697 | 0.036 | 74.24x | 2217.30 | [19, 12, 2048, 21, 26, 46, 136] |
| 19 | 232ed014 | 8 | 3.91e-03 | 2.914 | 0.040 | 73.45x | 1492.62 | [35, 54, 74, 19, 20, 403, 1091, 1] |
| 20 | 7a389715 | 8 | 1.95e-03 | 2.943 | 0.034 | 86.98x | 1817.00 | [8, 11, 11, 16, 1641, 73, 1, 1] |
| 21 | 5096e459 | 8 | 3.91e-03 | 2.925 | 0.049 | 59.58x | 3208.36 | [17, 13, 1887, 16, 180, 1986, 413, 1] |
| 22 | d57eb9e1 | 6 | 1.95e-03 | 2.495 | 0.045 | 54.95x | 2522.27 | [143, 139, 2013, 142, 306, 539] |
| 23 | 2207f0fd | 7 | 1.95e-03 | 2.742 | 0.043 | 64.07x | 2934.23 | [415, 131, 2011, 148, 263, 169, 462] |

## kernel5cl — 4-run average (no sentinel, class structure)

| # | Workload | Avg ms | Avg Speedup |
|---|----------|--------|-------------|
| 1 | 0c23b10c | 0.013 | 90.84x |
| 2 | 9d4a5f21 | 0.013 | 111.06x |
| 3 | b7668cfd | 0.013 | 103.62x |
| 4 | 0a63b87b | 0.014 | 100.69x |
| 5 | 05f6de65 | 0.024 | 57.75x |
| 6 | fc85411e | 0.013 | 108.36x |
| 7 | e6b849f2 | 0.015 | 92.84x |
| 8 | 9f3f891b | 0.022 | 63.25x |
| 9 | f77df5ce | 0.013 | 110.58x |
| 10 | 385742b2 | 0.029 | 102.38x |
| 11 | 4c46a94b | 0.028 | 106.05x |
| 12 | 38389961 | 0.035 | 77.76x |
| 13 | 02d6ae9c | 0.033 | 90.75x |
| 14 | ddfa9e34 | 0.029 | 85.78x |
| 15 | 78b2e11c | 0.037 | 80.77x |
| 16 | 68d6817d | 0.018 | 132.97x |
| 17 | 564007ac | 0.042 | 70.90x |
| 18 | ae4219a9 | 0.032 | 83.71x |
| 19 | 232ed014 | 0.036 | 82.51x |
| 20 | 7a389715 | 0.029 | 99.90x |
| 21 | 5096e459 | 0.044 | 67.19x |
| 22 | d57eb9e1 | 0.041 | 60.63x |
| 23 | 2207f0fd | 0.038 | 70.90x |
| | **Arithmetic Mean** | **0.027** | **89.18x** |
| | **Geometric Mean** | **0.024** | **87.18x** |

## kernel5cl — CUPTI measurement (B200, 23 workloads, ALL PASS)

| # | Workload | CUPTI ms | Speedup | MFLOPs ¹ | GFLOPS/s ² | MFU (FP32) ⁴ | GB/s ³ |
|---|----------|----------|---------|----------:|-----------:|-------------:|-------:|
| 1 | 0c23b10c | 0.005 | 240.30x | 0.07 | 13.96 | 0.02% | 9.1 |
| 2 | 9d4a5f21 | 0.006 | 310.56x | 1.01 | 168.66 | 0.23% | 19.9 |
| 3 | b7668cfd | 0.007 | 259.96x | 2.97 | 423.74 | 0.57% | 26.3 |
| 4 | 0a63b87b | 0.007 | 251.29x | 2.51 | 358.93 | 0.48% | 24.2 |
| 5 | 05f6de65 | 0.017 | 105.83x | 11.97 | 704.08 | 0.95% | 28.3 |
| 6 | fc85411e | 0.006 | 311.75x | 1.05 | 174.48 | 0.23% | 20.1 |
| 7 | e6b849f2 | 0.008 | 222.80x | 4.89 | 610.68 | 0.82% | 30.9 |
| 8 | 9f3f891b | 0.015 | 120.19x | 10.19 | 679.31 | 0.91% | 28.2 |
| 9 | f77df5ce | 0.006 | 310.27x | 1.29 | 215.19 | 0.29% | 21.5 |
| 10 | 385742b2 | 0.021 | 216.93x | 58.03 | 2763.43 | 3.71% | 107.6 |
| 11 | 4c46a94b | 0.020 | 231.70x | 49.66 | 2482.85 | 3.34% | 99.2 |
| 12 | 38389961 | 0.027 | 153.65x | 58.70 | 2173.89 | 2.92% | 82.9 |
| 13 | 02d6ae9c | 0.025 | 187.53x | 84.59 | 3383.52 | 4.55% | 125.5 |
| 14 | ddfa9e34 | 0.021 | 175.10x | 61.00 | 2904.68 | 3.90% | 108.2 |
| 15 | 78b2e11c | 0.028 | 162.10x | 91.71 | 3275.24 | 4.40% | 120.4 |
| 16 | 68d6817d | 0.009 | 396.47x | 3.87 | 430.38 | 0.58% | 42.9 |
| 17 | 564007ac | 0.034 | 137.34x | 166.04 | 4883.39 | 6.56% | 171.3 |
| 18 | ae4219a9 | 0.023 | 174.67x | 80.54 | 3501.74 | 4.70% | 128.7 |
| 19 | 232ed014 | 0.027 | 167.48x | 59.22 | 2193.28 | 2.95% | 85.2 |
| 20 | 7a389715 | 0.021 | 216.66x | 61.49 | 2927.94 | 3.93% | 113.1 |
| 21 | 5096e459 | 0.036 | 129.64x | 157.49 | 4374.60 | 5.88% | 154.0 |
| 22 | d57eb9e1 | 0.033 | 114.30x | 114.53 | 3470.57 | 4.66% | 122.4 |
| 23 | 2207f0fd | 0.031 | 137.57x | 125.59 | 4051.31 | 5.44% | 143.5 |
| | **Arithmetic Mean** | **0.019** | **205.83x** | **52.54** | **2007.21** | **2.70%** | **78.8** |
| | **Geometric Mean** | **0.016** | **193.37x** | **16.76** | **1076.77** | **1.45%** | **58.5** |



> ¹ **MFLOPs** — total FLOPs for the workload, computed directly from `seq_lens` with zero timing dependency:
> $$\text{FLOPs} = \sum_t V_t \cdot H \cdot (4D + 2D_p + 5) = \sum_t V_t \times 34{,}896$$
> where $V_t = \texttt{seq\_lens}[t]$ (valid KV tokens for request $t$), $H=16$, $D=512$, $D_p=64$.
>
> ² **GFLOPS/s** — achieved compute throughput: $\text{MFLOPs} \div 10^3 \div t_{\text{CUPTI}}$ (excludes host/launch overhead).
>
> ³ **GB/s** — achieved HBM read bandwidth: $B \div t_{\text{CUPTI}}$ where
> $$B = T \cdot \underbrace{(2HD + 2HD_p + 4\,\text{TOPK} + 2HD + 4H)}_{\text{per-request: }q_\text{nope},\, q_\text{pe},\, \text{indices},\, \text{output},\, \text{lse}} + V_{\text{total}} \cdot \underbrace{(2D + 2D_p)}_{\text{per-token: }ckv,\, kpe}$$
> $= T \times 43{,}072 + V_{\text{total}} \times 1{,}152$ bytes, with TOPK $= 2048$.
>
> ⁴ **MFU (FP32)** — $\text{GFLOPS/s} \div 74{,}450$ (B200 peak FP32 = 74.45 TFLOPS).
>
> **Best workload (#17, $V_\text{total}=4758$ tokens):** 4,883 GFLOPS/s vs B200 peak FP32 74.45 TFLOPS → **MFU ≈ 6.56%** (vs bf16 tensor-core peak 2,250 TFLOPS → 0.22%); 171 GB/s vs B200 peak HBM3e 8,000 GB/s → **HBM utilization ≈ 2.13%**. The bandwidth number is ~10× more meaningful than MFU, confirming this is a **memory-bandwidth-bound** kernel — the compute units are starved waiting for HBM, not the other way around. Low-token workloads (WL 1–9) are further bottlenecked by kernel **launch overhead** rather than either compute or bandwidth.

---

## kv_split_umma_v4_pdl — Benchmark Results (B200, 23 workloads, ALL PASS)

128-bit cpasync for KPE (upgraded from 32-bit): 8 threads/row → 4 KPE rows per warp per round (4 rounds total vs 16 in v3).

| # | Workload | ms | Speedup |
|---:|---|---:|---:|
|  1 | 0c23b10c | 0.008 | 147.48x |
|  2 | 9d4a5f21 | 0.010 | 157.94x |
|  3 | b7668cfd | 0.013 | 122.10x |
|  4 | 0a63b87b | 0.015 | 106.93x |
|  5 | 05f6de65 | 0.017 |  92.68x |
|  6 | fc85411e | 0.010 | 158.16x |
|  7 | e6b849f2 | 0.019 |  82.77x |
|  8 | 9f3f891b | 0.013 | 118.56x |
|  9 | f77df5ce | 0.010 | 153.27x |
| 10 | 385742b2 | 0.023 | 164.57x |
| 11 | 4c46a94b | 0.025 | 147.94x |
| 12 | 38389961 | 0.022 | 157.66x |
| 13 | 02d6ae9c | 0.022 | 166.80x |
| 14 | ddfa9e34 | 0.025 | 121.92x |
| 15 | 78b2e11c | 0.024 | 156.39x |
| 16 | 68d6817d | 0.011 | 277.55x |
| 17 | 564007ac | 0.033 | 115.92x |
| 18 | ae4219a9 | 0.023 | 148.03x |
| 19 | 232ed014 | 0.022 | 174.54x |
| 20 | 7a389715 | 0.025 | 149.63x |
| 21 | 5096e459 | 0.032 | 120.12x |
| 22 | d57eb9e1 | 0.030 | 101.49x |
| 23 | 2207f0fd | 0.025 | 142.04x |
| | **Arithmetic Mean** | **0.020** | **142.80x** |
| | **Geometric Mean** | **0.018** | **138.40x** |

---

## v4 vs kernel5cl — Comparison on High-Compute Workloads (MFLOPs > 50)

> **Note:** k5cl times are CUPTI-measured (pure GPU kernel execution, no launch overhead). v4 times are from the benchmarking framework. Both speedups are computed against the same per-workload reference time, so speedup is directly comparable; raw latency is not.

| # | UUID | MFLOPs | k5cl ms | GFLOPS/s | v4 ms | GFLOPS/s | Δ GFLOPS/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | 385742b2 |  58.03 | 0.021 | 2763 | 0.023 | 2523 |  −240 |
| 12 | 38389961 |  58.70 | 0.027 | 2174 | 0.022 | 2668 | **+494** |
| 13 | 02d6ae9c |  84.59 | 0.025 | 3384 | 0.022 | 3845 | **+461** |
| 14 | ddfa9e34 |  61.00 | 0.021 | 2905 | 0.025 | 2440 |  −465 |
| 15 | 78b2e11c |  91.71 | 0.028 | 3275 | 0.024 | 3821 | **+546** |
| 17 | 564007ac | 166.04 | 0.034 | 4883 | 0.033 | 5032 | **+149** |
| 18 | ae4219a9 |  80.54 | 0.023 | 3502 | 0.023 | 3502 |     ±0 |
| 19 | 232ed014 |  59.22 | 0.027 | 2193 | 0.022 | 2692 | **+499** |
| 20 | 7a389715 |  61.49 | 0.021 | 2928 | 0.025 | 2460 |  −468 |
| 21 | 5096e459 | 157.49 | 0.036 | 4375 | 0.032 | 4922 | **+547** |
| 22 | d57eb9e1 | 114.53 | 0.033 | 3471 | 0.030 | 3818 | **+347** |
| 23 | 2207f0fd | 125.59 | 0.031 | 4051 | 0.025 | 5024 | **+972** |
| | **Arith** | | **0.027** | **3325** | **0.026** | **3562** | **+237** |
| | **Geo**   | | **0.027** | **3233** | **0.025** | **3431** | **+198** |



I initially decided to go with the DSA track because I haven't written an attention kernel before, and this would be a very good opportunity for me to learn. To prep for this kernel, I wrote a naive implementation of flash attn in ... where we do softmax(Q@K.T) @ V in a single kernel. The flaw of this kernel is that we partition the output matrix to 2D, which hurts the performance as for each tile_N, we have to redo (N // tile_N) times the computation of Q @ K.T. Therefore, to ensure the best performance, we can only do 1D partition for the output by tiling it along the M dimension.

The concept is pretty much the same for Deepspeed Sparse Attention, except that here, we are doing Flash decoding instead, where we process only one token at a time, thanks to KV cache skip.

Can this kernel beat Flashinfer/TRT_LLM hand-crafted by Nvidia's ninjas on high throughput workloads (8x2048) ? No. But it can beat it in cases where the requests per workload are imbalanced. If these ninjas optimize this kernel further for real workloads given the ideas I presented here, it will be even harder to beat.
Decoding is really memory bound, and doesn't benefit much from B200's high throughput tensor core engines, that's why Jenson acquired Grod to offloading decoding to the GPUs.

Agentic coding:
Normally, writing kernel with a team of experienced kernel engineers to explore different ideas, different experiments. Now, with agents, a single engineer can do craft a 90-95% performance of the team-desinged kernel in the same amount of time. The engineer will first design the overall fused kernel, draft a first version by hand, then assign each step to a different agent for gradual optimization of this draft version, for even more speedup, then agent's ideas are not enough and requires human to give ideas or find ideas online and give it for it to implemenmt.

Human start with a first implementation that is runnable, shows then how to run it, it passed correctness with a base line duration (verifiable results). Intra kernel profiling can be really useful as we know the lateceny for each steps in the kernel and set that as a target for agents to optimize further, it's also important to create minimal version for each step in a separate and isolated file so each agent doesn't conflict each other.

They can be good at optimizing this early draft kernel, they did decent speedups for the softmax and really well for the output computation phase. 

Once reached a point, it's really hard for them to speed things up further, and need human guidance to use things like vectorized loads. Or for human to show them great ideas to experiment.

They can easily give up and not explore further and doesn't see the potential of a method if this experiment with this method yields slower speedup. One example is the persitent XOR KV-split kernel, they couldn't implement it to the ideas that I explained, I had to spend time wrote that entire kernel by hand, then run bench mark.

Flaws of agents: they seem to not able to search the internet and look at well optimized kernels and use ideas from those to adapt and experiment, instead they try to look for ideas already in the code base, from their kv cache or from their own weights.

What's next for optimization ?

1. For the kernel described in this doc:

We haven't tapped the tcgen05.mma, it will give us extra TMEM to store our GEMM accumulators, so the score computation can happen on tensor core (but we have to do B @ A.T = C.T, so we can use UMMA_M=128, UMMA_N can be divisible by 8, so we can make our 16 num heads divisble by UMMA_N) and overlap with the output computation happening on the CUDA core (or you can make it happen on tensor core too, if accuracies allow for it).

It's advised to use cp.async (non bulk) over TMA as the latency is much less for cp.async as we points to different KV cache page all the time due to the sparse KV cache nature of DSA, check this amazing blog from SemiAnalysis that explored these: https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor . 

Another thing is make the KV split persistent in SMEM, meaning we sub divide a block into sub-KV-split, so each 256 dim_split can be divided into 4 sub-splits, each has shape 512x64 (this constraint us to use the UMMA_M=64 instruction which is less efficient than UMMA_M=128), staging that SMEM, and overlap even more with fine-grained pipelines.

As we only use cp.async to load full rows of 512 ckv and 64 kpe to smem, this kernel essentially works for both PS = 1 and PS = 64.

2. For real-world workloads kernel:

Entropy aware compression. Entropy is a concept in information theory to determine the chaotic/randomness of data, or how much can be compressed from the data. Data with small entropy shows there's a structure/uniformity inside it and can be compressed really well (i.e. JPEG for images, MPEG for videos), but for random numbers, their entropy is infinite and can't be compressed (a large part inputs of this FlashInfer competition are sampled from `torch.rand`). For each workloads type like coding, excel sheets anaysis, law document proof read,... each has their own distribution, and thus their own entropy. If the kernel can exploit this data-dependant distribution (like how JPEG beautifully removed colors, patterns that human eyes can't see, or how MPEG only stores the difference between the key frame and its subsequent frames), we can compress really heavily the inputs to the kernel, to reduce latency moving data from GMEM to register files for these memory-bound decoding kernels. Props to the FlashInfer team to craft this real-world workloads test set, so new workload-aware kernels can be crafted that beats throughput-optimized/benchmarketing kernels, which waste too much compute calculating padded zeroes. Next step would be to craft a entropy-aware datasets that has the actual inputs distribution from various workload types. Eagle3 speculative decoding is a good example case of how different data distributions can affect speedup, with math/coding related, the symbols of output tokens are logically sequenced (think 1 + 1 = 2), hence gives better speedup on those distributions than from legal documents generation or shopping plannings.

3. Cost of agent helped coding

Coming to the competition, I received Modal GPU credits from the organizer, I also use my income to unleash all the Claude Opus that I needed. Below are the spending I used during the competition (using Github Copilot subscription summary).

Below is the budget I spent tackling these problem:
Subscription: 2 months of copilot pro +: 39 USD / moth
1. DSA attn: 
2. DSA Idxer:
3. Modal credits used:


Simulator results — under the simple model (5 µs per active split, 0 µs OOB; 16 CTAs, max-CTA latency = kernel total):

Workload 17 (T=8, mv=[288,4,1884,21,136,2048,42,335]; ideal balanced = 13.12 µs):

scheme	total µs	min/max active per CTA
baseline	40.00	1 / 8
xor	20.00	1 / 4
rot+1	20.00	2 / 4
stride=3	20.00	1 / 4
stride=5	20.00	1 / 4
stride=7	20.00	2 / 4
stride=9	20.00	2 / 4
Workload 23 (T=7, mv=[415,131,2011,148,263,169,462]; ideal balanced = 10.31 µs):

scheme	total µs	min/max active per CTA
baseline	35.00	1 / 7
xor	20.00	1 / 4
rot+1	20.00	1 / 4
stride=3	20.00	1 / 4
stride=5	20.00	1 / 4
stride=7	15.00	1 / 3
stride=9	15.00	1 / 3

---

## kernel5cl — CUPTI Latency Sorted (ascending), 23 workloads

| # | Workload | ms | MFLOPs | GFLOPS/s | GB/s | seq_lens |
|--:|---|---:|---:|---:|---:|---|
|  1 | 0c23b10c | 0.005 |   0.07 |    14.0 |   9.1 | [2] |
|  2 | 9d4a5f21 | 0.005 |   1.01 |   202.0 |  23.9 | [18, 11] |
|  3 | fc85411e | 0.005 |   1.05 |   210.0 |  24.1 | [17, 13] |
|  4 | f77df5ce | 0.005 |   1.29 |   258.0 |  25.8 | [18, 19] |
|  5 | b7668cfd | 0.007 |   2.97 |   424.3 |  26.3 | [33, 52] |
|  6 | 0a63b87b | 0.007 |   2.51 |   358.6 |  24.2 | [63, 9] |
|  7 | e6b849f2 | 0.008 |   4.89 |   611.3 |  30.9 | [92, 48] |
|  8 | 68d6817d | 0.009 |   3.87 |   430.0 |  42.9 | [19, 20, 32, 12, 25, 3] |
|  9 | 9f3f891b | 0.015 |  10.19 |   679.3 |  28.2 | [288, 4] |
| 10 | 05f6de65 | 0.017 |  11.97 |   704.1 |  28.3 | [6, 337] |
| 11 | 4c46a94b | 0.020 |  49.66 |  2483.0 |  99.2 | [18, 19, 1002, 31, 11, 316, 24, 2] |
| 12 | ddfa9e34 | 0.021 |  61.00 |  2904.8 | 108.2 | [6, 9, 9, 14, 1639, 71] |
| 13 | 385742b2 | 0.021 |  58.03 |  2763.3 | 107.6 | [92, 48, 1044, 14, 411, 30, 16, 8] |
| 14 | 7a389715 | 0.021 |  61.49 |  2928.1 | 113.1 | [8, 11, 11, 16, 1641, 73, 1, 1] |
| 15 | ae4219a9 | 0.023 |  80.54 |  3501.7 | 128.7 | [19, 12, 2048, 21, 26, 46, 136] |
| 16 | 02d6ae9c | 0.025 |  84.59 |  3383.6 | 125.5 | [63, 9, 2048, 212, 11, 25, 6, 50] |
| 17 | 38389961 | 0.027 |  58.70 |  2174.1 |  82.9 | [33, 52, 72, 17, 18, 401, 1089] |
| 18 | 232ed014 | 0.027 |  59.22 |  2193.3 |  85.2 | [35, 54, 74, 19, 20, 403, 1091, 1] |
| 19 | 78b2e11c | 0.028 |  91.71 |  3275.4 | 120.4 | [18, 11, 2048, 20, 25, 45, 135, 326] |
| 20 | 2207f0fd | 0.031 | 125.59 |  4051.3 | 143.5 | [415, 131, 2011, 148, 263, 169, 462] |
| 21 | d57eb9e1 | 0.033 | 114.53 |  3470.6 | 122.4 | [143, 139, 2013, 142, 306, 539] |
| 22 | 564007ac | 0.034 | 166.04 |  4883.5 | 171.3 | [288, 4, 1884, 21, 136, 2048, 42, 335] |
| 23 | 5096e459 | 0.036 | 157.49 |  4374.7 | 154.0 | [17, 13, 1887, 16, 180, 1986, 413, 1] |