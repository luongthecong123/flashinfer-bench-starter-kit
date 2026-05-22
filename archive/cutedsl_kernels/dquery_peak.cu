#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

__global__ void dummyKernel() {}

int main() {
    int devId;
    cudaGetDevice(&devId);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);

    int clockRateKHz, memClockRateKHz, memBusWidth;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, devId);
    cudaDeviceGetAttribute(&memClockRateKHz, cudaDevAttrMemoryClockRate, devId);
    cudaDeviceGetAttribute(&memBusWidth, cudaDevAttrGlobalMemoryBusWidth, devId);

    int smemPerBlock, smemPerSM, l2Size, maxThreadsBlock, maxBlocksSM, maxThreadsSM, regsSM;
    cudaDeviceGetAttribute(&smemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, devId);
    cudaDeviceGetAttribute(&smemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, devId);
    cudaDeviceGetAttribute(&l2Size, cudaDevAttrL2CacheSize, devId);
    cudaDeviceGetAttribute(&maxThreadsBlock, cudaDevAttrMaxThreadsPerBlock, devId);
    cudaDeviceGetAttribute(&maxBlocksSM, cudaDevAttrMaxBlocksPerMultiprocessor, devId);
    cudaDeviceGetAttribute(&maxThreadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, devId);
    cudaDeviceGetAttribute(&regsSM, cudaDevAttrMaxRegistersPerMultiprocessor, devId);

    int coresPerSM = 128, tcPerSM = 4, bf16FlopsPerTCPerCycle = 2048; // defaults (Blackwell)
    int cc = prop.major * 10 + prop.minor;
    switch (cc) {
        case 70: coresPerSM = 64;  tcPerSM = 8; bf16FlopsPerTCPerCycle = 128;  break; // V100
        case 75: coresPerSM = 64;  tcPerSM = 8; bf16FlopsPerTCPerCycle = 128;  break; // T4
        case 80: coresPerSM = 64;  tcPerSM = 4; bf16FlopsPerTCPerCycle = 512;  break; // A100
        case 86: coresPerSM = 128; tcPerSM = 4; bf16FlopsPerTCPerCycle = 256;  break; // 3090/A6000
        case 89: coresPerSM = 128; tcPerSM = 4; bf16FlopsPerTCPerCycle = 256;  break; // 4090/L40
        case 90: coresPerSM = 128; tcPerSM = 4; bf16FlopsPerTCPerCycle = 1024; break; // H100
        case 100:coresPerSM = 128; tcPerSM = 4; bf16FlopsPerTCPerCycle = 2048; break; // B200
    }

    int numSMs = prop.multiProcessorCount;
    int totalCores = coresPerSM * numSMs;
    int totalTC = tcPerSM * numSMs;
    double boostClockHz = clockRateKHz * 1e3;
    double boostClockMHz = clockRateKHz / 1000.0;

    // FP32 Peak = total_cores * 2 (FMA) * boost_clock
    double fp32Peak_TFLOPS = (double)totalCores * 2.0 * boostClockHz / 1e12;

    // BF16 TC Peak = num_TC * bf16_flops_per_TC_per_cycle * boost_clock
    double bf16Peak_TFLOPS = (double)totalTC * bf16FlopsPerTCPerCycle * boostClockHz / 1e12;

    // Memory BW = mem_clock * 2 (DDR) * bus_width / 8
    double memClockMHz = memClockRateKHz / 1000.0;
    double effectiveDataRateGbps = memClockMHz * 2.0 / 1000.0;
    double theoBW_GBs = effectiveDataRateGbps * memBusWidth / 8.0;

    // ── Print ──
    cout << "===============================" << endl;
    cout << " CUDA Device Report" << endl;
    cout << "===============================" << endl;

    cout << endl << "=== Device Overview ===" << endl;
    cout << "Device Name:          " << prop.name << endl;
    cout << "Compute Capability:   sm_" << prop.major << prop.minor << endl;

    cout << endl << "=== Memory ===" << endl;
    cout << "Global Memory:        " << prop.totalGlobalMem / (1024LL*1024*1024) << " GB" << endl;
    cout << "L2 Cache:             " << l2Size / (1024*1024) << " MB (" << l2Size / 1024 << " KB)" << endl;
    cout << "Shared Memory/Block:  " << smemPerBlock / 1024 << " KB" << endl;
    cout << "Shared Memory/SM:     " << smemPerSM / 1024 << " KB" << endl;
    cout << "Memory Bus Width:     " << memBusWidth << " bits" << endl;
    cout << "Memory Clock:         " << (int)memClockMHz << " MHz" << endl;
    cout << fixed << setprecision(1);
    cout << "Effective Data Rate:  " << effectiveDataRateGbps << " Gbps/pin" << endl;
    cout << "Theoretical BW:       " << theoBW_GBs << " GB/s (" << theoBW_GBs / 1000.0 << " TB/s)" << endl;

    cout << endl << "=== Compute ===" << endl;
    cout << "SM Count:             " << numSMs << endl;
    cout << "Boost Clock:          " << (int)boostClockMHz << " MHz" << endl;
    cout << "CUDA Cores/SM:        " << coresPerSM << endl;
    cout << "Total CUDA Cores:     " << totalCores << endl;
    cout << "Tensor Cores/SM:      " << tcPerSM << endl;
    cout << "Total Tensor Cores:   " << totalTC << endl;
    cout << "Max Threads/Block:    " << maxThreadsBlock << endl;
    cout << "Max Blocks/SM:        " << maxBlocksSM << endl;
    cout << "Max Threads/SM:       " << maxThreadsSM << endl;
    cout << "Warp Size:            " << prop.warpSize << endl;
    cout << "Registers/Block:      " << prop.regsPerBlock << endl;
    cout << "Registers/SM:         " << regsSM << endl;

    cout << endl << "=== Theoretical Peak FLOPS ===" << endl;
    cout << "FP32 Peak:            " << fp32Peak_TFLOPS << " TFLOPS" << endl;
    cout << "  = " << numSMs << " SMs * " << coresPerSM << " cores/SM * 2 FLOP/core/cycle * " << (int)boostClockMHz << " MHz" << endl;
    cout << "BF16 TC Peak:         " << bf16Peak_TFLOPS << " TFLOPS" << endl;
    cout << "  = " << (int)boostClockMHz << " MHz * " << totalTC << " TCs * " << bf16FlopsPerTCPerCycle << " FLOP/TC/cycle" << endl;

    cout << endl << "=== Roofline Ridge Points ===" << endl;
    double fp32Ridge = fp32Peak_TFLOPS * 1e12 / (theoBW_GBs * 1e9);
    double bf16Ridge = bf16Peak_TFLOPS * 1e12 / (theoBW_GBs * 1e9);
    cout << "FP32 Ridge:           " << fp32Ridge << " FLOP/byte" << endl;
    cout << "BF16 TC Ridge:        " << bf16Ridge << " FLOP/byte" << endl;

    return 0;
}