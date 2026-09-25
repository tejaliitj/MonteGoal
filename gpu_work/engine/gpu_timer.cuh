#pragma once

#include <cuda_runtime.h>

// Wraps CUDA events for GPU-side timing. Measures device work only --
// kernel launch through completion -- not host-side cost like malloc,
// memcpy, or printf. Reusable around any kernel launch, not just this one.
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { cudaEventRecord(start_); }

    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};
