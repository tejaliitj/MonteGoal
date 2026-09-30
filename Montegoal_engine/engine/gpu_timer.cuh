#pragma once

// =============================================================================
// engine/gpu_timer.cuh  --  GPU-side timing with CUDA events.
//
// CUDA events are timestamps recorded *on the GPU's own timeline*. Recording one
// before and one after a kernel launch measures just the device work between
// them (kernel start -> kernel end), with ~0.5 microsecond resolution, and
// excludes host-side costs such as cudaMalloc, cudaMemcpy, printf, or CUDA
// context creation. Reusable around any kernel launch, not just this project's.
//
// Usage:
//     GpuTimer t;
//     t.start();
//     my_kernel<<<blocks, threads>>>(...);
//     t.stop();                       // blocks the CPU until the GPU reaches the stop event
//     float ms = t.elapsed_ms();
// =============================================================================

#include <cuda_runtime.h>
#include "cuda_check.cuh"

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    // Events are owned handles -- forbid accidental copies (would double-destroy).
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // Enqueues the "start" timestamp on the default stream. Returns immediately.
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    // Enqueues the "stop" timestamp, then waits until the GPU has actually
    // reached it -- after this returns, the kernel is finished and
    // elapsed_ms() is valid.
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    // Milliseconds between the start and stop events (device time only).
    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};
