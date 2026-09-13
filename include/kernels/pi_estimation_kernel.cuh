// =============================================================================
// pi_estimation_kernel.cuh
// -----------------------------------------------------------------------------
// This is exactly your note's first task:
//   "Try estimation of Pi by uniform distribution (RNG test)"
//
// Each thread throws `trials_per_thread` random darts at the unit square
// using dist::uniform_signed() and counts how many land inside the unit
// circle. Per-thread hit fractions are written to d_out so the engine can
// both (a) reduce them into one Pi estimate and (b) push the per-thread
// fractions as a CSV "feature" dataset.
// =============================================================================
#pragma once
#include "kernel_interface.cuh"
#include "../distributions/uniform_dist.cuh"

__global__ void pi_estimation_device_kernel(curandStatePhilox4_32_10_t* states,
                                             int n_threads,
                                             int trials_per_thread,
                                             SimulationResult* out_results) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_threads) return;

    curandStatePhilox4_32_10_t local_state = states[id];

    // Measure simulation start time
    unsigned long long start_ns = get_globaltimer_ns();

    int hits = 0;
    for (int t = 0; t < trials_per_thread; ++t) {
        float x = dist::uniform_signed(&local_state);
        float y = dist::uniform_signed(&local_state);
        if (x * x + y * y <= 1.0f) hits++;
    }

    // Measure simulation end time
    unsigned long long end_ns = get_globaltimer_ns();
    float time_us = (end_ns >= start_ns) ? static_cast<float>(end_ns - start_ns) / 1000.0f : 0.0f;

    // 1 thread performs 1 simulation: compute estimate and error
    float pi_estimate = 4.0f * static_cast<float>(hits) / static_cast<float>(trials_per_thread);
    float error = fabsf(pi_estimate - 3.14159265358979323846f);

    out_results[id].estimate = pi_estimate;
    out_results[id].error = error;
    out_results[id].start_ns = start_ns;
    out_results[id].end_ns = end_ns;
    out_results[id].time_us = time_us;

    states[id] = local_state;   // persist the advanced RNG state
}

class PiEstimationKernel : public ApplicationKernel {
public:
    void launch(const RNGEngine& rng, int trials_per_thread,
                SimulationResult* d_out) const override {
        const int block_size = 256;
        const int grid_size  = (rng.num_threads() + block_size - 1) / block_size;
        pi_estimation_device_kernel<<<grid_size, block_size>>>(
            rng.states(), rng.num_threads(), trials_per_thread, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int out_size(const RNGEngine& rng) const override { return rng.num_threads(); }
    const char* csv_header() const override { return "thread_id,estimate,error,start_time_us,end_time_us,duration_us"; }
    const char* name() const override { return "pi_estimation"; }
    double ground_truth() const override { return 3.14159265358979323846; }
};
