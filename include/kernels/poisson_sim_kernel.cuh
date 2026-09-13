// =============================================================================
// poisson_sim_kernel.cuh
// -----------------------------------------------------------------------------
// A second "application" for the engine, showing how you swap kernels
// (your note: "Change kernel for different applications") without touching
// the RNG engine, the distribution modules, or main.cu's driver logic.
//
// Each thread draws `trials_per_thread` Poisson(lambda) samples and writes
// their mean to d_out -- a simple per-thread "feature" that gets pushed to
// the CSV dataset, e.g. simulating arrivals-per-interval in a queueing model.
// =============================================================================
#pragma once
#include "kernel_interface.cuh"
#include "../distributions/poisson_dist.cuh"

// lambda is fixed here for simplicity; expose it as a constructor parameter
// if you need it to vary per run.
__global__ void poisson_sim_device_kernel(curandStatePhilox4_32_10_t* states,
                                           int n_threads,
                                           int trials_per_thread,
                                           double lambda,
                                           SimulationResult* out_results) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_threads) return;

    curandStatePhilox4_32_10_t local_state = states[id];

    // Measure simulation start time
    unsigned long long start_ns = get_globaltimer_ns();

    unsigned long long total = 0;
    for (int t = 0; t < trials_per_thread; ++t) {
        total += dist::poisson(&local_state, lambda);
    }

    // Measure simulation end time
    unsigned long long end_ns = get_globaltimer_ns();
    float time_us = (end_ns >= start_ns) ? static_cast<float>(end_ns - start_ns) / 1000.0f : 0.0f;

    // 1 thread performs 1 simulation: compute mean count and error
    float mean_count = static_cast<float>(total) / static_cast<float>(trials_per_thread);
    float error = fabsf(mean_count - static_cast<float>(lambda));

    out_results[id].estimate = mean_count;
    out_results[id].error = error;
    out_results[id].start_ns = start_ns;
    out_results[id].end_ns = end_ns;
    out_results[id].time_us = time_us;

    states[id] = local_state;
}

class PoissonSimKernel : public ApplicationKernel {
public:
    explicit PoissonSimKernel(double lambda = 4.0) : lambda_(lambda) {}

    void launch(const RNGEngine& rng, int trials_per_thread,
                SimulationResult* d_out) const override {
        const int block_size = 256;
        const int grid_size  = (rng.num_threads() + block_size - 1) / block_size;
        poisson_sim_device_kernel<<<grid_size, block_size>>>(
            rng.states(), rng.num_threads(), trials_per_thread, lambda_, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int out_size(const RNGEngine& rng) const override { return rng.num_threads(); }
    const char* csv_header() const override { return "thread_id,estimate,error,start_time_us,end_time_us,duration_us"; }
    const char* name() const override { return "poisson_simulation"; }
    double ground_truth() const override { return lambda_; }

private:
    double lambda_;
};
