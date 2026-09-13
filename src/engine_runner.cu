// src/engine_runner.cu
#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "include/kernels/pi_estimation_kernel.cuh"

extern "C" {
    int run_pi_simulation(int num_threads, int trials_per_thread, unsigned long long seed, double* pi_out) {
        curandStatePhilox4_32_10_t* d_states = nullptr;
        int* d_results = nullptr;
        
        size_t states_size = num_threads * sizeof(curandStatePhilox4_32_10_t);
        size_t results_size = num_threads * sizeof(int);

        cudaMalloc(&d_states, states_size);
        cudaMalloc(&d_results, results_size);

        int threadsPerBlock = 256;
        int blocks = (num_threads + threadsPerBlock - 1) / threadsPerBlock;

        // Initialize Philox RNG per thread
        setup_philox_rng_kernel<<<blocks, threadsPerBlock>>>(d_states, seed, num_threads);
        cudaDeviceSynchronize();

        // Run Monte Carlo simulation
        pi_simulation_kernel<<<blocks, threadsPerBlock>>>(d_states, trials_per_thread, d_results, num_threads);
        cudaDeviceSynchronize();

        // Copy per-thread results back to host
        std::vector<int> h_results(num_threads);
        cudaMemcpy(h_results.data(), d_results, results_size, cudaMemcpyDeviceToHost);

        cudaFree(d_states);
        cudaFree(d_results);

        // Aggregate results across all threads
        long long total_inside = 0;
        for (int hits : h_results) {
            total_inside += hits;
        }

        long long total_trials = (long long)num_threads * trials_per_thread;
        *pi_out = 4.0 * ((double)total_inside / (double)total_trials);

        return 0;
    }
}