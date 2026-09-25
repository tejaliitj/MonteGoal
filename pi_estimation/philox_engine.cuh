#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "uniform_philox.cuh"
#include "pi_trials.cuh"
#include "cuda_check.cuh"

// Owns one seed and a running subsequence counter shared across every
// sample_* call, so consecutive kernel launches (e.g. successive
// tournament rounds) always draw from non-overlapping random streams.
// sample_poisson() will be added the same way once poisson_philox.cu
// exists -- same counter, same pattern, different kernel.
class PhiloxEngine {
public:
    explicit PhiloxEngine(unsigned long long seed) : seed_(seed), counter_(0) {}

    // Returns a device pointer to n uniform floats in (0, 1].
    // Caller owns the pointer and must cudaFree it.
    float* sample_uniform(int n) {
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

        launch_uniform_philox(d_out, n, seed_, counter_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        counter_ += n;  // next launch starts past every subsequence used here
        return d_out;
    }

    // Runs num_simulations independent Monte Carlo pi estimates in
    // parallel (one thread per simulation, each drawing
    // points_per_simulation points internally). Returns a device pointer
    // to num_simulations abs-error floats -- caller must cudaFree it.
    float* sample_pi_trials(int num_simulations, long long points_per_simulation,
                             double pi_actual) {
        float* d_abs_error;
        CUDA_CHECK(cudaMalloc(&d_abs_error, num_simulations * sizeof(float)));

        launch_pi_trials(d_abs_error, num_simulations, points_per_simulation,
                          seed_, counter_, pi_actual);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        counter_ += num_simulations;  // one subsequence slot per simulation/thread
        return d_abs_error;
    }

private:
    unsigned long long seed_;
    unsigned long long counter_;
};
