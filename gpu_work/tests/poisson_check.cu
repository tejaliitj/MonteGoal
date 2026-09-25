#include "poisson_check.cuh"
#include "poisson_sample.cuh"
#include <curand_kernel.h>
#include <math.h>

__global__ void poisson_check_kernel(float* abs_error_out, int num_trials,
                                      long long draws_per_trial, double lambda,
                                      unsigned long long seed,
                                      unsigned long long counter_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_trials) return;

    // One subsequence per trial -- same discipline as every other kernel
    // in this project. All draws_per_trial draws for this trial come
    // from this thread's own advancing Philox state.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, counter_offset + idx, 0, &state);

    long long total_goals = 0;
    for (long long i = 0; i < draws_per_trial; ++i) {
        total_goals += sample_goals(&state, lambda);
    }

    double empirical_mean = (double)total_goals / (double)draws_per_trial;
    abs_error_out[idx] = (float)fabs(empirical_mean - lambda);
}

void launch_poisson_check(float* d_abs_error, int num_trials,
                           long long draws_per_trial, double lambda,
                           unsigned long long seed,
                           unsigned long long counter_offset) {
    int threads = 128;
    int blocks = (num_trials + threads - 1) / threads;
    poisson_check_kernel<<<blocks, threads>>>(d_abs_error, num_trials,
                                               draws_per_trial, lambda,
                                               seed, counter_offset);
}
