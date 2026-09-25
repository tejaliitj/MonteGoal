#include "pi_trials.cuh"
#include <curand_kernel.h>
#include <math.h>

__global__ void pi_trials_kernel(float* abs_error_out, int num_simulations,
                                  long long points_per_simulation,
                                  unsigned long long seed,
                                  unsigned long long counter_offset,
                                  double pi_actual) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_simulations) return;

    // One subsequence per SIMULATION (not per draw). All
    // points_per_simulation draws for this simulation come from this
    // thread's own Philox stream -- curand_uniform() just advances the
    // state internally, no re-init between draws needed.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, counter_offset + idx, 0, &state);

    long long inside = 0;
    for (long long i = 0; i < points_per_simulation; ++i) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f) inside++;
    }

    double pi_estimate = 4.0 * (double)inside / (double)points_per_simulation;
    abs_error_out[idx] = (float)fabs(pi_estimate - pi_actual);
}

void launch_pi_trials(float* d_abs_error, int num_simulations,
                       long long points_per_simulation,
                       unsigned long long seed,
                       unsigned long long counter_offset,
                       double pi_actual) {
    int threads = 128;  // fewer than the uniform kernel -- each thread does real work now
    int blocks = (num_simulations + threads - 1) / threads;
    pi_trials_kernel<<<blocks, threads>>>(d_abs_error, num_simulations,
                                           points_per_simulation, seed,
                                           counter_offset, pi_actual);
}
