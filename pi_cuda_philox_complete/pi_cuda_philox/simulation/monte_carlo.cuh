#ifndef MONTE_CARLO_CUH
#define MONTE_CARLO_CUH

#include <curand_kernel.h>

__global__ void monte_carlo_kernel(
    curandStatePhilox4_32_10_t *states,
    int points_per_thread,
    int total_threads,
    float *thread_pi_results
);

#endif
