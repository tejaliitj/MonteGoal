#ifndef RNG_CUH
#define RNG_CUH

#include <curand_kernel.h>

__global__ void setup_rng(
    curandStatePhilox4_32_10_t *states,
    unsigned long long seed,
    int total_threads
);

#endif
