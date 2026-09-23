#ifndef PHILOX_RNG_CUH
#define PHILOX_RNG_CUH

#include <curand_kernel.h>
#include "utils.cuh"

__global__ void initializeRNG(curandStatePhilox4_32_10_t* state, unsigned long long seed) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, thread_id, 0, &state[thread_id]);
}

#endif