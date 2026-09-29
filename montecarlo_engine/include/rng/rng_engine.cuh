// #pragma once
// #include <curand_kernel.h>
// #include <cuda_runtime.h>

// unsigned long long generate_seed();

// __global__ void setup_philox_states(
//     curandStatePhilox4_32_10_t* states,
//     unsigned long long seed,
//     int n
// );

// void check_cuda(cudaError_t err, const char* operation);



// #pragma once

// #include <curand_kernel.h>

// __device__ inline void init_philox(
//     curandStatePhilox4_32_10_t* state,
//     unsigned long long seed,
//     unsigned long long sequence
// ) {
//     curand_init(seed, sequence, 0, state);
// }






#pragma once

#include <curand_kernel.h>

__device__ inline void init_philox(
    curandStatePhilox4_32_10_t* state,
    unsigned long long seed,
    unsigned long long sequence
) {
    curand_init(
        seed,
        sequence,
        0,
        state
    );
}