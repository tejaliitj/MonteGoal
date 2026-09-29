// #pragma once
// #include <curand_kernel.h>

// __device__ inline float sample_uniform(
//     curandStatePhilox4_32_10_t* state)
// {
//     return curand_uniform(state);
// }



#pragma once

#include <curand_kernel.h>

__device__ inline double sample_uniform(
    curandStatePhilox4_32_10_t* state
) {
    return static_cast<double>(
        curand_uniform(state)
    );
}
