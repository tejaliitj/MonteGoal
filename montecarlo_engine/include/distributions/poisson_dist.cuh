// #pragma once
// #include <curand_kernel.h>

// __device__ inline unsigned int sample_poisson(
//     curandStatePhilox4_32_10_t* state,
//     double lambda)
// {
//     return curand_poisson(state, lambda);
// }





#pragma once

#include <curand_kernel.h>

__device__ inline int sample_poisson(
    curandStatePhilox4_32_10_t* state,
    double lambda
) {
    if (lambda <= 0.0) {
        return 0;
    }

    return static_cast<int>(
        curand_poisson(state, lambda)
    );
}