#ifndef UNIFORM_DISTRIBUTION_CUH
#define UNIFORM_DISTRIBUTION_CUH

#include <curand_kernel.h>

__device__ inline void generateUniformXY(curandStatePhilox4_32_10_t* state, double& x, double& y) {
    double u1 = curand_uniform_double(state);
    double u2 = curand_uniform_double(state);
    
    x = 2.0 * u1 - 1.0;
    y = 2.0 * u2 - 1.0;
}

#endif