#ifndef UNIFORM_CUH
#define UNIFORM_CUH

#include <curand_kernel.h>

__device__ float generate_uniform(
    curandStatePhilox4_32_10_t *state
);

#endif
