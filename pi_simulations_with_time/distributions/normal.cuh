#ifndef NORMAL_CUH
#define NORMAL_CUH

#include <curand_kernel.h>

__device__ float generate_normal(
    curandStatePhilox4_32_10_t *state,
    float mean,
    float stddev
);

#endif
