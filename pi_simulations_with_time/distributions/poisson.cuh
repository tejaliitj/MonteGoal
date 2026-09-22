#ifndef POISSON_CUH
#define POISSON_CUH

#include <curand_kernel.h>

__device__ unsigned int generate_poisson(
    curandStatePhilox4_32_10_t *state,
    double lambda
);

#endif
