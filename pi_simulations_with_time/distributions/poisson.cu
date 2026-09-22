#include "poisson.cuh"

__device__ unsigned int generate_poisson(
    curandStatePhilox4_32_10_t *state,
    double lambda)
{
    return curand_poisson(state, lambda);
}
