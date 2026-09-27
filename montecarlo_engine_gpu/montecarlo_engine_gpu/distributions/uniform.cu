#include "uniform.cuh"

__device__ float generate_uniform(
    curandStatePhilox4_32_10_t *state)
{
    return curand_uniform(state);
}
