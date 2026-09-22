#include "normal.cuh"

__device__ float generate_normal(
    curandStatePhilox4_32_10_t *state,
    float mean,
    float stddev)
{
    return curand_normal(state) * stddev + mean;
}
