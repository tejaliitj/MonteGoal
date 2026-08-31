#include "rng.cuh"

__global__ void setup_rng(
    curandStatePhilox4_32_10_t *states,
    unsigned long long seed,
    int total_threads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_threads)
        return;

    curand_init(
        seed,
        idx,
        0,
        &states[idx]
    );
}
