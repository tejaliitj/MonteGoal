#include "rng/rng_engine.cuh"
#include <random>
#include <iostream>
#include <cstdlib>

unsigned long long generate_seed()
{
    std::random_device rd;
    return (static_cast<unsigned long long>(rd()) << 32) |
           static_cast<unsigned long long>(rd());
}

__global__ void setup_philox_states(
    curandStatePhilox4_32_10_t* states,
    unsigned long long seed,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        curand_init(seed, idx, 0, &states[idx]);
}

void check_cuda(cudaError_t err, const char* operation)
{
    if (err != cudaSuccess) {
        std::cerr << operation << " failed: "
                  << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}
