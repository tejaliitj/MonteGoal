#include "uniform_philox.cuh"
#include <curand_kernel.h>

__global__ void uniform_philox_kernel(float* out, int n,
                                       unsigned long long seed,
                                       unsigned long long counter_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Each thread owns a distinct subsequence: counter_offset (this
    // launch's slice of the stream) + idx (this thread's slot within it).
    // That is what makes launch #2's thread 0 statistically independent
    // from launch #1's thread 0.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, counter_offset + idx, 0, &state);

    out[idx] = curand_uniform(&state);
}

void launch_uniform_philox(float* d_out, int n,
                            unsigned long long seed,
                            unsigned long long counter_offset) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    uniform_philox_kernel<<<blocks, threads>>>(d_out, n, seed, counter_offset);
}
