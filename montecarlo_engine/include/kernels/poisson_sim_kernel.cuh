// #pragma once
// #include <curand_kernel.h>

// __global__ void generate_poisson_kernel(
//     curandStatePhilox4_32_10_t* states,
//     unsigned int* results,
//     double lambda,
//     int n
// );



#pragma once

#include <cuda_runtime.h>

void launch_poisson_simulation(
    double lambda,
    int* d_counts,
    int num_samples,
    unsigned long long seed,
    cudaStream_t stream = 0
);