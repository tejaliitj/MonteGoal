// #pragma once
// #include <curand_kernel.h>

// __global__ void estimate_pi_kernel(
//     curandStatePhilox4_32_10_t* states,
//     unsigned long long* inside,
//     int n
// );




#pragma once

#include <cuda_runtime.h>

void launch_pi_estimation(
    int* d_inside,
    unsigned long long points_per_thread,
    unsigned long long num_threads,
    unsigned long long seed,
    cudaStream_t stream = 0
);