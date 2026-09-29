// #pragma once

// constexpr int THREADS_PER_BLOCK = 256;

// inline int blocks_for(int n)
// {
//     return (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
// }





#pragma once

#include <cuda_runtime.h>

void check_cuda_error(
    cudaError_t error,
    const char* file,
    int line
);

#define CUDA_CHECK(call) \
    check_cuda_error((call), __FILE__, __LINE__)