#pragma once

// =============================================================================
// engine/cuda_check.cuh  --  error-checking macro for every CUDA runtime call.
//
// CUDA calls fail silently unless you look at the returned cudaError_t. Wrap
// any call in CUDA_CHECK(...) and the program prints the file/line and exits
// on the first failure instead of producing garbage results.
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(err), __FILE__, __LINE__);         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)
