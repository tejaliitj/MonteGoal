#pragma once

#include <curand_kernel.h>

// Draws one uniform float in (0, 1] from the calling thread's Philox state.
// Thin engine-owned wrapper, same pattern as poisson_sample.cuh's
// sample_goals() -- domain code (e.g. a penalty-shootout coin-flip) goes
// through this instead of calling curand_uniform directly, so every RNG draw
// in an application has exactly one engine-owned entry point per distribution.
__device__ __forceinline__ float sample_uniform01(curandStatePhilox4_32_10_t* state) {
    return curand_uniform(state);
}
