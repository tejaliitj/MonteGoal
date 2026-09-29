#pragma once

#include <curand_kernel.h>
#include "philox_rng.cuh"

// Draws one uniform float in (0, 1] from the calling thread's RngState.
// Thin engine-owned wrapper, same pattern as poisson_sample.cuh's
// sample_goals() -- domain code (e.g. a penalty-shootout coin-flip) goes
// through this instead of calling curand_uniform directly, so every RNG draw
// in an application has exactly one engine-owned entry point per distribution.
__device__ __forceinline__ float sample_uniform01(RngState* state) {
    return curand_uniform(state);
}
