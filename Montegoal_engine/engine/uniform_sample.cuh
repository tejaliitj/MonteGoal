#pragma once

// =============================================================================
// engine/uniform_sample.cuh  --  the one canonical uniform draw.
//
// Same pattern as poisson_sample.cuh: domain code (e.g. the penalty-shootout
// coin flip) goes through this instead of calling curand_uniform directly, so
// every RNG draw in an application has exactly one engine-owned entry point per
// distribution.
// =============================================================================

#include "rng.cuh"

// Draws one uniform float in (0, 1] from the calling thread's RNG state.
__device__ __forceinline__ float sample_uniform01(RngState* state) {
    return curand_uniform(state);
}
