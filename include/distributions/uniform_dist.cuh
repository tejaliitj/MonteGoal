// =============================================================================
// uniform_dist.cuh   (Distribution folder -> Uniform)
// -----------------------------------------------------------------------------
// Device-side helper functions for drawing uniformly distributed numbers from
// a per-thread Philox state. Every "application kernel" in include/kernels/
// includes this when it needs uniform randomness (e.g. the Pi-estimation
// RNG test).
// =============================================================================
#pragma once
#include <curand_kernel.h>

namespace dist {

// Uniform sample in (0, 1]  -- direct cuRAND primitive.
__device__ __forceinline__
float uniform01(curandStatePhilox4_32_10_t* state) {
    return curand_uniform(state);
}

// Uniform sample in [lo, hi)
__device__ __forceinline__
float uniform_range(curandStatePhilox4_32_10_t* state, float lo, float hi) {
    return lo + (hi - lo) * curand_uniform(state);
}

// Uniform sample mapped onto [-1, 1) -- used by the Pi-estimation "dart throw"
__device__ __forceinline__
float uniform_signed(curandStatePhilox4_32_10_t* state) {
    return curand_uniform(state) * 2.0f - 1.0f;
}

} // namespace dist
