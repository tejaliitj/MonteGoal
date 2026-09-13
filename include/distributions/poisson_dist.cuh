// =============================================================================
// poisson_dist.cuh   (Distribution folder -> Poisson)
// -----------------------------------------------------------------------------
// Device-side helper for drawing Poisson-distributed integers from a
// per-thread Philox state. cuRAND exposes curand_poisson() directly in the
// device API (curand_kernel.h) -- it internally picks the fastest algorithm
// for the given lambda (table lookup for small lambda, transformed rejection
// for large lambda), so we don't have to implement that ourselves.
// =============================================================================
#pragma once
#include <curand_kernel.h>

namespace dist {

// Draw one Poisson(lambda) sample.
__device__ __forceinline__
unsigned int poisson(curandStatePhilox4_32_10_t* state, double lambda) {
    return curand_poisson(state, lambda);
}

} // namespace dist
