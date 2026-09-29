#pragma once

#include <curand_kernel.h>

// This file IS the engine's RNG choice. Change what it defines here -- and
// nothing else -- to use a different generator throughout the whole engine.
// RngState / rng_init is the contract every other engine and domain file is
// written against; no other file names curandStatePhilox4_32_10_t directly.
// Swapping RNGs for a different application (or a different generator for
// this one) means writing a different version of this one file, not editing
// the kernel, the sampling primitives, or any domain logic.
using RngState = curandStatePhilox4_32_10_t;

// seed stays fixed for a whole run; subsequence is what gives each trial (or
// each kernel launch, via counter_offset) its own non-overlapping stream --
// see monte_carlo_kernel.cuh for how the two get combined.
__device__ __forceinline__ void rng_init(unsigned long long seed,
                                          unsigned long long subsequence,
                                          unsigned long long offset,
                                          RngState* state) {
    curand_init(seed, subsequence, offset, state);
}
