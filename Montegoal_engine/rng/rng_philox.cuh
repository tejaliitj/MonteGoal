#pragma once

// =============================================================================
// rng/rng_philox.cuh  --  Philox4x32-10 (cuRAND device API). DEFAULT generator.
//
// Why Philox: it is a COUNTER-BASED generator. Any (seed, subsequence, offset)
// triple directly addresses an independent block of random numbers, so every
// thread can own a distinct, non-overlapping stream just by using a different
// `subsequence` -- no mutable global state, no risk of two threads sharing a
// stream. curand_init for Philox is also cheap (no expensive skip-ahead), which
// matters here because every trial initialises its own state.
// =============================================================================

#include <curand_kernel.h>

#define RNG_NAME "philox4_32_10"

using RngState = curandStatePhilox4_32_10_t;

// Seeds one thread's stream.
//   seed        - fixed for a whole run (same seed => same results, always)
//   subsequence - per-trial stream id; the engine passes counter_offset + trial_index
//   offset      - starting position inside that stream (the engine always uses 0)
__device__ __forceinline__ void rng_init(unsigned long long seed,
                                          unsigned long long subsequence,
                                          unsigned long long offset,
                                          RngState* state) {
    curand_init(seed, subsequence, offset, state);
}
