#pragma once

// =============================================================================
// engine/poisson_sample.cuh  --  the one canonical Poisson draw.
//
// Every goal count drawn anywhere in the engine goes through sample_goals(), so
// call sites read in domain terms (goals, lambda) rather than raw cuRAND, and
// there is exactly one place to change if the sampling method ever changes.
//
// RngState comes from rng/rng.cuh, so this works unchanged with Philox, XORWOW
// or MRG32k3a -- curand_poisson is overloaded for all of them.
// =============================================================================

#include "rng.cuh"

// Draws one Poisson-distributed integer with mean `lambda` from the calling
// thread's own RNG state. The caller owns `state`: initialise it once per
// thread with rng_init(), then call this as often as needed -- the state
// advances internally on every draw.
__device__ __forceinline__ int sample_goals(RngState* state, double lambda) {
    return (int)curand_poisson(state, lambda);
}
