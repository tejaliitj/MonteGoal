#pragma once

// =============================================================================
// rng/rng_mrg32k3a.cuh  --  MRG32k3a (L'Ecuyer combined multiple recursive
// generator), cuRAND's other long-period generator with formal stream support.
//
// Included for the RNG speed comparison. Like XORWOW, curand_init() with a
// non-zero `subsequence` performs a skip-ahead (subsequences are 2^76 apart),
// which is costlier than Philox's counter-based start. Its state is small
// (6 x 32-bit words) but each draw does modular multiply arithmetic.
// =============================================================================

#include <curand_kernel.h>

#define RNG_NAME "mrg32k3a"

using RngState = curandStateMRG32k3a_t;

// Same contract as every other rng_<name>.cuh -- see rng_philox.cuh.
__device__ __forceinline__ void rng_init(unsigned long long seed,
                                          unsigned long long subsequence,
                                          unsigned long long offset,
                                          RngState* state) {
    curand_init(seed, subsequence, offset, state);
}
