#pragma once

// =============================================================================
// rng/rng_xorwow.cuh  --  XORWOW (cuRAND's classic default generator).
//
// Included for the RNG speed comparison. Things worth knowing when reading the
// benchmark numbers:
//   * Its state is larger than Philox's (6 x 32-bit words + a Weyl counter +
//     Box-Muller caches), so it uses more registers / local memory per thread.
//   * curand_init() with a non-zero `subsequence` must SKIP AHEAD by
//     subsequence * 2^67 steps (done with matrix powers), which is much more
//     expensive than Philox's initialisation. With one fresh state per trial
//     this init cost is part of what the benchmark measures.
// =============================================================================

#include <curand_kernel.h>

#define RNG_NAME "xorwow"

using RngState = curandStateXORWOW_t;

// Same contract as every other rng_<name>.cuh -- see rng_philox.cuh.
__device__ __forceinline__ void rng_init(unsigned long long seed,
                                          unsigned long long subsequence,
                                          unsigned long long offset,
                                          RngState* state) {
    curand_init(seed, subsequence, offset, state);
}
