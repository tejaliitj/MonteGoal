#pragma once

// =============================================================================
// rng/rng.cuh  --  THE ONE PLACE that decides which random number generator
// the whole engine uses.
//
// Every other file (engine/ and football/) is written against exactly this
// contract and never names a cuRAND generator type directly:
//
//     RngState                                   -- the per-thread generator state type
//     rng_init(seed, subsequence, offset, &st)   -- seed one thread's stream
//     RNG_NAME                                   -- short string, used in benchmark output
//
// The generator is chosen at COMPILE time with a -D flag (no runtime cost):
//
//     (nothing) or -DRNG_PHILOX    Philox4_32_10  (default, counter-based)
//     -DRNG_XORWOW                 XORWOW         (cuRAND's classic default)
//     -DRNG_MRG32K3A               MRG32k3a       (combined multiple recursive)
//
// Adding another generator = add rng_<name>.cuh next to these files and one more
// #elif below. Nothing else in the project changes.
// =============================================================================

#include <curand_kernel.h>

#if defined(RNG_XORWOW) && defined(RNG_MRG32K3A)
#error "Pick only one RNG: define RNG_PHILOX, RNG_XORWOW or RNG_MRG32K3A (not several)."
#endif
#if (defined(RNG_XORWOW) || defined(RNG_MRG32K3A)) && defined(RNG_PHILOX)
#error "Pick only one RNG: define RNG_PHILOX, RNG_XORWOW or RNG_MRG32K3A (not several)."
#endif

#if defined(RNG_XORWOW)
    #include "rng_xorwow.cuh"
#elif defined(RNG_MRG32K3A)
    #include "rng_mrg32k3a.cuh"
#else
    #ifndef RNG_PHILOX
    #define RNG_PHILOX   // default generator when no -D flag is given
    #endif
    #include "rng_philox.cuh"
#endif
