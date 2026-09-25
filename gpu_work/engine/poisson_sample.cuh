#pragma once

#include <curand_kernel.h>

// Draws one Poisson-distributed goal count with mean lambda, using the
// calling thread's own Philox state. Thin wrapper around cuRAND's
// device-side curand_poisson() so call sites in group-stage and
// knockout logic read in domain terms (goals, lambda) rather than the
// raw cuRAND API. The caller owns state -- init it once per thread,
// then call this as many times as needed; the state advances internally.
__device__ __forceinline__ int sample_goals(curandStatePhilox4_32_10_t* state,
                                             double lambda) {
    return (int)curand_poisson(state, lambda);
}
