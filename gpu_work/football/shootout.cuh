#pragma once

#include <curand_kernel.h>
#include <math.h>
#include "uniform_sample.cuh"

// Higher = more luck-dominated (probabilities closer to 50/50). 400 reproduces
// standard Elo win-probability scaling (used for full-match outcomes);
// shootouts are damped well above that -- ported value, same as penalty_model.py.
#define SHOOTOUT_SENSITIVITY 800.0f

// P(team A wins the shootout), dampened Elo logistic -- ported term-for-term
// from penalty_model.py's shootout_win_prob.
__device__ __forceinline__ float shootout_win_prob(float elo_a, float elo_b) {
    return 1.0f / (1.0f + powf(10.0f, -(elo_a - elo_b) / SHOOTOUT_SENSITIVITY));
}

// Draws one uniform from the calling thread's Philox state, via the engine's
// shared primitive (not curand_uniform directly -- see uniform_sample.cuh),
// and returns true if team A wins the shootout.
__device__ __forceinline__ bool sample_shootout(curandStatePhilox4_32_10_t* state,
                                                  float elo_a, float elo_b) {
    return sample_uniform01(state) < shootout_win_prob(elo_a, elo_b);
}
