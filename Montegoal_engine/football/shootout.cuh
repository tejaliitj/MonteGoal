#pragma once

// =============================================================================
// football/shootout.cuh  --  penalty shootout model.
//
// Only reached when a knockout match is still level after regulation AND extra
// time. Direct port of penalty_model.py.
// =============================================================================

#include <math.h>
#include "uniform_sample.cuh"
#include "rng.cuh"

// Higher = more luck-dominated (probabilities closer to 50/50). 400 reproduces
// standard Elo win-probability scaling (used for full-match outcomes);
// shootouts are damped well above that -- same value as penalty_model.py.
// Flagged for calibration once real shootout outcome data is available.
#define SHOOTOUT_SENSITIVITY 800.0f

// P(team A wins the shootout): dampened Elo logistic, ported term-for-term
// from penalty_model.py's shootout_win_prob:
//     P = 1 / (1 + 10^(-(elo_a - elo_b) / SENSITIVITY))
__device__ __forceinline__ float shootout_win_prob(float elo_a, float elo_b) {
    return 1.0f / (1.0f + powf(10.0f, -(elo_a - elo_b) / SHOOTOUT_SENSITIVITY));
}

// The shootout "coin flip": draws ONE uniform number from the thread's RNG
// (through the engine's shared primitive, not curand_uniform directly) and
// returns true if team A wins, i.e. if the draw falls under P(A wins).
__device__ __forceinline__ bool sample_shootout(RngState* state,
                                                  float elo_a, float elo_b) {
    return sample_uniform01(state) < shootout_win_prob(elo_a, elo_b);
}
