#pragma once

#include "tournament_data.cuh"
#include "lambda_matrix.cuh"

// Real model integration: looks up expected goals directly from the trained
// model's 48x48 lambda matrix (see lambda_matrix.cu for provenance), rather
// than the earlier placeholder Elo-based formula. Home advantage is already
// baked into the matrix values, so it is NOT re-applied here.
//
// is_knockout is accepted for interface parity but has no effect, same as
// before -- the matrix is a single general-purpose lambda per matchup, not
// split by context. extra_time is still applied here, since it's a
// match-state adjustment (a 30-minute period, not a full 90) rather than a
// pregame feature the model would have predicted.
__device__ __forceinline__ void predict_lambda(int team_a_id, int team_b_id,
                                                 bool is_knockout, bool extra_time,
                                                 float* lambda_a, float* lambda_b) {
    (void)is_knockout;  // unused -- kept for parity with the CPU model's signature

    float lam_a = d_lambda_matrix[team_a_id][team_b_id];
    float lam_b = d_lambda_matrix[team_b_id][team_a_id];

    if (extra_time) {
        lam_a *= 30.0f / 90.0f;
        lam_b *= 30.0f / 90.0f;
    }

    *lambda_a = lam_a;
    *lambda_b = lam_b;
}
