#pragma once

// =============================================================================
// football/lambda_model.cuh  --  expected goals (lambda) for a fixture.
//
// predict_lambda() is the ONE function every simulation module calls to get
// lambda_a / lambda_b for a match. Its signature is identical to the CPU
// reference's predict_lambda so downstream code never cares what sits behind it.
// Two interchangeable implementations, selected at COMPILE time:
//
//   default                 -> look up the trained model's 48x48 lambda matrix
//   -DUSE_ELO_ONLY_LAMBDA   -> the earlier placeholder Elo-difference formula
//                              (used for the ablation in report Section 7.2)
//
// In both cases `extra_time` scales lambda by 30/90 because extra time is a
// 30-minute period, not 90. That is a match-STATE adjustment applied at
// simulation time, not something either model predicts.
// =============================================================================

#include "tournament_data.cuh"
#include "lambda_matrix.cuh"

#ifdef USE_ELO_ONLY_LAMBDA
// ---- Placeholder formula constants (report Section 2.3 / lambda_model.py) ----
#define BASE_GOALS 1.35f   // average goals per team when Elo ratings are equal
#define ELO_SCALE  400.0f  // Elo difference that multiplies lambda by e
#define HOST_BOOST 1.10f   // flat multiplier for a host nation (Mexico, Canada, USA)
#endif

__device__ __forceinline__ void predict_lambda(int team_a_id, int team_b_id,
                                                 bool is_knockout, bool extra_time,
                                                 float* lambda_a, float* lambda_b) {
    (void)is_knockout;  // unused by both models -- kept for signature parity with the CPU model

#ifdef USE_ELO_ONLY_LAMBDA
    // lambda_a = BASE * exp(elo_diff / SCALE), lambda_b = BASE * exp(-elo_diff / SCALE)
    float elo_diff = d_team_elo[team_a_id] - d_team_elo[team_b_id];
    float lam_a = BASE_GOALS * expf(elo_diff / ELO_SCALE);
    float lam_b = BASE_GOALS * expf(-elo_diff / ELO_SCALE);
    // Host nations get a flat boost (applied per side, independently).
    if (d_team_is_host[team_a_id]) lam_a *= HOST_BOOST;
    if (d_team_is_host[team_b_id]) lam_b *= HOST_BOOST;
#else
    // Trained model: direct table lookup. d_lambda_matrix[a][b] = expected goals
    // for team a against team b. Home advantage is ALREADY baked into these
    // values by the model that produced them -- do not apply a host multiplier.
    float lam_a = d_lambda_matrix[team_a_id][team_b_id];
    float lam_b = d_lambda_matrix[team_b_id][team_a_id];
#endif

    if (extra_time) {
        lam_a *= 30.0f / 90.0f;
        lam_b *= 30.0f / 90.0f;
    }

    *lambda_a = lam_a;
    *lambda_b = lam_b;
}
