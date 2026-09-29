#pragma once

#include "tournament_data.cuh"
#include <math.h>

#ifdef USE_ELO_ONLY_LAMBDA
// Ablation build: the elo-difference-only placeholder formula used before
// the real trained model was integrated -- no CSV, no ML dependency at all.
// Kept here as an explicit, compile-time-selectable baseline specifically
// for comparing against the real matrix, rather than deleted once superseded.
#define BASE_GOALS 1.35f
#define ELO_SCALE  400.0f
#define HOST_BOOST 1.10f

__device__ __forceinline__ void predict_lambda(int team_a_id, int team_b_id,
                                                 bool is_knockout, bool extra_time,
                                                 float* lambda_a, float* lambda_b) {
    (void)is_knockout;  // unused -- kept for parity with the CPU model's signature

    float elo_diff = d_team_elo[team_a_id] - d_team_elo[team_b_id];
    float lam_a = BASE_GOALS * expf(elo_diff / ELO_SCALE);
    float lam_b = BASE_GOALS * expf(-elo_diff / ELO_SCALE);

    if (d_team_is_host[team_a_id]) lam_a *= HOST_BOOST;
    if (d_team_is_host[team_b_id]) lam_b *= HOST_BOOST;

    if (extra_time) {
        lam_a *= 30.0f / 90.0f;
        lam_b *= 30.0f / 90.0f;
    }

    *lambda_a = lam_a;
    *lambda_b = lam_b;
}

#else
// Real model integration: looks up expected goals directly from the trained
// model's 48x48 lambda matrix, loaded at runtime from lambda_matrix.csv (see
// lambda_matrix.cuh/.cu). Home advantage is already baked into the matrix
// values, so it is NOT re-applied here.
#include "lambda_matrix.cuh"

__device__ __forceinline__ void predict_lambda(int team_a_id, int team_b_id,
                                                 bool is_knockout, bool extra_time,
                                                 float* lambda_a, float* lambda_b) {
    (void)is_knockout;

    float lam_a = d_lambda_matrix[team_a_id][team_b_id];
    float lam_b = d_lambda_matrix[team_b_id][team_a_id];

    if (extra_time) {
        lam_a *= 30.0f / 90.0f;
        lam_b *= 30.0f / 90.0f;
    }

    *lambda_a = lam_a;
    *lambda_b = lam_b;
}
#endif
