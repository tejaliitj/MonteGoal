#pragma once

#include "tournament_data.cuh"
#include "lambda_model.cuh"
#include "poisson_sample.cuh"
#include <curand_kernel.h>

// Simulates all 72 group-stage fixtures for one trial, using the calling thread's
// own Philox state (advanced internally -- caller just passes it in already
// curand_init'd). Fills team_pts/team_gf/team_ga (length NUM_TEAMS, caller-owned,
// must be zero-initialized) with each team's points, goals for, and goals against.
// Also fills fixture_home_goals/fixture_away_goals (length NUM_GROUP_FIXTURES,
// caller-owned) with this trial's actual per-match scores -- standings.cuh needs
// these for head-to-head resolution, since aggregated team stats alone lose that.
// Ranking and tiebreaking is standings.cuh's job, not this function's -- this is
// the direct device equivalent of group_stage.py's simulate_group_matches + _table.
__device__ void simulate_group_stage(curandStatePhilox4_32_10_t* state,
                                      int* team_pts, int* team_gf, int* team_ga,
                                      int* fixture_home_goals, int* fixture_away_goals) {
    for (int f = 0; f < NUM_GROUP_FIXTURES; ++f) {
        int home_id = d_fixture_home[f];
        int away_id = d_fixture_away[f];

        float lam_home, lam_away;
        predict_lambda(home_id, away_id, /*is_knockout=*/false, /*extra_time=*/false,
                        &lam_home, &lam_away);

        int home_goals = sample_goals(state, lam_home);
        int away_goals = sample_goals(state, lam_away);

        fixture_home_goals[f] = home_goals;
        fixture_away_goals[f] = away_goals;

        team_gf[home_id] += home_goals;
        team_ga[home_id] += away_goals;
        team_gf[away_id] += away_goals;
        team_ga[away_id] += home_goals;

        if (home_goals > away_goals) {
            team_pts[home_id] += 3;
        } else if (home_goals < away_goals) {
            team_pts[away_id] += 3;
        } else {
            team_pts[home_id] += 1;
            team_pts[away_id] += 1;
        }
    }
}
