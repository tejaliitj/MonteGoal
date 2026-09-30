#pragma once

// =============================================================================
// football/group_stage.cuh  --  simulate all 72 group-stage matches for ONE trial.
//
// Runs on the GPU inside one thread (one thread = one whole tournament).
// Direct device equivalent of group_stage.py's simulate_group_matches + _table.
// =============================================================================

#include "tournament_data.cuh"
#include "lambda_model.cuh"
#include "poisson_sample.cuh"
#include "rng.cuh"

// Simulates all 72 group-stage fixtures using the calling thread's own RNG
// state (advanced internally -- the caller passes it in already rng_init'd).
//
// Outputs (all caller-owned, thread-local arrays):
//   team_pts / team_gf / team_ga : length NUM_TEAMS, MUST be zero-initialised by
//       the caller. Filled with each team's points, goals for, goals against.
//   fixture_home_goals / fixture_away_goals : length NUM_GROUP_FIXTURES. Each
//       match's actual scoreline. standings.cuh needs these for head-to-head
//       tiebreaks, which aggregated team totals alone cannot support.
//
// Ranking and tiebreaking is standings.cuh's job, not this function's.
__device__ void simulate_group_stage(RngState* state,
                                      int* team_pts, int* team_gf, int* team_ga,
                                      int* fixture_home_goals, int* fixture_away_goals) {
    for (int f = 0; f < NUM_GROUP_FIXTURES; ++f) {
        // Which two teams play fixture f (fixed schedule, read from constant memory).
        int home_id = d_fixture_home[f];
        int away_id = d_fixture_away[f];

        // Expected goals for each side, from the lambda model (trained matrix by
        // default, or the Elo-only placeholder if built with -DUSE_ELO_ONLY_LAMBDA).
        float lam_home, lam_away;
        predict_lambda(home_id, away_id, /*is_knockout=*/false, /*extra_time=*/false,
                        &lam_home, &lam_away);

        // Random step: draw each side's goals from Poisson(lambda).
        int home_goals = sample_goals(state, lam_home);
        int away_goals = sample_goals(state, lam_away);

        // Remember the exact scoreline (needed later for head-to-head tiebreaks).
        fixture_home_goals[f] = home_goals;
        fixture_away_goals[f] = away_goals;

        // Update goals-for / goals-against for both teams.
        team_gf[home_id] += home_goals;
        team_ga[home_id] += away_goals;
        team_gf[away_id] += away_goals;
        team_ga[away_id] += home_goals;

        // Award points: 3 for a win, 1 each for a draw.
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
