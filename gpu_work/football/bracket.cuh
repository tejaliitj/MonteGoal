#pragma once

#include "tournament_data.cuh"
#include "bracket_data.cuh"
#include "lambda_model.cuh"
#include "poisson_sample.cuh"
#include "shootout.cuh"
#include <curand_kernel.h>

#define DECIDED_REGULATION 0
#define DECIDED_EXTRA_TIME 1
#define DECIDED_PENALTIES  2

// Resolves one team-spec to a global team ID, given this trial's group_order,
// third-place slot assignment, and the knockout results built up so far.
// Direct device equivalent of bracket_engine.py's _resolve_team.
__device__ __forceinline__ int resolve_spec(int type, int arg,
                                             const int group_order[NUM_GROUPS][TEAMS_PER_GROUP],
                                             const int* slot_team_id,
                                             const int* match_winner, const int* match_loser) {
    switch (type) {
        case SPEC_GROUP_WINNER:      return group_order[arg][0];
        case SPEC_GROUP_RUNNERUP:    return group_order[arg][1];
        case SPEC_THIRDPLACE_LOOKUP: return slot_team_id[arg];
        case SPEC_WINNER_OF_MATCH:   return match_winner[arg];
        case SPEC_LOSER_OF_MATCH:    return match_loser[arg];
    }
    return -1;  // unreachable -- every type is one of the 5 above
}

// Simulates all 32 knockout matches (Round of 32 through the Final) for one
// trial, in match_id order 0..31 -- already dependency-safe, so a single
// forward pass resolves every match_a/match_b before it's needed. All output
// arrays are length NUM_KNOCKOUT_MATCHES, caller-owned. Direct device
// equivalent of bracket_engine.py's simulate_knockout + _play_match.
__device__ void simulate_knockout(curandStatePhilox4_32_10_t* state,
                                   const int group_order[NUM_GROUPS][TEAMS_PER_GROUP],
                                   const int* slot_team_id,
                                   int* match_team_a, int* match_team_b,
                                   int* match_goals_a, int* match_goals_b,
                                   int* match_decided_by,
                                   int* match_winner, int* match_loser) {
    for (int m = 0; m < NUM_KNOCKOUT_MATCHES; ++m) {
        int team_a = resolve_spec(d_bracket_team_a_type[m], d_bracket_team_a_arg[m],
                                   group_order, slot_team_id, match_winner, match_loser);
        int team_b = resolve_spec(d_bracket_team_b_type[m], d_bracket_team_b_arg[m],
                                   group_order, slot_team_id, match_winner, match_loser);
        match_team_a[m] = team_a;
        match_team_b[m] = team_b;

        float lam_a, lam_b;
        predict_lambda(team_a, team_b, /*is_knockout=*/true, /*extra_time=*/false, &lam_a, &lam_b);
        int goals_a = sample_goals(state, lam_a);
        int goals_b = sample_goals(state, lam_b);
        int decided_by = DECIDED_REGULATION;

        if (goals_a == goals_b) {
            float et_lam_a, et_lam_b;
            predict_lambda(team_a, team_b, /*is_knockout=*/true, /*extra_time=*/true, &et_lam_a, &et_lam_b);
            goals_a += sample_goals(state, et_lam_a);
            goals_b += sample_goals(state, et_lam_b);
            decided_by = DECIDED_EXTRA_TIME;

            if (goals_a == goals_b) {
                decided_by = DECIDED_PENALTIES;
                bool a_wins = sample_shootout(state, d_team_elo[team_a], d_team_elo[team_b]);
                match_goals_a[m] = goals_a;
                match_goals_b[m] = goals_b;
                match_decided_by[m] = decided_by;
                if (a_wins) { match_winner[m] = team_a; match_loser[m] = team_b; }
                else        { match_winner[m] = team_b; match_loser[m] = team_a; }
                continue;
            }
        }

        match_goals_a[m] = goals_a;
        match_goals_b[m] = goals_b;
        match_decided_by[m] = decided_by;
        if (goals_a > goals_b) { match_winner[m] = team_a; match_loser[m] = team_b; }
        else                   { match_winner[m] = team_b; match_loser[m] = team_a; }
    }
}
