#pragma once

#include <curand_kernel.h>
#include "group_stage.cuh"
#include "standings.cuh"
#include "third_place.cuh"
#include "bracket.cuh"
#include "tournament_data.cuh"
#include "bracket_data.cuh"

// One Monte Carlo trial = one complete simulated tournament: group stage ->
// standings -> third place -> knockout -> champion. This is the only
// football-specific piece the generic engine (engine/monte_carlo_kernel.cuh)
// ever touches -- everything about *how* trials run in parallel
// (thread-per-trial, subsequence assignment, launch configuration) lives
// there and knows nothing about football. Header-only, same as every other
// logic file in football/ -- only the static *_data files need a .cu for
// their single __constant__-memory definition point.
//
// Identical body to the old tournament_kernel.cu's per-thread logic --
// nothing about the simulation itself changed, only how it's launched.
struct SimulateTournament {
    __device__ int operator()(curandStatePhilox4_32_10_t* state) const {
        int team_pts[NUM_TEAMS] = {0};
        int team_gf[NUM_TEAMS] = {0};
        int team_ga[NUM_TEAMS] = {0};
        int fixture_home_goals[NUM_GROUP_FIXTURES];
        int fixture_away_goals[NUM_GROUP_FIXTURES];
        simulate_group_stage(state, team_pts, team_gf, team_ga,
                              fixture_home_goals, fixture_away_goals);

        int group_order[NUM_GROUPS][TEAMS_PER_GROUP];
        rank_all_groups(team_pts, team_gf, team_ga, fixture_home_goals, fixture_away_goals,
                         group_order);

        int slot_team_id[NUM_THIRDPLACE_SLOTS];
        resolve_third_place(group_order, team_pts, team_gf, team_ga, slot_team_id);

        int match_team_a[NUM_KNOCKOUT_MATCHES], match_team_b[NUM_KNOCKOUT_MATCHES];
        int match_goals_a[NUM_KNOCKOUT_MATCHES], match_goals_b[NUM_KNOCKOUT_MATCHES];
        int match_decided_by[NUM_KNOCKOUT_MATCHES];
        int match_winner[NUM_KNOCKOUT_MATCHES], match_loser[NUM_KNOCKOUT_MATCHES];
        simulate_knockout(state, group_order, slot_team_id,
                           match_team_a, match_team_b, match_goals_a, match_goals_b,
                           match_decided_by, match_winner, match_loser);

        return match_winner[MATCH_ID_FINAL];
    }
};
