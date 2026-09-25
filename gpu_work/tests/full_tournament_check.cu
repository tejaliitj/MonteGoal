#include "full_tournament_check.cuh"
#include "group_stage.cuh"
#include "standings.cuh"
#include "third_place.cuh"
#include "bracket.cuh"
#include "tournament_data.cuh"
#include <curand_kernel.h>

__global__ void full_tournament_check_kernel(unsigned long long seed, unsigned long long counter_offset,
                                              int* team_pts, int* team_gf, int* team_ga,
                                              int* group_order_flat, int* slot_team_id,
                                              int* match_team_a, int* match_team_b,
                                              int* match_goals_a, int* match_goals_b,
                                              int* match_decided_by,
                                              int* match_winner, int* match_loser) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, counter_offset, 0, &state);

    int fixture_home_goals[NUM_GROUP_FIXTURES];
    int fixture_away_goals[NUM_GROUP_FIXTURES];
    simulate_group_stage(&state, team_pts, team_gf, team_ga, fixture_home_goals, fixture_away_goals);

    int (*group_order)[TEAMS_PER_GROUP] = reinterpret_cast<int (*)[TEAMS_PER_GROUP]>(group_order_flat);
    rank_all_groups(team_pts, team_gf, team_ga, fixture_home_goals, fixture_away_goals, group_order);

    resolve_third_place(group_order, team_pts, team_gf, team_ga, slot_team_id);

    simulate_knockout(&state, group_order, slot_team_id,
                       match_team_a, match_team_b, match_goals_a, match_goals_b,
                       match_decided_by, match_winner, match_loser);
}

void launch_full_tournament_check(unsigned long long seed, unsigned long long counter_offset,
                                   int* d_team_pts, int* d_team_gf, int* d_team_ga,
                                   int* d_group_order_flat, int* d_slot_team_id,
                                   int* d_match_team_a, int* d_match_team_b,
                                   int* d_match_goals_a, int* d_match_goals_b,
                                   int* d_match_decided_by,
                                   int* d_match_winner, int* d_match_loser) {
    full_tournament_check_kernel<<<1, 1>>>(seed, counter_offset,
        d_team_pts, d_team_gf, d_team_ga, d_group_order_flat, d_slot_team_id,
        d_match_team_a, d_match_team_b, d_match_goals_a, d_match_goals_b,
        d_match_decided_by, d_match_winner, d_match_loser);
}
