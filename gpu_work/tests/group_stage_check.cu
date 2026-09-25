#include "group_stage_check.cuh"
#include "group_stage.cuh"
#include "tournament_data.cuh"
#include <curand_kernel.h>

__global__ void group_stage_check_kernel(int* total_goals_out, int num_trials,
                                          unsigned long long seed,
                                          unsigned long long counter_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_trials) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, counter_offset + idx, 0, &state);

    int team_pts[NUM_TEAMS] = {0};
    int team_gf[NUM_TEAMS] = {0};
    int team_ga[NUM_TEAMS] = {0};
    int fixture_home_goals[NUM_GROUP_FIXTURES];
    int fixture_away_goals[NUM_GROUP_FIXTURES];

    simulate_group_stage(&state, team_pts, team_gf, team_ga,
                          fixture_home_goals, fixture_away_goals);

    int total_goals = 0;
    for (int t = 0; t < NUM_TEAMS; ++t) total_goals += team_gf[t];

    total_goals_out[idx] = total_goals;
}

void launch_group_stage_check(int* d_total_goals, int num_trials,
                               unsigned long long seed,
                               unsigned long long counter_offset) {
    int threads = 128;
    int blocks = (num_trials + threads - 1) / threads;
    group_stage_check_kernel<<<blocks, threads>>>(d_total_goals, num_trials,
                                                    seed, counter_offset);
}
