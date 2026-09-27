#include "tournament.cuh"
#include "group_stage.cuh"
#include "bracket.cuh"
#include "match_engine.cuh"
#include <cuda_runtime.h>

__constant__ int d_group_teams[NUM_GROUPS][TEAMS_PER_GROUP];

void upload_group_teams(const int h_group_teams[NUM_GROUPS][TEAMS_PER_GROUP])
{
    cudaMemcpyToSymbol(d_group_teams, h_group_teams,
                        sizeof(int) * NUM_GROUPS * TEAMS_PER_GROUP);
}

__global__ void world_cup_kernel(
    curandStatePhilox4_32_10_t *states,
    int total_threads,
    int *champion_out,
    int *semis_out,
    int *finalists_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads)
        return;

    curandStatePhilox4_32_10_t local_state = states[idx];

    // ---------------- GROUP STAGE ----------------
    int group_winner[NUM_GROUPS];
    int group_runner_up[NUM_GROUPS];
    ThirdEntry thirds[NUM_GROUPS];

    for (int g = 0; g < NUM_GROUPS; g++)
    {
        int teams[TEAMS_PER_GROUP];
        for (int t = 0; t < TEAMS_PER_GROUP; t++)
            teams[t] = d_group_teams[g][t];

        GroupResult res = play_group_device(teams, &local_state);

        group_winner[g] = res.winner;
        group_runner_up[g] = res.runner_up;

        thirds[g].group_letter = g;
        thirds[g].team_id = res.third_team;
        thirds[g].pts = res.third_pts;
        thirds[g].gd = res.third_gd;
        thirds[g].gf = res.third_gf;
    }

    // ---------------- BEST THIRD-PLACED TEAMS ----------------
    rank_third_place_teams_device(thirds, &local_state);

    int qualifying_group_letters[8];
    int third_team_by_group[NUM_GROUPS];
    for (int i = 0; i < NUM_GROUPS; i++)
        third_team_by_group[i] = -1;

    for (int i = 0; i < 8; i++)
    {
        qualifying_group_letters[i] = thirds[i].group_letter;
        third_team_by_group[thirds[i].group_letter] = thirds[i].team_id;
    }

    int slot_to_third_group[NUM_GROUPS];
    assign_third_place_slots_device(qualifying_group_letters, slot_to_third_group);

    // ---------------- ROUND OF 32 ----------------
    int r32_winners[16];
    for (int m = 0; m < 16; m++)
    {
        int a = resolve_role_device(d_r32_kind[m][0], d_r32_letter[m][0],
                                     group_winner, group_runner_up,
                                     third_team_by_group, slot_to_third_group);
        int b = resolve_role_device(d_r32_kind[m][1], d_r32_letter[m][1],
                                     group_winner, group_runner_up,
                                     third_team_by_group, slot_to_third_group);
        r32_winners[m] = play_knockout_match_device(&local_state, a, b);
    }

    // ---------------- ROUND OF 16 ----------------
    int r16_winners[8];
    for (int m = 0; m < 8; m++)
    {
        int a = r32_winners[d_r16_pairs[m][0]];
        int b = r32_winners[d_r16_pairs[m][1]];
        r16_winners[m] = play_knockout_match_device(&local_state, a, b);
    }

    // ---------------- QUARTERFINALS ----------------
    int qf_winners[4];
    for (int m = 0; m < 4; m++)
    {
        int a = r16_winners[d_qf_pairs[m][0]];
        int b = r16_winners[d_qf_pairs[m][1]];
        qf_winners[m] = play_knockout_match_device(&local_state, a, b);
    }

    // ---------------- SEMIFINALS ----------------
    int finalists[2];
    for (int m = 0; m < 2; m++)
    {
        int a = qf_winners[d_sf_pairs[m][0]];
        int b = qf_winners[d_sf_pairs[m][1]];
        finalists[m] = play_knockout_match_device(&local_state, a, b);
    }

    // ---------------- FINAL ----------------
    // (third-place playoff is skipped -- tournament.py plays it but never
    // folds its result into the aggregated stats, so it doesn't affect any
    // of this project's output; see README for the full note.)
    int champion = play_knockout_match_device(&local_state, finalists[0], finalists[1]);

    // ---------------- WRITE RESULTS ----------------
    champion_out[idx] = champion;
    for (int i = 0; i < 4; i++)
        semis_out[idx * 4 + i] = qf_winners[i];
    for (int i = 0; i < 2; i++)
        finalists_out[idx * 2 + i] = finalists[i];

    states[idx] = local_state;
}
