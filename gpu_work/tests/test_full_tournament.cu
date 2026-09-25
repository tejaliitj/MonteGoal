#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "full_tournament_check.cuh"
#include "tournament_data.cuh"
#include "third_place_data.cuh"
#include "bracket_data.cuh"
#include "lambda_matrix.cuh"
#include "cuda_check.cuh"

static const char* decided_label(int d) {
    if (d == 1) return "  (extra time)";
    if (d == 2) return "  (penalties)";
    return "";
}

int main(int argc, char** argv) {
    unsigned long long seed = (argc > 1) ? (unsigned long long)atoll(argv[1]) : 42ULL;

    upload_tournament_data();
    upload_thirdplace_table();
    upload_bracket_data();
    upload_lambda_matrix();

    int *d_pts, *d_gf, *d_ga, *d_group_order, *d_slot_team_id;
    int *d_match_a, *d_match_b, *d_goals_a, *d_goals_b, *d_decided, *d_winner, *d_loser;

    CUDA_CHECK(cudaMalloc(&d_pts, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gf, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ga, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_group_order, NUM_GROUPS * TEAMS_PER_GROUP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_slot_team_id, NUM_THIRDPLACE_SLOTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_match_a, NUM_KNOCKOUT_MATCHES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_match_b, NUM_KNOCKOUT_MATCHES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_goals_a, NUM_KNOCKOUT_MATCHES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_goals_b, NUM_KNOCKOUT_MATCHES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_decided, NUM_KNOCKOUT_MATCHES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_winner, NUM_KNOCKOUT_MATCHES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_loser, NUM_KNOCKOUT_MATCHES * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_pts, 0, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_gf, 0, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ga, 0, NUM_TEAMS * sizeof(int)));

    launch_full_tournament_check(seed, 0ULL, d_pts, d_gf, d_ga, d_group_order, d_slot_team_id,
                                  d_match_a, d_match_b, d_goals_a, d_goals_b, d_decided,
                                  d_winner, d_loser);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_group_order[NUM_GROUPS][TEAMS_PER_GROUP];
    int h_slot_team_id[NUM_THIRDPLACE_SLOTS];
    int h_match_a[NUM_KNOCKOUT_MATCHES], h_match_b[NUM_KNOCKOUT_MATCHES];
    int h_goals_a[NUM_KNOCKOUT_MATCHES], h_goals_b[NUM_KNOCKOUT_MATCHES];
    int h_decided[NUM_KNOCKOUT_MATCHES], h_winner[NUM_KNOCKOUT_MATCHES], h_loser[NUM_KNOCKOUT_MATCHES];

    CUDA_CHECK(cudaMemcpy(h_group_order, d_group_order,
                          NUM_GROUPS * TEAMS_PER_GROUP * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_slot_team_id, d_slot_team_id,
                          NUM_THIRDPLACE_SLOTS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_match_a, d_match_a, NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_match_b, d_match_b, NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_goals_a, d_goals_a, NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_goals_b, d_goals_b, NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_decided, d_decided, NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_winner, d_winner, NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_loser, d_loser, NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyDeviceToHost));

    printf("=== Group stage (seed %llu) ===\n", seed);
    const char* GL = "ABCDEFGHIJKL";
    for (int g = 0; g < NUM_GROUPS; ++g) {
        printf("Group %c: 1st=%-16s 2nd=%-16s 3rd=%-16s 4th=%s\n", GL[g],
               h_team_names[h_group_order[g][0]], h_team_names[h_group_order[g][1]],
               h_team_names[h_group_order[g][2]], h_team_names[h_group_order[g][3]]);
    }

    printf("\n=== Round of 32 third-place slots ===\n");
    const char* slot_names[NUM_THIRDPLACE_SLOTS] = {"1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"};
    for (int k = 0; k < NUM_THIRDPLACE_SLOTS; ++k) {
        printf("  %s: %s\n", slot_names[k], h_team_names[h_slot_team_id[k]]);
    }

    const char* round_name[NUM_KNOCKOUT_MATCHES];
    for (int m = 0; m < 16; ++m) round_name[m] = "R32";
    for (int m = 16; m < 24; ++m) round_name[m] = "R16";
    for (int m = 24; m < 28; ++m) round_name[m] = "QF";
    for (int m = 28; m < 30; ++m) round_name[m] = "SF";
    round_name[MATCH_ID_THIRD_PLACE] = "3rd";
    round_name[MATCH_ID_FINAL] = "Final";

    printf("\n=== Knockout ===\n");
    for (int m = 0; m < NUM_KNOCKOUT_MATCHES; ++m) {
        printf("[%-5s %3d] %-16s %d-%d %-16s -> %s%s\n",
               round_name[m], m + MATCH_OFFSET,
               h_team_names[h_match_a[m]], h_goals_a[m], h_goals_b[m], h_team_names[h_match_b[m]],
               h_team_names[h_winner[m]], decided_label(h_decided[m]));
    }

    printf("\n*** Champion: %s ***\n", h_team_names[h_winner[MATCH_ID_FINAL]]);

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_gf)); CUDA_CHECK(cudaFree(d_ga));
    CUDA_CHECK(cudaFree(d_group_order)); CUDA_CHECK(cudaFree(d_slot_team_id));
    CUDA_CHECK(cudaFree(d_match_a)); CUDA_CHECK(cudaFree(d_match_b));
    CUDA_CHECK(cudaFree(d_goals_a)); CUDA_CHECK(cudaFree(d_goals_b));
    CUDA_CHECK(cudaFree(d_decided)); CUDA_CHECK(cudaFree(d_winner)); CUDA_CHECK(cudaFree(d_loser));

    return 0;
}
