#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "bracket_check.cuh"
#include "bracket_data.cuh"
#include "tournament_data.cuh"
#include "third_place_data.cuh"
#include "cuda_check.cuh"

int main() {
    upload_tournament_data();
    upload_bracket_data();

    // Synthetic group_order: group_order[g][k] = g*4+k -- lets us predict exactly
    // what any group_winner/group_runnerup resolution should return.
    int h_group_order[NUM_GROUPS][TEAMS_PER_GROUP];
    for (int g = 0; g < NUM_GROUPS; ++g)
        for (int k = 0; k < TEAMS_PER_GROUP; ++k) h_group_order[g][k] = g * TEAMS_PER_GROUP + k;

    // Marker values, chosen to be unmistakable and never collide with real IDs.
    int h_slot_team_id[NUM_THIRDPLACE_SLOTS];
    for (int k = 0; k < NUM_THIRDPLACE_SLOTS; ++k) h_slot_team_id[k] = 100 + k;

    int h_match_winner[NUM_KNOCKOUT_MATCHES], h_match_loser[NUM_KNOCKOUT_MATCHES];
    for (int i = 0; i < NUM_KNOCKOUT_MATCHES; ++i) {
        h_match_winner[i] = 200 + i;
        h_match_loser[i]  = 300 + i;
    }

    int h_match_ids[5] = {0, 1, 27, 30, 31};
    int expected_a[5]  = {1, 16, 222, 328, 228};
    int expected_b[5]  = {5, 103, 223, 329, 229};
    const char* labels[5] = {
        "match 73: group_runnerup(A) vs group_runnerup(B)",
        "match 74: group_winner(E) vs thirdplace_lookup(1E)",
        "match 100 (QF): winner_of(95) vs winner_of(96)",
        "match 103 (3rd place): loser_of(101) vs loser_of(102)",
        "match 104 (Final): winner_of(101) vs winner_of(102)",
    };

    int *d_match_ids, *d_group_order, *d_slot_team_id, *d_match_winner, *d_match_loser, *d_out_a, *d_out_b;
    CUDA_CHECK(cudaMalloc(&d_match_ids, 5 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_group_order, NUM_GROUPS * TEAMS_PER_GROUP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_slot_team_id, NUM_THIRDPLACE_SLOTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_match_winner, NUM_KNOCKOUT_MATCHES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_match_loser, NUM_KNOCKOUT_MATCHES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_a, 5 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_b, 5 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_match_ids, h_match_ids, 5 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_group_order, h_group_order,
                          NUM_GROUPS * TEAMS_PER_GROUP * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_team_id, h_slot_team_id,
                          NUM_THIRDPLACE_SLOTS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_match_winner, h_match_winner,
                          NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_match_loser, h_match_loser,
                          NUM_KNOCKOUT_MATCHES * sizeof(int), cudaMemcpyHostToDevice));

    launch_bracket_check(d_match_ids, 5, d_group_order, d_slot_team_id,
                          d_match_winner, d_match_loser, d_out_a, d_out_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out_a[5], h_out_b[5];
    CUDA_CHECK(cudaMemcpy(h_out_a, d_out_a, 5 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_b, d_out_b, 5 * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_match_ids)); CUDA_CHECK(cudaFree(d_group_order));
    CUDA_CHECK(cudaFree(d_slot_team_id)); CUDA_CHECK(cudaFree(d_match_winner));
    CUDA_CHECK(cudaFree(d_match_loser)); CUDA_CHECK(cudaFree(d_out_a)); CUDA_CHECK(cudaFree(d_out_b));

    bool all_pass = true;
    for (int i = 0; i < 5; ++i) {
        bool pass = (h_out_a[i] == expected_a[i]) && (h_out_b[i] == expected_b[i]);
        all_pass &= pass;
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", labels[i]);
        printf("  expected: a=%d b=%d\n", expected_a[i], expected_b[i]);
        printf("  got:      a=%d b=%d\n\n", h_out_a[i], h_out_b[i]);
    }

    printf("%s\n", all_pass ? "All cases passed." : "Some cases FAILED.");
    return all_pass ? 0 : 1;
}
