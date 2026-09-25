#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "third_place_check.cuh"
#include "third_place_data.cuh"
#include "tournament_data.cuh"
#include "cuda_check.cuh"

int main() {
    upload_tournament_data();
    upload_thirdplace_table();

    // group_order: resolve_third_place only reads index 2 (3rd place) of each
    // group. Set it to team (g*4 + 2) for every group -- an arbitrary but fixed
    // choice; the other 3 positions are unused here.
    int h_group_order[NUM_GROUPS][TEAMS_PER_GROUP];
    for (int g = 0; g < NUM_GROUPS; ++g) {
        for (int k = 0; k < TEAMS_PER_GROUP; ++k) h_group_order[g][k] = g * TEAMS_PER_GROUP + k;
    }

    // Groups A-H (indices 0-7) all outrank groups I-L (8-11) on points, all 12
    // values distinct -- no tie-break ambiguity. So the top-8 qualifying groups
    // are exactly A-H: qualifying_mask = 0b0000_1111_1111 = 255.
    int h_pts[NUM_TEAMS] = {0};
    int h_gf[NUM_TEAMS]  = {0};
    int h_ga[NUM_TEAMS]  = {0};
    int group_pts[NUM_GROUPS] = {23, 22, 21, 20, 19, 18, 17, 16, 5, 4, 3, 2};
    for (int g = 0; g < NUM_GROUPS; ++g) {
        int third_id = h_group_order[g][2];
        h_pts[third_id] = group_pts[g];
        h_gf[third_id] = 3;
        h_ga[third_id] = 1;
    }

    // Expected: read directly off the generated table's row for mask 255
    // (slot_group = {7, 6, 1, 2, 0, 5, 3, 4}), converted to team IDs via
    // group*4+2, in slot order 1A,1B,1D,1E,1G,1I,1K,1L. Not invented -- this is
    // the actual row wc2026_r32_thirdplace_lookup.csv produced for this combo.
    int expected[8] = {30, 26, 6, 10, 2, 22, 14, 18};

    int *d_group_order, *d_pts, *d_gf, *d_ga, *d_out;
    CUDA_CHECK(cudaMalloc(&d_group_order, NUM_GROUPS * TEAMS_PER_GROUP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pts, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gf, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ga, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, NUM_THIRDPLACE_SLOTS * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_group_order, h_group_order,
                          NUM_GROUPS * TEAMS_PER_GROUP * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, NUM_TEAMS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gf, h_gf, NUM_TEAMS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ga, h_ga, NUM_TEAMS * sizeof(int), cudaMemcpyHostToDevice));

    launch_thirdplace_check(d_group_order, d_pts, d_gf, d_ga, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out[NUM_THIRDPLACE_SLOTS];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, NUM_THIRDPLACE_SLOTS * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_group_order)); CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_gf)); CUDA_CHECK(cudaFree(d_ga)); CUDA_CHECK(cudaFree(d_out));

    bool pass = true;
    for (int k = 0; k < NUM_THIRDPLACE_SLOTS; ++k) {
        if (h_out[k] != expected[k]) pass = false;
    }

    const char* slot_names[NUM_THIRDPLACE_SLOTS] = {"1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"};
    printf("[%s] Groups A-H qualify (mask 255), slot assignment vs. real CSV row\n",
           pass ? "PASS" : "FAIL");
    for (int k = 0; k < NUM_THIRDPLACE_SLOTS; ++k) {
        printf("  slot %-2s  expected=%2d  got=%2d\n", slot_names[k], expected[k], h_out[k]);
    }

    return pass ? 0 : 1;
}
