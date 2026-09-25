#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "standings_check.cuh"
#include "tournament_data.cuh"
#include "cuda_check.cuh"

static bool run_case(const char* name,
                      const int* h_pts, const int* h_gf, const int* h_ga,
                      const int* h_fh, const int* h_fa,
                      const int* expected) {
    int *d_pts, *d_gf, *d_ga, *d_fh, *d_fa, *d_out;
    CUDA_CHECK(cudaMalloc(&d_pts, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gf, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ga, NUM_TEAMS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fh, NUM_GROUP_FIXTURES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fa, NUM_GROUP_FIXTURES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, NUM_TEAMS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gf, h_gf, NUM_TEAMS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ga, h_ga, NUM_TEAMS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fh, h_fh, NUM_GROUP_FIXTURES * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fa, h_fa, NUM_GROUP_FIXTURES * sizeof(int), cudaMemcpyHostToDevice));

    launch_standings_check(d_pts, d_gf, d_ga, d_fh, d_fa, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out[4];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 4 * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_gf)); CUDA_CHECK(cudaFree(d_ga));
    CUDA_CHECK(cudaFree(d_fh)); CUDA_CHECK(cudaFree(d_fa)); CUDA_CHECK(cudaFree(d_out));

    bool pass = true;
    for (int k = 0; k < 4; ++k) {
        if (h_out[k] != expected[k]) pass = false;
    }

    printf("[%s] %s\n", pass ? "PASS" : "FAIL", name);
    printf("  expected: %d %d %d %d\n", expected[0], expected[1], expected[2], expected[3]);
    printf("  got:      %d %d %d %d\n\n", h_out[0], h_out[1], h_out[2], h_out[3]);
    return pass;
}

int main() {
    upload_tournament_data();

    bool all_pass = true;

    // Case 1: no ties -- pts alone decides the order. Group A team IDs are 0-3
    // (Mexico, South Korea, South Africa, Czechia).
    {
        int pts[NUM_TEAMS] = {0}; pts[0] = 9; pts[1] = 6; pts[2] = 3; pts[3] = 0;
        int gf[NUM_TEAMS]  = {0}; gf[0] = 5; gf[1] = 4; gf[2] = 3; gf[3] = 2;
        int ga[NUM_TEAMS]  = {0}; ga[0] = 1; ga[1] = 2; ga[2] = 3; ga[3] = 4;
        int fh[NUM_GROUP_FIXTURES] = {0};
        int fa[NUM_GROUP_FIXTURES] = {0};
        int expected[4] = {0, 1, 2, 3};
        all_pass &= run_case("No ties (trivial)", pts, gf, ga, fh, fa, expected);
    }

    // Case 2: team 0 (Mexico, elo 1780) and team 1 (South Korea, elo 1740) tied
    // overall (pts=6, gd=+2, gf=5 each), but team 1 beat team 0 head-to-head 2-1.
    // If head-to-head resolution is broken and this silently fell through to
    // Elo instead, team 0 would win (higher Elo) -- wrong. Correct behavior is
    // team 1 first, since h2h overrides Elo in the cascade.
    {
        int pts[NUM_TEAMS] = {0}; pts[0] = 6; pts[1] = 6; pts[2] = 3; pts[3] = 0;
        int gf[NUM_TEAMS]  = {0}; gf[0] = 5; gf[1] = 5; gf[2] = 2; gf[3] = 1;
        int ga[NUM_TEAMS]  = {0}; ga[0] = 3; ga[1] = 3; ga[2] = 4; ga[3] = 5;
        int fh[NUM_GROUP_FIXTURES] = {0};
        int fa[NUM_GROUP_FIXTURES] = {0};
        fh[0] = 1; fa[0] = 2;  // fixture 0 = team0 (home) vs team1 (away): 1-2
        int expected[4] = {1, 0, 2, 3};
        all_pass &= run_case("2-way tie, head-to-head overrides Elo", pts, gf, ga, fh, fa, expected);
    }

    // Case 3: all 4 teams tied on everything (every group match a 1-1 draw), so
    // head-to-head among all 4 reproduces the exact same tie -- must fall through
    // to Elo. Group A Elo order: Mexico 1780 > South Korea 1740 > Czechia 1700 >
    // South Africa 1560.
    {
        int pts[NUM_TEAMS] = {0}; pts[0] = 3; pts[1] = 3; pts[2] = 3; pts[3] = 3;
        int gf[NUM_TEAMS]  = {0}; gf[0] = 3; gf[1] = 3; gf[2] = 3; gf[3] = 3;
        int ga[NUM_TEAMS]  = {0}; ga[0] = 3; ga[1] = 3; ga[2] = 3; ga[3] = 3;
        int fh[NUM_GROUP_FIXTURES] = {0};
        int fa[NUM_GROUP_FIXTURES] = {0};
        for (int m = 0; m < 6; ++m) { fh[m] = 1; fa[m] = 1; }
        int expected[4] = {0, 1, 3, 2};
        all_pass &= run_case("4-way tie, falls through to Elo", pts, gf, ga, fh, fa, expected);
    }

    printf("%s\n", all_pass ? "All cases passed." : "Some cases FAILED.");
    return all_pass ? 0 : 1;
}
