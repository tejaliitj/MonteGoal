#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include "monte_carlo_kernel.cuh"   // engine -- generic, no football knowledge
#include "tournament_kernel.cuh"    // football -- SimulateTournament functor
#include "tournament_data.cuh"
#include "third_place_data.cuh"
#include "bracket_data.cuh"
#include "lambda_matrix.cuh"
#include "cuda_check.cuh"
#include "gpu_timer.cuh"

int main(int argc, char** argv) {
    int num_trials = (argc > 1) ? atoi(argv[1]) : 100000;
    unsigned long long seed = (argc > 2) ? (unsigned long long)atoll(argv[2]) : 42ULL;

    auto wall_start = std::chrono::high_resolution_clock::now();

    upload_tournament_data();
    upload_thirdplace_table();
    upload_bracket_data();
    upload_lambda_matrix();

    int* d_champion;
    CUDA_CHECK(cudaMalloc(&d_champion, num_trials * sizeof(int)));

    GpuTimer timer;
    timer.start();
    launch_monte_carlo(seed, 0ULL, num_trials, d_champion, SimulateTournament{});
    CUDA_CHECK(cudaGetLastError());
    timer.stop();

    int* h_champion = (int*)malloc(num_trials * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_champion, d_champion, num_trials * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_champion));

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    int counts[NUM_TEAMS] = {0};
    for (int i = 0; i < num_trials; ++i) counts[h_champion[i]]++;
    free(h_champion);

    int order[NUM_TEAMS];
    for (int t = 0; t < NUM_TEAMS; ++t) order[t] = t;
    for (int i = 1; i < NUM_TEAMS; ++i) {
        int key = order[i];
        int j = i - 1;
        while (j >= 0 && counts[order[j]] < counts[key]) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }

    printf("=== MonteGoal: %d trials, seed %llu ===\n\n", num_trials, seed);
    printf("Champion probability (teams with at least one title):\n");
    for (int i = 0; i < NUM_TEAMS; ++i) {
        int t = order[i];
        if (counts[t] == 0) break;
        double pct = 100.0 * counts[t] / num_trials;
        printf("  %-18s %7d titles  (%5.2f%%)\n", h_team_names[t], counts[t], pct);
    }

    printf("\nGPU kernel time (malloc+launch+sync) = %.3f ms\n", timer.elapsed_ms());
    printf("Total wall-clock time (incl. setup)  = %.3f ms\n", wall_ms);

    return 0;
}
