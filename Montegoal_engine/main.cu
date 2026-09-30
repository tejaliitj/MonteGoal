// =============================================================================
// main.cu -- MonteGoal: simulate World Cup 2026 tournaments on the GPU and
//             report each team's probability of reaching QF, SF, Final and
//             winning the tournament.
//
// Usage:
//     montegoal [num_trials] [seed] [--block N]
//
// Defaults: 100000 trials, seed 42, 128 CUDA threads per block.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include "monte_carlo_kernel.cuh"
#include "tournament_kernel.cuh"
#include "tournament_data.cuh"
#include "third_place_data.cuh"
#include "bracket_data.cuh"
#include "lambda_matrix.cuh"
#include "cuda_check.cuh"
#include "gpu_timer.cuh"

int main(int argc, char** argv) {
    unsigned long long num_trials = 100000ULL;
    unsigned long long seed = 42ULL;
    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;

    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            threads_per_block = atoi(argv[++i]);
        } else if (positional == 0) {
            num_trials = strtoull(argv[i], nullptr, 10);
            ++positional;
        } else if (positional == 1) {
            seed = strtoull(argv[i], nullptr, 10);
            ++positional;
        } else {
            fprintf(stderr, "unexpected argument '%s'\n", argv[i]);
            return 1;
        }
    }

    if (num_trials == 0 || threads_per_block < 1 || threads_per_block > 1024) {
        fprintf(stderr, "need num_trials >= 1 and 1 <= --block <= 1024\n");
        return 1;
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    // 1. Upload all static tournament/model data once.
    upload_tournament_data();
    upload_thirdplace_table();
    upload_bracket_data();
    upload_lambda_matrix();

    // 2. Four GPU histograms. Each has one counter per team.
    unsigned long long* d_qf = nullptr;
    unsigned long long* d_sf = nullptr;
    unsigned long long* d_final = nullptr;
    unsigned long long* d_champion = nullptr;

    const size_t bins_bytes = NUM_TEAMS * sizeof(unsigned long long);
    CUDA_CHECK(cudaMalloc(&d_qf, bins_bytes));
    CUDA_CHECK(cudaMalloc(&d_sf, bins_bytes));
    CUDA_CHECK(cudaMalloc(&d_final, bins_bytes));
    CUDA_CHECK(cudaMalloc(&d_champion, bins_bytes));
    CUDA_CHECK(cudaMemset(d_qf, 0, bins_bytes));
    CUDA_CHECK(cudaMemset(d_sf, 0, bins_bytes));
    CUDA_CHECK(cudaMemset(d_final, 0, bins_bytes));
    CUDA_CHECK(cudaMemset(d_champion, 0, bins_bytes));

    GpuTimer timer;
    timer.start();

    launch_monte_carlo_four_stage_histogram<NUM_TEAMS, TournamentResult>(
        seed, 0ULL, num_trials,
        d_qf, d_sf, d_final, d_champion,
        SimulateTournament{}, threads_per_block);

    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    float kernel_ms = timer.elapsed_ms();

    // 3. Copy only 4 * 48 counters back to the CPU.
    unsigned long long qf[NUM_TEAMS] = {0};
    unsigned long long sf[NUM_TEAMS] = {0};
    unsigned long long final_counts[NUM_TEAMS] = {0};
    unsigned long long champions[NUM_TEAMS] = {0};

    CUDA_CHECK(cudaMemcpy(qf, d_qf, bins_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sf, d_sf, bins_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(final_counts, d_final, bins_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(champions, d_champion, bins_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_qf));
    CUDA_CHECK(cudaFree(d_sf));
    CUDA_CHECK(cudaFree(d_final));
    CUDA_CHECK(cudaFree(d_champion));

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // 4. Sort by champion probability, while displaying all four cumulative
    //    advancement probabilities for each team.
    int order[NUM_TEAMS];
    for (int t = 0; t < NUM_TEAMS; ++t) order[t] = t;
    for (int i = 1; i < NUM_TEAMS; ++i) {
        int key = order[i];
        int j = i - 1;
        while (j >= 0 && champions[order[j]] < champions[key]) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }

    printf("=== MonteGoal: %llu trials, seed %llu, RNG %s, block %d ===\n\n",
           num_trials, seed, RNG_NAME, threads_per_block);

    printf("Team                QF %%       SF %%     Final %%   Champion %%\n");
    printf("----------------------------------------------------------------\n");

    for (int i = 0; i < NUM_TEAMS; ++i) {
        int t = order[i];
        double qf_pct = 100.0 * (double)qf[t] / (double)num_trials;
        double sf_pct = 100.0 * (double)sf[t] / (double)num_trials;
        double final_pct = 100.0 * (double)final_counts[t] / (double)num_trials;
        double champion_pct = 100.0 * (double)champions[t] / (double)num_trials;

        printf("%-18s %8.2f%%   %8.2f%%   %8.2f%%   %8.2f%%\n",
               h_team_names[t], qf_pct, sf_pct, final_pct, champion_pct);
    }

    printf("\nGPU kernel time (CUDA events, kernel only) = %.3f ms\n", kernel_ms);
    printf("Tournaments per second (kernel only)       = %.3e\n",
           (double)num_trials / (kernel_ms / 1000.0));
    printf("Total wall-clock time (incl. setup)        = %.3f ms\n", wall_ms);

    return 0;
}
