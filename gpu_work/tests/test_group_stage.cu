#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "group_stage_check.cuh"
#include "tournament_data.cuh"
#include "lambda_matrix.cuh"
#include "cuda_check.cuh"
#include "gpu_timer.cuh"

int main(int argc, char** argv) {
    int num_trials = (argc > 1) ? atoi(argv[1]) : 100000;
    unsigned long long seed = 42ULL;

    upload_tournament_data();
    upload_lambda_matrix();

    // Analytic reference: every fixture's lambda is fixed (looked up from the
    // real trained-model matrix, same every trial), so the expected total goals
    // per trial is just the sum of every fixture's lambda_home + lambda_away,
    // read directly from the matrix -- independent of the device code path.
    double expected_total = 0.0;
    for (int f = 0; f < NUM_GROUP_FIXTURES; ++f) {
        int home_id = h_fixture_home[f];
        int away_id = h_fixture_away[f];
        expected_total += h_lambda_matrix[home_id][away_id] + h_lambda_matrix[away_id][home_id];
    }

    int* d_total_goals;
    CUDA_CHECK(cudaMalloc(&d_total_goals, num_trials * sizeof(int)));

    GpuTimer timer;
    timer.start();
    launch_group_stage_check(d_total_goals, num_trials, seed, 0ULL);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();

    int* h_total_goals = (int*)malloc(num_trials * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_total_goals, d_total_goals, num_trials * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_total_goals));

    double sum = 0.0;
    for (int i = 0; i < num_trials; ++i) sum += h_total_goals[i];
    double empirical_mean = sum / num_trials;

    printf("Trials                               = %d\n", num_trials);
    printf("Analytic expected total goals/trial  = %.4f\n", expected_total);
    printf("Empirical mean total goals/trial     = %.4f\n", empirical_mean);
    printf("Abs error                            = %.4f\n",
           fabs(expected_total - empirical_mean));
    printf("GPU kernel time                      = %.3f ms\n", timer.elapsed_ms());

    free(h_total_goals);
    return 0;
}
