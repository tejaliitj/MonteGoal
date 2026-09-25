#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "poisson_check.cuh"
#include "cuda_check.cuh"
#include "gpu_timer.cuh"

int main(int argc, char** argv) {
    int num_trials = (argc > 1) ? atoi(argv[1]) : 100;
    long long draws_per_trial = (argc > 2) ? atoll(argv[2]) : 1000000LL;
    double lambda = (argc > 3) ? atof(argv[3]) : 1.7;  // arbitrary test value, not real model output
    unsigned long long seed = 42ULL;
    unsigned long long counter_offset = 0ULL;

    float* d_abs_error;
    CUDA_CHECK(cudaMalloc(&d_abs_error, num_trials * sizeof(float)));

    GpuTimer timer;
    timer.start();
    launch_poisson_check(d_abs_error, num_trials, draws_per_trial, lambda,
                          seed, counter_offset);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();

    float* h_abs_error = (float*)malloc(num_trials * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_abs_error, d_abs_error, num_trials * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_abs_error));

    double sum = 0.0;
    double max_err = h_abs_error[0];
    for (int i = 0; i < num_trials; ++i) {
        sum += h_abs_error[i];
        if (h_abs_error[i] > max_err) max_err = h_abs_error[i];
    }
    double mean_err = sum / num_trials;

    printf("lambda          = %.3f\n", lambda);
    printf("Trials          = %d\n", num_trials);
    printf("Draws / trial   = %lld\n\n", draws_per_trial);
    printf("Mean abs error  = %.6f\n", mean_err);
    printf("Max  abs error  = %.6f\n", max_err);
    printf("GPU kernel time = %.3f ms\n", timer.elapsed_ms());

    free(h_abs_error);
    return 0;
}
