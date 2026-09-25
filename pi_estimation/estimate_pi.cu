#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include "philox_engine.cuh"
#include "gpu_timer.cuh"

int main(int argc, char** argv) {
    int num_simulations = (argc > 1) ? atoi(argv[1]) : 100;
    long long points_per_simulation = (argc > 2) ? atoll(argv[2]) : 1000000LL;
    unsigned long long seed = 42ULL;
    double pi_actual = 3.14159265358979323846;

    auto wall_start = std::chrono::high_resolution_clock::now();

    PhiloxEngine engine(seed);

    GpuTimer gpu_timer;
    gpu_timer.start();
    float* d_abs_error = engine.sample_pi_trials(num_simulations, points_per_simulation, pi_actual);
    gpu_timer.stop();

    float* h_abs_error = (float*)malloc(num_simulations * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_abs_error, d_abs_error, num_simulations * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_abs_error));

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    double sum = 0.0;
    double min_err = h_abs_error[0];
    double max_err = h_abs_error[0];
    for (int i = 0; i < num_simulations; ++i) {
        sum += h_abs_error[i];
        if (h_abs_error[i] < min_err) min_err = h_abs_error[i];
        if (h_abs_error[i] > max_err) max_err = h_abs_error[i];
    }
    double mean_err = sum / num_simulations;

    printf("Simulations         = %d\n", num_simulations);
    printf("Points / simulation = %lld\n\n", points_per_simulation);

    int print_count = (num_simulations < 20) ? num_simulations : 20;
    for (int i = 0; i < print_count; ++i) {
        printf("  sim[%2d] abs_error = %.6f\n", i, h_abs_error[i]);
    }
    if (num_simulations > print_count) {
        printf("  ... (%d more)\n", num_simulations - print_count);
    }

    printf("\nMean abs error = %.6f\n", mean_err);
    printf("Min  abs error = %.6f\n", min_err);
    printf("Max  abs error = %.6f\n\n", max_err);

    printf("GPU kernel time (malloc+launch+sync) = %.3f ms\n", gpu_timer.elapsed_ms());
    printf("Total wall-clock time (incl. memcpy)  = %.3f ms\n", wall_ms);

    free(h_abs_error);
    return 0;
}
