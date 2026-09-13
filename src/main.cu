// =============================================================================
// main.cu -- The Monte Carlo Engine
// -----------------------------------------------------------------------------
// This is the top-level "engine" from your notes:
//
//     Monte Carlo Engine
//       |-- Distribution folder (Uniform, Poisson)   [include/distributions/]
//       |-- Random Number Generation (part of engine) [include/rng/]
//       `-- swappable application kernels             [include/kernels/]
//
// Usage:
//   ./montecarlo_engine <application> [num_threads] [trials_per_thread] [seed]
//
//   application        = "pi"        -> runs the Pi-estimation RNG test
//                       = "poisson"   -> runs the Poisson simulation kernel
//   num_threads         (default 1,048,576   = 1024 blocks * 1024 threads)
//   trials_per_thread   (default 1000)
//   seed                (default 1234)
//
// Example:
//   ./montecarlo_engine pi 1048576 1000 42
// =============================================================================
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../include/rng/rng_engine.cuh"
#include "../include/kernels/kernel_interface.cuh"
#include "../include/kernels/pi_estimation_kernel.cuh"
#include "../include/kernels/poisson_sim_kernel.cuh"
#include "../include/csv_writer.hpp"

// -----------------------------------------------------------------------------
// Kernel registry -- this is the "change kernel for different applications"
// switch. To add a new application: write a new .cuh in include/kernels/ that
// implements ApplicationKernel, then add one line here.
// -----------------------------------------------------------------------------
static std::unique_ptr<ApplicationKernel> select_kernel(const std::string& app_name) {
    if (app_name == "pi") {
        return std::make_unique<PiEstimationKernel>();
    }
    if (app_name == "poisson") {
        return std::make_unique<PoissonSimKernel>(/*lambda=*/4.0);
    }
    std::fprintf(stderr, "[ERROR] Unknown application '%s'. Valid options: pi, poisson\n",
                 app_name.c_str());
    return nullptr;
}

extern "C" {
#if defined(_WIN32)
__declspec(dllexport)
#endif
int run_engine(const char* app, int num_threads, int trials_per_thread, unsigned long long seed) {
    std::string app_name = app ? app : "pi";
    if (num_threads <= 0) num_threads = 1048576;
    if (trials_per_thread <= 0) trials_per_thread = 1000;

    std::printf("=== Monte Carlo Engine (1 Thread = 1 Simulation) ===\n");
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::printf("GPU Device           : %s (Compute %d.%d, %d SMs)\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    }
    std::printf("Application          : %s\n", app_name.c_str());
    std::printf("Simulations (Threads): %d\n", num_threads);
    std::printf("Trials per simulation: %d\n", trials_per_thread);
    std::printf("Seed                 : %llu\n", seed);
    std::printf("RNG generator        : cuRAND Philox4x32-10 (per-thread independent streams)\n\n");

    // ------------------ 1) Random Number Generation (engine part) ----------
    RNGEngine rng;
    rng.init(num_threads, seed);

    // ------------------------- 2) Pick the application kernel --------------
    auto kernel = select_kernel(app_name);
    if (!kernel) {
        return 1;
    }
    std::printf("Running kernel: %s\n", kernel->name());

    // ------------------------- 3) Allocate output buffer --------------------
    int out_n = kernel->out_size(rng);
    SimulationResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(SimulationResult)));

    // ------------------------- 4) Launch (the swappable part) ---------------
    kernel->launch(rng, trials_per_thread, d_out);

    // ------------------------- 5) Copy results back to host -----------------
    std::vector<SimulationResult> h_out(out_n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_n * sizeof(SimulationResult),
                           cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    // ------------------------- 6) Report per-thread results ------------------
    unsigned long long min_start_ns = 0;
    if (!h_out.empty()) {
        min_start_ns = h_out[0].start_ns;
        for (const auto& r : h_out) {
            if (r.start_ns < min_start_ns) min_start_ns = r.start_ns;
        }
    }

    std::printf("\n--- Per-Thread Simulation Results ---\n");
    auto print_thread = [&](int i, int width) {
        double start_us = (h_out[i].start_ns >= min_start_ns)
            ? static_cast<double>(h_out[i].start_ns - min_start_ns) / 1000.0 : 0.0;
        double end_us = (h_out[i].end_ns >= min_start_ns)
            ? static_cast<double>(h_out[i].end_ns - min_start_ns) / 1000.0 : 0.0;
        std::printf("[Thread %*d] Estimate: %9.6f | Error: %9.6f | Start: %9.3f us | End: %9.3f us | Duration: %8.3f us\n",
                    width, i, h_out[i].estimate, h_out[i].error, start_us, end_us, h_out[i].time_us);
    };

    if (out_n <= 100) {
        for (int i = 0; i < out_n; ++i) {
            print_thread(i, 5);
        }
    } else {
        // Print first 10 threads
        for (int i = 0; i < 10; ++i) {
            print_thread(i, 7);
        }
        std::printf("... [%d threads omitted from console; full per-thread details saved to CSV] ...\n",
                    out_n - 20);
        // Print last 10 threads
        for (int i = out_n - 10; i < out_n; ++i) {
            print_thread(i, 7);
        }
    }

    // ------------------------- 7) Reduce & summary statistics ---------------
    double sum_est = 0.0;
    double sum_err = 0.0;
    double sum_time = 0.0;
    float min_time = h_out.empty() ? 0.0f : h_out[0].time_us;
    float max_time = h_out.empty() ? 0.0f : h_out[0].time_us;

    for (const auto& res : h_out) {
        sum_est += res.estimate;
        sum_err += res.error;
        sum_time += res.time_us;
        if (res.time_us < min_time) min_time = res.time_us;
        if (res.time_us > max_time) max_time = res.time_us;
    }
    double mean_est = out_n > 0 ? (sum_est / out_n) : 0.0;
    double mean_err = out_n > 0 ? (sum_err / out_n) : 0.0;
    double avg_time = out_n > 0 ? (sum_time / out_n) : 0.0;

    std::printf("\n--- Summary Statistics Across All %d Simulations ---\n", out_n);
    std::printf("Ground Truth          : %.6f\n", kernel->ground_truth());
    std::printf("Mean Estimate         : %.6f\n", mean_est);
    std::printf("Mean Absolute Error   : %.6f\n", mean_err);
    std::printf("Min Thread Time       : %.3f us\n", min_time);
    std::printf("Max Thread Time       : %.3f us\n", max_time);
    std::printf("Avg Thread Time       : %.3f us\n", avg_time);

    // ------------------------- 8) Push dataset (CSV) -------------------------
    write_csv("data", kernel->csv_header(), h_out);

    return 0;
}
}

int main(int argc, char** argv) {
    // --------------------------- Parse CLI args -----------------------------
    std::string app_name        = (argc > 1) ? argv[1] : "pi";
    int num_threads              = (argc > 2) ? std::atoi(argv[2]) : 1048576;
    int trials_per_thread        = (argc > 3) ? std::atoi(argv[3]) : 1000;
    unsigned long long seed      = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 1234ULL;

    return run_engine(app_name.c_str(), num_threads, trials_per_thread, seed);
}
