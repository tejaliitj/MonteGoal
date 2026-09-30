#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "monte_carlo_kernel.cuh"   // engine
#include "tournament_kernel.cuh"    // football
#include "tournament_data.cuh"
#include "third_place_data.cuh"
#include "bracket_data.cuh"
#include "cuda_check.cuh"
#include "gpu_timer.cuh"
#ifndef USE_ELO_ONLY_LAMBDA
#include "lambda_matrix.cuh"
#endif

struct Stats {
    double min_ms, max_ms, mean_ms, median_ms;
};

static Stats compute_stats(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    Stats s;
    s.min_ms = times.front();
    s.max_ms = times.back();
    double sum = 0.0;
    for (double t : times) sum += t;
    s.mean_ms = sum / times.size();
    size_t n = times.size();
    s.median_ms = (n % 2 == 0) ? (times[n / 2 - 1] + times[n / 2]) / 2.0 : times[n / 2];
    return s;
}

// Times ONLY the kernel itself (malloc + launch + sync, via CUDA events) for
// one (num_trials, threads_per_block) configuration, repeated `repeats`
// times. Deliberately skips the device->host copy and host-side aggregation
// every other binary in this project does -- irrelevant to kernel
// throughput. The seed varies per repeat (seed + r) rather than staying
// fixed, so these numbers reflect the variance you'd actually see across
// differently-seeded real runs, not just fixed-workload hardware noise.
//
// Returns false (leaving *out untouched) if allocation or launch fails at
// any repeat -- most likely out of device memory at large num_trials, since
// the output array alone is num_trials * 4 bytes (4 GB at 1B trials). A
// failure here is reported and skipped by the caller, not fatal to the rest
// of the sweep -- CUDA_CHECK is deliberately NOT used for these two calls,
// since CUDA_CHECK exits the whole process on any failure.
static bool try_benchmark_config(long long num_trials, int threads_per_block, int repeats,
                                  unsigned long long base_seed, Stats* out) {
    std::vector<double> times;
    times.reserve(repeats);

    for (int r = 0; r < repeats; ++r) {
        int* d_out = nullptr;
        cudaError_t err = cudaMalloc(&d_out, num_trials * sizeof(int));
        if (err != cudaSuccess) {
            cudaGetLastError();  // clear the sticky error so later configs aren't poisoned by this one
            return false;
        }

        GpuTimer timer;
        timer.start();
        launch_monte_carlo(base_seed + r, 0ULL, (int)num_trials, d_out, SimulateTournament{},
                            threads_per_block);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_out);
            return false;
        }
        timer.stop();

        times.push_back(timer.elapsed_ms());
        CUDA_CHECK(cudaFree(d_out));
    }

    *out = compute_stats(times);
    return true;
}

int main(int argc, char** argv) {
    int repeats = (argc > 1) ? atoi(argv[1]) : 30;
#ifndef USE_ELO_ONLY_LAMBDA
    const char* lambda_csv_path = (argc > 2) ? argv[2] : "data/lambda_matrix.csv";
#endif

    upload_tournament_data();
    upload_thirdplace_table();
    upload_bracket_data();
#ifndef USE_ELO_ONLY_LAMBDA
    upload_lambda_matrix(lambda_csv_path);
#endif

    fprintf(stderr, "=== Benchmark: %d repeats per configuration, CUDA-event kernel timing ===\n", repeats);

    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    fprintf(stderr, "GPU memory: %.2f GB free / %.2f GB total\n", free_bytes / 1e9, total_bytes / 1e9);
    fprintf(stderr, "(each config needs num_trials * 4 bytes for its output array -- e.g. ~4 GB at 1B trials)\n\n");

    // ---- Phase 1: trial-count sweep at a fixed block size ----
    long long trial_counts[] = {10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    int num_trial_counts = (int)(sizeof(trial_counts) / sizeof(trial_counts[0]));
    int default_block_size = 128;

    FILE* f1 = fopen("benchmark_trials_sweep.csv", "w");
    fprintf(f1, "num_trials,min_ms,max_ms,mean_ms,median_ms,tournaments_per_sec\n");

    fprintf(stderr, "--- Phase 1: trial-count sweep (block size = %d) ---\n", default_block_size);
    printf("%-15s %10s %10s %10s %10s %18s\n",
           "Trials", "Min(ms)", "Max(ms)", "Mean(ms)", "Median(ms)", "Tournaments/sec");

    for (int i = 0; i < num_trial_counts; ++i) {
        long long n = trial_counts[i];
        fprintf(stderr, "  running N=%lld ...\n", n);
        Stats s;
        if (!try_benchmark_config(n, default_block_size, repeats, 42ULL, &s)) {
            printf("%-15lld %10s %10s %10s %10s %18s\n", n, "SKIPPED", "SKIPPED", "SKIPPED", "SKIPPED", "(alloc/launch failed)");
            fprintf(f1, "%lld,SKIPPED,SKIPPED,SKIPPED,SKIPPED,SKIPPED\n", n);
            fprintf(stderr, "    skipped -- allocation or launch failed (likely out of GPU memory at this size)\n");
            continue;
        }
        double throughput = n / (s.mean_ms / 1000.0);

        printf("%-15lld %10.3f %10.3f %10.3f %10.3f %18.0f\n",
               n, s.min_ms, s.max_ms, s.mean_ms, s.median_ms, throughput);
        fprintf(f1, "%lld,%.4f,%.4f,%.4f,%.4f,%.1f\n",
                n, s.min_ms, s.max_ms, s.mean_ms, s.median_ms, throughput);
    }
    fclose(f1);

    // ---- Phase 2: block-size sweep at a fixed, representative trial count ----
    long long fixed_trials = 10000000;  // large enough for a real signal, fast enough for many repeats
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_block_sizes = (int)(sizeof(block_sizes) / sizeof(block_sizes[0]));

    FILE* f2 = fopen("benchmark_blocksize_sweep.csv", "w");
    fprintf(f2, "block_size,min_ms,max_ms,mean_ms,median_ms,tournaments_per_sec\n");

    fprintf(stderr, "\n--- Phase 2: block-size sweep (trials = %lld) ---\n", fixed_trials);
    printf("\n%-15s %10s %10s %10s %10s %18s\n",
           "BlockSize", "Min(ms)", "Max(ms)", "Mean(ms)", "Median(ms)", "Tournaments/sec");

    for (int i = 0; i < num_block_sizes; ++i) {
        int bs = block_sizes[i];
        fprintf(stderr, "  running block_size=%d ...\n", bs);
        Stats s;
        if (!try_benchmark_config(fixed_trials, bs, repeats, 42ULL, &s)) {
            printf("%-15d %10s %10s %10s %10s %18s\n", bs, "SKIPPED", "SKIPPED", "SKIPPED", "SKIPPED", "(alloc/launch failed)");
            fprintf(f2, "%d,SKIPPED,SKIPPED,SKIPPED,SKIPPED,SKIPPED\n", bs);
            fprintf(stderr, "    skipped -- allocation or launch failed\n");
            continue;
        }
        double throughput = fixed_trials / (s.mean_ms / 1000.0);

        printf("%-15d %10.3f %10.3f %10.3f %10.3f %18.0f\n",
               bs, s.min_ms, s.max_ms, s.mean_ms, s.median_ms, throughput);
        fprintf(f2, "%d,%.4f,%.4f,%.4f,%.4f,%.1f\n",
                bs, s.min_ms, s.max_ms, s.mean_ms, s.median_ms, throughput);
    }
    fclose(f2);

    fprintf(stderr, "\nWrote benchmark_trials_sweep.csv and benchmark_blocksize_sweep.csv\n");
    return 0;
}
