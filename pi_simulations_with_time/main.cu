#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <curand_kernel.h>

#include "rng.cuh"
#include "simulation/monte_carlo.cuh"

using namespace std;

// Small helper so we don't repeat the same 6-line error-check block
// after every single CUDA call.
static bool checkCuda(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
    {
        cout << what << ": " << cudaGetErrorString(err) << endl;
        return false;
    }
    return true;
}

int main()
{
    // ---------------- User settings ----------------
    int total_threads      = 16384;   // one "simulation" per thread
    int points_per_thread  = 50000;   // iterations per simulation
    unsigned long long seed = 42;
    int threads_per_block  = 256;

    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    cout << "============================================\n";
    cout << "        CUDA MONTE CARLO PI SIMULATION\n";
    cout << "============================================\n";
    cout << "RNG               : Philox\n";
    cout << "Seed              : " << seed << endl;
    cout << "Total Threads     : " << total_threads << endl;
    cout << "Points/Thread     : " << points_per_thread << endl;
    cout << "Threads/Block     : " << threads_per_block << endl;
    cout << "Blocks            : " << blocks << endl;
    cout << "Total Points      : "
         << (long long)total_threads * points_per_thread << endl;
    cout << "============================================\n";

    // ---------------- Device allocations ----------------
    curandStatePhilox4_32_10_t *d_states = nullptr;
    int *d_hits = nullptr;
    unsigned long long *d_start_ns = nullptr;
    unsigned long long *d_end_ns   = nullptr;

    if (!checkCuda(cudaMalloc(&d_states, total_threads * sizeof(curandStatePhilox4_32_10_t)), "RNG Memory Allocation Error"))
        return 1;

    if (!checkCuda(cudaMalloc(&d_hits, total_threads * sizeof(int)), "Hits Memory Allocation Error"))
    {
        cudaFree(d_states);
        return 1;
    }

    if (!checkCuda(cudaMalloc(&d_start_ns, total_threads * sizeof(unsigned long long)), "Start Timestamp Allocation Error"))
    {
        cudaFree(d_states);
        cudaFree(d_hits);
        return 1;
    }

    if (!checkCuda(cudaMalloc(&d_end_ns, total_threads * sizeof(unsigned long long)), "End Timestamp Allocation Error"))
    {
        cudaFree(d_states);
        cudaFree(d_hits);
        cudaFree(d_start_ns);
        return 1;
    }

    // ---------------- RNG setup ----------------
    setup_rng<<<blocks, threads_per_block>>>(d_states, seed, total_threads);

    if (!checkCuda(cudaGetLastError(), "RNG Kernel Error") ||
        !checkCuda(cudaDeviceSynchronize(), "RNG Synchronization Error"))
    {
        cudaFree(d_states); cudaFree(d_hits); cudaFree(d_start_ns); cudaFree(d_end_ns);
        return 1;
    }

    // ---------------- Monte Carlo kernel (timed) ----------------
    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    cudaEventRecord(evStart);

    monte_carlo_kernel<<<blocks, threads_per_block>>>(
        d_states, points_per_thread, total_threads,
        d_hits, d_start_ns, d_end_ns
    );

    cudaEventRecord(evStop);

    if (!checkCuda(cudaGetLastError(), "Monte Carlo Kernel Error") ||
        !checkCuda(cudaEventSynchronize(evStop), "Monte Carlo Synchronization Error"))
    {
        cudaFree(d_states); cudaFree(d_hits); cudaFree(d_start_ns); cudaFree(d_end_ns);
        return 1;
    }

    float gpu_exec_ms = 0.0f;
    cudaEventElapsedTime(&gpu_exec_ms, evStart, evStop);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);

    // ---------------- Copy results back ----------------
    vector<int> h_hits(total_threads);
    vector<unsigned long long> h_start_ns(total_threads);
    vector<unsigned long long> h_end_ns(total_threads);

    bool copyOk =
        checkCuda(cudaMemcpy(h_hits.data(), d_hits, total_threads * sizeof(int), cudaMemcpyDeviceToHost), "Hits Copy Error") &&
        checkCuda(cudaMemcpy(h_start_ns.data(), d_start_ns, total_threads * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "Start Timestamp Copy Error") &&
        checkCuda(cudaMemcpy(h_end_ns.data(), d_end_ns, total_threads * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "End Timestamp Copy Error");

    cudaFree(d_states);
    cudaFree(d_hits);
    cudaFree(d_start_ns);
    cudaFree(d_end_ns);

    if (!copyOk)
        return 1;

    // ---------------- Per-thread derived stats ----------------
    const double actual_pi = acos(-1.0);

    vector<double> thread_pi(total_threads);
    vector<double> abs_error(total_threads);
    vector<double> pct_error(total_threads);
    vector<double> start_ms(total_threads);
    vector<double> end_ms(total_threads);
    vector<double> duration_ms(total_threads);

    long long total_hits = 0;

    for (int i = 0; i < total_threads; i++)
    {
        thread_pi[i]   = 4.0 * (double)h_hits[i] / (double)points_per_thread;
        abs_error[i]   = fabs(actual_pi - thread_pi[i]);
        pct_error[i]   = (abs_error[i] / actual_pi) * 100.0;
        start_ms[i]    = (double)h_start_ns[i] / 1.0e6;
        end_ms[i]      = (double)h_end_ns[i]   / 1.0e6;
        duration_ms[i] = (double)(h_end_ns[i] - h_start_ns[i]) / 1.0e6;

        total_hits += h_hits[i];
    }

    double estimated_pi = 4.0 * (double)total_hits /
                           ((double)total_threads * (double)points_per_thread);
    double abs_summary_error = fabs(actual_pi - estimated_pi);
    double pct_summary_error = (abs_summary_error / actual_pi) * 100.0;

    // ---------------- Aggregated summary (console) ----------------
    cout << "\n============================================\n";
    cout << "               AGGREGATED SUMMARY\n";
    cout << "============================================\n";
    cout << "Number of Blocks       : " << blocks << endl;
    cout << "Block Size (CUDA Threads): " << threads_per_block << endl;
    cout << "Total CUDA Threads     : " << total_threads << endl;
    cout << "Iterations Per Thread  : " << points_per_thread << endl;
    cout << "Total Iterations       : "
         << (long long)total_threads * points_per_thread << endl;
    cout << fixed << setprecision(4);
    cout << "GPU Execution Time     : " << gpu_exec_ms << " ms\n";
    cout << setprecision(6);
    cout << "Estimated Pi           : " << estimated_pi << endl;
    cout << "True Pi                : " << actual_pi << endl;
    cout << scientific << setprecision(5);
    cout << "Absolute Summary Error : " << abs_summary_error << endl;
    cout << fixed << setprecision(5);
    cout << "Percentage Error       : " << pct_summary_error << " %\n";
    cout << "============================================\n";

    // ---------------- First-N preview (console) ----------------
    int preview_n = min(total_threads, 10);

    cout << "\nFirst " << preview_n << " Threads Preview:\n\n";
    cout << left
         << setw(8)  << "Sim_ID"
         << setw(8)  << "Hits"
         << setw(12) << "Est_Pi"
         << setw(14) << "Abs_Error"
         << setw(10) << "Error (%)"
         << setw(16) << "Start (ms)"
         << setw(16) << "End (ms)"
         << setw(12) << "Duration (ms)"
         << endl;
    cout << string(96, '-') << endl;

    for (int i = 0; i < preview_n; i++)
    {
        cout << left
             << setw(8)  << i
             << setw(8)  << h_hits[i]
             << fixed << setprecision(6) << setw(12) << thread_pi[i]
             << setprecision(8) << setw(14) << abs_error[i]
             << setprecision(4) << setw(10) << pct_error[i]
             << setprecision(4) << setw(16) << start_ms[i]
             << setw(16) << end_ms[i]
             << setw(12) << duration_ms[i]
             << endl;
    }
    cout << string(96, '-') << endl;

    // ---------------- Concurrency analysis ----------------
    // Sweep-line over every thread's [start, end] interval: +1 when a
    // thread starts, -1 when it ends. The running total's peak value is
    // the max number of threads that were *simultaneously* executing.
    double min_start = start_ms[0], max_start = start_ms[0];
    double min_end   = end_ms[0],   max_end   = end_ms[0];

    vector<pair<unsigned long long, int>> events;
    events.reserve(2 * total_threads);

    for (int i = 0; i < total_threads; i++)
    {
        events.push_back({h_start_ns[i], +1});
        events.push_back({h_end_ns[i],   -1});

        min_start = min(min_start, start_ms[i]);
        max_start = max(max_start, start_ms[i]);
        min_end   = min(min_end,   end_ms[i]);
        max_end   = max(max_end,   end_ms[i]);
    }

    // At equal timestamps, process an "end" before a "start" so a thread
    // finishing at exactly time t isn't double-counted with one starting
    // at the same instant.
    sort(events.begin(), events.end(), [](const pair<unsigned long long, int> &a,
                                           const pair<unsigned long long, int> &b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    int running = 0, max_concurrent = 0;
    for (auto &e : events)
    {
        running += e.second;
        max_concurrent = max(max_concurrent, running);
    }

    cout << "\n============================================\n";
    cout << "          THREAD CONCURRENCY ANALYSIS\n";
    cout << "============================================\n";
    cout << fixed << setprecision(4);
    cout << "Earliest Thread Start (ms) : " << min_start << endl;
    cout << "Latest Thread Start (ms)   : " << max_start << endl;
    cout << "Start Time Spread (ms)     : " << (max_start - min_start) << endl;
    cout << "Earliest Thread End (ms)   : " << min_end << endl;
    cout << "Latest Thread End (ms)     : " << max_end << endl;
    cout << "Max Concurrent Threads     : " << max_concurrent
         << " / " << total_threads << endl;
    cout << "============================================\n";
    cout << "(A small Start Time Spread + Max Concurrent Threads close to\n";
    cout << " Total CUDA Threads means sim[0], sim[1], ... genuinely ran\n";
    cout << " together, not one after another.)\n";

    // ---------------- Full per-thread CSV export ----------------
    const string csv_path = "simulation_results.csv";
    ofstream csv(csv_path);

    if (!csv.is_open())
    {
        cout << "\nWarning: could not open " << csv_path << " for writing.\n";
    }
    else
    {
        csv << "Simulation_ID,Hits,Iterations_Per_Simulation,Estimated_Pi,Actual_Pi,"
               "Absolute_Error,Percentage_Error_Percent,Start_Time_ms,End_Time_ms,Duration_ms\n";

        csv << fixed << setprecision(11);

        for (int i = 0; i < total_threads; i++)
        {
            csv << i << ","
                << h_hits[i] << ","
                << points_per_thread << ","
                << setprecision(6) << thread_pi[i] << ","
                << setprecision(11) << actual_pi << ","
                << setprecision(8) << abs_error[i] << ","
                << setprecision(6) << pct_error[i] << ","
                << setprecision(6) << start_ms[i] << ","
                << end_ms[i] << ","
                << duration_ms[i] << "\n";
        }

        csv.close();
        cout << "\nFull per-simulation results written to: " << csv_path << endl;
    }

    return 0;
}
