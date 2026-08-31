#include <iostream>
#include <cmath>
#include <iomanip>
#include <curand_kernel.h>

#include "rng.cuh"
#include "simulation/monte_carlo.cuh"

using namespace std;

int main()
{
    // User settings
    int total_threads = 1000;
    int points_per_thread = 1000;
    unsigned long long seed = 42;
    int threads_per_block = 256;

    int blocks =
        (total_threads + threads_per_block - 1) /
        threads_per_block;

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
         << (long long)total_threads * points_per_thread
         << endl;

    cout << "============================================\n";

    // Allocate Philox RNG states on GPU
    curandStatePhilox4_32_10_t *d_states;

    cudaError_t error = cudaMalloc(
        &d_states,
        total_threads * sizeof(curandStatePhilox4_32_10_t)
    );

    if (error != cudaSuccess)
    {
        cout << "RNG Memory Allocation Error: "
             << cudaGetErrorString(error) << endl;
        return 1;
    }

    // Allocate thread-wise Pi results
    float *d_thread_pi;

    error = cudaMalloc(
        &d_thread_pi,
        total_threads * sizeof(float)
    );

    if (error != cudaSuccess)
    {
        cout << "Pi Memory Allocation Error: "
             << cudaGetErrorString(error) << endl;

        cudaFree(d_states);
        return 1;
    }

    // Initialize one Philox RNG state per thread
    setup_rng<<<blocks, threads_per_block>>>(
        d_states,
        seed,
        total_threads
    );

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        cout << "RNG Kernel Error: "
             << cudaGetErrorString(error) << endl;

        cudaFree(d_states);
        cudaFree(d_thread_pi);
        return 1;
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        cout << "RNG Synchronization Error: "
             << cudaGetErrorString(error) << endl;

        cudaFree(d_states);
        cudaFree(d_thread_pi);
        return 1;
    }

    // Run Monte Carlo simulation
    monte_carlo_kernel<<<blocks, threads_per_block>>>(
        d_states,
        points_per_thread,
        total_threads,
        d_thread_pi
    );

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        cout << "Monte Carlo Kernel Error: "
             << cudaGetErrorString(error) << endl;

        cudaFree(d_states);
        cudaFree(d_thread_pi);
        return 1;
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        cout << "Monte Carlo Synchronization Error: "
             << cudaGetErrorString(error) << endl;

        cudaFree(d_states);
        cudaFree(d_thread_pi);
        return 1;
    }

    // Allocate CPU memory
    float *h_thread_pi = new float[total_threads];

    // Copy results GPU -> CPU
    error = cudaMemcpy(
        h_thread_pi,
        d_thread_pi,
        total_threads * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess)
    {
        cout << "Memory Copy Error: "
             << cudaGetErrorString(error) << endl;

        delete[] h_thread_pi;
        cudaFree(d_states);
        cudaFree(d_thread_pi);
        return 1;
    }

    // Print thread-wise Pi
    cout << "\nTHREAD-WISE PI ESTIMATES\n";
    cout << "--------------------------------------------\n";

    cout << left
         << setw(12) << "Thread"
         << setw(15) << "Pi Estimate"
         << endl;

    cout << "--------------------------------------------\n";

    double sum_pi = 0.0;

    for (int i = 0; i < total_threads; i++)
    {
        cout << left
             << setw(12) << i
             << fixed
             << setprecision(6)
             << setw(15) << h_thread_pi[i]
             << endl;

        sum_pi += h_thread_pi[i];
    }

    // Final result
    double average_pi = sum_pi / total_threads;
    double actual_pi = acos(-1.0);

    double absolute_error =
        fabs(actual_pi - average_pi);

    double percentage_error =
        (absolute_error / actual_pi) * 100.0;

    cout << "\n============================================\n";
    cout << "              FINAL RESULT\n";
    cout << "============================================\n";

    cout << "Average Pi        : "
         << fixed << setprecision(10)
         << average_pi << endl;

    cout << "Actual Pi         : "
         << actual_pi << endl;

    cout << "Absolute Error    : "
         << absolute_error << endl;

    cout << "Percentage Error  : "
         << percentage_error << "%" << endl;

    cout << "============================================\n";

    // Free memory
    cudaFree(d_states);
    cudaFree(d_thread_pi);
    delete[] h_thread_pi;

    return 0;
}
