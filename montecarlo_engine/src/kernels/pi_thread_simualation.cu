#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>

// ============================================================
// Configuration
// ============================================================

#define NUM_THREADS 1024
#define POINTS_PER_THREAD 1000

constexpr double ACTUAL_PI = 3.14159265358979323846;

// ============================================================
// CUDA error checking
// ============================================================

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ============================================================
// Result structure
// One result for every CUDA thread
// ============================================================

struct ThreadResult
{
    int thread_id;

    double estimated_pi;
    double actual_pi;
    double absolute_error;
    double percentage_error;

    unsigned long long elapsed_cycles;
};

// ============================================================
// Kernel 1: Initialize one Philox state per CUDA thread
// ============================================================

__global__
void initialize_rng(
    curandStatePhilox4_32_10_t* states,
    unsigned long long seed)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= NUM_THREADS)
        return;

    /*
        Every thread gets:

        states[0] -> Thread 0
        states[1] -> Thread 1
        states[2] -> Thread 2
        ...

        The sequence number is thread_id, giving every
        thread a different Philox random-number sequence.
    */

    curand_init(
        seed,
        thread_id,      // sequence number
        0,              // offset
        &states[thread_id]
    );
}

// ============================================================
// Kernel 2: One π simulation per CUDA thread
// ============================================================

__global__
void pi_simulation(
    curandStatePhilox4_32_10_t* states,
    ThreadResult* results)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= NUM_THREADS)
        return;

    // --------------------------------------------------------
    // Copy this thread's Philox state into a local variable
    // --------------------------------------------------------

    curandStatePhilox4_32_10_t local_state = states[thread_id];

    // --------------------------------------------------------
    // Start timing
    // --------------------------------------------------------

    unsigned long long start = clock64();

    // --------------------------------------------------------
    // Monte Carlo π calculation
    //
    // This ONE thread generates exactly:
    //
    // 1000 points
    //
    // Each point has:
    //     x = random number
    //     y = random number
    // --------------------------------------------------------

    int inside_circle = 0;

    for (int i = 0; i < POINTS_PER_THREAD; i++)
    {
        double x = curand_uniform_double(&local_state);
        double y = curand_uniform_double(&local_state);

        double distance_squared = x * x + y * y;

        if (distance_squared <= 1.0)
        {
            inside_circle++;
        }
    }

    // --------------------------------------------------------
    // Calculate estimated π
    // --------------------------------------------------------

    double estimated_pi =
        4.0 * static_cast<double>(inside_circle)
        / static_cast<double>(POINTS_PER_THREAD);

    // --------------------------------------------------------
    // Calculate errors
    // --------------------------------------------------------

    double absolute_error =
        fabs(estimated_pi - ACTUAL_PI);

    double percentage_error =
        (absolute_error / ACTUAL_PI) * 100.0;

    // --------------------------------------------------------
    // Stop timing
    // --------------------------------------------------------

    unsigned long long end = clock64();

    unsigned long long elapsed_cycles = end - start;

    // --------------------------------------------------------
    // Save updated Philox state
    // --------------------------------------------------------

    states[thread_id] = local_state;

    // --------------------------------------------------------
    // Store this thread's result
    // --------------------------------------------------------

    results[thread_id].thread_id = thread_id;

    results[thread_id].estimated_pi = estimated_pi;
    results[thread_id].actual_pi = ACTUAL_PI;
    results[thread_id].absolute_error = absolute_error;
    results[thread_id].percentage_error = percentage_error;

    results[thread_id].elapsed_cycles = elapsed_cycles;
}

// ============================================================
// Main
// ============================================================

int main()
{
    std::cout << "=============================================\n";
    std::cout << " CUDA Monte Carlo Pi Simulation\n";
    std::cout << "=============================================\n\n";

    std::cout << "CUDA Threads       : " << NUM_THREADS << "\n";
    std::cout << "Points per Thread  : " << POINTS_PER_THREAD << "\n";
    std::cout << "Total Points       : "
              << static_cast<long long>(NUM_THREADS)
                 * POINTS_PER_THREAD
              << "\n\n";

    // ========================================================
    // Allocate GPU memory
    // ========================================================

    curandStatePhilox4_32_10_t* d_states = nullptr;
    ThreadResult* d_results = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_states,
        NUM_THREADS * sizeof(curandStatePhilox4_32_10_t)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_results,
        NUM_THREADS * sizeof(ThreadResult)
    ));

    // ========================================================
    // Generate a different seed for every program execution
    // ========================================================

    unsigned long long seed =
        static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count()
        );

    std::cout << "Random Seed        : " << seed << "\n\n";

    // ========================================================
    // CUDA launch configuration
    // ========================================================

    int block_size = 256;

    int num_blocks =
        (NUM_THREADS + block_size - 1) / block_size;

    std::cout << "Block Size         : " << block_size << "\n";
    std::cout << "Number of Blocks   : " << num_blocks << "\n\n";

    // ========================================================
    // Initialize Philox states
    // ========================================================

    initialize_rng<<<num_blocks, block_size>>>(
        d_states,
        seed
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================
    // Run π simulation
    // ========================================================

    auto cpu_start = std::chrono::high_resolution_clock::now();

    pi_simulation<<<num_blocks, block_size>>>(
        d_states,
        d_results
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cpu_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total_gpu_time =
        cpu_end - cpu_start;

    std::cout << "Kernel completed.\n";
    std::cout << "Total kernel time  : "
              << total_gpu_time.count()
              << " ms\n\n";

    // ========================================================
    // Copy results back to CPU
    // ========================================================

    ThreadResult* h_results =
        new ThreadResult[NUM_THREADS];

    CUDA_CHECK(cudaMemcpy(
        h_results,
        d_results,
        NUM_THREADS * sizeof(ThreadResult),
        cudaMemcpyDeviceToHost
    ));

    // ========================================================
    // Get GPU clock frequency
    // ========================================================

    cudaDeviceProp device_properties;

    CUDA_CHECK(cudaGetDeviceProperties(
        &device_properties,
        0
    ));

    /*
        clockRate is in kHz.

        cycles / (clockRate * 1000)
        = seconds

        Therefore:

        microseconds =
        cycles / clockRate / 1000
    */

    double clock_rate_khz =
        static_cast<double>(device_properties.clockRate);

    // ========================================================
    // Open CSV file
    // ========================================================

    std::ofstream csv("pi_thread_results.csv");

    if (!csv.is_open())
    {
        std::cerr << "ERROR: Could not create CSV file.\n";

        delete[] h_results;

        CUDA_CHECK(cudaFree(d_states));
        CUDA_CHECK(cudaFree(d_results));

        return EXIT_FAILURE;
    }

    // ========================================================
    // CSV Header
    // ========================================================

    csv << "Thread_ID,"
        << "Estimated_Pi,"
        << "Actual_Pi,"
        << "Absolute_Error,"
        << "Percentage_Error,"
        << "Time_us\n";

    csv << std::setprecision(15);

    // ========================================================
    // Write one row per CUDA thread
    // ========================================================

    for (int i = 0; i < NUM_THREADS; i++)
    {
        double time_us =
            static_cast<double>(
                h_results[i].elapsed_cycles
            )
            / clock_rate_khz
            / 1000.0;

        csv << h_results[i].thread_id << ","
            << h_results[i].estimated_pi << ","
            << h_results[i].actual_pi << ","
            << h_results[i].absolute_error << ","
            << h_results[i].percentage_error << ","
            << time_us
            << "\n";
    }

    csv.close();

    // ========================================================
    // Print first 10 results
    // ========================================================

    std::cout << "First 10 thread results:\n\n";

    std::cout
        << std::left
        << std::setw(10) << "Thread"
        << std::setw(18) << "Estimated Pi"
        << std::setw(18) << "Abs Error"
        << std::setw(18) << "% Error"
        << std::setw(15) << "Time (us)"
        << "\n";

    std::cout
        << "--------------------------------------------------------------------------\n";

    for (int i = 0; i < 10 && i < NUM_THREADS; i++)
    {
        double time_us =
            static_cast<double>(
                h_results[i].elapsed_cycles
            )
            / clock_rate_khz
            / 1000.0;

        std::cout
            << std::left
            << std::setw(10) << h_results[i].thread_id
            << std::setw(18) << h_results[i].estimated_pi
            << std::setw(18) << h_results[i].absolute_error
            << std::setw(18) << h_results[i].percentage_error
            << std::setw(15) << time_us
            << "\n";
    }

    // ========================================================
    // Cleanup
    // ========================================================

    delete[] h_results;

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_results));

    std::cout << "\nResults saved to:\n";
    std::cout << "pi_thread_results.csv\n";

    std::cout << "\nSimulation finished successfully.\n";

    return 0;
}