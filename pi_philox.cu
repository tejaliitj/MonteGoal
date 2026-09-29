#include <iostream>
#include <random>
#include <iomanip>

#include <curand_kernel.h>
#include <cuda_runtime.h>

#define N 10000000

// ------------------------------------------------------------
// CUDA kernel
// Generate random points and count points inside unit circle
// ------------------------------------------------------------
__global__ void estimate_pi(
    curandStatePhilox4_32_10_t* states,
    unsigned long long* inside,
    unsigned long long seed
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // Each thread gets a unique Philox sequence
    curand_init(
        seed,
        idx,
        0,
        &states[idx]
    );

    // Generate two random numbers in (0,1]
    float u1 = curand_uniform(&states[idx]);
    float u2 = curand_uniform(&states[idx]);

    // Convert them from (0,1] to (-1,1]
    float x = 2.0f * u1 - 1.0f;
    float y = 2.0f * u2 - 1.0f;

    // Check if point is inside unit circle
    float distance_squared = x * x + y * y;

    if (distance_squared <= 1.0f)
    {
        atomicAdd(inside, 1ULL);
    }
}


// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main()
{
    // --------------------------------------------------------
    // Generate a NEW seed every program execution
    // --------------------------------------------------------

    std::random_device rd;

    unsigned long long seed =
        (static_cast<unsigned long long>(rd()) << 32) |
        static_cast<unsigned long long>(rd());

    std::cout << "Monte Carlo Pi Estimation using Philox\n";
    std::cout << "========================================\n";

    std::cout << "Seed: "
              << seed
              << "\n\n";

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------

    curandStatePhilox4_32_10_t* d_states = nullptr;

    unsigned long long* d_inside = nullptr;

    cudaError_t err;

    err = cudaMalloc(
        &d_states,
        N * sizeof(curandStatePhilox4_32_10_t)
    );

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate RNG states: "
                  << cudaGetErrorString(err)
                  << '\n';

        return 1;
    }

    err = cudaMalloc(
        &d_inside,
        sizeof(unsigned long long)
    );

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate counter: "
                  << cudaGetErrorString(err)
                  << '\n';

        cudaFree(d_states);

        return 1;
    }

    // --------------------------------------------------------
    // Initialize counter to zero
    // --------------------------------------------------------

    err = cudaMemset(
        d_inside,
        0,
        sizeof(unsigned long long)
    );

    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemset failed: "
                  << cudaGetErrorString(err)
                  << '\n';

        cudaFree(d_states);
        cudaFree(d_inside);

        return 1;
    }

    // --------------------------------------------------------
    // CUDA configuration
    // --------------------------------------------------------

    const int threadsPerBlock = 256;

    const int blocks =
        (N + threadsPerBlock - 1) /
        threadsPerBlock;

    std::cout << "Generating "
              << N
              << " random points...\n";

    // --------------------------------------------------------
    // Launch kernel
    // --------------------------------------------------------

    estimate_pi<<<blocks, threadsPerBlock>>>(
        d_states,
        d_inside,
        seed
    );

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: "
                  << cudaGetErrorString(err)
                  << '\n';

        cudaFree(d_states);
        cudaFree(d_inside);

        return 1;
    }

    // Wait for GPU
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cerr << "Kernel execution failed: "
                  << cudaGetErrorString(err)
                  << '\n';

        cudaFree(d_states);
        cudaFree(d_inside);

        return 1;
    }

    // --------------------------------------------------------
    // Copy result GPU -> CPU
    // --------------------------------------------------------

    unsigned long long h_inside = 0;

    err = cudaMemcpy(
        &h_inside,
        d_inside,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(err)
                  << '\n';

        cudaFree(d_states);
        cudaFree(d_inside);

        return 1;
    }

    // --------------------------------------------------------
    // Calculate Pi
    // --------------------------------------------------------

    double pi =
        4.0 *
        static_cast<double>(h_inside) /
        static_cast<double>(N);

    // --------------------------------------------------------
    // Calculate error
    // --------------------------------------------------------

    constexpr double PI_TRUE =
        3.14159265358979323846;

    double absolute_error =
        std::abs(pi - PI_TRUE);

    double percentage_error =
        (absolute_error / PI_TRUE) * 100.0;

    // --------------------------------------------------------
    // Display results
    // --------------------------------------------------------

    std::cout << "\n";
    std::cout << "Results\n";
    std::cout << "-------\n";

    std::cout << "Total points       : "
              << N
              << '\n';

    std::cout << "Points inside      : "
              << h_inside
              << '\n';

    std::cout << std::fixed
              << std::setprecision(12);

    std::cout << "Estimated Pi       : "
              << pi
              << '\n';

    std::cout << "Actual Pi          : "
              << PI_TRUE
              << '\n';

    std::cout << "Absolute error     : "
              << absolute_error
              << '\n';

    std::cout << "Percentage error   : "
              << percentage_error
              << "%\n";

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    cudaFree(d_states);
    cudaFree(d_inside);

    return 0;
}