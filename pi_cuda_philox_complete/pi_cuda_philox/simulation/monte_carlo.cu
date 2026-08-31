#include "monte_carlo.cuh"
#include "../distributions/uniform.cuh"

__global__ void monte_carlo_kernel(
    curandStatePhilox4_32_10_t *states,
    int points_per_thread,
    int total_threads,
    float *thread_pi_results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_threads)
        return;

    // Copy this thread's Philox state
    curandStatePhilox4_32_10_t local_state = states[idx];

    int inside = 0;

    // Generate random points
    for (int i = 0; i < points_per_thread; i++)
    {
        float rx = generate_uniform(&local_state);
        float ry = generate_uniform(&local_state);

        // Convert U(0,1) to U(-1,1)
        float x = 2.0f * rx - 1.0f;
        float y = 2.0f * ry - 1.0f;

        // Check if point is inside unit circle
        if (x * x + y * y <= 1.0f)
            inside++;
    }

    // Pi estimate for this thread
    thread_pi_results[idx] =
        4.0f * inside / points_per_thread;

    // Save updated RNG state
    states[idx] = local_state;
}
