#include "monte_carlo.cuh"
#include "../distributions/uniform.cuh"

// Reads the GPU's global nanosecond timer via inline PTX.
// %globaltimer is a single free-running counter shared across the whole
// device (every SM reads the same clock), so timestamps taken by
// different threads are directly comparable -- unlike clock64(), which
// is per-SM and not synchronized across SMs.
__device__ __forceinline__ unsigned long long global_timer_ns()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void monte_carlo_kernel(
    curandStatePhilox4_32_10_t *states,
    int points_per_thread,
    int total_threads,
    int *d_hits,
    unsigned long long *d_start_ns,
    unsigned long long *d_end_ns)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_threads)
        return;

    // Timestamp the instant this thread actually began doing work.
    unsigned long long t_start = global_timer_ns();

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

    // Timestamp the instant this thread finished its work.
    unsigned long long t_end = global_timer_ns();

    d_hits[idx]     = inside;
    d_start_ns[idx] = t_start;
    d_end_ns[idx]   = t_end;

    // Save updated RNG state
    states[idx] = local_state;
}
