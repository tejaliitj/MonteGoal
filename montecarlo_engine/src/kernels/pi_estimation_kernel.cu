// #include "kernels/pi_estimation_kernel.cuh"
// #include "distributions/uniform_dist.cuh"

// __global__ void estimate_pi_kernel(
//     curandStatePhilox4_32_10_t* states,
//     unsigned long long* inside,
//     int n)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n) return;

//     float u1 = sample_uniform(&states[idx]);
//     float u2 = sample_uniform(&states[idx]);

//     float x = 2.0f * u1 - 1.0f;
//     float y = 2.0f * u2 - 1.0f;

//     if (x * x + y * y <= 1.0f)
//         atomicAdd(inside, 1ULL);
// }



#include "kernels/pi_estimation_kernel.cuh"

#include "rng/rng_engine.cuh"
#include "distributions/uniform_dist.cuh"

#include <cuda_runtime.h>

__global__ void pi_estimation_kernel(
    int* inside,
    unsigned long long points_per_thread,
    unsigned long long seed
) {
    unsigned long long idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;

    init_philox(
        &state,
        seed,
        idx
    );

    int local_inside = 0;

    for (unsigned long long i = 0;
         i < points_per_thread;
         ++i) {

        double x =
            2.0 * sample_uniform(&state) - 1.0;

        double y =
            2.0 * sample_uniform(&state) - 1.0;

        double distance_squared =
            x * x + y * y;

        if (distance_squared <= 1.0) {
            ++local_inside;
        }
    }

    atomicAdd(
        inside,
        local_inside
    );
}

void launch_pi_estimation(
    int* d_inside,
    unsigned long long points_per_thread,
    unsigned long long num_threads,
    unsigned long long seed,
    cudaStream_t stream
) {
    constexpr int THREADS_PER_BLOCK = 256;

    int blocks =
        static_cast<int>(
            (num_threads + THREADS_PER_BLOCK - 1)
            / THREADS_PER_BLOCK
        );

    pi_estimation_kernel<<<
        blocks,
        THREADS_PER_BLOCK,
        0,
        stream
    >>>(
        d_inside,
        points_per_thread,
        seed
    );
}