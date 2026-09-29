// #include "kernels/poisson_sim_kernel.cuh"
// #include "distributions/poisson_dist.cuh"

// __global__ void generate_poisson_kernel(
//     curandStatePhilox4_32_10_t* states,
//     unsigned int* results,
//     double lambda,
//     int n)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n)
//         results[idx] = sample_poisson(&states[idx], lambda);
// }



#include "kernels/poisson_sim_kernel.cuh"

#include "rng/rng_engine.cuh"
#include "distributions/poisson_dist.cuh"

__global__ void poisson_sim_kernel(
    double lambda,
    int* counts,
    int num_samples,
    unsigned long long seed
) {
    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_samples) {
        return;
    }

    curandStatePhilox4_32_10_t state;

    init_philox(
        &state,
        seed,
        idx
    );

    int value =
        sample_poisson(
            &state,
            lambda
        );

    atomicAdd(
        &counts[value],
        1
    );
}

void launch_poisson_simulation(
    double lambda,
    int* d_counts,
    int num_samples,
    unsigned long long seed,
    cudaStream_t stream
) {
    constexpr int THREADS_PER_BLOCK = 256;

    int blocks =
        (num_samples + THREADS_PER_BLOCK - 1)
        / THREADS_PER_BLOCK;

    poisson_sim_kernel<<<
        blocks,
        THREADS_PER_BLOCK,
        0,
        stream
    >>>(
        lambda,
        d_counts,
        num_samples,
        seed
    );
}