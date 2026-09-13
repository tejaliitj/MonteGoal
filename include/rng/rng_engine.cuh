// =============================================================================
// rng_engine.cuh
// -----------------------------------------------------------------------------
// The "Random Number Generation" box in your notes — this is the RNG piece
// that the Monte Carlo Engine calls into. It wraps cuRAND's Philox4x32-10
// generator, which is the recommended choice for GPU thread parallelism
// because it is COUNTER-BASED:
//   - No thread needs to depend on another thread's state.
//   - Each thread just needs a unique "subsequence" number, and it can jump
//     directly to its own independent random stream with O(1) cost.
//   - This is exactly why the note says: "Thread parallelism by using
//     cuRAND Philox".
// =============================================================================
#pragma once
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// -----------------------------------------------------------------------------
// Error-checking macros (wrap every CUDA / cuRAND call in these)
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__,\
                         cudaGetErrorString(err));                              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (0)

// Each thread gets seed = shared seed, subsequence = its own thread id,
// offset = 0. Different subsequence numbers guarantee non-overlapping
// streams -- this is what makes Philox safe for massive parallelism.
__global__ void setup_states_kernel(curandStatePhilox4_32_10_t* states,
                                    int n, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, /*subsequence=*/id, /*offset=*/0, &states[id]);
    }
}

// -----------------------------------------------------------------------------
// RNGEngine
// -----------------------------------------------------------------------------
// Manages one curandStatePhilox4_32_10_t per GPU thread. Call init() once at
// the start of your program; every kernel in include/kernels/ receives
// engine.states() and indexes into it with its own global thread id.
// -----------------------------------------------------------------------------
class RNGEngine {
public:
    RNGEngine() : d_states_(nullptr), num_threads_(0) {}

    ~RNGEngine() {
        if (d_states_) cudaFree(d_states_);
    }

    // Allocate + initialize one independent Philox stream per thread.
    void init(int num_threads, unsigned long long seed) {
        num_threads_ = num_threads;
        CUDA_CHECK(cudaMalloc(&d_states_,
                               num_threads_ * sizeof(curandStatePhilox4_32_10_t)));

        const int block_size = 256;
        const int grid_size  = (num_threads_ + block_size - 1) / block_size;
        setup_states_kernel<<<grid_size, block_size>>>(d_states_, num_threads_, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    curandStatePhilox4_32_10_t* states() const { return d_states_; }
    int num_threads() const { return num_threads_; }

private:
    curandStatePhilox4_32_10_t* d_states_;
    int num_threads_;
};
