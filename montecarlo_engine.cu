#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "utils.cuh"
#include "philox_rng.cuh"
#include "uniform_distribution.cuh"

constexpr int DARTS_PER_THREAD = 50000;

__global__ void monteCarloPiKernel(curandStatePhilox4_32_10_t* rng_states, ThreadResult* results, double clock_rate_khz) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t local_state = rng_states[thread_id];
    unsigned long long start_cycles = clock64();
    
    int inside_circle = 0;
    for (int i = 0; i < DARTS_PER_THREAD; i++) {
        double x, y;
        generateUniformXY(&local_state, x, y);
        if (x * x + y * y <= 1.0) {
            inside_circle++;
        }
    }
    
    unsigned long long end_cycles = clock64();
    unsigned long long elapsed = end_cycles - start_cycles;
    rng_states[thread_id] = local_state;
    
    double estimated_pi = 4.0 * inside_circle / (double)DARTS_PER_THREAD;
    double percentage_error = fabs(estimated_pi - M_PI_CONST) / M_PI_CONST * 100.0;
    
    results[thread_id].thread_id = thread_id;
    results[thread_id].block_id = blockIdx.x;
    results[thread_id].thread_index = threadIdx.x;
    results[thread_id].darts = DARTS_PER_THREAD;
    results[thread_id].inside_circle = inside_circle;
    results[thread_id].estimated_pi = estimated_pi;
    results[thread_id].actual_pi = M_PI_CONST;
    results[thread_id].percentage_error = percentage_error;
    results[thread_id].elapsed_cycles = elapsed;
    results[thread_id].thread_time = (double)elapsed / clock_rate_khz;
}

int main(int argc, char** argv) {
    int blocks = 128, threads_per_block = 256;
    std::string output_file = "data.csv";
    
    if (argc >= 3) {
        blocks = std::stoi(argv[1]);
        threads_per_block = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        output_file = argv[3];
    }
    
    int total_threads = blocks * threads_per_block;
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    double clock_rate_khz = prop.clockRate;
    
    curandStatePhilox4_32_10_t* d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, total_threads * sizeof(curandStatePhilox4_32_10_t)));
    
    ThreadResult* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, total_threads * sizeof(ThreadResult)));
    
    unsigned long long seed = 2024ULL;
    initializeRNG<<<blocks, threads_per_block>>>(d_rng_states, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    monteCarloPiKernel<<<blocks, threads_per_block>>>(d_rng_states, d_results, clock_rate_khz);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float overall_kernel_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&overall_kernel_time_ms, start, stop));
    
    std::vector<ThreadResult> h_results(total_threads);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, total_threads * sizeof(ThreadResult), cudaMemcpyDeviceToHost));
    
    generateCSV(h_results, overall_kernel_time_ms, output_file);
    printSummary(h_results, overall_kernel_time_ms, blocks, threads_per_block, total_threads, output_file);
    
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}