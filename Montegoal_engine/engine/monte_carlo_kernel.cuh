#pragma once

// =============================================================================
// engine/monte_carlo_kernel.cuh  --  generic Monte Carlo launch patterns.
//
// In addition to the original one-result and one-histogram launchers, this file
// provides a four-stage histogram launcher. The four stages are intentionally
// generic: the engine does not know that they mean QF/SF/Final/Champion; the
// trial functor simply returns four team-id bitmasks/counters through a result
// object. MonteGoal uses those four outputs for stage probabilities.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include "rng.cuh"
#include "cuda_check.cuh"

#define DEFAULT_THREADS_PER_BLOCK 128

// -----------------------------------------------------------------------------
// Original style: one result per trial.
// -----------------------------------------------------------------------------
template <typename ResultT, typename TrialFn>
__global__ void monte_carlo_kernel(unsigned long long seed, unsigned long long counter_offset,
                                    unsigned long long num_trials, ResultT* results_out,
                                    TrialFn trial_fn) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_trials) return;
    RngState state;
    rng_init(seed, counter_offset + idx, 0, &state);
    results_out[idx] = trial_fn(&state);
}

template <typename ResultT, typename TrialFn>
void launch_monte_carlo(unsigned long long seed, unsigned long long counter_offset,
                         unsigned long long num_trials, ResultT* d_results_out,
                         TrialFn trial_fn, int threads_per_block = DEFAULT_THREADS_PER_BLOCK) {
    unsigned long long blocks = (num_trials + threads_per_block - 1) / threads_per_block;
    if (blocks > 2147483647ULL) {
        fprintf(stderr, "launch_monte_carlo: %llu blocks exceeds grid limit; use a larger block size\n", blocks);
        exit(1);
    }
    monte_carlo_kernel<ResultT, TrialFn><<<(unsigned int)blocks, threads_per_block>>>(
        seed, counter_offset, num_trials, d_results_out, trial_fn);
}

// -----------------------------------------------------------------------------
// Original style: one integer result per trial counted into one histogram.
// -----------------------------------------------------------------------------
template <int NUM_BINS, typename TrialFn>
__global__ void monte_carlo_histogram_kernel(unsigned long long seed,
                                              unsigned long long counter_offset,
                                              unsigned long long num_trials,
                                              unsigned long long* d_bins,
                                              TrialFn trial_fn) {
    __shared__ unsigned int s_bins[NUM_BINS];
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) s_bins[b] = 0u;
    __syncthreads();

    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_trials) {
        RngState state;
        rng_init(seed, counter_offset + idx, 0, &state);
        int bin = trial_fn(&state);
        atomicAdd(&s_bins[bin], 1u);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (s_bins[b] != 0u) atomicAdd(&d_bins[b], (unsigned long long)s_bins[b]);
    }
}

template <int NUM_BINS, typename TrialFn>
void launch_monte_carlo_histogram(unsigned long long seed, unsigned long long counter_offset,
                                   unsigned long long num_trials, unsigned long long* d_bins,
                                   TrialFn trial_fn,
                                   int threads_per_block = DEFAULT_THREADS_PER_BLOCK) {
    unsigned long long blocks = (num_trials + threads_per_block - 1) / threads_per_block;
    if (blocks > 2147483647ULL) {
        fprintf(stderr, "launch_monte_carlo_histogram: %llu blocks exceeds grid limit; use a larger block size\n", blocks);
        exit(1);
    }
    monte_carlo_histogram_kernel<NUM_BINS, TrialFn><<<(unsigned int)blocks, threads_per_block>>>(
        seed, counter_offset, num_trials, d_bins, trial_fn);
}

// -----------------------------------------------------------------------------
// Four-stage histogram.
// ResultT must contain four unsigned 64-bit bitmasks named stage0..stage3.
// Bit t is 1 when team t reached that stage. The engine remains agnostic to
// what the four stages mean.
// -----------------------------------------------------------------------------
template <int NUM_TEAMS, typename ResultT, typename TrialFn>
__global__ void monte_carlo_four_stage_histogram_kernel(
    unsigned long long seed, unsigned long long counter_offset,
    unsigned long long num_trials,
    unsigned long long* d_stage0,
    unsigned long long* d_stage1,
    unsigned long long* d_stage2,
    unsigned long long* d_stage3,
    TrialFn trial_fn) {

    __shared__ unsigned int s_stage0[NUM_TEAMS];
    __shared__ unsigned int s_stage1[NUM_TEAMS];
    __shared__ unsigned int s_stage2[NUM_TEAMS];
    __shared__ unsigned int s_stage3[NUM_TEAMS];

    for (int t = threadIdx.x; t < NUM_TEAMS; t += blockDim.x) {
        s_stage0[t] = 0u;
        s_stage1[t] = 0u;
        s_stage2[t] = 0u;
        s_stage3[t] = 0u;
    }
    __syncthreads();

    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_trials) {
        RngState state;
        rng_init(seed, counter_offset + idx, 0, &state);
        ResultT result = trial_fn(&state);

        for (int t = 0; t < NUM_TEAMS; ++t) {
            if ((result.stage0 >> t) & 1ULL) atomicAdd(&s_stage0[t], 1u);
            if ((result.stage1 >> t) & 1ULL) atomicAdd(&s_stage1[t], 1u);
            if ((result.stage2 >> t) & 1ULL) atomicAdd(&s_stage2[t], 1u);
            if ((result.stage3 >> t) & 1ULL) atomicAdd(&s_stage3[t], 1u);
        }
    }
    __syncthreads();

    for (int t = threadIdx.x; t < NUM_TEAMS; t += blockDim.x) {
        if (s_stage0[t]) atomicAdd(&d_stage0[t], (unsigned long long)s_stage0[t]);
        if (s_stage1[t]) atomicAdd(&d_stage1[t], (unsigned long long)s_stage1[t]);
        if (s_stage2[t]) atomicAdd(&d_stage2[t], (unsigned long long)s_stage2[t]);
        if (s_stage3[t]) atomicAdd(&d_stage3[t], (unsigned long long)s_stage3[t]);
    }
}

template <int NUM_TEAMS, typename ResultT, typename TrialFn>
void launch_monte_carlo_four_stage_histogram(
    unsigned long long seed, unsigned long long counter_offset,
    unsigned long long num_trials,
    unsigned long long* d_stage0,
    unsigned long long* d_stage1,
    unsigned long long* d_stage2,
    unsigned long long* d_stage3,
    TrialFn trial_fn,
    int threads_per_block = DEFAULT_THREADS_PER_BLOCK) {
    unsigned long long blocks = (num_trials + threads_per_block - 1) / threads_per_block;
    if (blocks > 2147483647ULL) {
        fprintf(stderr, "launch_monte_carlo_four_stage_histogram: %llu blocks exceeds grid limit; use a larger block size\n", blocks);
        exit(1);
    }
    monte_carlo_four_stage_histogram_kernel<NUM_TEAMS, ResultT, TrialFn>
        <<< (unsigned int)blocks, threads_per_block >>>(
            seed, counter_offset, num_trials,
            d_stage0, d_stage1, d_stage2, d_stage3, trial_fn);
}
