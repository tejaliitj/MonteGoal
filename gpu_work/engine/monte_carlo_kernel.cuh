#pragma once

#include "philox_rng.cuh"

// Generic one-trial-per-thread Monte Carlo kernel. This file has zero
// domain knowledge and must stay that way -- the only thing it needs from
// the caller is TrialFn: a callable (a struct with a __device__ operator())
// that takes an RngState* and returns one ResultT per trial. Whatever
// "a trial" means -- a simulated tournament, a stock-price path, anything
// else -- is entirely the caller's business. Which RNG is in use is entirely
// philox_rng.cuh's business -- this file only ever calls rng_init(), never
// names a specific generator.
//
// Every trial gets its own non-overlapping subsequence: seed stays fixed,
// counter_offset + thread_id gives each trial a distinct slot. Same
// discipline PhiloxEngine used throughout the rehearsal phase, generalized
// so the launch pattern itself is reusable, not just the RNG.
template <typename ResultT, typename TrialFn>
__global__ void monte_carlo_kernel(unsigned long long seed, unsigned long long counter_offset,
                                    int num_trials, ResultT* results_out, TrialFn trial_fn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_trials) return;

    RngState state;
    rng_init(seed, counter_offset + idx, 0, &state);

    results_out[idx] = trial_fn(&state);
}

// Host launcher. trial_fn is passed by value -- fine as long as domain
// functors stay small/stateless (as SimulateTournament is), since it becomes
// a kernel argument copied to device, not something requiring a separate
// upload step. threads_per_block defaults to 128 (the value used everywhere
// else so far); pass a different value to benchmark other block sizes.
template <typename ResultT, typename TrialFn>
void launch_monte_carlo(unsigned long long seed, unsigned long long counter_offset,
                         int num_trials, ResultT* d_results_out, TrialFn trial_fn,
                         int threads_per_block = 128) {
    int blocks = (num_trials + threads_per_block - 1) / threads_per_block;
    monte_carlo_kernel<ResultT, TrialFn><<<blocks, threads_per_block>>>(
        seed, counter_offset, num_trials, d_results_out, trial_fn);
}
