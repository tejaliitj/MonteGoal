#pragma once

// Validation-only harness, not part of the real tournament kernel's API.
// Runs num_trials independent checks in parallel (one thread per trial).
// Each thread draws draws_per_trial samples from Poisson(lambda) using
// its own Philox subsequence, and writes |empirical_mean - lambda| to
// abs_error_out[idx].
void launch_poisson_check(float* d_abs_error, int num_trials,
                           long long draws_per_trial, double lambda,
                           unsigned long long seed,
                           unsigned long long counter_offset);
