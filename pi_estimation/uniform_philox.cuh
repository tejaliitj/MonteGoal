#pragma once

// Fills d_out with n independent uniform floats in (0, 1], using the
// Philox4x32-10 counter-based generator. counter_offset shifts every
// thread's subsequence so a later launch never reuses an earlier
// launch's random stream.
void launch_uniform_philox(float* d_out, int n,
                            unsigned long long seed,
                            unsigned long long counter_offset);
