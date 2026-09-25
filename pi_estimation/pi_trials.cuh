#pragma once

// Runs num_simulations independent Monte Carlo pi estimates in parallel,
// one thread per simulation. Writes each simulation's |estimate - pi_actual|
// into abs_error_out[idx].
void launch_pi_trials(float* d_abs_error, int num_simulations,
                       long long points_per_simulation,
                       unsigned long long seed,
                       unsigned long long counter_offset,
                       double pi_actual);
