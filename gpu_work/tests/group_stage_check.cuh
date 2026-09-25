#pragma once

// Validation-only harness, not part of the real tournament kernel's API. Runs
// num_trials independent group-stage simulations (one thread per trial) and
// writes each trial's total goals scored, summed across all 72 matches, to
// total_goals_out[idx].
void launch_group_stage_check(int* d_total_goals, int num_trials,
                               unsigned long long seed,
                               unsigned long long counter_offset);
