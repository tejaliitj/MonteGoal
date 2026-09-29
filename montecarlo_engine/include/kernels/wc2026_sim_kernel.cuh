#pragma once

#include "../wc2026_types.cuh"

#include <cuda_runtime.h>

void launch_wc2026_simulation(
    const TeamGPU* d_teams,
    const double* d_lambda_matrix,
    const FixtureGPU* d_fixtures,
    int num_simulations,
    unsigned long long seed,
    int* d_champion_counts,
    int* d_round_counts,
    cudaStream_t stream = 0
);