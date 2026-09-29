#pragma once

#include "wc2026_types.cuh"

#include <string>
#include <vector>

struct HostTeam {
    TeamGPU gpu;

    std::string group_name;
    std::string team_name;
};

struct WC2026Data {

    std::vector<HostTeam> teams;

    /*
     * Flattened 48 x 48 lambda matrix.
     *
     * lambda_matrix[i * 48 + j]
     *
     * = expected goals for team i against team j.
     */
    std::vector<double> lambda_matrix;

    std::vector<FixtureGPU> fixtures;
};

WC2026Data load_wc2026_data(
    const std::string& lambda_matrix_path
);

void write_wc2026_results(
    const WC2026Data& data,
    const std::vector<int>& champion_counts,
    const std::vector<int>& round_counts,
    int num_simulations
);

