#include "wc2026_io.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <iomanip>

namespace {

std::vector<std::string> split_csv_line(
    const std::string& line
) {
    std::vector<std::string> result;

    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
        result.push_back(cell);
    }

    return result;
}

std::string trim(
    const std::string& value
) {
    std::string s = value;

    while (!s.empty() &&
           (s.back() == '\r' ||
            s.back() == '\n' ||
            s.back() == ' ')) {
        s.pop_back();
    }

    size_t start = 0;

    while (start < s.size() &&
           s[start] == ' ') {
        ++start;
    }

    return s.substr(start);
}

}


/* ============================================================
   Load lambda matrix
   ============================================================ */

WC2026Data load_wc2026_data(
    const std::string& lambda_matrix_path
) {
    WC2026Data data;

    std::ifstream file(
        lambda_matrix_path
    );

    if (!file.is_open()) {

        throw std::runtime_error(
            "Could not open lambda matrix: "
            + lambda_matrix_path
        );
    }

    std::string line;

    /*
     * ----------------------------------------------------------
     * Header
     * ----------------------------------------------------------
     */

    if (!std::getline(file, line)) {

        throw std::runtime_error(
            "Lambda matrix is empty."
        );
    }

    std::vector<std::string> header =
        split_csv_line(line);

    if (header.size() != WC_NUM_TEAMS + 1) {

        throw std::runtime_error(
            "Lambda matrix must contain "
            "49 columns: one row-name column + 48 teams."
        );
    }

    /*
     * Column 0 = Unnamed: 0
     *
     * Columns 1..48 = team names
     */

    std::vector<std::string> team_names;

    for (int i = 1;
         i <= WC_NUM_TEAMS;
         ++i) {

        team_names.push_back(
            trim(header[i])
        );
    }

    /*
     * ----------------------------------------------------------
     * Team name -> matrix index
     * ----------------------------------------------------------
     */

    std::unordered_map<
        std::string,
        int
    > team_to_index;

    for (int i = 0;
         i < WC_NUM_TEAMS;
         ++i) {

        team_to_index[
            team_names[i]
        ] = i;
    }

    /*
     * ----------------------------------------------------------
     * Read matrix
     * ----------------------------------------------------------
     */

    data.lambda_matrix.resize(
        WC_NUM_TEAMS * WC_NUM_TEAMS
    );

    for (int row = 0;
         row < WC_NUM_TEAMS;
         ++row) {

        if (!std::getline(file, line)) {

            throw std::runtime_error(
                "Lambda matrix has fewer than 48 rows."
            );
        }

        std::vector<std::string> cells =
            split_csv_line(line);

        if (cells.size() != WC_NUM_TEAMS + 1) {

            throw std::runtime_error(
                "Invalid number of columns in lambda matrix row "
                + std::to_string(row)
            );
        }

        std::string row_team =
            trim(cells[0]);

        auto it =
            team_to_index.find(row_team);

        if (it == team_to_index.end()) {

            throw std::runtime_error(
                "Unknown team in lambda matrix row: "
                + row_team
            );
        }

        int row_index = it->second;

        for (int col = 0;
             col < WC_NUM_TEAMS;
             ++col) {

            double lambda =
                std::stod(
                    trim(cells[col + 1])
                );

            data.lambda_matrix[
                row_index * WC_NUM_TEAMS
                + col
            ] = lambda;
        }
    }

    /*
     * ----------------------------------------------------------
     * Create team objects
     * ----------------------------------------------------------
     *
     * IMPORTANT:
     *
     * We currently do not have group information inside
     * lambda_matrix.csv.
     *
     * Therefore this section must be populated according
     * to your actual World Cup 2026 team/group file.
     *
     * For the moment, teams are assigned sequentially:
     *
     * 0-3   -> Group 0
     * 4-7   -> Group 1
     * ...
     *
     * Replace this once your official group allocation file
     * is connected.
     */

    data.teams.resize(
        WC_NUM_TEAMS
    );

    for (int i = 0;
         i < WC_NUM_TEAMS;
         ++i) {

        data.teams[i].team_name =
            team_names[i];

        data.teams[i].gpu.id = i;

        data.teams[i].gpu.group_id =
            i / WC_GROUP_SIZE;

        data.teams[i].group_name =
            "Group_" +
            std::to_string(
                i / WC_GROUP_SIZE
            );
    }

    /*
     * ----------------------------------------------------------
     * Generate group fixtures
     * ----------------------------------------------------------
     */

    data.fixtures.reserve(
        WC_NUM_FIXTURES
    );

    for (int group = 0;
         group < WC_NUM_GROUPS;
         ++group) {

        int start =
            group * WC_GROUP_SIZE;

        for (int i = 0;
             i < WC_GROUP_SIZE;
             ++i) {

            for (int j = i + 1;
                 j < WC_GROUP_SIZE;
                 ++j) {

                FixtureGPU fixture;

                fixture.home_idx =
                    start + i;

                fixture.away_idx =
                    start + j;

                data.fixtures.push_back(
                    fixture
                );
            }
        }
    }

    if (
        static_cast<int>(
            data.fixtures.size()
        )
        != WC_NUM_FIXTURES
    ) {
        throw std::runtime_error(
            "Incorrect number of fixtures generated."
        );
    }

    return data;
}


/* ============================================================
   Write results
   ============================================================ */

void write_wc2026_results(
    const WC2026Data& data,
    const std::vector<int>& champion_counts,
    const std::vector<int>& round_counts,
    int num_simulations
) {
    /*
     * ----------------------------------------------------------
     * Champion probabilities
     * ----------------------------------------------------------
     */

    std::ofstream champion_file(
        "champion_probabilities_cuda.csv"
    );

    if (!champion_file.is_open()) {

        throw std::runtime_error(
            "Could not create champion probability CSV."
        );
    }

    champion_file
        << "team,champion_count,champion_probability\n";

    champion_file
        << std::fixed
        << std::setprecision(8);

    for (int i = 0;
         i < WC_NUM_TEAMS;
         ++i) {

        double probability =
            static_cast<double>(
                champion_counts[i]
            )
            / static_cast<double>(
                num_simulations
            );

        champion_file
            << data.teams[i].team_name
            << ","
            << champion_counts[i]
            << ","
            << probability
            << "\n";
    }

    champion_file.close();


    /*
     * ----------------------------------------------------------
     * Round reach probabilities
     * ----------------------------------------------------------
     */

    std::ofstream round_file(
        "round_reach_probabilities_cuda.csv"
    );

    if (!round_file.is_open()) {

        throw std::runtime_error(
            "Could not create round probability CSV."
        );
    }

    round_file
        << "team,"
        << "r32_probability,"
        << "r16_probability,"
        << "qf_probability,"
        << "sf_probability,"
        << "final_probability,"
        << "champion_probability\n";

    round_file
        << std::fixed
        << std::setprecision(8);

    for (int team = 0;
         team < WC_NUM_TEAMS;
         ++team) {

        auto count_stage =
            [&](int stage) {

                return static_cast<double>(
                    round_counts[
                        team * WC_NUM_STAGES
                        + stage
                    ]
                )
                /
                static_cast<double>(
                    num_simulations
                );
            };

        round_file
            << data.teams[team].team_name
            << ","
            << count_stage(STAGE_R32)
            << ","
            << count_stage(STAGE_R16)
            << ","
            << count_stage(STAGE_QF)
            << ","
            << count_stage(STAGE_SF)
            << ","
            << count_stage(STAGE_FINAL)
            << ","
            << count_stage(STAGE_CHAMPION)
            << "\n";
    }

    round_file.close();
}