#include "lambda_matrix.cuh"
#include "cuda_check.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

__constant__ float d_lambda_matrix[NUM_TEAMS][NUM_TEAMS];
float h_lambda_matrix[NUM_TEAMS][NUM_TEAMS];

namespace {

// Explicit aliases for CSV names that don't match tournament_data.cu's team
// list verbatim -- no fuzzy matching, same discipline the original Python
// generator used. Add an entry here if a future CSV uses another spelling.
const std::map<std::string, std::string>& alias_table() {
    static const std::map<std::string, std::string> aliases = {
        {"Bosnia and Herzegovina", "Bosnia-Herzegovina"},
        {"United States", "USA"},
    };
    return aliases;
}

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string canonical(const std::string& raw) {
    std::string name = trim(raw);
    auto it = alias_table().find(name);
    return (it != alias_table().end()) ? it->second : name;
}

// Fields here never contain commas (team names only), so a plain split is
// safe -- this CSV does not need quoted-field handling the way the
// third-place table's multi-value column did.
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    return fields;
}

int find_team_id(const std::string& canon_name) {
    for (int i = 0; i < NUM_TEAMS; ++i) {
        if (canon_name == h_team_names[i]) return i;
    }
    return -1;
}

}  // namespace

void upload_lambda_matrix(const char* csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open lambda matrix CSV: %s\n", csv_path);
        exit(1);
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        fprintf(stderr, "Lambda matrix CSV is empty: %s\n", csv_path);
        exit(1);
    }
    std::vector<std::string> header = split_csv_line(header_line);
    if ((int)header.size() != NUM_TEAMS + 1) {
        fprintf(stderr, "Expected %d columns in header, got %d (%s)\n",
                NUM_TEAMS, (int)header.size() - 1, csv_path);
        exit(1);
    }

    int col_ids[NUM_TEAMS];
    bool col_seen[NUM_TEAMS] = {false};
    for (int c = 0; c < NUM_TEAMS; ++c) {
        std::string canon = canonical(header[c + 1]);
        int id = find_team_id(canon);
        if (id < 0) {
            fprintf(stderr, "Column name '%s' (canonical '%s') not found in team list\n",
                    header[c + 1].c_str(), canon.c_str());
            exit(1);
        }
        col_ids[c] = id;
        col_seen[id] = true;
    }
    for (int t = 0; t < NUM_TEAMS; ++t) {
        if (!col_seen[t]) {
            fprintf(stderr, "Column names don't cover team ID %d (%s)\n", t, h_team_names[t]);
            exit(1);
        }
    }

    bool row_seen[NUM_TEAMS] = {false};
    std::string line;
    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;
        std::vector<std::string> row = split_csv_line(line);
        std::string canon = canonical(row[0]);
        int row_id = find_team_id(canon);
        if (row_id < 0) {
            fprintf(stderr, "Row name '%s' (canonical '%s') not found in team list\n",
                    row[0].c_str(), canon.c_str());
            exit(1);
        }
        if ((int)row.size() != NUM_TEAMS + 1) {
            fprintf(stderr, "Row '%s' has %d values, expected %d\n",
                    row[0].c_str(), (int)row.size() - 1, NUM_TEAMS);
            exit(1);
        }
        row_seen[row_id] = true;
        for (int c = 0; c < NUM_TEAMS; ++c) {
            h_lambda_matrix[row_id][col_ids[c]] = std::stof(row[c + 1]);
        }
    }

    for (int t = 0; t < NUM_TEAMS; ++t) {
        if (!row_seen[t]) {
            fprintf(stderr, "Missing row for team ID %d (%s)\n", t, h_team_names[t]);
            exit(1);
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_lambda_matrix, h_lambda_matrix, sizeof(h_lambda_matrix)));
}
