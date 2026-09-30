// =============================================================================
// football/lambda_matrix.cu  --  runtime CSV loader for the trained lambda matrix.
//
// Data flow:   data/lambda_matrix.csv  --(host parser)-->  h_lambda_matrix
//              --(cudaMemcpyToSymbol)-->  d_lambda_matrix (device constant memory)
//
// The CSV is the SINGLE SOURCE OF TRUTH for model output. It can be replaced
// (new model, new run) without recompiling anything. The CSV was originally
// produced by the modelling phase; the values in the shipped file are the exact
// ones that used to be compiled into this file.
// =============================================================================

#include "lambda_matrix.cuh"
#include "cuda_check.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Host mirror + device copy. The device copy is defined HERE (one translation
// unit) and declared extern in the header; this is why the whole project must be
// compiled with nvcc -rdc=true (see report Section 5.4 and the Makefile).
float h_lambda_matrix[NUM_TEAMS][NUM_TEAMS];
__constant__ float d_lambda_matrix[NUM_TEAMS][NUM_TEAMS];

// ----------------------------------------------------------------------------
// Team-name matching. The modelling CSV spells two teams differently from this
// project's registry (tournament_data.cu); everything else matches verbatim.
// {name as it appears in the CSV, name in the registry}
// ----------------------------------------------------------------------------
static const char* NAME_ALIASES[][2] = {
    {"Bosnia and Herzegovina", "Bosnia-Herzegovina"},
    {"United States",          "USA"},
};

static std::string g_csv_path_override;

void set_lambda_matrix_path(const char* path) { g_csv_path_override = path ? path : ""; }

// Prints an error mentioning the file (and line, if known) and exits.
static void csv_fail(const std::string& path, int line, const std::string& msg) {
    if (line > 0) fprintf(stderr, "lambda matrix CSV error (%s, line %d): %s\n", path.c_str(), line, msg.c_str());
    else          fprintf(stderr, "lambda matrix CSV error (%s): %s\n", path.c_str(), msg.c_str());
    exit(1);
}

static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

// Splits one CSV line into cells. Supports simple "quoted, cells" so a team name
// containing a comma would still parse.
static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cells;
    std::string cur;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"')                    in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) { cells.push_back(trim(cur)); cur.clear(); }
        else                             cur.push_back(c);
    }
    cells.push_back(trim(cur));
    return cells;
}

// Registry team id (0..47) for a name from the CSV, or -1 if unrecognised.
static int team_id_from_name(const std::string& name) {
    std::string n = name;
    for (const auto& alias : NAME_ALIASES) {
        if (n == alias[0]) { n = alias[1]; break; }
    }
    for (int t = 0; t < NUM_TEAMS; ++t) {
        if (n == h_team_names[t]) return t;
    }
    return -1;
}

static bool file_readable(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

// Decides which CSV file to read (see the header for the search order).
static std::string resolve_csv_path() {
    if (!g_csv_path_override.empty()) return g_csv_path_override;
    const char* env = getenv("MONTEGOAL_LAMBDA_CSV");
    if (env && *env) return env;
    const char* candidates[] = {"data/lambda_matrix.csv", "../data/lambda_matrix.csv",
                                "../../data/lambda_matrix.csv"};
    for (const char* c : candidates) {
        if (file_readable(c)) return c;
    }
    return "data/lambda_matrix.csv";   // not found -- load_lambda_matrix_csv reports it
}

void load_lambda_matrix_csv(const char* path_c) {
    const std::string path = path_c;
    std::ifstream in(path);
    if (!in.good()) csv_fail(path, 0, "cannot open file (set MONTEGOAL_LAMBDA_CSV or run from the project root)");

    // seen[r][c] records that cell (r, c) was filled, so we can prove all 2,304 are present.
    static bool seen[NUM_TEAMS][NUM_TEAMS];
    memset(seen, 0, sizeof(seen));
    memset(h_lambda_matrix, 0, sizeof(h_lambda_matrix));

    std::string line;
    int line_no = 0;

    // ---- Header row: "team,<name>,<name>,..." -> column index -> team id ----
    int col_team[NUM_TEAMS];
    while (std::getline(in, line)) {
        ++line_no;
        if (line_no == 1 && line.size() >= 3 && (unsigned char)line[0] == 0xEF) line.erase(0, 3);  // strip UTF-8 BOM
        if (!trim(line).empty()) break;
    }
    std::vector<std::string> header = split_csv_line(line);
    if ((int)header.size() != NUM_TEAMS + 1)
        csv_fail(path, line_no, "header must have 1 label cell + 48 team names, found " + std::to_string(header.size()) + " cells");
    bool col_used[NUM_TEAMS] = {false};
    for (int c = 0; c < NUM_TEAMS; ++c) {
        int id = team_id_from_name(header[c + 1]);
        if (id < 0)          csv_fail(path, line_no, "unrecognised team name in header: '" + header[c + 1] + "'");
        if (col_used[id])    csv_fail(path, line_no, "duplicate team in header: '" + header[c + 1] + "'");
        col_used[id] = true;
        col_team[c] = id;
    }

    // ---- Data rows: "<row team>,<48 values>" ----
    bool row_used[NUM_TEAMS] = {false};
    int rows = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (trim(line).empty()) continue;
        std::vector<std::string> cells = split_csv_line(line);
        if ((int)cells.size() != NUM_TEAMS + 1)
            csv_fail(path, line_no, "expected 49 cells (name + 48 values), found " + std::to_string(cells.size()));
        int row_id = team_id_from_name(cells[0]);
        if (row_id < 0)        csv_fail(path, line_no, "unrecognised team name: '" + cells[0] + "'");
        if (row_used[row_id])  csv_fail(path, line_no, "duplicate row for team '" + cells[0] + "'");
        row_used[row_id] = true;
        ++rows;

        for (int c = 0; c < NUM_TEAMS; ++c) {
            const std::string& cell = cells[c + 1];
            if (cell.empty()) csv_fail(path, line_no, "missing value in column '" + header[c + 1] + "'");
            char* end = nullptr;
            // strtof (not strtod + cast) so the parsed float is bit-identical to the
            // compile-time literal it replaced -- keeps seeded results reproducible.
            float v = strtof(cell.c_str(), &end);
            if (end == cell.c_str() || *end != '\0' || !std::isfinite(v) || v < 0.0f)
                csv_fail(path, line_no, "bad value '" + cell + "' in column '" + header[c + 1] + "'");
            h_lambda_matrix[row_id][col_team[c]] = v;   // re-indexed into registry team-id order
            seen[row_id][col_team[c]] = true;
        }
    }

    if (rows != NUM_TEAMS) csv_fail(path, 0, "expected 48 data rows, found " + std::to_string(rows));
    for (int r = 0; r < NUM_TEAMS; ++r)
        for (int c = 0; c < NUM_TEAMS; ++c)
            if (!seen[r][c]) csv_fail(path, 0, "cell missing: " + std::string(h_team_names[r]) + " vs " + h_team_names[c]);
}

void upload_lambda_matrix() {
    // 1) Parse the CSV on the host (validates all 2,304 cells)...
    load_lambda_matrix_csv(resolve_csv_path().c_str());
    // 2) ...then copy the validated matrix into device constant memory (once).
    CUDA_CHECK(cudaMemcpyToSymbol(d_lambda_matrix, h_lambda_matrix, sizeof(h_lambda_matrix)));
}
