#pragma once

// =============================================================================
// football/lambda_matrix.cuh  --  the trained model's 48x48 lambda matrix.
//
// The matrix is loaded AT RUNTIME from a CSV (data/lambda_matrix.csv) rather
// than compiled in, so model output stays editable/shareable independent of the
// CUDA build (mentor requirement, report Section 6.3).
// =============================================================================

#include "tournament_data.cuh"

// d_lambda_matrix[team_a][team_b] = expected goals for team_a against team_b.
// Home advantage is already baked in by the model that produced the values --
// lambda_model.cuh must not apply an additional host multiplier on top.
extern __constant__ float d_lambda_matrix[NUM_TEAMS][NUM_TEAMS];

// Host-side mirror of the same data -- filled by load_lambda_matrix_csv(). Used
// by validation harnesses that need a CPU-side reference computed independently.
extern float h_lambda_matrix[NUM_TEAMS][NUM_TEAMS];

// Overrides where the CSV is read from. Optional: with no override, the loader
// tries the MONTEGOAL_LAMBDA_CSV environment variable, then data/lambda_matrix.csv
// relative to the current directory and its parents (so it works whether you run
// from the project root, build/, tests/ or benchmarks/).
void set_lambda_matrix_path(const char* path);

// Parses the CSV into h_lambda_matrix. Layout: first row = "team,<48 team names>",
// then 48 rows of "<team name>,<48 values>", where cell (row r, column c) is the
// expected goals of row-team against column-team. Team names are matched to the
// registry BY NAME (row/column order in the file does not matter). Validates that
// every one of the 2,304 cells is present and finite and every name is
// recognised; prints a clear error and exits otherwise.
void load_lambda_matrix_csv(const char* path);

// Loads the CSV (see above) and uploads the result to device constant memory.
// Call once at startup, alongside upload_tournament_data() and the others.
void upload_lambda_matrix();
