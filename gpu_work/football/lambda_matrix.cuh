#pragma once

#include "tournament_data.cuh"

// d_lambda_matrix[team_a][team_b] = expected goals for team_a against team_b.
// Home advantage is already baked in by the model that produced these values
// -- lambda_model.cuh must not apply an additional host multiplier on top.
extern __constant__ float d_lambda_matrix[NUM_TEAMS][NUM_TEAMS];

// Host-side mirror, populated by upload_lambda_matrix() -- exposed for
// validation harnesses that need an independent reference value.
extern float h_lambda_matrix[NUM_TEAMS][NUM_TEAMS];

// Loads the lambda matrix from a CSV file at runtime (see lambda_matrix.cu)
// and uploads it to device constant memory. Call once at startup. csv_path
// points at lambda_matrix.csv -- header row and first column are team names,
// matched against tournament_data.cu's team list via a small explicit alias
// table for known naming differences (e.g. "United States" -> "USA"). No
// fuzzy matching -- an unrecognized name is a hard error, not a guess.
void upload_lambda_matrix(const char* csv_path);
