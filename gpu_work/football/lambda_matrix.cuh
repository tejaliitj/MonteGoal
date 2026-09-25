#pragma once

#include "tournament_data.cuh"

// d_lambda_matrix[team_a][team_b] = expected goals for team_a against team_b,
// from the trained model (see lambda_matrix.cu for provenance). Home advantage
// is already baked in by the model -- lambda_model.cuh must not apply an
// additional host multiplier on top of these values.
extern __constant__ float d_lambda_matrix[NUM_TEAMS][NUM_TEAMS];

// Host-side mirror of the same data -- for validation harnesses that need a
// reference value computed independently on the CPU side.
extern const float h_lambda_matrix[NUM_TEAMS][NUM_TEAMS];

// Uploads the matrix to device constant memory. Call once at startup,
// alongside upload_tournament_data() and the others.
void upload_lambda_matrix();
