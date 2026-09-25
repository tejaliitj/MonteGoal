#pragma once

#define NUM_KNOCKOUT_MATCHES 32
#define MATCH_OFFSET 73  // original match numbers are 73..104; match_id = match_no - 73
#define MATCH_ID_THIRD_PLACE 30
#define MATCH_ID_FINAL 31

// Team-spec types, matching bracket_engine.py's _resolve_team spec["type"] values.
#define SPEC_GROUP_WINNER       0
#define SPEC_GROUP_RUNNERUP     1
#define SPEC_THIRDPLACE_LOOKUP  2
#define SPEC_WINNER_OF_MATCH    3
#define SPEC_LOSER_OF_MATCH     4

extern __constant__ int d_bracket_team_a_type[NUM_KNOCKOUT_MATCHES];
extern __constant__ int d_bracket_team_a_arg[NUM_KNOCKOUT_MATCHES];
extern __constant__ int d_bracket_team_b_type[NUM_KNOCKOUT_MATCHES];
extern __constant__ int d_bracket_team_b_arg[NUM_KNOCKOUT_MATCHES];

extern const int h_bracket_team_a_type[NUM_KNOCKOUT_MATCHES];
extern const int h_bracket_team_a_arg[NUM_KNOCKOUT_MATCHES];
extern const int h_bracket_team_b_type[NUM_KNOCKOUT_MATCHES];
extern const int h_bracket_team_b_arg[NUM_KNOCKOUT_MATCHES];

// Uploads the table (baked in from wc2026_bracket_structure.json, see
// bracket_data.cu) to device constant memory. Call once at startup, alongside
// upload_tournament_data() and upload_thirdplace_table().
void upload_bracket_data();
