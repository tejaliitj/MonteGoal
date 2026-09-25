#pragma once

#define NUM_TEAMS 48
#define NUM_GROUPS 12
#define TEAMS_PER_GROUP 4
#define NUM_GROUP_FIXTURES 72

// Device-resident constant data, uploaded once at startup by upload_tournament_data().
// Team IDs 0..47 are assigned by group order A..L, then draw order within each
// group -- this ordering is the single source of truth every other file indexes
// against (see tournament_data.cu for the actual WC2026 draw).
extern __constant__ float d_team_elo[NUM_TEAMS];
extern __constant__ int   d_team_is_host[NUM_TEAMS];   // 0/1
extern __constant__ int   d_team_group[NUM_TEAMS];     // 0..11 (A..L)
extern __constant__ int   d_team_alpha_rank[NUM_TEAMS]; // 0 = alphabetically first name
extern __constant__ int   d_fixture_home[NUM_GROUP_FIXTURES];
extern __constant__ int   d_fixture_away[NUM_GROUP_FIXTURES];
extern __constant__ int   d_group_teams[NUM_GROUPS][TEAMS_PER_GROUP];

// Host-side mirrors of the same data -- for printing results by name, and for
// validation harnesses that need a CPU-side reference computed independently.
extern const char* h_team_names[NUM_TEAMS];
extern const float h_team_elo[NUM_TEAMS];
extern const int   h_team_is_host[NUM_TEAMS];
extern int h_team_alpha_rank[NUM_TEAMS];
extern int h_fixture_home[NUM_GROUP_FIXTURES];
extern int h_fixture_away[NUM_GROUP_FIXTURES];

// Builds every array above and uploads the device copies via cudaMemcpyToSymbol.
// Call once before any kernel launch.
void upload_tournament_data();
