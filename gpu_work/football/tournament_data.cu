#include "tournament_data.cuh"
#include "cuda_check.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

__constant__ float d_team_elo[NUM_TEAMS];
__constant__ int   d_team_is_host[NUM_TEAMS];
__constant__ int   d_team_group[NUM_TEAMS];
__constant__ int   d_team_alpha_rank[NUM_TEAMS];
__constant__ int   d_fixture_home[NUM_GROUP_FIXTURES];
__constant__ int   d_fixture_away[NUM_GROUP_FIXTURES];
__constant__ int   d_group_teams[NUM_GROUPS][TEAMS_PER_GROUP];

// Real WC2026 draw (FIFA final draw, Dec 5 2025), ported from teams.py's
// WC2026_GROUPS. Team IDs 0-47 assigned by group order A..L, draw order within
// each group.
const char* h_team_names[NUM_TEAMS] = {
    "Mexico", "South Korea", "South Africa", "Czechia",               // A: 0-3
    "Canada", "Switzerland", "Qatar", "Bosnia-Herzegovina",           // B: 4-7
    "Brazil", "Morocco", "Scotland", "Haiti",                         // C: 8-11
    "USA", "Paraguay", "Australia", "Turkiye",                        // D: 12-15
    "Germany", "Ecuador", "Ivory Coast", "Curacao",                   // E: 16-19
    "Netherlands", "Japan", "Tunisia", "Sweden",                      // F: 20-23
    "Belgium", "Iran", "Egypt", "New Zealand",                        // G: 24-27
    "Spain", "Uruguay", "Saudi Arabia", "Cape Verde",                 // H: 28-31
    "France", "Senegal", "Norway", "Iraq",                            // I: 32-35
    "Argentina", "Austria", "Algeria", "Jordan",                      // J: 36-39
    "Portugal", "Colombia", "Uzbekistan", "DR Congo",                 // K: 40-43
    "England", "Croatia", "Panama", "Ghana",                          // L: 44-47
};

// PLACEHOLDER Elo -- same values and same caveat as teams.py: order-of-magnitude
// plausible, not the real elo10-25.csv. Swap this array when the real model lands;
// nothing downstream needs to change shape.
const float h_team_elo[NUM_TEAMS] = {
    1780, 1740, 1560, 1700,   // A
    1740, 1790, 1630, 1700,   // B
    2050, 1830, 1720, 1500,   // C
    1790, 1690, 1690, 1780,   // D
    1930, 1750, 1740, 1500,   // E
    1930, 1800, 1650, 1720,   // F
    1900, 1720, 1690, 1500,   // G
    2020, 1870, 1620, 1560,   // H
    2000, 1780, 1780, 1580,   // I
    2050, 1800, 1720, 1560,   // J
    1970, 1850, 1650, 1580,   // K
    1980, 1870, 1620, 1650,   // L
};

// Host nations: Mexico (0), Canada (4), USA (12).
const int h_team_is_host[NUM_TEAMS] = {
    1, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
};

int h_fixture_home[NUM_GROUP_FIXTURES];
int h_fixture_away[NUM_GROUP_FIXTURES];
int h_team_alpha_rank[NUM_TEAMS];

void upload_tournament_data() {
    // Team group assignment and group->team-ID table both fall straight out of
    // team IDs already being laid out in group order.
    int h_team_group[NUM_TEAMS];
    int h_group_teams[NUM_GROUPS][TEAMS_PER_GROUP];
    for (int g = 0; g < NUM_GROUPS; ++g) {
        for (int k = 0; k < TEAMS_PER_GROUP; ++k) {
            int team_id = g * TEAMS_PER_GROUP + k;
            h_team_group[team_id] = g;
            h_group_teams[g][k] = team_id;
        }
    }

    // 72 group fixtures: within each group, every (i, j) pair with i < j,
    // home = i, away = j -- exactly teams.py's group_round_robin().
    int f = 0;
    for (int g = 0; g < NUM_GROUPS; ++g) {
        for (int i = 0; i < TEAMS_PER_GROUP; ++i) {
            for (int j = i + 1; j < TEAMS_PER_GROUP; ++j) {
                h_fixture_home[f] = h_group_teams[g][i];
                h_fixture_away[f] = h_group_teams[g][j];
                ++f;
            }
        }
    }
    if (f != NUM_GROUP_FIXTURES) {
        fprintf(stderr, "fixture count mismatch: built %d, expected %d\n",
                f, NUM_GROUP_FIXTURES);
        exit(1);
    }

    // Alphabetical rank by team name -- the FIFA WC2026 cascade's final,
    // purely-for-determinism tiebreak (see standings.cuh). Two of the real
    // groups have an exact Elo collision in this placeholder data (Group D:
    // Paraguay/Australia both 1690; Group I: Senegal/Norway both 1780), so
    // this genuinely gets reached, not just a theoretical fallback.
    int team_order[NUM_TEAMS];
    for (int t = 0; t < NUM_TEAMS; ++t) team_order[t] = t;
    std::sort(team_order, team_order + NUM_TEAMS, [](int a, int b) {
        return strcmp(h_team_names[a], h_team_names[b]) < 0;
    });
    for (int rank = 0; rank < NUM_TEAMS; ++rank) {
        h_team_alpha_rank[team_order[rank]] = rank;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_team_elo, h_team_elo, sizeof(h_team_elo)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_team_is_host, h_team_is_host, sizeof(h_team_is_host)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_team_group, h_team_group, sizeof(h_team_group)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_team_alpha_rank, h_team_alpha_rank, sizeof(h_team_alpha_rank)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fixture_home, h_fixture_home, sizeof(h_fixture_home)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fixture_away, h_fixture_away, sizeof(h_fixture_away)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_group_teams, h_group_teams, sizeof(h_group_teams)));
}
