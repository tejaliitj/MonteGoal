#pragma once

#include "tournament_data.cuh"

// C(TEAMS_PER_GROUP, 2) -- number of fixtures within one group. Fixed at 6 because
// TEAMS_PER_GROUP is fixed at 4; matches tournament_data.cu's fixture-building loop.
#define FIXTURES_PER_GROUP 6

// True if team "a" ranks strictly ahead of team "b" on points/GD/GS alone.
__device__ __forceinline__ bool primary_better(int pts_a, int gd_a, int gf_a,
                                                 int pts_b, int gd_b, int gf_b) {
    if (pts_a != pts_b) return pts_a > pts_b;
    if (gd_a != gd_b) return gd_a > gd_b;
    return gf_a > gf_b;
}

// True if team "a" ranks strictly ahead of team "b" on the full tiebreak key:
// head-to-head points/GD/GS, then Elo (FIFA-ranking stand-in), then alphabetical
// name as the final, purely-for-determinism tiebreak. "Fair play" is a deliberate
// no-op here, same as group_stage.py -- not modeled, falls straight through.
__device__ __forceinline__ bool tie_better(int pts_a, int gd_a, int gf_a, float elo_a, int alpha_a,
                                            int pts_b, int gd_b, int gf_b, float elo_b, int alpha_b) {
    if (pts_a != pts_b) return pts_a > pts_b;
    if (gd_a != gd_b) return gd_a > gd_b;
    if (gf_a != gf_b) return gf_a > gf_b;
    if (elo_a != elo_b) return elo_a > elo_b;
    return alpha_a < alpha_b;  // alphabetically-earlier name wins
}

// Ranks one group's 4 teams for one trial. team_pts/team_gf/team_ga are the full
// NUM_TEAMS-length arrays simulate_group_stage filled in; fixture_home_goals/
// fixture_away_goals are its NUM_GROUP_FIXTURES-length per-match output, needed
// here for head-to-head. Writes the 4 teams' global IDs into ranked_team_ids,
// best (group winner) first -- this is group_stage.py's rank_group, one pass of
// head-to-head only (not the recursive re-application FIFA's actual rule uses),
// same documented simplification as the source.
__device__ void rank_group(int group_id,
                            const int* team_pts, const int* team_gf, const int* team_ga,
                            const int* fixture_home_goals, const int* fixture_away_goals,
                            int* ranked_team_ids) {
    int gid[4], pts[4], gf[4], gd[4];
    for (int k = 0; k < 4; ++k) {
        gid[k] = d_group_teams[group_id][k];
        pts[k] = team_pts[gid[k]];
        gf[k]  = team_gf[gid[k]];
        gd[k]  = team_gf[gid[k]] - team_ga[gid[k]];
    }

    // Primary sort (pts desc, gd desc, gf desc) -- insertion sort, 4 elements.
    int order[4] = {0, 1, 2, 3};
    for (int i = 1; i < 4; ++i) {
        int key = order[i];
        int j = i - 1;
        while (j >= 0 && primary_better(pts[key], gd[key], gf[key],
                                         pts[order[j]], gd[order[j]], gf[order[j]])) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }

    // Local (within-group) fixture pattern: fixture m's two local team indices.
    // Matches exactly how tournament_data.cu builds each group's 6 fixtures
    // (the i < j loop over local indices 0..3) -- holds for every group.
    const int local_home[FIXTURES_PER_GROUP] = {0, 0, 0, 1, 1, 2};
    const int local_away[FIXTURES_PER_GROUP] = {1, 2, 3, 2, 3, 3};

    // Find tie runs under the primary key and resolve each with head-to-head.
    int i = 0;
    while (i < 4) {
        int j = i + 1;
        while (j < 4 && pts[order[j]] == pts[order[i]] &&
               gd[order[j]] == gd[order[i]] && gf[order[j]] == gf[order[i]]) {
            ++j;
        }
        int run_len = j - i;
        if (run_len > 1) {
            int tied_mask = 0;
            for (int k = i; k < j; ++k) tied_mask |= (1 << order[k]);

            int h2h_pts[4] = {0, 0, 0, 0};
            int h2h_gf[4]  = {0, 0, 0, 0};
            int h2h_ga[4]  = {0, 0, 0, 0};
            for (int m = 0; m < FIXTURES_PER_GROUP; ++m) {
                int li = local_home[m], lj = local_away[m];
                if ((tied_mask & (1 << li)) && (tied_mask & (1 << lj))) {
                    int f = group_id * FIXTURES_PER_GROUP + m;
                    int hg = fixture_home_goals[f], ag = fixture_away_goals[f];
                    h2h_gf[li] += hg; h2h_ga[li] += ag;
                    h2h_gf[lj] += ag; h2h_ga[lj] += hg;
                    if (hg > ag) h2h_pts[li] += 3;
                    else if (hg < ag) h2h_pts[lj] += 3;
                    else { h2h_pts[li] += 1; h2h_pts[lj] += 1; }
                }
            }

            // Re-sort just this tied run by the full tiebreak key.
            for (int a = i + 1; a < j; ++a) {
                int key = order[a];
                int b = a - 1;
                while (b >= i &&
                       tie_better(h2h_pts[key], h2h_gf[key] - h2h_ga[key], h2h_gf[key],
                                  d_team_elo[gid[key]], d_team_alpha_rank[gid[key]],
                                  h2h_pts[order[b]], h2h_gf[order[b]] - h2h_ga[order[b]], h2h_gf[order[b]],
                                  d_team_elo[gid[order[b]]], d_team_alpha_rank[gid[order[b]]])) {
                    order[b + 1] = order[b];
                    --b;
                }
                order[b + 1] = key;
            }
        }
        i = j;
    }

    for (int k = 0; k < 4; ++k) ranked_team_ids[k] = gid[order[k]];
}

// Ranks all 12 groups for one trial. group_order[g][0..3] = that group's 4 teams'
// global IDs, ranked 1st..4th.
__device__ void rank_all_groups(const int* team_pts, const int* team_gf, const int* team_ga,
                                 const int* fixture_home_goals, const int* fixture_away_goals,
                                 int group_order[NUM_GROUPS][TEAMS_PER_GROUP]) {
    for (int g = 0; g < NUM_GROUPS; ++g) {
        rank_group(g, team_pts, team_gf, team_ga, fixture_home_goals, fixture_away_goals,
                   group_order[g]);
    }
}
