#pragma once

#include "tournament_data.cuh"
#include "third_place_data.cuh"

// True if third-place candidate "a" ranks strictly ahead of "b": points -> GD ->
// GS -> Elo (FIFA-ranking stand-in) -> alphabetical name. No head-to-head --
// third-place teams are from different groups, so it generally doesn't apply,
// same simplification third_place.py documents.
__device__ __forceinline__ bool thirdplace_better(int pts_a, int gd_a, int gf_a, float elo_a, int alpha_a,
                                                    int pts_b, int gd_b, int gf_b, float elo_b, int alpha_b) {
    if (pts_a != pts_b) return pts_a > pts_b;
    if (gd_a != gd_b) return gd_a > gd_b;
    if (gf_a != gf_b) return gf_a > gf_b;
    if (elo_a != elo_b) return elo_a > elo_b;
    return alpha_a < alpha_b;
}

// Ranks all 12 groups' third-place teams for one trial, resolves which 8
// qualify via the official 495-combo lookup, and writes which team fills each
// of the 8 fixed R32 slots. group_order is rank_all_groups' output; team_pts/
// team_gf/team_ga are simulate_group_stage's output. slot_team_id[0..7] comes
// out in slot order 1A,1B,1D,1E,1G,1I,1K,1L (global team IDs) -- the device
// equivalent of third_place.py's rank_third_place_teams + resolve_third_place_slots.
__device__ void resolve_third_place(const int group_order[NUM_GROUPS][TEAMS_PER_GROUP],
                                     const int* team_pts, const int* team_gf, const int* team_ga,
                                     int* slot_team_id) {
    int third_gid[NUM_GROUPS], pts[NUM_GROUPS], gf[NUM_GROUPS], gd[NUM_GROUPS];
    for (int g = 0; g < NUM_GROUPS; ++g) {
        int gid = group_order[g][2];  // index 2 = 3rd place
        third_gid[g] = gid;
        pts[g] = team_pts[gid];
        gf[g]  = team_gf[gid];
        gd[g]  = team_gf[gid] - team_ga[gid];
    }

    // Rank all 12 by the cascade -- insertion sort, 12 elements.
    int order[NUM_GROUPS];
    for (int g = 0; g < NUM_GROUPS; ++g) order[g] = g;
    for (int i = 1; i < NUM_GROUPS; ++i) {
        int key = order[i];
        int j = i - 1;
        while (j >= 0 &&
               thirdplace_better(pts[key], gd[key], gf[key],
                                  d_team_elo[third_gid[key]], d_team_alpha_rank[third_gid[key]],
                                  pts[order[j]], gd[order[j]], gf[order[j]],
                                  d_team_elo[third_gid[order[j]]], d_team_alpha_rank[third_gid[order[j]]])) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }

    // Top 8 groups (by their 3rd-place team's rank) qualify.
    int qualifying_mask = 0;
    for (int k = 0; k < 8; ++k) qualifying_mask |= (1 << order[k]);

    // Linear scan of the 495-row table -- simple and correct; the table is
    // exhaustive over every C(12,8)=495 combination, so exactly one row always
    // matches. Worth revisiting only if profiling ever shows this scan costing
    // something real -- 495 int comparisons is cheap next to the rest of a trial.
    int row = -1;
    for (int r = 0; r < NUM_THIRDPLACE_COMBOS; ++r) {
        if (d_thirdplace_combo_mask[r] == qualifying_mask) { row = r; break; }
    }

    for (int k = 0; k < NUM_THIRDPLACE_SLOTS; ++k) {
        int group = d_thirdplace_slot_group[row][k];
        slot_team_id[k] = third_gid[group];
    }
}
