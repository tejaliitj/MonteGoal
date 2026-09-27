/*
 * group_stage.cuh -- plays one group's round robin and ranks the 4 teams
 * using the same real FIFA tiebreakers as group_stage.py: Points -> Goal
 * Difference -> Goals For -> Head-to-head result -> random draw of lots.
 */
#ifndef GROUP_STAGE_CUH
#define GROUP_STAGE_CUH

#include <curand_kernel.h>
#include "../config.cuh"

struct GroupResult
{
    int winner;
    int runner_up;
    int third_team;
    int third_pts;
    int third_gd;
    int third_gf;
};

// One qualifying candidate for the "best third-placed teams" ranking.
struct ThirdEntry
{
    int group_letter; // 0=A .. 11=L
    int team_id;
    int pts;
    int gd;
    int gf;
};

// Plays every match in a 4-team round robin and ranks the teams.
__device__ GroupResult play_group_device(
    const int teams[TEAMS_PER_GROUP],
    curandStatePhilox4_32_10_t *state
);

// Ranks all 12 third-place entries the same way group standings are ranked
// (Pts -> GD -> GF -> random), since third-place teams from different
// groups never played each other so no head-to-head is possible. Sorts
// `entries` in place, best team first, mirrors rank_third_place_teams().
__device__ void rank_third_place_teams_device(
    ThirdEntry entries[NUM_GROUPS],
    curandStatePhilox4_32_10_t *state
);

#endif
