/*
 * tournament.cuh -- GPU version of tournament.py. world_cup_kernel plays
 * ONE full 48-team World Cup per thread: group stage, best-third-place
 * selection, the real R32 bracket, and every knockout round through to
 * the final. Ties bracket.cuh / group_stage.cuh / match_engine.cuh
 * together exactly the way tournament.py ties its three modules together.
 */
#ifndef TOURNAMENT_CUH
#define TOURNAMENT_CUH

#include <curand_kernel.h>
#include "../config.cuh"

extern __constant__ int d_group_teams[NUM_GROUPS][TEAMS_PER_GROUP];

// Host-side: uploads which team ids belong to which group, once at startup.
void upload_group_teams(const int h_group_teams[NUM_GROUPS][TEAMS_PER_GROUP]);

// One thread = one tournament.
//   champion_out[idx]        -> winning team id
//   semis_out[idx*4 .. +3]   -> the 4 quarterfinal winners (semifinalists)
//   finalists_out[idx*2 .. +1] -> the 2 finalists
__global__ void world_cup_kernel(
    curandStatePhilox4_32_10_t *states,
    int total_threads,
    int *champion_out,
    int *semis_out,
    int *finalists_out
);

#endif
