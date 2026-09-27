/*
 * bracket.cuh -- direct translation of bracket.py's fixed FIFA Round-of-32
 * structure and fixed progression graph through to the final.
 *
 * Roles are encoded as (kind, letter) instead of strings like "W:A":
 *     kind ROLE_W = that group's winner
 *     kind ROLE_R = that group's runner-up
 *     kind ROLE_T = the third-place qualifier assigned to that slot letter
 * letter is a group-letter index, A=0 .. L=11.
 *
 * R16_PAIRS / QF_PAIRS / SF_PAIRS mean exactly what they do in bracket.py:
 * which match's winner plays which other match's winner, fixed by position,
 * not by team identity.
 *
 * NOTE on third-place slot assignment: FIFA's exact rule depends on a
 * published 495-row combination table. As in bracket.py, this uses the same
 * documented simplified substitute (assign_third_place_slots_device) rather
 * than embedding that table. Everything else here (who plays whom by role,
 * and the match graph through R16/QF/SF/Final) is the real FIFA structure.
 */
#ifndef BRACKET_CUH
#define BRACKET_CUH

#include <curand_kernel.h>
#include "../config.cuh"

#define ROLE_W 0
#define ROLE_R 1
#define ROLE_T 2

// Definitions live in bracket.cu; every other translation unit sees them
// via these extern declarations (requires -rdc=true, same as this whole project).
extern __constant__ int d_r32_kind[16][2];
extern __constant__ int d_r32_letter[16][2];
extern __constant__ int d_r16_pairs[8][2];
extern __constant__ int d_qf_pairs[4][2];
extern __constant__ int d_sf_pairs[2][2];
extern __constant__ int d_third_slot_order[8];

// Host-side: fills the __constant__ arrays above once at startup.
void upload_bracket_constants();

// Resolves one role (kind, letter) to an actual team id.
__device__ int resolve_role_device(
    int kind, int letter,
    const int group_winner[NUM_GROUPS],
    const int group_runner_up[NUM_GROUPS],
    const int third_team_by_group[NUM_GROUPS],
    const int slot_to_third_group[NUM_GROUPS]
);

// Simplified substitute for FIFA's full 495-combination table (see module
// docstring above / bracket.py's assign_third_place_slots). Takes the 8
// qualifying third-place teams' group letters (best-to-worst order, as
// produced by rank_third_place_teams_device), fills slot_to_third_group
// indexed by slot letter (size NUM_GROUPS, unused entries left at -1).
__device__ void assign_third_place_slots_device(
    const int qualifying_group_letters[8],
    int slot_to_third_group[NUM_GROUPS]
);

#endif
