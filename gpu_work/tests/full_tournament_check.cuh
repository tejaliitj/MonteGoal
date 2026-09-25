#pragma once

// Runs one complete tournament trial (group stage -> standings -> third place
// -> knockout) on a single thread. Every array is device memory the caller
// allocates: team_pts/gf/ga length NUM_TEAMS, group_order_flat length
// NUM_GROUPS*TEAMS_PER_GROUP, slot_team_id length NUM_THIRDPLACE_SLOTS, every
// match_* array length NUM_KNOCKOUT_MATCHES. team_pts/gf/ga must be
// zero-initialized by the caller before the call.
void launch_full_tournament_check(unsigned long long seed, unsigned long long counter_offset,
                                   int* d_team_pts, int* d_team_gf, int* d_team_ga,
                                   int* d_group_order_flat, int* d_slot_team_id,
                                   int* d_match_team_a, int* d_match_team_b,
                                   int* d_match_goals_a, int* d_match_goals_b,
                                   int* d_match_decided_by,
                                   int* d_match_winner, int* d_match_loser);
