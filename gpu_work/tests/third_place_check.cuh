#pragma once

// Validation-only harness. Launches a single thread that calls
// resolve_third_place() with the given (caller-constructed) group_order and
// team stats, writing the resulting 8 slot team IDs to d_slot_team_id[0..7].
void launch_thirdplace_check(const int* d_group_order_flat,
                              const int* d_team_pts, const int* d_team_gf, const int* d_team_ga,
                              int* d_slot_team_id);
