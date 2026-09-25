#include "bracket_check.cuh"
#include "bracket.cuh"
#include "tournament_data.cuh"

__global__ void bracket_check_kernel(const int* match_ids, int num_checks,
                                      const int (*group_order)[TEAMS_PER_GROUP],
                                      const int* slot_team_id,
                                      const int* match_winner, const int* match_loser,
                                      int* resolved_a, int* resolved_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_checks) return;

    int m = match_ids[idx];
    resolved_a[idx] = resolve_spec(d_bracket_team_a_type[m], d_bracket_team_a_arg[m],
                                    group_order, slot_team_id, match_winner, match_loser);
    resolved_b[idx] = resolve_spec(d_bracket_team_b_type[m], d_bracket_team_b_arg[m],
                                    group_order, slot_team_id, match_winner, match_loser);
}

void launch_bracket_check(const int* d_match_ids, int num_checks,
                           const int* d_group_order_flat, const int* d_slot_team_id,
                           const int* d_match_winner, const int* d_match_loser,
                           int* d_resolved_a, int* d_resolved_b) {
    const int (*group_order)[TEAMS_PER_GROUP] =
        reinterpret_cast<const int (*)[TEAMS_PER_GROUP]>(d_group_order_flat);
    int threads = 32;
    int blocks = (num_checks + threads - 1) / threads;
    bracket_check_kernel<<<blocks, threads>>>(d_match_ids, num_checks, group_order,
                                               d_slot_team_id, d_match_winner, d_match_loser,
                                               d_resolved_a, d_resolved_b);
}
