#include "third_place_check.cuh"
#include "third_place.cuh"
#include "tournament_data.cuh"

__global__ void thirdplace_check_kernel(const int (*group_order)[TEAMS_PER_GROUP],
                                         const int* team_pts, const int* team_gf, const int* team_ga,
                                         int* slot_team_id) {
    resolve_third_place(group_order, team_pts, team_gf, team_ga, slot_team_id);
}

void launch_thirdplace_check(const int* d_group_order_flat,
                              const int* d_team_pts, const int* d_team_gf, const int* d_team_ga,
                              int* d_slot_team_id) {
    const int (*group_order)[TEAMS_PER_GROUP] =
        reinterpret_cast<const int (*)[TEAMS_PER_GROUP]>(d_group_order_flat);
    thirdplace_check_kernel<<<1, 1>>>(group_order, d_team_pts, d_team_gf, d_team_ga, d_slot_team_id);
}
