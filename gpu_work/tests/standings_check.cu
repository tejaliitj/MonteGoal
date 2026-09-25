#include "standings_check.cuh"
#include "standings.cuh"

__global__ void standings_check_kernel(const int* team_pts, const int* team_gf, const int* team_ga,
                                        const int* fixture_home_goals, const int* fixture_away_goals,
                                        int* ranked_out) {
    rank_group(0, team_pts, team_gf, team_ga, fixture_home_goals, fixture_away_goals, ranked_out);
}

void launch_standings_check(const int* d_team_pts, const int* d_team_gf, const int* d_team_ga,
                             const int* d_fixture_home_goals, const int* d_fixture_away_goals,
                             int* d_ranked_out) {
    standings_check_kernel<<<1, 1>>>(d_team_pts, d_team_gf, d_team_ga,
                                      d_fixture_home_goals, d_fixture_away_goals, d_ranked_out);
}
