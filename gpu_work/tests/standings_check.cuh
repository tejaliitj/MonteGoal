#pragma once

// Validation-only harness. Launches a single thread that calls rank_group() on
// group 0 with the given (caller-constructed) stats and fixture-goal arrays, and
// writes the resulting ranked team IDs to d_ranked_out[0..3]. Used with
// hand-crafted inputs where the expected order is known by construction, to
// unit-test the tiebreak cascade directly rather than only checking it
// statistically the way the group-stage harness does.
void launch_standings_check(const int* d_team_pts, const int* d_team_gf, const int* d_team_ga,
                             const int* d_fixture_home_goals, const int* d_fixture_away_goals,
                             int* d_ranked_out);
