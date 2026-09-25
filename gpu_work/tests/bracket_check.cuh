#pragma once

// Validation-only harness. Runs resolve_spec() for a handful of real match_ids
// from the uploaded bracket data, using caller-supplied synthetic group_order/
// slot_team_id/match_winner/match_loser (with distinguishable marker values) so
// the exact resolved team_a/team_b can be checked against a hand-computed
// expectation. No RNG involved -- this only tests the bracket data + resolution
// logic, not match simulation.
void launch_bracket_check(const int* d_match_ids, int num_checks,
                           const int* d_group_order_flat, const int* d_slot_team_id,
                           const int* d_match_winner, const int* d_match_loser,
                           int* d_resolved_a, int* d_resolved_b);
