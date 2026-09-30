#pragma once

// =============================================================================
// football/bracket.cuh  --  knockout stage (Round of 32 -> Final) for ONE trial.
//
// Direct device equivalent of bracket_engine.py. The 32 matches are stored in
// bracket_data.cu in dependency-safe order (every winner_of/loser_of reference
// points to an EARLIER match), so a single forward pass 0..31 always has both
// teams of a match resolved before that match is played.
// =============================================================================

#include "tournament_data.cuh"
#include "bracket_data.cuh"
#include "lambda_model.cuh"
#include "poisson_sample.cuh"
#include "shootout.cuh"
#include "rng.cuh"

// How a match was decided (recorded per match for inspection/testing).
#define DECIDED_REGULATION 0
#define DECIDED_EXTRA_TIME 1
#define DECIDED_PENALTIES  2

// Resolves one "team spec" -- a (type, arg) pair from the bracket table -- into
// a global team ID, using this trial's group results, third-place slot
// assignment and the knockout results built up so far.
// Direct device equivalent of bracket_engine.py's _resolve_team.
__device__ __forceinline__ int resolve_spec(int type, int arg,
                                             const int group_order[NUM_GROUPS][TEAMS_PER_GROUP],
                                             const int* slot_team_id,
                                             const int* match_winner, const int* match_loser) {
    switch (type) {
        case SPEC_GROUP_WINNER:      return group_order[arg][0];   // arg = group index; 1st place
        case SPEC_GROUP_RUNNERUP:    return group_order[arg][1];   // arg = group index; 2nd place
        case SPEC_THIRDPLACE_LOOKUP: return slot_team_id[arg];     // arg = third-place slot 0..7
        case SPEC_WINNER_OF_MATCH:   return match_winner[arg];     // arg = earlier match id
        case SPEC_LOSER_OF_MATCH:    return match_loser[arg];      // arg = earlier match id
    }
    return -1;  // unreachable -- every type is one of the 5 above
}

// Simulates all 32 knockout matches for one trial, in match_id order 0..31.
// All output arrays are length NUM_KNOCKOUT_MATCHES and caller-owned.
// Each match: regulation -> (if level) extra time -> (if still level) shootout.
// Direct device equivalent of bracket_engine.py's simulate_knockout + _play_match.
__device__ void simulate_knockout(RngState* state,
                                   const int group_order[NUM_GROUPS][TEAMS_PER_GROUP],
                                   const int* slot_team_id,
                                   int* match_team_a, int* match_team_b,
                                   int* match_goals_a, int* match_goals_b,
                                   int* match_decided_by,
                                   int* match_winner, int* match_loser) {
    for (int m = 0; m < NUM_KNOCKOUT_MATCHES; ++m) {
        // 1) Work out who plays in match m (earlier results are already filled in).
        int team_a = resolve_spec(d_bracket_team_a_type[m], d_bracket_team_a_arg[m],
                                   group_order, slot_team_id, match_winner, match_loser);
        int team_b = resolve_spec(d_bracket_team_b_type[m], d_bracket_team_b_arg[m],
                                   group_order, slot_team_id, match_winner, match_loser);
        match_team_a[m] = team_a;
        match_team_b[m] = team_b;

        // 2) Regulation time: Poisson goals for each side.
        float lam_a, lam_b;
        predict_lambda(team_a, team_b, /*is_knockout=*/true, /*extra_time=*/false, &lam_a, &lam_b);
        int goals_a = sample_goals(state, lam_a);
        int goals_b = sample_goals(state, lam_b);
        int decided_by = DECIDED_REGULATION;

        // 3) Level after 90 minutes -> extra time (lambda scaled to 30 minutes).
        //    This branch is what makes threads in a warp diverge: only some
        //    trials need it.
        if (goals_a == goals_b) {
            float et_lam_a, et_lam_b;
            predict_lambda(team_a, team_b, /*is_knockout=*/true, /*extra_time=*/true, &et_lam_a, &et_lam_b);
            goals_a += sample_goals(state, et_lam_a);
            goals_b += sample_goals(state, et_lam_b);
            decided_by = DECIDED_EXTRA_TIME;

            // 4) Still level -> penalty shootout (dampened Elo coin flip).
            if (goals_a == goals_b) {
                decided_by = DECIDED_PENALTIES;
                bool a_wins = sample_shootout(state, d_team_elo[team_a], d_team_elo[team_b]);
                match_goals_a[m] = goals_a;
                match_goals_b[m] = goals_b;
                match_decided_by[m] = decided_by;
                if (a_wins) { match_winner[m] = team_a; match_loser[m] = team_b; }
                else        { match_winner[m] = team_b; match_loser[m] = team_a; }
                continue;   // shootout decided it -- move on to the next match
            }
        }

        // 5) Decided in regulation or extra time: more goals wins.
        match_goals_a[m] = goals_a;
        match_goals_b[m] = goals_b;
        match_decided_by[m] = decided_by;
        if (goals_a > goals_b) { match_winner[m] = team_a; match_loser[m] = team_b; }
        else                   { match_winner[m] = team_b; match_loser[m] = team_a; }
    }
}
