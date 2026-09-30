#pragma once

// =============================================================================
// football/tournament_kernel.cuh  --  what ONE Monte Carlo trial means for football.
//
// One trial = one complete simulated World Cup:
//     group stage -> group standings -> third-place resolution -> knockout
//
// In addition to the champion, this version records which teams reached the
// quarterfinals, semifinals and final. The generic Monte Carlo engine aggregates
// those four stage results into GPU histograms.
// =============================================================================

#include "group_stage.cuh"
#include "standings.cuh"
#include "third_place.cuh"
#include "bracket.cuh"
#include "tournament_data.cuh"
#include "bracket_data.cuh"
#include "rng.cuh"

// Four 48-bit masks are enough because there are only 48 teams.
// bit t == 1 means team t reached that stage in this trial.
struct TournamentResult {
    unsigned long long stage0; // Quarterfinals
    unsigned long long stage1; // Semifinals
    unsigned long long stage2; // Final
    unsigned long long stage3; // Champion
};

struct SimulateTournament {
    // Runs one tournament using the calling thread's RNG state.
    // Returns the teams that reached QF/SF/Final and the champion.
    __device__ TournamentResult operator()(RngState* state) const {
        TournamentResult result{0ULL, 0ULL, 0ULL, 0ULL};

        // ---- 1. Group stage: play 72 matches, tally points / goals -----------
        int team_pts[NUM_TEAMS] = {0};
        int team_gf[NUM_TEAMS] = {0};
        int team_ga[NUM_TEAMS] = {0};
        int fixture_home_goals[NUM_GROUP_FIXTURES];
        int fixture_away_goals[NUM_GROUP_FIXTURES];
        simulate_group_stage(state, team_pts, team_gf, team_ga,
                              fixture_home_goals, fixture_away_goals);

        // ---- 2. Rank each group with the FIFA tiebreaker cascade -------------
        int group_order[NUM_GROUPS][TEAMS_PER_GROUP];
        rank_all_groups(team_pts, team_gf, team_ga, fixture_home_goals, fixture_away_goals,
                         group_order);

        // ---- 3. Pick the best 8 third-placed teams and assign R32 slots ------
        int slot_team_id[NUM_THIRDPLACE_SLOTS];
        resolve_third_place(group_order, team_pts, team_gf, team_ga, slot_team_id);

        // ---- 4. Knockout bracket: R32 -> R16 -> QF -> SF -> 3rd place -> Final
        int match_team_a[NUM_KNOCKOUT_MATCHES], match_team_b[NUM_KNOCKOUT_MATCHES];
        int match_goals_a[NUM_KNOCKOUT_MATCHES], match_goals_b[NUM_KNOCKOUT_MATCHES];
        int match_decided_by[NUM_KNOCKOUT_MATCHES];
        int match_winner[NUM_KNOCKOUT_MATCHES], match_loser[NUM_KNOCKOUT_MATCHES];
        simulate_knockout(state, group_order, slot_team_id,
                           match_team_a, match_team_b, match_goals_a, match_goals_b,
                           match_decided_by, match_winner, match_loser);

        // Bracket data uses match IDs 0..31:
        //   0..15 = Round of 32
        //  16..23 = Round of 16
        //  24..27 = Quarterfinals
        //  28..29 = Semifinals
        //       30 = Third-place match
        //       31 = Final
        for (int m = 24; m <= 27; ++m) {
            result.stage0 |= (1ULL << match_team_a[m]);
            result.stage0 |= (1ULL << match_team_b[m]);
        }
        for (int m = 28; m <= 29; ++m) {
            result.stage1 |= (1ULL << match_team_a[m]);
            result.stage1 |= (1ULL << match_team_b[m]);
        }

        result.stage2 |= (1ULL << match_team_a[MATCH_ID_FINAL]);
        result.stage2 |= (1ULL << match_team_b[MATCH_ID_FINAL]);
        result.stage3 |= (1ULL << match_winner[MATCH_ID_FINAL]);

        return result;
    }
};
