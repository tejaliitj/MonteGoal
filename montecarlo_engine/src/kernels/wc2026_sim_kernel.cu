#include "kernels/wc2026_sim_kernel.cuh"

#include "rng/rng_engine.cuh"
#include "distributions/poisson_dist.cuh"
#include "distributions/uniform_dist.cuh"

#include <cuda_runtime.h>

#include <cmath>

/* ============================================================
   Utility functions
   ============================================================ */

__device__ inline double get_lambda(
    const double* lambda_matrix,
    int home,
    int away
) {
    return lambda_matrix[
        home * WC_NUM_TEAMS + away
    ];
}


/* ============================================================
   Standings comparison
   ============================================================ */

__device__ bool better_team(
    int a,
    int b,
    const StatsGPU* stats,
    curandStatePhilox4_32_10_t* state
) {
    if (stats[a].Pts != stats[b].Pts) {
        return stats[a].Pts > stats[b].Pts;
    }

    if (stats[a].GD != stats[b].GD) {
        return stats[a].GD > stats[b].GD;
    }

    if (stats[a].GF != stats[b].GF) {
        return stats[a].GF > stats[b].GF;
    }

    /*
     * Random tiebreak.
     */
    return sample_uniform(state) < 0.5;
}


/* ============================================================
   Sort four teams inside a group
   ============================================================ */

__device__ void sort_group(
    int* teams,
    int count,
    const StatsGPU* stats,
    curandStatePhilox4_32_10_t* state
) {
    for (int i = 0; i < count - 1; ++i) {

        for (int j = i + 1; j < count; ++j) {

            if (better_team(
                    teams[j],
                    teams[i],
                    stats,
                    state
                )) {

                int temp = teams[i];

                teams[i] = teams[j];
                teams[j] = temp;
            }
        }
    }
}


/* ============================================================
   Update standings
   ============================================================ */

__device__ void update_match_stats(
    StatsGPU* home,
    StatsGPU* away,
    int home_goals,
    int away_goals
) {
    home->P++;
    away->P++;

    home->GF += home_goals;
    home->GA += away_goals;

    away->GF += away_goals;
    away->GA += home_goals;

    home->GD =
        home->GF - home->GA;

    away->GD =
        away->GF - away->GA;

    if (home_goals > away_goals) {

        home->W++;
        away->L++;

        home->Pts += 3;

    } else if (home_goals < away_goals) {

        away->W++;
        home->L++;

        away->Pts += 3;

    } else {

        home->D++;
        away->D++;

        home->Pts++;
        away->Pts++;
    }
}


/* ============================================================
   Group-stage match
   ============================================================ */

__device__ void play_group_match(
    int home,
    int away,
    const double* lambda_matrix,
    StatsGPU* stats,
    curandStatePhilox4_32_10_t* state
) {
    double lambda_home =
        get_lambda(
            lambda_matrix,
            home,
            away
        );

    double lambda_away =
        get_lambda(
            lambda_matrix,
            away,
            home
        );

    int goals_home =
        sample_poisson(
            state,
            lambda_home
        );

    int goals_away =
        sample_poisson(
            state,
            lambda_away
        );

    update_match_stats(
        &stats[home],
        &stats[away],
        goals_home,
        goals_away
    );
}


/* ============================================================
   Knockout match
   ============================================================ */

__device__ int play_knockout_match(
    int team_a,
    int team_b,
    const double* lambda_matrix,
    curandStatePhilox4_32_10_t* state
) {
    double lambda_a =
        get_lambda(
            lambda_matrix,
            team_a,
            team_b
        );

    double lambda_b =
        get_lambda(
            lambda_matrix,
            team_b,
            team_a
        );

    /*
     * Normal time
     */
    int goals_a =
        sample_poisson(
            state,
            lambda_a
        );

    int goals_b =
        sample_poisson(
            state,
            lambda_b
        );

    if (goals_a > goals_b) {
        return team_a;
    }

    if (goals_b > goals_a) {
        return team_b;
    }

    /*
     * Extra time.
     *
     * Preserve the CPU simulator's
     * lambda / 3 convention.
     */
    int et_a =
        sample_poisson(
            state,
            lambda_a / 3.0
        );

    int et_b =
        sample_poisson(
            state,
            lambda_b / 3.0
        );

    goals_a += et_a;
    goals_b += et_b;

    if (goals_a > goals_b) {
        return team_a;
    }

    if (goals_b > goals_a) {
        return team_b;
    }

    /*
     * Shootout.
     *
     * We no longer have to calculate Elo here
     * because the lambda matrix is being used as
     * the matchup model.
     *
     * For now, use the relative lambda values to
     * determine shootout probability.
     */
    double total_lambda =
        lambda_a + lambda_b;

    double p_a;

    if (total_lambda > 0.0) {

        p_a =
            lambda_a / total_lambda;

    } else {

        p_a = 0.5;
    }

    return (
        sample_uniform(state) < p_a
    )
        ? team_a
        : team_b;
}


/* ============================================================
   Fisher-Yates shuffle
   ============================================================ */

__device__ void shuffle_array(
    int* array,
    int n,
    curandStatePhilox4_32_10_t* state
) {
    for (int i = n - 1; i > 0; --i) {

        int j =
            static_cast<int>(
                sample_uniform(state)
                * (i + 1)
            );

        if (j > i) {
            j = i;
        }

        int temp = array[i];

        array[i] = array[j];
        array[j] = temp;
    }
}


/* ============================================================
   Main tournament kernel
   ============================================================ */

__global__ void wc2026_simulation_kernel(
    const TeamGPU* teams,
    const double* lambda_matrix,
    const FixtureGPU* fixtures,
    int num_simulations,
    unsigned long long seed,
    int* champion_counts,
    int* round_counts
) {
    int sim_id =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (sim_id >= num_simulations) {
        return;
    }

    /*
     * One independent Philox stream per tournament.
     */
    curandStatePhilox4_32_10_t state;

    init_philox(
        &state,
        seed,
        static_cast<unsigned long long>(sim_id)
    );

    /*
     * Local standings.
     *
     * Each CUDA thread has its own tournament.
     */
    StatsGPU stats[WC_NUM_TEAMS];

    for (int i = 0;
         i < WC_NUM_TEAMS;
         ++i) {

        stats[i].P = 0;
        stats[i].W = 0;
        stats[i].D = 0;
        stats[i].L = 0;

        stats[i].GF = 0;
        stats[i].GA = 0;
        stats[i].GD = 0;

        stats[i].Pts = 0;
    }

    /* ========================================================
       GROUP STAGE
       ======================================================== */

    for (int f = 0;
         f < WC_NUM_FIXTURES;
         ++f) {

        int home =
            fixtures[f].home_idx;

        int away =
            fixtures[f].away_idx;

        play_group_match(
            home,
            away,
            lambda_matrix,
            stats,
            &state
        );
    }

    /* ========================================================
       GROUP RANKINGS
       ======================================================== */

    int group_teams[WC_NUM_GROUPS][WC_GROUP_SIZE];

    for (int g = 0;
         g < WC_NUM_GROUPS;
         ++g) {

        int count = 0;

        for (int t = 0;
             t < WC_NUM_TEAMS;
             ++t) {

            if (teams[t].group_id == g) {

                group_teams[g][count] = t;

                count++;
            }
        }

        sort_group(
            group_teams[g],
            WC_GROUP_SIZE,
            stats,
            &state
        );
    }

    /* ========================================================
       QUALIFIED TEAMS
       ======================================================== */

    int winners[WC_NUM_GROUPS];
    int runners[WC_NUM_GROUPS];
    int thirds[WC_NUM_GROUPS];

    for (int g = 0;
         g < WC_NUM_GROUPS;
         ++g) {

        winners[g] =
            group_teams[g][0];

        runners[g] =
            group_teams[g][1];

        thirds[g] =
            group_teams[g][2];
    }

    /*
     * Select best 8 third-place teams.
     */
    int third_candidates[WC_NUM_GROUPS];

    for (int i = 0;
         i < WC_NUM_GROUPS;
         ++i) {

        third_candidates[i] = thirds[i];
    }

    sort_group(
        third_candidates,
        WC_NUM_GROUPS,
        stats,
        &state
    );

    int qualified_thirds[8];

    for (int i = 0; i < 8; ++i) {

        qualified_thirds[i] =
            third_candidates[i];
    }

    /* ========================================================
       R32
       ======================================================== */

    int r32_a[WC_NUM_R32_MATCHES];
    int r32_b[WC_NUM_R32_MATCHES];

    int match_count = 0;

    /*
     * Winner vs runner pairings.
     *
     * Start with shuffled runners.
     */
    int shuffled_runners[WC_NUM_GROUPS];

    for (int i = 0;
         i < WC_NUM_GROUPS;
         ++i) {

        shuffled_runners[i] =
            runners[i];
    }

    shuffle_array(
        shuffled_runners,
        WC_NUM_GROUPS,
        &state
    );

    for (int i = 0;
         i < WC_NUM_GROUPS;
         ++i) {

        int winner =
            winners[i];

        int runner =
            shuffled_runners[i];

        /*
         * Avoid same-group pairing.
         */
        if (
            teams[winner].group_id ==
            teams[runner].group_id
        ) {
            int next =
                (i + 1) % WC_NUM_GROUPS;

            int temp =
                shuffled_runners[i];

            shuffled_runners[i] =
                shuffled_runners[next];

            shuffled_runners[next] =
                temp;

            runner =
                shuffled_runners[i];
        }

        r32_a[match_count] = winner;
        r32_b[match_count] = runner;

        match_count++;
    }

    /*
     * Third-place teams form the remaining 4 matches.
     */
    int shuffled_thirds[8];

    for (int i = 0; i < 8; ++i) {

        shuffled_thirds[i] =
            qualified_thirds[i];
    }

    shuffle_array(
        shuffled_thirds,
        8,
        &state
    );

    for (int i = 0; i < 8; i += 2) {

        r32_a[match_count] =
            shuffled_thirds[i];

        r32_b[match_count] =
            shuffled_thirds[i + 1];

        match_count++;
    }

    /*
     * Shuffle R32 matches.
     */
    for (int i = WC_NUM_R32_MATCHES - 1;
         i > 0;
         --i) {

        int j =
            static_cast<int>(
                sample_uniform(&state)
                * (i + 1)
            );

        if (j > i) {
            j = i;
        }

        int temp_a = r32_a[i];
        int temp_b = r32_b[i];

        r32_a[i] = r32_a[j];
        r32_b[i] = r32_b[j];

        r32_a[j] = temp_a;
        r32_b[j] = temp_b;
    }

    /* ========================================================
       R32
       ======================================================== */

    int r16[16];

    for (int i = 0; i < 16; ++i) {

        r16[i] =
            play_knockout_match(
                r32_a[i],
                r32_b[i],
                lambda_matrix,
                &state
            );
    }

    /* ========================================================
       R16
       ======================================================== */

    int qf[8];

    for (int i = 0; i < 8; ++i) {

        qf[i] =
            play_knockout_match(
                r16[2 * i],
                r16[2 * i + 1],
                lambda_matrix,
                &state
            );
    }

    /* ========================================================
       QUARTER FINALS
       ======================================================== */

    int sf[4];

    for (int i = 0; i < 4; ++i) {

        sf[i] =
            play_knockout_match(
                qf[2 * i],
                qf[2 * i + 1],
                lambda_matrix,
                &state
            );
    }

    /* ========================================================
       SEMI FINALS
       ======================================================== */

    int finalists[2];

    finalists[0] =
        play_knockout_match(
            sf[0],
            sf[1],
            lambda_matrix,
            &state
        );

    finalists[1] =
        play_knockout_match(
            sf[2],
            sf[3],
            lambda_matrix,
            &state
        );

    /* ========================================================
       FINAL
       ======================================================== */

    int champion =
        play_knockout_match(
            finalists[0],
            finalists[1],
            lambda_matrix,
            &state
        );

    /*
     * Champion counter.
     */
    atomicAdd(
        &champion_counts[champion],
        1
    );

    /*
     * For now we count the stages reached by
     * looking at the tournament progression.
     *
     * Champion:
     */
    atomicAdd(
        &round_counts[
            champion * WC_NUM_STAGES
            + STAGE_CHAMPION
        ],
        1
    );

    /*
     * Finalists.
     */
    atomicAdd(
        &round_counts[
            finalists[0] * WC_NUM_STAGES
            + STAGE_FINAL
        ],
        1
    );

    atomicAdd(
        &round_counts[
            finalists[1] * WC_NUM_STAGES
            + STAGE_FINAL
        ],
        1
    );

    /*
     * Semi-finalists.
     */
    for (int i = 0; i < 4; ++i) {

        atomicAdd(
            &round_counts[
                sf[i] * WC_NUM_STAGES
                + STAGE_SF
            ],
            1
        );
    }

    /*
     * Quarter-finalists.
     */
    for (int i = 0; i < 8; ++i) {

        atomicAdd(
            &round_counts[
                qf[i] * WC_NUM_STAGES
                + STAGE_QF
            ],
            1
        );
    }

    /*
     * R16 participants.
     */
    for (int i = 0; i < 16; ++i) {

        atomicAdd(
            &round_counts[
                r16[i] * WC_NUM_STAGES
                + STAGE_R16
            ],
            1
        );
    }

    /*
     * R32 participants.
     */
    for (int i = 0; i < 16; ++i) {

        atomicAdd(
            &round_counts[
                r32_a[i] * WC_NUM_STAGES
                + STAGE_R32
            ],
            1
        );

        atomicAdd(
            &round_counts[
                r32_b[i] * WC_NUM_STAGES
                + STAGE_R32
            ],
            1
        );
    }

    /*
     * All qualified teams reached at least R32.
     *
     * Group-stage participants that did not qualify
     * do not get a knockout-stage count.
     */
}


/* ============================================================
   Host launcher
   ============================================================ */

void launch_wc2026_simulation(
    const TeamGPU* d_teams,
    const double* d_lambda_matrix,
    const FixtureGPU* d_fixtures,
    int num_simulations,
    unsigned long long seed,
    int* d_champion_counts,
    int* d_round_counts,
    cudaStream_t stream
) {
    constexpr int THREADS_PER_BLOCK = 256;

    int blocks =
        (
            num_simulations
            + THREADS_PER_BLOCK
            - 1
        ) / THREADS_PER_BLOCK;

    wc2026_simulation_kernel<<<
        blocks,
        THREADS_PER_BLOCK,
        0,
        stream
    >>>(
        d_teams,
        d_lambda_matrix,
        d_fixtures,
        num_simulations,
        seed,
        d_champion_counts,
        d_round_counts
    );
}