/*
 * match_engine.cuh -- everything about simulating ONE match, GPU version
 * of match_engine.py. Knows nothing about groups, brackets, or tournaments.
 *
 *     get_lambda(home_id, away_id)                    -> lambda lookup
 *     play_regulation_device(state, home, away, gh, ga)
 *     play_knockout_match_device(state, home, away)   -> winner id, handles
 *         regulation -> extra time -> penalties automatically.
 *
 * expected_goals()'s (home,away)/(away,home) fallback from match_engine.py
 * is gone here: the lambda matrix is dense and directional (home advantage
 * is already baked into its asymmetry), so every ordered pair is always
 * present -- no fallback needed.
 */
#ifndef MATCH_ENGINE_CUH
#define MATCH_ENGINE_CUH

#include <curand_kernel.h>
#include "../config.cuh"

extern __constant__ float d_lambda[NUM_TEAMS * NUM_TEAMS];

// Host-side: uploads the flattened lambda matrix once at startup.
void upload_lambda_matrix(const float h_lambda[NUM_TEAMS * NUM_TEAMS]);

__device__ float get_lambda(int home_id, int away_id);

__device__ void play_regulation_device(
    curandStatePhilox4_32_10_t *state,
    int home_id, int away_id,
    int &goals_home, int &goals_away
);

// Plays regulation, then extra time if level, then penalties if still
// level. Returns the winning team's id.
__device__ int play_knockout_match_device(
    curandStatePhilox4_32_10_t *state,
    int home_id, int away_id
);

#endif
