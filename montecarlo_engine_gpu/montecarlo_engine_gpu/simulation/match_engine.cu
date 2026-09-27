#include "match_engine.cuh"
#include "../distributions/poisson.cuh"
#include "../distributions/uniform.cuh"
#include <cuda_runtime.h>

__constant__ float d_lambda[NUM_TEAMS * NUM_TEAMS];

void upload_lambda_matrix(const float h_lambda[NUM_TEAMS * NUM_TEAMS])
{
    cudaMemcpyToSymbol(d_lambda, h_lambda, sizeof(float) * NUM_TEAMS * NUM_TEAMS);
}

__device__ float get_lambda(int home_id, int away_id)
{
    return d_lambda[home_id * NUM_TEAMS + away_id];
}

__device__ void play_regulation_device(
    curandStatePhilox4_32_10_t *state,
    int home_id, int away_id,
    int &goals_home, int &goals_away)
{
    float lam_home = get_lambda(home_id, away_id);
    float lam_away = get_lambda(away_id, home_id);

    goals_home = (int)generate_poisson(state, (double)lam_home);
    goals_away = (int)generate_poisson(state, (double)lam_away);
}

// Best-of-5 then sudden death. Returns 0 if home wins, 1 if away wins.
// Mirrors match_engine.py's simulate_penalty_shootout.
static __device__ int simulate_penalty_shootout_device(
    curandStatePhilox4_32_10_t *state,
    int home_id, int away_id)
{
    float lam_home = get_lambda(home_id, away_id);
    float lam_away = get_lambda(away_id, home_id);
    float gap = lam_home - lam_away;

    float rate_home = PENALTY_BASE_RATE + gap * PENALTY_GAP_SENSITIVITY;
    rate_home = fminf(fmaxf(rate_home, PENALTY_RATE_MIN), PENALTY_RATE_MAX);

    float rate_away = PENALTY_BASE_RATE - gap * PENALTY_GAP_SENSITIVITY;
    rate_away = fminf(fmaxf(rate_away, PENALTY_RATE_MIN), PENALTY_RATE_MAX);

    int pen_home = 0, pen_away = 0;

    for (int i = 0; i < 5; i++)
    {
        if (generate_uniform(state) < rate_home) pen_home++;
        if (generate_uniform(state) < rate_away) pen_away++;
    }

    while (pen_home == pen_away)
    {
        if (generate_uniform(state) < rate_home) pen_home++;
        if (generate_uniform(state) < rate_away) pen_away++;
    }

    return pen_home > pen_away ? 0 : 1;
}

__device__ int play_knockout_match_device(
    curandStatePhilox4_32_10_t *state,
    int home_id, int away_id)
{
    int gh, ga;
    play_regulation_device(state, home_id, away_id, gh, ga);

    if (gh != ga)
        return gh > ga ? home_id : away_id;

    // Level after 90 -- extra time
    float lam_home_et = get_lambda(home_id, away_id) * EXTRA_TIME_SCALE;
    float lam_away_et = get_lambda(away_id, home_id) * EXTRA_TIME_SCALE;

    gh += (int)generate_poisson(state, (double)lam_home_et);
    ga += (int)generate_poisson(state, (double)lam_away_et);

    if (gh != ga)
        return gh > ga ? home_id : away_id;

    // Still level -- penalties
    int who = simulate_penalty_shootout_device(state, home_id, away_id);
    return who == 0 ? home_id : away_id;
}
