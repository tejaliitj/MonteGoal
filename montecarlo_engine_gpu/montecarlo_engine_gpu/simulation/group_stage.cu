#include "group_stage.cuh"
#include "match_engine.cuh"
#include "../distributions/uniform.cuh"

__device__ GroupResult play_group_device(
    const int teams[TEAMS_PER_GROUP],
    curandStatePhilox4_32_10_t *state)
{
    int pts[4] = {0, 0, 0, 0};
    int gf[4] = {0, 0, 0, 0};
    int ga[4] = {0, 0, 0, 0};

    // head-to-head goal difference, from row i's perspective vs j
    int h2h_diff[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            h2h_diff[i][j] = 0;

    for (int i = 0; i < 4; i++)
    {
        for (int j = i + 1; j < 4; j++)
        {
            int a_id = teams[i], b_id = teams[j];
            int ga_, gb_;
            play_regulation_device(state, a_id, b_id, ga_, gb_);

            gf[i] += ga_; ga[i] += gb_;
            gf[j] += gb_; ga[j] += ga_;

            if (ga_ > gb_)
                pts[i] += 3;
            else if (gb_ > ga_)
                pts[j] += 3;
            else
            {
                pts[i] += 1;
                pts[j] += 1;
            }

            h2h_diff[i][j] = ga_ - gb_;
            h2h_diff[j][i] = gb_ - ga_;
        }
    }

    // One random tiebreak value per team, drawn once -- mirrors
    // group_stage.py's `rnd = {t: rng.random() for t in teams}`.
    float rnd[4];
    for (int i = 0; i < 4; i++)
        rnd[i] = generate_uniform(state);

    // Insertion sort (4 elements) using the same precedence as
    // group_stage.py's sort_key_cmp: Pts desc -> GD desc -> GF desc ->
    // head-to-head goal diff -> random draw.
    int order[4] = {0, 1, 2, 3};
    for (int i = 1; i < 4; i++)
    {
        int key = order[i];
        int j = i - 1;
        while (j >= 0)
        {
            int x = key, y = order[j];
            bool key_before_y;

            if (pts[x] != pts[y])
            {
                key_before_y = pts[x] > pts[y];
            }
            else
            {
                int gdx = gf[x] - ga[x], gdy = gf[y] - ga[y];
                if (gdx != gdy)
                    key_before_y = gdx > gdy;
                else if (gf[x] != gf[y])
                    key_before_y = gf[x] > gf[y];
                else if (h2h_diff[x][y] != 0)
                    key_before_y = h2h_diff[x][y] > 0;
                else
                    key_before_y = rnd[x] > rnd[y];
            }

            if (!key_before_y)
                break;

            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    GroupResult r;
    r.winner = teams[order[0]];
    r.runner_up = teams[order[1]];
    r.third_team = teams[order[2]];
    r.third_pts = pts[order[2]];
    r.third_gd = gf[order[2]] - ga[order[2]];
    r.third_gf = gf[order[2]];
    return r;
}

__device__ void rank_third_place_teams_device(
    ThirdEntry entries[NUM_GROUPS],
    curandStatePhilox4_32_10_t *state)
{
    // Insertion sort (12 elements). No head-to-head (these teams never
    // played each other); random tiebreak drawn fresh per comparison,
    // same as rank_third_place_teams()'s `rng.random() > 0.5`.
    for (int i = 1; i < NUM_GROUPS; i++)
    {
        ThirdEntry key = entries[i];
        int j = i - 1;
        while (j >= 0)
        {
            ThirdEntry &y = entries[j];
            bool key_before_y;

            if (key.pts != y.pts)
                key_before_y = key.pts > y.pts;
            else if (key.gd != y.gd)
                key_before_y = key.gd > y.gd;
            else if (key.gf != y.gf)
                key_before_y = key.gf > y.gf;
            else
                key_before_y = generate_uniform(state) > 0.5f;

            if (!key_before_y)
                break;

            entries[j + 1] = entries[j];
            j--;
        }
        entries[j + 1] = key;
    }
}
