#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int WC_NUM_TEAMS = 48;
constexpr int WC_NUM_GROUPS = 12;
constexpr int WC_GROUP_SIZE = 4;
constexpr int WC_NUM_FIXTURES = 72;
constexpr int WC_NUM_R32_MATCHES = 16;
constexpr int WC_NUM_STAGES = 7;

// Tournament stages
enum WCStage {
    STAGE_GROUP = 0,
    STAGE_R32 = 1,
    STAGE_R16 = 2,
    STAGE_QF = 3,
    STAGE_SF = 4,
    STAGE_FINAL = 5,
    STAGE_CHAMPION = 6
};

// Basic team information used on GPU
struct TeamGPU {
    int id;
    int group_id;
};

// One match fixture
struct FixtureGPU {
    int home_idx;
    int away_idx;
};

// Match/tournament statistics
struct StatsGPU {
    int P;
    int W;
    int D;
    int L;

    int GF;
    int GA;
    int GD;

    int Pts;
};
