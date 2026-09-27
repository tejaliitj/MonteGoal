/*
 * config.cuh -- all tunable constants for the WC2026 GPU simulator in one
 * place. Direct translation of config.py: change N_RUNS or the CSV paths
 * here, nothing else in the project should hardcode these values.
 */
#ifndef CONFIG_CUH
#define CONFIG_CUH

#define NUM_TEAMS       48
#define NUM_GROUPS      12
#define TEAMS_PER_GROUP 4

// ============================================================
// >>> CHANGE THIS NUMBER TO RUN MORE OR FEWER TOURNAMENTS <<<
// One thread = one full tournament, so this is also the thread count.
// ============================================================
#define N_RUNS 10000

#define THREADS_PER_BLOCK 256

// Default data paths (same role as config.py's MATCH_XG_CSV_PATH / GROUPS_CSV_PATH)
#define LAMBDA_MATRIX_CSV_PATH "data/lambda_matrix.csv"
#define GROUPS_CSV_PATH        "data/teams_groups_2026.csv"

// Extra time is 1/3 the length of regulation, so goal rates scale down accordingly.
#define EXTRA_TIME_SCALE (1.0f / 3.0f)

// Baseline penalty conversion rate, nudged by the match's own lambda gap.
#define PENALTY_BASE_RATE         0.75f
#define PENALTY_GAP_SENSITIVITY   0.05f
#define PENALTY_RATE_MIN          0.4f
#define PENALTY_RATE_MAX          0.95f

#endif
