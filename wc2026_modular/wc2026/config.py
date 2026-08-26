"""
config.py -- all tunable constants for the WC2026 simulator in one place.

Change N_RUNS, the CSV paths, or the extra-time/penalty tuning here.
Nothing else in the package should hardcode these values.
"""

# ============================================================
# >>> CHANGE THIS NUMBER TO RUN MORE OR FEWER TOURNAMENTS <<<
# ============================================================
N_RUNS = 10000

# Default CSV paths (overridable via --match-xg-csv / --groups-csv on the CLI)
MATCH_XG_CSV_PATH = "match_xg_2026.csv"
GROUPS_CSV_PATH = "teams_groups_2026.csv"

# Extra time is 1/3 the length of regulation, so goal rates scale down accordingly.
EXTRA_TIME_SCALE = 1.0 / 3.0

# Baseline penalty conversion rate, nudged by the match's own lambda gap.
PENALTY_BASE_RATE = 0.75
PENALTY_GAP_SENSITIVITY = 0.05
PENALTY_RATE_MIN = 0.4
PENALTY_RATE_MAX = 0.95
