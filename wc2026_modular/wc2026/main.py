"""
main.py -- CLI entry point. Loads the two CSVs, runs N tournaments, prints
the championship-count and furthest-stage-reached summary.

USAGE
-----
    python3 main.py --seed 1000
    python3 main.py --seed 1000 --verbose
    python3 main.py --runs 50 --seed 1
    python3 main.py --match-xg-csv other.csv --groups-csv other_groups.csv
"""

import argparse
import random
import sys
from collections import defaultdict

import config
from csv_loader import load_match_xg_csv, load_groups_csv, flatten_teams, validate_all_pairs_present
from tournament import run_tournament

# this is basically helping read inputs from the terminal
def main():
    parser = argparse.ArgumentParser(description="Simulate the 2026 World Cup N times.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Start seed for run #1 (later runs use seed+1, seed+2, ...)")
    parser.add_argument("--verbose", action="store_true", help="Print every match of every run")
    parser.add_argument("--runs", type=int, default=None, help="How many tournaments to simulate (overrides config.N_RUNS)")
    parser.add_argument("--match-xg-csv", type=str, default=config.MATCH_XG_CSV_PATH,
                         help="Path to MatchID,HomeTeam,AwayTeam,HomeXG,AwayXG CSV")
    parser.add_argument("--groups-csv", type=str, default=config.GROUPS_CSV_PATH,
                         help="Path to Team,Group CSV")
    args = parser.parse_args()

    n_runs = args.runs if args.runs is not None else config.N_RUNS

    print(f"Loading groups from: {args.groups_csv}")
    groups = load_groups_csv(args.groups_csv)
    if groups is None:
        print("Failed to load groups CSV. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading match expected-goals pairs from: {args.match_xg_csv}")
    match_xg = load_match_xg_csv(args.match_xg_csv)
    if match_xg is None:
        print("Failed to load match-xG CSV. Exiting.", file=sys.stderr)
        sys.exit(1)

    all_teams = flatten_teams(groups) #get a list of all teams
    if not validate_all_pairs_present(match_xg, all_teams):
        print("Some team-pair combinations missing from match-xG CSV. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_teams)} teams / {len(groups)} groups, all {len(match_xg)} possible pairings covered.\n")

    rng = random.Random()
    champions = []
    stats = defaultdict(lambda: {"semis": 0, "finals": 0, "champs": 0})

    for run in range(n_runs):
        if args.seed is not None:
            rng.seed(args.seed + run)
        else:
            rng.seed()

        champ = run_tournament(match_xg, groups, rng, args.verbose, stats)
        champions.append(champ)

        if not args.verbose:
            if args.seed is not None:
                print(f"Run {run + 1} (seed {args.seed + run}): {champ}")
            else:
                print(f"Run {run + 1}: {champ}")

    champ_count = defaultdict(int)
    for c in champions:
        champ_count[c] += 1
    champ_ranked = sorted(champ_count.items(), key=lambda kv: -kv[1])

    print("\n" + "=" * 60)
    print(f" SUMMARY -- {n_runs} tournaments simulated")
    print("=" * 60)
    print("\nChampionship counts:")
    for team, count in champ_ranked:
        print(f"   {team:<24} {count}/{n_runs}")

    by_stage = [(team, sc) for team, sc in stats.items() if sc["semis"] > 0]
    by_stage.sort(key=lambda ts: -ts[1]["semis"])

    print("\nFurthest stage reached (teams with >=1 semifinal appearance):")
    print(f"{'Team':<24} {'SF':>6} {'Final':>6} {'Champ':>6}")
    for team, sc in by_stage:
        print(f"{team:<24} {sc['semis']:>6} {sc['finals']:>6} {sc['champs']:>6}")


if __name__ == "__main__":
    main()
