"""
csv_loader.py -- reads and validates the two input CSVs.

    load_match_xg_csv(path)  -> dict[(home, away)] = (home_xg, away_xg)
    load_groups_csv(path)    -> list[(group_label, [team, team, team, team])]
    validate_all_pairs_present(match_xg, all_teams) -> bool

Nothing here runs a simulation -- this module's only job is turning CSV rows
into clean Python data structures, with clear errors if something's wrong.
"""

import csv
import sys
from collections import defaultdict


def load_match_xg_csv(path):
    """Load MatchID,HomeTeam,AwayTeam,HomeXG,AwayXG rows.
    Returns a dict keyed by (home_team, away_team) -> (home_xg, away_xg),
    or None if the file couldn't be loaded."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        print(f"ERROR: could not open CSV file '{path}'", file=sys.stderr)
        return None

    if not rows:
        print(f"ERROR: '{path}' is empty", file=sys.stderr)
        return None

    start = 0
    header = " ".join(c.strip().lower() for c in rows[0])
    if "home" in header and "away" in header:
        start = 1

    match_xg = {}
    loaded = 0
    for line_num, row in enumerate(rows[start:], start=start + 1):
        if not row or all(c.strip() == "" for c in row):
            continue
        if len(row) < 5:
            print(f"WARNING: line {line_num} - not enough columns, skipping", file=sys.stderr)
            continue
        _match_id, home, away, home_xg_str, away_xg_str = row[0], row[1], row[2], row[3], row[4]
        home, away = home.strip(), away.strip()
        if not home or not away:
            continue
        try:
            home_xg = float(home_xg_str.strip())
            away_xg = float(away_xg_str.strip())
        except ValueError:
            print(f"WARNING: line {line_num} - bad XG values for '{home}' vs '{away}', skipping", file=sys.stderr)
            continue
        home_xg = max(home_xg, 0.05)
        away_xg = max(away_xg, 0.05)
        match_xg[(home, away)] = (home_xg, away_xg)
        loaded += 1

    print(f"Loaded {loaded} match-xG rows.")
    return match_xg


def load_groups_csv(path):
    """Load Team,Group rows. Returns a list of (group_label, [4 teams]),
    sorted by group label, or None if the file couldn't be loaded / is invalid."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        print(f"ERROR: could not open CSV file '{path}'", file=sys.stderr)
        return None

    if not rows:
        print(f"ERROR: '{path}' is empty", file=sys.stderr)
        return None

    start = 0
    header = " ".join(c.strip().lower() for c in rows[0])
    if "team" in header and "group" in header:
        start = 1

    by_group = defaultdict(list)
    for row in rows[start:]:
        if not row or all(c.strip() == "" for c in row):
            continue
        if len(row) < 2:
            continue
        team, group = row[0].strip(), row[1].strip()
        if not team or not group:
            continue
        by_group[group].append(team)

    groups = []
    for label in sorted(by_group.keys()):
        teams = by_group[label]
        if len(teams) != 4:
            print(f"ERROR: Group {label} has {len(teams)} teams (expected 4).", file=sys.stderr)
            return None
        groups.append((label, teams))

    if len(groups) != 12:
        print(f"ERROR: found {len(groups)} groups (expected 12).", file=sys.stderr)
        return None

    return groups


def flatten_teams(groups):
    """groups: list of (label, [teams]) -> flat list of all teams."""
    return [t for _, teams in groups for t in teams]


def validate_all_pairs_present(match_xg, all_teams):
    """Every possible ordered pair among the 48 tournament teams must have a
    row in match_xg, since any two of them could face each other in the
    knockout stage once earlier rounds are resolved."""
    ok = True
    missing_count = 0
    MAX_REPORTED = 20
    for home in all_teams:
        for away in all_teams:
            if home == away:
                continue
            if (home, away) not in match_xg:
                if missing_count < MAX_REPORTED:
                    print(f"ERROR: missing match-xG row for '{home}' (home) vs '{away}' (away).", file=sys.stderr)
                missing_count += 1
                ok = False
    if missing_count > MAX_REPORTED:
        print(f"...and {missing_count - MAX_REPORTED} more missing pairs (only first {MAX_REPORTED} shown).", file=sys.stderr)
    return ok
