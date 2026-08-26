"""
group_stage.py -- plays one group's round robin and ranks the 4 teams using
real FIFA tiebreakers: Points -> Goal Difference -> Goals For -> Head-to-head
result -> random draw of lots as the final fallback.
"""

import functools

from match_engine import play_regulation


def _new_row():
    return {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}


def play_group(match_xg, rng, teams, verbose):
    """Plays every match in a 4-team round robin.
    Returns (ranked_teams, table) where table[team] holds W/D/L/GF/GA/Pts."""
    table = {t: _new_row() for t in teams}
    h2h = {}  # (teamA, teamB) -> (golsFor, golsAgainst) from teamA's perspective

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            ga, gb = play_regulation(match_xg, rng, a, b)
            if verbose:
                print(f"    {a} {ga}-{gb} {b}")

            table[a]["P"] += 1
            table[b]["P"] += 1
            table[a]["GF"] += ga
            table[a]["GA"] += gb
            table[b]["GF"] += gb
            table[b]["GA"] += ga

            if ga > gb:
                table[a]["W"] += 1
                table[a]["Pts"] += 3
                table[b]["L"] += 1
            elif gb > ga:
                table[b]["W"] += 1
                table[b]["Pts"] += 3
                table[a]["L"] += 1
            else:
                table[a]["D"] += 1
                table[b]["D"] += 1
                table[a]["Pts"] += 1
                table[b]["Pts"] += 1

            h2h[(a, b)] = (ga, gb)
            h2h[(b, a)] = (gb, ga)

    rnd = {t: rng.random() for t in teams}

    def sort_key_cmp(x, y):
        rx, ry = table[x], table[y]
        if rx["Pts"] != ry["Pts"]:
            return ry["Pts"] - rx["Pts"]
        gdx, gdy = rx["GF"] - rx["GA"], ry["GF"] - ry["GA"]
        if gdx != gdy:
            return gdy - gdx
        if rx["GF"] != ry["GF"]:
            return ry["GF"] - rx["GF"]
        if (x, y) in h2h:
            diff = h2h[(x, y)][0] - h2h[(x, y)][1]
            if diff != 0:
                return -diff  # x's h2h goal diff positive => x ranks first
        return -1 if rnd[x] > rnd[y] else 1

    ranked = sorted(teams, key=functools.cmp_to_key(sort_key_cmp))
    return ranked, table


def rank_third_place_teams(thirds, rng):
    """thirds: list of dicts {group, team, row}. Ranks them the same way
    group standings are ranked (Pts -> GD -> GF -> random), since third-place
    teams from different groups never played each other so no head-to-head
    is possible. Returns the sorted list."""
    def cmp(x, y):
        rx, ry = x["row"], y["row"]
        if rx["Pts"] != ry["Pts"]:
            return ry["Pts"] - rx["Pts"]
        gdx, gdy = rx["GF"] - rx["GA"], ry["GF"] - ry["GA"]
        if gdx != gdy:
            return gdy - gdx
        if rx["GF"] != ry["GF"]:
            return ry["GF"] - rx["GF"]
        return -1 if rng.random() > 0.5 else 1

    return sorted(thirds, key=functools.cmp_to_key(cmp))
