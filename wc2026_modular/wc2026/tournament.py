"""
tournament.py -- orchestrates ONE full 48-team World Cup: group stage,
best-third-place selection, the real R32 bracket, and every knockout round
through to the final. This is the only module that ties the others together;
it contains no simulation math of its own.
"""

from group_stage import play_group, rank_third_place_teams
from bracket import (
    R32_FIXTURES, R16_PAIRS, QF_PAIRS, SF_PAIRS,
    assign_third_place_slots, resolve_role,
)
from match_engine import play_knockout_match


def run_tournament(match_xg, groups, rng, verbose, stats):
    """Plays one full tournament. `stats` is a dict[team] -> {semis, finals,
    champs} that gets updated in place across possibly many calls (for the
    multi-run summary). Returns the champion's name."""
    if verbose:
        print("=" * 60)
        print(" FIFA WORLD CUP 2026 -- CPU SIMULATION")
        print("=" * 60)

    group_winner = {}
    group_runner_up = {}
    thirds = []  # list of dicts: group, team, row

    if verbose:
        print("\n--- GROUP STAGE ---")
    for label, teams in groups:
        if verbose:
            print(f"\nGroup {label}: {', '.join(teams)}")
        ranked, table = play_group(match_xg, rng, teams, verbose)
        if verbose:
            for pos, t in enumerate(ranked, start=1):
                r = table[t]
                gd = r["GF"] - r["GA"]
                print(f"   {pos}. {t:<24} Pts {r['Pts']}  GD {gd:+d}  ({r['W']}W {r['D']}D {r['L']}L)")
        group_winner[label] = ranked[0]
        group_runner_up[label] = ranked[1]
        thirds.append({"group": label, "team": ranked[2], "row": table[ranked[2]]})

    thirds_sorted = rank_third_place_teams(thirds, rng)

    if verbose:
        print("\n--- BEST THIRD-PLACED TEAMS (advancing) ---")
    qualifying_third_groups = []
    third_team_by_group = {}
    for e in thirds_sorted[:8]:
        if verbose:
            gd = e["row"]["GF"] - e["row"]["GA"]
            print(f"   Group {e['group']}: {e['team']} (Pts {e['row']['Pts']}, GD {gd:+d})")
        qualifying_third_groups.append(e["group"])
        third_team_by_group[e["group"]] = e["team"]

    slot_to_third_group = assign_third_place_slots(qualifying_third_groups)

    r32_pairs = [
        (
            resolve_role(role_a, group_winner, group_runner_up, third_team_by_group, slot_to_third_group),
            resolve_role(role_b, group_winner, group_runner_up, third_team_by_group, slot_to_third_group),
        )
        for role_a, role_b in R32_FIXTURES
    ]

    if verbose:
        print("\n=== ROUND OF 32 ===")
    r32_winners = [play_knockout_match(match_xg, rng, a, b, verbose)["winner"] for a, b in r32_pairs]

    if verbose:
        print("\n=== ROUND OF 16 ===")
    r16_winners = []
    for i, j in R16_PAIRS:
        res = play_knockout_match(match_xg, rng, r32_winners[i], r32_winners[j], verbose)
        r16_winners.append(res["winner"])

    if verbose:
        print("\n=== QUARTERFINALS ===")
    qf_winners = []
    for i, j in QF_PAIRS:
        res = play_knockout_match(match_xg, rng, r16_winners[i], r16_winners[j], verbose)
        qf_winners.append(res["winner"])
    for w in qf_winners:
        stats[w]["semis"] += 1

    if verbose:
        print("\n=== SEMIFINALS ===")
    finalists = []
    sf_pairs = []
    for i, j in SF_PAIRS:
        a, b = qf_winners[i], qf_winners[j]
        sf_pairs.append((a, b))
        res = play_knockout_match(match_xg, rng, a, b, verbose)
        finalists.append(res["winner"])
    for f in finalists:
        stats[f]["finals"] += 1

    losers_sf = []
    for idx, (a, b) in enumerate(sf_pairs):
        losers_sf.append(b if finalists[idx] == a else a)

    if verbose:
        print("\n=== THIRD-PLACE PLAY-OFF ===")
    third_res = play_knockout_match(match_xg, rng, losers_sf[0], losers_sf[1], verbose)

    if verbose:
        print("\n=== FINAL ===")
    final_res = play_knockout_match(match_xg, rng, finalists[0], finalists[1], verbose)
    champion = final_res["winner"]
    runner_up = finalists[1] if champion == finalists[0] else finalists[0]

    stats[champion]["champs"] += 1

    if verbose:
        print("\n" + "=" * 60)
        print(f" CHAMPION: {champion}")
        print(f" Runner-up: {runner_up}")
        print(f" Third place: {third_res['winner']}")
        print("=" * 60)

    return champion
