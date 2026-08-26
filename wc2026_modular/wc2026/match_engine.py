"""
match_engine.py -- everything about simulating ONE match.

    expected_goals(match_xg, home, away)      -> (home_lambda, away_lambda)
    poisson_sample(rng, lam)                   -> random goal count
    play_regulation(match_xg, rng, home, away) -> (gols_home, gols_away)
    play_knockout_match(match_xg, rng, a, b, verbose) -> result dict, handles
        regulation -> extra time -> penalties automatically.

This module knows nothing about groups, brackets, or tournaments -- it only
simulates individual matches given a pre-computed lambda pair.
"""

import sys

from config import (
    EXTRA_TIME_SCALE,
    PENALTY_BASE_RATE,
    PENALTY_GAP_SENSITIVITY,
    PENALTY_RATE_MIN,
    PENALTY_RATE_MAX,
)


def expected_goals(match_xg, home, away): #extracts lambda for both teams
    """Look up the pre-computed match-specific lambda pair.
    `home` and `away` are the two sides for this lookup."""
    if (home, away) in match_xg:
        return match_xg[(home, away)]
    if (away, home) in match_xg:
        # Only the swapped orientation is available -- invert which lambda
        # belongs to which side.
        away_xg, home_xg = match_xg[(away, home)]
        return (home_xg, away_xg)
    print(f"WARNING: no match-xG data for '{home}' vs '{away}', using default 1.0/1.0", file=sys.stderr)
    return (1.0, 1.0)


def poisson_sample(rng, lam):
    """Knuth's algorithm -- avoids needing numpy."""
    l = pow(2.718281828, -lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1


def play_regulation(match_xg, rng, home, away):
    lam_home, lam_away = expected_goals(match_xg, home, away)
    return poisson_sample(rng, lam_home), poisson_sample(rng, lam_away)


def simulate_penalty_shootout(match_xg, rng, home, away):
    """Best-of-5 then sudden death. Returns (winner_index, pen_home, pen_away)
    where winner_index is 0 for home, 1 for away."""
    lam_home, lam_away = expected_goals(match_xg, home, away)
    gap = lam_home - lam_away
    rate_home = min(max(PENALTY_BASE_RATE + gap * PENALTY_GAP_SENSITIVITY, PENALTY_RATE_MIN), PENALTY_RATE_MAX)
    rate_away = min(max(PENALTY_BASE_RATE - gap * PENALTY_GAP_SENSITIVITY, PENALTY_RATE_MIN), PENALTY_RATE_MAX)

    pen_home, pen_away = 0, 0
    for _ in range(5):
        if rng.random() < rate_home:
            pen_home += 1
        if rng.random() < rate_away:
            pen_away += 1

    while pen_home == pen_away:
        if rng.random() < rate_home:
            pen_home += 1
        if rng.random() < rate_away:
            pen_away += 1

    return (0 if pen_home > pen_away else 1), pen_home, pen_away


def play_knockout_match(match_xg, rng, home, away, verbose):
    """Plays regulation, then extra time if level, then penalties if still
    level. Returns a result dict with the winner and how it was decided."""
    gols_home, gols_away = play_regulation(match_xg, rng, home, away)

    if gols_home != gols_away:
        winner = home if gols_home > gols_away else away
        if verbose:
            print(f"    {home} {gols_home}-{gols_away} {away}  -> {winner} advances")
        return {
            "winner": winner, "gols_home": gols_home, "gols_away": gols_away,
            "went_to_et": False, "went_to_pens": False, "pen_home": 0, "pen_away": 0,
        }

    # Level after 90 -- extra time
    lam_home, lam_away = expected_goals(match_xg, home, away)
    et_home = poisson_sample(rng, lam_home * EXTRA_TIME_SCALE)
    et_away = poisson_sample(rng, lam_away * EXTRA_TIME_SCALE)
    gols_home += et_home
    gols_away += et_away

    if gols_home != gols_away:
        winner = home if gols_home > gols_away else away
        if verbose:
            print(f"    {home} {gols_home}-{gols_away} {away} (AET)  -> {winner} advances")
        return {
            "winner": winner, "gols_home": gols_home, "gols_away": gols_away,
            "went_to_et": True, "went_to_pens": False, "pen_home": 0, "pen_away": 0,
        }

    # Still level -- penalties
    who, pen_home, pen_away = simulate_penalty_shootout(match_xg, rng, home, away)
    winner = home if who == 0 else away
    if verbose:
        print(f"    {home} {gols_home}-{gols_away} {away} (AET, {winner} win {pen_home}-{pen_away} on penalties)  -> {winner} advances")
    return {
        "winner": winner, "gols_home": gols_home, "gols_away": gols_away,
        "went_to_et": True, "went_to_pens": True, "pen_home": pen_home, "pen_away": pen_away,
    }
