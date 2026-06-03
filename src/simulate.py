"""
Series simulator.

Given a win probability for the home team in a single game,
Monte Carlo-simulates a best-of-7 series and returns series win
probabilities and expected game count distributions.
"""

import random
from collections import Counter
from typing import NamedTuple

# NBA Finals home-court schedule: 2-2-1-1-1
# True = home team (higher seed) has home court
HOME_SCHEDULE = [True, True, False, False, True, False, True]


class SeriesResult(NamedTuple):
    team1_win_pct: float       # fraction of simulations team1 wins the series
    team2_win_pct: float
    avg_games: float
    game_dist: dict[int, float]  # {4: 0.21, 5: 0.28, 6: 0.27, 7: 0.24}


def simulate_series(
    p_team1_home: float,
    p_team1_away: float,
    n: int = 100_000,
) -> SeriesResult:
    """
    Simulate a best-of-7 series n times.

    Args:
        p_team1_home: team1's win probability when playing at home.
        p_team1_away: team1's win probability when playing away.
        n: number of Monte Carlo iterations.
    """
    team1_wins = 0
    game_counts: Counter = Counter()

    for _ in range(n):
        w1 = w2 = 0
        for game_idx in range(7):
            is_home_for_team1 = HOME_SCHEDULE[game_idx]
            p = p_team1_home if is_home_for_team1 else p_team1_away
            if random.random() < p:
                w1 += 1
            else:
                w2 += 1
            if w1 == 4 or w2 == 4:
                game_counts[w1 + w2] += 1
                break
        if w1 == 4:
            team1_wins += 1

    total = sum(game_counts.values())
    return SeriesResult(
        team1_win_pct=team1_wins / n,
        team2_win_pct=1 - team1_wins / n,
        avg_games=sum(g * c for g, c in game_counts.items()) / total,
        game_dist={g: c / total for g, c in sorted(game_counts.items())},
    )


def series_from_single_prob(p_home: float, n: int = 100_000) -> SeriesResult:
    """
    Convenience wrapper when you only have one win probability
    (assumes team1 always has home court advantage baked in externally,
    so we derive away prob by removing the HCA effect).
    """
    # Rough HCA removal: flip the advantage for away games
    # home advantage ~ 3-4 pts => ~0.06 in probability terms
    hca_delta = 0.06
    p_away = max(0.01, min(0.99, p_home - hca_delta * 2))
    return simulate_series(p_home, p_away, n)
