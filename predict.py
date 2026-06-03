"""
NBA Finals Predictor — Elo baseline.

Usage:
    python predict.py                        # full series prediction
    python predict.py --game 1               # single game prediction
    python predict.py --refresh              # force-rebuild Elo from API
"""

import argparse

from src.elo import build_ratings, win_prob
from src.simulate import simulate_series

# ── 2025 Finals matchup ────────────────────────────────────────────────────────
# Knicks have home court (better record)
TEAM1 = "NYK"   # home court
TEAM2 = "SAS"

# NBA Finals 2-2-1-1-1 home schedule (True = TEAM1 at home)
HOME_SCHEDULE = [True, True, False, False, True, False, True]
GAME_LABELS = [
    "Game 1  (NYK home)",
    "Game 2  (NYK home)",
    "Game 3  (SAS home)",
    "Game 4  (SAS home)",
    "Game 5  (NYK home)",
    "Game 6  (SAS home)*",
    "Game 7  (NYK home)*",
]


def print_banner(text: str) -> None:
    print("\n" + "─" * 52)
    print(f"  {text}")
    print("─" * 52)


def run(game: int | None = None, refresh: bool = False) -> None:
    ratings = build_ratings(force=refresh)

    r1 = ratings.get(TEAM1, 1500)
    r2 = ratings.get(TEAM2, 1500)
    p_home = win_prob(TEAM1, TEAM2, ratings)   # TEAM1 at home
    p_away = win_prob(TEAM2, TEAM1, ratings)   # TEAM1 away = TEAM2 home prob flipped
    p_team1_away = 1 - p_away

    print_banner("ELO RATINGS")
    print(f"  {TEAM1}: {r1:.1f}")
    print(f"  {TEAM2}: {r2:.1f}")
    print(f"  Elo gap: {r1 - r2:+.1f} pts")

    if game is not None:
        idx = game - 1
        if not 0 <= idx < 7:
            print("Game number must be 1–7.")
            return
        is_home = HOME_SCHEDULE[idx]
        p = p_home if is_home else p_team1_away
        print_banner(f"{GAME_LABELS[idx]}")
        print(f"  {TEAM1} win probability : {p*100:.1f}%")
        print(f"  {TEAM2} win probability : {(1-p)*100:.1f}%")
        return

    # Per-game probabilities
    print_banner("PER-GAME WIN PROBABILITIES (NYK)")
    for i, (label, is_home) in enumerate(zip(GAME_LABELS, HOME_SCHEDULE)):
        p = p_home if is_home else p_team1_away
        bar = "█" * int(p * 20)
        print(f"  {label:<26} {p*100:5.1f}%  {bar}")

    # Series simulation
    result = simulate_series(p_home, p_team1_away, n=100_000)

    print_banner("SERIES PREDICTION  (100k simulations)")
    print(f"  {TEAM1} win series : {result.team1_win_pct*100:.1f}%")
    print(f"  {TEAM2} win series : {result.team2_win_pct*100:.1f}%")
    print(f"  Expected games   : {result.avg_games:.2f}")
    print()
    print("  Series length distribution:")
    for g, pct in result.game_dist.items():
        bar = "█" * int(pct * 40)
        print(f"    {g} games  {pct*100:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Finals predictor (Elo baseline)")
    parser.add_argument("--game", type=int, help="Predict a specific game (1–7)")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh Elo from API")
    args = parser.parse_args()
    run(game=args.game, refresh=args.refresh)
