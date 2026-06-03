"""
NBA Finals Predictor — Level 1 + Level 2 + Level 3.

Level 1 (smart Elo):
  - Recency-weighted K-factor
  - Margin-of-victory multiplier
  - Per-team home court advantage

Level 2 (team context):
  - Blended net rating (65% last-15 games, 35% full season)
  - Rest days & back-to-back flag
  - Pace differential

Level 3 (lineup & injuries):
  - Per-player impact penalty for injuries (scaled by status severity)
  - Wembanyama defensive gravity adjustment (rim deterrence above net rating)

Usage:
    python predict.py                        # full series prediction
    python predict.py --game 1               # single game prediction
    python predict.py --refresh              # force-rebuild all data from API
"""

import argparse

from src.elo import build_ratings, win_prob
from src.features import build_features
from src.injuries import compute_injury_adjustments
from src.matchup import compute_matchup_adjustments
from src.simulate import simulate_series

# ── 2025-26 Finals matchup ─────────────────────────────────────────────────────
# Spurs have home court — they had the better record and eliminated OKC (top West seed)
TEAM1       = "SAS"   # home court
TEAM2       = "NYK"
FINALS_DATE = "2026-06-04"   # Game 1 tip-off date

# NBA Finals 2-2-1-1-1 home schedule (True = TEAM1 / SAS at home)
HOME_SCHEDULE = [True, True, False, False, True, False, True]
GAME_LABELS = [
    "Game 1  (SAS home)",
    "Game 2  (SAS home)",
    "Game 3  (NYK home)",
    "Game 4  (NYK home)",
    "Game 5  (SAS home)",
    "Game 6  (NYK home)*",
    "Game 7  (SAS home)*",
]


def print_banner(text: str) -> None:
    print("\n" + "─" * 56)
    print(f"  {text}")
    print("─" * 56)


def net_rating_adjustment(blended_delta: float) -> float:
    """
    Convert blended net rating delta into an Elo-point adjustment.

    Calibration: 1 pt of net rating ≈ 8 Elo pts in a playoff context.
    Full-season conversion (FiveThirtyEight) is ~25 pts, but in a Finals
    series both teams are elite and the sample is small — dampening prevents
    recent-form noise from overwhelming Elo signal.
    Capped at ±50 so no single feature dominates.
    """
    return max(-50.0, min(50.0, blended_delta * 8))


def rest_adjustment(rest_delta: int, t1_b2b: bool, t2_b2b: bool) -> float:
    """
    Elo-point adjustment for rest advantage.
    Each extra rest day ≈ +10 Elo pts, capped at ±40.
    B2B team loses ~30 Elo pts on top of rest adjustment.
    """
    adj = max(-40.0, min(40.0, rest_delta * 10.0))
    if t2_b2b:
        adj += 30.0   # opponent is on b2b, good for t1
    if t1_b2b:
        adj -= 30.0   # we're on b2b, bad for t1
    return adj


def run(game: int | None = None, refresh: bool = False, season: str = "2025-26") -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    ratings, hca = build_ratings(season=season, force=refresh)
    feats = build_features(TEAM1, TEAM2, FINALS_DATE, season=season, elo_ratings=ratings, force=refresh)

    r1 = ratings.get(TEAM1, 1500)
    r2 = ratings.get(TEAM2, 1500)

    # ── Level 2 adjustments ────────────────────────────────────────────────────
    nr_adj   = net_rating_adjustment(feats["blended_delta"])
    rest_adj = rest_adjustment(feats["rest_delta"], feats["t1_b2b"], feats["t2_b2b"])

    # ── Level 3 adjustments ────────────────────────────────────────────────────
    inj = compute_injury_adjustments(TEAM1, TEAM2)
    inj_adj = inj["net_adj"]

    # ── Level 4 adjustments ────────────────────────────────────────────────────
    matchup = compute_matchup_adjustments(
        TEAM1, TEAM2, r1, r2, season=season,
        robinson_questionable=(inj["team1_penalty"] > 0),
    )
    matchup_adj = matchup["net_adj"]

    total_adj = nr_adj + rest_adj + inj_adj + matchup_adj

    # Apply total adjustment — positive = benefits TEAM1 (home)
    p_home       = win_prob(TEAM1, TEAM2, ratings, hca, extra_home_adj=total_adj)
    p_team1_away = win_prob(TEAM1, TEAM2, ratings, hca, extra_home_adj=total_adj - 2 * hca.get(TEAM1, 59))

    # ── Print: Elo ─────────────────────────────────────────────────────────────
    print_banner("ELO RATINGS  (Level 1 — smart Elo)")
    print(f"  {TEAM1}: {r1:.1f}  (HCA: +{hca.get(TEAM1, 100):.0f} pts)")
    print(f"  {TEAM2}: {r2:.1f}  (HCA: +{hca.get(TEAM2, 100):.0f} pts)")
    print(f"  Raw Elo gap: {r1 - r2:+.1f} pts")

    # ── Print: Level 2 context ─────────────────────────────────────────────────
    print_banner("TEAM CONTEXT  (Level 2)")
    print(f"  {'':30} {TEAM1:>8} {TEAM2:>8}")
    print(f"  {'Season net rating':30} {feats['t1_net_rating']:>8.1f} {feats['t2_net_rating']:>8.1f}")
    print(f"  {'Last-15 avg margin':30} {feats['t1_last15_margin']:>8.1f} {feats['t2_last15_margin']:>8.1f}")
    print(f"  {'Blended net rating':30} {feats['t1_blended']:>8.1f} {feats['t2_blended']:>8.1f}")
    print(f"  {'Pace':30} {feats['t1_pace']:>8.1f} {feats['t2_pace']:>8.1f}")
    print(f"  {'Rest days (before G1)':30} {feats['t1_rest_days']:>8} {feats['t2_rest_days']:>8}")
    print(f"  {'Back-to-back':30} {'YES' if feats['t1_b2b'] else 'no':>8} {'YES' if feats['t2_b2b'] else 'no':>8}")
    print()
    print(f"  Blended net rating adj : {nr_adj:+.1f} Elo pts  (favours {TEAM1 if nr_adj > 0 else TEAM2})")
    print(f"  Rest adj               : {rest_adj:+.1f} Elo pts")

    print_banner("INJURY REPORT  (Level 3)")
    for line in inj["breakdown"]:
        print(line)
    print()
    print(f"  {TEAM1} injury penalty  : -{inj['team1_penalty']:.0f} Elo pts")
    print(f"  {TEAM2} injury penalty  : -{inj['team2_penalty']:.0f} Elo pts")
    print(f"  Wemby gravity adj      : -{inj['wemby_gravity']:.0f} Elo pts to {TEAM2}")
    print(f"  Net injury adj ({TEAM1}) : {inj_adj:+.1f} Elo pts")
    print()
    print_banner("MATCHUP FACTORS  (Level 4)")
    for factor, desc in matchup["breakdown"].items():
        print(f"  {factor:<18} {desc}")
    print(f"\n  Net matchup adj ({TEAM1}) : {matchup_adj:+.1f} Elo pts")
    print(f"\n  ── Total adjustment (L2 + L3 + L4): {total_adj:+.1f} Elo pts ──")

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

    # ── Print: per-game probabilities ──────────────────────────────────────────
    print_banner(f"PER-GAME WIN PROBABILITIES ({TEAM1})")
    for label, is_home in zip(GAME_LABELS, HOME_SCHEDULE):
        p = p_home if is_home else p_team1_away
        bar = "█" * int(p * 20)
        print(f"  {label:<26} {p*100:5.1f}%  {bar}")

    # ── Print: series simulation ───────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="NBA Finals predictor")
    parser.add_argument("--game",    type=int,          help="Predict a specific game (1–7)")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh all data from API")
    parser.add_argument("--season",  default="2025-26", help="NBA season (default: 2025-26)")
    args = parser.parse_args()
    run(game=args.game, refresh=args.refresh, season=args.season)
