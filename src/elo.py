"""
Elo rating engine — Level 1 (smart Elo).

Improvements over baseline:
  1. Recency-weighted K-factor  — games from 3+ months ago count less
  2. Margin of victory multiplier — blowouts shift ratings more than 1-pt wins
  3. Per-team home court advantage — calibrated from each team's home/away record
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_ELO   = 1500
K_REGULAR     = 20
K_PLAYOFF     = 15
DEFAULT_HCA   = 100   # Elo points, used when per-team HCA isn't available

ELO_CACHE     = Path(__file__).parent.parent / "data" / "elo_ratings.json"
HCA_CACHE     = Path(__file__).parent.parent / "data" / "hca_ratings.json"

# Recency decay: a game played HALF_LIFE days ago counts at 50% weight.
# 120 days ≈ 4 months — early season games matter half as much as recent ones.
RECENCY_HALF_LIFE = 120


# ── Core Elo math ──────────────────────────────────────────────────────────────

def expected(rating_a: float, rating_b: float) -> float:
    """Expected win probability for team A against team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update(rating: float, actual: float, expected_: float, k: float) -> float:
    """Standard Elo update."""
    return rating + k * (actual - expected_)


# ── Level 1 helpers ────────────────────────────────────────────────────────────

def recency_weight(game_date: str, latest_date: str) -> float:
    """
    Exponential decay weight in [0, 1] based on how long ago the game was played.
    A game played RECENCY_HALF_LIFE days before the latest game gets weight 0.5.
    """
    d_game  = datetime.strptime(game_date,  "%Y-%m-%d")
    d_last  = datetime.strptime(latest_date, "%Y-%m-%d")
    days_ago = (d_last - d_game).days
    return math.exp(-math.log(2) * days_ago / RECENCY_HALF_LIFE)


def mov_multiplier(margin: float, elo_diff: float) -> float:
    """
    Margin-of-victory multiplier (based on FiveThirtyEight's formula).
    Larger wins shift ratings more, but autocorrelation correction
    prevents good teams from being over-rewarded for beating weak ones badly.

    margin   : winning team's point margin (always positive)
    elo_diff : winner's pre-game Elo minus loser's pre-game Elo
    """
    # Autocorrelation correction — big favourites get diminishing returns
    autocorr = 2.2 / (elo_diff * 0.001 + 2.2)
    return math.log(abs(margin) + 1) * autocorr


def build_hca(df: pd.DataFrame) -> dict[str, float]:
    """
    Estimate each team's home court advantage in Elo points from
    their home vs away win rates this season.

    League average HCA ≈ 100 Elo pts → ~57% home win rate.
    We scale each team's observed home win rate to that same Elo space.
    """
    hca: dict[str, float] = {}
    for team, group in df.groupby("TEAM_ABBREVIATION"):
        home_games = group[~group["MATCHUP"].str.contains("@")]
        away_games = group[group["MATCHUP"].str.contains("@")]
        home_wr = (home_games["WL"] == "W").mean() if len(home_games) else 0.5
        away_wr = (away_games["WL"] == "W").mean() if len(away_games) else 0.5
        # Convert win-rate gap to Elo points (400 * log10 scale)
        # Clamp to avoid log(0) — floor/ceil at 5%/95% win rates
        home_wr = max(0.05, min(0.95, home_wr))
        away_wr = max(0.05, min(0.95, away_wr))
        elo_home = -400 * math.log10(1 / home_wr - 1)
        elo_away = -400 * math.log10(1 / away_wr - 1)
        hca[team] = max(0.0, (elo_home - elo_away) / 2)
    return hca


# ── Rating builder ─────────────────────────────────────────────────────────────

def build_ratings(season: str = "2024-25", force: bool = False) -> tuple[dict[str, float], dict[str, float]]:
    """
    Fetch every game for *season*, replay them in chronological order using
    Level-1 smart Elo, and return (elo_ratings, hca_per_team).

    Both dicts are cached to data/ so repeated calls are instant.
    """
    if ELO_CACHE.exists() and HCA_CACHE.exists() and not force:
        with open(ELO_CACHE) as f:
            elo = json.load(f)
        with open(HCA_CACHE) as f:
            hca = json.load(f)
        return elo, hca

    print(f"Fetching game log for {season} …")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]

    df = df.sort_values("GAME_DATE")
    df["IS_HOME"]    = ~df["MATCHUP"].str.contains("@")
    df["IS_PLAYOFF"] = df["SEASON_ID"].astype(str).str.startswith("42")

    latest_date = df["GAME_DATE"].max()

    # Build per-team HCA from the full season record
    hca = build_hca(df)

    home = df[df["IS_HOME"]].copy()
    away = df[~df["IS_HOME"]].copy()

    games = home[["GAME_ID", "TEAM_ABBREVIATION", "WL", "PLUS_MINUS", "GAME_DATE", "IS_PLAYOFF"]].merge(
        away[["GAME_ID", "TEAM_ABBREVIATION", "WL"]],
        on="GAME_ID",
        suffixes=("_home", "_away"),
    )

    ratings: dict[str, float] = {}

    for _, row in games.iterrows():
        h = row["TEAM_ABBREVIATION_home"]
        a = row["TEAM_ABBREVIATION_away"]
        ratings.setdefault(h, INITIAL_ELO)
        ratings.setdefault(a, INITIAL_ELO)

        # 1. Recency weight — scale K by how recent this game is
        rw = recency_weight(row["GAME_DATE"], latest_date)
        k  = (K_PLAYOFF if row["IS_PLAYOFF"] else K_REGULAR) * rw

        # 2. Per-team HCA for expected probability
        team_hca = (hca.get(h, DEFAULT_HCA) + hca.get(a, DEFAULT_HCA)) / 2
        e_h = expected(ratings[h] + team_hca, ratings[a])
        e_a = 1.0 - e_h

        home_won = row["WL_home"] == "W"
        margin   = abs(row["PLUS_MINUS"])

        # 3. Margin-of-victory multiplier
        winner_elo = ratings[h] if home_won else ratings[a]
        loser_elo  = ratings[a] if home_won else ratings[h]
        mov = mov_multiplier(margin, winner_elo - loser_elo)

        ratings[h] = update(ratings[h], 1.0 if home_won else 0.0, e_h, k * mov)
        ratings[a] = update(ratings[a], 0.0 if home_won else 1.0, e_a, k * mov)

    ELO_CACHE.parent.mkdir(exist_ok=True)
    with open(ELO_CACHE, "w") as f:
        json.dump(ratings, f, indent=2)
    with open(HCA_CACHE, "w") as f:
        json.dump(hca, f, indent=2)

    print(f"Ratings built for {len(ratings)} teams. Cached to {ELO_CACHE}")
    return ratings, hca


# ── Prediction helpers ─────────────────────────────────────────────────────────

def win_prob(
    home_team: str,
    away_team: str,
    ratings: Optional[dict[str, float]] = None,
    hca: Optional[dict[str, float]] = None,
) -> float:
    """Win probability for the home team (0–1), using per-team HCA if available."""
    if ratings is None or hca is None:
        ratings, hca = build_ratings()
    r_h = ratings.get(home_team, INITIAL_ELO)
    r_a = ratings.get(away_team, INITIAL_ELO)
    team_hca = (hca.get(home_team, DEFAULT_HCA) + hca.get(away_team, DEFAULT_HCA)) / 2
    return expected(r_h + team_hca, r_a)
