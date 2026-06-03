"""
Elo rating engine.

Builds team Elo ratings from the 2024-25 regular season + playoffs
and exposes win probability for any matchup.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_ELO = 1500
K_REGULAR = 20
K_PLAYOFF = 15
HOME_ADVANTAGE = 100   # Elo points added to home team's rating for prediction
ELO_CACHE = Path(__file__).parent.parent / "data" / "elo_ratings.json"

# ── Core Elo math ──────────────────────────────────────────────────────────────

def expected(rating_a: float, rating_b: float) -> float:
    """Expected win probability for team A against team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update(rating: float, actual: float, expected_: float, k: float) -> float:
    return rating + k * (actual - expected_)


# ── Rating builder ─────────────────────────────────────────────────────────────

def build_ratings(season: str = "2024-25", force: bool = False) -> dict[str, float]:
    """
    Fetch every game for *season*, replay them in chronological order,
    and return a dict of {team_abbreviation: elo_rating}.

    Results are cached to data/elo_ratings.json so repeated calls are instant.
    """
    if ELO_CACHE.exists() and not force:
        with open(ELO_CACHE) as f:
            return json.load(f)

    print(f"Fetching game log for {season} …")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",          # NBA
    )
    df = finder.get_data_frames()[0]

    # Keep only one row per game (home team perspective) to avoid double-counting
    # SEASON_ID prefix: "42" = playoffs, "22" = regular season
    df = df.sort_values("GAME_DATE")
    df["IS_HOME"] = ~df["MATCHUP"].str.contains("@")
    df["IS_PLAYOFF"] = df["SEASON_ID"].astype(str).str.startswith("42")
    home = df[df["IS_HOME"]].copy()
    away = df[~df["IS_HOME"]].copy()

    # Merge so each row = one game with both teams' stats
    # IS_PLAYOFF comes from the home row (same game, same value)
    games = home[["GAME_ID", "TEAM_ABBREVIATION", "WL", "IS_PLAYOFF"]].merge(
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

        k = K_PLAYOFF if row["IS_PLAYOFF"] else K_REGULAR

        # Home team gets the advantage baked into expected probability
        e_h = expected(ratings[h] + HOME_ADVANTAGE, ratings[a])
        e_a = 1.0 - e_h

        home_won = row["WL_home"] == "W"
        ratings[h] = update(ratings[h], 1.0 if home_won else 0.0, e_h, k)
        ratings[a] = update(ratings[a], 0.0 if home_won else 1.0, e_a, k)

    ELO_CACHE.parent.mkdir(exist_ok=True)
    with open(ELO_CACHE, "w") as f:
        json.dump(ratings, f, indent=2)

    print(f"Ratings built for {len(ratings)} teams. Cached to {ELO_CACHE}")
    return ratings


# ── Prediction helpers ─────────────────────────────────────────────────────────

def win_prob(
    home_team: str,
    away_team: str,
    ratings: Optional[dict[str, float]] = None,
) -> float:
    """Win probability for the home team (0–1)."""
    if ratings is None:
        ratings = build_ratings()
    r_h = ratings.get(home_team, INITIAL_ELO)
    r_a = ratings.get(away_team, INITIAL_ELO)
    return expected(r_h + HOME_ADVANTAGE, r_a)
