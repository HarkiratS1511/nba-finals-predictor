"""
Level 2 feature engine.

Computes team-level context features for a given matchup:
  - Season net rating (OFF - DEF)
  - Last-N-games net rating
  - Blended net rating (recent form weighted more heavily)
  - Rest days before the game
  - Back-to-back flag
  - Pace differential
"""

import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamefinder

FEATURES_CACHE = Path(__file__).parent.parent / "data" / "features.json"

# How much to weight recent form vs full season (must sum to 1.0)
RECENT_WEIGHT = 0.65
SEASON_WEIGHT = 0.35

LAST_N_GAMES = 15


# ── Data fetchers ──────────────────────────────────────────────────────────────

def _fetch_advanced_stats(season: str, retries: int = 3) -> pd.DataFrame:
    """
    Fetch full-season advanced stats (net rating, pace) from the NBA API.
    Retries on failure with a short sleep between attempts.
    """
    for attempt in range(retries):
        try:
            time.sleep(1 + attempt)   # back off on retries
            df = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                timeout=60,
            ).get_data_frames()[0]
            # Normalise: add TEAM_ABBREVIATION via a name→abbrev map built from game log
            return df
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch advanced stats after {retries} attempts: {e}")


def _fetch_game_log(season: str, retries: int = 3) -> pd.DataFrame:
    """Fetch the full game log for the season, with retries."""
    for attempt in range(retries):
        try:
            time.sleep(1 + attempt)
            df = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                timeout=60,
            ).get_data_frames()[0]
            return df.sort_values("GAME_DATE")
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch game log after {retries} attempts: {e}")


# ── Feature builders ───────────────────────────────────────────────────────────

def _rest_days(last_game_date: str, game_date: str) -> int:
    """Days of rest between last game and the upcoming game."""
    d1 = datetime.strptime(last_game_date, "%Y-%m-%d").date()
    d2 = datetime.strptime(game_date, "%Y-%m-%d").date()
    return (d2 - d1).days - 1   # -1 because game day itself is not a rest day


def _last_n_margin(game_log: pd.DataFrame, team_abbrev: str, n: int = LAST_N_GAMES) -> float:
    """Average point margin over the last N games for a team."""
    team_games = game_log[game_log["TEAM_ABBREVIATION"] == team_abbrev]
    return team_games.tail(n)["PLUS_MINUS"].mean()


def _build_name_to_abbrev(game_log: pd.DataFrame) -> dict[str, str]:
    """Build a TEAM_NAME → TEAM_ABBREVIATION lookup from the game log."""
    # leaguegamefinder has TEAM_NAME and TEAM_ABBREVIATION
    if "TEAM_NAME" in game_log.columns:
        return dict(zip(game_log["TEAM_NAME"], game_log["TEAM_ABBREVIATION"]))
    return {}


# ── Main entry point ───────────────────────────────────────────────────────────

def build_features(
    team1: str,
    team2: str,
    game_date: str,          # "YYYY-MM-DD" — date of the upcoming game
    season: str = "2025-26",
    force: bool = False,
) -> dict:
    """
    Build a feature dict for a team1 (home) vs team2 (away) matchup.

    Returns a dict with keys for each feature, plus a blended_net_rating_delta
    which is the single most useful scalar for the predictor.
    """
    if FEATURES_CACHE.exists() and not force:
        with open(FEATURES_CACHE) as f:
            return json.load(f)

    print("Fetching advanced stats and game log …")
    adv   = _fetch_advanced_stats(season)
    glog  = _fetch_game_log(season)

    # Map team names in advanced stats → abbreviations via game log
    name_to_abbrev = _build_name_to_abbrev(glog)
    adv["TEAM_ABBREVIATION"] = adv["TEAM_NAME"].map(name_to_abbrev)

    def team_adv(abbrev: str) -> pd.Series:
        row = adv[adv["TEAM_ABBREVIATION"] == abbrev]
        if row.empty:
            raise ValueError(f"No advanced stats found for {abbrev}")
        return row.iloc[0]

    t1, t2 = team_adv(team1), team_adv(team2)

    # Last game dates for rest calculation
    def last_game(abbrev: str) -> str:
        games = glog[glog["TEAM_ABBREVIATION"] == abbrev]
        return games["GAME_DATE"].iloc[-1]

    t1_last = last_game(team1)
    t2_last = last_game(team2)

    t1_rest = _rest_days(t1_last, game_date)
    t2_rest = _rest_days(t2_last, game_date)

    # Last-N margin as a proxy for recent net rating
    t1_recent = _last_n_margin(glog, team1)
    t2_recent = _last_n_margin(glog, team2)

    # Blended net rating: weight recent form more than full season
    t1_blended = RECENT_WEIGHT * t1_recent + SEASON_WEIGHT * t1["NET_RATING"]
    t2_blended = RECENT_WEIGHT * t2_recent + SEASON_WEIGHT * t2["NET_RATING"]

    features = {
        # Full-season ratings
        "t1_off_rating":      round(float(t1["OFF_RATING"]), 2),
        "t1_def_rating":      round(float(t1["DEF_RATING"]), 2),
        "t1_net_rating":      round(float(t1["NET_RATING"]), 2),
        "t2_off_rating":      round(float(t2["OFF_RATING"]), 2),
        "t2_def_rating":      round(float(t2["DEF_RATING"]), 2),
        "t2_net_rating":      round(float(t2["NET_RATING"]), 2),

        # Recent form
        "t1_last15_margin":   round(t1_recent, 2),
        "t2_last15_margin":   round(t2_recent, 2),

        # Blended rating
        "t1_blended":         round(t1_blended, 2),
        "t2_blended":         round(t2_blended, 2),
        "blended_delta":      round(t1_blended - t2_blended, 2),  # positive = t1 stronger

        # Rest
        "t1_rest_days":       t1_rest,
        "t2_rest_days":       t2_rest,
        "t1_b2b":             t1_rest == 0,
        "t2_b2b":             t2_rest == 0,
        "rest_delta":         t1_rest - t2_rest,   # positive = t1 more rested

        # Pace
        "t1_pace":            round(float(t1["PACE"]), 2),
        "t2_pace":            round(float(t2["PACE"]), 2),
        "pace_delta":         round(float(t1["PACE"]) - float(t2["PACE"]), 2),
    }

    FEATURES_CACHE.parent.mkdir(exist_ok=True)
    with open(FEATURES_CACHE, "w") as f:
        json.dump(features, f, indent=2)

    return features
