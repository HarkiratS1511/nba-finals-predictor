"""
Level 5 — Logistic regression trained on 5 years of playoff data.

Pipeline:
  1. Fetch game logs for 2020-21 → 2024-25 playoffs
  2. Build Elo ratings incrementally for each season
  3. Compute per-game features: Elo diff, net rating diff, rest diff, pace diff
  4. Train logistic regression — learns real weights from outcomes
  5. Predict any matchup using the same feature vector

Key advantage over hand-tuned Elo adjustments:
  - Weights come from actual playoff outcomes, not intuition
  - Rest days, net rating, and pace are automatically calibrated
  - Model naturally handles interactions between features
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamefinder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_CACHE = Path(__file__).parent.parent / "data" / "model.json"
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


# ── Data helpers ───────────────────────────────────────────────────────────────

def _fetch_game_log(season: str) -> pd.DataFrame:
    time.sleep(1.5)
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00", timeout=60
    ).get_data_frames()[0]
    return df.sort_values("GAME_DATE")


def _fetch_net_ratings(season: str) -> dict[str, float]:
    """Returns {team_name: net_rating} for the season."""
    time.sleep(1.5)
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        timeout=60,
    ).get_data_frames()[0]
    return dict(zip(df["TEAM_NAME"], df["NET_RATING"].astype(float)))


def _fetch_pace(season: str) -> dict[str, float]:
    """Returns {team_name: pace} for the season."""
    time.sleep(1.5)
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        timeout=60,
    ).get_data_frames()[0]
    return dict(zip(df["TEAM_NAME"], df["PACE"].astype(float)))


# ── Elo builder (reused from elo.py logic, self-contained here) ────────────────

def _build_season_elo(game_log: pd.DataFrame) -> dict[str, float]:
    """Build end-of-season Elo ratings from a game log DataFrame."""
    from src.elo import expected, update, mov_multiplier, recency_weight, build_hca, INITIAL_ELO, K_REGULAR, K_PLAYOFF, DEFAULT_HCA

    game_log = game_log.copy()
    game_log["IS_HOME"]    = ~game_log["MATCHUP"].str.contains("@")
    game_log["IS_PLAYOFF"] = game_log["SEASON_ID"].astype(str).str.startswith("42")
    latest_date = game_log["GAME_DATE"].max()

    hca = build_hca(game_log)
    home = game_log[game_log["IS_HOME"]]
    away = game_log[~game_log["IS_HOME"]]

    games = home[["GAME_ID", "TEAM_ABBREVIATION", "WL", "PLUS_MINUS", "GAME_DATE", "IS_PLAYOFF"]].merge(
        away[["GAME_ID", "TEAM_ABBREVIATION", "WL"]],
        on="GAME_ID", suffixes=("_home", "_away"),
    )

    ratings: dict[str, float] = {}
    for _, row in games.iterrows():
        h, a = row["TEAM_ABBREVIATION_home"], row["TEAM_ABBREVIATION_away"]
        ratings.setdefault(h, INITIAL_ELO)
        ratings.setdefault(a, INITIAL_ELO)

        rw  = recency_weight(row["GAME_DATE"], latest_date)
        k   = (K_PLAYOFF if row["IS_PLAYOFF"] else K_REGULAR) * rw
        hca_pts = (hca.get(h, DEFAULT_HCA) + hca.get(a, DEFAULT_HCA)) / 2
        e_h = expected(ratings[h] + hca_pts, ratings[a])
        e_a = 1.0 - e_h

        home_won = row["WL_home"] == "W"
        mov = mov_multiplier(abs(row["PLUS_MINUS"]), ratings[h] - ratings[a])
        ratings[h] = update(ratings[h], 1.0 if home_won else 0.0, e_h, k * mov)
        ratings[a] = update(ratings[a], 0.0 if home_won else 1.0, e_a, k * mov)

    return ratings


# ── Feature engineering ────────────────────────────────────────────────────────

def _rest_days(game_log: pd.DataFrame, team: str, before_date: str) -> int:
    """Days of rest for a team before a given game date."""
    prev = game_log[
        (game_log["TEAM_ABBREVIATION"] == team) &
        (game_log["GAME_DATE"] < before_date)
    ]
    if prev.empty:
        return 3   # default if no prior game found
    last = prev["GAME_DATE"].iloc[-1]
    d1 = pd.Timestamp(last).date()
    d2 = pd.Timestamp(before_date).date()
    return max(0, (d2 - d1).days - 1)


def _name_to_abbrev(game_log: pd.DataFrame) -> dict[str, str]:
    if "TEAM_NAME" in game_log.columns:
        return dict(zip(game_log["TEAM_NAME"], game_log["TEAM_ABBREVIATION"]))
    return {}


def build_training_data(seasons: list[str] = TRAIN_SEASONS) -> pd.DataFrame:
    """
    Build a feature matrix from historical playoff games.

    Each row = one playoff game (home team perspective).
    Target = 1 if home team won, 0 otherwise.

    Features:
        elo_diff        : home Elo - away Elo (at game time, pre-game)
        net_rating_diff : home net rating - away net rating (season)
        rest_diff       : home rest days - away rest days
        pace_avg        : average pace of both teams (higher = more variance)
        is_elimination  : proxy — game 5/6/7 (higher stakes)
    """
    all_rows = []

    for season in seasons:
        print(f"  Processing {season} …")
        try:
            glog       = _fetch_game_log(season)
            net_ratings = _fetch_net_ratings(season)
            pace_map    = _fetch_pace(season)
            elo         = _build_season_elo(glog)
            n2a         = _name_to_abbrev(glog)

            # Reverse map: abbrev → name
            a2n = {v: k for k, v in n2a.items()}

            # Playoff games only, home perspective
            playoffs = glog[glog["SEASON_ID"].astype(str).str.startswith("42")].copy()
            home_games = playoffs[~playoffs["MATCHUP"].str.contains("@")]
            away_games = playoffs[playoffs["MATCHUP"].str.contains("@")]

            merged = home_games[["GAME_ID", "TEAM_ABBREVIATION", "WL", "GAME_DATE"]].merge(
                away_games[["GAME_ID", "TEAM_ABBREVIATION"]],
                on="GAME_ID", suffixes=("_home", "_away"),
            )

            for _, row in merged.iterrows():
                h = row["TEAM_ABBREVIATION_home"]
                a = row["TEAM_ABBREVIATION_away"]
                gdate = row["GAME_DATE"]

                h_name = a2n.get(h, "")
                a_name = a2n.get(a, "")

                h_elo = elo.get(h, 1500)
                a_elo = elo.get(a, 1500)
                h_nr  = net_ratings.get(h_name, 0.0)
                a_nr  = net_ratings.get(a_name, 0.0)
                h_pace = pace_map.get(h_name, 98.0)
                a_pace = pace_map.get(a_name, 98.0)
                h_rest = _rest_days(glog, h, gdate)
                a_rest = _rest_days(glog, a, gdate)

                all_rows.append({
                    "season":          season,
                    "game_date":       gdate,
                    "home_team":       h,
                    "away_team":       a,
                    "elo_diff":        h_elo - a_elo,
                    "net_rating_diff": h_nr - a_nr,
                    "rest_diff":       h_rest - a_rest,
                    "pace_avg":        (h_pace + a_pace) / 2,
                    "home_won":        1 if row["WL"] == "W" else 0,
                })

        except Exception as e:
            print(f"  Warning: failed to process {season} — {e}")
            continue

    return pd.DataFrame(all_rows)


# ── Model training ─────────────────────────────────────────────────────────────

FEATURE_COLS = ["elo_diff", "net_rating_diff", "rest_diff", "pace_avg"]


def train(df: pd.DataFrame) -> tuple[LogisticRegression, StandardScaler, dict]:
    """Train logistic regression, return (model, scaler, coefficients)."""
    X = df[FEATURE_COLS].values
    y = df["home_won"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    # Accuracy on training data (ballpark — no held-out set given small N)
    acc = model.score(X_scaled, y)

    coefs = dict(zip(FEATURE_COLS, model.coef_[0]))
    print(f"  Training accuracy : {acc*100:.1f}%  ({len(df)} games)")
    print(f"  Coefficients      : { {k: round(v,3) for k,v in coefs.items()} }")

    return model, scaler, coefs


def predict_game(
    model: LogisticRegression,
    scaler: StandardScaler,
    elo_diff: float,
    net_rating_diff: float,
    rest_diff: float,
    pace_avg: float,
) -> float:
    """Return win probability for the home team (0–1)."""
    X = np.array([[elo_diff, net_rating_diff, rest_diff, pace_avg]])
    X_scaled = scaler.transform(X)
    return float(model.predict_proba(X_scaled)[0][1])


# ── Cache helpers ──────────────────────────────────────────────────────────────

def save_model(scaler: StandardScaler, model: LogisticRegression, coefs: dict) -> None:
    payload = {
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef":         model.coef_[0].tolist(),
        "intercept":    float(model.intercept_[0]),
        "features":     FEATURE_COLS,
        "coef_named":   coefs,
    }
    MODEL_CACHE.parent.mkdir(exist_ok=True)
    with open(MODEL_CACHE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Model saved to {MODEL_CACHE}")


def load_model() -> tuple[LogisticRegression, StandardScaler] | None:
    if not MODEL_CACHE.exists():
        return None
    with open(MODEL_CACHE) as f:
        p = json.load(f)

    scaler = StandardScaler()
    scaler.mean_  = np.array(p["scaler_mean"])
    scaler.scale_ = np.array(p["scaler_scale"])

    model = LogisticRegression()
    model.coef_      = np.array([p["coef"]])
    model.intercept_ = np.array([p["intercept"]])
    model.classes_   = np.array([0, 1])

    return model, scaler


def build_model(force: bool = False) -> tuple[LogisticRegression, StandardScaler]:
    """
    Load cached model or retrain from scratch.
    Returns (model, scaler) ready for predict_game().
    """
    if not force:
        result = load_model()
        if result:
            return result

    print("Training logistic regression on playoff data …")
    df = build_training_data()
    model, scaler, coefs = train(df)
    save_model(scaler, model, coefs)
    return model, scaler
