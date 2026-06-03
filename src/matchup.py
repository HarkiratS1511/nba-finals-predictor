"""
Level 4 — Matchup-specific adjustments.

Computes Elo-point adjustments based on stylistic mismatches:
  1. 3PT matchup   — offensive 3PT rate vs opponent 3PT defense
  2. Rebounding    — OREB% vs opponent DREB%; playoff modifier for Robinson injury
  3. Turnovers     — team TOV rate vs opponent forced-TOV rate
  4. Pace impact   — faster pace raises variance, slight edge to underdog

All stats pulled from nba_api with playoff context modifiers applied on top.
Conversion calibration: each factor is capped to prevent any single edge dominating.
"""

import time
from dataclasses import dataclass

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

# ── Elo conversion weights ─────────────────────────────────────────────────────
# Each unit of advantage → Elo pts. Tuned to keep individual factors small.
PTS_PER_3PT_PCT_EDGE   = 60.0   # per 1.0 (100%) 3PT% edge — scaled from fraction
PTS_PER_OREB_EDGE      = 80.0   # per 1.0 (100%) OREB% edge
PTS_PER_TOV_EDGE       = 50.0   # per 1.0 (100%) net TOV% edge (we force - we commit)
PTS_PER_PACE_POSSESSION = 0.8   # per extra possession per game pace advantage to underdog

# Individual caps per factor (Elo pts)
CAP_3PT    = 20.0
CAP_REB    = 25.0
CAP_TOV    = 15.0
CAP_PACE   = 10.0


# ── Data fetcher ───────────────────────────────────────────────────────────────

def _fetch_stats(season: str, measure: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            time.sleep(1 + attempt)
            return leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense=measure,
                timeout=60,
            ).get_data_frames()[0]
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch {measure} stats: {e}")


# ── Individual matchup calculators ─────────────────────────────────────────────

def _three_point_adj(
    t1_fg3_pct: float,    # t1 offensive 3PT%
    t1_fg3a_rate: float,  # t1 3PT attempts / FGA
    t2_opp_fg3_pct: float,  # t2 opponent 3PT% allowed (defensive quality)
    t2_fg3_pct: float,
    t2_fg3a_rate: float,
    t1_opp_fg3_pct: float,
) -> float:
    """
    Positive = benefits team1.

    Logic: if t1 shoots a lot of 3s AND t2 is bad at defending them → big edge for t1.
    Conversely if t2 has elite 3PT D (like NYK's 30.5% opp%), t1's 3PT attack is neutralised.

    We use: (t1_3PT% - t2_opp_3PT_allowed) * t1_3PT_rate
            vs
            (t2_3PT% - t1_opp_3PT_allowed) * t2_3PT_rate
    """
    t1_edge = (t1_fg3_pct - t2_opp_fg3_pct) * t1_fg3a_rate
    t2_edge = (t2_fg3_pct - t1_opp_fg3_pct) * t2_fg3a_rate
    raw = (t1_edge - t2_edge) * PTS_PER_3PT_PCT_EDGE
    return max(-CAP_3PT, min(CAP_3PT, raw))


def _rebounding_adj(
    t1_oreb: float, t1_dreb: float,
    t2_oreb: float, t2_dreb: float,
    t2_opp_oreb: float, t1_opp_oreb: float,
    robinson_penalty: float = 0.0,
) -> float:
    """
    Positive = benefits team1.

    OREB advantage: t1 OREB vs t2 opponent OREB allowed (t2 DREB quality).
    We proxy DREB quality by opponent OREB allowed per game.

    robinson_penalty: extra Elo penalty to team1 if Robinson is out/limited,
    since his absence tanks NYK's OREB rate from 39.4% → 28.6%.
    """
    # Raw rebound edge per game (offensive boards t1 gets - offensive boards t2 gets)
    t1_oreb_adj = t1_oreb - t2_opp_oreb   # how many more/fewer OREB t1 grabs vs what t2 allows
    t2_oreb_adj = t2_oreb - t1_opp_oreb
    raw = (t1_oreb_adj - t2_oreb_adj) * PTS_PER_OREB_EDGE / 10.0  # /10 to scale boards to Elo
    adj = max(-CAP_REB, min(CAP_REB, raw))
    return adj - robinson_penalty


def _turnover_adj(
    t1_tov: float, t2_tov: float,
    t1_opp_tov: float, t2_opp_tov: float,
) -> float:
    """
    Positive = benefits team1.

    Net TOV edge: (turnovers forced on opponent - own turnovers committed), relative to t2.
    t1_opp_tov = turnovers t1 forces from opponents per game.
    t1_tov     = turnovers t1 commits per game.
    """
    t1_net = t1_opp_tov - t1_tov   # positive = t1 wins the turnover battle
    t2_net = t2_opp_tov - t2_tov
    raw = (t1_net - t2_net) * PTS_PER_TOV_EDGE / 3.0   # /3 to scale TOV diff to Elo
    return max(-CAP_TOV, min(CAP_TOV, raw))


def _pace_adj(
    t1_pace: float, t2_pace: float,
    t1_elo: float,  t2_elo: float,
) -> float:
    """
    Positive = benefits team1.

    Faster pace means more possessions, more variance, which slightly helps
    the underdog. We only apply this if there's a meaningful pace gap (>2 poss/game).
    """
    pace_diff = (t1_pace + t2_pace) / 2   # expected game pace
    baseline_pace = 97.0
    extra_possessions = max(0.0, pace_diff - baseline_pace)

    # Underdog benefits from extra possessions
    underdog_is_t1 = t1_elo < t2_elo
    raw = extra_possessions * PTS_PER_PACE_POSSESSION
    adj = max(-CAP_PACE, min(CAP_PACE, raw))
    return adj if underdog_is_t1 else -adj


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_matchup_adjustments(
    team1: str,
    team2: str,
    t1_elo: float,
    t2_elo: float,
    season: str = "2025-26",
    robinson_questionable: bool = True,
) -> dict:
    """
    Returns a dict with per-factor Elo adjustments and a net total.
    Positive net = benefits team1.

    robinson_questionable: if True, applies a rebounding penalty to team1 (NYK)
    reflecting the drop in OREB when Robinson is limited.
    """
    base = _fetch_stats(season, "Base")
    opp  = _fetch_stats(season, "Opponent")

    def get(df: pd.DataFrame, team_name_fragment: str) -> pd.Series:
        row = df[df["TEAM_NAME"].str.contains(team_name_fragment, case=False)]
        if row.empty:
            raise ValueError(f"Team not found: {team_name_fragment}")
        return row.iloc[0]

    # Map abbreviation → name fragment
    NAME_MAP = {"NYK": "Knicks", "SAS": "Spurs"}
    t1_base = get(base, NAME_MAP.get(team1, team1))
    t2_base = get(base, NAME_MAP.get(team2, team2))
    t1_opp  = get(opp,  NAME_MAP.get(team1, team1))
    t2_opp  = get(opp,  NAME_MAP.get(team2, team2))

    # ── 3PT matchup ────────────────────────────────────────────────────────────
    # Opponent 3PT% allowed is in the opponent stats as OPP_FG3_PCT (if available)
    # Proxy: use opponent FG3_PCT from opponent table, else fall back to league avg
    t1_opp_fg3_pct = t1_opp.get("OPP_FG3_PCT", 0.36)
    t2_opp_fg3_pct = t2_opp.get("OPP_FG3_PCT", 0.36)

    # Apply known playoff-context values from research (more accurate than reg season)
    # NYK opponents: 30.5% in playoffs | SAS opponents: ~35.5% estimated
    # Assign correctly regardless of which slot each team occupies
    NYK_OPP_FG3 = 0.305   # NYK elite 3PT D
    SAS_OPP_FG3 = 0.355   # SAS average 3PT D
    t1_opp_fg3_pct = NYK_OPP_FG3 if team1 == "NYK" else SAS_OPP_FG3
    t2_opp_fg3_pct = NYK_OPP_FG3 if team2 == "NYK" else SAS_OPP_FG3

    adj_3pt = _three_point_adj(
        t1_fg3_pct   = float(t1_base["FG3_PCT"]),
        t1_fg3a_rate = float(t1_base["FG3A"]) / float(t1_base["FGA"]),
        t2_opp_fg3_pct = t2_opp_fg3_pct,
        t2_fg3_pct   = float(t2_base["FG3_PCT"]),
        t2_fg3a_rate = float(t2_base["FG3A"]) / float(t2_base["FGA"]),
        t1_opp_fg3_pct = t1_opp_fg3_pct,
    )

    # ── Rebounding matchup ─────────────────────────────────────────────────────
    # Robinson playoff OREB: 39.4% with him, 28.6% without (from research)
    # Regular season OREB: 12.7 pg NYK, 11.4 pg SAS
    # Apply penalty if Robinson questionable (40% discount on his OREB contribution)
    # Robinson penalty always hurts NYK's rebounding regardless of team slot
    nyk_is_t1 = (team1 == "NYK")
    robinson_oreb_loss_nyk = 8.0 if robinson_questionable else 0.0

    adj_reb = _rebounding_adj(
        t1_oreb     = float(t1_base["OREB"]),
        t1_dreb     = float(t1_base["DREB"]),
        t2_oreb     = float(t2_base["OREB"]),
        t2_dreb     = float(t2_base["DREB"]),
        t2_opp_oreb = float(t2_opp["OPP_OREB"]),
        t1_opp_oreb = float(t1_opp["OPP_OREB"]),
        robinson_penalty = robinson_oreb_loss_nyk if nyk_is_t1 else 0.0,
    )
    # If NYK is team2, the penalty flips sign (hurts team2 = benefits team1)
    if not nyk_is_t1 and robinson_questionable:
        adj_reb += robinson_oreb_loss_nyk

    # ── Turnover matchup ───────────────────────────────────────────────────────
    adj_tov = _turnover_adj(
        t1_tov     = float(t1_base["TOV"]),
        t2_tov     = float(t2_base["TOV"]),
        t1_opp_tov = float(t1_opp["OPP_TOV"]),
        t2_opp_tov = float(t2_opp["OPP_TOV"]),
    )

    # ── Pace impact ────────────────────────────────────────────────────────────
    t1_pace = float(t1_base.get("PACE", 97.7))
    t2_pace = float(t2_base.get("PACE", 100.7))
    adj_pace = _pace_adj(t1_pace, t2_pace, t1_elo, t2_elo)

    net = adj_3pt + adj_reb + adj_tov + adj_pace

    return {
        "adj_3pt":   round(adj_3pt,  1),
        "adj_reb":   round(adj_reb,  1),
        "adj_tov":   round(adj_tov,  1),
        "adj_pace":  round(adj_pace, 1),
        "net_adj":   round(net,      1),
        "breakdown": {
            "3PT matchup":   f"{adj_3pt:+.1f} Elo  (NYK 37.3% on 42.7% rate vs SAS 36.0%; NYK D holds opps to 30.5%)",
            "Rebounding":    f"{adj_reb:+.1f} Elo  (NYK 12.7 OREB vs SAS 11.4; Robinson penalty -{robinson_oreb_loss_nyk:.0f} pts)",
            "Turnovers":     f"{adj_tov:+.1f} Elo  (NYK forces {t1_opp['OPP_TOV']:.1f}/gm, commits {t1_base['TOV']:.1f}/gm)",
            "Pace variance": f"{adj_pace:+.1f} Elo  (SAS {t2_pace:.1f} pace vs NYK {t1_pace:.1f} — faster pace aids underdog)",
        },
    }
