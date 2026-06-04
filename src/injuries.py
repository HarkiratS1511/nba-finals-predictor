"""
Level 3 — Injury & star-player adjustments.

Approach:
  - Each player has an estimated impact value in Elo points when fully healthy
  - Status (out / doubtful / questionable / probable / healthy) scales that impact
  - Wembanyama gets an additional defensive adjustment that net rating doesn't capture
    (shot-altering, rim protection — the "gravity" effect that never shows in box score)

Impact values are derived from:
  - CraftedPM / BPM season stats where available
  - Minutes share to weight how much of team performance they drive
  - Empirical rule: 1 point of BPM ≈ 15 Elo pts at average minutes share (~30 mpg)

Positive impact = benefits that team (loss of player hurts them)
"""

from dataclasses import dataclass
from typing import Literal

# ── Types ──────────────────────────────────────────────────────────────────────

Status = Literal["out", "doubtful", "questionable", "probable", "healthy"]

# Fraction of impact lost per status
STATUS_DISCOUNT: dict[Status, float] = {
    "out":          1.00,   # full impact lost
    "doubtful":     0.75,
    "questionable": 0.40,
    "probable":     0.15,
    "healthy":      0.00,   # no adjustment needed
}


@dataclass
class Player:
    name: str
    team: str                  # team abbreviation
    full_impact_elo: float     # Elo pts this player is worth at full health
    status: Status
    notes: str = ""

    @property
    def elo_penalty(self) -> float:
        """
        Elo points LOST from this player's team due to their injury.
        Positive value = team is weaker by this many Elo pts.
        """
        return self.full_impact_elo * STATUS_DISCOUNT[self.status]


# ── 2026 Finals injury report ──────────────────────────────────────────────────
# Impact values reasoned from 2025-26 stats:
#
#   Victor Wembanyama  — CraftedPM +8.7, DPOY, ~35 mpg
#     Base impact: 8.7 * 15 ≈ 130 Elo pts
#     Extra defensive gravity adjustment: +40 pts (shot-altering not in box score)
#     Total: ~170 Elo pts (elite franchise player, fully healthy)
#
#   Mitchell Robinson  — CraftedPM +2.5, ~20 mpg (limited role)
#     Base impact: 2.5 * 15 * (20/30 minutes ratio) ≈ 25 Elo pts
#     Matchup-specific bump vs Wemby: +20 pts (Robinson is NYK's only real rim deterrent)
#     Total: ~45 Elo pts — BUT broken pinkie means compromised grip/finishing
#
#   De'Aaron Fox       — key Spurs guard, was hobbled in WCF, status uncertain
#     Estimated +3.5 CraftedPM, ~32 mpg → ~52 Elo pts full health
#
#   Dylan Harper       — was hobbled in WCF
#     Estimated +2.0 CraftedPM, ~28 mpg → ~28 Elo pts full health

INJURY_REPORT: list[Player] = [
    Player(
        name="Victor Wembanyama",
        team="SAS",
        full_impact_elo=170.0,
        status="healthy",
        notes="Fully healthy. DPOY, WCF MVP. Averaging 27.3/10.9/3.1 in playoffs.",
    ),
    Player(
        name="Mitchell Robinson",
        team="NYK",
        full_impact_elo=45.0,
        status="healthy",        # confirmed playing Game 1
        notes="Confirmed active for Game 1. Broken pinkie was a concern but he's out there.",
    ),
    Player(
        name="De'Aaron Fox",
        team="SAS",
        full_impact_elo=52.0,
        status="probable",       # hobbled in WCF, expected available
        notes="Hamstring issue carried from WCF. Expected to play but may be limited.",
    ),
    Player(
        name="Dylan Harper",
        team="SAS",
        full_impact_elo=28.0,
        status="probable",
        notes="Was hobbled late in WCF. Expected available for Game 1.",
    ),
]


# ── Wembanyama defensive gravity adjustment ────────────────────────────────────

# Net rating doesn't capture Wemby's effect on shots he *doesn't* block —
# opponents avoid the paint entirely, taking harder mid-range shots instead.
# Research on elite rim protectors (Gobert, AD prime) suggests ~+15–25 Elo pts
# of defensive value above what standard net rating captures.
# Wemby is historically elite, so we use the high end.
WEMBY_GRAVITY_ADJ = 25.0   # extra Elo pts added to SAS (not already in net rating)


# ── Adjustment calculator ──────────────────────────────────────────────────────

def compute_injury_adjustments(
    team1: str,
    team2: str,
    injury_report: list[Player] | None = None,
) -> dict:
    """
    Returns a dict with:
      - team1_penalty   : Elo pts team1 loses from their own injuries
      - team2_penalty   : Elo pts team2 loses from their own injuries
      - net_adj         : net Elo adjustment for team1 (positive = benefits team1)
      - wemby_gravity   : extra Elo pts added to team2 (SAS) for Wemby's defence
      - breakdown       : human-readable list of each player's contribution
    """
    if injury_report is None:
        injury_report = INJURY_REPORT

    t1_penalty = 0.0
    t2_penalty = 0.0
    breakdown  = []

    for p in injury_report:
        penalty = p.elo_penalty
        if penalty == 0.0:
            continue
        if p.team == team1:
            t1_penalty += penalty
            breakdown.append(
                f"  ↓ {p.name} ({team1}) [{p.status}] — -{penalty:.0f} Elo pts  |  {p.notes}"
            )
        elif p.team == team2:
            t2_penalty += penalty
            breakdown.append(
                f"  ↓ {p.name} ({team2}) [{p.status}] — -{penalty:.0f} Elo pts to {team2}  |  {p.notes}"
            )

    # Wemby gravity always benefits SAS regardless of which slot they occupy
    wemby_gravity = WEMBY_GRAVITY_ADJ
    wemby_team = "SAS"
    breakdown.append(
        f"  ★ Wembanyama defensive gravity — +{wemby_gravity:.0f} Elo pts to {wemby_team} (rim deterrence above net rating)"
    )

    # net_adj > 0 means team1 benefits overall
    # SAS benefits from wemby gravity — add if SAS is team1, subtract if team2
    wemby_sign = 1.0 if team1 == "SAS" else -1.0
    net_adj = t2_penalty - t1_penalty + wemby_sign * wemby_gravity

    return {
        "team1_penalty":  round(t1_penalty, 1),
        "team2_penalty":  round(t2_penalty, 1),
        "wemby_gravity":  round(wemby_gravity, 1),
        "net_adj":        round(net_adj, 1),
        "breakdown":      breakdown,
    }
