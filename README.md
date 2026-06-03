# NBA Finals Predictor

Game-by-game and series win probabilities for the **2026 NBA Finals — San Antonio Spurs vs New York Knicks**.

Built across 5 progressively smarter levels, from pure Elo to a logistic regression trained on 5 seasons of real playoff data.

**Current prediction: SAS 75.1% — NYK 24.9%**

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full series prediction (all levels)
python predict.py

# Single game prediction
python predict.py --game 1

# Rebuild all data + retrain model from NBA API
python predict.py --refresh

# Retrain model only (keep cached data)
python predict.py --retrain
```

---

## How it works — the 5 levels

### Level 1 — Smart Elo
> `src/elo.py`

The foundation. Replays every 2025-26 regular season and playoff game chronologically to build Elo ratings for all 30 teams.

Three improvements over plain Elo:

- **Recency-weighted K-factor** — games from 4+ months ago count at half weight via exponential decay. A hot October means less than a hot May.
- **Margin-of-victory multiplier** — a 30-point blowout shifts ratings more than a 1-point squeaker. Uses FiveThirtyEight's autocorrelation correction so dominant teams don't get over-rewarded for beating weak opponents badly.
- **Per-team home court advantage** — calibrated from each team's actual home/away win rate this season, converted to Elo points. Some arenas genuinely matter more than others.

---

### Level 2 — Team Context
> `src/features.py`

Adds three real-world context signals on top of Elo:

- **Blended net rating** — 50% last-15-game adjusted margin + 50% full-season net rating. The last-15 margin is **opponent-adjusted**: each game's margin is scaled by `opponent_elo / league_avg_elo`, so blowing out a bottom-feeder counts less than beating a contender.
- **Rest days** — days since each team's last game before Game 1.
- **Pace differential** — faster pace means more possessions, more variance, slight edge to the underdog.

These are converted to Elo-point adjustments and fed into the model. The net rating → Elo conversion is deliberately conservative (8 pts per 1pt delta, capped at ±50) to prevent recent-form noise drowning the Elo signal.

---

### Level 3 — Injuries & Star Player Adjustments
> `src/injuries.py`

Each player on the injury report has an estimated Elo-point impact based on their CraftedPM/BPM and minutes share. Status scales the impact:

| Status | Impact discount |
|---|---|
| Out | 100% |
| Doubtful | 75% |
| Questionable | 40% |
| Probable | 15% |
| Healthy | 0% |

**2026 Finals injury report:**
- Mitchell Robinson (NYK) — broken right pinkie, questionable. Primary interior defender vs Wemby. NYK went 0-1 vs SAS without him this season (Wemby had 31 pts on 12 shots).
- De'Aaron Fox (SAS) — hamstring, probable.
- Dylan Harper (SAS) — general soreness, probable.

**Wembanyama defensive gravity adjustment (+25 Elo pts to SAS):** standard net rating doesn't capture the effect Wemby has on shots he *doesn't* block — opponents avoid the paint entirely and take harder mid-range shots instead. He was unanimous Defensive Player of the Year and this effect is real and unquantified by box scores.

---

### Level 4 — Matchup-Specific Factors
> `src/matchup.py`

Four stylistic matchup edges, each capped to prevent any one factor dominating:

- **3PT matchup** — offensive 3PT% vs opponent 3PT% allowed. NYK holds opponents to 30.5% from three in the playoffs (historically low). SAS shoots 36% on 42% of attempts.
- **Rebounding** — OREB per game vs opponent OREB allowed. Robinson's injury penalty applied here too — his absence tanks NYK's OREB rate from 39.4% → 28.6%.
- **Turnovers** — turnovers forced minus turnovers committed, relative to opponent. NYK forces 14.4/game vs SAS 13.0/game.
- **Pace variance** — faster average pace benefits the underdog slightly (more possessions = more chances to close the gap).

---

### Level 5 — Logistic Regression
> `src/model.py`

The primary predictor. Trained on **335 playoff games across 5 seasons (2020-21 → 2024-25)**.

Features: Elo difference, net rating difference, rest difference, average pace.

Learned coefficients (standardised):

| Feature | Weight | Meaning |
|---|---|---|
| `elo_diff` | **+1.179** | By far the strongest signal |
| `pace_avg` | +0.303 | Higher pace benefits home team slightly |
| `net_rating_diff` | -0.306 | Already baked into Elo, slight overcorrection |
| `rest_diff` | **+0.013** | Nearly zero — rest barely matters in playoffs |

The model confirmed what was wrong with our hand-tuned approach: rest days were worth +40 Elo pts in our Level 2 formula, but the actual historical weight is nearly zero. The model fixed this automatically.

L3 (injuries) and L4 (matchup) adjustments are fed in as Elo-point shifts before the model predicts — these are the factors that historical data can't capture directly (player-specific injuries, matchup context).

Training accuracy: **68.1%** on playoff games.

---

## Current prediction breakdown

```
SAS Elo: 1802  (home court advantage)
NYK Elo: 1784

Injuries/Wemby: +31 Elo pts → SAS
Matchup:        -15 Elo pts → SAS (NYK turnover edge)

Model output:
  Game 1 (SAS home)  SAS 69.1%
  Game 2 (SAS home)  SAS 69.1%
  Game 3 (NYK home)  SAS 52.5%
  Game 4 (NYK home)  SAS 52.5%
  Game 5 (SAS home)  SAS 69.1%
  Game 6 (NYK home)* SAS 52.5%
  Game 7 (SAS home)* SAS 69.1%

Series: SAS 75.1%  —  NYK 24.9%
Expected length: 5.68 games
```

Vegas implied: SAS ~65%. We're at 75%, slightly more bullish on the Spurs — likely because Wembanyama is underrepresented in 2-year-old Elo history.

---

## Project structure

```
predict.py          main entry point
src/
  elo.py            Level 1 — smart Elo engine
  features.py       Level 2 — net rating, rest, pace
  injuries.py       Level 3 — injury report + Wemby adjustment
  matchup.py        Level 4 — 3PT, rebounding, turnovers, pace
  model.py          Level 5 — logistic regression trainer + predictor
  simulate.py       Monte Carlo series simulator
data/
  elo_ratings.json  cached Elo ratings (rebuilt with --refresh)
  hca_ratings.json  per-team home court advantage
  features.json     cached team context features
  model.json        trained logistic regression weights
requirements.txt
```

---

## Why the Spurs are favoured

1. **Best player** — Wemby is the best player on the floor by a wide margin. Unanimous DPOY, WCF MVP averaging 27.3/10.9/3.1 in the playoffs.
2. **Home court** — Spurs have Games 1, 2, 5, 7 at home.
3. **Elo edge** — 18.6 pts, reflecting a better season record.
4. **Robinson injured** — NYK's only real interior answer to Wemby is compromised.
5. **Season net rating** — SAS +8.4 vs NYK +6.4.

NYK's advantages (recent form, rest, turnover rate, 3PT defense) are real but not enough to overcome the above.
