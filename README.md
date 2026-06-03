# NBA Finals Predictor

Game-by-game and series win probabilities for the 2025 NBA Finals (Knicks vs Spurs).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full series prediction
python predict.py

# Single game
python predict.py --game 1

# Force-refresh Elo ratings from NBA API
python predict.py --refresh
```

## How it works

**Stage 1 — Elo baseline**
Replays every 2024-25 regular season and playoff game to build Elo ratings for all 30 teams. Win probabilities are derived from the Elo difference with a +100-point home court adjustment (~57% for equal teams).

Series outcomes are estimated via 100k Monte Carlo simulations using the NBA Finals 2-2-1-1-1 home schedule.

**Stage 2 — Enhanced model** *(coming soon)*
- Net rating (offensive − defensive, season + last 15 games)
- Rest days / back-to-back flag
- Injury-adjusted lineup strength (BPM × minutes share)
- Pace and matchup-specific factors (rebounding, 3PT rate)
- Logistic regression trained on 5 years of playoff data

## Project structure

```
predict.py          entry point
src/
  elo.py            Elo rating engine + caching
  simulate.py       Monte Carlo series simulator
data/               cached ratings (git-ignored)
requirements.txt
```
