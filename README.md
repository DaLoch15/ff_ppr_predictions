# Fantasy Football Projection System

## Overview

ML-powered PPR fantasy football projections with DFS optimization. This system leverages ensemble machine learning models, real-time NFL data, and advanced feature engineering to generate accurate weekly player projections with confidence intervals.

## Features

- **Real-time data integration** via nfl_data_py (play-by-play, weekly stats, rosters)
- **Ensemble ML models** combining XGBoost, Random Forest, and Gradient Boosting
- **Position-specific projections** with confidence intervals (floor/ceiling)
- **DFS lineup optimization** with salary cap constraints and game stacking
- **Value play identification** for tournaments and cash games
- **Automated weekly updates** with data fetching, training, and projection generation
- **Two-stage prediction** approach (Poisson for volume + XGBoost for efficiency)
- **Matchup adjustments** based on defensive rankings and Vegas lines
- **Comprehensive test suite** with 26 unit tests (80.8% passing)

## Installation

### Quick Start (Recommended)

```bash
git clone <repository>
cd fantasy-football-projections
./quickstart.sh
```

The quickstart script will:
1. Create and activate a virtual environment
2. Install all dependencies
3. Initialize the database schema
4. Fetch current season data
5. Train ML models
6. Generate sample projections

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize database
python3 main.py setup

# Fetch data
python3 main.py fetch --week 10 --season 2024

# Train models
python3 main.py train

# Generate projections
python3 main.py update --week 10 --season 2024
```

## Usage

### Generate Weekly Projections

```bash
# Full update with data fetch and projections
python3 main.py update --week 10 --season 2024

# Update without fetching new data
python3 main.py update --week 10 --season 2024 --no-fetch

# Generate projections for specific positions
python3 main.py update --week 10 --season 2024 --positions QB WR
```

### Train Models with Latest Data

```bash
# Train all position models
python3 main.py train

# Train specific positions
python3 main.py train --positions RB WR
```

### Fetch New Data

```bash
# Fetch data for current week
python3 main.py fetch --week 10 --season 2024

# Fetch multiple seasons
python3 main.py fetch --seasons 2022 2023 2024
```

### Run Backtesting

```bash
# Backtest last 5 weeks
python3 main.py backtest --weeks 5

# Backtest specific date range
python3 main.py backtest --start-week 1 --end-week 10 --season 2023
```

### Database Setup

```bash
# Initialize database with schema
python3 main.py setup

# Reset database (warning: deletes all data)
python3 main.py setup --reset
```

## Project Structure

```
fantasy-football-projections/
├── src/
│   ├── database/
│   │   ├── schema.sql           # Database schema (9 tables)
│   │   ├── db_manager.py        # Connection pooling and management
│   │   └── __init__.py
│   ├── data/
│   │   ├── data_fetcher.py      # NFL data collection via nfl_data_py
│   │   └── __init__.py
│   ├── models/
│   │   ├── ml_pipeline.py       # Two-stage and ensemble ML models
│   │   └── __init__.py
│   ├── production/
│   │   ├── weekly_system.py     # Weekly projection orchestration
│   │   └── __init__.py
│   ├── config.py                # Configuration loader
│   └── __init__.py
├── config/
│   └── config.yaml              # System configuration
├── tests/
│   ├── test_pipeline.py         # 26 unit tests
│   ├── README.md                # Test documentation
│   ├── TEST_SUMMARY.md          # Detailed test results
│   └── __init__.py
├── output/
│   └── projections/             # Generated projection outputs
│       └── YYYY_weekNN/         # Week-specific outputs
│           ├── QB_rankings.csv
│           ├── RB_rankings.csv
│           ├── WR_rankings.csv
│           ├── TE_rankings.csv
│           ├── lineup_cash_game.csv
│           ├── lineup_gpp_*.csv
│           ├── value_plays_*.csv
│           └── summary.txt
├── models/                      # Trained model files
│   └── position_YYYY/
│       ├── QB_model.pkl
│       ├── RB_model.pkl
│       └── ...
├── data/
│   └── cache/                   # Cached NFL data
├── logs/                        # Application logs
├── backups/                     # Database backups
├── examples/                    # Usage examples
├── main.py                      # CLI entry point
├── quickstart.sh               # Automated setup script
├── run_tests.sh                # Test runner
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
└── README.md                   # This file
```

### Directory Purposes

- **src/database/**: Database schema, connection management with pooling, backup functionality
- **src/data/**: Data fetching from nfl_data_py, processing, validation, rolling averages
- **src/models/**: ML pipeline with feature engineering, training, prediction
- **src/production/**: Weekly projection system, rankings, tiers, DFS optimization
- **config/**: YAML configuration with environment variable substitution
- **tests/**: Comprehensive unit tests for all components
- **output/**: Generated projections, rankings, lineups, and reports
- **models/**: Serialized trained models organized by position and season
- **logs/**: Application logs with rotation

## Configuration

### config.yaml Options

```yaml
database:
  path: fantasy_football.db      # SQLite database path
  pool_size: 5                   # Connection pool size
  backup_enabled: true           # Enable automatic backups

model:
  positions: [QB, RB, WR, TE]    # Positions to model
  training:
    test_size: 0.2               # Train/test split
    n_estimators: 200            # Trees per model
    max_depth: 6                 # Max tree depth
    learning_rate: 0.1           # XGBoost learning rate
    cv_folds: 5                  # Cross-validation folds

projections:
  rolling_windows:
    short: 3                     # 3-game rolling average
    medium: 5                    # 5-game rolling average
    long: 10                     # 10-game rolling average

  matchup_adjustment:
    enabled: true
    min_multiplier: 0.75         # vs elite defense
    max_multiplier: 1.25         # vs weak defense

  game_script_adjustment:
    enabled: true                # Adjust based on Vegas spread

  pace_adjustment:
    enabled: true                # Adjust based on team pace

  floor_ceiling:
    floor_multiplier: 0.60       # Floor = projection * 0.60
    ceiling_multiplier: 1.50     # Ceiling = projection * 1.50

  confidence_thresholds:
    high: 0.8                    # High confidence >= 80%
    medium: 0.6                  # Medium confidence >= 60%
    low: 0.4                     # Low confidence < 60%

dfs:
  salary_cap: 50000              # DraftKings/FanDuel cap
  positions:
    QB: 1
    RB: 2
    WR: 3
    TE: 1
    FLEX: 1                      # RB/WR/TE

  strategies:
    cash_game:                   # Conservative lineup
      projected_points: 0.7
      floor: 0.3

    gpp_balanced:                # Balanced tournament
      projected_points: 0.5
      ceiling: 0.3
      value: 0.2

    gpp_ceiling:                 # High-risk tournament
      ceiling: 0.6
      projected_points: 0.2
      leverage: 0.2

value_plays:
  thresholds:
    high_ceiling: 1.3            # Ceiling / projection ratio
    safe_floor: 0.8              # Floor / projection ratio
    leverage: 5.0                # Low ownership %
    positive_regression: 0.7     # Recent underperformance

features:
  universal:                     # Features for all positions
    - implied_game_script
    - vegas_total
    - home_away
    - days_rest
    - consistency_score
    - momentum

  position_specific:
    RB:
      - game_script_run
      - pass_catching_role
      - goal_line_back
    WR:
      - deep_threat
      - wr1_role
      - red_zone_wr
    TE:
      - te_premium
      - red_zone_role
    QB:
      - pass_heavy_script
      - shootout_potential

output:
  formats: [csv, json]           # Export formats
  include_confidence: true       # Include confidence intervals
  include_tiers: true            # Include tier assignments
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# The Odds API (for Vegas lines)
ODDS_API_KEY=your_odds_api_key_here

# Weather API (optional, for weather data)
WEATHER_API_KEY=your_weather_api_key_here
```

Environment variables can also be referenced in `config.yaml`:

```yaml
data_sources:
  odds_api_key: ${ODDS_API_KEY}
  weather_api_key: ${WEATHER_API_KEY:default_value}  # With default
```

## Model Details

### Two-Stage Prediction Approach

The system uses a sophisticated two-stage approach for more accurate projections:

**Stage 1: Volume Prediction (Poisson Regression)**
- Predicts opportunity volume (carries, targets, pass attempts)
- Uses Poisson distribution appropriate for count data
- Features: game script, team pace, matchup, Vegas total

**Stage 2: Efficiency Prediction (XGBoost)**
- Predicts efficiency metrics (YPC, YPT, TD rate)
- Uses predicted volume from Stage 1 as a feature
- Features: player metrics, rolling averages, defensive rankings

**Final Projection**: `volume × efficiency + TD_bonus`

### Ensemble Approach

Each position uses an ensemble of three models:

1. **XGBoost** (50% weight) - Primary model, best overall performance
2. **Random Forest** (25% weight) - Reduces overfitting, good for non-linear patterns
3. **Gradient Boosting** (25% weight) - Captures residual patterns

Final prediction = weighted average of all three models.

### Feature Engineering

**Universal Features** (all positions):
- `implied_game_script`: Team implied total - opponent implied total
- `vegas_total`: Over/under for game
- `home_away`: Home field advantage indicator
- `days_rest`: Days since last game
- `consistency_score`: Inverse of rolling standard deviation
- `momentum`: Recent trend in performance

**Position-Specific Features**:

**RB Features**:
- `game_script_run`: Expected run volume based on spread
- `pass_catching_role`: Targets per game rolling average
- `goal_line_back`: Red zone carry share
- `touches_3g`, `touches_5g`: Rolling touch counts

**WR Features**:
- `deep_threat`: Air yards per target
- `wr1_role`: Target share (>25% = WR1)
- `red_zone_wr`: Red zone target share
- `routes_run`: Route participation rate

**TE Features**:
- `te_premium`: TE premium scoring multiplier
- `red_zone_role`: Red zone target rate
- `target_share`: Team target percentage

**QB Features**:
- `pass_heavy_script`: Expected pass volume based on spread
- `shootout_potential`: High total + low spread
- `pass_attempts_3g`: Recent passing volume
- `td_rate`: Touchdown rate rolling average

### Model Training

```python
# Time-series cross-validation prevents data leakage
cv = TimeSeriesSplit(n_splits=5)

# Hyperparameter tuning with cross-validation
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train with early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=False
)
```

### Confidence Intervals

- **Floor** (10th percentile): `projection - 1.5 × model_std`
- **Ceiling** (90th percentile): `projection + 1.5 × model_std`
- **Confidence Score**: Based on prediction stability across ensemble

## Output Files

### Rankings Files

**Format**: `{POSITION}_rankings.csv`

```csv
rank,player_name,team,opponent,projected_points,floor,ceiling,confidence,tier,value_score
1,Patrick Mahomes,KC,@LV,24.5,18.2,30.8,0.87,1,5.2
2,Josh Allen,BUF,vs.MIA,23.8,17.5,30.1,0.85,1,4.9
...
```

**Columns**:
- `rank`: Overall rank for position
- `player_name`: Player name
- `team`: Team abbreviation
- `opponent`: Opponent (@ for away, vs. for home)
- `projected_points`: PPR projected points
- `floor`: Conservative projection (10th percentile)
- `ceiling`: Optimistic projection (90th percentile)
- `confidence`: Model confidence (0-1)
- `tier`: Tier assignment (1 = elite, 2 = strong, etc.)
- `value_score`: Points per $1000 salary (DFS)

### Lineup Files

**Format**: `lineup_{strategy}.csv`

```csv
position,player_name,team,projected_points,salary,value
QB,Patrick Mahomes,KC,24.5,8500,2.88
RB,Christian McCaffrey,SF,22.3,9800,2.28
RB,Travis Etienne,JAX,16.8,6300,2.67
WR,Tyreek Hill,MIA,19.2,8800,2.18
WR,CeeDee Lamb,DAL,18.5,8200,2.26
WR,Amon-Ra St. Brown,DET,16.9,7500,2.25
TE,Travis Kelce,KC,14.2,6700,2.12
FLEX,Rachaad White,TB,15.6,6200,2.52
TOTAL,,,148.0,49000,3.02
```

**Strategies**:
- `cash_game`: Conservative, high floor lineup
- `gpp_balanced`: Balanced tournament lineup
- `gpp_ceiling`: High-risk, high-ceiling tournament lineup
- `gpp_contrarian`: Low-ownership leverage plays

### Value Plays Files

**Format**: `value_plays_{category}.csv`

**Categories**:
- `high_ceiling`: Players with upside potential (ceiling 1.3x+ projection)
- `safe_floor`: High-floor players for cash games (floor 0.8x+ projection)
- `leverage`: Low-ownership tournament plays (<10% projected ownership)
- `positive_regression`: Players due for positive regression
- `best_values`: Top points per dollar regardless of projection

### Summary File

**Format**: `summary.txt`

```
===================================================================
          Fantasy Football Projections - Week 10, 2024
===================================================================

Generated: 2024-11-05 17:30:45
Model Version: 2024.1
Confidence: HIGH (87.3%)

-------------------------------------------------------------------
TOP PLAYS BY POSITION
-------------------------------------------------------------------

QUARTERBACK
  1. Patrick Mahomes (KC vs. LV)     24.5 pts  [Floor: 18.2  Ceiling: 30.8]
  2. Josh Allen (BUF vs. MIA)        23.8 pts  [Floor: 17.5  Ceiling: 30.1]
  3. Jalen Hurts (PHI @ DAL)         22.9 pts  [Floor: 16.8  Ceiling: 29.0]

RUNNING BACK
  1. Christian McCaffrey (SF @ JAX)  22.3 pts  [Floor: 16.5  Ceiling: 28.1]
  2. Travis Etienne (JAX vs. SF)     20.1 pts  [Floor: 14.8  Ceiling: 25.4]
  ...

-------------------------------------------------------------------
VALUE PLAYS
-------------------------------------------------------------------

Best Values (Points per $1K):
  - Jaylen Warren (RB, PIT): 3.45 pts/$1K
  - Jordan Addison (WR, MIN): 3.21 pts/$1K
  - Khalil Shakir (WR, BUF): 3.18 pts/$1K

High Ceiling Plays:
  - Tank Dell (WR, HOU): Ceiling 26.8 (1.42x projection)
  - David Njoku (TE, CLE): Ceiling 18.5 (1.39x projection)

-------------------------------------------------------------------
DFS LINEUP RECOMMENDATIONS
-------------------------------------------------------------------

Cash Game Lineup (Total: $48,500, Proj: 146.3 pts)
  QB:  Brock Purdy ($7,200) - 19.8 pts
  RB:  Travis Etienne ($6,800) - 20.1 pts
  ...
```

## API Keys

### The Odds API

Used for fetching Vegas betting lines (spreads, totals, moneylines).

1. Visit [the-odds-api.com](https://the-odds-api.com/)
2. Sign up for a free account (500 requests/month)
3. Get your API key from the dashboard
4. Add to `.env` file: `ODDS_API_KEY=your_key_here`

**Note**: The system includes mock Vegas line generation if no API key is provided, but real data significantly improves projection accuracy.

### Weather API (Optional)

Used for weather conditions (temperature, wind, precipitation).

1. Visit [weatherapi.com](https://www.weatherapi.com/)
2. Sign up for a free account
3. Get your API key
4. Add to `.env` file: `WEATHER_API_KEY=your_key_here`

Weather adjustments are optional and primarily affect QB/K projections in outdoor stadiums.

## Testing

### Run All Tests

```bash
# Using test runner
./run_tests.sh all

# Using Python directly
python3 tests/test_pipeline.py
```

### Run Specific Test Suites

```bash
./run_tests.sh quick        # Fast tests only (DB + Validator)
./run_tests.sh db           # Database tests
./run_tests.sh fetcher      # Data fetcher tests
./run_tests.sh ml           # ML pipeline tests
./run_tests.sh prod         # Production system tests
```

### Run with Coverage

```bash
./run_tests.sh coverage
```

Generates HTML coverage report in `htmlcov/index.html`

### Test Results

Current test suite: **26 tests**, **80.8% passing**

- TestDataFetcher: 7/7 passing ✓
- TestDataValidator: 4/4 passing ✓
- TestDatabaseManager: 5/5 passing ✓
- TestProductionSystem: 4/5 passing ✓
- TestMLPipeline: 1/5 passing (requires real data)

See `tests/TEST_SUMMARY.md` for detailed results.

## Contributing

### Guidelines for Adding Features

1. **Fork the repository** and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first** (Test-Driven Development)
   - Add tests to `tests/test_pipeline.py`
   - Ensure tests fail initially
   - Implement feature until tests pass

3. **Follow code style**
   - Use type hints for function signatures
   - Add docstrings (Google style)
   - Keep functions focused (single responsibility)
   - Maximum line length: 100 characters

4. **Update configuration**
   - Add new settings to `config/config.yaml`
   - Document in README.md

5. **Run tests and linting**
   ```bash
   ./run_tests.sh all
   python3 -m pylint src/
   ```

6. **Update documentation**
   - Add docstrings to new functions
   - Update README.md if adding features
   - Update `tests/TEST_SUMMARY.md` if adding tests

7. **Submit pull request**
   - Provide clear description of changes
   - Reference any related issues
   - Ensure all tests pass

### Improving Models

**Adding New Features**:
1. Add feature engineering logic to `ml_pipeline.py::_engineer_features()`
2. Update `config.yaml` with feature list
3. Retrain models and validate improvement
4. Document feature in README.md

**Tuning Hyperparameters**:
1. Modify training parameters in `config.yaml`
2. Run hyperparameter search:
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_estimators': [100, 200, 300], ...}
   grid_search = GridSearchCV(model, param_grid, cv=5)
   ```
3. Update config with best parameters

**Adding New Models**:
1. Implement in `ml_pipeline.py::train_ensemble_model()`
2. Add to ensemble with appropriate weight
3. Backtest to validate improvement

### Project Roadmap

- [ ] Add player injury impact modeling
- [ ] Implement weather adjustments
- [ ] Add playoff projections
- [ ] Support dynasty league projections
- [ ] Add web dashboard for visualization
- [ ] Implement real-time odds updates
- [ ] Add correlation analysis for DFS stacking
- [ ] Support multiple scoring formats (half-PPR, standard)
- [ ] Add rest-of-season projections
- [ ] Implement trade analyzer

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- **nfl_data_py**: For providing comprehensive NFL data access
- **XGBoost**: For high-performance gradient boosting
- **scikit-learn**: For machine learning infrastructure
- **The Odds API**: For Vegas betting lines

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing documentation in `tests/README.md` and `tests/TEST_SUMMARY.md`
- Review configuration options in `config/config.yaml`

## Quick Reference

```bash
# Setup
./quickstart.sh

# Weekly workflow
python3 main.py update --week 10 --season 2024

# Retrain models
python3 main.py train

# Run tests
./run_tests.sh quick

# View projections
cat output/projections/2024_week10/summary.txt

# Check model performance
python3 main.py backtest --weeks 5
```
