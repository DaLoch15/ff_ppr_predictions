

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
