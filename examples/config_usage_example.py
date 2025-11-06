"""
Configuration Usage Example

This file demonstrates how to use the configuration system in your code.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import get_config


def example_database_setup():
    """Example: Using config for database setup."""
    config = get_config()

    # Get database configuration
    db_config = config.get_database_config()

    db_path = db_config['path']
    backup_path = db_config['backup_path']
    pool_size = db_config['pool_size']

    print(f"Database: {db_path}")
    print(f"Backup location: {backup_path}")
    print(f"Connection pool size: {pool_size}")


def example_model_training():
    """Example: Using config for model training."""
    config = get_config()

    # Get positions to train
    positions = config.get_positions()

    # Get training parameters
    training_config = config.get('model.training')
    n_estimators = training_config['n_estimators']
    max_depth = training_config['max_depth']
    learning_rate = training_config['learning_rate']

    # Get ensemble weights
    ensemble_weights = training_config['ensemble_weights']

    print(f"Training models for: {positions}")
    print(f"N estimators: {n_estimators}")
    print(f"Max depth: {max_depth}")
    print(f"Learning rate: {learning_rate}")
    print(f"Ensemble weights: {ensemble_weights}")


def example_projections():
    """Example: Using config for projections."""
    config = get_config()

    # Get rolling windows
    windows = config.get_rolling_windows()
    short_window = windows['short']
    medium_window = windows['medium']
    long_window = windows['long']

    # Get matchup adjustment settings
    if config.is_matchup_adjustment_enabled():
        min_mult, max_mult = config.get_matchup_adjustment_range()
        print(f"Matchup adjustments: {min_mult} - {max_mult}")

    # Get floor/ceiling multipliers
    floor, ceiling = config.get_floor_ceiling_multipliers()

    print(f"Rolling windows: {short_window}, {medium_window}, {long_window}")
    print(f"Floor: {floor*100:.0f}%, Ceiling: {ceiling*100:.0f}%")


def example_dfs_lineup():
    """Example: Using config for DFS lineup generation."""
    config = get_config()

    # Get DFS settings
    salary_cap = config.get_dfs_salary_cap()
    position_reqs = config.get_dfs_position_requirements()

    # Get lineup strategy weights
    cash_weights = config.get_lineup_strategy_weights('cash_game')
    gpp_weights = config.get_lineup_strategy_weights('gpp_ceiling')

    print(f"Salary cap: ${salary_cap:,}")
    print(f"Position requirements: {position_reqs}")
    print(f"Cash game weights: {cash_weights}")
    print(f"GPP weights: {gpp_weights}")


def example_api_access():
    """Example: Using config for API keys."""
    config = get_config()

    # Get API keys (with environment variable substitution)
    odds_api_key = config.get_api_key('odds_api')
    weather_api_key = config.get_api_key('weather_api')

    if odds_api_key:
        print(f"Odds API key is set")
    else:
        print("Odds API key not set - using mock data")

    if weather_api_key:
        print(f"Weather API key is set")
    else:
        print("Weather API key not set - using mock data")


def example_feature_engineering():
    """Example: Using config for feature engineering."""
    config = get_config()

    # Get universal features
    features_config = config.get_features_config()
    universal_features = features_config.get('universal', [])

    # Get position-specific features
    rb_features = features_config.get('rb_features', [])
    wr_features = features_config.get('wr_features', [])

    print(f"Universal features ({len(universal_features)}): {', '.join(universal_features[:3])}...")
    print(f"RB-specific features ({len(rb_features)}): {', '.join(rb_features)}")
    print(f"WR-specific features ({len(wr_features)}): {', '.join(wr_features)}")


def example_value_plays():
    """Example: Using config for value play identification."""
    config = get_config()

    # Get value plays configuration
    value_config = config.get_value_plays_config()

    # Get specific category settings
    high_ceiling_config = value_config['categories']['high_ceiling']
    leverage_config = value_config['categories']['leverage']

    print(f"High ceiling plays:")
    print(f"  Enabled: {high_ceiling_config['enabled']}")
    print(f"  Top N: {high_ceiling_config['top_n']}")
    print(f"  Min percentile: {high_ceiling_config['min_ceiling_percentile']}")

    print(f"\nLeverage plays:")
    print(f"  Min rank: {leverage_config['min_position_rank']}")
    print(f"  Min ceiling percentile: {leverage_config['min_ceiling_percentile']}")


def main():
    """Run all examples."""
    print("="*60)
    print("Configuration Usage Examples")
    print("="*60)

    examples = [
        ("Database Setup", example_database_setup),
        ("Model Training", example_model_training),
        ("Projections", example_projections),
        ("DFS Lineup", example_dfs_lineup),
        ("API Access", example_api_access),
        ("Feature Engineering", example_feature_engineering),
        ("Value Plays", example_value_plays),
    ]

    for name, example_func in examples:
        print(f"\n{name}:")
        print("-" * 60)
        example_func()

    print("\n" + "="*60)
    print("✓ All examples completed")
    print("="*60)


if __name__ == "__main__":
    main()
