"""
Configuration Module

This module handles loading and parsing the YAML configuration file with
environment variable substitution and type validation.
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    logging.warning("PyYAML not installed. Run: pip install pyyaml")
    yaml = None


logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager with environment variable substitution.

    Loads configuration from YAML file and provides easy access to settings
    with support for nested keys and environment variable substitution.

    Example:
        >>> config = Config()
        >>> db_path = config.get('database.path')
        >>> positions = config.get('model.positions')
        >>> api_key = config.get('data_sources.odds_api_key')  # Substitutes ${ODDS_API_KEY}
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / 'config' / 'config.yaml'

        self.config_path = Path(config_path)
        self._config_data = {}
        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if yaml is None:
            logger.error("PyYAML is not installed. Cannot load configuration.")
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                self._config_data = yaml.safe_load(f)

            # Substitute environment variables
            self._config_data = self._substitute_env_vars(self._config_data)

            logger.info(f"Configuration loaded from: {self.config_path}")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _substitute_env_vars(self, data: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Supports ${VAR_NAME} syntax with optional defaults: ${VAR_NAME:default_value}

        Args:
            data: Configuration data (dict, list, str, or other)

        Returns:
            Data with environment variables substituted
        """
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._substitute_env_var_in_string(data)
        else:
            return data

    def _substitute_env_var_in_string(self, text: str) -> str:
        """
        Substitute environment variables in a string.

        Supports:
        - ${VAR_NAME} - Required variable
        - ${VAR_NAME:default} - Variable with default value

        Args:
            text: String that may contain environment variables

        Returns:
            String with environment variables substituted
        """
        # Pattern matches ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2)

            # Get environment variable
            value = os.getenv(var_name)

            if value is not None:
                return value
            elif default_value is not None:
                return default_value
            else:
                # Variable not set and no default
                logger.warning(f"Environment variable '{var_name}' not set and no default provided")
                return match.group(0)  # Return original ${VAR_NAME}

        return re.sub(pattern, replace_var, text)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports dot notation for nested keys (e.g., 'database.path').

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get('database.path')
            'fantasy_football.db'
            >>> config.get('model.positions')
            ['QB', 'RB', 'WR', 'TE']
            >>> config.get('nonexistent.key', 'default_value')
            'default_value'
        """
        keys = key.split('.')
        value = self._config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get('database', {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('model', {})

    def get_projections_config(self) -> Dict[str, Any]:
        """Get projections configuration."""
        return self.get('projections', {})

    def get_dfs_config(self) -> Dict[str, Any]:
        """Get DFS configuration."""
        return self.get('dfs', {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get('output', {})

    def get_features_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get('features', {})

    def get_value_plays_config(self) -> Dict[str, Any]:
        """Get value plays configuration."""
        return self.get('value_plays', {})

    def get_positions(self) -> list:
        """Get list of positions to model."""
        return self.get('model.positions', ['QB', 'RB', 'WR', 'TE'])

    def get_dfs_salary_cap(self) -> int:
        """Get DFS salary cap."""
        return self.get('dfs.salary_cap', 50000)

    def get_dfs_position_requirements(self) -> Dict[str, int]:
        """Get DFS position requirements."""
        return self.get('dfs.positions', {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1
        })

    def get_rolling_windows(self) -> Dict[str, int]:
        """Get rolling average windows."""
        return self.get('projections.rolling_windows', {
            'short': 3, 'medium': 5, 'long': 10
        })

    def get_confidence_thresholds(self) -> Dict[str, float]:
        """Get confidence score thresholds."""
        return self.get('projections.confidence_thresholds', {
            'high': 0.8, 'medium': 0.6, 'low': 0.4
        })

    def get_matchup_adjustment_range(self) -> tuple:
        """Get matchup adjustment multiplier range."""
        min_mult = self.get('projections.matchup_adjustment.min_multiplier', 0.75)
        max_mult = self.get('projections.matchup_adjustment.max_multiplier', 1.25)
        return (min_mult, max_mult)

    def is_matchup_adjustment_enabled(self) -> bool:
        """Check if matchup adjustments are enabled."""
        return self.get('projections.matchup_adjustment.enabled', True)

    def is_game_script_adjustment_enabled(self) -> bool:
        """Check if game script adjustments are enabled."""
        return self.get('projections.game_script_adjustment.enabled', True)

    def is_pace_adjustment_enabled(self) -> bool:
        """Check if pace adjustments are enabled."""
        return self.get('projections.pace_adjustment.enabled', True)

    def get_floor_ceiling_multipliers(self) -> tuple:
        """Get floor and ceiling multipliers."""
        floor = self.get('projections.floor_ceiling.floor_multiplier', 0.60)
        ceiling = self.get('projections.floor_ceiling.ceiling_multiplier', 1.50)
        return (floor, ceiling)

    def get_api_key(self, api_name: str) -> Optional[str]:
        """
        Get API key from configuration.

        Args:
            api_name: Name of the API (e.g., 'odds_api', 'weather_api')

        Returns:
            API key or None if not set
        """
        key = self.get(f'data_sources.{api_name}_key')
        if key and key.startswith('${'):
            # Environment variable not substituted (not set)
            return None
        return key

    def get_lineup_strategy_weights(self, strategy: str) -> Dict[str, float]:
        """
        Get weights for a DFS lineup strategy.

        Args:
            strategy: Strategy name (e.g., 'cash_game', 'gpp_ceiling')

        Returns:
            Dictionary of weights
        """
        return self.get(f'dfs.strategies.{strategy}', {})

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config_data.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(config_path='{self.config_path}')"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"Configuration loaded from {self.config_path}"


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern).

    Args:
        config_path: Optional path to config file (only used on first call)

    Returns:
        Config instance

    Example:
        >>> from config import get_config
        >>> config = get_config()
        >>> db_path = config.get('database.path')
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance


def reset_config() -> None:
    """Reset global configuration instance (mainly for testing)."""
    global _config_instance
    _config_instance = None


# Convenience function for common use case
def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.

    This is an alias for get_config() for backwards compatibility.

    Args:
        config_path: Optional path to config file

    Returns:
        Config instance
    """
    return get_config(config_path)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Load configuration
        print("Loading configuration...")
        config = get_config()

        print(f"\n{config}")
        print("="*60)

        # Test basic access
        print("\n1. Basic Configuration Access:")
        print(f"   Database path: {config.get('database.path')}")
        print(f"   Positions: {config.get_positions()}")
        print(f"   DFS salary cap: ${config.get_dfs_salary_cap():,}")

        # Test nested access
        print("\n2. Nested Configuration:")
        print(f"   Test size: {config.get('model.training.test_size')}")
        print(f"   N estimators: {config.get('model.training.n_estimators')}")
        print(f"   Learning rate: {config.get('model.training.learning_rate')}")

        # Test DFS configuration
        print("\n3. DFS Configuration:")
        pos_reqs = config.get_dfs_position_requirements()
        for pos, count in pos_reqs.items():
            print(f"   {pos}: {count}")

        # Test rolling windows
        print("\n4. Rolling Windows:")
        windows = config.get_rolling_windows()
        for name, window in windows.items():
            print(f"   {name}: {window} games")

        # Test confidence thresholds
        print("\n5. Confidence Thresholds:")
        thresholds = config.get_confidence_thresholds()
        for level, threshold in thresholds.items():
            print(f"   {level}: {threshold}")

        # Test matchup adjustments
        print("\n6. Matchup Adjustments:")
        min_mult, max_mult = config.get_matchup_adjustment_range()
        print(f"   Range: {min_mult} - {max_mult}")
        print(f"   Enabled: {config.is_matchup_adjustment_enabled()}")

        # Test floor/ceiling
        print("\n7. Floor/Ceiling Multipliers:")
        floor, ceiling = config.get_floor_ceiling_multipliers()
        print(f"   Floor: {floor} ({floor*100:.0f}%)")
        print(f"   Ceiling: {ceiling} ({ceiling*100:.0f}%)")

        # Test API keys
        print("\n8. API Keys:")
        odds_key = config.get_api_key('odds_api')
        print(f"   Odds API key: {'Set' if odds_key else 'Not set'}")

        # Test lineup strategies
        print("\n9. DFS Strategies:")
        for strategy in ['cash_game', 'gpp_balanced', 'gpp_ceiling']:
            weights = config.get_lineup_strategy_weights(strategy)
            if weights:
                print(f"   {strategy}:")
                for metric, weight in weights.items():
                    print(f"      {metric}: {weight}")

        # Test default values
        print("\n10. Default Values:")
        print(f"   Nonexistent key: {config.get('nonexistent.key', 'DEFAULT_VALUE')}")

        print("\n" + "="*60)
        print("✓ Configuration loaded and tested successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
