# Configuration System

This directory contains the configuration system for the Fantasy Football Projections project.

## Files

- **config.yaml** - Main configuration file with all project settings
- **README.md** - This file

## Configuration Structure

The configuration is organized into the following sections:

### 1. Database Configuration
```yaml
database:
  path: fantasy_football.db
  backup_path: backups/
  pool_size: 5
```

### 2. Model Configuration
```yaml
model:
  positions: [QB, RB, WR, TE]
  min_games_for_projection: 2
  training:
    n_estimators: 200
    max_depth: 6
```

### 3. Data Sources
```yaml
data_sources:
  odds_api_key: ${ODDS_API_KEY}
  weather_api_key: ${WEATHER_API_KEY}
```

### 4. Projections
```yaml
projections:
  rolling_windows:
    short: 3
    medium: 5
  matchup_adjustment:
    min_multiplier: 0.75
    max_multiplier: 1.25
```

### 5. DFS (Daily Fantasy Sports)
```yaml
dfs:
  salary_cap: 50000
  positions:
    QB: 1
    RB: 2
    WR: 3
    TE: 1
    FLEX: 1
```

## Environment Variable Substitution

The configuration system supports environment variable substitution using the syntax:

- `${VAR_NAME}` - Required variable (must be set in environment)
- `${VAR_NAME:default}` - Variable with default value

### Example

In config.yaml:
```yaml
data_sources:
  odds_api_key: ${ODDS_API_KEY}
  cache_dir: ${CACHE_DIR:./cache}
```

In your shell:
```bash
export ODDS_API_KEY="your_api_key_here"
```

## Usage in Python

### Basic Usage

```python
from config import get_config

# Load configuration (singleton pattern)
config = get_config()

# Access configuration values
db_path = config.get('database.path')
positions = config.get('model.positions')
salary_cap = config.get('dfs.salary_cap')
```

### Using Helper Methods

```python
from config import get_config

config = get_config()

# Get position list
positions = config.get_positions()  # ['QB', 'RB', 'WR', 'TE']

# Get DFS settings
salary_cap = config.get_dfs_salary_cap()  # 50000
pos_reqs = config.get_dfs_position_requirements()  # {'QB': 1, 'RB': 2, ...}

# Get rolling windows
windows = config.get_rolling_windows()  # {'short': 3, 'medium': 5, 'long': 10}

# Get confidence thresholds
thresholds = config.get_confidence_thresholds()  # {'high': 0.8, 'medium': 0.6, 'low': 0.4}

# Get matchup adjustment range
min_mult, max_mult = config.get_matchup_adjustment_range()  # (0.75, 1.25)

# Check if adjustments are enabled
if config.is_matchup_adjustment_enabled():
    # Apply matchup adjustments
    pass

# Get API keys
odds_api_key = config.get_api_key('odds_api')
```

### Default Values

```python
# Get with default value if key doesn't exist
value = config.get('some.nonexistent.key', 'default_value')
```

### Reloading Configuration

```python
# Reload configuration from file (if changed)
config.reload()
```

## Integration Examples

### In DatabaseManager

```python
from config import get_config

class DatabaseManager:
    def __init__(self):
        config = get_config()
        db_config = config.get_database_config()

        self.db_path = db_config.get('path', 'fantasy_football.db')
        self.backup_path = db_config.get('backup_path', 'backups/')
        self.pool_size = db_config.get('pool_size', 5)
```

### In ML Pipeline

```python
from config import get_config

class FantasyProjectionPipeline:
    def __init__(self):
        config = get_config()
        model_config = config.get_model_config()

        self.positions = config.get_positions()
        self.n_estimators = model_config['training']['n_estimators']
        self.max_depth = model_config['training']['max_depth']
        self.ensemble_weights = model_config['training']['ensemble_weights']
```

### In Weekly System

```python
from config import get_config

class WeeklyProjectionSystem:
    def __init__(self):
        config = get_config()

        self.salary_cap = config.get_dfs_salary_cap()
        self.position_requirements = config.get_dfs_position_requirements()
        self.output_dir = config.get('output.directory')

        # Get lineup strategy weights
        self.cash_weights = config.get_lineup_strategy_weights('cash_game')
        self.gpp_weights = config.get_lineup_strategy_weights('gpp_ceiling')
```

## Testing Configuration

Run the config module directly to test:

```bash
python src/config.py
```

This will display all configuration values and test environment variable substitution.

## Configuration Validation

The Config class includes automatic validation:

- Environment variables are substituted recursively
- Missing required environment variables are logged as warnings
- Type checking is performed through Python's type system
- Default values can be provided for any key

## Best Practices

1. **Environment Variables**: Store sensitive data (API keys, passwords) in environment variables, not in config.yaml
2. **Version Control**: Commit config.yaml but not .env files
3. **Defaults**: Provide sensible defaults for optional configuration
4. **Documentation**: Document new configuration options in this README
5. **Testing**: Test configuration changes with `python src/config.py`

## Adding New Configuration

To add new configuration:

1. Add the setting to `config/config.yaml`
2. Optionally add a helper method to `src/config.py`
3. Update this README with documentation
4. Test with `python src/config.py`

Example:

```python
# In src/config.py
def get_new_feature_config(self) -> Dict[str, Any]:
    """Get new feature configuration."""
    return self.get('new_feature', {})
```

## Troubleshooting

### Environment variable not substituted

If you see `${VAR_NAME}` in your configuration values:
- Check that the environment variable is set: `echo $VAR_NAME`
- Set it: `export VAR_NAME=value`
- Alternatively, provide a default in config.yaml: `${VAR_NAME:default_value}`

### Configuration not loading

- Check that `config/config.yaml` exists
- Verify YAML syntax: `python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"`
- Check file permissions
- Look for error messages in logs

### PyYAML not installed

```bash
pip install pyyaml
```

## Environment Setup

Create a `.env` file in the project root:

```bash
# .env
ODDS_API_KEY=your_odds_api_key_here
WEATHER_API_KEY=your_weather_api_key_here
```

Load environment variables:

```python
from dotenv import load_dotenv
load_dotenv()

from config import get_config
config = get_config()
```

Or in shell:

```bash
source .env  # or
export $(cat .env | xargs)
```
