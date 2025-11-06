# Fantasy Football Projections - Test Suite

This directory contains comprehensive unit tests for the fantasy football projection system.

## Test Structure

### Test Classes

1. **TestDataFetcher** - Tests data fetching and processing
   - `test_process_player_games_non_empty` - Validates data processing returns results
   - `test_process_player_games_maintains_integrity` - Checks data integrity
   - `test_calculate_rolling_averages_produces_correct_windows` - Validates rolling averages
   - `test_calculate_rolling_averages_edge_case_single_game` - Edge case: single game
   - `test_calculate_rolling_averages_missing_data` - Edge case: missing data
   - `test_generate_mock_vegas_lines` - Tests mock Vegas line generation
   - `test_calculate_defensive_rankings_non_empty` - Validates defensive rankings

2. **TestDataValidator** - Tests data validation
   - `test_validate_player_data_valid` - Validates correct data passes
   - `test_validate_player_data_missing_columns` - Catches missing columns
   - `test_validate_player_data_invalid_positions` - Catches invalid positions
   - `test_validate_player_data_duplicates` - Catches duplicate player IDs

3. **TestMLPipeline** - Tests machine learning pipeline
   - `test_feature_engineering_creates_expected_columns` - Validates feature creation
   - `test_feature_engineering_different_positions` - Tests position-specific features
   - `test_generate_projections_returns_reasonable_values` - Validates projection ranges (0-50 pts)
   - `test_train_ensemble_model_produces_valid_models` - Tests model training

4. **TestProductionSystem** - Tests production system
   - `test_calculate_rankings_and_tiers_assigns_correct_ranks` - Validates ranking logic
   - `test_calculate_rankings_and_tiers_creates_tiers` - Tests tier creation
   - `test_calculate_rankings_value_scores` - Validates value score calculation
   - `test_identify_value_plays_finds_non_empty_results` - Tests value play identification
   - `test_generate_optimal_lineups_respects_salary_cap` - Validates salary cap constraint
   - `test_generate_optimal_lineups_respects_position_limits` - Validates position requirements

5. **TestDatabaseManager** - Tests database operations
   - `test_database_initialization_creates_all_tables` - Validates schema creation
   - `test_bulk_insert_functionality` - Tests bulk insert
   - `test_backup_functionality_works_correctly` - Tests database backup
   - `test_execute_query_with_parameters` - Tests parameterized queries
   - `test_connection_pool` - Tests connection pooling

## Running Tests

### Run All Tests

```bash
python3 -m unittest discover tests -v
```

Or use the test runner:

```bash
python3 tests/test_pipeline.py
```

### Run Specific Test Class

```bash
python3 -m unittest tests.test_pipeline.TestDataFetcher -v
python3 -m unittest tests.test_pipeline.TestDatabaseManager -v
```

### Run Specific Test Method

```bash
python3 -m unittest tests.test_pipeline.TestDataFetcher.test_process_player_games_non_empty -v
```

### Run with Coverage (if installed)

```bash
pip install coverage
coverage run -m unittest discover tests
coverage report
coverage html  # Generate HTML report
```

## Test Features

### Mock Data
- All tests use mock/synthetic data
- No external API calls required
- Tests are fast and deterministic

### Edge Cases
Tests include edge case handling for:
- Single game samples
- Missing data columns
- Empty DataFrames
- Invalid positions
- Duplicate records
- Salary cap constraints
- Position requirements

### Validation Ranges
- Projections: 0-50 points (reasonable NFL fantasy range)
- Confidence scores: 0-100%
- Value scores: Positive values
- Salary cap: ≤ $50,000
- Position counts: Match requirements (1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX)

### Fixtures
Tests use `setUp()` and `tearDown()` methods to:
- Create temporary databases
- Initialize test data
- Clean up after tests
- Ensure test isolation

## Test Coverage

Current test coverage includes:

✓ Data Fetching and Processing
✓ Data Validation
✓ Feature Engineering
✓ Model Training
✓ Projection Generation
✓ Rankings and Tiers
✓ Value Play Identification
✓ DFS Lineup Generation
✓ Database Operations
✓ Connection Pooling
✓ Backup Functionality

## Adding New Tests

### Template for New Test Class

```python
class TestNewComponent(unittest.TestCase):
    """Test suite for NewComponent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Initialize your component
        pass

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_component_functionality(self):
        """Test that component works correctly."""
        # Arrange
        test_data = self._create_test_data()

        # Act
        result = self.component.process(test_data)

        # Assert
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
```

### Best Practices

1. **Test Isolation** - Each test should be independent
2. **Descriptive Names** - Use clear, descriptive test names
3. **AAA Pattern** - Arrange, Act, Assert
4. **Mock External Dependencies** - Don't rely on external APIs
5. **Test Edge Cases** - Include boundary conditions and error cases
6. **Clean Up** - Always clean up in tearDown()
7. **Document Tests** - Include docstrings explaining what's tested

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python3 -m unittest discover tests -v
```

## Troubleshooting

### Tests Fail Due to Missing Data

Some tests may be skipped if insufficient training data is available. This is expected behavior and uses `self.skipTest()`.

### Database Lock Issues

If tests hang or fail with database locks:
- Ensure all database connections are closed in tearDown()
- Check that no background processes are accessing the test database
- Use separate database files for each test

### Memory Issues

For tests with large datasets:
- Use smaller mock datasets
- Limit the number of test samples
- Run tests individually if needed

## Performance

Test suite typically completes in:
- DatabaseManager tests: ~0.3s
- DataFetcher tests: ~0.9s
- MLPipeline tests: ~5-10s (due to model training)
- ProductionSystem tests: ~2-3s
- Total: ~8-15 seconds

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure tests pass locally
3. Maintain >80% code coverage
4. Document test purpose and edge cases
5. Update this README if needed
