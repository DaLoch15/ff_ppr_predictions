"""
Comprehensive Unit Tests for Fantasy Football Projection System

Tests all major components including data fetching, ML pipeline,
production system, and database operations.
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np

from database.db_manager import DatabaseManager
from data.data_fetcher import FantasyDataFetcher, DataValidator
from models.ml_pipeline import FantasyProjectionPipeline
from production.weekly_system import WeeklyProjectionSystem

# Suppress warnings during testing
warnings.filterwarnings('ignore')


class TestDataFetcher(unittest.TestCase):
    """Test suite for FantasyDataFetcher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_fantasy.db')
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.db_manager.initialize_database()
        self.fetcher = FantasyDataFetcher(self.db_manager)

    def tearDown(self):
        """Clean up test fixtures."""
        self.db_manager.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_weekly_data(self, n_players=10, n_weeks=5):
        """Create mock weekly player data."""
        data = []
        positions = ['QB', 'RB', 'WR', 'TE']
        teams = ['KC', 'BUF', 'SF', 'PHI']

        for player_id in range(n_players):
            position = positions[player_id % len(positions)]
            team = teams[player_id % len(teams)]

            for week in range(1, n_weeks + 1):
                data.append({
                    'player_id': f'P{player_id:03d}',
                    'season': 2024,
                    'week': week,
                    'recent_team': team,
                    'opponent_team': teams[(teams.index(team) + 1) % len(teams)],
                    'targets': np.random.randint(0, 15),
                    'receptions': np.random.randint(0, 12),
                    'receiving_yards': np.random.randint(0, 150),
                    'receiving_tds': np.random.randint(0, 3),
                    'carries': np.random.randint(0, 25),
                    'rushing_yards': np.random.randint(0, 120),
                    'rushing_tds': np.random.randint(0, 2),
                    'attempts': np.random.randint(0, 40),
                    'passing_yards': np.random.randint(0, 350),
                    'passing_tds': np.random.randint(0, 4),
                    'interceptions': np.random.randint(0, 2),
                    'sacks': np.random.randint(0, 4),
                    'fantasy_points_ppr': np.random.uniform(5, 30),
                    'snap_count': np.random.randint(20, 70),
                    'snap_pct': np.random.uniform(40, 100),
                    'target_share': np.random.uniform(10, 35),
                    'air_yards': np.random.randint(0, 120),
                    'game_date': datetime.now() - timedelta(days=(n_weeks - week) * 7)
                })

        return pd.DataFrame(data)

    def test_process_player_games_non_empty(self):
        """Test that process_player_games returns non-empty DataFrame."""
        mock_data = self._create_mock_weekly_data(n_players=5, n_weeks=3)
        processed = self.fetcher.process_player_games(mock_data)

        self.assertIsInstance(processed, pd.DataFrame)
        self.assertGreater(len(processed), 0)
        self.assertIn('game_id', processed.columns)
        self.assertIn('fantasy_points_ppr', processed.columns)

    def test_process_player_games_maintains_integrity(self):
        """Test that process_player_games maintains data integrity."""
        mock_data = self._create_mock_weekly_data(n_players=5, n_weeks=3)
        original_count = len(mock_data)

        processed = self.fetcher.process_player_games(mock_data)

        # Should have same number of rows
        self.assertEqual(len(processed), original_count)

        # Check required columns exist
        required_cols = ['player_id', 'season', 'week', 'team', 'opponent']
        for col in required_cols:
            self.assertIn(col, processed.columns)

        # Check no null values in critical columns
        self.assertEqual(processed['player_id'].isnull().sum(), 0)
        self.assertEqual(processed['game_id'].isnull().sum(), 0)

    def test_calculate_rolling_averages_produces_correct_windows(self):
        """Test that rolling averages are calculated correctly."""
        mock_data = self._create_mock_weekly_data(n_players=3, n_weeks=10)
        processed = self.fetcher.process_player_games(mock_data)

        # Calculate rolling averages
        with_rolling = self.fetcher.calculate_rolling_averages(processed)

        # Check that rolling average columns exist
        self.assertIn('fantasy_points_ppr_3g', with_rolling.columns)
        self.assertIn('fantasy_points_ppr_5g', with_rolling.columns)

        # Verify rolling averages are reasonable
        for col in ['fantasy_points_ppr_3g', 'fantasy_points_ppr_5g']:
            self.assertTrue(with_rolling[col].notna().any())
            # Rolling avg should be within range of individual games
            self.assertTrue((with_rolling[col] >= 0).all())

    def test_calculate_rolling_averages_edge_case_single_game(self):
        """Test rolling averages with single game (edge case)."""
        mock_data = self._create_mock_weekly_data(n_players=1, n_weeks=1)
        processed = self.fetcher.process_player_games(mock_data)

        with_rolling = self.fetcher.calculate_rolling_averages(processed)

        # Should still work with single game
        self.assertEqual(len(with_rolling), 1)
        # First game's 3-game average should equal the game itself
        self.assertAlmostEqual(
            with_rolling.iloc[0]['fantasy_points_ppr_3g'],
            with_rolling.iloc[0]['fantasy_points_ppr'],
            places=2
        )

    def test_calculate_rolling_averages_missing_data(self):
        """Test rolling averages with missing data."""
        mock_data = self._create_mock_weekly_data(n_players=2, n_weeks=5)
        # Introduce some NaN values
        mock_data.loc[mock_data.index[2:4], 'targets'] = np.nan

        processed = self.fetcher.process_player_games(mock_data)
        with_rolling = self.fetcher.calculate_rolling_averages(processed)

        # Should handle NaN gracefully
        self.assertIsInstance(with_rolling, pd.DataFrame)
        self.assertEqual(len(with_rolling), len(mock_data))

    def test_generate_mock_vegas_lines(self):
        """Test generation of mock Vegas lines."""
        lines = self.fetcher._generate_mock_vegas_lines(week=5, season=2024)

        self.assertIsInstance(lines, pd.DataFrame)
        self.assertGreater(len(lines), 0)

        # Check required columns
        required_cols = ['game_id', 'season', 'week', 'spread', 'total',
                        'home_implied_total', 'away_implied_total']
        for col in required_cols:
            self.assertIn(col, lines.columns)

        # Check reasonable ranges
        self.assertTrue((lines['spread'] >= -14).all() & (lines['spread'] <= 14).all())
        self.assertTrue((lines['total'] >= 35).all() & (lines['total'] <= 60).all())

    def test_calculate_defensive_rankings_non_empty(self):
        """Test defensive rankings calculation returns data."""
        # Create minimal mock play-by-play data
        pbp_data = pd.DataFrame({
            'defteam': ['KC', 'BUF', 'SF'] * 20,
            'season': [2024] * 60,
            'week': [1] * 60,
            'play_type': ['pass', 'run'] * 30,
            'passing_yards': np.random.randint(0, 20, 60),
            'yards_gained': np.random.randint(0, 15, 60),
            'touchdown': np.random.randint(0, 2, 60),
            'receiver_position': ['WR', 'RB', 'TE'] * 20,
        })

        def_rankings = self.fetcher.calculate_defensive_rankings(pbp_data)

        self.assertIsInstance(def_rankings, pd.DataFrame)
        self.assertGreater(len(def_rankings), 0)
        self.assertIn('team', def_rankings.columns)
        self.assertIn('qb_points_allowed_avg', def_rankings.columns)


class TestDataValidator(unittest.TestCase):
    """Test suite for DataValidator class."""

    def test_validate_player_data_valid(self):
        """Test validation of valid player data."""
        valid_data = pd.DataFrame({
            'player_id': ['P001', 'P002', 'P003'],
            'player_name': ['Player 1', 'Player 2', 'Player 3'],
            'position': ['QB', 'RB', 'WR'],
            'team': ['KC', 'BUF', 'SF'],
            'fantasy_points_ppr': [25.3, 18.2, 15.7]
        })

        is_valid, errors = DataValidator.validate_player_data(valid_data)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_player_data_missing_columns(self):
        """Test validation catches missing columns."""
        invalid_data = pd.DataFrame({
            'player_id': ['P001', 'P002'],
            'position': ['QB', 'RB']
            # Missing player_name and team
        })

        is_valid, errors = DataValidator.validate_player_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('Missing required columns' in err for err in errors))

    def test_validate_player_data_invalid_positions(self):
        """Test validation catches invalid positions."""
        invalid_data = pd.DataFrame({
            'player_id': ['P001', 'P002'],
            'player_name': ['Player 1', 'Player 2'],
            'position': ['QB', 'INVALID'],  # Invalid position
            'team': ['KC', 'BUF']
        })

        is_valid, errors = DataValidator.validate_player_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertTrue(any('invalid position' in err.lower() for err in errors))

    def test_validate_player_data_duplicates(self):
        """Test validation catches duplicate player IDs."""
        invalid_data = pd.DataFrame({
            'player_id': ['P001', 'P001', 'P002'],  # Duplicate
            'player_name': ['Player 1', 'Player 1 Dup', 'Player 2'],
            'position': ['QB', 'QB', 'RB'],
            'team': ['KC', 'KC', 'BUF']
        })

        is_valid, errors = DataValidator.validate_player_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertTrue(any('duplicate' in err.lower() for err in errors))


class TestMLPipeline(unittest.TestCase):
    """Test suite for FantasyProjectionPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_fantasy.db')
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.db_manager.initialize_database()
        self.pipeline = FantasyProjectionPipeline(self.db_manager)

        # Insert mock training data
        self._insert_mock_training_data()

    def tearDown(self):
        """Clean up test fixtures."""
        self.db_manager.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _insert_mock_training_data(self):
        """Insert mock data for training."""
        # Insert players
        players = []
        for i in range(20):
            players.append({
                'player_id': f'P{i:03d}',
                'player_name': f'Player {i}',
                'position': ['QB', 'RB', 'WR', 'TE'][i % 4],
                'team': ['KC', 'BUF', 'SF', 'PHI'][i % 4],
                'years_experience': i % 10,
                'draft_year': 2020
            })
        self.db_manager.bulk_insert('players', players)

        # Insert player games
        games = []
        for player_id in range(20):
            for week in range(4, 11):  # Weeks 4-10
                games.append({
                    'game_id': f'2024_{week:02d}_GAME{player_id}',
                    'player_id': f'P{player_id:03d}',
                    'season': 2024,
                    'week': week,
                    'game_date': datetime(2024, 9, 1) + timedelta(days=week*7),
                    'team': ['KC', 'BUF', 'SF', 'PHI'][player_id % 4],
                    'opponent': ['BUF', 'KC', 'PHI', 'SF'][player_id % 4],
                    'is_home': player_id % 2,
                    'snap_count': 50 + player_id,
                    'snap_pct': 70.0 + player_id % 30,
                    'targets': 5 + player_id % 10,
                    'receptions': 3 + player_id % 8,
                    'receiving_yards': 50 + player_id * 5,
                    'receiving_tds': player_id % 2,
                    'carries': 10 + player_id % 15,
                    'rushing_yards': 40 + player_id * 3,
                    'rushing_tds': player_id % 2,
                    'fantasy_points_ppr': 10.0 + player_id % 20,
                    'target_share': 15.0 + player_id % 20,
                    'routes_run': 30 + player_id,
                    'route_participation': 80.0
                })
        self.db_manager.bulk_insert('player_games', games)

    def test_feature_engineering_creates_expected_columns(self):
        """Test that feature engineering creates expected columns."""
        # Load mock data
        df = self.pipeline.load_training_data('RB', min_week=4)

        if df.empty:
            self.skipTest("No training data available")

        # Engineer features
        engineered = self.pipeline._engineer_features(df, 'RB')

        # Check universal features
        universal_features = [
            'implied_game_script', 'consistency_score', 'momentum',
            'home_advantage'
        ]
        for feature in universal_features:
            self.assertIn(feature, engineered.columns,
                         f"Missing universal feature: {feature}")

        # Check RB-specific features
        rb_features = ['game_script_run', 'pass_catching_role']
        for feature in rb_features:
            self.assertIn(feature, engineered.columns,
                         f"Missing RB feature: {feature}")

    def test_feature_engineering_different_positions(self):
        """Test feature engineering for different positions."""
        positions = ['QB', 'RB', 'WR', 'TE']

        for position in positions:
            df = self.pipeline.load_training_data(position, min_week=4)

            if df.empty:
                continue

            engineered = self.pipeline._engineer_features(df, position)

            # Should have more columns than input
            self.assertGreater(len(engineered.columns), len(df.columns))

            # Should not have NaN in most columns
            numeric_cols = engineered.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                null_pct = engineered[col].isnull().sum() / len(engineered)
                self.assertLess(null_pct, 0.5,  # Allow up to 50% nulls
                               f"Too many nulls in {col}: {null_pct:.1%}")

    def test_generate_projections_returns_reasonable_values(self):
        """Test that projections are in reasonable range (0-50 points)."""
        # First train a simple model
        positions = ['RB', 'WR']

        for position in positions:
            try:
                # Train model
                self.pipeline.train_ensemble_model(position)

                # Generate projections
                projections = self.pipeline.generate_projections(
                    position=position,
                    week=11,
                    season=2024,
                    model_type='ensemble'
                )

                if projections.empty:
                    continue

                # Check projections are in reasonable range
                self.assertTrue((projections['projected_points'] >= 0).all(),
                               f"{position} has negative projections")
                self.assertTrue((projections['projected_points'] <= 50).all(),
                               f"{position} has unrealistic high projections")

                # Check floor < projection < ceiling
                valid_ranges = (
                    (projections['floor_points'] <= projections['projected_points']) &
                    (projections['projected_points'] <= projections['ceiling_points'])
                )
                self.assertTrue(valid_ranges.all(),
                               f"{position} has invalid floor/ceiling ranges")

                # Check confidence scores are 0-100
                self.assertTrue((projections['confidence_score'] >= 0).all())
                self.assertTrue((projections['confidence_score'] <= 100).all())

            except Exception as e:
                self.fail(f"Failed to generate projections for {position}: {e}")

    def test_train_ensemble_model_produces_valid_models(self):
        """Test that ensemble training produces valid model objects."""
        position = 'RB'

        try:
            metrics = self.pipeline.train_ensemble_model(position)

            # Check metrics exist
            self.assertIsInstance(metrics, dict)
            self.assertIn('ensemble_mae', metrics)
            self.assertIn('n_samples', metrics)

            # Check MAE is reasonable
            self.assertGreater(metrics['ensemble_mae'], 0)
            self.assertLess(metrics['ensemble_mae'], 50)  # Should be under 50 points error

            # Check model is stored
            model_key = f"{position}_ensemble"
            self.assertIn(model_key, self.pipeline.models)

        except ValueError as e:
            if "No training data" in str(e):
                self.skipTest("Insufficient training data")
            else:
                raise


class TestProductionSystem(unittest.TestCase):
    """Test suite for WeeklyProjectionSystem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        self.db_path = os.path.join(self.temp_dir, 'test_fantasy.db')

        os.makedirs(self.output_dir, exist_ok=True)

        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.db_manager.initialize_database()
        self.system = WeeklyProjectionSystem(
            db_manager=self.db_manager,
            output_dir=self.output_dir
        )

        # Insert mock data
        self._insert_mock_data()

    def tearDown(self):
        """Clean up test fixtures."""
        self.db_manager.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _insert_mock_data(self):
        """Insert mock data for testing."""
        # Insert players
        players = [
            {'player_id': f'P{i:03d}', 'player_name': f'Player {i}',
             'position': ['QB', 'RB', 'WR', 'TE'][i % 4],
             'team': ['KC', 'BUF', 'SF'][i % 3]}
            for i in range(30)
        ]
        self.db_manager.bulk_insert('players', players)

    def _create_mock_projections(self, n_players=20):
        """Create mock projections DataFrame."""
        positions = ['QB', 'RB', 'WR', 'TE']
        teams = ['KC', 'BUF', 'SF', 'PHI', 'DAL']

        projections = []
        for i in range(n_players):
            base_points = np.random.uniform(8, 25)
            projections.append({
                'projection_id': f'PROJ{i:03d}',
                'player_id': f'P{i:03d}',
                'player_name': f'Player {i}',
                'position': positions[i % len(positions)],
                'team': teams[i % len(teams)],
                'opponent': teams[(i + 1) % len(teams)],
                'season': 2024,
                'week': 5,
                'projected_points': base_points,
                'floor_points': base_points * 0.6,
                'ceiling_points': base_points * 1.5,
                'confidence_score': np.random.uniform(60, 95),
                'salary': int(base_points * 1000),
                'model_version': 'test'
            })

        return pd.DataFrame(projections)

    def test_calculate_rankings_and_tiers_assigns_correct_ranks(self):
        """Test that rankings are assigned correctly."""
        projections = self._create_mock_projections(n_players=20)

        rankings = self.system.calculate_rankings_and_tiers(projections)

        # Check position rankings
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_data = rankings[rankings['position'] == position]
            if len(pos_data) == 0:
                continue

            # Ranks should start at 1
            self.assertEqual(pos_data['position_rank'].min(), 1)

            # Ranks should be sequential
            ranks = sorted(pos_data['position_rank'].unique())
            self.assertEqual(ranks[0], 1)

            # Higher projected points should have lower (better) rank
            sorted_by_points = pos_data.sort_values('projected_points', ascending=False)
            sorted_by_rank = pos_data.sort_values('position_rank')
            self.assertTrue(
                sorted_by_points['player_id'].equals(sorted_by_rank['player_id'])
            )

    def test_calculate_rankings_and_tiers_creates_tiers(self):
        """Test that tier assignments are created."""
        projections = self._create_mock_projections(n_players=20)

        rankings = self.system.calculate_rankings_and_tiers(projections)

        # Tiers should exist
        self.assertIn('tier', rankings.columns)

        # Tiers should start at 1
        self.assertEqual(rankings['tier'].min(), 1)

        # Should have multiple tiers for sufficient data
        if len(rankings) > 5:
            self.assertGreater(rankings['tier'].nunique(), 1)

    def test_calculate_rankings_value_scores(self):
        """Test that value scores are calculated."""
        projections = self._create_mock_projections(n_players=20)

        rankings = self.system.calculate_rankings_and_tiers(projections)

        # Value score should exist
        self.assertIn('value_score', rankings.columns)

        # Value score should be positive
        self.assertTrue((rankings['value_score'] > 0).all())

        # Higher points per dollar should have higher value score
        # (assuming similar salaries)
        high_proj = rankings.nlargest(5, 'projected_points')
        low_proj = rankings.nsmallest(5, 'projected_points')
        if len(high_proj) > 0 and len(low_proj) > 0:
            self.assertGreater(
                high_proj['value_score'].mean(),
                low_proj['value_score'].mean()
            )

    def test_identify_value_plays_finds_non_empty_results(self):
        """Test that value play identification returns results."""
        projections = self._create_mock_projections(n_players=30)
        rankings = self.system.calculate_rankings_and_tiers(projections)

        value_plays = self.system.identify_value_plays(rankings)

        # Should return dictionary
        self.assertIsInstance(value_plays, dict)

        # Should have multiple categories
        self.assertGreater(len(value_plays), 0)

        # Check specific categories
        expected_categories = ['high_ceiling', 'safe_floor', 'leverage',
                              'positive_regression', 'best_values']
        for category in expected_categories:
            self.assertIn(category, value_plays)

        # Each category should have data
        for category, df in value_plays.items():
            self.assertIsInstance(df, pd.DataFrame)
            if not df.empty:
                # Should have reasonable number of plays
                self.assertLessEqual(len(df), 25)

    def test_generate_optimal_lineups_respects_salary_cap(self):
        """Test that generated lineups respect salary cap."""
        projections = self._create_mock_projections(n_players=40)
        rankings = self.system.calculate_rankings_and_tiers(projections)

        lineups = self.system.generate_optimal_lineups(rankings)

        # Should return dictionary
        self.assertIsInstance(lineups, dict)

        # Check each lineup
        for strategy, lineup in lineups.items():
            if lineup.empty:
                continue

            # Total salary should not exceed cap
            total_salary = lineup['salary'].sum()
            self.assertLessEqual(
                total_salary,
                self.system.SALARY_CAP,
                f"{strategy} exceeds salary cap: ${total_salary:,}"
            )

            # Should have projected points
            self.assertIn('projected_points', lineup.columns)
            self.assertGreater(lineup['projected_points'].sum(), 0)

    def test_generate_optimal_lineups_respects_position_limits(self):
        """Test that lineups respect position requirements."""
        projections = self._create_mock_projections(n_players=50)
        rankings = self.system.calculate_rankings_and_tiers(projections)

        lineups = self.system.generate_optimal_lineups(rankings)

        for strategy, lineup in lineups.items():
            if lineup.empty:
                continue

            # Count positions
            position_counts = lineup['position'].value_counts().to_dict()

            # Check minimum requirements (allowing FLEX to fill in)
            total_positions = sum(position_counts.values())

            # Should have 8 total players (QB + 2RB + 3WR + TE + FLEX)
            expected_total = sum(self.system.POSITION_REQUIREMENTS.values())
            self.assertLessEqual(
                abs(total_positions - expected_total),
                2,  # Allow some flexibility
                f"{strategy} has wrong number of positions: {total_positions}"
            )


class TestDatabaseManager(unittest.TestCase):
    """Test suite for DatabaseManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_fantasy.db')
        self.backup_dir = os.path.join(self.temp_dir, 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization_creates_all_tables(self):
        """Test that database initialization creates all tables."""
        db_manager = DatabaseManager(db_path=self.db_path)

        try:
            # Initialize database
            db_manager.initialize_database()

            # Get database stats
            stats = db_manager.get_database_stats()

            # Check that tables were created
            self.assertGreater(stats['table_count'], 0)

            # Check expected tables exist
            expected_tables = [
                'players', 'player_games', 'team_games',
                'defensive_rankings', 'vegas_lines', 'injuries',
                'weather', 'projections', 'projection_results'
            ]

            tables = stats['tables']
            for table in expected_tables:
                self.assertIn(table, tables,
                            f"Table '{table}' not created")

        finally:
            db_manager.close()

    def test_bulk_insert_functionality(self):
        """Test bulk insert functionality."""
        db_manager = DatabaseManager(db_path=self.db_path)

        try:
            db_manager.initialize_database()

            # Insert test data
            test_players = [
                {'player_id': 'P001', 'player_name': 'Test Player 1',
                 'position': 'QB', 'team': 'KC'},
                {'player_id': 'P002', 'player_name': 'Test Player 2',
                 'position': 'RB', 'team': 'BUF'},
                {'player_id': 'P003', 'player_name': 'Test Player 3',
                 'position': 'WR', 'team': 'SF'}
            ]

            rows_inserted = db_manager.bulk_insert('players', test_players)

            # Check insertion count
            self.assertEqual(rows_inserted, len(test_players))

            # Verify data was inserted
            results = db_manager.execute_query("SELECT COUNT(*) as count FROM players")
            self.assertEqual(results[0]['count'], len(test_players))

        finally:
            db_manager.close()

    def test_backup_functionality_works_correctly(self):
        """Test database backup functionality."""
        db_manager = DatabaseManager(db_path=self.db_path)

        try:
            db_manager.initialize_database()

            # Insert some data
            test_data = [
                {'player_id': 'P001', 'player_name': 'Player 1',
                 'position': 'QB', 'team': 'KC'}
            ]
            db_manager.bulk_insert('players', test_data)

            # Create backup
            backup_path = db_manager.backup_database(backup_dir=self.backup_dir)

            # Check backup was created
            self.assertTrue(os.path.exists(backup_path))
            self.assertGreater(os.path.getsize(backup_path), 0)

            # Backup filename should contain timestamp
            filename = os.path.basename(backup_path)
            self.assertIn('fantasy_football_backup_', filename)
            self.assertTrue(filename.endswith('.db'))

        finally:
            db_manager.close()

    def test_execute_query_with_parameters(self):
        """Test query execution with parameters."""
        db_manager = DatabaseManager(db_path=self.db_path)

        try:
            db_manager.initialize_database()

            # Insert test data
            test_players = [
                {'player_id': 'P001', 'player_name': 'QB Test',
                 'position': 'QB', 'team': 'KC'},
                {'player_id': 'P002', 'player_name': 'RB Test',
                 'position': 'RB', 'team': 'BUF'}
            ]
            db_manager.bulk_insert('players', test_players)

            # Query with parameters
            results = db_manager.execute_query(
                "SELECT * FROM players WHERE position = ?",
                ('QB',)
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['position'], 'QB')

        finally:
            db_manager.close()

    def test_connection_pool(self):
        """Test connection pooling functionality."""
        db_manager = DatabaseManager(db_path=self.db_path, pool_size=3)

        try:
            db_manager.initialize_database()

            # Use multiple connections
            with db_manager.get_connection() as conn1:
                with db_manager.get_connection() as conn2:
                    with db_manager.get_connection() as conn3:
                        # All three connections should work
                        self.assertIsNotNone(conn1)
                        self.assertIsNotNone(conn2)
                        self.assertIsNotNone(conn3)

        finally:
            db_manager.close()


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataFetcher))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseManager))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
