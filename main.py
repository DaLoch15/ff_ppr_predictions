#!/usr/bin/env python3
"""
Fantasy Football Projections - Main Entry Point

This is the main command-line interface for the fantasy football projection system.
Provides commands for setup, data updates, model training, and backtesting.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import project modules
from src.database.db_manager import DatabaseManager
from src.data.data_fetcher import FantasyDataFetcher
from src.models.ml_pipeline import FantasyProjectionPipeline
from src.production.weekly_system import WeeklyProjectionSystem


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_project(db_path: str = 'fantasy_football.db') -> bool:
    """
    Initialize project with database schema and required directories.

    Args:
        db_path: Path to SQLite database file

    Returns:
        bool: True if setup successful, False otherwise

    Example:
        >>> setup_project()
    """
    try:
        logger.info("="*60)
        logger.info("FANTASY FOOTBALL PROJECTIONS - PROJECT SETUP")
        logger.info("="*60)

        # Create required directories
        logger.info("\nStep 1: Creating directory structure...")
        directories = [
            'output',
            'backups',
            'logs',
            'models',
            'data/cache'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"  ✓ Created: {directory}/")

        # Initialize database
        logger.info("\nStep 2: Initializing database...")
        db_manager = DatabaseManager(db_path=db_path)

        try:
            db_manager.initialize_database()
            logger.info("  ✓ Database schema initialized")

            # Verify database structure
            stats = db_manager.get_database_stats()
            logger.info(f"  ✓ Database created: {stats['database_path']}")
            logger.info(f"  ✓ Tables created: {stats['table_count']}")

        finally:
            db_manager.close()

        # Validate environment
        logger.info("\nStep 3: Validating environment...")

        # Check Python version
        python_version = sys.version_info
        logger.info(f"  ✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Check required packages
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'xgboost',
            'nfl_data_py', 'requests', 'dotenv'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('_', '.') if '.' in package else package)
                logger.info(f"  ✓ Package installed: {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"  ✗ Package missing: {package}")

        if missing_packages:
            logger.warning("\nMissing packages detected. Install with:")
            logger.warning(f"  pip install {' '.join(missing_packages)}")

        # Create .env template if it doesn't exist
        logger.info("\nStep 4: Checking environment configuration...")
        env_file = Path('.env')

        if not env_file.exists():
            logger.info("  Creating .env template...")
            with open(env_file, 'w') as f:
                f.write("# Fantasy Football Projections - Environment Variables\n\n")
                f.write("# The Odds API key (get from https://the-odds-api.com/)\n")
                f.write("ODDS_API_KEY=your_api_key_here\n\n")
                f.write("# Database configuration\n")
                f.write("DATABASE_PATH=fantasy_football.db\n\n")
            logger.info("  ✓ Created .env template")
            logger.info("  ⚠  Edit .env file and add your API keys")
        else:
            logger.info("  ✓ .env file exists")

        logger.info("\n" + "="*60)
        logger.info("✓ PROJECT SETUP COMPLETE")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("  1. Edit .env file and add your API keys (optional)")
        logger.info("  2. Fetch data: python main.py fetch --week 1 --season 2024")
        logger.info("  3. Train models: python main.py train")
        logger.info("  4. Run projections: python main.py update --week 1 --season 2024")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        return False


def run_weekly_update(
    week: int,
    season: int,
    db_path: str = 'fantasy_football.db',
    train: bool = False,
    fetch: bool = True
) -> dict:
    """
    Run complete weekly projection workflow.

    Args:
        week: NFL week number
        season: NFL season year
        db_path: Path to database file
        train: Whether to train models before projections
        fetch: Whether to fetch new data

    Returns:
        Dictionary with results

    Example:
        >>> results = run_weekly_update(week=5, season=2024)
    """
    try:
        logger.info("="*60)
        logger.info(f"WEEKLY PROJECTION UPDATE - {season} Week {week}")
        logger.info("="*60)

        # Initialize database manager
        db_manager = DatabaseManager(db_path=db_path)

        try:
            # Create weekly projection system
            system = WeeklyProjectionSystem(
                db_manager=db_manager,
                output_dir='output',
                log_file=f'logs/weekly_{season}_w{week}.log'
            )

            # Run projections
            results = system.run_weekly_projections(
                week=week,
                season=season,
                update_data=fetch,
                train_models=train
            )

            if results['success']:
                logger.info("\n" + "="*60)
                logger.info("✓ WEEKLY UPDATE COMPLETED SUCCESSFULLY")
                logger.info("="*60)
                logger.info(f"Duration: {results['duration_seconds']:.2f}s")
                logger.info(f"Projections generated: {len(results.get('projections', []))}")

                if 'export_paths' in results:
                    logger.info("\nOutput files:")
                    for key, path in results['export_paths'].items():
                        logger.info(f"  • {key}: {path}")

                return results
            else:
                logger.error("\n✗ Weekly update failed")
                logger.error(f"Error: {results.get('error', 'Unknown error')}")
                return results

        finally:
            db_manager.close()

    except Exception as e:
        logger.error(f"Error in weekly update: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'week': week,
            'season': season
        }


def train_models(
    db_path: str = 'fantasy_football.db',
    positions: list = None
) -> dict:
    """
    Train ML models for fantasy projections.

    Args:
        db_path: Path to database file
        positions: List of positions to train (default: all)

    Returns:
        Dictionary with training results

    Example:
        >>> results = train_models()
    """
    try:
        logger.info("="*60)
        logger.info("MODEL TRAINING")
        logger.info("="*60)

        if positions is None:
            positions = ['QB', 'RB', 'WR', 'TE']

        # Initialize database manager
        db_manager = DatabaseManager(db_path=db_path)

        try:
            # Create ML pipeline
            pipeline = FantasyProjectionPipeline(db_manager)

            training_results = {}

            for position in positions:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {position} Models")
                logger.info('='*60)

                # Train ensemble model
                logger.info(f"\n1. Training {position} Ensemble Model...")
                try:
                    ensemble_metrics = pipeline.train_ensemble_model(position)
                    training_results[f'{position}_ensemble'] = ensemble_metrics

                    logger.info(f"\n✓ {position} Ensemble Model Trained:")
                    logger.info(f"  MAE: {ensemble_metrics['ensemble_mae']:.3f}")
                    logger.info(f"  Std: {ensemble_metrics['ensemble_std']:.3f}")
                    logger.info(f"  Samples: {ensemble_metrics['n_samples']}")
                    logger.info(f"  Features: {ensemble_metrics['n_features']}")

                except Exception as e:
                    logger.error(f"✗ Failed to train {position} ensemble: {e}")
                    training_results[f'{position}_ensemble'] = {'error': str(e)}

                # Train two-stage model
                logger.info(f"\n2. Training {position} Two-Stage Model...")
                try:
                    two_stage_metrics = pipeline.train_two_stage_model(position)
                    training_results[f'{position}_two_stage'] = two_stage_metrics

                    logger.info(f"\n✓ {position} Two-Stage Model Trained:")
                    logger.info(f"  Volume MAE: {two_stage_metrics['volume_mae']:.3f}")
                    logger.info(f"  Efficiency MAE: {two_stage_metrics['efficiency_mae']:.3f}")
                    logger.info(f"  Efficiency RMSE: {two_stage_metrics['efficiency_rmse']:.3f}")

                except Exception as e:
                    logger.error(f"✗ Failed to train {position} two-stage: {e}")
                    training_results[f'{position}_two_stage'] = {'error': str(e)}

            logger.info("\n" + "="*60)
            logger.info("✓ MODEL TRAINING COMPLETE")
            logger.info("="*60)

            # Summary
            successful = sum(1 for v in training_results.values() if 'error' not in v)
            total = len(training_results)
            logger.info(f"\nResults: {successful}/{total} models trained successfully")

            return {
                'success': successful > 0,
                'results': training_results,
                'timestamp': datetime.now().isoformat()
            }

        finally:
            db_manager.close()

    except Exception as e:
        logger.error(f"Error in model training: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def backtest(
    weeks: int,
    db_path: str = 'fantasy_football.db',
    positions: list = None
) -> dict:
    """
    Run backtest on historical data.

    Args:
        weeks: Number of previous weeks to test
        db_path: Path to database file
        positions: List of positions to test (default: all)

    Returns:
        Dictionary with backtest results

    Example:
        >>> results = backtest(weeks=4)
    """
    try:
        logger.info("="*60)
        logger.info(f"BACKTESTING - Last {weeks} Weeks")
        logger.info("="*60)

        if positions is None:
            positions = ['QB', 'RB', 'WR', 'TE']

        # Initialize database manager
        db_manager = DatabaseManager(db_path=db_path)

        try:
            # Create ML pipeline
            pipeline = FantasyProjectionPipeline(db_manager)

            # Calculate start date (approximate)
            current_year = datetime.now().year
            current_month = datetime.now().month

            # Determine season and approximate week
            if current_month >= 9:
                season = current_year
                current_week = min((current_month - 9) * 4 + 1, 18)
            else:
                season = current_year - 1
                current_week = 18

            start_week = max(1, current_week - weeks)

            backtest_results = {}

            for position in positions:
                logger.info(f"\n{'='*60}")
                logger.info(f"Backtesting {position}")
                logger.info('='*60)

                try:
                    results = pipeline.backtest_projections(
                        position=position,
                        start_season=season,
                        start_week=start_week,
                        end_season=season,
                        end_week=current_week - 1,
                        model_type='ensemble'
                    )

                    backtest_results[position] = results

                    if results:
                        logger.info(f"\n✓ {position} Backtest Results:")
                        logger.info(f"  Projections: {results.get('n_projections', 0)}")
                        logger.info(f"  MAE: {results.get('mae', 0):.3f}")
                        logger.info(f"  RMSE: {results.get('rmse', 0):.3f}")
                        logger.info(f"  Within Range: {results.get('within_range_pct', 0):.1f}%")
                        logger.info(f"  Date Range: {results.get('date_range', 'N/A')}")
                    else:
                        logger.warning(f"✗ No backtest results for {position}")

                except Exception as e:
                    logger.error(f"✗ Backtest failed for {position}: {e}")
                    backtest_results[position] = {'error': str(e)}

            logger.info("\n" + "="*60)
            logger.info("✓ BACKTEST COMPLETE")
            logger.info("="*60)

            # Overall summary
            successful = sum(1 for v in backtest_results.values() if v and 'error' not in v)
            logger.info(f"\nResults: {successful}/{len(positions)} positions tested successfully")

            if successful > 0:
                avg_mae = sum(r.get('mae', 0) for r in backtest_results.values() if r and 'mae' in r) / successful
                logger.info(f"Average MAE: {avg_mae:.3f}")

            return {
                'success': successful > 0,
                'results': backtest_results,
                'weeks_tested': weeks,
                'timestamp': datetime.now().isoformat()
            }

        finally:
            db_manager.close()

    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def fetch_data(
    week: int,
    season: int,
    db_path: str = 'fantasy_football.db'
) -> dict:
    """
    Fetch and update data without generating projections.

    Args:
        week: NFL week number
        season: NFL season year
        db_path: Path to database file

    Returns:
        Dictionary with fetch results

    Example:
        >>> results = fetch_data(week=5, season=2024)
    """
    try:
        logger.info("="*60)
        logger.info(f"DATA FETCH - {season} Week {week}")
        logger.info("="*60)

        # Initialize database manager
        db_manager = DatabaseManager(db_path=db_path)

        try:
            # Create data fetcher
            fetcher = FantasyDataFetcher(db_manager)

            # Run weekly update
            logger.info("\nFetching NFL data...")
            results = fetcher.run_weekly_update(week, season)

            if results['success']:
                logger.info("\n✓ DATA FETCH COMPLETE")
                logger.info(f"Duration: {results['duration_seconds']:.2f}s")
                logger.info(f"Total rows inserted: {results['total_rows']}")
                logger.info("\nBreakdown:")
                for table, count in results['insert_counts'].items():
                    logger.info(f"  • {table}: {count} rows")

                return results
            else:
                logger.error("\n✗ Data fetch failed")
                logger.error(f"Error: {results.get('error', 'Unknown error')}")
                return results

        finally:
            db_manager.close()

    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'week': week,
            'season': season
        }


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Fantasy Football Projection System',
        epilog="""
Examples:
  # Initial setup
  python main.py setup

  # Fetch data for a week
  python main.py fetch --week 5 --season 2024

  # Train all models
  python main.py train

  # Train specific positions
  python main.py train --positions QB RB

  # Run weekly projections (with data fetch)
  python main.py update --week 5 --season 2024

  # Run projections with model training
  python main.py update --week 5 --season 2024 --train

  # Run projections without fetching new data
  python main.py update --week 5 --season 2024 --no-fetch

  # Backtest on last 4 weeks
  python main.py backtest --weeks 4

  # Backtest specific positions
  python main.py backtest --weeks 4 --positions RB WR

For more information, visit: https://github.com/yourusername/fantasy-football-projections
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Setup command
    setup_parser = subparsers.add_parser(
        'setup',
        help='Initialize project with database and directories'
    )
    setup_parser.add_argument(
        '--db-path',
        default='fantasy_football.db',
        help='Path to database file (default: fantasy_football.db)'
    )

    # Update command
    update_parser = subparsers.add_parser(
        'update',
        help='Run weekly projections'
    )
    update_parser.add_argument(
        '--week',
        type=int,
        required=True,
        help='NFL week number (1-18)'
    )
    update_parser.add_argument(
        '--season',
        type=int,
        required=True,
        help='NFL season year (e.g., 2024)'
    )
    update_parser.add_argument(
        '--train',
        action='store_true',
        help='Train models before generating projections'
    )
    update_parser.add_argument(
        '--no-fetch',
        action='store_true',
        help='Skip data fetching, use existing data'
    )
    update_parser.add_argument(
        '--db-path',
        default='fantasy_football.db',
        help='Path to database file'
    )

    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train ML models'
    )
    train_parser.add_argument(
        '--positions',
        nargs='+',
        choices=['QB', 'RB', 'WR', 'TE'],
        help='Positions to train (default: all)'
    )
    train_parser.add_argument(
        '--db-path',
        default='fantasy_football.db',
        help='Path to database file'
    )

    # Backtest command
    backtest_parser = subparsers.add_parser(
        'backtest',
        help='Run historical validation'
    )
    backtest_parser.add_argument(
        '--weeks',
        type=int,
        required=True,
        help='Number of previous weeks to test'
    )
    backtest_parser.add_argument(
        '--positions',
        nargs='+',
        choices=['QB', 'RB', 'WR', 'TE'],
        help='Positions to test (default: all)'
    )
    backtest_parser.add_argument(
        '--db-path',
        default='fantasy_football.db',
        help='Path to database file'
    )

    # Fetch command
    fetch_parser = subparsers.add_parser(
        'fetch',
        help='Fetch data without generating projections'
    )
    fetch_parser.add_argument(
        '--week',
        type=int,
        required=True,
        help='NFL week number (1-18)'
    )
    fetch_parser.add_argument(
        '--season',
        type=int,
        required=True,
        help='NFL season year (e.g., 2024)'
    )
    fetch_parser.add_argument(
        '--db-path',
        default='fantasy_football.db',
        help='Path to database file'
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate function
    try:
        if args.command == 'setup':
            success = setup_project(db_path=args.db_path)
            sys.exit(0 if success else 1)

        elif args.command == 'update':
            results = run_weekly_update(
                week=args.week,
                season=args.season,
                db_path=args.db_path,
                train=args.train,
                fetch=not args.no_fetch
            )
            sys.exit(0 if results.get('success') else 1)

        elif args.command == 'train':
            results = train_models(
                db_path=args.db_path,
                positions=args.positions
            )
            sys.exit(0 if results.get('success') else 1)

        elif args.command == 'backtest':
            results = backtest(
                weeks=args.weeks,
                db_path=args.db_path,
                positions=args.positions
            )
            sys.exit(0 if results.get('success') else 1)

        elif args.command == 'fetch':
            results = fetch_data(
                week=args.week,
                season=args.season,
                db_path=args.db_path
            )
            sys.exit(0 if results.get('success') else 1)

        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
