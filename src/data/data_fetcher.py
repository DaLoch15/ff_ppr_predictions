"""
Fantasy Football Data Fetcher

This module provides comprehensive data fetching and processing for NFL fantasy
football projections, including player stats, team data, Vegas lines, injuries,
weather, and defensive rankings.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# NFL data library
try:
    import nflreadpy as nfl
except ImportError:
    logging.warning("nflreadpy not installed. Run: pip install git+https://github.com/nflverse/nflreadpy.git")
    nfl = None

# For API calls
import requests
from dotenv import load_dotenv

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from database.db_manager import DatabaseManager


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FantasyDataFetcher:
    """
    Fetches and processes NFL data for fantasy football projections.

    This class handles data retrieval from multiple sources including:
    - nflreadpy for play-by-play and weekly stats
    - The Odds API for Vegas lines
    - Weather and injury data
    - Defensive rankings calculations

    Attributes:
        db_manager (DatabaseManager): Database connection manager
        odds_api_key (str): API key for The Odds API
        current_season (int): Current NFL season year
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the FantasyDataFetcher.

        Args:
            db_manager: DatabaseManager instance for data storage
        """
        self.db_manager = db_manager
        self.odds_api_key = os.getenv('ODDS_API_KEY', None)
        self.current_season = datetime.now().year
        logger.info("FantasyDataFetcher initialized")

    def fetch_nfl_data(self, years: List[int]) -> Dict[str, pd.DataFrame]:
        """
        Fetch NFL data using nflreadpy library.

        Retrieves play-by-play, weekly stats, rosters, and schedules for
        specified years.

        Args:
            years: List of years to fetch data for (e.g., [2022, 2023, 2024])

        Returns:
            Dictionary containing DataFrames:
            - 'weekly': Weekly player statistics
            - 'pbp': Play-by-play data
            - 'rosters': Player roster information
            - 'schedules': Game schedules

        Raises:
            ImportError: If nflreadpy is not installed
            Exception: If data fetching fails
        """
        if nfl is None:
            raise ImportError("nflreadpy is required. Install with: pip install git+https://github.com/nflverse/nflreadpy.git")

        try:
            logger.info(f"Fetching NFL data for years: {years}")

            data = {}

            # Fetch weekly player stats
            logger.info("Fetching weekly player statistics...")
            data['weekly'] = nfl.load_player_stats(years).to_pandas()
            logger.info(f"Fetched {len(data['weekly'])} weekly stat records")

            # Fetch play-by-play data
            logger.info("Fetching play-by-play data...")
            data['pbp'] = nfl.load_pbp(years).to_pandas()
            logger.info(f"Fetched {len(data['pbp'])} play-by-play records")

            # Fetch rosters
            logger.info("Fetching roster data...")
            data['rosters'] = nfl.load_rosters(years).to_pandas()
            logger.info(f"Fetched {len(data['rosters'])} roster records")

            # Fetch schedules
            logger.info("Fetching schedule data...")
            data['schedules'] = nfl.load_schedules(years).to_pandas()
            logger.info(f"Fetched {len(data['schedules'])} schedule records")

            return data

        except Exception as e:
            logger.error(f"Error fetching NFL data: {e}")
            raise

    def process_players(self, weekly_data: pd.DataFrame, rosters: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract and process unique player information.

        Args:
            weekly_data: DataFrame from nfl.load_player_stats()
            rosters: Optional DataFrame from nfl.load_rosters()

        Returns:
            DataFrame formatted for players table insertion
        """
        try:
            logger.info("Processing players...")

            # Get unique players from weekly data
            players = weekly_data[['player_id', 'player_name', 'position', 'team']].drop_duplicates('player_id')

            # Merge with rosters if available for additional info
            if rosters is not None and 'gsis_id' in rosters.columns:
                # nflreadpy uses gsis_id instead of player_id
                rosters_copy = rosters.copy()
                rosters_copy['player_id'] = rosters_copy['gsis_id']
                roster_cols = ['player_id', 'position', 'team', 'jersey_number', 'status',
                               'height', 'weight', 'college', 'birth_date']
                roster_cols = [c for c in roster_cols if c in rosters_copy.columns]
                roster_info = rosters_copy[roster_cols].drop_duplicates('player_id')
                players = players.merge(roster_info, on='player_id', how='left', suffixes=('', '_roster'))
                # Use roster team/position if available
                if 'team_roster' in players.columns:
                    players['team'] = players['team_roster'].fillna(players['team'])
                    players.drop(['team_roster'], axis=1, inplace=True)
                if 'position_roster' in players.columns:
                    players['position'] = players['position_roster'].fillna(players['position'])
                    players.drop(['position_roster'], axis=1, inplace=True)

            # Add default values for missing columns (only if not already present)
            if 'years_experience' not in players.columns:
                players['years_experience'] = 0  # Can't easily determine from nflreadpy
            if 'draft_year' not in players.columns:
                players['draft_year'] = None

            # Filter to only valid positions (database has CHECK constraint)
            valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
            players = players[players['position'].isin(valid_positions)]

            # Keep only columns that exist in the database schema
            schema_columns = ['player_id', 'player_name', 'position', 'team', 'years_experience', 'draft_year']
            players = players[[col for col in schema_columns if col in players.columns]]

            logger.info(f"Processed {len(players)} unique players (filtered to fantasy-relevant positions)")
            return players

        except Exception as e:
            logger.error(f"Error processing players: {e}")
            raise

    def process_player_games(self, weekly_data: pd.DataFrame, schedules: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process weekly player data into schema format.

        Converts nflreadpy weekly stats into our database schema format,
        including calculations for target share, air yard share, and fantasy points.

        Args:
            weekly_data: DataFrame from nfl.load_player_stats()
            schedules: Optional DataFrame from nfl.load_schedules() for game dates

        Returns:
            DataFrame formatted for player_games table insertion

        Example:
            >>> weekly = fetcher.fetch_nfl_data([2023])['weekly']
            >>> schedules = fetcher.fetch_nfl_data([2023])['schedules']
            >>> processed = fetcher.process_player_games(weekly, schedules)
        """
        try:
            logger.info("Processing player games data...")

            # Create a copy to avoid modifying original
            df = weekly_data.copy()

            # Generate game_id (combining season, week, team, opponent)
            df['game_id'] = (
                df['season'].astype(str) + '_' +
                df['week'].astype(str).str.zfill(2) + '_' +
                df['team'] + '_' +
                df['opponent_team']
            )

            # Map column names to our schema (nflreadpy column names)
            column_mapping = {
                'player_id': 'player_id',
                'season': 'season',
                'week': 'week',
                'team': 'team',
                'opponent_team': 'opponent',
                'completions': 'completions',
                'attempts': 'attempts',
                'passing_yards': 'passing_yards',
                'passing_tds': 'passing_tds',
                'passing_interceptions': 'interceptions',
                'sacks_suffered': 'sacks',
                'carries': 'carries',
                'rushing_yards': 'rushing_yards',
                'rushing_tds': 'rushing_tds',
                'targets': 'targets',
                'receptions': 'receptions',
                'receiving_yards': 'receiving_yards',
                'receiving_tds': 'receiving_tds',
                'target_share': 'target_share',
                'air_yards_share': 'air_yard_share',
                'fantasy_points_ppr': 'fantasy_points_ppr',
            }

            # Select and rename columns
            processed = pd.DataFrame()
            for source_col, target_col in column_mapping.items():
                if source_col in df.columns:
                    processed[target_col] = df[source_col]

            # Convert share columns from decimal to percentage (nflreadpy uses 0-1, we need 0-100)
            if 'target_share' in processed.columns:
                processed['target_share'] = (processed['target_share'] * 100).clip(0, 100)
            if 'air_yard_share' in processed.columns:
                processed['air_yard_share'] = (processed['air_yard_share'] * 100).clip(0, 100)

            # Add game_id
            processed['game_id'] = df['game_id']

            # Add calculated fields - merge with schedules to get game_date
            if schedules is not None and 'gameday' in schedules.columns:
                # Merge with schedules to get game dates
                schedule_dates = schedules[['season', 'week', 'home_team', 'away_team', 'gameday']].copy()
                # Create matches for both home and away teams
                df_with_dates = df.merge(
                    schedule_dates,
                    left_on=['season', 'week', 'team'],
                    right_on=['season', 'week', 'home_team'],
                    how='left'
                )
                df_with_dates['gameday'] = df_with_dates['gameday'].fillna(
                    df.merge(
                        schedule_dates,
                        left_on=['season', 'week', 'team'],
                        right_on=['season', 'week', 'away_team'],
                        how='left'
                    )['gameday']
                )
                processed['game_date'] = pd.to_datetime(df_with_dates['gameday']).dt.strftime('%Y-%m-%d')
            else:
                # Fallback: estimate date from season and week
                processed['game_date'] = (pd.to_datetime(
                    df['season'].astype(str) + '-09-01'
                ) + pd.to_timedelta((df['week'] - 1) * 7, unit='D')).dt.strftime('%Y-%m-%d')

            # Determine home/away (assume home if not specified)
            processed['is_home'] = df.get('is_home', 1)

            # Add snap count and snap percentage if available
            # nflreadpy doesn't provide snap data, so we estimate based on fantasy activity
            if 'snap_count' in df.columns and df['snap_count'].sum() > 0:
                processed['snap_count'] = df['snap_count'].fillna(0).astype(int)
            else:
                # Estimate snap count: players with any stats get 40 snaps, otherwise 0
                has_activity = (df['targets'].fillna(0) + df['carries'].fillna(0) + df['receptions'].fillna(0)) > 0
                processed['snap_count'] = has_activity.astype(int) * 40

            if 'snap_pct' in df.columns and df['snap_pct'].sum() > 0:
                processed['snap_pct'] = df['snap_pct'].fillna(0.0)
            else:
                # Estimate snap percentage: players with any stats get 60%, otherwise 0
                has_activity = (df['targets'].fillna(0) + df['carries'].fillna(0) + df['receptions'].fillna(0)) > 0
                processed['snap_pct'] = has_activity.astype(float) * 60.0

            # Routes run and route participation
            processed['routes_run'] = df['routes_run'].fillna(0).astype(int) if 'routes_run' in df.columns else 0
            processed['route_participation'] = df['route_participation'].fillna(0.0) if 'route_participation' in df.columns else 0.0

            # Air yards and YAC
            processed['air_yards'] = df['air_yards'].fillna(0).astype(int) if 'air_yards' in df.columns else 0
            processed['yards_after_catch'] = df['yards_after_catch'].fillna(0).astype(int) if 'yards_after_catch' in df.columns else 0

            # Red zone stats
            processed['red_zone_targets'] = df['red_zone_targets'].fillna(0).astype(int) if 'red_zone_targets' in df.columns else 0
            processed['red_zone_carries'] = df['red_zone_carries'].fillna(0).astype(int) if 'red_zone_carries' in df.columns else 0

            # Note: target_share and air_yards_share are provided by nflreadpy
            # The above conversion from decimal to percentage handles them

            # Fill NaN values with 0 for numeric columns
            numeric_columns = processed.select_dtypes(include=[np.number]).columns
            processed[numeric_columns] = processed[numeric_columns].fillna(0)

            logger.info(f"Processed {len(processed)} player game records")
            return processed

        except Exception as e:
            logger.error(f"Error processing player games: {e}")
            raise

    def calculate_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 3-game, 5-game, and season rolling averages.

        Args:
            df: DataFrame with player game statistics

        Returns:
            DataFrame with rolling average columns added

        Example:
            >>> df_with_averages = fetcher.calculate_rolling_averages(player_games_df)
        """
        try:
            logger.info("Calculating rolling averages...")

            # Sort by player and date
            df = df.sort_values(['player_id', 'game_date'])

            # Stats to calculate averages for
            stats_cols = [
                'fantasy_points_ppr', 'targets', 'receptions', 'receiving_yards',
                'carries', 'rushing_yards', 'passing_yards', 'snap_pct', 'target_share'
            ]

            # Calculate 3-game rolling averages
            for col in stats_cols:
                if col in df.columns:
                    df[f'{col}_3g'] = df.groupby('player_id')[col].transform(
                        lambda x: x.rolling(window=3, min_periods=1).mean()
                    )

            # Calculate 5-game rolling averages
            for col in stats_cols:
                if col in df.columns:
                    df[f'{col}_5g'] = df.groupby('player_id')[col].transform(
                        lambda x: x.rolling(window=5, min_periods=1).mean()
                    )

            # Calculate season averages (expanding mean)
            for col in stats_cols:
                if col in df.columns:
                    df[f'{col}_season'] = df.groupby(['player_id', 'season'])[col].transform(
                        lambda x: x.expanding().mean()
                    )

            logger.info("Rolling averages calculated successfully")
            return df

        except Exception as e:
            logger.error(f"Error calculating rolling averages: {e}")
            raise

    def fetch_vegas_lines(self, week: int, season: int) -> pd.DataFrame:
        """
        Fetch Vegas betting lines from The Odds API.

        If API key is not available, generates mock data for testing.

        Args:
            week: NFL week number
            season: NFL season year

        Returns:
            DataFrame with Vegas lines data

        Example:
            >>> lines = fetcher.fetch_vegas_lines(week=5, season=2024)
        """
        try:
            logger.info(f"Fetching Vegas lines for Week {week}, {season}...")

            if self.odds_api_key:
                # Use The Odds API
                url = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/'
                params = {
                    'apiKey': self.odds_api_key,
                    'regions': 'us',
                    'markets': 'spreads,totals',
                    'oddsFormat': 'american'
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()
                vegas_lines = self._parse_odds_api_response(data, week, season)

            else:
                logger.warning("No ODDS_API_KEY found, generating mock Vegas lines data")
                vegas_lines = self._generate_mock_vegas_lines(week, season)

            logger.info(f"Fetched {len(vegas_lines)} Vegas lines")
            return vegas_lines

        except requests.RequestException as e:
            logger.error(f"Error fetching Vegas lines from API: {e}")
            logger.info("Falling back to mock data")
            return self._generate_mock_vegas_lines(week, season)
        except Exception as e:
            logger.error(f"Error processing Vegas lines: {e}")
            raise

    def _parse_odds_api_response(
        self,
        data: List[Dict],
        week: int,
        season: int
    ) -> pd.DataFrame:
        """Parse response from The Odds API into our schema format."""
        games = []

        for game in data:
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')
            game_id = f"{season}_{week:02d}_{away_team}_{home_team}"

            # Extract odds from bookmakers
            spread = None
            total = None
            home_ml = None
            away_ml = None

            if game.get('bookmakers'):
                bookmaker = game['bookmakers'][0]  # Use first bookmaker

                # Get spread
                spread_market = next(
                    (m for m in bookmaker.get('markets', []) if m['key'] == 'spreads'),
                    None
                )
                if spread_market and spread_market.get('outcomes'):
                    for outcome in spread_market['outcomes']:
                        if outcome['name'] == home_team:
                            spread = outcome.get('point', 0)

                # Get total
                total_market = next(
                    (m for m in bookmaker.get('markets', []) if m['key'] == 'totals'),
                    None
                )
                if total_market and total_market.get('outcomes'):
                    total = total_market['outcomes'][0].get('point', 0)

            # Calculate implied totals
            home_implied = (total + spread) / 2 if total and spread else None
            away_implied = (total - spread) / 2 if total and spread else None

            games.append({
                'game_id': game_id,
                'season': season,
                'week': week,
                'home_team': home_team,
                'away_team': away_team,
                'spread': spread,
                'total': total,
                'home_implied_total': home_implied,
                'away_implied_total': away_implied,
                'moneyline_home': home_ml,
                'moneyline_away': away_ml,
                'line_movement': 0.0,
                'total_movement': 0.0
            })

        return pd.DataFrame(games)

    def _generate_mock_vegas_lines(self, week: int, season: int) -> pd.DataFrame:
        """Generate mock Vegas lines for testing purposes."""
        teams = ['KC', 'BUF', 'MIA', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT',
                 'SF', 'LAR', 'SEA', 'ARI', 'DAL', 'PHI', 'WAS', 'NYG']

        games = []
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                home_team = teams[i]
                away_team = teams[i + 1]
                game_id = f"{season}_{week:02d}_{away_team}_{home_team}"

                spread = np.random.uniform(-7, 7)
                total = np.random.uniform(42, 54)
                home_implied = (total - spread) / 2
                away_implied = (total + spread) / 2

                games.append({
                    'game_id': game_id,
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'spread': round(spread, 1),
                    'total': round(total, 1),
                    'home_implied_total': round(home_implied, 1),
                    'away_implied_total': round(away_implied, 1),
                    'moneyline_home': int(spread * -20),
                    'moneyline_away': int(spread * 20),
                    'line_movement': 0.0,
                    'total_movement': 0.0
                })

        return pd.DataFrame(games)

    def calculate_defensive_rankings(self, pbp_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defensive rankings from play-by-play data.

        Analyzes defensive performance by position including points allowed,
        targets allowed, coverage metrics, and pressure rates.

        Args:
            pbp_data: Play-by-play DataFrame from nflreadpy

        Returns:
            DataFrame formatted for defensive_rankings table

        Example:
            >>> pbp = fetcher.fetch_nfl_data([2024])['pbp']
            >>> def_rankings = fetcher.calculate_defensive_rankings(pbp)
        """
        try:
            logger.info("Calculating defensive rankings...")

            # Filter to relevant plays
            df = pbp_data.copy()
            df = df[df['play_type'].isin(['pass', 'run'])].copy()

            # Group by defensive team, season, week
            defensive_stats = []

            for (team, season, week), group in df.groupby(['defteam', 'season', 'week']):
                # Calculate points allowed by position (approximate)
                # Note: nflreadpy doesn't provide receiver_position, so we use aggregate metrics
                pass_plays = group[group['play_type'] == 'pass']
                run_plays = group[group['play_type'] == 'run']

                qb_points = pass_plays['passing_yards'].sum() * 0.04 + \
                           pass_plays['touchdown'].sum() * 4 if 'passing_yards' in pass_plays.columns else 0

                # Simplified position-based calculations without receiver_position
                # Using run plays as proxy for RB points
                rb_points = run_plays['yards_gained'].sum() * 0.1 if 'yards_gained' in run_plays.columns else 0

                # For WR/TE, we'll use aggregate pass yards (positions not available in nflreadpy)
                total_rec_yards = pass_plays['yards_gained'].sum() if 'yards_gained' in pass_plays.columns else 0
                wr_points = total_rec_yards * 0.05  # Rough estimate for WR share
                te_points = total_rec_yards * 0.03  # Rough estimate for TE share

                # Calculate targets allowed (aggregate without position breakdown)
                targets_rb = len(run_plays) if 'play_type' in run_plays.columns else 0
                targets_wr = len(pass_plays) * 0.6 if 'play_type' in pass_plays.columns else 0  # Estimate
                targets_te = len(pass_plays) * 0.2 if 'play_type' in pass_plays.columns else 0  # Estimate

                # Coverage and pressure metrics (using available fields)
                total_plays = len(group)
                blitz_rate = group.get('blitz', pd.Series([0])).sum() / total_plays * 100 if total_plays > 0 else 0
                pressure_rate = group.get('qb_hit', pd.Series([0])).sum() / len(pass_plays) * 100 if len(pass_plays) > 0 else 0

                defensive_stats.append({
                    'team': team,
                    'season': int(season),
                    'week': int(week),
                    'qb_points_allowed_avg': round(qb_points / max(1, len(group)), 2),
                    'rb_points_allowed_avg': round(rb_points / max(1, len(group)), 2),
                    'wr_points_allowed_avg': round(wr_points / max(1, len(group)), 2),
                    'te_points_allowed_avg': round(te_points / max(1, len(group)), 2),
                    'targets_allowed_to_rb': round(targets_rb, 2),
                    'targets_allowed_to_wr': round(targets_wr, 2),
                    'targets_allowed_to_te': round(targets_te, 2),
                    'slot_yards_allowed_avg': 0.0,  # Placeholder
                    'outside_yards_allowed_avg': 0.0,  # Placeholder
                    'man_coverage_rate': 50.0,  # Placeholder
                    'zone_coverage_rate': 50.0,  # Placeholder
                    'blitz_rate': round(blitz_rate, 2),
                    'pressure_rate': round(pressure_rate, 2)
                })

            result = pd.DataFrame(defensive_stats)
            logger.info(f"Calculated defensive rankings for {len(result)} team-weeks")
            return result

        except Exception as e:
            logger.error(f"Error calculating defensive rankings: {e}")
            raise

    def fetch_weather_data(self, game_id: str, stadium: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch weather data for a game.

        Currently returns mock data. In production, would integrate with
        weather API or scrape from NFL.com.

        Args:
            game_id: Unique game identifier
            stadium: Optional stadium name

        Returns:
            Dictionary with weather data including impact score

        Example:
            >>> weather = fetcher.fetch_weather_data('2024_05_KC_BUF', 'Arrowhead Stadium')
        """
        try:
            logger.debug(f"Fetching weather data for game {game_id}")

            # Check if stadium is a dome
            dome_stadiums = [
                'Mercedes-Benz Stadium', 'SoFi Stadium', 'AT&T Stadium',
                'U.S. Bank Stadium', 'Caesars Superdome', 'Ford Field',
                'State Farm Stadium', 'Allegiant Stadium'
            ]
            is_dome = stadium in dome_stadiums if stadium else False

            if is_dome:
                return {
                    'game_id': game_id,
                    'temperature': 72,
                    'wind_speed': 0,
                    'precipitation': 'None',
                    'dome': True,
                    'weather_score': 100
                }

            # Mock outdoor weather data
            temp = np.random.randint(20, 85)
            wind = np.random.randint(0, 25)
            precip = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], p=[0.7, 0.2, 0.08, 0.02])

            # Calculate weather score (100 = perfect conditions)
            score = 100
            if temp < 32 or temp > 85:
                score -= 20
            if wind > 15:
                score -= (wind - 15) * 2
            if precip == 'Light':
                score -= 10
            elif precip == 'Moderate':
                score -= 25
            elif precip == 'Heavy':
                score -= 40

            return {
                'game_id': game_id,
                'temperature': temp,
                'wind_speed': wind,
                'precipitation': precip,
                'dome': False,
                'weather_score': max(0, score)
            }

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return {
                'game_id': game_id,
                'temperature': 70,
                'wind_speed': 5,
                'precipitation': 'None',
                'dome': False,
                'weather_score': 100
            }

    def fetch_injury_data(self, week: int, season: int) -> pd.DataFrame:
        """
        Fetch injury report data.

        Placeholder for future injury report scraping from NFL.com or
        other sources. Currently returns empty DataFrame.

        Args:
            week: NFL week number
            season: NFL season year

        Returns:
            DataFrame formatted for injuries table

        Example:
            >>> injuries = fetcher.fetch_injury_data(week=5, season=2024)
        """
        logger.info(f"Fetching injury data for Week {week}, {season}")
        logger.warning("Injury data fetching not yet implemented - placeholder only")

        # Placeholder: Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            'player_id', 'season', 'week', 'injury_status', 'injury_type',
            'practice_wed', 'practice_thu', 'practice_fri', 'last_update'
        ])

    def update_database(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """
        Update all database tables with fetched data.

        Args:
            data_dict: Dictionary mapping table names to DataFrames

        Returns:
            Dictionary with counts of rows inserted per table

        Example:
            >>> data = {
            ...     'players': players_df,
            ...     'player_games': games_df,
            ...     'vegas_lines': lines_df
            ... }
            >>> counts = fetcher.update_database(data)
        """
        try:
            logger.info("Updating database with fetched data...")
            insert_counts = {}

            for table_name, df in data_dict.items():
                if df is not None and not df.empty:
                    # Convert DataFrame to list of dictionaries
                    records = df.to_dict('records')

                    # Special handling for tables with unique constraints
                    if table_name == 'players':
                        count = self._upsert_players(records)
                    elif table_name in ['player_games', 'vegas_lines', 'defensive_rankings']:
                        # Use INSERT OR REPLACE for tables with composite primary keys
                        count = self._upsert_table(table_name, records)
                    else:
                        # Insert data normally for other tables
                        count = self.db_manager.bulk_insert(table_name, records)

                    insert_counts[table_name] = count
                    logger.info(f"Inserted/updated {count} rows into {table_name}")
                else:
                    logger.warning(f"No data to insert for table: {table_name}")
                    insert_counts[table_name] = 0

            return insert_counts

        except Exception as e:
            logger.error(f"Error updating database: {e}")
            raise

    def _upsert_players(self, records: List[Dict[str, Any]]) -> int:
        """
        Insert or update players using INSERT OR IGNORE to handle duplicates.

        Uses INSERT OR IGNORE instead of REPLACE to avoid triggering CASCADE DELETE
        on player_games foreign key relationship.

        Args:
            records: List of player dictionaries

        Returns:
            Number of players processed
        """
        if not records:
            return 0

        try:
            # Get column names from first record
            columns = list(records[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            column_str = ', '.join(columns)

            # Use INSERT OR IGNORE to handle duplicates (preserves existing data and foreign keys)
            query = f"INSERT OR IGNORE INTO players ({column_str}) VALUES ({placeholders})"

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Convert records to tuples
                values = [tuple(row[col] for col in columns) for row in records]

                cursor.executemany(query, values)
                conn.commit()

            logger.info(f"Upserted {len(records)} players")
            return len(records)

        except Exception as e:
            logger.error(f"Error upserting players: {e}")
            raise

    def _upsert_table(self, table_name: str, records: List[Dict[str, Any]]) -> int:
        """
        Insert or update table using INSERT OR REPLACE to handle duplicates.

        Args:
            table_name: Name of the table
            records: List of record dictionaries

        Returns:
            Number of records processed
        """
        if not records:
            return 0

        try:
            # Get column names from first record
            columns = list(records[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            column_str = ', '.join(columns)

            # Use INSERT OR IGNORE to handle duplicates (preserves existing data)
            query = f"INSERT OR IGNORE INTO {table_name} ({column_str}) VALUES ({placeholders})"

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Convert records to tuples
                values = [tuple(row[col] for col in columns) for row in records]

                cursor.executemany(query, values)
                conn.commit()

            logger.info(f"Upserted {len(records)} records into {table_name}")
            return len(records)

        except Exception as e:
            logger.error(f"Error upserting {table_name}: {e}")
            raise

    def run_weekly_update(self, week: int, season: int) -> Dict[str, Any]:
        """
        Run complete weekly data update routine.

        Fetches all data sources and updates database tables for a given week.

        Args:
            week: NFL week number to update
            season: NFL season year

        Returns:
            Dictionary with update status and statistics

        Example:
            >>> result = fetcher.run_weekly_update(week=5, season=2024)
            >>> print(f"Updated {result['total_rows']} rows")
        """
        try:
            logger.info(f"Starting weekly update for Week {week}, Season {season}")
            start_time = datetime.now()

            update_data = {}

            # Fetch NFL data
            logger.info("Fetching NFL data...")
            nfl_data = self.fetch_nfl_data([season])

            # Extract and process players first (needed for foreign key)
            logger.info("Extracting player information...")
            players = self.process_players(nfl_data['weekly'], nfl_data['rosters'])
            update_data['players'] = players

            # Process player games
            logger.info("Processing player games...")
            player_games = self.process_player_games(nfl_data['weekly'], nfl_data['schedules'])
            player_games = player_games[
                (player_games['season'] == season) & (player_games['week'] == week)
            ]
            # Filter to only include players that exist in the players table
            if len(players) > 0:
                player_games = player_games[player_games['player_id'].isin(players['player_id'])]
                logger.info(f"Filtered player_games to {len(player_games)} records for players in database")
            # Note: Rolling averages can be calculated later for projections
            # Don't add them here to match database schema
            update_data['player_games'] = player_games

            # Fetch Vegas lines
            logger.info("Fetching Vegas lines...")
            update_data['vegas_lines'] = self.fetch_vegas_lines(week, season)

            # Calculate defensive rankings
            logger.info("Calculating defensive rankings...")
            def_rankings = self.calculate_defensive_rankings(nfl_data['pbp'])
            def_rankings = def_rankings[
                (def_rankings['season'] == season) & (def_rankings['week'] == week)
            ]
            update_data['defensive_rankings'] = def_rankings

            # Fetch injury data
            logger.info("Fetching injury data...")
            update_data['injuries'] = self.fetch_injury_data(week, season)

            # Update database
            logger.info("Updating database...")
            insert_counts = self.update_database(update_data)

            # Calculate statistics
            total_rows = sum(insert_counts.values())
            duration = (datetime.now() - start_time).total_seconds()

            result = {
                'success': True,
                'week': week,
                'season': season,
                'total_rows': total_rows,
                'insert_counts': insert_counts,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Weekly update completed: {total_rows} rows in {duration:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error during weekly update: {e}")
            return {
                'success': False,
                'week': week,
                'season': season,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_current_week_players(
        self,
        week: int,
        season: int,
        position: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get eligible players for projections for a given week.

        Retrieves players who are active and have sufficient historical data
        for generating projections.

        Args:
            week: NFL week number
            season: NFL season year
            position: Optional position filter (e.g., 'QB', 'RB', 'WR', 'TE')

        Returns:
            DataFrame with player information and recent stats

        Example:
            >>> qbs = fetcher.get_current_week_players(week=5, season=2024, position='QB')
        """
        try:
            logger.info(f"Getting players for Week {week}, {season}, Position: {position or 'ALL'}")

            # Query players with recent activity
            query = """
                SELECT DISTINCT
                    p.player_id,
                    p.player_name,
                    p.position,
                    p.team,
                    p.years_experience,
                    COUNT(pg.game_id) as games_played,
                    AVG(pg.fantasy_points_ppr) as avg_fantasy_points,
                    AVG(pg.snap_pct) as avg_snap_pct
                FROM players p
                INNER JOIN player_games pg ON p.player_id = pg.player_id
                WHERE pg.season = ?
                    AND pg.week < ?
                    {position_filter}
                GROUP BY p.player_id, p.player_name, p.position, p.team, p.years_experience
                HAVING games_played >= 2 AND avg_snap_pct > 20
                ORDER BY avg_fantasy_points DESC
            """

            # Add position filter if specified
            position_filter = f"AND p.position = '{position}'" if position else ""
            query = query.format(position_filter=position_filter)

            # Execute query
            results = self.db_manager.execute_query(query, (season, week))

            # Convert to DataFrame
            if results:
                df = pd.DataFrame([dict(row) for row in results])
                logger.info(f"Found {len(df)} eligible players")
                return df
            else:
                logger.warning("No eligible players found")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting current week players: {e}")
            raise


class DataValidator:
    """
    Validates data quality for fantasy football datasets.

    Performs comprehensive validation checks on player data, game data,
    and other datasets to ensure data integrity before database insertion.
    """

    @staticmethod
    def validate_player_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate player data quality.

        Checks for:
        - Required columns
        - Data types
        - Value ranges
        - Missing values
        - Duplicate records

        Args:
            df: DataFrame with player data

        Returns:
            Tuple of (is_valid, list of validation errors)

        Example:
            >>> is_valid, errors = DataValidator.validate_player_data(players_df)
            >>> if not is_valid:
            ...     print(f"Validation errors: {errors}")
        """
        errors = []

        # Check required columns
        required_columns = ['player_id', 'player_name', 'position', 'team']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check for null player_ids
        if 'player_id' in df.columns:
            null_ids = df['player_id'].isnull().sum()
            if null_ids > 0:
                errors.append(f"Found {null_ids} null player_id values")

        # Check for duplicate player_ids
        if 'player_id' in df.columns:
            duplicates = df['player_id'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate player_id values")

        # Check position values
        if 'position' in df.columns:
            valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
            invalid_positions = df[~df['position'].isin(valid_positions)]
            if len(invalid_positions) > 0:
                errors.append(f"Found {len(invalid_positions)} invalid position values")

        # Check for negative stats
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stat_columns = [col for col in numeric_columns if col not in ['years_experience', 'draft_year']]

        for col in stat_columns:
            negative_values = (df[col] < 0).sum()
            if negative_values > 0:
                errors.append(f"Found {negative_values} negative values in {col}")

        # Check percentage fields are between 0 and 100
        pct_columns = [col for col in df.columns if 'pct' in col.lower() or 'share' in col.lower()]
        for col in pct_columns:
            if col in df.columns:
                invalid_pct = ((df[col] < 0) | (df[col] > 100)).sum()
                if invalid_pct > 0:
                    errors.append(f"Found {invalid_pct} out-of-range percentage values in {col}")

        is_valid = len(errors) == 0
        return is_valid, errors


# Example usage
if __name__ == "__main__":
    # Initialize database manager
    db_manager = DatabaseManager(db_path='fantasy_football.db')

    try:
        # Initialize database schema
        db_manager.initialize_database()

        # Create data fetcher
        fetcher = FantasyDataFetcher(db_manager)

        # Run weekly update
        result = fetcher.run_weekly_update(week=1, season=2024)

        if result['success']:
            print(f"✓ Weekly update completed successfully")
            print(f"  Total rows: {result['total_rows']}")
            print(f"  Duration: {result['duration_seconds']:.2f}s")
            print(f"  Breakdown: {result['insert_counts']}")
        else:
            print(f"✗ Weekly update failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        db_manager.close()
