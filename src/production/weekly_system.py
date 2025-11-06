"""
Weekly Fantasy Football Projection System

This module provides a production-ready system for generating weekly fantasy
football projections, rankings, tiers, value plays, and optimal DFS lineups.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from database.db_manager import DatabaseManager
from data.data_fetcher import FantasyDataFetcher
from models.ml_pipeline import FantasyProjectionPipeline


class WeeklyProjectionSystem:
    """
    Complete weekly projection system for fantasy football.

    This system orchestrates:
    - Data fetching and validation
    - ML projection generation
    - Matchup and game script adjustments
    - Rankings and tier calculations
    - Value play identification
    - DFS lineup optimization
    - Results export

    Attributes:
        db_manager (DatabaseManager): Database connection manager
        data_fetcher (FantasyDataFetcher): Data fetching component
        ml_pipeline (FantasyProjectionPipeline): ML projection component
        output_dir (str): Directory for output files
        logger (logging.Logger): Logger instance
    """

    # DFS Configuration
    SALARY_CAP = 50000
    POSITION_REQUIREMENTS = {
        'QB': 1,
        'RB': 2,
        'WR': 3,
        'TE': 1,
        'FLEX': 1  # RB, WR, or TE
    }

    def __init__(
        self,
        db_manager: DatabaseManager,
        output_dir: str = 'output',
        log_file: Optional[str] = None
    ):
        """
        Initialize the WeeklyProjectionSystem.

        Args:
            db_manager: DatabaseManager instance
            output_dir: Directory for output files (default: 'output')
            log_file: Optional path to log file
        """
        self.db_manager = db_manager
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging to both file and console
        self.logger = self._setup_logging(log_file)

        # Initialize components
        self.data_fetcher = FantasyDataFetcher(db_manager)
        self.ml_pipeline = FantasyProjectionPipeline(db_manager)

        self.logger.info("WeeklyProjectionSystem initialized")

    def _setup_logging(self, log_file: Optional[str] = None) -> logging.Logger:
        """
        Setup logging to both file and console.

        Args:
            log_file: Optional path to log file

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Clear any existing handlers
        logger.handlers = []

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file is None:
            log_file = os.path.join(self.output_dir, 'weekly_system.log')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def run_weekly_projections(
        self,
        week: int,
        season: int,
        update_data: bool = True,
        train_models: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete weekly projection workflow.

        Args:
            week: NFL week number
            season: NFL season year
            update_data: Whether to fetch and update data (default: True)
            train_models: Whether to retrain models (default: False)

        Returns:
            Dictionary with results including projections, rankings, and lineups

        Example:
            >>> system = WeeklyProjectionSystem(db_manager)
            >>> results = system.run_weekly_projections(week=5, season=2024)
        """
        try:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Starting Weekly Projections: {season} Week {week}")
            self.logger.info(f"{'='*60}")
            start_time = datetime.now()

            results = {
                'week': week,
                'season': season,
                'timestamp': start_time.isoformat(),
                'success': False
            }

            # Step 1: Data fetching and validation
            if update_data:
                self.logger.info("Step 1: Fetching and validating data...")
                data_result = self.data_fetcher.run_weekly_update(week, season)
                results['data_update'] = data_result

                if not data_result['success']:
                    self.logger.error("Data update failed")
                    return results

            # Step 2: Model training (if requested)
            if train_models:
                self.logger.info("Step 2: Training ML models...")
                train_results = self._train_all_models()
                results['training'] = train_results

            # Step 3: Generate projections for all positions
            self.logger.info("Step 3: Generating projections...")
            all_projections = []
            positions = ['QB', 'RB', 'WR', 'TE']

            for position in positions:
                self.logger.info(f"  Generating {position} projections...")
                position_proj = self.generate_position_projections(position, week, season)

                if not position_proj.empty:
                    all_projections.append(position_proj)
                    self.logger.info(f"  ✓ Generated {len(position_proj)} {position} projections")
                else:
                    self.logger.warning(f"  ✗ No projections for {position}")

            if not all_projections:
                self.logger.error("No projections generated")
                return results

            projections_df = pd.concat(all_projections, ignore_index=True)
            results['projections'] = projections_df

            # Step 4: Calculate rankings and tiers
            self.logger.info("Step 4: Calculating rankings and tiers...")
            rankings_df = self.calculate_rankings_and_tiers(projections_df)
            results['rankings'] = rankings_df

            # Step 5: Identify value plays
            self.logger.info("Step 5: Identifying value plays...")
            value_plays = self.identify_value_plays(rankings_df)
            results['value_plays'] = value_plays

            # Step 6: Generate optimal DFS lineups
            self.logger.info("Step 6: Generating optimal DFS lineups...")
            lineups = self.generate_optimal_lineups(rankings_df)
            results['lineups'] = lineups

            # Step 7: Export results
            self.logger.info("Step 7: Exporting results...")
            export_paths = self.export_results(results, week, season)
            results['export_paths'] = export_paths

            # Calculate execution time
            duration = (datetime.now() - start_time).total_seconds()
            results['duration_seconds'] = duration
            results['success'] = True

            self.logger.info(f"{'='*60}")
            self.logger.info(f"✓ Weekly projections completed in {duration:.2f}s")
            self.logger.info(f"{'='*60}")

            return results

        except Exception as e:
            self.logger.error(f"Error in weekly projection workflow: {e}", exc_info=True)
            results['error'] = str(e)
            return results

    def _train_all_models(self) -> Dict[str, Any]:
        """Train ML models for all positions."""
        training_results = {}
        positions = ['QB', 'RB', 'WR', 'TE']

        for position in positions:
            try:
                self.logger.info(f"  Training {position} ensemble model...")
                metrics = self.ml_pipeline.train_ensemble_model(position)
                training_results[position] = metrics
                self.logger.info(f"  ✓ {position} MAE: {metrics['ensemble_mae']:.3f}")
            except Exception as e:
                self.logger.error(f"  ✗ Failed to train {position}: {e}")
                training_results[position] = {'error': str(e)}

        return training_results

    def generate_position_projections(
        self,
        position: str,
        week: int,
        season: int
    ) -> pd.DataFrame:
        """
        Generate adjusted projections for a position.

        Applies matchup, game script, and pace adjustments to base projections.

        Args:
            position: Player position
            week: NFL week number
            season: NFL season year

        Returns:
            DataFrame with adjusted projections

        Example:
            >>> rb_projections = system.generate_position_projections('RB', 5, 2024)
        """
        try:
            # Get base ML projections
            base_projections = self.ml_pipeline.generate_projections(
                position, week, season, model_type='ensemble'
            )

            if base_projections.empty:
                self.logger.warning(f"No base projections for {position}")
                return pd.DataFrame()

            # Load additional context (player info, matchup data, vegas lines)
            context_query = """
                SELECT
                    p.player_id,
                    p.player_name,
                    p.position,
                    p.team,
                    s.opponent,
                    s.game_date,
                    dr.qb_points_allowed_avg,
                    dr.rb_points_allowed_avg,
                    dr.wr_points_allowed_avg,
                    dr.te_points_allowed_avg,
                    vl.spread,
                    vl.total,
                    vl.home_implied_total,
                    vl.away_implied_total
                FROM players p
                LEFT JOIN (
                    SELECT DISTINCT player_id, team, opponent, game_date
                    FROM player_games
                    WHERE season = ? AND week = ?
                ) s ON p.player_id = s.player_id
                LEFT JOIN defensive_rankings dr ON s.opponent = dr.team
                    AND dr.season = ? AND dr.week = ?
                LEFT JOIN vegas_lines vl ON vl.season = ? AND vl.week = ?
                    AND (vl.home_team = p.team OR vl.away_team = p.team)
                WHERE p.position = ?
            """

            context_results = self.db_manager.execute_query(
                context_query,
                (season, week, season, week, season, week, position)
            )

            if not context_results:
                # If no context data, return base projections with defaults
                return self._add_default_adjustments(base_projections)

            context_df = pd.DataFrame([dict(row) for row in context_results])

            # Merge projections with context
            merged = base_projections.merge(
                context_df,
                on='player_id',
                how='left'
            )

            # Apply adjustments
            merged = self._apply_matchup_adjustments(merged, position)
            merged = self._apply_game_script_adjustments(merged, position)
            merged = self._apply_pace_adjustments(merged)

            # Recalculate floor and ceiling
            merged['floor_points'] = merged['projected_points'] * 0.60
            merged['ceiling_points'] = merged['projected_points'] * 1.50

            # Add mock DFS salaries (in production, fetch from DFS sites)
            merged['salary'] = self._generate_mock_salaries(merged)

            return merged

        except Exception as e:
            self.logger.error(f"Error generating {position} projections: {e}")
            return pd.DataFrame()

    def _add_default_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default adjustments when context data is missing."""
        df['player_name'] = 'Unknown'
        df['team'] = 'UNK'
        df['opponent'] = 'UNK'
        df['matchup_adjustment'] = 1.0
        df['game_script_adjustment'] = 1.0
        df['pace_adjustment'] = 1.0
        df['adjusted_projection'] = df['projected_points']
        df['salary'] = 5000
        return df

    def _apply_matchup_adjustments(
        self,
        df: pd.DataFrame,
        position: str
    ) -> pd.DataFrame:
        """
        Apply matchup adjustments based on defensive rankings.

        Adjustments range from 0.75 (tough matchup) to 1.25 (great matchup).
        """
        # Map position to defensive points allowed column
        def_col_map = {
            'QB': 'qb_points_allowed_avg',
            'RB': 'rb_points_allowed_avg',
            'WR': 'wr_points_allowed_avg',
            'TE': 'te_points_allowed_avg'
        }

        def_col = def_col_map.get(position)

        if def_col and def_col in df.columns:
            # Calculate adjustment based on points allowed
            # Higher points allowed = easier matchup = higher multiplier
            avg_points = df[def_col].mean()

            df['matchup_adjustment'] = 1.0 + (
                (df[def_col] - avg_points) / avg_points * 0.5
            )

            # Clip to reasonable range
            df['matchup_adjustment'] = df['matchup_adjustment'].clip(0.75, 1.25)
        else:
            df['matchup_adjustment'] = 1.0

        df['matchup_adjustment'] = df['matchup_adjustment'].fillna(1.0)
        return df

    def _apply_game_script_adjustments(
        self,
        df: pd.DataFrame,
        position: str
    ) -> pd.DataFrame:
        """
        Apply game script adjustments based on spread.

        RBs benefit from positive spread (team favored = more rushing)
        WRs benefit from negative spread (team underdog = more passing)
        """
        if 'spread' not in df.columns:
            df['game_script_adjustment'] = 1.0
            return df

        df['game_script_adjustment'] = 1.0

        # Determine team's spread perspective
        # Positive spread = team is underdog, negative = favored
        # Note: This is simplified; need to check if team is home/away

        if position == 'RB':
            # RBs benefit when team is favored (negative spread)
            # Adjustment: +3 point favorite = +5% projection
            df['game_script_adjustment'] = 1.0 - (df['spread'] / 100)
            df['game_script_adjustment'] = df['game_script_adjustment'].clip(0.85, 1.15)

        elif position in ['WR', 'TE', 'QB']:
            # Pass catchers benefit when team is underdog (positive spread)
            # Adjustment: +3 point underdog = +5% projection
            df['game_script_adjustment'] = 1.0 + (df['spread'] / 100)
            df['game_script_adjustment'] = df['game_script_adjustment'].clip(0.85, 1.15)

        df['game_script_adjustment'] = df['game_script_adjustment'].fillna(1.0)
        return df

    def _apply_pace_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pace adjustments based on game total.

        Higher totals = more plays = more opportunity.
        """
        if 'total' not in df.columns:
            df['pace_adjustment'] = 1.0
            return df

        # Average game total is around 45-47 points
        avg_total = 46.0

        # Higher total = higher adjustment (max +10% for 55+ total)
        df['pace_adjustment'] = 1.0 + (df['total'] - avg_total) / 100
        df['pace_adjustment'] = df['pace_adjustment'].clip(0.90, 1.10)
        df['pace_adjustment'] = df['pace_adjustment'].fillna(1.0)

        return df

    def _generate_mock_salaries(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate mock DFS salaries based on projections.

        In production, fetch actual salaries from DraftKings, FanDuel, etc.
        """
        # Base salary on projected points
        # Rough formula: $1000 per projected point + position adjustment
        base_salary = df['projected_points'] * 1000

        # Add position adjustments (QBs and TEs typically cheaper per point)
        position_multipliers = {'QB': 0.9, 'RB': 1.0, 'WR': 1.0, 'TE': 0.85}
        position_mult = df['position'].map(position_multipliers).fillna(1.0)

        salary = (base_salary * position_mult).clip(3000, 10000).round(-2)
        return salary

    def calculate_rankings_and_tiers(self, projections: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position and overall rankings with tier groupings.

        Args:
            projections: DataFrame with player projections

        Returns:
            DataFrame with rankings and tiers

        Example:
            >>> rankings = system.calculate_rankings_and_tiers(projections_df)
        """
        try:
            df = projections.copy()

            # Calculate position rankings
            df['position_rank'] = df.groupby('position')['projected_points'].rank(
                ascending=False,
                method='min'
            ).astype(int)

            # Calculate overall rankings (flex positions)
            flex_positions = ['RB', 'WR', 'TE']
            df['overall_rank'] = 0
            df.loc[df['position'].isin(flex_positions), 'overall_rank'] = (
                df[df['position'].isin(flex_positions)]['projected_points']
                .rank(ascending=False, method='min')
                .astype(int)
            )

            # Calculate tiers based on significant point drops
            df = self._calculate_tiers(df)

            # Calculate value scores
            df = self._calculate_value_scores(df)

            # Sort by position and rank
            df = df.sort_values(['position', 'position_rank'])

            self.logger.info(f"Calculated rankings for {len(df)} players")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating rankings: {e}")
            return projections

    def _calculate_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate tier groupings based on significant point drops.

        Uses 75th percentile of point differences as tier threshold.
        """
        for position in df['position'].unique():
            pos_mask = df['position'] == position
            pos_data = df[pos_mask].sort_values('projected_points', ascending=False)

            if len(pos_data) < 2:
                df.loc[pos_mask, 'tier'] = 1
                continue

            # Calculate point differences between consecutive players
            point_diffs = pos_data['projected_points'].diff().abs()

            # Significant drop threshold (75th percentile)
            threshold = point_diffs.quantile(0.75)

            # Assign tiers
            tiers = [1]
            current_tier = 1

            for diff in point_diffs.iloc[1:]:
                if diff >= threshold:
                    current_tier += 1
                tiers.append(current_tier)

            df.loc[pos_mask, 'tier'] = pd.Series(
                tiers,
                index=pos_data.index
            )

        df['tier'] = df['tier'].fillna(1).astype(int)
        return df

    def _calculate_value_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate value scores based on points per salary dollar."""
        df['value_score'] = (df['projected_points'] / df['salary'] * 1000).round(2)
        return df

    def identify_value_plays(self, rankings: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Identify different categories of value plays.

        Args:
            rankings: DataFrame with player rankings

        Returns:
            Dictionary with different value play categories

        Example:
            >>> value_plays = system.identify_value_plays(rankings_df)
            >>> print(value_plays['high_ceiling'].head())
        """
        try:
            value_plays = {}

            # 1. High ceiling plays (top 20 by ceiling)
            high_ceiling = rankings.nlargest(20, 'ceiling_points')[
                ['player_name', 'position', 'team', 'opponent', 'projected_points',
                 'ceiling_points', 'confidence_score', 'salary', 'value_score']
            ].copy()
            value_plays['high_ceiling'] = high_ceiling

            # 2. Safe floor plays (high floor + confidence > 70)
            safe_floor = rankings[
                (rankings['confidence_score'] > 70) &
                (rankings['floor_points'] >= rankings['floor_points'].quantile(0.70))
            ].nlargest(20, 'floor_points')[
                ['player_name', 'position', 'team', 'opponent', 'projected_points',
                 'floor_points', 'confidence_score', 'salary', 'value_score']
            ].copy()
            value_plays['safe_floor'] = safe_floor

            # 3. Leverage plays (low rank but high ceiling)
            leverage_candidates = rankings[
                (rankings['position_rank'] > 15) &
                (rankings['ceiling_points'] >= rankings['ceiling_points'].quantile(0.75))
            ].nlargest(15, 'ceiling_points')[
                ['player_name', 'position', 'team', 'opponent', 'position_rank',
                 'projected_points', 'ceiling_points', 'salary', 'value_score']
            ].copy()
            value_plays['leverage'] = leverage_candidates

            # 4. Positive regression candidates (high ceiling relative to projection)
            rankings['ceiling_ratio'] = rankings['ceiling_points'] / rankings['projected_points']
            regression_candidates = rankings[
                rankings['ceiling_ratio'] >= rankings['ceiling_ratio'].quantile(0.80)
            ].nlargest(15, 'ceiling_ratio')[
                ['player_name', 'position', 'team', 'opponent', 'projected_points',
                 'ceiling_points', 'ceiling_ratio', 'salary', 'value_score']
            ].copy()
            value_plays['positive_regression'] = regression_candidates

            # 5. Best values (points per dollar)
            best_values = rankings.nlargest(20, 'value_score')[
                ['player_name', 'position', 'team', 'opponent', 'projected_points',
                 'salary', 'value_score', 'confidence_score']
            ].copy()
            value_plays['best_values'] = best_values

            self.logger.info(f"Identified value plays in {len(value_plays)} categories")
            return value_plays

        except Exception as e:
            self.logger.error(f"Error identifying value plays: {e}")
            return {}

    def generate_optimal_lineups(self, rankings: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate optimal DFS lineups with different strategies.

        Args:
            rankings: DataFrame with player rankings and salaries

        Returns:
            Dictionary with different lineup strategies

        Example:
            >>> lineups = system.generate_optimal_lineups(rankings_df)
            >>> print(lineups['cash_game'])
        """
        try:
            lineups = {}

            # Prepare player pool
            pool = rankings[rankings['salary'] > 0].copy()

            # 1. Cash game lineup (prioritize floor and consistency)
            cash_lineup = self._build_lineup(
                pool,
                strategy='cash',
                weights={'projected_points': 0.4, 'floor_points': 0.4, 'confidence_score': 0.2}
            )
            lineups['cash_game'] = cash_lineup

            # 2. GPP balanced (mix of projection and ceiling)
            gpp_balanced = self._build_lineup(
                pool,
                strategy='gpp_balanced',
                weights={'projected_points': 0.5, 'ceiling_points': 0.5}
            )
            lineups['gpp_balanced'] = gpp_balanced

            # 3. GPP ceiling (maximum ceiling approach)
            gpp_ceiling = self._build_lineup(
                pool,
                strategy='gpp_ceiling',
                weights={'ceiling_points': 0.8, 'projected_points': 0.2}
            )
            lineups['gpp_ceiling'] = gpp_ceiling

            # 4. GPP contrarian (low ownership, high ceiling)
            gpp_contrarian = self._build_lineup(
                pool,
                strategy='gpp_contrarian',
                weights={'ceiling_points': 0.6, 'value_score': 0.4}
            )
            lineups['gpp_contrarian'] = gpp_contrarian

            # 5. Game stack (QB + pass catchers from same team)
            game_stack = self._build_game_stack(pool)
            lineups['game_stack'] = game_stack

            self.logger.info(f"Generated {len(lineups)} optimal lineups")
            return lineups

        except Exception as e:
            self.logger.error(f"Error generating lineups: {e}")
            return {}

    def _build_lineup(
        self,
        pool: pd.DataFrame,
        _strategy: str,
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Build optimal lineup using greedy algorithm.

        This is a simplified greedy approach. For production, use linear
        programming (PuLP or similar) for true optimization.
        """
        # Calculate composite score
        pool = pool.copy()
        pool['composite_score'] = 0.0

        for metric, weight in weights.items():
            if metric in pool.columns:
                # Normalize metric to 0-1 range
                normalized = (pool[metric] - pool[metric].min()) / (
                    pool[metric].max() - pool[metric].min() + 0.001
                )
                pool['composite_score'] += normalized * weight

        # Sort by value (composite score per salary dollar)
        pool['score_per_dollar'] = pool['composite_score'] / pool['salary'] * 1000
        pool = pool.sort_values('score_per_dollar', ascending=False)

        # Greedy selection
        lineup = []
        remaining_salary = self.SALARY_CAP
        positions_needed = self.POSITION_REQUIREMENTS.copy()

        # First pass: fill required positions
        for _, player in pool.iterrows():
            if player['salary'] > remaining_salary:
                continue

            position = player['position']

            # Check if we need this position
            if position in positions_needed and positions_needed[position] > 0:
                lineup.append(player)
                remaining_salary -= player['salary']
                positions_needed[position] -= 1

            # Check if done
            if all(count == 0 for pos, count in positions_needed.items() if pos != 'FLEX'):
                break

        # Second pass: fill FLEX
        if positions_needed.get('FLEX', 0) > 0:
            flex_eligible = ['RB', 'WR', 'TE']
            for _, player in pool.iterrows():
                if player['salary'] > remaining_salary:
                    continue

                # Skip if already in lineup
                if player['player_id'] in [p['player_id'] for p in lineup]:
                    continue

                if player['position'] in flex_eligible:
                    lineup.append(player)
                    remaining_salary -= player['salary']
                    positions_needed['FLEX'] -= 1
                    break

        # Convert to DataFrame
        lineup_df = pd.DataFrame(lineup)

        if not lineup_df.empty:
            lineup_df['lineup_position'] = lineup_df.apply(
                lambda row: 'FLEX' if len([p for p in lineup[:lineup.index(row)]
                                          if p['position'] == row['position']]) >=
                                          self.POSITION_REQUIREMENTS.get(row['position'], 0)
                else row['position'],
                axis=1
            )

        return lineup_df

    def _build_game_stack(self, pool: pd.DataFrame) -> pd.DataFrame:
        """
        Build lineup with QB + pass catchers from same team correlation.
        """
        # Find best QB by projected points
        qbs = pool[pool['position'] == 'QB'].nlargest(5, 'projected_points')

        best_stack = None
        best_score = 0

        for _, qb in qbs.iterrows():
            # Get pass catchers from same team
            team_stack = pool[
                (pool['team'] == qb['team']) &
                (pool['position'].isin(['WR', 'TE']))
            ].nlargest(3, 'projected_points')

            if len(team_stack) < 2:
                continue

            # Build rest of lineup
            stack_players = [qb] + team_stack.to_dict('records')
            used_salary = sum(p['salary'] for p in stack_players)
            remaining_salary = self.SALARY_CAP - used_salary

            # Fill remaining positions from other teams
            other_positions = pool[
                (~pool['player_id'].isin([p['player_id'] for p in stack_players])) &
                (pool['salary'] <= remaining_salary)
            ].sort_values('value_score', ascending=False)

            lineup = stack_players.copy()
            positions_needed = {'RB': 2, 'WR': 1, 'TE': 0, 'FLEX': 1}

            for _, player in other_positions.iterrows():
                if player['salary'] > remaining_salary:
                    continue

                pos = player['position']
                if pos in positions_needed and positions_needed[pos] > 0:
                    lineup.append(player)
                    remaining_salary -= player['salary']
                    positions_needed[pos] -= 1

                if sum(positions_needed.values()) == 0:
                    break

            # Calculate total score
            total_score = sum(p['projected_points'] for p in lineup)

            if total_score > best_score:
                best_score = total_score
                best_stack = lineup

        if best_stack:
            return pd.DataFrame(best_stack)
        else:
            return pd.DataFrame()

    def export_results(
        self,
        results: Dict[str, Any],
        week: int,
        season: int
    ) -> Dict[str, str]:
        """
        Export results to CSV files and text summary.

        Args:
            results: Dictionary with all results
            week: NFL week number
            season: NFL season year

        Returns:
            Dictionary with paths to exported files

        Example:
            >>> paths = system.export_results(results, week=5, season=2024)
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = f"{season}_week{week}_{timestamp}"
            export_paths = {}

            # Create week-specific subdirectory
            week_dir = os.path.join(self.output_dir, f"{season}_week{week}")
            os.makedirs(week_dir, exist_ok=True)

            # Export rankings
            if 'rankings' in results and not results['rankings'].empty:
                rankings_path = os.path.join(week_dir, f"{prefix}_rankings.csv")
                results['rankings'].to_csv(rankings_path, index=False)
                export_paths['rankings'] = rankings_path
                self.logger.info(f"  Exported rankings: {rankings_path}")

            # Export value plays
            if 'value_plays' in results:
                for category, df in results['value_plays'].items():
                    if not df.empty:
                        value_path = os.path.join(week_dir, f"{prefix}_value_{category}.csv")
                        df.to_csv(value_path, index=False)
                        export_paths[f'value_{category}'] = value_path

                self.logger.info(f"  Exported {len(results['value_plays'])} value play categories")

            # Export lineups
            if 'lineups' in results:
                for strategy, df in results['lineups'].items():
                    if not df.empty:
                        lineup_path = os.path.join(week_dir, f"{prefix}_lineup_{strategy}.csv")
                        df.to_csv(lineup_path, index=False)
                        export_paths[f'lineup_{strategy}'] = lineup_path

                self.logger.info(f"  Exported {len(results['lineups'])} lineup strategies")

            # Export text summary
            summary_path = os.path.join(week_dir, f"{prefix}_summary.txt")
            self._write_summary_report(results, summary_path, week, season)
            export_paths['summary'] = summary_path
            self.logger.info(f"  Exported summary: {summary_path}")

            return export_paths

        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return {}

    def _write_summary_report(
        self,
        results: Dict[str, Any],
        file_path: str,
        week: int,
        season: int
    ) -> None:
        """Write text summary report with key insights."""
        with open(file_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"FANTASY FOOTBALL PROJECTIONS - {season} WEEK {week}\n")
            f.write("="*70 + "\n\n")

            # Top projected players by position
            if 'rankings' in results and not results['rankings'].empty:
                rankings = results['rankings']

                for position in ['QB', 'RB', 'WR', 'TE']:
                    f.write(f"\n{position} TOP 10\n")
                    f.write("-"*70 + "\n")

                    pos_data = rankings[rankings['position'] == position].head(10)

                    for _, player in pos_data.iterrows():
                        f.write(
                            f"{int(player['position_rank']):2d}. {player['player_name']:25s} "
                            f"({player['team']:3s} vs {player.get('opponent', 'UNK'):3s}) - "
                            f"Proj: {player['projected_points']:5.1f} "
                            f"(Floor: {player['floor_points']:5.1f}, "
                            f"Ceil: {player['ceiling_points']:5.1f}) "
                            f"${int(player['salary']):,}\n"
                        )

            # Key value plays
            if 'value_plays' in results:
                f.write("\n" + "="*70 + "\n")
                f.write("KEY VALUE PLAYS\n")
                f.write("="*70 + "\n")

                if 'best_values' in results['value_plays']:
                    f.write("\nBEST VALUE (Points per $1000)\n")
                    f.write("-"*70 + "\n")

                    for _, player in results['value_plays']['best_values'].head(10).iterrows():
                        f.write(
                            f"{player['player_name']:25s} ({player['position']:2s}) - "
                            f"Proj: {player['projected_points']:5.1f} - "
                            f"${int(player['salary']):,} - "
                            f"Value: {player['value_score']:.2f}\n"
                        )

                if 'high_ceiling' in results['value_plays']:
                    f.write("\nHIGH CEILING UPSIDE\n")
                    f.write("-"*70 + "\n")

                    for _, player in results['value_plays']['high_ceiling'].head(10).iterrows():
                        f.write(
                            f"{player['player_name']:25s} ({player['position']:2s}) - "
                            f"Ceiling: {player['ceiling_points']:5.1f} - "
                            f"Proj: {player['projected_points']:5.1f} - "
                            f"${int(player['salary']):,}\n"
                        )

            # Sample lineup
            if 'lineups' in results and 'cash_game' in results['lineups']:
                lineup = results['lineups']['cash_game']

                if not lineup.empty:
                    f.write("\n" + "="*70 + "\n")
                    f.write("OPTIMAL CASH GAME LINEUP\n")
                    f.write("="*70 + "\n\n")

                    total_salary = lineup['salary'].sum()
                    total_projected = lineup['projected_points'].sum()

                    for _, player in lineup.iterrows():
                        f.write(
                            f"{player['position']:4s} {player['player_name']:25s} - "
                            f"Proj: {player['projected_points']:5.1f} - "
                            f"${int(player['salary']):,}\n"
                        )

                    f.write("-"*70 + "\n")
                    f.write(f"Total Salary: ${int(total_salary):,} / ${self.SALARY_CAP:,}\n")
                    f.write(f"Remaining: ${self.SALARY_CAP - int(total_salary):,}\n")
                    f.write(f"Total Projected: {total_projected:.1f} points\n")

            # Footer
            f.write("\n" + "="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize database
    db_manager = DatabaseManager(db_path='fantasy_football.db')

    try:
        # Initialize database schema (if not already done)
        db_manager.initialize_database()

        # Create weekly projection system
        system = WeeklyProjectionSystem(
            db_manager=db_manager,
            output_dir='output',
            log_file='output/weekly_system.log'
        )

        # Run complete weekly projections
        print("Running weekly projection system...")
        results = system.run_weekly_projections(
            week=5,
            season=2024,
            update_data=True,
            train_models=True  # Set to True for first run or retraining
        )

        if results['success']:
            print("\n✓ Weekly projections completed successfully!")
            print(f"  Duration: {results['duration_seconds']:.2f}s")
            print(f"  Projections: {len(results.get('projections', []))} players")
            print(f"  Export paths: {results.get('export_paths', {})}")
        else:
            print("\n✗ Weekly projections failed")
            print(f"  Error: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db_manager.close()
