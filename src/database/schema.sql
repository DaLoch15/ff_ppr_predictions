-- Fantasy Football Projections Database Schema
-- Comprehensive schema for storing player stats, team data, and projections

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Players: Core player information
CREATE TABLE players (
    player_id VARCHAR(50) PRIMARY KEY,
    player_name VARCHAR(100) NOT NULL,
    position VARCHAR(5) NOT NULL CHECK (position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DST')),
    team VARCHAR(3),
    years_experience INTEGER DEFAULT 0,
    draft_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player Games: Detailed game-level statistics for players
CREATE TABLE player_games (
    game_id VARCHAR(50) NOT NULL,
    player_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL CHECK (week BETWEEN 1 AND 18),
    game_date DATE NOT NULL,
    team VARCHAR(3) NOT NULL,
    opponent VARCHAR(3) NOT NULL,
    is_home BOOLEAN NOT NULL,

    -- Snap and route data
    snap_count INTEGER DEFAULT 0,
    snap_pct DECIMAL(5,2) CHECK (snap_pct BETWEEN 0 AND 100),
    routes_run INTEGER DEFAULT 0,
    route_participation DECIMAL(5,2) CHECK (route_participation BETWEEN 0 AND 100),

    -- Passing stats
    completions INTEGER DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    passing_yards INTEGER DEFAULT 0,
    passing_tds INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    sacks INTEGER DEFAULT 0,

    -- Rushing stats
    carries INTEGER DEFAULT 0,
    rushing_yards INTEGER DEFAULT 0,
    rushing_tds INTEGER DEFAULT 0,

    -- Receiving stats
    targets INTEGER DEFAULT 0,
    receptions INTEGER DEFAULT 0,
    receiving_yards INTEGER DEFAULT 0,
    receiving_tds INTEGER DEFAULT 0,
    air_yards INTEGER DEFAULT 0,
    yards_after_catch INTEGER DEFAULT 0,

    -- Advanced metrics
    target_share DECIMAL(5,2) CHECK (target_share BETWEEN 0 AND 100),
    air_yard_share DECIMAL(5,2) CHECK (air_yard_share BETWEEN 0 AND 100),
    red_zone_targets INTEGER DEFAULT 0,
    red_zone_carries INTEGER DEFAULT 0,

    -- Fantasy points
    fantasy_points_ppr DECIMAL(6,2) DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (game_id, player_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE
);

-- Team Games: Team-level offensive statistics
CREATE TABLE team_games (
    game_id VARCHAR(50) NOT NULL,
    team VARCHAR(3) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL CHECK (week BETWEEN 1 AND 18),
    opponent VARCHAR(3) NOT NULL,
    is_home BOOLEAN NOT NULL,

    -- Offensive stats
    total_plays INTEGER DEFAULT 0,
    pass_attempts INTEGER DEFAULT 0,
    rush_attempts INTEGER DEFAULT 0,
    pass_rate DECIMAL(5,2) CHECK (pass_rate BETWEEN 0 AND 100),
    total_yards INTEGER DEFAULT 0,
    points_scored INTEGER DEFAULT 0,

    -- Pace metrics
    time_of_possession INTEGER, -- in seconds
    seconds_per_play DECIMAL(5,2),
    no_huddle_rate DECIMAL(5,2) CHECK (no_huddle_rate BETWEEN 0 AND 100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (game_id, team)
);

-- Defensive Rankings: Week-by-week defensive performance metrics
CREATE TABLE defensive_rankings (
    team VARCHAR(3) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL CHECK (week BETWEEN 1 AND 18),

    -- Fantasy points allowed by position
    qb_points_allowed_avg DECIMAL(6,2) DEFAULT 0,
    rb_points_allowed_avg DECIMAL(6,2) DEFAULT 0,
    wr_points_allowed_avg DECIMAL(6,2) DEFAULT 0,
    te_points_allowed_avg DECIMAL(6,2) DEFAULT 0,

    -- Targets allowed
    targets_allowed_to_rb DECIMAL(6,2) DEFAULT 0,
    targets_allowed_to_wr DECIMAL(6,2) DEFAULT 0,
    targets_allowed_to_te DECIMAL(6,2) DEFAULT 0,

    -- Coverage metrics
    slot_yards_allowed_avg DECIMAL(6,2) DEFAULT 0,
    outside_yards_allowed_avg DECIMAL(6,2) DEFAULT 0,
    man_coverage_rate DECIMAL(5,2) CHECK (man_coverage_rate BETWEEN 0 AND 100),
    zone_coverage_rate DECIMAL(5,2) CHECK (zone_coverage_rate BETWEEN 0 AND 100),

    -- Pressure metrics
    blitz_rate DECIMAL(5,2) CHECK (blitz_rate BETWEEN 0 AND 100),
    pressure_rate DECIMAL(5,2) CHECK (pressure_rate BETWEEN 0 AND 100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (team, season, week)
);

-- Vegas Lines: Betting lines and implied totals
CREATE TABLE vegas_lines (
    game_id VARCHAR(50) PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL CHECK (week BETWEEN 1 AND 18),
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,

    -- Lines
    spread DECIMAL(5,2), -- positive means home team is underdog
    total DECIMAL(5,2),
    home_implied_total DECIMAL(5,2),
    away_implied_total DECIMAL(5,2),

    -- Moneylines
    moneyline_home INTEGER,
    moneyline_away INTEGER,

    -- Line movement
    line_movement DECIMAL(5,2), -- change from opening
    total_movement DECIMAL(5,2), -- change from opening

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Injuries: Player injury status and practice participation
CREATE TABLE injuries (
    player_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL CHECK (week BETWEEN 1 AND 18),
    injury_status VARCHAR(20) CHECK (injury_status IN ('Out', 'Doubtful', 'Questionable', 'Probable', 'IR', 'PUP', 'Healthy')),
    injury_type VARCHAR(100),

    -- Practice participation
    practice_wed VARCHAR(20) CHECK (practice_wed IN ('DNP', 'Limited', 'Full', 'N/A')),
    practice_thu VARCHAR(20) CHECK (practice_thu IN ('DNP', 'Limited', 'Full', 'N/A')),
    practice_fri VARCHAR(20) CHECK (practice_fri IN ('DNP', 'Limited', 'Full', 'N/A')),

    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (player_id, season, week),
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE
);

-- Weather: Game weather conditions
CREATE TABLE weather (
    game_id VARCHAR(50) PRIMARY KEY,
    temperature INTEGER, -- in Fahrenheit
    wind_speed INTEGER, -- in MPH
    precipitation VARCHAR(20) CHECK (precipitation IN ('None', 'Light', 'Moderate', 'Heavy')),
    dome BOOLEAN NOT NULL DEFAULT FALSE,
    weather_score INTEGER CHECK (weather_score BETWEEN 0 AND 100), -- 100 = ideal conditions

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (game_id) REFERENCES vegas_lines(game_id) ON DELETE CASCADE
);

-- Projections: Model predictions for player performance
CREATE TABLE projections (
    projection_id VARCHAR(50) PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL CHECK (week BETWEEN 1 AND 18),

    -- Overall projections
    projected_points DECIMAL(6,2) NOT NULL,
    floor_points DECIMAL(6,2),
    ceiling_points DECIMAL(6,2),

    -- Position-specific projections
    projected_targets INTEGER,
    projected_receptions INTEGER,
    projected_rec_yards INTEGER,
    projected_rec_tds DECIMAL(4,2),
    projected_rush_attempts INTEGER,
    projected_rush_yards INTEGER,
    projected_rush_tds DECIMAL(4,2),

    -- Metadata
    confidence_score DECIMAL(5,2) CHECK (confidence_score BETWEEN 0 AND 100),
    projection_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(20) NOT NULL,

    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    UNIQUE (player_id, season, week, model_version)
);

-- Projection Results: Track accuracy of projections
CREATE TABLE projection_results (
    projection_id VARCHAR(50) PRIMARY KEY,
    actual_points DECIMAL(6,2),
    error DECIMAL(6,2), -- actual - projected
    absolute_error DECIMAL(6,2), -- |actual - projected|
    percentile_result INTEGER CHECK (percentile_result BETWEEN 0 AND 100), -- where actual falls in projection range
    beat_espn BOOLEAN, -- did we beat ESPN's projection?
    beat_market BOOLEAN, -- did we beat market consensus?

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (projection_id) REFERENCES projections(projection_id) ON DELETE CASCADE
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Players indexes
CREATE INDEX idx_players_position ON players(position);
CREATE INDEX idx_players_team ON players(team);
CREATE INDEX idx_players_name ON players(player_name);

-- Player games indexes
CREATE INDEX idx_player_games_player ON player_games(player_id);
CREATE INDEX idx_player_games_season_week ON player_games(season, week);
CREATE INDEX idx_player_games_date ON player_games(game_date);
CREATE INDEX idx_player_games_team ON player_games(team);
CREATE INDEX idx_player_games_opponent ON player_games(opponent);
CREATE INDEX idx_player_games_player_season ON player_games(player_id, season);

-- Team games indexes
CREATE INDEX idx_team_games_team ON team_games(team);
CREATE INDEX idx_team_games_season_week ON team_games(season, week);
CREATE INDEX idx_team_games_opponent ON team_games(opponent);

-- Defensive rankings indexes
CREATE INDEX idx_defensive_rankings_team ON defensive_rankings(team);
CREATE INDEX idx_defensive_rankings_season_week ON defensive_rankings(season, week);

-- Vegas lines indexes
CREATE INDEX idx_vegas_lines_season_week ON vegas_lines(season, week);
CREATE INDEX idx_vegas_lines_home_team ON vegas_lines(home_team);
CREATE INDEX idx_vegas_lines_away_team ON vegas_lines(away_team);

-- Injuries indexes
CREATE INDEX idx_injuries_player ON injuries(player_id);
CREATE INDEX idx_injuries_season_week ON injuries(season, week);
CREATE INDEX idx_injuries_status ON injuries(injury_status);

-- Projections indexes
CREATE INDEX idx_projections_player ON projections(player_id);
CREATE INDEX idx_projections_season_week ON projections(season, week);
CREATE INDEX idx_projections_player_season_week ON projections(player_id, season, week);
CREATE INDEX idx_projections_timestamp ON projections(projection_timestamp);
CREATE INDEX idx_projections_model_version ON projections(model_version);

-- Projection results indexes
CREATE INDEX idx_projection_results_accuracy ON projection_results(absolute_error);
CREATE INDEX idx_projection_results_beat_espn ON projection_results(beat_espn);

-- ============================================================================
-- MATERIALIZED VIEW: Rolling Averages
-- ============================================================================

CREATE MATERIALIZED VIEW player_rolling_avg AS
WITH game_stats AS (
    SELECT
        player_id,
        season,
        week,
        game_date,
        fantasy_points_ppr,
        targets,
        receptions,
        receiving_yards,
        receiving_tds,
        carries,
        rushing_yards,
        rushing_tds,
        attempts as pass_attempts,
        passing_yards,
        passing_tds,
        interceptions,
        snap_pct,
        target_share,
        ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) as game_number
    FROM player_games
    WHERE snap_count > 0  -- Only include games where player was active
),
rolling_3game AS (
    SELECT
        player_id,
        season,
        week,
        AVG(fantasy_points_ppr) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_fantasy_points_3g,
        AVG(targets) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_targets_3g,
        AVG(receptions) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_receptions_3g,
        AVG(receiving_yards) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_receiving_yards_3g,
        AVG(carries) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_carries_3g,
        AVG(rushing_yards) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_rushing_yards_3g,
        AVG(passing_yards) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_passing_yards_3g,
        AVG(snap_pct) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_snap_pct_3g,
        AVG(target_share) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as avg_target_share_3g
    FROM game_stats
),
rolling_5game AS (
    SELECT
        player_id,
        season,
        week,
        AVG(fantasy_points_ppr) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_fantasy_points_5g,
        AVG(targets) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_targets_5g,
        AVG(receptions) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_receptions_5g,
        AVG(receiving_yards) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_receiving_yards_5g,
        AVG(carries) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_carries_5g,
        AVG(rushing_yards) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_rushing_yards_5g,
        AVG(passing_yards) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_passing_yards_5g,
        AVG(snap_pct) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_snap_pct_5g,
        AVG(target_share) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as avg_target_share_5g
    FROM game_stats
),
season_avg AS (
    SELECT
        player_id,
        season,
        week,
        AVG(fantasy_points_ppr) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_fantasy_points_season,
        AVG(targets) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_targets_season,
        AVG(receptions) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_receptions_season,
        AVG(receiving_yards) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_receiving_yards_season,
        AVG(carries) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_carries_season,
        AVG(rushing_yards) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_rushing_yards_season,
        AVG(passing_yards) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_passing_yards_season,
        AVG(snap_pct) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_snap_pct_season,
        AVG(target_share) OVER (
            PARTITION BY player_id, season
            ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as avg_target_share_season
    FROM game_stats
)
SELECT
    r3.player_id,
    r3.season,
    r3.week,
    -- 3-game rolling averages
    ROUND(CAST(r3.avg_fantasy_points_3g AS NUMERIC), 2) as avg_fantasy_points_3g,
    ROUND(CAST(r3.avg_targets_3g AS NUMERIC), 2) as avg_targets_3g,
    ROUND(CAST(r3.avg_receptions_3g AS NUMERIC), 2) as avg_receptions_3g,
    ROUND(CAST(r3.avg_receiving_yards_3g AS NUMERIC), 2) as avg_receiving_yards_3g,
    ROUND(CAST(r3.avg_carries_3g AS NUMERIC), 2) as avg_carries_3g,
    ROUND(CAST(r3.avg_rushing_yards_3g AS NUMERIC), 2) as avg_rushing_yards_3g,
    ROUND(CAST(r3.avg_passing_yards_3g AS NUMERIC), 2) as avg_passing_yards_3g,
    ROUND(CAST(r3.avg_snap_pct_3g AS NUMERIC), 2) as avg_snap_pct_3g,
    ROUND(CAST(r3.avg_target_share_3g AS NUMERIC), 2) as avg_target_share_3g,
    -- 5-game rolling averages
    ROUND(CAST(r5.avg_fantasy_points_5g AS NUMERIC), 2) as avg_fantasy_points_5g,
    ROUND(CAST(r5.avg_targets_5g AS NUMERIC), 2) as avg_targets_5g,
    ROUND(CAST(r5.avg_receptions_5g AS NUMERIC), 2) as avg_receptions_5g,
    ROUND(CAST(r5.avg_receiving_yards_5g AS NUMERIC), 2) as avg_receiving_yards_5g,
    ROUND(CAST(r5.avg_carries_5g AS NUMERIC), 2) as avg_carries_5g,
    ROUND(CAST(r5.avg_rushing_yards_5g AS NUMERIC), 2) as avg_rushing_yards_5g,
    ROUND(CAST(r5.avg_passing_yards_5g AS NUMERIC), 2) as avg_passing_yards_5g,
    ROUND(CAST(r5.avg_snap_pct_5g AS NUMERIC), 2) as avg_snap_pct_5g,
    ROUND(CAST(r5.avg_target_share_5g AS NUMERIC), 2) as avg_target_share_5g,
    -- Season averages
    ROUND(CAST(sa.avg_fantasy_points_season AS NUMERIC), 2) as avg_fantasy_points_season,
    ROUND(CAST(sa.avg_targets_season AS NUMERIC), 2) as avg_targets_season,
    ROUND(CAST(sa.avg_receptions_season AS NUMERIC), 2) as avg_receptions_season,
    ROUND(CAST(sa.avg_receiving_yards_season AS NUMERIC), 2) as avg_receiving_yards_season,
    ROUND(CAST(sa.avg_carries_season AS NUMERIC), 2) as avg_carries_season,
    ROUND(CAST(sa.avg_rushing_yards_season AS NUMERIC), 2) as avg_rushing_yards_season,
    ROUND(CAST(sa.avg_passing_yards_season AS NUMERIC), 2) as avg_passing_yards_season,
    ROUND(CAST(sa.avg_snap_pct_season AS NUMERIC), 2) as avg_snap_pct_season,
    ROUND(CAST(sa.avg_target_share_season AS NUMERIC), 2) as avg_target_share_season
FROM rolling_3game r3
JOIN rolling_5game r5 ON r3.player_id = r5.player_id
    AND r3.season = r5.season
    AND r3.week = r5.week
JOIN season_avg sa ON r3.player_id = sa.player_id
    AND r3.season = sa.season
    AND r3.week = sa.week;

-- Index for the materialized view
CREATE UNIQUE INDEX idx_player_rolling_avg_player_season_week
    ON player_rolling_avg(player_id, season, week);
CREATE INDEX idx_player_rolling_avg_season_week
    ON player_rolling_avg(season, week);

-- ============================================================================
-- HELPFUL FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_players_updated_at BEFORE UPDATE ON players
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_defensive_rankings_updated_at BEFORE UPDATE ON defensive_rankings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vegas_lines_updated_at BEFORE UPDATE ON vegas_lines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE players IS 'Core player information including position, team, and experience';
COMMENT ON TABLE player_games IS 'Game-level statistics for individual players including all fantasy-relevant stats';
COMMENT ON TABLE team_games IS 'Team-level offensive statistics and pace metrics';
COMMENT ON TABLE defensive_rankings IS 'Week-by-week defensive performance metrics and coverage statistics';
COMMENT ON TABLE vegas_lines IS 'Betting lines, spreads, totals, and implied scoring projections';
COMMENT ON TABLE injuries IS 'Player injury status and weekly practice participation';
COMMENT ON TABLE weather IS 'Game weather conditions that may impact fantasy performance';
COMMENT ON TABLE projections IS 'Model-generated fantasy point projections with confidence intervals';
COMMENT ON TABLE projection_results IS 'Actual results and accuracy metrics for model projections';
COMMENT ON MATERIALIZED VIEW player_rolling_avg IS 'Pre-computed 3-game, 5-game, and season rolling averages for player performance';
