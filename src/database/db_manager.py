"""
Database Manager for Fantasy Football Projections

This module provides a comprehensive database management class for handling
SQLite database operations including initialization, connection pooling,
querying, bulk inserts, and backups.
"""

import sqlite3
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from threading import Lock
import queue


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database connections and operations for fantasy football data.

    Features:
    - Connection pooling for concurrent access
    - Automatic schema initialization
    - Bulk insert operations
    - Database backup functionality
    - Comprehensive error handling and logging

    Attributes:
        db_path (str): Path to the SQLite database file
        schema_path (str): Path to the SQL schema file
        pool_size (int): Maximum number of connections in the pool
        connection_pool (queue.Queue): Pool of available database connections
        lock (Lock): Thread lock for connection pool management
    """

    def __init__(
        self,
        db_path: str = 'fantasy_football.db',
        schema_path: Optional[str] = None,
        pool_size: int = 5
    ):
        """
        Initialize the DatabaseManager.

        Args:
            db_path: Path to the SQLite database file (default: 'fantasy_football.db')
            schema_path: Path to the schema.sql file. If None, uses default location
            pool_size: Maximum number of connections to maintain in pool (default: 5)
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.lock = Lock()

        # Set default schema path if not provided
        if schema_path is None:
            current_dir = Path(__file__).parent
            self.schema_path = str(current_dir / 'schema.sql')
        else:
            self.schema_path = schema_path

        # Initialize connection pool
        self.connection_pool = queue.Queue(maxsize=pool_size)
        self._initialize_pool()

        logger.info(f"DatabaseManager initialized with db_path: {db_path}")

    def _initialize_pool(self) -> None:
        """
        Initialize the connection pool with database connections.

        Creates pool_size connections and adds them to the pool queue.
        Each connection is configured with:
        - Row factory for dictionary-like access
        - Foreign key constraints enabled
        - WAL mode for better concurrent access
        """
        try:
            for _ in range(self.pool_size):
                conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for concurrency
                self.connection_pool.put(conn)

            logger.info(f"Connection pool initialized with {self.pool_size} connections")
        except sqlite3.Error as e:
            logger.error(f"Error initializing connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool (context manager).

        This method implements the context manager protocol, ensuring
        connections are properly returned to the pool after use.

        Yields:
            sqlite3.Connection: A database connection from the pool

        Example:
            >>> with db_manager.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM players")
        """
        conn = None
        try:
            # Get connection from pool (blocks if pool is empty)
            conn = self.connection_pool.get(timeout=10)
            yield conn
        except queue.Empty:
            logger.error("Connection pool exhausted - timeout waiting for connection")
            raise RuntimeError("Unable to get database connection - pool exhausted")
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise
        finally:
            # Return connection to pool
            if conn is not None:
                try:
                    self.connection_pool.put(conn, timeout=1)
                except queue.Full:
                    # Should not happen, but close connection if pool is somehow full
                    conn.close()
                    logger.warning("Connection pool full - closed extra connection")

    def initialize_database(self) -> bool:
        """
        Initialize the database by executing the schema.sql file.

        Reads the SQL schema file and executes all statements to create
        tables, indexes, views, and triggers. Handles SQLite-specific
        adaptations of PostgreSQL syntax.

        Returns:
            bool: True if initialization successful, False otherwise

        Raises:
            FileNotFoundError: If schema file doesn't exist
            sqlite3.Error: If there's an error executing SQL
        """
        try:
            # Check if schema file exists
            if not os.path.exists(self.schema_path):
                raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

            # Read schema file
            with open(self.schema_path, 'r') as f:
                schema_sql = f.read()

            # Adapt PostgreSQL syntax to SQLite
            schema_sql = self._adapt_schema_for_sqlite(schema_sql)

            # Execute schema
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Split and execute statements
                # SQLite executescript doesn't support parameterized queries but is safe for schema
                cursor.executescript(schema_sql)
                conn.commit()

            logger.info("Database schema initialized successfully")
            return True

        except FileNotFoundError as e:
            logger.error(f"Schema file not found: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            raise

    def _adapt_schema_for_sqlite(self, schema_sql: str) -> str:
        """
        Adapt PostgreSQL schema syntax to SQLite.

        Args:
            schema_sql: The original SQL schema string

        Returns:
            str: Adapted SQL schema for SQLite
        """
        import re

        # Remove PostgreSQL-specific syntax
        adaptations = [
            # Remove MATERIALIZED keyword (SQLite doesn't support it)
            ('CREATE MATERIALIZED VIEW', 'CREATE VIEW'),
            # Remove IF NOT EXISTS for triggers (will be skipped if exist)
            ('CREATE OR REPLACE FUNCTION', '-- CREATE OR REPLACE FUNCTION'),
            # Comment out trigger creation (SQLite syntax is different)
            ('CREATE TRIGGER update_', '-- CREATE TRIGGER update_'),
            # Remove function language specification
            ("language 'plpgsql'", ''),
            # Convert DECIMAL to REAL
            ('DECIMAL', 'REAL'),
            # Convert VARCHAR to TEXT
            ('VARCHAR', 'TEXT'),
        ]

        adapted_sql = schema_sql
        for old, new in adaptations:
            adapted_sql = adapted_sql.replace(old, new)

        # Comment out entire function and trigger section (not supported in SQLite)
        # Find the section between "HELPFUL FUNCTIONS AND TRIGGERS" and "COMMENTS FOR DOCUMENTATION"
        function_section_pattern = r'(-- HELPFUL FUNCTIONS AND TRIGGERS.*?-- COMMENTS FOR DOCUMENTATION)'

        def comment_section(match):
            section = match.group(1)
            # Comment out each line in the section except lines that are already comments
            lines = section.split('\n')
            commented_lines = []
            for line in lines:
                if line.strip().startswith('--') or line.strip() == '':
                    commented_lines.append(line)
                else:
                    commented_lines.append('-- ' + line)
            return '\n'.join(commented_lines)

        adapted_sql = re.sub(
            function_section_pattern,
            comment_section,
            adapted_sql,
            flags=re.DOTALL
        )

        # Comment out view indexes (multi-line statements)
        # This regex matches CREATE INDEX statements on player_rolling_avg including multi-line
        adapted_sql = re.sub(
            r'CREATE (?:UNIQUE )?INDEX idx_player_rolling_avg[^;]*;',
            lambda m: '-- ' + m.group(0).replace('\n', '\n-- '),
            adapted_sql,
            flags=re.DOTALL
        )

        # Comment out COMMENT ON statements (not supported in SQLite)
        adapted_sql = re.sub(
            r'COMMENT ON (?:TABLE|MATERIALIZED VIEW)[^;]+;',
            lambda m: '-- ' + m.group(0),
            adapted_sql,
            flags=re.DOTALL
        )

        return adapted_sql

    def execute_query(
        self,
        query: str,
        params: Optional[Union[Tuple, Dict]] = None,
        fetch: bool = True
    ) -> Optional[List[sqlite3.Row]]:
        """
        Execute a SQL query with error handling.

        Args:
            query: SQL query string
            params: Query parameters (tuple for positional, dict for named)
            fetch: Whether to fetch and return results (default: True)

        Returns:
            List of sqlite3.Row objects if fetch=True, None otherwise

        Example:
            >>> # Positional parameters
            >>> results = db_manager.execute_query(
            ...     "SELECT * FROM players WHERE position = ?",
            ...     ('QB',)
            ... )
            >>>
            >>> # Named parameters
            >>> results = db_manager.execute_query(
            ...     "SELECT * FROM players WHERE position = :pos",
            ...     {'pos': 'QB'}
            ... )
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Execute query with or without parameters
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Fetch results if requested
                if fetch:
                    results = cursor.fetchall()
                    logger.debug(f"Query executed successfully, returned {len(results)} rows")
                    return results
                else:
                    conn.commit()
                    logger.debug(f"Query executed successfully, {cursor.rowcount} rows affected")
                    return None

        except sqlite3.Error as e:
            logger.error(f"Database error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            raise

    def bulk_insert(
        self,
        table: str,
        data: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Perform efficient bulk insert operation.

        Uses executemany for optimal performance and processes data in batches
        to avoid memory issues with large datasets.

        Args:
            table: Name of the table to insert into
            data: List of dictionaries containing row data
            batch_size: Number of rows to insert per batch (default: 1000)

        Returns:
            int: Total number of rows inserted

        Example:
            >>> players_data = [
            ...     {'player_id': 'P1', 'player_name': 'John Doe', 'position': 'QB'},
            ...     {'player_id': 'P2', 'player_name': 'Jane Smith', 'position': 'RB'}
            ... ]
            >>> rows_inserted = db_manager.bulk_insert('players', players_data)
        """
        if not data:
            logger.warning("No data provided for bulk insert")
            return 0

        try:
            # Get column names from first row
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            column_str = ', '.join(columns)

            # Build INSERT query
            query = f"INSERT INTO {table} ({column_str}) VALUES ({placeholders})"

            total_inserted = 0

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Process in batches
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]

                    # Convert dict values to tuples in correct order
                    values = [tuple(row[col] for col in columns) for row in batch]

                    cursor.executemany(query, values)
                    conn.commit()

                    total_inserted += len(batch)
                    logger.debug(f"Inserted batch of {len(batch)} rows into {table}")

            logger.info(f"Bulk insert completed: {total_inserted} rows inserted into {table}")
            return total_inserted

        except sqlite3.Error as e:
            logger.error(f"Database error during bulk insert: {e}")
            logger.error(f"Table: {table}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during bulk insert: {e}")
            raise

    def backup_database(self, backup_dir: str = 'backups') -> str:
        """
        Create a timestamped backup of the database.

        Creates a backup directory if it doesn't exist and copies the database
        file with a timestamp in the filename.

        Args:
            backup_dir: Directory to store backups (default: 'backups')

        Returns:
            str: Path to the backup file

        Example:
            >>> backup_path = db_manager.backup_database()
            >>> print(f"Backup created at: {backup_path}")
        """
        try:
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)

            # Generate timestamp for backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"fantasy_football_backup_{timestamp}.db"
            backup_path = os.path.join(backup_dir, backup_filename)

            # Close all connections and create backup
            # Use VACUUM INTO for SQLite 3.27+, otherwise use file copy
            try:
                with self.get_connection() as conn:
                    conn.execute(f"VACUUM INTO '{backup_path}'")
                logger.info(f"Database backed up using VACUUM INTO: {backup_path}")
            except sqlite3.OperationalError:
                # Fallback to file copy for older SQLite versions
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Database backed up using file copy: {backup_path}")

            # Verify backup was created
            if os.path.exists(backup_path):
                size_mb = os.path.getsize(backup_path) / (1024 * 1024)
                logger.info(f"Backup successful: {backup_path} ({size_mb:.2f} MB)")
                return backup_path
            else:
                raise RuntimeError("Backup file was not created")

        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            raise

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of dictionaries containing column information
        """
        try:
            query = f"PRAGMA table_info({table_name})"
            results = self.execute_query(query)

            columns = []
            for row in results:
                columns.append({
                    'name': row['name'],
                    'type': row['type'],
                    'notnull': bool(row['notnull']),
                    'default': row['dflt_value'],
                    'pk': bool(row['pk'])
                })

            return columns

        except sqlite3.Error as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            raise

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.

        Returns:
            Dictionary containing database statistics including table counts
            and database size
        """
        try:
            stats = {}

            # Get list of tables
            tables_query = """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
            tables = self.execute_query(tables_query)
            table_names = [row['name'] for row in tables]

            # Get row count for each table
            table_counts = {}
            for table in table_names:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = self.execute_query(count_query)
                table_counts[table] = result[0]['count']

            # Get database file size
            db_size_bytes = os.path.getsize(self.db_path)
            db_size_mb = db_size_bytes / (1024 * 1024)

            stats['database_path'] = self.db_path
            stats['database_size_mb'] = round(db_size_mb, 2)
            stats['table_count'] = len(table_names)
            stats['tables'] = table_counts
            stats['total_rows'] = sum(table_counts.values())

            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            raise

    def close(self) -> None:
        """
        Close all database connections in the pool.

        This should be called when shutting down the application to ensure
        all connections are properly closed.
        """
        try:
            closed_count = 0

            # Close all connections in the pool
            while not self.connection_pool.empty():
                try:
                    conn = self.connection_pool.get_nowait()
                    conn.close()
                    closed_count += 1
                except queue.Empty:
                    break

            logger.info(f"DatabaseManager closed: {closed_count} connections closed")

        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit - close all connections."""
        self.close()

    def __del__(self):
        """Destructor - ensure connections are closed."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup


# Example usage and testing
if __name__ == "__main__":
    # Initialize database manager
    db_manager = DatabaseManager(
        db_path='fantasy_football.db',
        pool_size=5
    )

    try:
        # Initialize database schema
        print("Initializing database schema...")
        db_manager.initialize_database()

        # Get database statistics
        print("\nDatabase Statistics:")
        stats = db_manager.get_database_stats()
        print(f"Database Size: {stats['database_size_mb']} MB")
        print(f"Total Tables: {stats['table_count']}")
        print(f"Total Rows: {stats['total_rows']}")

        # Example: Insert sample player data
        print("\nInserting sample data...")
        sample_players = [
            {
                'player_id': 'P001',
                'player_name': 'Patrick Mahomes',
                'position': 'QB',
                'team': 'KC',
                'years_experience': 7,
                'draft_year': 2017
            },
            {
                'player_id': 'P002',
                'player_name': 'Christian McCaffrey',
                'position': 'RB',
                'team': 'SF',
                'years_experience': 7,
                'draft_year': 2017
            }
        ]

        rows_inserted = db_manager.bulk_insert('players', sample_players)
        print(f"Inserted {rows_inserted} players")

        # Query data
        print("\nQuerying players...")
        players = db_manager.execute_query("SELECT * FROM players")
        for player in players:
            print(f"  {player['player_name']} ({player['position']}) - {player['team']}")

        # Create backup
        print("\nCreating backup...")
        backup_path = db_manager.backup_database()
        print(f"Backup created: {backup_path}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close all connections
        db_manager.close()
        print("\nDatabase connections closed")
