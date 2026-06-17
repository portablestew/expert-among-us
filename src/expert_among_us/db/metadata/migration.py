"""Migration script for existing single-project expert databases to the multi-project schema.

Transforms the old schema:
  - experts(name, workspace_path, subdirs, vcs_type, last_processed_commit_hash,
            first_processed_commit_hash, created_at, last_indexed_at)
  - changelists(id, expert_name, timestamp, author, message, diff, files,
                review_comments, generated_prompt)

Into the new multi-project schema:
  - experts(name, description, created_at, last_indexed_at)
  - projects(expert_name, name, workspace_path, subdirs, vcs_type,
             last_processed_commit_hash, first_processed_commit_hash,
             has_vector_metadata, created_at, last_indexed_at)
  - changelists(id, expert_name, project_name, ...)

The migration is:
  - Atomic: wrapped in a single SQLite transaction
  - Idempotent: uses IF NOT EXISTS / column-existence checks so re-running is a no-op
"""

import sqlite3
from pathlib import Path
from typing import Optional


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    """Check if a table exists in the database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    return cursor.fetchone() is not None


def migrate_to_multi_project(db_path: str) -> None:
    """Migrate an existing single-project expert database to the multi-project schema.

    This function:
      1. Creates the `projects` table if it does not exist.
      2. For each expert row that still has workspace_path (old schema),
         inserts a project with name=expert_name, inheriting workspace_path,
         subdirs, vcs_type, and commit tracking fields.
      3. Adds `project_name` column to changelists and populates it with expert_name.
      4. Recreates the experts table without workspace_path, subdirs, vcs_type
         and adds a `description` column.
      5. Sets has_vector_metadata = 0 for all migrated projects.

    The entire migration runs in a single transaction for atomicity.
    IF NOT EXISTS and column-existence checks ensure idempotency.

    Args:
        db_path: Absolute path to the metadata.db file to migrate.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Use a single transaction for atomicity
        cursor.execute("BEGIN IMMEDIATE")

        # --- Step 1: Create projects table if not exists ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                expert_name TEXT NOT NULL,
                name TEXT NOT NULL,
                workspace_path TEXT NOT NULL,
                subdirs TEXT,
                vcs_type TEXT NOT NULL,
                last_indexed_at TIMESTAMP,
                last_processed_commit_hash TEXT,
                first_processed_commit_hash TEXT,
                has_vector_metadata INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (expert_name, name)
            );
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_projects_expert ON projects(expert_name);"
        )

        # --- Step 2: Migrate expert data into projects ---
        # Only migrate if the experts table still has workspace_path (old schema)
        if _column_exists(cursor, "experts", "workspace_path"):
            cursor.execute("""
                SELECT name, workspace_path, subdirs, vcs_type,
                       last_processed_commit_hash, first_processed_commit_hash,
                       created_at
                FROM experts
            """)
            expert_rows = cursor.fetchall()

            for row in expert_rows:
                expert_name = row["name"]
                workspace_path = row["workspace_path"] or ""
                subdirs = row["subdirs"] or ""
                vcs_type = row["vcs_type"] or "git"
                last_processed = row["last_processed_commit_hash"]
                first_processed = row["first_processed_commit_hash"]
                created_at = row["created_at"]

                # Insert project with name=expert_name (idempotent via INSERT OR IGNORE)
                cursor.execute("""
                    INSERT OR IGNORE INTO projects
                        (expert_name, name, workspace_path, subdirs, vcs_type,
                         last_processed_commit_hash, first_processed_commit_hash,
                         has_vector_metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (
                    expert_name, expert_name, workspace_path, subdirs,
                    vcs_type, last_processed, first_processed, created_at
                ))

        # --- Step 3: Add project_name column to changelists ---
        if _table_exists(cursor, "changelists"):
            if not _column_exists(cursor, "changelists", "project_name"):
                cursor.execute(
                    "ALTER TABLE changelists ADD COLUMN project_name TEXT"
                )
                # Populate project_name with expert_name for all existing rows
                cursor.execute("""
                    UPDATE changelists
                    SET project_name = expert_name
                    WHERE project_name IS NULL
                """)
            else:
                # Column exists but might have NULL values from a partial migration
                cursor.execute("""
                    UPDATE changelists
                    SET project_name = expert_name
                    WHERE project_name IS NULL
                """)

        # --- Step 4: Recreate experts table without workspace/VCS fields ---
        if _column_exists(cursor, "experts", "workspace_path"):
            # Read all existing expert data we want to keep
            cursor.execute("""
                SELECT name, created_at, last_indexed_at
                FROM experts
            """)
            experts_data = cursor.fetchall()

            # Drop and recreate the experts table with new schema
            cursor.execute("DROP TABLE IF EXISTS experts")
            cursor.execute("""
                CREATE TABLE experts (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_indexed_at TIMESTAMP
                );
            """)

            # Re-insert expert rows with new schema
            for row in experts_data:
                cursor.execute("""
                    INSERT OR IGNORE INTO experts (name, description, created_at, last_indexed_at)
                    VALUES (?, NULL, ?, ?)
                """, (row["name"], row["created_at"], row["last_indexed_at"]))

        else:
            # Experts table already has new schema; ensure description column exists
            if not _column_exists(cursor, "experts", "description"):
                cursor.execute(
                    "ALTER TABLE experts ADD COLUMN description TEXT"
                )

        # --- Step 5: Ensure has_vector_metadata = 0 for all migrated projects ---
        # This is safe to run idempotently: any project that was created by
        # this migration (not by new indexing) should have has_vector_metadata=0.
        # We only touch rows where has_vector_metadata might still be the default (1)
        # and the project was created by migration (name == expert_name pattern).
        # For safety, set all existing projects to 0 if they don't have vector metadata
        # (projects created via new indexing will have it set to 1 explicitly).
        # The INSERT OR IGNORE in step 2 already sets has_vector_metadata=0,
        # so this is a no-op guard for idempotency.
        cursor.execute("""
            UPDATE projects
            SET has_vector_metadata = 0
            WHERE has_vector_metadata = 1
              AND expert_name = name
        """)

        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
