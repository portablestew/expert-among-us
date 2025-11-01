import sqlite3
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
from expert_among_us.db.metadata.base import MetadataDB
from expert_among_us.models.changelist import Changelist
from expert_among_us.utils.compression import compress_diff, decompress_diff

class SQLiteMetadataDB(MetadataDB):
    def __init__(self, expert_name: str, data_dir: Optional[Path] = None):
        # Use Path.home() to properly expand home directory on all platforms (Windows, Linux, Mac)
        base_dir = data_dir or Path.home() / ".expert-among-us"
        self.db_path = str(base_dir / "data" / expert_name / "metadata.db")
        self.conn = None
        self.expert_name = expert_name
    
    def exists(self) -> bool:
        """Check if the database file exists."""
        return Path(self.db_path).exists()
        
    def _connect(self, require_exists: bool = False):
        """Connect to the database if not already connected.
        
        Args:
            require_exists: If True, raise an error if the database doesn't exist
        """
        if self.conn is None:
            if require_exists and not self.exists():
                raise FileNotFoundError(
                    f"Expert database '{self.expert_name}' does not exist. "
                    f"Please run 'populate' command first to create the expert index."
                )
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False
        
    def initialize(self):
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experts (
                name TEXT PRIMARY KEY,
                workspace_path TEXT NOT NULL,
                subdirs TEXT,
                vcs_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_indexed_at TIMESTAMP,
                last_commit_time TIMESTAMP,
                last_commit_hash TEXT,
                first_commit_time TIMESTAMP,
                first_commit_hash TEXT,
                max_commits INTEGER DEFAULT 10000
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS changelists (
                id TEXT PRIMARY KEY,
                expert_name TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                author TEXT NOT NULL,
                message TEXT NOT NULL,
                diff BLOB NOT NULL,
                files TEXT NOT NULL,
                review_comments TEXT,
                generated_prompt TEXT,
                FOREIGN KEY (expert_name) REFERENCES experts(name) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_changelists_expert ON changelists(expert_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_changelists_timestamp ON changelists(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_changelists_author ON changelists(author);")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS changelist_files (
                changelist_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                PRIMARY KEY (changelist_id, file_path),
                FOREIGN KEY (changelist_id) REFERENCES changelists(id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON changelist_files(file_path);")
        self.conn.commit()
        
    def create_expert(self, name: str, workspace_path: str, subdirs: list[str], vcs_type: str) -> None:
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO experts (name, workspace_path, subdirs, vcs_type)
            VALUES (?, ?, ?, ?)
        """, (
            name,
            workspace_path,
            ",".join(subdirs),
            vcs_type
        ))
        self.conn.commit()
    
    def get_expert(self, name: str) -> Optional[dict]:
        self._connect(require_exists=True)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, workspace_path, subdirs, vcs_type, last_indexed_at, last_commit_time, last_commit_hash, first_commit_time, first_commit_hash
            FROM experts
            WHERE name = ?
        """, (name,))
        row = cursor.fetchone()
        if row:
            # Parse timestamp strings back to datetime objects
            last_indexed_at = None
            if row['last_indexed_at']:
                try:
                    last_indexed_at = datetime.fromisoformat(row['last_indexed_at'])
                except (ValueError, TypeError):
                    last_indexed_at = None
            
            last_commit_time = None
            if row['last_commit_time']:
                try:
                    last_commit_time = datetime.fromisoformat(row['last_commit_time'])
                except (ValueError, TypeError):
                    last_commit_time = None
            
            first_commit_time = None
            if row['first_commit_time']:
                try:
                    first_commit_time = datetime.fromisoformat(row['first_commit_time'])
                except (ValueError, TypeError):
                    first_commit_time = None
            
            return {
                'name': row['name'],
                'workspace_path': row['workspace_path'],
                'subdirs': row['subdirs'].split(',') if row['subdirs'] else [],
                'vcs_type': row['vcs_type'],
                'last_indexed_at': last_indexed_at,
                'last_commit_time': last_commit_time,
                'last_commit_hash': row['last_commit_hash'],
                'first_commit_time': first_commit_time,
                'first_commit_hash': row['first_commit_hash']
            }
        return None
    
    def update_expert_index_time(self, name: str, timestamp: datetime) -> None:
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE experts
            SET last_indexed_at = ?
            WHERE name = ?
        """, (timestamp.isoformat(), name))
        self.conn.commit()
    
    def update_expert_last_commit(self, name: str, last_commit_time: datetime, last_commit_hash: str) -> None:
        """Update the last commit time and hash for an expert."""
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE experts
            SET last_commit_time = ?, last_commit_hash = ?
            WHERE name = ?
        """, (last_commit_time.isoformat(), last_commit_hash, name))
        self.conn.commit()
    
    def update_expert_first_commit(self, name: str, first_commit_time: datetime, first_commit_hash: str) -> None:
        """Update the first commit time and hash for an expert."""
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE experts
            SET first_commit_time = ?, first_commit_hash = ?
            WHERE name = ?
        """, (first_commit_time.isoformat(), first_commit_hash, name))
        self.conn.commit()
        
    def insert_changelists(self, changelists: list[Changelist]) -> None:
        self._connect()
        cursor = self.conn.cursor()
        for changelist in changelists:
            # Compress diff before storing
            diff_compressed = compress_diff(changelist.diff)
            
            cursor.execute("""
                INSERT OR REPLACE INTO changelists (id, expert_name, timestamp, author, message, diff, files, review_comments, generated_prompt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                changelist.id,
                getattr(changelist, 'expert_name', ''),
                changelist.timestamp.isoformat(),
                changelist.author,
                changelist.message,
                diff_compressed,
                ",".join(changelist.files),
                getattr(changelist, 'review_comments', None),
                getattr(changelist, 'generated_prompt', None)
            ))
            for file in changelist.files:
                cursor.execute("""
                    INSERT OR IGNORE INTO changelist_files (changelist_id, file_path)
                    VALUES (?, ?)
                """, (changelist.id, file))
        self.conn.commit()
    
    def get_changelist(self, changelist_id: str) -> Optional[Changelist]:
        changelists = self.get_changelists_by_ids([changelist_id])
        return changelists[0] if changelists else None
    
    def get_changelists_by_ids(self, ids: list[str]) -> list[Changelist]:
        """Retrieve multiple changelists by their IDs."""
        if not ids:
            return []
        
        self._connect(require_exists=True)
        cursor = self.conn.cursor()
        
        # Use parameterized query with placeholders
        placeholders = ','.join('?' * len(ids))
        query = f"""
            SELECT id, expert_name, timestamp, author, message, diff, files, review_comments, generated_prompt
            FROM changelists
            WHERE id IN ({placeholders})
        """
        
        cursor.execute(query, ids)
        rows = cursor.fetchall()
        
        changelists = []
        for row in rows:
            # Always decompress - all diffs are stored compressed
            diff = decompress_diff(row['diff'])
            
            # Parse files list, ensuring at least one element to satisfy validation
            # This matches the write behavior in git.py where empty files get [""]
            files_str = row['files']
            if files_str:
                files = files_str.split(',')
            else:
                files = [""]  # Ensure at least one file to satisfy validation
            
            changelist = Changelist(
                id=row['id'],
                expert_name=row['expert_name'],
                timestamp=row['timestamp'],
                author=row['author'],
                message=row['message'],
                diff=diff,
                files=files,
                review_comments=row['review_comments'],
                generated_prompt=row['generated_prompt']
            )
            changelists.append(changelist)
        
        return changelists
    
    def query_changelists_by_author(self, author: str) -> list[str]:
        self._connect(require_exists=True)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM changelists
            WHERE author = ?
        """, (author,))
        return [row['id'] for row in cursor.fetchall()]
    
    def query_changelists_by_files(self, file_paths: list[str]) -> list[str]:
        if not file_paths:
            return []
        self._connect(require_exists=True)
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(file_paths))
        query = f"""
            SELECT DISTINCT changelist_id
            FROM changelist_files
            WHERE file_path IN ({placeholders})
        """
        cursor.execute(query, file_paths)
        return [row['changelist_id'] for row in cursor.fetchall()]
    
    def get_generated_prompt(self, query_id: str) -> Optional[str]:
        self._connect(require_exists=True)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT generated_prompt FROM changelists
            WHERE id = ?
        """, (query_id,))
        row = cursor.fetchone()
        return row['generated_prompt'] if row else None
    
    def cache_generated_prompt(self, query_id: str, prompt: str) -> None:
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE changelists
            SET generated_prompt = ?
            WHERE id = ?
        """, (prompt, query_id))
        self.conn.commit()
        
    def close(self):
        if self.conn:
            self.conn.close()