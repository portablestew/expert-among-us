import sqlite3
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
from expert_among_us.db.metadata.base import MetadataDB
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.file_chunk import FileChunk
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
                last_processed_commit_hash TEXT,
                first_processed_commit_hash TEXT,
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
        
        # File content indexing tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_contents (
                file_path TEXT PRIMARY KEY,
                expert_name TEXT NOT NULL,
                revision_id TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                last_indexed_at TIMESTAMP,
                FOREIGN KEY (expert_name) REFERENCES experts(name) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_contents_expert ON file_contents(expert_name);")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (file_path) REFERENCES file_contents(file_path) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_chunks_path ON file_chunks(file_path);")
        
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
            SELECT name, workspace_path, subdirs, vcs_type, last_indexed_at, last_processed_commit_hash, first_processed_commit_hash
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
            
            return {
                'name': row['name'],
                'workspace_path': row['workspace_path'],
                'subdirs': row['subdirs'].split(',') if row['subdirs'] else [],
                'vcs_type': row['vcs_type'],
                'last_indexed_at': last_indexed_at,
                'last_processed_commit_hash': row['last_processed_commit_hash'],
                'first_processed_commit_hash': row['first_processed_commit_hash']
            }
        return None
    
    def list_all_experts(self) -> List[dict]:
        """Retrieve all experts with their metadata.
        
        Returns:
            List of expert dictionaries with all fields from experts table
        """
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, workspace_path, subdirs, vcs_type, created_at,
                   last_indexed_at, last_processed_commit_hash, first_processed_commit_hash, max_commits
            FROM experts
            ORDER BY name ASC
        """)
        rows = cursor.fetchall()
        
        experts = []
        for row in rows:
            # Parse timestamp strings back to datetime objects
            last_indexed_at = None
            if row['last_indexed_at']:
                try:
                    last_indexed_at = datetime.fromisoformat(row['last_indexed_at'])
                except (ValueError, TypeError):
                    pass
            
            created_at = None
            if row['created_at']:
                try:
                    created_at = datetime.fromisoformat(row['created_at'])
                except (ValueError, TypeError):
                    pass
            
            experts.append({
                'name': row['name'],
                'workspace_path': row['workspace_path'],
                'subdirs': row['subdirs'].split(',') if row['subdirs'] else [],
                'vcs_type': row['vcs_type'],
                'created_at': created_at,
                'last_indexed_at': last_indexed_at,
                'last_processed_commit_hash': row['last_processed_commit_hash'],
                'first_processed_commit_hash': row['first_processed_commit_hash'],
                'max_commits': row['max_commits']
            })
        
        return experts
    
    def get_commit_count(self, expert_name: str) -> int:
        """Get the number of commits indexed for an expert.
        
        Args:
            expert_name: Name of the expert (not used, kept for API compatibility)
            
        Returns:
            Number of commits in the changelists table for this expert
        """
        self._connect()
        cursor = self.conn.cursor()
        # Each expert has its own database, so count all changelists
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM changelists
        """)
        row = cursor.fetchone()
        return row['count'] if row else 0
    
    def update_expert_index_time(self, name: str, timestamp: datetime) -> None:
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE experts
            SET last_indexed_at = ?
            WHERE name = ?
        """, (timestamp.isoformat(), name))
        self.conn.commit()
    
    def get_last_processed_commit_hash(self, expert_name: str) -> Optional[str]:
        """Get the last processed commit hash."""
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT last_processed_commit_hash FROM experts WHERE name = ?",
            (expert_name,)
        )
        result = cursor.fetchone()
        return result[0] if result and result[0] else None

    def update_last_processed_commit(self, expert_name: str, commit_hash: str) -> None:
        """Update the last processed commit hash. Also sets first_processed_commit_hash if not set."""
        self._connect()
        cursor = self.conn.cursor()
        # Update last_processed_commit_hash
        cursor.execute(
            "UPDATE experts SET last_processed_commit_hash = ? WHERE name = ?",
            (commit_hash, expert_name)
        )
        # Set first_processed_commit_hash if not already set (only on first run)
        cursor.execute(
            "UPDATE experts SET first_processed_commit_hash = ? WHERE name = ? AND first_processed_commit_hash IS NULL",
            (commit_hash, expert_name)
        )
        self.conn.commit()
        
    def get_connection(self):
        """Get database connection, creating it if needed."""
        self._connect()
        return self.conn

    def insert_changelists(self, changelists: list[Changelist]) -> None:
        """Insert or update multiple changelists.

        Notes on expert_name handling:
        - VCS providers like Git._parse_commits() may set expert_name="" because they
          do not know which expert the commits belong to.
        - Callers that know the expert (e.g. Indexer, batch pipelines) SHOULD set
          changelist.expert_name explicitly before calling this method.
        - As a safety net, when changelist.expert_name is empty we fall back to this
          SQLiteMetadataDB's expert_name so all rows are properly attributed.
        """
        self._connect()
        cursor = self.conn.cursor()
        for changelist in changelists:
            # Compress diff before storing
            diff_compressed = compress_diff(changelist.diff)

            # Ensure expert_name is populated:
            # - Prefer the value on the Changelist (set by caller).
            # - Fallback to this DB's expert_name for safety when empty.
            expert_name = (getattr(changelist, "expert_name", "") or self.expert_name)
            
            cursor.execute("""
                INSERT OR REPLACE INTO changelists (id, expert_name, timestamp, author, message, diff, files, review_comments, generated_prompt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                changelist.id,
                expert_name,
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
            
            # Parse files list from comma-separated string.
            # With updated Changelist validation, an empty list is allowed for
            # metadata-only or diff-only changelists where no file paths are present.
            files_str = row['files']
            if files_str:
                files = [f.strip() for f in files_str.split(',') if f.strip()]
            else:
                files = []
            
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
    
    def insert_file_content(self, file_path: str, expert_name: str, revision_id: str,
                           file_size: int, chunk_count: int) -> None:
        """Insert or update file content metadata.
        
        Args:
            file_path: Relative path to the file in the repository
            expert_name: Name of the expert this file belongs to
            revision_id: Revision identifier when indexed (e.g. commit hash or changelist ID)
            file_size: Size of the file in bytes
            chunk_count: Number of chunks this file was split into
        """
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO file_contents
            (file_path, expert_name, revision_id, file_size, chunk_count, last_indexed_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (file_path, expert_name, revision_id, file_size, chunk_count))
        self.conn.commit()
    
    def insert_file_chunks(self, chunks: List[FileChunk]) -> None:
        """Insert file chunks in batch.
        
        Args:
            chunks: List of FileChunk objects to insert
        """
        self._connect()
        cursor = self.conn.cursor()
        for chunk in chunks:
            cursor.execute("""
                INSERT OR REPLACE INTO file_chunks (id, file_path, chunk_index, line_start, line_end, content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk.get_chroma_id(), chunk.file_path, chunk.chunk_index,
                  chunk.line_start, chunk.line_end, chunk.content))
        self.conn.commit()
    
    def get_file_chunk(self, chunk_id: str) -> Optional[FileChunk]:
        """Retrieve a single file chunk by ID.

        Public API for future file-scoped retrieval features.

        This helper is available for use by higher-level search/query flows
        that want to hydrate a specific file chunk by its stored chunk ID, but
        is not yet wired into the main ExpertAmongUs pipeline.

        Args:
            chunk_id: The stored chunk ID of the chunk to retrieve.

        Returns:
            FileChunk object if found, None otherwise.
        """
        self._connect(require_exists=True)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fc.file_path, fc.chunk_index, fc.content, fc.line_start, fc.line_end, f.revision_id
            FROM file_chunks fc
            JOIN file_contents f ON fc.file_path = f.file_path
            WHERE fc.id = ?
        """, (chunk_id,))
        row = cursor.fetchone()
        if row:
            return FileChunk(
                file_path=row['file_path'],
                chunk_index=row['chunk_index'],
                content=row['content'],
                line_start=row['line_start'],
                line_end=row['line_end'],
                revision_id=row['revision_id']
            )
        return None
    
    def get_file_chunks_by_ids(self, chunk_ids: List[str]) -> List[FileChunk]:
        """Retrieve multiple file chunks by IDs.

        Public API for future file-scoped retrieval features.

        This helper is intended for callers that perform vector search over file
        chunks (e.g. via Chroma) and need to map resulting IDs back to full
        FileChunk records. It is currently not integrated into the core query
        flow but is safe for external and future internal use.

        Args:
            chunk_ids: List of stored chunk IDs to retrieve.

        Returns:
            List of FileChunk objects, ordered by file path and line number.
        """
        if not chunk_ids:
            return []
        
        self._connect(require_exists=True)
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(chunk_ids))
        query = f"""
            SELECT fc.file_path, fc.chunk_index, fc.content, fc.line_start, fc.line_end, f.revision_id
            FROM file_chunks fc
            JOIN file_contents f ON fc.file_path = f.file_path
            WHERE fc.id IN ({placeholders})
            ORDER BY fc.file_path, fc.line_start
        """
        cursor.execute(query, chunk_ids)
        rows = cursor.fetchall()
        
        return [FileChunk(
            file_path=row['file_path'],
            chunk_index=row['chunk_index'],
            content=row['content'],
            line_start=row['line_start'],
            line_end=row['line_end'],
            revision_id=row['revision_id']
        ) for row in rows]
    
    def delete_file_chunks_by_path(self, file_path: str) -> int:
        """Delete all chunks for a specific file path.
        
        Args:
            file_path: Path of the file whose chunks should be deleted
            
        Returns:
            Number of chunks deleted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete from file_chunks
            cursor.execute(
                "DELETE FROM file_chunks WHERE file_path = ?",
                (file_path,)
            )
            chunks_deleted = cursor.rowcount
            
            # Also delete from file_contents
            cursor.execute(
                "DELETE FROM file_contents WHERE file_path = ?",
                (file_path,)
            )
            
            conn.commit()
            return chunks_deleted
        
    def close(self):
        if self.conn:
            self.conn.close()