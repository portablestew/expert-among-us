"""Abstract base class for metadata database implementations.

This module defines the MetadataDB interface that all metadata storage backends
must implement. Metadata includes expert configurations, changelist data, and
cached generated prompts.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from expert_among_us.models.changelist import Changelist


class MetadataDB(ABC):
    """Abstract base class for metadata database operations.
    
    Implementations must provide storage for:
    - Expert configurations (name, workspace path, VCS type, etc.)
    - Changelist metadata (commits with messages, diffs, files, authors)
    - Generated prompt cache (for impostor mode)
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize database schema/tables.
        
        Creates all necessary tables, indexes, and constraints required for
        storing expert and changelist metadata. Should be idempotent.
        """
        pass

    @abstractmethod
    def create_expert(
        self,
        name: str,
        workspace_path: str,
        subdirs: list[str],
        vcs_type: str
    ) -> None:
        """Create a new expert entry.
        
        Args:
            name: Unique identifier for the expert
            workspace_path: Path to the workspace/repository
            subdirs: List of subdirectories to filter (empty for all)
            vcs_type: Version control system type ('git', 'p4', etc.)
        """
        pass

    @abstractmethod
    def get_expert(self, name: str) -> Optional[dict]:
        """Retrieve expert configuration.
        
        Args:
            name: Expert identifier
            
        Returns:
            Dictionary with keys: name, workspace_path, subdirs, vcs_type, 
            last_indexed_at. Returns None if expert doesn't exist.
        """
        pass

    @abstractmethod
    def update_expert_index_time(
        self,
        name: str,
        timestamp: datetime
    ) -> None:
        """Update expert's last indexed timestamp.
        
        Args:
            name: Expert identifier
            timestamp: Timestamp when indexing completed
        """
        pass

    @abstractmethod
    def insert_changelists(self, changelists: list[Changelist]) -> None:
        """Insert multiple changelists in a batch operation.
        
        Should use transactions for atomicity and performance. Duplicate
        changelist IDs should be handled gracefully (update or skip).
        
        Args:
            changelists: List of Changelist objects to insert
        """
        pass

    @abstractmethod
    def get_changelist(self, changelist_id: str) -> Optional[Changelist]:
        """Retrieve a single changelist by ID.
        
        Args:
            changelist_id: Unique changelist identifier (commit hash or CL number)
            
        Returns:
            Changelist object if found, None otherwise
        """
        pass

    @abstractmethod
    def get_changelists_by_ids(
        self,
        ids: list[str]
    ) -> list[Changelist]:
        """Retrieve multiple changelists by IDs.
        
        Args:
            ids: List of changelist IDs to retrieve
            
        Returns:
            List of Changelist objects (may be shorter than ids if some not found)
        """
        pass

    @abstractmethod
    def query_changelists_by_author(
        self,
        author: str
    ) -> list[str]:
        """Get changelist IDs by author filter.
        
        Args:
            author: Author name/email to filter by
            
        Returns:
            List of changelist IDs matching the author
        """
        pass

    @abstractmethod
    def query_changelists_by_files(
        self,
        file_paths: list[str]
    ) -> list[str]:
        """Get changelist IDs by file path filter using OR logic.
        
        Returns changelists that modified ANY of the specified files.
        
        Args:
            file_paths: List of file paths to search for
            
        Returns:
            List of changelist IDs that modified at least one of the files
        """
        pass

    @abstractmethod
    def get_generated_prompt(self, query_id: str) -> Optional[str]:
        """Retrieve cached generated prompt.
        
        Used in impostor mode to avoid regenerating prompts for the same
        changelist.
        
        Args:
            query_id: Identifier for the cached prompt (typically changelist_id)
            
        Returns:
            Generated prompt string if cached, None otherwise
        """
        pass

    @abstractmethod
    def cache_generated_prompt(
        self,
        query_id: str,
        prompt: str
    ) -> None:
        """Store generated prompt in cache.
        
        Args:
            query_id: Identifier for the cached prompt (typically changelist_id)
            prompt: Generated prompt text to cache
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection and release resources.
        
        Should be called when done with the database to ensure proper cleanup.
        """
        pass