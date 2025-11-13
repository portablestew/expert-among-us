"""Base abstract class for version control system providers.

This module defines the VCSProvider abstract base class that all VCS
implementations must inherit from. It provides a consistent interface
for interacting with different version control systems (Git, Perforce, etc.).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional

from expert_among_us.models.changelist import Changelist


class VCSProvider(ABC):
    """Abstract base class for version control system providers.
    
    All VCS implementations (Git, Perforce, etc.) must inherit from this
    class and implement all abstract methods to provide a consistent
    interface for retrieving commit/changelist history.
    """

    @staticmethod
    @abstractmethod
    def detect(workspace_path: str) -> bool:
        """Detect if this VCS is used in the given workspace.
        
        Args:
            workspace_path: Path to the workspace directory to check
            
        Returns:
            True if this VCS is detected in the workspace, False otherwise
        """
        pass

    @abstractmethod
    def get_commits_after(
        self,
        workspace_path: str,
        after_hash: str | None,
        batch_size: int,
        subdirs: Optional[list[str]] = None,
    ) -> list[Changelist]:
        """Get commits after a specific hash in chronological order (oldest → newest).

        Args:
            workspace_path: Path to the workspace/repository
            after_hash: Get commits after this hash (None = from beginning)
            batch_size: Maximum number of commits to return
            subdirs: Optional list of subdirectories to filter commits by

        Returns:
            List of Changelist objects in chronological order (oldest → newest)
        """
        pass

    @abstractmethod
    def get_tracked_files_at_commit(
        self,
        workspace_path: str,
        commit_hash: str,
        subdirs: Optional[list[str]] = None,
    ) -> list[str]:
        """Get list of tracked files at a specific commit.

        Args:
            workspace_path: Path to the workspace/repository
            commit_hash: Commit hash to inspect
            subdirs: Optional list of subdirectories to filter by

        Returns:
            List of file paths (relative to workspace root) tracked at the commit
        """
        pass

    @abstractmethod
    def get_file_content_at_commit(
        self,
        workspace_path: str,
        file_path: str,
        commit_hash: str,
    ) -> Optional[str]:
        """Get file content at a specific commit.

        Args:
            workspace_path: Path to the workspace/repository
            file_path: Relative path to file
            commit_hash: Commit hash to read from

        Returns:
            File content as text, or None if missing or not readable
        """
        pass

    @abstractmethod
    def get_files_content_at_commit(
        self,
        workspace_path: str,
        file_paths: list[str],
        commit_hash: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, Optional[str]]:
        """Get content for multiple files at a specific commit (batched operation).

        Args:
            workspace_path: Path to the workspace/repository
            file_paths: List of relative file paths to fetch
            commit_hash: Commit hash to read from
            progress_callback: Optional callback(current, total) called after each batch.
                             Receives the number of files processed so far and total files.

        Returns:
            Dictionary mapping file_path -> content (or None if missing/binary)

        Notes:
            - Implementations should fetch all files in a single operation when possible
            - Binary files should return None in the result dict
            - Missing files should return None in the result dict
            - The returned dict should contain an entry for every input file_path
            - If progress_callback is provided, it will be called after each batch with (current, total)
        """
        pass

    @abstractmethod
    def get_latest_commit_time(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent commit.
        
        Args:
            workspace_path: Path to the workspace/repository
            subdirs: Optional list of subdirectories to filter commits by.
                    Only commits that touched files in these directories will be considered.
            
        Returns:
            Datetime of the most recent commit, or None if no commits found
        """
        pass

    @abstractmethod
    def get_total_commit_count(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> int:
        """Return the total number of commits to consider for indexing.

        Implementations should:
        - Count only commits that match the same semantics as get_commits_after()
          (e.g. exclude merges, respect subdir filters if supported).
        - Return 0 if the repository has no matching commits or cannot be read.

        Args:
            workspace_path: Path to the workspace / repository root.
            subdirs: Optional list of subdirectories to filter by.

        Returns:
            Integer count of commits.
        """
        pass