"""Base abstract class for version control system providers.

This module defines the VCSProvider abstract base class that all VCS
implementations must inherit from. It provides a consistent interface
for interacting with different version control systems (Git, Perforce, etc.).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

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
    def get_commits_page(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]],
        page_size: int,
        skip: int,
        since: Optional[datetime],
    ) -> list[Changelist]:
        """Retrieve a page of commits from the VCS.
        
        Args:
            workspace_path: Path to the workspace/repository
            subdirs: Optional list of subdirectories to filter commits by
            page_size: Number of commits to retrieve in this page
            skip: Number of commits to skip (for pagination)
            since: Optional datetime to filter commits after this time
            
        Returns:
            List of Changelist objects representing the commits (up to page_size)
        """
        pass

    @abstractmethod
    def get_commits_page_before(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]],
        page_size: int,
        skip: int,
        before: datetime,
    ) -> list[Changelist]:
        """Retrieve a page of commits older than the specified timestamp.
        
        Args:
            workspace_path: Path to the workspace/repository
            subdirs: Optional list of subdirectories to filter commits by
            page_size: Number of commits to retrieve in this page
            skip: Number of commits to skip (for pagination)
            before: Fetch commits before this timestamp
            
        Returns:
            List of Changelist objects in chronological order (oldest first, up to page_size)
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