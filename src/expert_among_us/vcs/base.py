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
    def get_commits(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
        max_commits: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> list[Changelist]:
        """Retrieve commits from the VCS, optionally filtered.
        
        Args:
            workspace_path: Path to the workspace/repository
            subdirs: Optional list of subdirectories to filter commits by.
                    Only commits that touched files in these directories will be included.
            max_commits: Optional maximum number of commits to retrieve
            since: Optional datetime to filter commits after this time
            
        Returns:
            List of Changelist objects representing the commits
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