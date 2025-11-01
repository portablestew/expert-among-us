import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone, timedelta
from expert_among_us.vcs.base import VCSProvider
from expert_among_us.models.changelist import Changelist
from expert_among_us.utils.truncate import filter_binary_from_diff

class Git(VCSProvider):
    """Git VCS provider implementation."""
    
    @staticmethod
    def detect(workspace_path: str) -> bool:
        """Detect if this workspace uses Git."""
        git_dir = Path(workspace_path) / ".git"
        return git_dir.exists() and git_dir.is_dir()
    
    def get_commits_page(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]],
        page_size: int,
        since_hash: Optional[str] = None,
        debug: bool = False,
    ) -> list[Changelist]:
        """Retrieve a page of commits from Git repository using topological traversal.
        
        Args:
            workspace_path: Path to the git repository
            subdirs: Optional subdirectories to filter
            page_size: Number of commits to fetch
            since_hash: Optional commit hash to use as exclusive boundary
            debug: Enable debug logging
        
        Returns:
            List of Changelist objects (excludes merge commits)
        """
        if page_size == 0:
            return []
        
        cmd = ["git", "-C", workspace_path, "log", "--no-merges", "--pretty=format:%H|%an|%ae|%at|%s"]
        
        # Limit number of commits
        cmd.extend(["-n", str(page_size)])
        
        # Use commit hash range if provided
        if since_hash:
            # Use hash..HEAD syntax to get commits after since_hash (exclusive)
            # Topological ordering respects parent-child relationships
            cmd.append(f"{since_hash}..HEAD")
        
        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)
        
        if debug:
            from expert_among_us.utils.progress import log_info
            log_info(f"[DEBUG] Git command: {' '.join(cmd)}")
        
        # Use errors='replace' to handle non-UTF-8 characters gracefully
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            return []
        
        return self._parse_commits(workspace_path, result.stdout)
    
    def get_commits_page_before(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]],
        page_size: int,
        before_hash: Optional[str] = None,
        debug: bool = False,
    ) -> list[Changelist]:
        """Retrieve a page of commits older than the specified boundary using topological traversal.
        
        Args:
            workspace_path: Path to the git repository
            subdirs: Optional subdirectories to filter
            page_size: Number of commits to fetch
            before_hash: Optional commit hash to use as exclusive boundary
            debug: Enable debug logging
        
        Returns:
            List of Changelist objects in topological order (oldest first, excludes merge commits)
        """
        if page_size == 0:
            return []
        
        cmd = ["git", "-C", workspace_path, "log", "--no-merges", "--reverse", "--pretty=format:%H|%an|%ae|%at|%s"]
        
        # Limit number of commits
        cmd.extend(["-n", str(page_size)])
        
        # Use commit hash range if provided
        if before_hash:
            # Use hash^ to start from the parent of before_hash
            # This gives us commits older than before_hash
            # Topological ordering respects parent-child relationships
            cmd.append(f"{before_hash}^")
        # else: No boundary - fetch from repository beginning (for initial indexing)
        
        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)
        
        if debug:
            from expert_among_us.utils.progress import log_info
            log_info(f"[DEBUG] Git command: {' '.join(cmd)}")
        
        # Use errors='replace' to handle non-UTF-8 characters gracefully
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            return []
        
        return self._parse_commits(workspace_path, result.stdout)
    
    def _parse_commits(self, workspace_path: str, stdout: str) -> list[Changelist]:
        """Parse commit output and return list of Changelist objects."""
        changelists = []
        for line in stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('|', 4)
            if len(parts) < 5:
                continue
            
            commit_hash, author_name, author_email, timestamp_str, message = parts
            # Create timezone-aware datetime in UTC
            timestamp = datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)
            
            # Get diff and files for this commit
            diff_cmd = ["git", "-C", workspace_path, "show", "--format=", commit_hash]
            # Use errors='replace' to handle non-UTF-8 characters in diffs
            diff_result = subprocess.run(diff_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            diff = diff_result.stdout if diff_result.returncode == 0 else ""
            
            # Filter binary content from diff
            diff, binary_files = filter_binary_from_diff(diff)
            
            # Skip commits with empty diffs (merge commits are already excluded by --no-merges)
            if not diff or len(diff.strip()) == 0:
                continue
            
            # Get list of changed files
            files_cmd = ["git", "-C", workspace_path, "show", "--name-status", "--format=", commit_hash]
            # Use errors='replace' to handle non-UTF-8 characters in file names
            files_result = subprocess.run(files_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            files = []
            if files_result.returncode == 0:
                for file_line in files_result.stdout.strip().split('\n'):
                    if not file_line:
                        continue
                    parts = file_line.split('\t', 1)
                    if len(parts) == 2:
                        status, filepath = parts
                        files.append(filepath)
            
            changelist = Changelist(
                id=commit_hash,
                expert_name="",  # Will be set by the caller
                timestamp=timestamp,
                author=f"{author_name} <{author_email}>",
                message=message,
                diff=diff,
                files=files if files else [""]  # Ensure at least one file to satisfy validation
            )
            changelists.append(changelist)
        
        return changelists
    
    def get_latest_commit_time(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent commit."""
        cmd = ["git", "-C", workspace_path, "log", "-1", "--format=%at"]
        
        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)
        
        # Use errors='replace' to handle non-UTF-8 characters gracefully
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        timestamp_str = result.stdout.strip()
        # Create timezone-aware datetime in UTC
        return datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)