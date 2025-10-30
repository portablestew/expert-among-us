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
    
    def get_commits(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
        max_commits: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> list[Changelist]:
        """Retrieve commits from Git repository."""
        # Handle max_commits=0 case
        if max_commits is not None and max_commits == 0:
            return []
        
        cmd = ["git", "-C", workspace_path, "log", "--pretty=format:%H|%an|%ae|%at|%s"]
        
        if max_commits:
            cmd.extend(["-n", str(max_commits)])
        
        if since:
            # Use --after instead of --since to exclude the boundary commit
            # Add 1 second to ensure we don't re-process the last commit
            since_after = since + timedelta(seconds=1)
            cmd.append(f"--after={since_after.isoformat()}")
        
        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            return []
        
        changelists = []
        for line in result.stdout.strip().split('\n'):
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
            diff_result = subprocess.run(diff_cmd, capture_output=True, text=True, encoding='utf-8')
            diff = diff_result.stdout if diff_result.returncode == 0 else ""
            
            # Filter binary content from diff
            diff, binary_files = filter_binary_from_diff(diff)
            
            # Get list of changed files
            files_cmd = ["git", "-C", workspace_path, "show", "--name-status", "--format=", commit_hash]
            files_result = subprocess.run(files_cmd, capture_output=True, text=True, encoding='utf-8')
            
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
    
    def get_commits_before(
        self,
        workspace_path: str,
        before: datetime,
        subdirs: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[Changelist]:
        """Retrieve commits older than the specified timestamp in chronological order (oldest first).
        
        Args:
            workspace_path: Path to the repository
            before: Fetch commits before this timestamp
            subdirs: Optional list of subdirectories to filter
            limit: Optional maximum number of commits to fetch
            
        Returns:
            List of Changelist objects in chronological order (oldest first)
        """
        # Handle limit=0 case
        if limit is not None and limit == 0:
            return []
        
        cmd = ["git", "-C", workspace_path, "log", "--reverse", "--pretty=format:%H|%an|%ae|%at|%s"]
        
        if limit:
            cmd.extend(["-n", str(limit)])
        
        # Use --before to exclude the boundary commit
        # Subtract 1 second to ensure we don't re-process the first commit
        before_minus = before - timedelta(seconds=1)
        cmd.append(f"--before={before_minus.isoformat()}")
        
        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            return []
        
        changelists = []
        for line in result.stdout.strip().split('\n'):
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
            diff_result = subprocess.run(diff_cmd, capture_output=True, text=True, encoding='utf-8')
            diff = diff_result.stdout if diff_result.returncode == 0 else ""
            
            # Filter binary content from diff
            diff, binary_files = filter_binary_from_diff(diff)
            
            # Get list of changed files
            files_cmd = ["git", "-C", workspace_path, "show", "--name-status", "--format=", commit_hash]
            files_result = subprocess.run(files_cmd, capture_output=True, text=True, encoding='utf-8')
            
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
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        timestamp_str = result.stdout.strip()
        # Create timezone-aware datetime in UTC
        return datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)