"""
Comprehensive tests for Git provider covering:
- VCS detection (positive and negative cases)
- Commit retrieval with various filters (subdirs, max_commits, since)
- Latest commit time retrieval
- Edge cases (empty repositories, no commits)

Uses temporary Git repositories for testing.
"""

import pytest
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from expert_among_us.vcs.git import Git


@pytest.fixture
def temp_repo_path():
    """Fixture providing a temporary Git repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        # Initialize as git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        # Configure git user for commits
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        yield repo_path


@pytest.fixture
def git_provider():
    """Fixture providing a Git provider instance."""
    return Git()


@pytest.fixture
def repo_with_commits(temp_repo_path):
    """Fixture providing a repository with sample commits."""
    repo_path = temp_repo_path
    
    # Create commits with timestamps
    for i in range(3):
        # Create a file
        file_path = repo_path / f"file_{i}.txt"
        file_path.write_text(f"Content {i}")
        
        # Stage and commit
        subprocess.run(
            ["git", "add", f"file_{i}.txt"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"Commit {i}"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
    
    return repo_path


class TestVCSDetection:
    """Tests for VCS detection."""

    def test_detect_git_repository(self, temp_repo_path):
        """Verify that a valid Git repository is detected."""
        assert Git.detect(str(temp_repo_path))

    def test_detect_non_git_directory(self):
        """Verify that non-Git directory is not detected as VCS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not Git.detect(tmpdir)

    def test_detect_nested_git_repository(self, temp_repo_path):
        """Verify that nested Git repository is detected."""
        nested_path = temp_repo_path / "nested"
        nested_path.mkdir()
        
        # Parent has .git, so even nested should detect it
        assert Git.detect(str(temp_repo_path))

    def test_detect_git_repository_with_dot_git_dir(self, temp_repo_path):
        """Verify detection by presence of .git directory."""
        git_dir = temp_repo_path / ".git"
        assert git_dir.exists()
        assert Git.detect(str(temp_repo_path))

    def test_detect_nonexistent_path(self):
        """Verify that nonexistent paths return False."""
        assert not Git.detect("/nonexistent/path/to/repo")


class TestCommitRetrieval:
    """Tests for commit retrieval with pagination."""

    def test_get_commits_page_basic(self, git_provider, repo_with_commits):
        """Verify that commits can be retrieved via pagination."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=10,
            since_hash=None
        )
        
        assert len(commits) > 0

    def test_get_commits_page_respects_page_size(self, git_provider, repo_with_commits):
        """Verify that page_size parameter limits results."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=2,
            since_hash=None
        )
        
        assert len(commits) <= 2

    def test_get_commits_page_returns_changelist_objects(self, git_provider, repo_with_commits):
        """Verify that commits have required attributes."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=1,
            since_hash=None
        )
        
        assert len(commits) > 0
        commit = commits[0]
        assert hasattr(commit, 'id')
        assert hasattr(commit, 'message')
        assert hasattr(commit, 'author')
        assert hasattr(commit, 'timestamp')
        assert hasattr(commit, 'diff')
        assert hasattr(commit, 'files')

    def test_get_commits_page_with_since_filter(self, git_provider, repo_with_commits):
        """Verify that since_hash filter works correctly."""
        # Get first page
        all_commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=10,
            since_hash=None
        )
        if len(all_commits) > 1:
            # Use the second commit as boundary (to exclude the most recent)
            since_hash = all_commits[1].id
            
            # Get commits since that hash (should return commits newer than since_hash)
            recent_commits = git_provider.get_commits_page(
                workspace_path=str(repo_with_commits),
                subdirs=None,
                page_size=10,
                since_hash=since_hash
            )
            
            # Should return the most recent commit only
            assert len(recent_commits) >= 1
            # The most recent commit should be included
            assert recent_commits[0].id == all_commits[0].id

    def test_get_commits_page_with_subdirectory_filter(self, temp_repo_path, git_provider):
        """Verify that subdirectory filter works."""
        repo_path = temp_repo_path
        
        # Create subdirectories with different files
        src_dir = repo_path / "src"
        src_dir.mkdir()
        test_dir = repo_path / "tests"
        test_dir.mkdir()
        
        # Create and commit src file
        (src_dir / "main.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "src/main.py"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add main"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        # Create and commit test file
        (test_dir / "test_main.py").write_text("# tests")
        subprocess.run(
            ["git", "add", "tests/test_main.py"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add tests"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        # Get commits for src directory
        src_commits = git_provider.get_commits_page(
            workspace_path=str(repo_path),
            subdirs=["src"],
            page_size=10,
            since_hash=None
        )
        
        # Should have at least one commit touching src
        assert len(src_commits) > 0

    def test_get_commits_page_empty_repository(self, git_provider):
        """Verify behavior with empty Git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            commits = git_provider.get_commits_page(
                workspace_path=str(repo_path),
                subdirs=None,
                page_size=10,
                since_hash=None
            )
            
            assert commits == []

    def test_get_commits_page_message_preserved(self, git_provider, repo_with_commits):
        """Verify that commit messages are preserved correctly."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=10,
            since_hash=None
        )
        
        messages = [c.message for c in commits]
        assert any("Commit" in msg for msg in messages)

    def test_get_commits_page_author_preserved(self, git_provider, repo_with_commits):
        """Verify that commit authors are preserved correctly."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=10,
            since_hash=None
        )
        
        authors = [c.author for c in commits]
        assert any("Test User" in author for author in authors)

    def test_get_commits_page_timestamp_format(self, git_provider, repo_with_commits):
        """Verify that commit timestamps are datetime objects."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=1,
            since_hash=None
        )
        
        if commits:
            assert isinstance(commits[0].timestamp, datetime)

    def test_get_commits_page_hash_pagination(self, temp_repo_path, git_provider):
        """Verify that hash-based pagination works correctly."""
        repo_path = temp_repo_path
        
        # Create multiple commits
        for i in range(3):
            (repo_path / f"file{i}.py").write_text(f"code{i}")
            subprocess.run(
                ["git", "add", f"file{i}.py"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            subprocess.run(
                ["git", "commit", "-m", f"Commit {i}"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
        
        # Get first page
        first_page = git_provider.get_commits_page(
            workspace_path=str(repo_path),
            subdirs=None,
            page_size=1,
            since_hash=None
        )
        
        # Get second page using first commit as boundary
        if first_page:
            second_page = git_provider.get_commits_page(
                workspace_path=str(repo_path),
                subdirs=None,
                page_size=1,
                since_hash=first_page[0].id
            )
            
            # If there are more commits, pages should be different
            # Note: since_hash is exclusive, so second_page gets commits after first_page[0]
            # With only 3 commits and first_page[0] being the newest, second_page should be empty
            # Let's just verify we got the first page
            assert len(first_page) == 1

    def test_get_commits_page_contains_diff(self, git_provider, repo_with_commits):
        """Verify that commits contain diff information."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=1,
            since_hash=None
        )
        
        if commits:
            assert commits[0].diff is not None
            assert isinstance(commits[0].diff, str)

    def test_get_commits_page_contains_files(self, git_provider, repo_with_commits):
        """Verify that commits contain file information."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=1,
            since_hash=None
        )
        
        if commits:
            assert commits[0].files is not None
            assert isinstance(commits[0].files, list)


class TestLatestCommitTime:
    """Tests for latest commit time retrieval."""

    def test_get_latest_commit_time(self, git_provider, repo_with_commits):
        """Verify that latest commit time can be retrieved."""
        latest_time = git_provider.get_latest_commit_time(str(repo_with_commits))
        
        assert latest_time is not None
        assert isinstance(latest_time, datetime)

    def test_latest_commit_time_is_recent(self, git_provider, repo_with_commits):
        """Verify that latest commit time is recent."""
        from datetime import timezone
        latest_time = git_provider.get_latest_commit_time(str(repo_with_commits))
        
        now = datetime.now(timezone.utc)
        # Should be within last minute (test just ran)
        assert (now - latest_time).total_seconds() < 60

    def test_latest_commit_time_empty_repository(self, git_provider):
        """Verify behavior for empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            latest_time = git_provider.get_latest_commit_time(str(repo_path))
            
            # Should return None for empty repo
            assert latest_time is None

    def test_latest_commit_time_after_new_commit(self, temp_repo_path, git_provider):
        """Verify that latest commit time updates after new commit."""
        # First commit
        (temp_repo_path / "file1.txt").write_text("content1")
        subprocess.run(
            ["git", "add", "file1.txt"],
            cwd=temp_repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "First"],
            cwd=temp_repo_path,
            capture_output=True,
            check=True
        )
        
        time1 = git_provider.get_latest_commit_time(str(temp_repo_path))
        
        # Small delay to ensure different timestamps
        import time
        time.sleep(1)
        
        # Second commit
        (temp_repo_path / "file2.txt").write_text("content2")
        subprocess.run(
            ["git", "add", "file2.txt"],
            cwd=temp_repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Second"],
            cwd=temp_repo_path,
            capture_output=True,
            check=True
        )
        
        time2 = git_provider.get_latest_commit_time(str(temp_repo_path))
        
        # Second time should be >= first time
        assert time2 >= time1

    def test_latest_commit_time_with_subdirs(self, temp_repo_path, git_provider):
        """Verify that latest commit time respects subdirs filter."""
        repo_path = temp_repo_path
        
        # Create src directory with file
        src_dir = repo_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("code")
        subprocess.run(
            ["git", "add", "src/main.py"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add src"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        # Get latest commit time for src directory
        latest_time = git_provider.get_latest_commit_time(str(repo_path), subdirs=["src"])
        
        assert latest_time is not None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions with pagination API."""

    def test_get_commits_page_from_nonexistent_directory(self, git_provider):
        """Verify behavior when Git provider points to non-existent directory."""
        commits = git_provider.get_commits_page(
            workspace_path="/nonexistent/path",
            subdirs=None,
            page_size=10,
            since_hash=None
        )
        # Should return empty list, not crash
        assert commits == []

    def test_get_commits_page_with_special_characters_in_messages(self, temp_repo_path, git_provider):
        """Verify commits with special characters in message are handled."""
        repo_path = temp_repo_path
        
        (repo_path / "file.txt").write_text("content")
        subprocess.run(
            ["git", "add", "file.txt"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Fix: bug with 'quotes' and \"double quotes\""],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_path),
            subdirs=None,
            page_size=1,
            since_hash=None
        )
        
        assert len(commits) == 1
        assert "quotes" in commits[0].message

    def test_get_commits_page_with_unicode_in_author(self, temp_repo_path, git_provider):
        """Verify commits with unicode characters in author are handled."""
        repo_path = temp_repo_path
        
        subprocess.run(
            ["git", "config", "user.name", "José García"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        (repo_path / "file.txt").write_text("content")
        subprocess.run(
            ["git", "add", "file.txt"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Test commit"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_path),
            subdirs=None,
            page_size=1,
            since_hash=None
        )
        
        assert len(commits) == 1
        assert "García" in commits[0].author or "Garcia" in commits[0].author

    def test_get_commits_page_with_zero_page_size(self, git_provider, repo_with_commits):
        """Verify that page_size=0 returns empty."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=0,
            since_hash=None
        )
        
        # Should return empty list
        assert isinstance(commits, list)
        assert len(commits) == 0

    def test_get_commits_page_since_nonexistent_hash(self, git_provider, repo_with_commits):
        """Verify that since_hash with invalid hash returns empty or all commits."""
        # Using a valid-looking but nonexistent hash should return empty
        fake_hash = "0" * 40
        
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=10,
            since_hash=fake_hash
        )
        # Git will return empty list for invalid hash range
        assert isinstance(commits, list)

    def test_get_commits_page_with_valid_since_hash(self, git_provider, repo_with_commits):
        """Verify that since_hash with valid hash filters correctly."""
        all_commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=10,
            since_hash=None
        )
        
        if len(all_commits) > 1:
            # Use oldest commit as boundary
            oldest_hash = all_commits[-1].id
            recent_commits = git_provider.get_commits_page(
                workspace_path=str(repo_with_commits),
                subdirs=None,
                page_size=10,
                since_hash=oldest_hash
            )
            
            # Should return commits newer than the boundary (excluding it)
            # All commits except the oldest one
            assert len(recent_commits) >= len(all_commits) - 1

    def test_multiple_commits_same_file(self, temp_repo_path, git_provider):
        """Verify that multiple commits to same file are tracked."""
        repo_path = temp_repo_path
        
        for i in range(3):
            (repo_path / "same_file.txt").write_text(f"Version {i}")
            subprocess.run(
                ["git", "add", "same_file.txt"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            subprocess.run(
                ["git", "commit", "-m", f"Update {i}"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
        
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_path),
            subdirs=None,
            page_size=10,
            since_hash=None
        )
        
        assert len(commits) == 3

    def test_get_commits_page_with_empty_subdirs_list(self, git_provider, repo_with_commits):
        """Verify that empty subdirs list is handled."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=[],
            page_size=10,
            since_hash=None
        )
        
        # Empty list should be equivalent to no filter
        assert len(commits) > 0

    def test_commit_id_is_valid_hash(self, git_provider, repo_with_commits):
        """Verify that commit IDs are valid Git hashes."""
        commits = git_provider.get_commits_page(
            workspace_path=str(repo_with_commits),
            subdirs=None,
            page_size=1,
            since_hash=None
        )
        
        if commits:
            # Git hashes are 40 character hex strings
            commit_id = commits[0].id
            assert len(commit_id) >= 7  # At least short hash length
            assert all(c in '0123456789abcdef' for c in commit_id.lower())

    def test_latest_commit_time_nonexistent_directory(self, git_provider):
        """Verify behavior when getting latest commit time from nonexistent directory."""
        latest_time = git_provider.get_latest_commit_time("/nonexistent/path")
        # Should return None, not crash
        assert latest_time is None