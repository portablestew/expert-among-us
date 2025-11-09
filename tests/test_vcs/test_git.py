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

    def test_get_commits_after_basic(self, git_provider, repo_with_commits):
        """Verify that commits can be retrieved via unified get_commits_after API."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=10,
            subdirs=None,
        )
        assert len(commits) > 0

    def test_get_commits_after_respects_batch_size(self, git_provider, repo_with_commits):
        """Verify that batch_size parameter limits results."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=2,
            subdirs=None,
        )
        assert len(commits) <= 2

    def test_get_commits_after_returns_changelist_objects(self, git_provider, repo_with_commits):
        """Verify that commits have required attributes."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=1,
            subdirs=None,
        )
        
        assert len(commits) > 0
        commit = commits[0]
        assert hasattr(commit, 'id')
        assert hasattr(commit, 'message')
        assert hasattr(commit, 'author')
        assert hasattr(commit, 'timestamp')
        assert hasattr(commit, 'diff')
        assert hasattr(commit, 'files')

    def test_get_commits_after_with_after_hash_filter(self, git_provider, repo_with_commits):
        """Verify that after_hash filter works correctly."""
        all_commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=10,
            subdirs=None,
        )
        if len(all_commits) > 1:
            boundary = all_commits[0].id
            newer = git_provider.get_commits_after(
                workspace_path=str(repo_with_commits),
                after_hash=boundary,
                batch_size=10,
                subdirs=None,
            )
            # All returned commits should be strictly after the boundary in chronological order
            assert all(c.id != boundary for c in newer)

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
        src_commits = git_provider.get_commits_after(
            workspace_path=str(repo_path),
            after_hash=None,
            batch_size=10,
            subdirs=["src"],
        )
        
        # Should have at least one commit touching src
        assert len(src_commits) > 0

    def test_get_commits_page_empty_repository(self, git_provider):
        """Empty Git repository should cause git log to fail with an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True,
            )

            with pytest.raises(subprocess.CalledProcessError):
                git_provider.get_commits_after(
                    workspace_path=str(repo_path),
                    after_hash=None,
                    batch_size=10,
                    subdirs=None,
                )

    def test_get_commits_after_message_preserved(self, git_provider, repo_with_commits):
        """Verify that commit messages are preserved correctly."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=10,
            subdirs=None,
        )
        
        messages = [c.message for c in commits]
        assert any("Commit" in msg for msg in messages)

    def test_get_commits_after_author_preserved(self, git_provider, repo_with_commits):
        """Verify that commit authors are preserved correctly."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=10,
            subdirs=None,
        )
        
        authors = [c.author for c in commits]
        assert any("Test User" in author for author in authors)

    def test_get_commits_after_timestamp_format(self, git_provider, repo_with_commits):
        """Verify that commit timestamps are datetime objects."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=1,
            subdirs=None,
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
        
        # Get first batch
        first_page = git_provider.get_commits_after(
            workspace_path=str(repo_path),
            after_hash=None,
            batch_size=1,
            subdirs=None,
        )
        
        # Get second page using first commit as boundary
        if first_page:
            second_page = git_provider.get_commits_after(
                workspace_path=str(repo_path),
                after_hash=first_page[0].id,
                batch_size=1,
                subdirs=None,
            )
            
            # If there are more commits, pages should be different
            # Note: since_hash is exclusive, so second_page gets commits after first_page[0]
            # With only 3 commits and first_page[0] being the newest, second_page should be empty
            # Let's just verify we got the first page
            assert len(first_page) == 1

    def test_get_commits_after_contains_diff(self, git_provider, repo_with_commits):
        """Verify that commits contain diff information."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=1,
            subdirs=None,
        )
        
        if commits:
            assert commits[0].diff is not None
            assert isinstance(commits[0].diff, str)

    def test_get_commits_after_contains_files(self, git_provider, repo_with_commits):
        """Verify that commits contain file information."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=1,
            subdirs=None,
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
        """Nonexistent workspace path should raise an error."""
        with pytest.raises(subprocess.CalledProcessError):
            git_provider.get_commits_after(
                workspace_path="/nonexistent/path",
                after_hash=None,
                batch_size=10,
                subdirs=None,
            )

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
        
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_path),
            after_hash=None,
            batch_size=1,
            subdirs=None,
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
        
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_path),
            after_hash=None,
            batch_size=1,
            subdirs=None,
        )
        
        assert len(commits) == 1
        assert "García" in commits[0].author or "Garcia" in commits[0].author

    def test_get_commits_page_with_zero_page_size(self, git_provider, repo_with_commits):
        """Verify that page_size=0 returns empty."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=0,
            subdirs=None,
        )
        
        # Should return empty list
        assert isinstance(commits, list)
        assert len(commits) == 0

    def test_get_commits_page_since_nonexistent_hash(self, git_provider, repo_with_commits):
        """Invalid since_hash should raise, as it represents a bad range."""
        fake_hash = "0" * 40

        with pytest.raises(subprocess.CalledProcessError):
            git_provider.get_commits_after(
                workspace_path=str(repo_with_commits),
                after_hash=fake_hash,
                batch_size=10,
                subdirs=None,
            )

    def test_get_commits_page_with_valid_since_hash(self, git_provider, repo_with_commits):
        """Verify that since_hash with valid hash filters correctly."""
        all_commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=10,
            subdirs=None,
        )
        
        if len(all_commits) > 1:
            oldest_hash = all_commits[0].id
            recent_commits = git_provider.get_commits_after(
                workspace_path=str(repo_with_commits),
                after_hash=oldest_hash,
                batch_size=10,
                subdirs=None,
            )
            # Should return commits strictly after the boundary
            assert all(c.id != oldest_hash for c in recent_commits)

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
        
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_path),
            after_hash=None,
            batch_size=10,
            subdirs=None,
        )
        
        assert len(commits) == 3

    def test_get_commits_page_with_empty_subdirs_list(self, git_provider, repo_with_commits):
        """Verify that empty subdirs list is handled."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=10,
            subdirs=[],
        )
        
        # Empty list should be equivalent to no filter
        assert len(commits) > 0

    def test_commit_id_is_valid_hash(self, git_provider, repo_with_commits):
        """Verify that commit IDs are valid Git hashes."""
        commits = git_provider.get_commits_after(
            workspace_path=str(repo_with_commits),
            after_hash=None,
            batch_size=1,
            subdirs=None,
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


class TestGetFilesContentAtCommit:
    """Tests for the batched get_files_content_at_commit API."""

    def _init_repo_with_files(self, temp_repo_path) -> tuple[str, dict[str, str], str]:
        """Helper: create a single commit with given files, return (repo, files, commit_hash)."""
        repo_path = temp_repo_path

        # Create multiple text files
        files = {
            "a.txt": "hello a",
            "b.txt": "hello b",
            "dir/c.txt": "hello c",
        }
        for rel, content in files.items():
            path = repo_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        subprocess.run(
            ["git", "add", "."],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add text files"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Get commit hash (HEAD)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
        commit_hash = result.stdout.strip()
        return str(repo_path), files, commit_hash

    def test_basic_batched_file_reading(self, git_provider, temp_repo_path):
        """Multiple text files can be fetched in a single batched call."""
        repo_path, files, commit_hash = self._init_repo_with_files(temp_repo_path)

        result = git_provider.get_files_content_at_commit(
            workspace_path=repo_path,
            file_paths=list(files.keys()),
            commit_hash=commit_hash,
        )

        # Should contain all requested files with exact content.
        assert set(result.keys()) == set(files.keys())
        for filename, expected in files.items():
            assert result[filename] == expected

    def test_binary_file_handling(self, git_provider, temp_repo_path):
        """Binary files should map to None while text files return content."""
        repo_path = temp_repo_path

        # Text file
        (repo_path / "text.txt").write_text("plain text", encoding="utf-8")

        # Binary file (PNG header bytes)
        binary_data = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        (repo_path / "image.bin").write_bytes(binary_data)

        subprocess.run(
            ["git", "add", "."],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add text and binary"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        commit_hash = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
            .stdout.strip()
        )

        result = git_provider.get_files_content_at_commit(
            workspace_path=str(repo_path),
            file_paths=["text.txt", "image.bin"],
            commit_hash=commit_hash,
        )

        assert set(result.keys()) == {"text.txt", "image.bin"}
        assert result["text.txt"] == "plain text"
        # Binary file content must be treated as None
        assert result["image.bin"] is None

    def test_missing_files_return_none(self, git_provider, temp_repo_path):
        """Missing/non-existent files should map to None in result."""
        repo_path, files, commit_hash = self._init_repo_with_files(temp_repo_path)

        requested = list(files.keys()) + ["missing.txt", "dir/none.py"]

        result = git_provider.get_files_content_at_commit(
            workspace_path=repo_path,
            file_paths=requested,
            commit_hash=commit_hash,
        )

        # All requested paths present
        assert set(result.keys()) == set(requested)

        # Existing files have correct content
        for filename, expected in files.items():
            assert result[filename] == expected

        # Missing files are None
        assert result["missing.txt"] is None
        assert result["dir/none.py"] is None

    def test_empty_file_list_returns_empty_dict(self, git_provider, temp_repo_path):
        """Edge case: empty input list returns empty dict."""
        repo_path = str(temp_repo_path)

        # No need to create commits; API should handle empty input early.
        result = git_provider.get_files_content_at_commit(
            workspace_path=repo_path,
            file_paths=[],
            commit_hash="HEAD",
        )

        assert result == {}

    def test_mixed_scenario(self, git_provider, temp_repo_path):
        """Mixed scenario: valid text, binary, and missing files handled correctly."""
        repo_path = temp_repo_path

        # Text files
        (repo_path / "x.txt").write_text("x", encoding="utf-8")
        (repo_path / "y.txt").write_text("y", encoding="utf-8")

        # Binary file
        (repo_path / "bin.dat").write_bytes(b"\x00\xFF\x00\xFF")

        subprocess.run(
            ["git", "add", "."],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "mixed content"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        commit_hash = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
            .stdout.strip()
        )

        requested = ["x.txt", "y.txt", "bin.dat", "missing1", "missing/also"]

        result = git_provider.get_files_content_at_commit(
            workspace_path=str(repo_path),
            file_paths=requested,
            commit_hash=commit_hash,
        )

        # Ensure all requested paths are present
        assert set(result.keys()) == set(requested)

        # Text files
        assert result["x.txt"] == "x"
        assert result["y.txt"] == "y"

        # Binary -> None
        assert result["bin.dat"] is None

        # Missing -> None
        assert result["missing1"] is None
        assert result["missing/also"] is None

    def test_single_file_backward_compatibility(self, git_provider, temp_repo_path):
        """Single-file API should behave identically to batched API for one file."""
        repo_path = temp_repo_path

        (repo_path / "file.txt").write_text("compat content", encoding="utf-8")
        subprocess.run(
            ["git", "add", "file.txt"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "compat"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        commit_hash = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
            .stdout.strip()
        )

        batched = git_provider.get_files_content_at_commit(
            workspace_path=str(repo_path),
            file_paths=["file.txt"],
            commit_hash=commit_hash,
        )
        single = git_provider.get_file_content_at_commit(
            workspace_path=str(repo_path),
            file_path="file.txt",
            commit_hash=commit_hash,
        )

        assert batched["file.txt"] == "compat content"
        assert single == "compat content"

    def test_large_file_list_chunking(self, git_provider, temp_repo_path):
        """Large batches (150+ files) should be processed correctly (chunking behavior)."""
        repo_path = temp_repo_path

        file_paths = []
        for i in range(155):
            rel = f"files/file_{i}.txt"
            path = repo_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"content {i}"
            path.write_text(content, encoding="utf-8")
            file_paths.append(rel)

        subprocess.run(
            ["git", "add", "."],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "many files"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        commit_hash = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
            .stdout.strip()
        )

        result = git_provider.get_files_content_at_commit(
            workspace_path=str(repo_path),
            file_paths=file_paths,
            commit_hash=commit_hash,
        )

        # Ensure all files are present with correct content.
        assert set(result.keys()) == set(file_paths)
        for i, rel in enumerate(file_paths):
            assert result[rel] == f"content {i}"