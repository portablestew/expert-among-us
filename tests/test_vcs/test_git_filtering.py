"""Tests for Git subdirectory-based file filtering.

These tests verify that file and diff filtering works correctly when subdirs is specified,
ensuring only files within those subdirectories are included in diffs and file lists.
"""

import pytest
from unittest.mock import Mock, patch, call
from pathlib import Path

from expert_among_us.vcs.git import Git
from expert_among_us.config.settings import Settings


@pytest.fixture
def settings():
    """Fixture providing a Settings instance for tests."""
    return Settings()


@pytest.fixture
def git_provider(settings):
    """Fixture providing a Git provider instance."""
    return Git(settings)


@pytest.fixture
def tmp_git_repo(tmp_path):
    """Fixture providing a temporary git repository."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    return tmp_path


class TestGitSubdirFiltering:
    """Tests for subdirectory filtering in Git diffs and file lists."""
    
    def test_subdirs_filters_diffs(self, git_provider, tmp_git_repo):
        """Verify subdirs parameter filters diff output to only include specified directories."""
        with patch('subprocess.run') as mock_run:
            # Mock git log to get commit metadata
            meta_output = "abc123|Author Name|author@example.com|1705334400|Commit message"
            
            # Mock git show for diffs - includes files from multiple directories
            diff_output = """commit abc123

diff --git a/localization/en.json b/localization/en.json
--- a/localization/en.json
+++ b/localization/en.json
@@ -1 +1 @@
-{"greeting": "Hello"}
+{"greeting": "Hi"}
diff --git a/ui/styles.css b/ui/styles.css
--- a/ui/styles.css
+++ b/ui/styles.css
@@ -1 +1 @@
-.button { color: red; }
+.button { color: blue; }"""
            
            # Mock git show for files
            files_output = """commit abc123

M\tlocalization/en.json
M\tui/styles.css"""
            
            mock_run.side_effect = [
                Mock(returncode=0, stdout=meta_output),  # git log for metadata
                Mock(returncode=0, stdout=diff_output),   # git show for diffs
                Mock(returncode=0, stdout=files_output),  # git show for files
            ]
            
            # Fetch commits with subdirs=["localization"]
            changelists = git_provider._fetch_single_commit_batch(
                workspace_path=str(tmp_git_repo),
                hashes=["abc123"],
                subdirs=["localization"],
                embed_diffs=True,
            )
            
            # Verify git show commands included subdirs filter
            calls = mock_run.call_args_list
            
            # Check diff command (2nd call)
            diff_cmd = calls[1][0][0]
            assert "show" in diff_cmd
            assert "--" in diff_cmd
            assert "localization" in diff_cmd
            
            # Check files command (3rd call)
            files_cmd = calls[2][0][0]
            assert "show" in files_cmd
            assert "--" in files_cmd
            assert "localization" in files_cmd
    
    def test_subdirs_none_no_filtering(self, git_provider, tmp_git_repo):
        """Verify subdirs=None does not add filtering to git commands."""
        with patch('subprocess.run') as mock_run:
            meta_output = "abc123|Author|author@example.com|1705334400|Message"
            diff_output = "commit abc123\n\ndiff --git a/file.txt b/file.txt"
            files_output = "commit abc123\n\nM\tfile.txt"
            
            mock_run.side_effect = [
                Mock(returncode=0, stdout=meta_output),
                Mock(returncode=0, stdout=diff_output),
                Mock(returncode=0, stdout=files_output),
            ]
            
            git_provider._fetch_single_commit_batch(
                workspace_path=str(tmp_git_repo),
                hashes=["abc123"],
                subdirs=None,  # No subdirs filtering
                embed_diffs=True,
            )
            
            calls = mock_run.call_args_list
            
            # Verify NO subdirs filter in diff command
            diff_cmd = calls[1][0][0]
            assert "--" not in diff_cmd or "localization" not in diff_cmd
            
            # Verify NO subdirs filter in files command
            files_cmd = calls[2][0][0]
            assert "--" not in files_cmd or "localization" not in files_cmd
    
    def test_subdirs_filters_multiple_directories(self, git_provider, tmp_git_repo):
        """Verify multiple subdirs are correctly passed to git commands."""
        with patch('subprocess.run') as mock_run:
            meta_output = "abc123|Author|author@example.com|1705334400|Message"
            diff_output = "commit abc123\n\ndiff --git a/game/player.cpp b/game/player.cpp"
            files_output = "commit abc123\n\nM\tgame/player.cpp"
            
            mock_run.side_effect = [
                Mock(returncode=0, stdout=meta_output),
                Mock(returncode=0, stdout=diff_output),
                Mock(returncode=0, stdout=files_output),
            ]
            
            git_provider._fetch_single_commit_batch(
                workspace_path=str(tmp_git_repo),
                hashes=["abc123"],
                subdirs=["game", "localization"],
                embed_diffs=True,
            )
            
            calls = mock_run.call_args_list
            
            # Check both subdirs are in diff command
            diff_cmd = calls[1][0][0]
            assert "--" in diff_cmd
            assert "game" in diff_cmd
            assert "localization" in diff_cmd
            
            # Check both subdirs are in files command
            files_cmd = calls[2][0][0]
            assert "--" in files_cmd
            assert "game" in files_cmd
            assert "localization" in files_cmd
    
    def test_filtering_prevents_cross_directory_pollution(self, git_provider, tmp_git_repo):
        """Verify filtering prevents including files from outside specified subdirs."""
        with patch('subprocess.run') as mock_run:
            # Commit that touches localization and UI files
            meta_output = "abc123|Developer|dev@example.com|1705334400|Update translations and UI"
            
            # Git show output with multiple files (but filtered by subdirs)
            # When subdirs=["localization"], git show should only return localization diffs
            diff_output = """commit abc123

diff --git a/localization/en.json b/localization/en.json
--- a/localization/en.json
+++ b/localization/en.json
@@ -1 +1 @@
-{"key": "value"}
+{"key": "new_value"}"""
            
            files_output = """commit abc123

M\tlocalization/en.json"""
            
            mock_run.side_effect = [
                Mock(returncode=0, stdout=meta_output),
                Mock(returncode=0, stdout=diff_output),
                Mock(returncode=0, stdout=files_output),
            ]
            
            changelists = git_provider._fetch_single_commit_batch(
                workspace_path=str(tmp_git_repo),
                hashes=["abc123"],
                subdirs=["localization"],
                embed_diffs=True,
            )
            
            assert len(changelists) == 1
            cl = changelists[0]
            
            # CRITICAL: Only localization file should be in files list
            assert len(cl.files) == 1
            assert "localization/en.json" in cl.files[0]
            
            # CRITICAL: Only localization diff should be present
            assert "new_value" in cl.diff
            # UI file should NOT be in diff (git filtered it out)
            assert "styles.css" not in cl.diff
            assert "button" not in cl.diff
    
    def test_embed_diffs_false_skips_filtering(self, git_provider, tmp_git_repo):
        """Verify when embed_diffs=False, diff command is not called."""
        with patch('subprocess.run') as mock_run:
            meta_output = "abc123|Author|author@example.com|1705334400|Message"
            files_output = "commit abc123\n\nM\tlocalization/en.json"
            
            mock_run.side_effect = [
                Mock(returncode=0, stdout=meta_output),
                # No diff command call
                Mock(returncode=0, stdout=files_output),
            ]
            
            changelists = git_provider._fetch_single_commit_batch(
                workspace_path=str(tmp_git_repo),
                hashes=["abc123"],
                subdirs=["localization"],
                embed_diffs=False,  # Diffs disabled
            )
            
            # Should only have 2 calls: metadata and files (no diffs)
            assert mock_run.call_count == 2
            
            # Verify changelist has no diff
            assert len(changelists) == 1
            assert changelists[0].diff == ""


class TestGitPerforceParityBehavior:
    """Tests ensuring Git and Perforce have equivalent filtering behavior."""
    
    def test_git_matches_perforce_filtering_logic(self, git_provider, tmp_git_repo):
        """Verify Git filtering matches Perforce for consistency."""
        with patch('subprocess.run') as mock_run:
            # Same scenario as Perforce test: commit with mixed files
            meta_output = "abc123|User|user@example.com|1705334400|Mixed commit"
            
            # Only localization files should be in filtered output
            diff_output = """commit abc123

diff --git a/Assets/Localization/en-US/strings.xml b/Assets/Localization/en-US/strings.xml
--- a/Assets/Localization/en-US/strings.xml
+++ b/Assets/Localization/en-US/strings.xml
@@ -1 +1 @@
-<string>Old</string>
+<string>New</string>"""
            
            files_output = """commit abc123

M\tAssets/Localization/en-US/strings.xml"""
            
            mock_run.side_effect = [
                Mock(returncode=0, stdout=meta_output),
                Mock(returncode=0, stdout=diff_output),
                Mock(returncode=0, stdout=files_output),
            ]
            
            changelists = git_provider._fetch_single_commit_batch(
                workspace_path=str(tmp_git_repo),
                hashes=["abc123"],
                subdirs=["Assets/Localization"],
                embed_diffs=True,
            )
            
            # Same assertions as Perforce test
            assert len(changelists) == 1
            cl = changelists[0]
            
            # Only localization file
            assert len(cl.files) == 1
            assert "Localization" in cl.files[0]
            assert "strings.xml" in cl.files[0]
            
            # Only localization changes in diff
            assert "New" in cl.diff
            assert "Old" in cl.diff