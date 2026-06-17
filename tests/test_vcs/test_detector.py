"""
Tests for VCS detection functionality covering:
- Detecting Git repositories correctly
- Returning None when no VCS is found
- Testing the extensibility of the provider registry
"""

import pytest
import tempfile
import subprocess
from pathlib import Path

from expert_among_us.vcs.detector import detect_vcs
from expert_among_us.vcs.git import Git
from expert_among_us.config.settings import Settings


@pytest.fixture
def settings():
    """Fixture providing a Settings instance for tests."""
    return Settings()


class TestVCSDetection:
    """Tests for VCS detection functionality."""

    def test_detect_git_repository(self, settings):
        """Verify that Git repositories are correctly detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            vcs = detect_vcs(str(repo_path), settings)
            
            assert vcs is not None
            assert isinstance(vcs, Git)

    def test_detect_no_vcs_in_empty_directory(self, settings):
        """Verify that empty directories return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vcs = detect_vcs(tmpdir, settings)
            
            assert vcs is None

    def test_detect_vcs_with_nonexistent_path(self, settings):
        """Verify behavior when path doesn't exist."""
        vcs = detect_vcs("/nonexistent/path", settings)
        
        assert vcs is None

    def test_detect_vcs_multiple_repositories_root_wins(self, settings):
        """Verify that when in a git directory, it's detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outer_repo = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=outer_repo,
                capture_output=True,
                check=True
            )
            
            inner_dir = outer_repo / "nested"
            inner_dir.mkdir()
            
            subprocess.run(
                ["git", "init"],
                cwd=inner_dir,
                capture_output=True,
                check=True
            )
            
            vcs = detect_vcs(str(inner_dir), settings)
            
            assert vcs is not None
            assert isinstance(vcs, Git)

    def test_detect_vcs_with_file_not_directory(self, settings):
        """Verify that passing a file path (not directory) returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("test content")
            
            vcs = detect_vcs(str(file_path), settings)
            
            assert vcs is None


class TestEdgeCases:
    """Tests for edge cases in VCS detection."""

    def test_detect_vcs_with_relative_path(self, settings):
        """Verify VCS detection works with relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(repo_path)
                vcs = detect_vcs(".", settings)
                assert vcs is not None
                assert isinstance(vcs, Git)
            finally:
                os.chdir(original_cwd)

    def test_detect_vcs_with_trailing_slash(self, settings):
        """Verify VCS detection works with trailing slashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            path_with_slash = str(repo_path) + "/"
            vcs = detect_vcs(path_with_slash, settings)
            assert vcs is not None
            assert isinstance(vcs, Git)

    def test_detect_vcs_empty_string_path(self, settings):
        """Verify behavior with empty string path."""
        vcs = detect_vcs("", settings)
        
        assert vcs is None

    def test_detect_vcs_whitespace_path(self, settings):
        """Verify behavior with whitespace path."""
        vcs = detect_vcs("   ", settings)
        
        assert vcs is None


class TestProviderRegistry:
    """Tests for the provider registry extensibility."""

    def test_provider_registry_contains_git(self):
        """Verify that Git is in the provider registry."""
        from expert_among_us.vcs.detector import VCS_PROVIDERS
        
        assert len(VCS_PROVIDERS) > 0
        
        provider_names = [p.__name__ for p in VCS_PROVIDERS]
        assert any("Git" in name for name in provider_names)
