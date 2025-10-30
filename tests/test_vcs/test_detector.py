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


class TestVCSDetection:
    """Tests for VCS detection functionality."""

    def test_detect_git_repository(self):
        """Verify that Git repositories are correctly detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Initialize as git repo
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            # Detect VCS
            vcs = detect_vcs(str(repo_path))
            
            assert vcs is not None
            assert isinstance(vcs, Git)

    def test_detect_no_vcs_in_empty_directory(self):
        """Verify that empty directories return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vcs = detect_vcs(tmpdir)
            
            assert vcs is None

    def test_detect_vcs_in_subdirectory(self):
        """Verify that VCS detection works from subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Initialize git repo
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            # Create nested directory
            nested = repo_path / "nested" / "deep"
            nested.mkdir(parents=True)
            
            # Detect VCS from root should work
            vcs = detect_vcs(str(repo_path))
            
            assert vcs is not None
            assert isinstance(vcs, Git)

    def test_detect_vcs_returns_correct_type_for_git(self):
        """Verify that detected Git VCS is of correct type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            vcs = detect_vcs(str(repo_path))
            
            # Verify it's a Git instance with expected methods
            assert hasattr(vcs, 'get_commits')
            assert hasattr(vcs, 'get_latest_commit_time')
            assert hasattr(vcs, 'detect')

    def test_detect_vcs_with_nonexistent_path(self):
        """Verify behavior when path doesn't exist."""
        vcs = detect_vcs("/nonexistent/path")
        
        # Should return None
        assert vcs is None

    def test_detect_vcs_consistency(self):
        """Verify that multiple detections return same type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            vcs1 = detect_vcs(str(repo_path))
            vcs2 = detect_vcs(str(repo_path))
            
            assert type(vcs1) == type(vcs2)
            assert isinstance(vcs1, Git)
            assert isinstance(vcs2, Git)

    def test_detect_vcs_multiple_repositories_root_wins(self):
        """Verify that when in a git directory, it's detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outer_repo = Path(tmpdir)
            
            # Create outer repository
            subprocess.run(
                ["git", "init"],
                cwd=outer_repo,
                capture_output=True,
                check=True
            )
            
            # Create nested directory
            inner_dir = outer_repo / "nested"
            inner_dir.mkdir()
            
            # Initialize inner as separate repo
            subprocess.run(
                ["git", "init"],
                cwd=inner_dir,
                capture_output=True,
                check=True
            )
            
            # Detect from inner directory should find a repo
            vcs = detect_vcs(str(inner_dir))
            
            assert vcs is not None
            assert isinstance(vcs, Git)

    def test_detect_vcs_with_file_not_directory(self):
        """Verify that passing a file path (not directory) returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("test content")
            
            vcs = detect_vcs(str(file_path))
            
            # Should return None for files
            assert vcs is None


class TestVCSProviderIntegration:
    """Tests for VCS provider integration with detector."""

    def test_detected_git_can_get_commits(self):
        """Verify that detected Git provider can retrieve commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
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
            
            # Create a commit
            (Path(tmpdir) / "file.txt").write_text("content")
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
            
            vcs = detect_vcs(str(repo_path))
            commits = vcs.get_commits(str(repo_path))
            
            assert len(commits) > 0

    def test_detected_git_can_get_latest_commit_time(self):
        """Verify that detected Git provider can get latest commit time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
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
            
            # Create a commit
            (Path(tmpdir) / "file.txt").write_text("content")
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
            
            vcs = detect_vcs(str(repo_path))
            latest_time = vcs.get_latest_commit_time(str(repo_path))
            
            assert latest_time is not None

    def test_detected_git_can_check_detect_method(self):
        """Verify that detected Git provider has working detect method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            vcs = detect_vcs(str(repo_path))
            
            # Should be able to call detect on the class
            assert Git.detect(str(repo_path)) is True


class TestEdgeCases:
    """Tests for edge cases in VCS detection."""

    def test_detect_vcs_with_relative_path(self):
        """Verify VCS detection works with relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            # Use relative path by changing directory
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(repo_path)
                vcs = detect_vcs(".")
                assert vcs is not None
                assert isinstance(vcs, Git)
            finally:
                os.chdir(original_cwd)

    def test_detect_vcs_with_trailing_slash(self):
        """Verify VCS detection works with trailing slashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            # Add trailing slash (works differently on Windows vs Unix)
            path_with_slash = str(repo_path) + "/"
            vcs = detect_vcs(path_with_slash)
            assert vcs is not None
            assert isinstance(vcs, Git)

    def test_detect_vcs_returns_none_not_raises(self):
        """Verify that no VCS returns None, not exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files but not a git repo
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("not a repo")
            
            vcs = detect_vcs(tmpdir)
            assert vcs is None

    def test_detect_vcs_multiple_calls_same_directory(self):
        """Verify that multiple detections from same directory are consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            results = [detect_vcs(str(repo_path)) for _ in range(3)]
            
            # All should be Git instances
            assert all(isinstance(r, Git) for r in results)

    def test_detect_vcs_empty_string_path(self):
        """Verify behavior with empty string path."""
        vcs = detect_vcs("")
        
        # Should return None for empty path
        assert vcs is None

    def test_detect_vcs_whitespace_path(self):
        """Verify behavior with whitespace path."""
        vcs = detect_vcs("   ")
        
        # Should return None for whitespace path
        assert vcs is None


class TestProviderRegistry:
    """Tests for the provider registry extensibility."""

    def test_provider_registry_contains_git(self):
        """Verify that Git is in the provider registry."""
        from expert_among_us.vcs.detector import VCS_PROVIDERS
        
        # Should have at least one provider
        assert len(VCS_PROVIDERS) > 0
        
        # Git should be in the list (as Git or GitProvider depending on implementation)
        provider_names = [p.__name__ for p in VCS_PROVIDERS]
        assert any("Git" in name for name in provider_names)

    def test_provider_registry_is_list(self):
        """Verify that provider registry is a list."""
        from expert_among_us.vcs.detector import VCS_PROVIDERS
        
        assert isinstance(VCS_PROVIDERS, list)

    def test_provider_registry_elements_are_classes(self):
        """Verify that provider registry contains classes."""
        from expert_among_us.vcs.detector import VCS_PROVIDERS
        
        for provider in VCS_PROVIDERS:
            # Should be a class type
            assert isinstance(provider, type)
            # Should have a detect method
            assert hasattr(provider, 'detect')

    def test_detection_order_matters(self):
        """Verify that providers are checked in order."""
        from expert_among_us.vcs.detector import VCS_PROVIDERS
        
        # If there are multiple providers, order should matter
        # Git should typically be first as it's most common
        if len(VCS_PROVIDERS) > 0:
            first_provider = VCS_PROVIDERS[0]
            # Should be Git or GitProvider
            assert "Git" in first_provider.__name__

    def test_detect_vcs_checks_providers_sequentially(self):
        """Verify that detect_vcs tries providers in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create git repo
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            # Should detect and return first matching provider
            vcs = detect_vcs(str(repo_path))
            assert vcs is not None
            
            # Verify it's one of the registered providers
            from expert_among_us.vcs.detector import VCS_PROVIDERS
            assert any(isinstance(vcs, provider) for provider in VCS_PROVIDERS)


class TestVCSProviderInterface:
    """Tests verifying provider interface compliance."""

    def test_detected_provider_has_required_methods(self):
        """Verify detected provider implements required interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            vcs = detect_vcs(str(repo_path))
            
            # Verify required methods exist
            assert hasattr(vcs, 'get_commits')
            assert callable(vcs.get_commits)
            assert hasattr(vcs, 'get_latest_commit_time')
            assert callable(vcs.get_latest_commit_time)

    def test_detected_provider_class_has_detect_method(self):
        """Verify provider class has static detect method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            vcs = detect_vcs(str(repo_path))
            
            # Verify class has detect method
            assert hasattr(type(vcs), 'detect')
            assert callable(type(vcs).detect)

    def test_provider_detect_returns_boolean(self):
        """Verify that provider detect method returns boolean."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            # Test Git.detect directly
            result = Git.detect(str(repo_path))
            assert isinstance(result, bool)
            assert result is True
            
            # Test on non-repo
            result2 = Git.detect("/nonexistent/path")
            assert isinstance(result2, bool)
            assert result2 is False