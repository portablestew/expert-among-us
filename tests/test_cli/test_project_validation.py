"""Unit tests for CLI --project validation on the populate command.

Validates: Requirements 7.1, 7.4, 2.1, 2.2, 2.3
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from expert_among_us.__main__ import main


@pytest.fixture
def runner():
    """Create a Click CliRunner for testing CLI commands."""
    return CliRunner()


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary directory to act as a valid workspace path."""
    return str(tmp_path)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory so tests don't touch real user data."""
    data_dir = tmp_path / "expert-data"
    data_dir.mkdir()
    return str(data_dir)


class TestMissingProjectFlag:
    """Test that missing --project produces error exit."""

    def test_populate_without_project_flag_fails(self, runner, temp_workspace):
        """Invoking populate without --project should fail with non-zero exit code."""
        result = runner.invoke(main, ["populate", "TestExpert", temp_workspace])
        assert result.exit_code != 0
        # Click reports missing required option
        assert "Missing option" in result.output or "--project" in result.output

    def test_populate_without_project_shows_error_message(self, runner, temp_workspace):
        """The error message should reference the --project option."""
        result = runner.invoke(main, ["populate", "TestExpert", temp_workspace])
        assert result.exit_code != 0
        assert "--project" in result.output or "project" in result.output.lower()


class TestInvalidProjectNames:
    """Test that invalid project names are rejected with exit code 1."""

    @pytest.mark.parametrize("invalid_name,description", [
        ("has/slash", "contains forward slash"),
        ("has\\backslash", "contains backslash"),
        ("-starts-hyphen", "starts with hyphen"),
        ("_starts-underscore", "starts with underscore"),
        ("has space", "contains space"),
        ("has.dot", "contains dot"),
        ("has@symbol", "contains @ symbol"),
        ("has!bang", "contains exclamation mark"),
    ])
    def test_invalid_project_name_rejected(
        self, runner, temp_workspace, temp_data_dir, invalid_name, description
    ):
        """Project names with invalid characters should be rejected."""
        result = runner.invoke(
            main,
            ["--data-dir", temp_data_dir, "populate", "TestExpert", temp_workspace,
             "--project", invalid_name],
        )
        assert result.exit_code == 1, (
            f"Expected exit code 1 for project name '{invalid_name}' ({description}), "
            f"got {result.exit_code}. Output: {result.output}"
        )

    def test_empty_project_name_rejected(self, runner, temp_workspace, temp_data_dir):
        """Empty project names should be rejected."""
        result = runner.invoke(
            main,
            ["--data-dir", temp_data_dir, "populate", "TestExpert", temp_workspace,
             "--project", ""],
        )
        assert result.exit_code == 1

    def test_invalid_project_name_shows_error_message(
        self, runner, temp_workspace, temp_data_dir
    ):
        """Invalid project name should show descriptive error message."""
        result = runner.invoke(
            main,
            ["--data-dir", temp_data_dir, "populate", "TestExpert", temp_workspace,
             "--project", "has/slash"],
        )
        assert result.exit_code == 1
        # Should mention it's invalid
        assert "invalid" in result.output.lower() or "Invalid" in result.output


class TestValidProjectNames:
    """Test that valid project names pass validation (may fail later due to missing VCS)."""

    @pytest.mark.parametrize("valid_name", [
        "my-project",
        "MyProject",
        "project123",
        "a",
        "A",
        "project_name",
        "Test-Project_123",
        "9starts-with-digit",
    ])
    @patch("expert_among_us.__main__.detect_vcs")
    @patch("expert_among_us.__main__.create_embedder")
    @patch("expert_among_us.__main__.SQLiteMetadataDB")
    @patch("expert_among_us.__main__.ChromaVectorDB")
    def test_valid_project_name_passes_validation(
        self,
        mock_chroma_db,
        mock_sqlite_db,
        mock_create_embedder,
        mock_detect_vcs,
        runner,
        temp_workspace,
        temp_data_dir,
        valid_name,
    ):
        """Valid project names should pass the regex validation step.

        We mock out DB and VCS to isolate project name validation.
        The test verifies we get past validation (exit code != 1 for name rejection).
        """
        # Mock VCS detection to return a mock provider
        mock_vcs = MagicMock()
        mock_vcs.__class__.__name__ = "Git"
        mock_vcs.get_total_commit_count.return_value = 0
        mock_detect_vcs.return_value = mock_vcs

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_create_embedder.return_value = mock_embedder

        # Mock SQLiteMetadataDB
        mock_db_instance = MagicMock()
        mock_db_instance.exists.return_value = False
        mock_db_instance.get_expert.return_value = None
        mock_db_instance.get_project.return_value = None
        mock_db_instance.get_project_commit_count.return_value = 0
        mock_sqlite_db.return_value = mock_db_instance

        # Mock ChromaVectorDB
        mock_vector_instance = MagicMock()
        mock_chroma_db.return_value = mock_vector_instance

        # Mock the Indexer (imported locally inside populate via
        # `from expert_among_us.core.indexer import Indexer`)
        with patch("expert_among_us.core.indexer.Indexer") as mock_indexer_class:
            mock_indexer = MagicMock()
            mock_indexer.index_unified.return_value = False  # no more remain
            mock_indexer_class.return_value = mock_indexer

            result = runner.invoke(
                main,
                ["--data-dir", temp_data_dir, "populate", "TestExpert", temp_workspace,
                 "--project", valid_name],
            )

            # Should NOT exit with code 1 (name validation failure)
            # Exit code 0 means success, any other non-1 code is a different error
            assert result.exit_code != 1 or "Invalid project name" not in result.output, (
                f"Valid project name '{valid_name}' was incorrectly rejected. "
                f"Output: {result.output}"
            )


class TestNonExistentWorkspace:
    """Test behavior when workspace doesn't exist and project not found."""

    def test_nonexistent_workspace_shows_error(self, runner, temp_data_dir):
        """A non-existent workspace path should exit with error."""
        result = runner.invoke(
            main,
            ["--data-dir", temp_data_dir, "populate", "TestExpert", "/nonexistent/path/xyz",
             "--project", "valid-project"],
        )
        assert result.exit_code != 0

    def test_no_workspace_no_existing_project_shows_error(self, runner, temp_data_dir):
        """When no workspace is provided and project doesn't exist, should error."""
        result = runner.invoke(
            main,
            ["--data-dir", temp_data_dir, "populate", "TestExpert",
             "--project", "nonexistent-project"],
        )
        assert result.exit_code != 0
        # Should mention workspace is required or project doesn't exist
        assert "workspace" in result.output.lower() or "does not exist" in result.output.lower()
