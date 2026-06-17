"""Tests for ExpertConfig and ProjectConfig models."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from expert_among_us.models.expert import ExpertConfig, ProjectConfig


# ============================================================
# ExpertConfig Tests
# ============================================================


def test_expert_config_creation():
    """Test basic expert config creation."""
    config = ExpertConfig(name="TestExpert")

    assert config.name == "TestExpert"
    assert config.description is None


def test_expert_config_with_description():
    """Test expert config creation with description."""
    config = ExpertConfig(
        name="TestExpert",
        description="A test expert for unit tests",
    )

    assert config.name == "TestExpert"
    assert config.description == "A test expert for unit tests"


def test_expert_config_validation_invalid_name():
    """Test that invalid names raise validation errors."""
    # Empty name
    with pytest.raises(ValidationError):
        ExpertConfig(name="")

    # Invalid characters
    with pytest.raises(ValidationError):
        ExpertConfig(name="Test@Expert")

    # Path separators
    with pytest.raises(ValidationError):
        ExpertConfig(name="path/to/expert")

    with pytest.raises(ValidationError):
        ExpertConfig(name="path\\to\\expert")

    # Spaces
    with pytest.raises(ValidationError):
        ExpertConfig(name="has space")


def test_expert_config_validation_valid_names():
    """Test that valid names are accepted."""
    # Alphanumeric
    config = ExpertConfig(name="MyExpert123")
    assert config.name == "MyExpert123"

    # With hyphens
    config = ExpertConfig(name="my-expert")
    assert config.name == "my-expert"

    # With underscores
    config = ExpertConfig(name="my_expert")
    assert config.name == "my_expert"

    # Single character
    config = ExpertConfig(name="a")
    assert config.name == "a"


def test_expert_config_defaults():
    """Test default values."""
    config = ExpertConfig(name="TestExpert")

    assert config.max_embedding_text_size == 100000
    assert config.embed_diffs is True
    assert config.embed_metadata is True
    assert config.last_indexed_at is None
    assert config.description is None


def test_expert_config_get_storage_dir():
    """Test storage directory path generation."""
    config = ExpertConfig(name="TestExpert")
    storage_dir = config.get_storage_dir()

    assert storage_dir.name == "TestExpert"
    assert ".expert-among-us" in str(storage_dir)
    assert "data" in str(storage_dir)


def test_expert_config_get_metadata_db_path():
    """Test metadata database path generation."""
    config = ExpertConfig(name="TestExpert")
    db_path = config.get_metadata_db_path()

    assert db_path.name == "metadata.db"
    assert "TestExpert" in str(db_path)


def test_expert_config_get_vector_db_path():
    """Test vector database path generation."""
    config = ExpertConfig(name="TestExpert")
    db_path = config.get_vector_db_path()

    assert db_path.name == "chroma"
    assert "TestExpert" in str(db_path)


def test_expert_config_ensure_storage_exists():
    """Test storage directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExpertConfig(name="TestExpert", data_dir=Path(tmpdir))

        storage_dir = config.get_storage_dir()
        assert not storage_dir.exists()

        # Create storage
        config.ensure_storage_exists()

        # Now it exists
        assert storage_dir.exists()
        assert storage_dir.is_dir()

        # Vector DB directory also created
        vector_dir = config.get_vector_db_path()
        assert vector_dir.exists()
        assert vector_dir.is_dir()


def test_expert_config_timestamps():
    """Test timestamp handling."""
    config = ExpertConfig(name="TestExpert")

    # created_at should be set
    assert config.created_at is not None
    assert isinstance(config.created_at, datetime)

    # Should be recent (within last minute)
    now = datetime.now(timezone.utc)
    time_diff = (now - config.created_at).total_seconds()
    assert time_diff < 60


# ============================================================
# ProjectConfig Tests
# ============================================================


def test_project_config_creation():
    """Test basic project config creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        config = ProjectConfig(
            name="payment-service",
            expert_name="my-team",
            project_root=workspace,
            vcs_type="git",
        )

        assert config.name == "payment-service"
        assert config.expert_name == "my-team"
        assert config.project_root == workspace
        assert config.vcs_type == "git"
        assert config.has_vector_metadata is True


def test_project_config_defaults():
    """Test default field values for ProjectConfig."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        config = ProjectConfig(
            name="myproject",
            expert_name="myexpert",
            project_root=workspace,
        )

        assert config.vcs_type == "git"
        assert config.last_indexed_at is None
        assert config.last_processed_commit_hash is None
        assert config.first_processed_commit_hash is None
        assert config.has_vector_metadata is True
        assert config.created_at is not None


def test_project_config_valid_names():
    """Test that valid project names are accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        valid_names = [
            "a",
            "A",
            "abc",
            "my-project",
            "my_project",
            "MyProject123",
            "a1-b2_c3",
        ]

        for name in valid_names:
            config = ProjectConfig(
                name=name,
                expert_name="expert1",
                project_root=workspace,
            )
            assert config.name == name


def test_project_config_invalid_names():
    """Test that invalid project names are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        invalid_names = [
            "",           # empty
            " ",          # whitespace only
            "has space",  # contains space
            "has/slash",  # path separator
            "has\\back",  # backslash
            "has.dot",    # period
            "@special",   # special char start
            "name@here",  # special char middle
        ]

        for name in invalid_names:
            with pytest.raises(ValidationError, match="Project name"):
                ProjectConfig(
                    name=name,
                    expert_name="expert1",
                    project_root=workspace,
                )


def test_project_config_invalid_expert_name():
    """Test that invalid expert names in ProjectConfig are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        with pytest.raises(ValidationError, match="Expert name"):
            ProjectConfig(
                name="valid-project",
                expert_name="",
                project_root=workspace,
            )

        with pytest.raises(ValidationError, match="Expert name"):
            ProjectConfig(
                name="valid-project",
                expert_name="bad/name",
                project_root=workspace,
            )


def test_project_config_nonexistent_workspace():
    """Test that nonexistent workspace raises validation error."""
    nonexistent = Path("/nonexistent/path/12345")

    with pytest.raises(ValidationError, match="does not exist"):
        ProjectConfig(
            name="myproject",
            expert_name="myexpert",
            project_root=nonexistent,
        )


def test_project_config_workspace_not_directory():
    """Test that file path (not directory) raises validation error."""
    with tempfile.NamedTemporaryFile() as tmpfile:
        file_path = Path(tmpfile.name)

        with pytest.raises(ValidationError, match="not a directory"):
            ProjectConfig(
                name="myproject",
                expert_name="myexpert",
                project_root=file_path,
            )


def test_project_config_perforce_vcs():
    """Test project config with perforce VCS type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        config = ProjectConfig(
            name="shared-lib",
            expert_name="myexpert",
            project_root=workspace,
            vcs_type="p4",
        )

        assert config.vcs_type == "p4"


def test_project_config_has_vector_metadata_false():
    """Test project config with has_vector_metadata=False (legacy/migrated)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        config = ProjectConfig(
            name="legacy-project",
            expert_name="myexpert",
            project_root=workspace,
            has_vector_metadata=False,
        )

        assert config.has_vector_metadata is False


def test_project_config_timestamps():
    """Test timestamp handling for ProjectConfig."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        config = ProjectConfig(
            name="myproject",
            expert_name="myexpert",
            project_root=workspace,
        )

        # created_at should be set automatically
        assert config.created_at is not None
        assert isinstance(config.created_at, datetime)

        # Should be recent (within last minute)
        now = datetime.now(timezone.utc)
        time_diff = (now - config.created_at).total_seconds()
        assert time_diff < 60
