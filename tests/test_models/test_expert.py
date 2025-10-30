"""Tests for ExpertConfig model."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from expert_among_us.models.expert import ExpertConfig


def test_expert_config_creation():
    """Test basic expert config creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        config = ExpertConfig(
            name="TestExpert",
            workspace_path=workspace,
            vcs_type="git",
            max_commits=5000,
        )
        
        assert config.name == "TestExpert"
        assert config.workspace_path == workspace
        assert config.vcs_type == "git"
        assert config.max_commits == 5000


def test_expert_config_validation_invalid_name():
    """Test that invalid names raise validation errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Empty name
        with pytest.raises(ValidationError):
            ExpertConfig(name="", workspace_path=workspace)
        
        # Invalid characters
        with pytest.raises(ValidationError):
            ExpertConfig(name="Test@Expert", workspace_path=workspace)


def test_expert_config_validation_valid_names():
    """Test that valid names are accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Alphanumeric
        config = ExpertConfig(name="MyExpert123", workspace_path=workspace)
        assert config.name == "MyExpert123"
        
        # With hyphens
        config = ExpertConfig(name="my-expert", workspace_path=workspace)
        assert config.name == "my-expert"
        
        # With underscores
        config = ExpertConfig(name="my_expert", workspace_path=workspace)
        assert config.name == "my_expert"


def test_expert_config_validation_nonexistent_workspace():
    """Test that nonexistent workspace raises validation error."""
    nonexistent = Path("/nonexistent/path/12345")
    
    with pytest.raises(ValidationError, match="does not exist"):
        ExpertConfig(name="TestExpert", workspace_path=nonexistent)


def test_expert_config_validation_workspace_not_directory():
    """Test that file path (not directory) raises validation error."""
    with tempfile.NamedTemporaryFile() as tmpfile:
        file_path = Path(tmpfile.name)
        
        with pytest.raises(ValidationError, match="not a directory"):
            ExpertConfig(name="TestExpert", workspace_path=file_path)


def test_expert_config_defaults():
    """Test default values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        config = ExpertConfig(name="TestExpert", workspace_path=workspace)
        
        assert config.vcs_type == "git"
        assert config.max_commits == 10000
        assert config.max_diff_size == 100000
        assert config.max_embedding_text_size == 30000
        assert config.embed_diffs is True
        assert config.embed_metadata is True
        assert config.subdirs == []
        assert config.last_indexed_at is None
        assert config.last_commit_time is None


def test_expert_config_get_storage_dir():
    """Test storage directory path generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        config = ExpertConfig(name="TestExpert", workspace_path=workspace)
        storage_dir = config.get_storage_dir()
        
        assert storage_dir.name == "TestExpert"
        assert ".expert-among-us" in str(storage_dir)
        assert "data" in str(storage_dir)


def test_expert_config_get_metadata_db_path():
    """Test metadata database path generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        config = ExpertConfig(name="TestExpert", workspace_path=workspace)
        db_path = config.get_metadata_db_path()
        
        assert db_path.name == "metadata.db"
        assert "TestExpert" in str(db_path)


def test_expert_config_get_vector_db_path():
    """Test vector database path generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        config = ExpertConfig(name="TestExpert", workspace_path=workspace)
        db_path = config.get_vector_db_path()
        
        assert db_path.name == "chroma"
        assert "TestExpert" in str(db_path)


def test_expert_config_ensure_storage_exists():
    """Test storage directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        config = ExpertConfig(name="TestExpert", workspace_path=workspace)
        
        storage_dir = config.get_storage_dir()
        
        # Ensure a clean state for the test: remove the directory if it exists from a prior run
        if storage_dir.exists():
            import shutil
            shutil.rmtree(storage_dir)
            
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


def test_expert_config_with_subdirs():
    """Test expert config with subdirectory filters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        config = ExpertConfig(
            name="TestExpert",
            workspace_path=workspace,
            subdirs=["src/main/", "src/resources/"],
        )
        
        assert len(config.subdirs) == 2
        assert "src/main/" in config.subdirs
        assert "src/resources/" in config.subdirs


def test_expert_config_timestamps():
    """Test timestamp handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        config = ExpertConfig(name="TestExpert", workspace_path=workspace)
        
        # created_at should be set
        assert config.created_at is not None
        assert isinstance(config.created_at, datetime)
        
        # Should be recent (within last minute)
        now = datetime.now(timezone.utc)
        time_diff = (now - config.created_at).total_seconds()
        assert time_diff < 60