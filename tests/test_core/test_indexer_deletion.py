"""Tests for file chunk deletion synchronization between SQLite and ChromaDB."""

import datetime
from unittest.mock import Mock, MagicMock

import pytest

from expert_among_us.core.indexer import Indexer
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.file_chunk import FileChunk
from expert_among_us.config.settings import Settings


@pytest.fixture
def expert_config(tmp_path):
    return {
        "name": "TestExpert",
        "workspace_path": str(tmp_path),
        "max_commits": 10,
    }


@pytest.fixture
def mock_vcs():
    """Mock VCS provider that can simulate files being deleted."""
    vcs = Mock()
    vcs.get_total_commit_count.return_value = 2
    vcs.get_tracked_files_at_commit.return_value = []  # Simulates deleted file
    return vcs


@pytest.fixture
def mock_metadata_db():
    """Mock metadata DB that tracks chunk IDs."""
    db = Mock()
    db.get_last_processed_commit_hash.return_value = None
    db.get_commit_count.return_value = 0
    
    # Simulate file chunks existing in sqlite
    chunk_ids = ["file:test.py:chunk_0", "file:test.py:chunk_1", "file:test.py:chunk_2"]
    db.get_file_chunk_ids.return_value = chunk_ids
    db.delete_file_chunks_by_path.return_value = len(chunk_ids)
    
    return db


@pytest.fixture
def mock_vector_db():
    """Mock vector DB that tracks deletions."""
    db = Mock()
    return db


@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    embedder = Mock()
    embedder.embed_batch.return_value = [[0.1, 0.2]]
    embedder.dimension = 2
    return embedder


@pytest.fixture
def indexer(expert_config, mock_vcs, mock_metadata_db, mock_vector_db, mock_embedder):
    settings = Settings()
    return Indexer(
        expert_config=expert_config,
        vcs=mock_vcs,
        metadata_db=mock_metadata_db,
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        settings=settings,
    )


def test_delete_file_chunks_uses_sqlite_ids(indexer, mock_metadata_db, mock_vector_db):
    """
    Verify that _delete_file_chunks uses SQLite as source of truth.
    
    This tests the fix for the bug where ChromaDB deletions failed because
    they relied on metadata that was never written. The corrected flow:
    1. Get chunk IDs from SQLite (source of truth)
    2. Delete those IDs from ChromaDB
    3. Delete rows from SQLite
    """
    file_path = "test.py"
    expected_ids = ["file:test.py:chunk_0", "file:test.py:chunk_1", "file:test.py:chunk_2"]
    
    # Execute deletion
    indexer._delete_file_chunks(file_path)
    
    # Verify SQLite was queried for chunk IDs
    mock_metadata_db.get_file_chunk_ids.assert_called_once_with(file_path)
    
    # Verify ChromaDB deletion used the exact IDs from SQLite
    mock_vector_db.delete_file_chunks.assert_called_once_with(expected_ids)
    
    # Verify SQLite deletion happened after ChromaDB
    mock_metadata_db.delete_file_chunks_by_path.assert_called_once_with(file_path)
    
    # Verify call order: get_ids -> delete_chroma -> delete_sqlite
    calls = [
        mock_metadata_db.get_file_chunk_ids.call_args_list,
        mock_vector_db.delete_file_chunks.call_args_list,
        mock_metadata_db.delete_file_chunks_by_path.call_args_list,
    ]
    assert all(calls), "All deletion steps should be called"


def test_delete_file_chunks_handles_empty_results(indexer, mock_metadata_db, mock_vector_db):
    """
    Verify that deletion handles the case when no chunks exist in SQLite.
    """
    file_path = "nonexistent.py"
    mock_metadata_db.get_file_chunk_ids.return_value = []
    
    # Execute deletion
    indexer._delete_file_chunks(file_path)
    
    # ChromaDB delete_file_chunks should not be called with empty list
    mock_vector_db.delete_file_chunks.assert_not_called()
    
    # SQLite deletion should still be called (idempotent)
    mock_metadata_db.delete_file_chunks_by_path.assert_called_once_with(file_path)


def test_indexer_detects_missing_files_and_deletes(indexer, mock_vcs, mock_metadata_db, mock_vector_db):
    """
    Integration test: verify that when a file is missing from VCS HEAD,
    the indexer deletes its chunks from both databases.
    """
    base_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    
    # Setup: VCS reports commit with file that no longer exists
    commit = Changelist(
        id="commit1",
        expert_name="TestExpert",
        timestamp=base_time,
        author="tester",
        message="Test commit",
        diff="diff content",
        files=["deleted_file.py"],  # This file will be reported as missing
    )
    
    mock_vcs.get_commits_after.return_value = [commit]
    mock_vcs.get_tracked_files_at_commit.return_value = []  # File doesn't exist at HEAD
    
    # Setup metadata DB to report chunks exist
    mock_metadata_db.get_file_chunk_ids.return_value = ["file:deleted_file.py:chunk_0"]
    
    # Run indexing
    indexer.index_unified(batch_size=10)
    
    # Verify deletion was triggered for the missing file
    mock_metadata_db.get_file_chunk_ids.assert_called_with("deleted_file.py")
    mock_vector_db.delete_file_chunks.assert_called_with(["file:deleted_file.py:chunk_0"])
    mock_metadata_db.delete_file_chunks_by_path.assert_called_with("deleted_file.py")