"""Tests for ChromaDB batching behavior to handle large insert operations."""

import pytest
from pathlib import Path
import tempfile
import shutil
import time
import gc
from expert_among_us.db.vector.chroma import ChromaVectorDB, CHROMA_MAX_BATCH_SIZE


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    
    # Give ChromaDB time to release file handles on Windows
    gc.collect()
    time.sleep(0.1)
    
    # Retry cleanup with exponential backoff for Windows file locking
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))
            else:
                # Last attempt failed, but don't fail the test
                pass


@pytest.fixture
def chroma_db(temp_data_dir):
    """Create a ChromaVectorDB instance for testing."""
    db = ChromaVectorDB("test_batching", data_dir=temp_data_dir)
    db.initialize(dimension=128)  # Smaller dimension for faster tests
    yield db
    db.close()
    # Force garbage collection to release resources
    gc.collect()


def test_small_batch_insert_metadata(chroma_db):
    """Test that small batches work correctly."""
    vectors = [(f"commit_{i}", [0.1] * 128) for i in range(100)]
    
    chroma_db.insert_metadata(vectors)
    
    assert chroma_db.metadata_collection.count() == 100


def test_large_batch_insert_diffs(chroma_db):
    """Test that large diff batches are automatically chunked and inserted correctly."""
    # Create batch larger than CHROMA_MAX_BATCH_SIZE to trigger batching
    large_batch = [
        (f"commit_{i}_chunk_{j}", [0.1] * 128)
        for i in range(100)
        for j in range(60)  # 6000 vectors total
    ]
    
    # Should not raise BatchSizeError
    chroma_db.insert_diffs(large_batch)
    
    # Verify all vectors were inserted
    assert chroma_db.diff_collection.count() == 6000


def test_large_batch_insert_files(chroma_db):
    """Test that large file batches are automatically chunked."""
    large_batch = [
        (f"file_{i}:chunk_{j}", [0.2] * 128)
        for i in range(50)
        for j in range(120)  # 6000 vectors total
    ]
    
    chroma_db.insert_files(large_batch)
    
    assert chroma_db.file_collection.count() == 6000


def test_exact_batch_size_insert(chroma_db):
    """Test insert at exactly the batch size limit."""
    vectors = [(f"commit_{i}", [0.1] * 128) for i in range(CHROMA_MAX_BATCH_SIZE)]
    
    chroma_db.insert_metadata(vectors)
    
    assert chroma_db.metadata_collection.count() == CHROMA_MAX_BATCH_SIZE


def test_just_over_batch_size_insert(chroma_db):
    """Test insert just over the batch size limit."""
    vectors = [(f"commit_{i}", [0.1] * 128) for i in range(CHROMA_MAX_BATCH_SIZE + 1)]
    
    chroma_db.insert_metadata(vectors)
    
    assert chroma_db.metadata_collection.count() == CHROMA_MAX_BATCH_SIZE + 1


def test_empty_batch_insert(chroma_db):
    """Test that empty batches don't cause errors."""
    chroma_db.insert_metadata([])
    chroma_db.insert_diffs([])
    chroma_db.insert_files([])
    
    assert chroma_db.metadata_collection.count() == 0
    assert chroma_db.diff_collection.count() == 0
    assert chroma_db.file_collection.count() == 0


