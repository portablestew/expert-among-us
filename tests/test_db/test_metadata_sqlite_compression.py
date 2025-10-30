"""Integration tests for SQLite metadata database with diff compression."""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.models.changelist import Changelist


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test expert name
        expert_name = "test_expert"
        
        # Create the directory structure
        db_dir = Path(tmpdir) / ".expert-among-us" / "data" / expert_name
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the database
        db = SQLiteMetadataDB(expert_name)
        # Override the db_path to use our temp directory
        db.db_path = str(db_dir / "metadata.db")
        db.initialize()
        
        yield db
        
        # Cleanup
        db.close()


def test_insert_retrieve_compressed_changelist(temp_db):
    """Test that changelists are stored compressed and retrieved correctly."""
    # Create changelist with large diff
    diff = "diff --git a/file.py b/file.py\n" * 500
    changelist = Changelist(
        id="abc123",
        expert_name="test_expert",
        timestamp=datetime.now(timezone.utc),
        author="test_author",
        message="test message",
        diff=diff,
        files=["file.py"]
    )
    
    # Insert
    temp_db.insert_changelists([changelist])
    
    # Retrieve
    retrieved = temp_db.get_changelist("abc123")
    assert retrieved is not None
    assert retrieved.diff == diff
    assert retrieved.id == "abc123"
    assert retrieved.author == "test_author"
    assert retrieved.message == "test message"
    assert retrieved.files == ["file.py"]
    
    # Verify it was stored compressed (BLOB should be smaller than original)
    cursor = temp_db.conn.cursor()
    cursor.execute("SELECT diff FROM changelists WHERE id = ?", ("abc123",))
    row = cursor.fetchone()
    assert isinstance(row['diff'], bytes)
    assert len(row['diff']) < len(diff.encode('utf-8'))


def test_multiple_changelists_compression(temp_db):
    """Test compression with multiple changelists."""
    changelists = []
    for i in range(5):
        diff = f"diff --git a/file{i}.py b/file{i}.py\n" * 100
        changelist = Changelist(
            id=f"id_{i}",
            expert_name="test_expert",
            timestamp=datetime.now(timezone.utc),
            author=f"author_{i}",
            message=f"message {i}",
            diff=diff,
            files=[f"file{i}.py"]
        )
        changelists.append(changelist)
    
    # Insert all
    temp_db.insert_changelists(changelists)
    
    # Retrieve all by IDs
    retrieved_ids = [f"id_{i}" for i in range(5)]
    retrieved = temp_db.get_changelists_by_ids(retrieved_ids)
    
    assert len(retrieved) == 5
    for i, changelist in enumerate(retrieved):
        # Verify content
        expected_diff = f"diff --git a/file{i}.py b/file{i}.py\n" * 100
        assert changelist.diff == expected_diff
        assert changelist.id == f"id_{i}"


def test_large_diff_compression(temp_db):
    """Test compression of large diffs near 100KB limit."""
    # Create a large diff (~100KB)
    large_diff = "line content " * 8000  # Approximately 100KB
    
    changelist = Changelist(
        id="large_diff_id",
        expert_name="test_expert",
        timestamp=datetime.now(timezone.utc),
        author="test_author",
        message="large diff test",
        diff=large_diff,
        files=["large_file.py"]
    )
    
    # Insert
    temp_db.insert_changelists([changelist])
    
    # Retrieve
    retrieved = temp_db.get_changelist("large_diff_id")
    assert retrieved is not None
    assert retrieved.diff == large_diff
    
    # Verify compression ratio
    cursor = temp_db.conn.cursor()
    cursor.execute("SELECT diff FROM changelists WHERE id = ?", ("large_diff_id",))
    row = cursor.fetchone()
    compressed_size = len(row['diff'])
    original_size = len(large_diff.encode('utf-8'))
    compression_ratio = compressed_size / original_size
    
    # Should achieve at least 50% compression on repetitive content
    assert compression_ratio < 0.5


def test_unicode_diff_compression(temp_db):
    """Test compression with unicode characters in diff."""
    unicode_diff = "diff --git a/file.py b/file.py\n" \
                   "- old line with émojis 🎉\n" \
                   "+ new line with émojis 🚀\n" * 100
    
    changelist = Changelist(
        id="unicode_id",
        expert_name="test_expert",
        timestamp=datetime.now(timezone.utc),
        author="test_author",
        message="unicode test",
        diff=unicode_diff,
        files=["file.py"]
    )
    
    # Insert
    temp_db.insert_changelists([changelist])
    
    # Retrieve
    retrieved = temp_db.get_changelist("unicode_id")
    assert retrieved is not None
    assert retrieved.diff == unicode_diff


def test_minimal_diff_compression(temp_db):
    """Test compression with minimal diff."""
    minimal_diff = "diff --git a/file.py b/file.py\nindex 123..456\n"
    
    changelist = Changelist(
        id="minimal_id",
        expert_name="test_expert",
        timestamp=datetime.now(timezone.utc),
        author="test_author",
        message="minimal test",
        diff=minimal_diff,
        files=["file.py"]
    )
    
    # Insert
    temp_db.insert_changelists([changelist])
    
    # Retrieve
    retrieved = temp_db.get_changelist("minimal_id")
    assert retrieved is not None
    assert retrieved.diff == minimal_diff


def test_realistic_git_diff_compression(temp_db):
    """Test compression with realistic git diff format."""
    git_diff = """diff --git a/src/main.py b/src/main.py
index 1234567..abcdefg 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@ class MyClass:
     def method(self):
-        old_code = "something"
+        new_code = "something else"
         return result
 
@@ -20,3 +20,4 @@ def another_function():
     pass
+    # Added comment
"""
    
    changelist = Changelist(
        id="git_diff_id",
        expert_name="test_expert",
        timestamp=datetime.now(timezone.utc),
        author="test_author",
        message="realistic git diff",
        diff=git_diff,
        files=["src/main.py"]
    )
    
    # Insert
    temp_db.insert_changelists([changelist])
    
    # Retrieve
    retrieved = temp_db.get_changelist("git_diff_id")
    assert retrieved is not None
    assert retrieved.diff == git_diff
    
    # Verify compression occurred
    cursor = temp_db.conn.cursor()
    cursor.execute("SELECT diff FROM changelists WHERE id = ?", ("git_diff_id",))
    row = cursor.fetchone()
    assert len(row['diff']) < len(git_diff.encode('utf-8'))


def test_schema_uses_blob_type(temp_db):
    """Verify that the schema uses BLOB type for diff column."""
    cursor = temp_db.conn.cursor()
    cursor.execute("PRAGMA table_info(changelists)")
    columns = cursor.fetchall()
    
    # Find the diff column
    diff_column = None
    for col in columns:
        if col['name'] == 'diff':
            diff_column = col
            break
    
    assert diff_column is not None
    # SQLite stores BLOB type as 'BLOB' in the type field
    assert diff_column['type'] == 'BLOB'