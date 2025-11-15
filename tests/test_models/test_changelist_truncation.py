"""Tests for Changelist metadata truncation with 4KB limits."""

from datetime import datetime, timezone

import pytest

from expert_among_us.models.changelist import Changelist


def test_get_metadata_text_message_truncation():
    """Test that commit messages are truncated at 4KB."""
    # Create a message larger than 4KB
    large_message = "X" * 5000  # 5KB
    
    cl = Changelist(
        id="test123",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message=large_message,
        diff="diff --git a/test.py...",
        files=["test.py"],
    )
    
    metadata_text = cl.get_metadata_text()
    
    # Check that message was truncated
    assert "bytes truncated]" in metadata_text
    
    # Verify total size is reasonable (should be around 4KB for message + small overhead)
    metadata_bytes = len(metadata_text.encode('utf-8'))
    assert metadata_bytes < 6000  # Should be well under 6KB


def test_get_metadata_text_files_truncation():
    """Test that file lists are truncated at 4KB."""
    # Create a list with many files that exceeds 4KB
    long_filename = "src/very/long/path/to/file_with_long_name_" + "x" * 50 + ".py"
    many_files = [f"{long_filename}_{i}" for i in range(100)]
    
    cl = Changelist(
        id="test456",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Short message",
        diff="diff --git a/test.py...",
        files=many_files,
    )
    
    metadata_text = cl.get_metadata_text()
    
    # Check that files were truncated
    assert "files omitted]" in metadata_text
    
    # Count included files - should be less than total
    files_section = metadata_text.split("Files: ")[1].split("\n")[0]
    included_count = files_section.count(',') + 1
    assert included_count < len(many_files)
    
    # Verify file list size is within limit
    files_bytes = files_section.encode('utf-8')
    # Should be around 4KB plus marker text
    assert len(files_bytes) < 5000


def test_get_metadata_text_both_truncated():
    """Test that both message and files can be truncated simultaneously."""
    large_message = "Y" * 5000  # 5KB message
    long_filename = "src/another/very/long/path/" + "z" * 60 + ".py"
    many_files = [f"{long_filename}_{i}" for i in range(100)]
    
    cl = Changelist(
        id="test789",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message=large_message,
        diff="diff --git a/test.py...",
        files=many_files,
    )
    
    metadata_text = cl.get_metadata_text()
    
    # Both should be truncated
    assert "bytes truncated]" in metadata_text
    assert "files omitted]" in metadata_text
    
    # Total metadata should be within reasonable bounds (around 8KB + overhead)
    metadata_bytes = len(metadata_text.encode('utf-8'))
    assert metadata_bytes < 10000  # Should be well under 10KB


def test_get_metadata_text_small_data_no_truncation():
    """Test that small messages and file lists are not truncated."""
    small_message = "This is a normal commit message"
    small_files = ["src/file1.py", "src/file2.py", "tests/test_file.py"]
    
    cl = Changelist(
        id="test000",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message=small_message,
        diff="diff --git a/test.py...",
        files=small_files,
        review_comments="LGTM",
    )
    
    metadata_text = cl.get_metadata_text()
    
    # No truncation markers
    assert "bytes truncated]" not in metadata_text
    assert "files omitted]" not in metadata_text
    
    # All files should be present
    for file in small_files:
        assert file in metadata_text
    
    # Review comments should be present
    assert "LGTM" in metadata_text


def test_get_metadata_text_custom_limits():
    """Test that custom limits can be specified."""
    message = "X" * 2000  # 2KB
    files = [f"file_{i}.py" for i in range(50)]
    
    cl = Changelist(
        id="test_custom",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message=message,
        diff="diff --git a/test.py...",
        files=files,
    )
    
    # Use custom smaller limits
    metadata_text = cl.get_metadata_text(
        max_files=10,
        max_message_bytes=1000,
        max_files_bytes=500
    )
    
    # Both should be truncated with custom limits
    assert "bytes truncated]" in metadata_text
    assert "files omitted]" in metadata_text


def test_get_metadata_text_exactly_at_limit():
    """Test edge case where content is exactly at the limit."""
    # Create message exactly 4096 bytes
    message = "A" * 4096
    
    cl = Changelist(
        id="test_exact",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message=message,
        diff="diff --git a/test.py...",
        files=["test.py"],
    )
    
    metadata_text = cl.get_metadata_text()
    
    # Should not be truncated at exactly 4096 bytes
    assert "bytes truncated]" not in metadata_text
    assert message in metadata_text


def test_get_metadata_text_empty_files():
    """Test that empty files list doesn't cause issues."""
    cl = Changelist(
        id="test_empty",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Commit with no files",
        diff="diff --git a/test.py...",
        files=[],
    )
    
    metadata_text = cl.get_metadata_text()
    
    # Should handle empty files gracefully
    assert "Files: " in metadata_text
    assert "Commit with no files" in metadata_text