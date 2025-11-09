"""Tests for Changelist model."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from expert_among_us.models.changelist import Changelist


def test_changelist_creation():
    """Test basic changelist creation."""
    cl = Changelist(
        id="abc123",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Test commit",
        diff="diff --git a/test.py...",
        files=["test.py"],
    )
    
    assert cl.id == "abc123"
    assert cl.expert_name == "TestExpert"
    assert cl.author == "test_user"
    assert cl.message == "Test commit"
    assert len(cl.files) == 1


def test_changelist_validation_empty_id():
    """Test that empty ID raises validation error."""
    with pytest.raises(ValidationError):
        Changelist(
            id="",
            expert_name="TestExpert",
            timestamp=datetime.now(timezone.utc),
            author="test_user",
            message="Test commit",
            diff="diff --git a/test.py...",
            files=["test.py"],
        )


def test_changelist_validation_empty_message():
    """Test that empty message raises validation error."""
    with pytest.raises(ValidationError):
        Changelist(
            id="abc123",
            expert_name="TestExpert",
            timestamp=datetime.now(timezone.utc),
            author="test_user",
            message="",
            diff="diff --git a/test.py...",
            files=["test.py"],
        )


def test_changelist_allows_empty_files_list():
    """Empty files list is allowed for metadata-only or diff-only changelists."""
    cl = Changelist(
        id="abc123",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Test commit",
        diff="diff --git a/test.py...",
        files=[],
    )
    assert cl.files == []


def test_changelist_embedding_dimension_validation():
    """Test that embeddings must be 1024 dimensions."""
    # Valid 1024D embedding
    embedding = [0.1] * 1024
    cl = Changelist(
        id="abc123",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Test commit",
        diff="diff --git a/test.py...",
        files=["test.py"],
        metadata_embedding=embedding,
    )
    assert len(cl.metadata_embedding) == 1024
    
    # Invalid dimension
    with pytest.raises(ValidationError):
        Changelist(
            id="abc123",
            expert_name="TestExpert",
            timestamp=datetime.now(timezone.utc),
            author="test_user",
            message="Test commit",
            diff="diff --git a/test.py...",
            files=["test.py"],
            metadata_embedding=[0.1] * 512,  # Wrong dimension
        )


def test_changelist_get_metadata_text():
    """Test metadata text generation."""
    cl = Changelist(
        id="abc123",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Test commit",
        diff="diff --git a/test.py...",
        files=["test.py", "test2.py"],
        review_comments="LGTM",
    )
    
    metadata_text = cl.get_metadata_text()
    
    assert "Test commit" in metadata_text
    assert "test.py" in metadata_text
    assert "test2.py" in metadata_text
    assert "LGTM" in metadata_text


def test_changelist_get_truncated_diff():
    """Test diff truncation."""
    # Small diff - no truncation
    small_diff = "a" * 100
    cl = Changelist(
        id="abc123",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Test commit",
        diff=small_diff,
        files=["test.py"],
    )
    
    truncated = cl.get_truncated_diff(max_size=200)
    assert truncated == small_diff
    assert "[TRUNCATED" not in truncated
    
    # Large diff - truncation
    large_diff = "a" * 200
    cl_large = Changelist(
        id="abc123_large",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Test large commit",
        diff=large_diff,
        files=["large_test.py"],
    )
    truncated = cl_large.get_truncated_diff(max_size=100)
    assert len(truncated.encode("utf-8")) < 200
    assert "[TRUNCATED" in truncated


def test_changelist_to_dict_from_dict():
    """Test serialization and deserialization."""
    cl = Changelist(
        id="abc123",
        expert_name="TestExpert",
        timestamp=datetime.now(timezone.utc),
        author="test_user",
        message="Test commit",
        diff="diff --git a/test.py...",
        files=["test.py"],
    )
    
    # Convert to dict
    cl_dict = cl.to_dict()
    assert cl_dict["id"] == "abc123"
    assert cl_dict["author"] == "test_user"
    
    # Convert back from dict
    cl2 = Changelist.from_dict(cl_dict)
    assert cl2.id == cl.id
    assert cl2.author == cl.author
    assert cl2.message == cl.message