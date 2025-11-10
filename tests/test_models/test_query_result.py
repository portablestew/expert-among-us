"""Tests for QueryResult abstraction."""

import pytest
from datetime import datetime

from expert_among_us.models.changelist import Changelist
from expert_among_us.models.file_chunk import FileChunk
from expert_among_us.models.query_result import (
    CommitResult,
    FileChunkResult,
    QueryResult,
    QueryResultBase,
)


def test_commit_result_interface():
    """Test CommitResult implements QueryResultBase interface."""
    cl = Changelist(
        id="abc123def456",
        expert_name="test",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        author="alice",
        message="Add new feature",
        diff="diff --git a/file.py...",
        files=["file1.py", "file2.py"]
    )
    
    result = CommitResult(
        changelist=cl,
        similarity_score=0.85,
        source="metadata"
    )
    
    # Test interface methods
    assert result.get_id() == "abc123def456"
    assert result.get_author() == "alice"
    assert result.get_timestamp() == cl.timestamp
    assert result.get_result_type() == "commit"
    assert "Add new feature" in result.get_display_title()
    assert "Add new feature" in result.get_preview_text()
    
    # Test commit-specific methods
    assert result.get_files() == ["file1.py", "file2.py"]
    assert "diff --git" in result.get_diff()


def test_commit_result_preview_truncation():
    """Test CommitResult truncates long messages in preview."""
    long_message = "A" * 300
    cl = Changelist(
        id="abc123",
        expert_name="test",
        timestamp=datetime.now(),
        author="bob",
        message=long_message,
        diff="test diff",
        files=["test.py"]
    )
    
    result = CommitResult(
        changelist=cl,
        similarity_score=0.75,
        source="diff"
    )
    
    # Preview should be truncated
    preview = result.get_preview_text(max_len=200)
    assert len(preview) <= 200
    assert preview.endswith("...")


def test_file_chunk_result_interface():
    """Test FileChunkResult implements QueryResultBase interface."""
    chunk = FileChunk(
        file_path="src/main.py",
        chunk_index=0,
        content="def main():\n    pass",
        line_start=1,
        line_end=2,
        revision_id="abc123def456"
    )
    
    result = FileChunkResult(
        file_chunk=chunk,
        similarity_score=0.75,
        source="file"
    )
    
    # Test interface methods
    assert "src/main.py" in result.get_id()
    assert result.get_author() is None
    assert result.get_timestamp() is None
    assert result.get_result_type() == "file_chunk"
    assert "src/main.py" in result.get_display_title()
    assert "lines 1-2" in result.get_display_title()
    assert "def main()" in result.get_preview_text()
    
    # Test file-specific methods
    assert result.get_file_path() == "src/main.py"
    assert result.get_line_range() == (1, 2)
    assert result.get_content() == "def main():\n    pass"
    assert result.get_revision_id() == "abc123def456"


def test_file_chunk_result_preview_truncation():
    """Test FileChunkResult truncates long content in preview."""
    long_content = "x" * 300
    chunk = FileChunk(
        file_path="test.py",
        chunk_index=0,
        content=long_content,
        line_start=1,
        line_end=50,
        revision_id="abc123"
    )
    
    result = FileChunkResult(
        file_chunk=chunk,
        similarity_score=0.65,
        source="file"
    )
    
    # Preview should be truncated
    preview = result.get_preview_text(max_len=200)
    assert len(preview) <= 200
    assert preview.endswith("...")


def test_polymorphic_handling():
    """Test handling mixed result types in a list."""
    # Create mixed results
    commit_cl = Changelist(
        id="commit123",
        expert_name="test",
        timestamp=datetime.now(),
        author="alice",
        message="Fix bug",
        diff="test diff",
        files=["fix.py"]
    )
    
    file_chunk = FileChunk(
        file_path="src/lib.py",
        chunk_index=0,
        content="def helper(): pass",
        line_start=10,
        line_end=11,
        revision_id="file123"
    )
    
    results: list[QueryResult] = [
        CommitResult(changelist=commit_cl, similarity_score=0.9, source="metadata"),
        FileChunkResult(file_chunk=file_chunk, similarity_score=0.8, source="file"),
        CommitResult(changelist=commit_cl, similarity_score=0.7, source="diff"),
    ]
    
    # Test common interface works for all types
    for result in results:
        assert isinstance(result, QueryResultBase)
        assert isinstance(result.get_id(), str)
        assert isinstance(result.similarity_score, float)
        assert result.source in ["metadata", "diff", "file", "combined"]
        assert isinstance(result.get_display_title(), str)
        assert isinstance(result.get_preview_text(), str)
    
    # Test type-specific access
    commit_count = 0
    file_count = 0
    
    for result in results:
        if isinstance(result, CommitResult):
            commit_count += 1
            assert hasattr(result, 'get_diff')
            assert hasattr(result, 'get_files')
            assert isinstance(result.get_files(), list)
        elif isinstance(result, FileChunkResult):
            file_count += 1
            assert hasattr(result, 'get_line_range')
            assert hasattr(result, 'get_content')
            assert isinstance(result.get_content(), str)
    
    assert commit_count == 2
    assert file_count == 1


def test_commit_result_with_chroma_id():
    """Test CommitResult with optional chroma_id."""
    cl = Changelist(
        id="test123",
        expert_name="test",
        timestamp=datetime.now(),
        author="bob",
        message="Test commit",
        diff="test",
        files=["test.py"]
    )
    
    result = CommitResult(
        changelist=cl,
        similarity_score=0.8,
        source="metadata",
        chroma_id="metadata:test123"
    )
    
    assert result.chroma_id == "metadata:test123"


def test_file_chunk_result_with_chroma_id():
    """Test FileChunkResult with optional chroma_id."""
    chunk = FileChunk(
        file_path="test.py",
        chunk_index=0,
        content="test content",
        line_start=1,
        line_end=5,
        revision_id="abc123"
    )
    
    result = FileChunkResult(
        file_chunk=chunk,
        similarity_score=0.7,
        source="file",
        chroma_id="file:test.py:chunk_0"
    )
    
    assert result.chroma_id == "file:test.py:chunk_0"


def test_file_chunk_result_empty_content():
    """Test FileChunkResult handles empty content gracefully."""
    chunk = FileChunk(
        file_path="empty.py",
        chunk_index=0,
        content="",
        line_start=1,
        line_end=1,
        revision_id="abc123"
    )
    
    result = FileChunkResult(
        file_chunk=chunk,
        similarity_score=0.3,
        source="file"
    )
    
    # Should not crash
    assert result.get_preview_text() == ""
    assert result.get_content() == ""


def test_result_type_literals():
    """Test result type returns correct literals."""
    commit_cl = Changelist(
        id="test",
        expert_name="test",
        timestamp=datetime.now(),
        author="test",
        message="test",
        diff="test",
        files=[]
    )
    
    file_chunk = FileChunk(
        file_path="test.py",
        chunk_index=0,
        content="test",
        line_start=1,
        line_end=1,
        revision_id="test"
    )
    
    commit_result = CommitResult(
        changelist=commit_cl,
        similarity_score=0.5,
        source="metadata"
    )
    
    file_result = FileChunkResult(
        file_chunk=file_chunk,
        similarity_score=0.5,
        source="file"
    )
    
    # Test literal types
    assert commit_result.get_result_type() == "commit"
    assert file_result.get_result_type() == "file_chunk"


def test_similarity_scores():
    """Test similarity scores are properly stored."""
    cl = Changelist(
        id="test",
        expert_name="test",
        timestamp=datetime.now(),
        author="test",
        message="test",
        diff="test",
        files=[]
    )
    
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for score in scores:
        result = CommitResult(
            changelist=cl,
            similarity_score=score,
            source="metadata"
        )
        assert result.similarity_score == score


def test_source_types():
    """Test different source types are properly stored."""
    cl = Changelist(
        id="test",
        expert_name="test",
        timestamp=datetime.now(),
        author="test",
        message="test",
        diff="test",
        files=[]
    )
    
    sources = ["metadata", "diff", "file", "combined"]
    
    for source in sources:
        result = CommitResult(
            changelist=cl,
            similarity_score=0.5,
            source=source  # type: ignore
        )
        assert result.source == source