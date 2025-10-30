"""Tests for QueryParams and VectorSearchResult models."""

import pytest
from pydantic import ValidationError

from expert_among_us.models.query import QueryParams, VectorSearchResult


def test_query_params_creation():
    """Test basic query params creation."""
    params = QueryParams(
        prompt="How to add a new feature?",
        max_changes=15,
        users=["user1", "user2"],
        files=["file1.py", "file2.py"],
    )
    
    assert params.prompt == "How to add a new feature?"
    assert params.max_changes == 15
    assert params.users == ["user1", "user2"]
    assert params.files == ["file1.py", "file2.py"]


def test_query_params_defaults():
    """Test default values."""
    params = QueryParams(prompt="Test query")
    
    assert params.max_changes == 10
    assert params.users is None
    assert params.files is None
    assert params.amogus is False
    assert params.temperature == 0.7


def test_query_params_validation_empty_prompt():
    """Test that empty prompt raises validation error."""
    with pytest.raises(ValidationError):
        QueryParams(prompt="")
    
    with pytest.raises(ValidationError):
        QueryParams(prompt="   ")


def test_query_params_validation_max_changes_range():
    """Test max_changes range validation."""
    # Valid range
    params = QueryParams(prompt="Test", max_changes=1)
    assert params.max_changes == 1
    
    params = QueryParams(prompt="Test", max_changes=100)
    assert params.max_changes == 100
    
    # Below minimum
    with pytest.raises(ValidationError):
        QueryParams(prompt="Test", max_changes=0)
    
    # Above maximum
    with pytest.raises(ValidationError):
        QueryParams(prompt="Test", max_changes=101)


def test_query_params_validation_temperature_range():
    """Test temperature range validation."""
    # Valid range
    params = QueryParams(prompt="Test", temperature=0.0)
    assert params.temperature == 0.0
    
    params = QueryParams(prompt="Test", temperature=1.0)
    assert params.temperature == 1.0
    
    # Below minimum
    with pytest.raises(ValidationError):
        QueryParams(prompt="Test", temperature=-0.1)
    
    # Above maximum
    with pytest.raises(ValidationError):
        QueryParams(prompt="Test", temperature=1.1)


def test_query_params_prompt_trimming():
    """Test that prompt is trimmed."""
    params = QueryParams(prompt="  Test query  ")
    assert params.prompt == "Test query"


def test_vector_search_result_creation():
    """Test basic vector search result creation."""
    result = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.85,
        source="metadata",
    )
    
    assert result.changelist_id == "abc123"
    assert result.similarity_score == 0.85
    assert result.source == "metadata"


def test_vector_search_result_defaults():
    """Test default source value."""
    result = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.85,
    )
    
    assert result.source == "metadata"


def test_vector_search_result_validation_score_range():
    """Test similarity score range validation."""
    # Valid range
    result = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.0,
    )
    assert result.similarity_score == 0.0
    
    result = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=1.0,
    )
    assert result.similarity_score == 1.0
    
    # Below minimum
    with pytest.raises(ValidationError):
        VectorSearchResult(
            changelist_id="abc123",
            similarity_score=-0.1,
        )
    
    # Above maximum
    with pytest.raises(ValidationError):
        VectorSearchResult(
            changelist_id="abc123",
            similarity_score=1.1,
        )


def test_vector_search_result_validation_source():
    """Test source validation (must be 'metadata' or 'diff')."""
    # Valid sources
    result = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.85,
        source="metadata",
    )
    assert result.source == "metadata"
    
    result = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.85,
        source="diff",
    )
    assert result.source == "diff"
    
    # Invalid source
    with pytest.raises(ValidationError):
        VectorSearchResult(
            changelist_id="abc123",
            similarity_score=0.85,
            source="invalid",  # type: ignore
        )


def test_vector_search_result_sorting():
    """Test that results can be sorted by similarity score (descending)."""
    result1 = VectorSearchResult(changelist_id="a", similarity_score=0.9)
    result2 = VectorSearchResult(changelist_id="b", similarity_score=0.7)
    result3 = VectorSearchResult(changelist_id="c", similarity_score=0.85)
    
    results = [result2, result3, result1]
    sorted_results = sorted(results)
    
    # Should be sorted by score descending (highest first)
    assert sorted_results[0].changelist_id == "a"  # 0.9
    assert sorted_results[1].changelist_id == "c"  # 0.85
    assert sorted_results[2].changelist_id == "b"  # 0.7


def test_vector_search_result_equality():
    """Test equality based on changelist_id and source."""
    result1 = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.9,
        source="metadata",
    )
    result2 = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.85,  # Different score
        source="metadata",
    )
    result3 = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.9,
        source="diff",  # Different source
    )
    
    # Same ID and source = equal (even with different scores)
    assert result1 == result2
    
    # Different source = not equal
    assert result1 != result3


def test_vector_search_result_hashable():
    """Test that results can be used in sets/dicts."""
    result1 = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.9,
        source="metadata",
    )
    result2 = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.85,
        source="metadata",
    )
    result3 = VectorSearchResult(
        changelist_id="abc123",
        similarity_score=0.9,
        source="diff",
    )
    
    # Can create a set
    result_set = {result1, result2, result3}
    
    # result1 and result2 are equal, so only 2 unique items
    assert len(result_set) == 2