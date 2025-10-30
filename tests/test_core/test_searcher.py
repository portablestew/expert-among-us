"""Tests for the search engine module."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from expert_among_us.core.searcher import Searcher, SearchResult
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.query import QueryParams, VectorSearchResult


class TestSearcher:
    """Test cases for the Searcher class."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder that returns dummy embeddings."""
        embedder = Mock()
        embedder.embed.return_value = [0.1] * 1024
        embedder.embed_batch.return_value = [[0.1] * 1024, [0.2] * 1024]
        embedder.dimension = 1024
        return embedder
    
    @pytest.fixture
    def mock_metadata_db(self):
        """Mock metadata database."""
        db = Mock()
        db.get_changelists_by_ids.return_value = [
            Changelist(
                id="abc123",
                expert_name="TestExpert",
                timestamp=datetime.now(),
                author="john",
                message="Add new feature",
                diff="diff content",
                files=["src/main.py", "src/utils.py"]
            ),
            Changelist(
                id="def456",
                expert_name="TestExpert",
                timestamp=datetime.now(),
                author="jane",
                message="Fix bug in parser",
                diff="diff content 2",
                files=["src/parser.py"]
            )
        ]
        return db
    
    @pytest.fixture
    def mock_vector_db(self):
        """Mock vector database."""
        db = Mock()
        db.search_metadata.return_value = [
            VectorSearchResult(changelist_id="abc123", similarity_score=0.95, source="metadata"),
            VectorSearchResult(changelist_id="def456", similarity_score=0.85, source="metadata")
        ]
        db.search_diffs.return_value = [
            VectorSearchResult(changelist_id="abc123", similarity_score=0.90, source="diff")
        ]
        return db
    
    @pytest.fixture
    def searcher(self, mock_embedder, mock_metadata_db, mock_vector_db):
        """Create a searcher instance with mocked dependencies."""
        return Searcher(
            expert_name="TestExpert",
            embedder=mock_embedder,
            metadata_db=mock_metadata_db,
            vector_db=mock_vector_db,
            enable_diff_search=True
        )
    
    def test_search_basic(self, searcher, mock_embedder, mock_vector_db, mock_metadata_db):
        """Test basic search functionality."""
        params = QueryParams(prompt="How to add new feature?", max_changes=10)
        
        results = searcher.search(params)
        
        # Verify embedder was called
        mock_embedder.embed.assert_called_once_with("How to add new feature?")
        
        # Verify vector search was called
        mock_vector_db.search_metadata.assert_called_once()
        
        # Verify results
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.changelist is not None for r in results)
    
    def test_merge_scores_metadata_only(self, searcher):
        """Test score merging with metadata results only."""
        metadata_results = [
            VectorSearchResult(changelist_id="abc123", similarity_score=0.95, source="metadata"),
            VectorSearchResult(changelist_id="def456", similarity_score=0.85, source="metadata")
        ]
        diff_results = []
        
        merged = searcher._merge_scores(metadata_results, diff_results)
        
        assert len(merged) == 2
        assert merged["abc123"]["score"] == 0.95
        assert merged["abc123"]["source"] == "metadata"
        assert merged["def456"]["score"] == 0.85
        assert merged["def456"]["source"] == "metadata"
    
    def test_merge_scores_combined(self, searcher):
        """Test score merging with both metadata and diff results."""
        metadata_results = [
            VectorSearchResult(changelist_id="abc123", similarity_score=0.90, source="metadata"),
            VectorSearchResult(changelist_id="def456", similarity_score=0.80, source="metadata")
        ]
        diff_results = [
            VectorSearchResult(changelist_id="abc123", similarity_score=0.85, source="diff")
        ]
        
        merged = searcher._merge_scores(metadata_results, diff_results)
        
        # abc123 should have combined score: 0.90*0.6 + 0.85*0.4 = 0.54 + 0.34 = 0.88
        # Source should be "metadata" since metadata_score (0.90) > diff_score (0.85)
        assert len(merged) == 2
        assert merged["abc123"]["source"] == "metadata"
        assert abs(merged["abc123"]["score"] - 0.88) < 0.01
        
        # def456 should only have metadata score
        assert merged["def456"]["source"] == "metadata"
        assert merged["def456"]["score"] == 0.80
    
    def test_apply_filters_users(self, searcher, mock_metadata_db):
        """Test filtering by user."""
        changelists = mock_metadata_db.get_changelists_by_ids.return_value
        scores = {
            "abc123": {"score": 0.95, "source": "metadata"},
            "def456": {"score": 0.85, "source": "metadata"}
        }
        params = QueryParams(prompt="test", max_changes=10, users=["john"])
        
        results = searcher._apply_filters(changelists, scores, params)
        
        # Should only include john's changelist
        assert len(results) == 1
        assert results[0].changelist.author == "john"
    
    def test_apply_filters_files(self, searcher, mock_metadata_db):
        """Test filtering by files."""
        changelists = mock_metadata_db.get_changelists_by_ids.return_value
        scores = {
            "abc123": {"score": 0.95, "source": "metadata"},
            "def456": {"score": 0.85, "source": "metadata"}
        }
        params = QueryParams(prompt="test", max_changes=10, files=["src/parser.py"])
        
        results = searcher._apply_filters(changelists, scores, params)
        
        # Should only include changelist affecting parser.py
        assert len(results) == 1
        assert "src/parser.py" in results[0].changelist.files
    
    def test_apply_filters_no_match(self, searcher, mock_metadata_db):
        """Test filtering with no matches."""
        changelists = mock_metadata_db.get_changelists_by_ids.return_value
        scores = {
            "abc123": {"score": 0.95, "source": "metadata"},
            "def456": {"score": 0.85, "source": "metadata"}
        }
        params = QueryParams(prompt="test", max_changes=10, users=["nonexistent"])
        
        results = searcher._apply_filters(changelists, scores, params)
        
        # Should be empty
        assert len(results) == 0
    
    def test_search_with_filters(self, searcher, mock_embedder, mock_vector_db, mock_metadata_db):
        """Test search with user and file filters."""
        params = QueryParams(
            prompt="How to fix bug?",
            max_changes=5,
            users=["john"],
            files=["src/main.py"]
        )
        
        results = searcher.search(params)
        
        # Verify only filtered results returned
        assert all(r.changelist.author == "john" for r in results)
        assert all("src/main.py" in r.changelist.files for r in results)
    
    def test_search_respects_max_changes(self, searcher):
        """Test that search respects max_changes limit."""
        params = QueryParams(prompt="test query", max_changes=1)
        
        results = searcher.search(params)
        
        # Should return at most 1 result
        assert len(results) <= 1
    
    def test_close(self, searcher, mock_metadata_db, mock_vector_db):
        """Test cleanup of resources."""
        searcher.close()
        
        mock_metadata_db.close.assert_called_once()
        mock_vector_db.close.assert_called_once()


class TestSearchResult:
    """Test cases for SearchResult dataclass."""
    
    def test_create_search_result(self):
        """Test creating a search result."""
        changelist = Changelist(
            id="abc123",
            expert_name="TestExpert",
            timestamp=datetime.now(),
            author="john",
            message="Test change",
            diff="diff content",
            files=["test.py"]
        )
        
        result = SearchResult(
            changelist=changelist,
            similarity_score=0.95,
            source="metadata"
        )
        
        assert result.changelist == changelist
        assert result.similarity_score == 0.95
        assert result.source == "metadata"
    
    def test_search_result_sorting(self):
        """Test that search results can be sorted by score."""
        cl1 = Changelist(
            id="1", expert_name="Test", timestamp=datetime.now(),
            author="john", message="m1", diff="d1", files=["f1"]
        )
        cl2 = Changelist(
            id="2", expert_name="Test", timestamp=datetime.now(),
            author="jane", message="m2", diff="d2", files=["f2"]
        )
        
        results = [
            SearchResult(changelist=cl1, similarity_score=0.85, source="metadata"),
            SearchResult(changelist=cl2, similarity_score=0.95, source="diff")
        ]
        
        sorted_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
        assert sorted_results[0].similarity_score == 0.95
        assert sorted_results[1].similarity_score == 0.85