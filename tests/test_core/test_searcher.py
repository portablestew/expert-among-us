"""Tests for the search engine module."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from expert_among_us.core.searcher import Searcher
from expert_among_us.models.query_result import CommitResult, FileChunkResult, QueryResultBase
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
        
        # Create all possible changelists
        all_changelists = {
            "abc123": Changelist(
                id="abc123",
                expert_name="TestExpert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="john",
                message="Add new feature",
                diff="diff content",
                files=["src/main.py", "src/utils.py"]
            ),
            "def456": Changelist(
                id="def456",
                expert_name="TestExpert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="jane",
                message="Fix bug in parser",
                diff="diff content 2",
                files=["src/parser.py"]
            )
        }
        
        # Mock get_changelists_by_ids to properly filter based on input IDs
        def mock_get_changelists_by_ids(ids):
            return [all_changelists[id] for id in ids if id in all_changelists]
        
        db.get_changelists_by_ids.side_effect = mock_get_changelists_by_ids
        return db
    
    @pytest.fixture
    def mock_vector_db(self):
        """Mock vector database."""
        db = Mock()
        db.search_metadata.return_value = [
            VectorSearchResult(result_id="abc123", similarity_score=0.95),
            VectorSearchResult(result_id="def456", similarity_score=0.85)
        ]
        db.search_diffs.return_value = [
            VectorSearchResult(result_id="abc123", similarity_score=0.90)
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
            enable_diff_search=True,
            enable_query_expansion=False  # Disable expansion for existing tests to maintain compatibility
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

        # New abstraction: all results must implement QueryResultBase
        assert all(isinstance(r, QueryResultBase) for r in results)

        # At least one commit-style result with attached changelist data
        assert any(isinstance(r, CommitResult) for r in results)
        for r in results:
            if isinstance(r, CommitResult):
                assert r.changelist is not None
    
    def test_merge_scores_metadata_only(self, searcher):
        """Test score merging with metadata results only."""
        metadata_results = [
            VectorSearchResult(result_id="abc123", similarity_score=0.95),
            VectorSearchResult(result_id="def456", similarity_score=0.85)
        ]
        diff_results = []
        
        merged = searcher._merge_commit_scores(metadata_results, diff_results)
        
        assert len(merged) == 2
        assert merged["abc123"]["score"] == 0.95
        assert merged["def456"]["score"] == 0.85
    
    def test_merge_scores_combined(self, searcher):
        """Test score merging with both metadata and diff results."""
        metadata_results = [
            VectorSearchResult(result_id="abc123", similarity_score=0.90),
            VectorSearchResult(result_id="def456", similarity_score=0.80)
        ]
        diff_results = [
            VectorSearchResult(result_id="abc123", similarity_score=0.85)
        ]
        
        merged = searcher._merge_commit_scores(metadata_results, diff_results)
        
        # abc123 should have combined score: 0.90*0.6 + 0.85*0.4 = 0.54 + 0.34 = 0.88
        assert len(merged) == 2
        assert abs(merged["abc123"]["score"] - 0.88) < 0.01
        
        # def456 should only have metadata score
        assert merged["def456"]["score"] == 0.80
    
    def test_apply_filters_users(self, searcher, mock_metadata_db):
        """Test filtering by user."""
        changelists = mock_metadata_db.get_changelists_by_ids(["abc123", "def456"])
        scores = {
            "abc123": {"score": 0.95, "source": "metadata"},
            "def456": {"score": 0.85, "source": "metadata"}
        }
        params = QueryParams(prompt="test", max_changes=10, users=["john"])
        
        results = searcher._apply_commit_filters(changelists, scores, params)
        
        # Should only include john's changelist
        assert len(results) == 1
        assert results[0].changelist.author == "john"
    
    def test_apply_filters_files(self, searcher, mock_metadata_db):
        """Test filtering by files."""
        changelists = mock_metadata_db.get_changelists_by_ids(["abc123", "def456"])
        scores = {
            "abc123": {"score": 0.95, "source": "metadata"},
            "def456": {"score": 0.85, "source": "metadata"}
        }
        params = QueryParams(prompt="test", max_changes=10, files=["src/parser.py"])
        
        results = searcher._apply_commit_filters(changelists, scores, params)
        
        # Should only include changelist affecting parser.py
        assert len(results) == 1
        assert "src/parser.py" in results[0].changelist.files
    
    def test_apply_filters_no_match(self, searcher, mock_metadata_db):
        """Test filtering with no matches."""
        changelists = mock_metadata_db.get_changelists_by_ids(["abc123", "def456"])
        scores = {
            "abc123": {"score": 0.95, "source": "metadata"},
            "def456": {"score": 0.85, "source": "metadata"}
        }
        params = QueryParams(prompt="test", max_changes=10, users=["nonexistent"])
        
        results = searcher._apply_commit_filters(changelists, scores, params)
        
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
    """Tests for legacy SearchResult behavior, now mapped onto CommitResult."""
    
    def test_create_search_result(self):
        """Test creating a commit-style search result."""
        changelist = Changelist(
            id="abc123",
            expert_name="TestExpert",
            project_name="test-project",
            timestamp=datetime.now(),
            author="john",
            message="Test change",
            diff="diff content",
            files=["test.py"]
        )
        
        # Legacy SearchResult dataclass is replaced by CommitResult
        result = CommitResult(
            changelist=changelist,
            similarity_score=0.95,
            source="metadata"
        )
        
        assert result.changelist == changelist
        assert result.similarity_score == 0.95
        assert result.source == "metadata"
    
    def test_search_result_sorting(self):
        """Test that commit-style results can be sorted by score."""
        cl1 = Changelist(
            id="1", expert_name="Test", project_name="test-project", timestamp=datetime.now(),
            author="john", message="m1", diff="d1", files=["f1"]
        )
        cl2 = Changelist(
            id="2", expert_name="Test", project_name="test-project", timestamp=datetime.now(),
            author="jane", message="m2", diff="d2", files=["f2"]
        )
        
        results = [
            CommitResult(changelist=cl1, similarity_score=0.85, source="metadata"),
            CommitResult(changelist=cl2, similarity_score=0.95, source="diff")
        ]
        
        sorted_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
        assert sorted_results[0].similarity_score == 0.95
        assert sorted_results[1].similarity_score == 0.85



class TestBuildWhereClause:
    """Test cases for the Searcher._build_where_clause method."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = Mock()
        embedder.embed.return_value = [0.1] * 1024
        return embedder

    @pytest.fixture
    def mock_vector_db(self):
        db = Mock()
        db.search_metadata.return_value = []
        db.search_diffs.return_value = []
        return db

    def _make_searcher(self, mock_embedder, mock_vector_db, projects, has_vector_metadata=True):
        """Helper to create a Searcher with specified known projects."""
        mock_metadata_db = Mock()
        mock_metadata_db.list_projects.return_value = [{"name": p} for p in projects]
        return Searcher(
            expert_name="TestExpert",
            embedder=mock_embedder,
            metadata_db=mock_metadata_db,
            vector_db=mock_vector_db,
            enable_query_expansion=False,
            has_vector_metadata=has_vector_metadata
        )

    def test_returns_none_when_has_vector_metadata_false(self, mock_embedder, mock_vector_db):
        """Legacy experts with has_vector_metadata=False should never produce a where clause."""
        searcher = self._make_searcher(mock_embedder, mock_vector_db, ["payment-service"], has_vector_metadata=False)
        result = searcher._build_where_clause(["payment-service/src/handler.py"])
        assert result is None

    def test_returns_none_when_files_is_none(self, mock_embedder, mock_vector_db):
        """No files parameter means no where clause."""
        searcher = self._make_searcher(mock_embedder, mock_vector_db, ["payment-service"])
        result = searcher._build_where_clause(None)
        assert result is None

    def test_returns_none_when_files_is_empty(self, mock_embedder, mock_vector_db):
        """Empty files list means no where clause."""
        searcher = self._make_searcher(mock_embedder, mock_vector_db, ["payment-service"])
        result = searcher._build_where_clause([])
        assert result is None

    def test_returns_none_when_no_prefix_matches_known_projects(self, mock_embedder, mock_vector_db):
        """Files with prefixes not matching any known project should produce None."""
        searcher = self._make_searcher(mock_embedder, mock_vector_db, ["payment-service", "user-service"])
        result = searcher._build_where_clause(["unknown-project/src/file.py"])
        assert result is None

    def test_extracts_single_project(self, mock_embedder, mock_vector_db):
        """Files matching a single known project produce the correct where clause."""
        searcher = self._make_searcher(mock_embedder, mock_vector_db, ["payment-service", "user-service"])
        result = searcher._build_where_clause(["payment-service/src/handler.py"])
        assert result is not None
        assert result["project"]["$in"] == ["payment-service"]

    def test_extracts_multiple_projects(self, mock_embedder, mock_vector_db):
        """Files from multiple known projects produce a multi-project where clause."""
        searcher = self._make_searcher(mock_embedder, mock_vector_db, ["payment-service", "user-service", "shared-lib"])
        result = searcher._build_where_clause([
            "payment-service/src/handler.py",
            "user-service/src/auth.py"
        ])
        assert result is not None
        assert set(result["project"]["$in"]) == {"payment-service", "user-service"}

    def test_project_prefix_only_with_trailing_slash(self, mock_embedder, mock_vector_db):
        """A bare project prefix like 'payment-service/' should still extract the project."""
        searcher = self._make_searcher(mock_embedder, mock_vector_db, ["payment-service"])
        result = searcher._build_where_clause(["payment-service/"])
        assert result is not None
        assert result["project"]["$in"] == ["payment-service"]

    def test_ignores_non_project_files_in_mixed_list(self, mock_embedder, mock_vector_db):
        """Only known project prefixes should appear in where clause, others ignored."""
        searcher = self._make_searcher(mock_embedder, mock_vector_db, ["payment-service"])
        result = searcher._build_where_clause([
            "payment-service/src/handler.py",
            "nonexistent/src/other.py"
        ])
        assert result is not None
        assert result["project"]["$in"] == ["payment-service"]

    def test_known_projects_fetched_at_construction(self, mock_embedder, mock_vector_db):
        """Verify the constructor fetches known projects from metadata_db."""
        mock_metadata_db = Mock()
        mock_metadata_db.list_projects.return_value = [
            {"name": "proj-a"},
            {"name": "proj-b"},
        ]
        searcher = Searcher(
            expert_name="TestExpert",
            embedder=mock_embedder,
            metadata_db=mock_metadata_db,
            vector_db=mock_vector_db,
            enable_query_expansion=False,
        )
        mock_metadata_db.list_projects.assert_called_once_with("TestExpert")
        assert searcher.known_projects == {"proj-a", "proj-b"}

    def test_constructor_graceful_with_no_list_projects(self, mock_embedder, mock_vector_db):
        """Constructor handles databases without list_projects gracefully."""
        mock_metadata_db = Mock(spec=[])  # No methods at all
        searcher = Searcher(
            expert_name="TestExpert",
            embedder=mock_embedder,
            metadata_db=mock_metadata_db,
            vector_db=mock_vector_db,
            enable_query_expansion=False,
        )
        assert searcher.known_projects == set()

    def test_has_vector_metadata_defaults_to_true(self, mock_embedder, mock_vector_db):
        """has_vector_metadata should default to True."""
        mock_metadata_db = Mock()
        mock_metadata_db.list_projects.return_value = []
        searcher = Searcher(
            expert_name="TestExpert",
            embedder=mock_embedder,
            metadata_db=mock_metadata_db,
            vector_db=mock_vector_db,
            enable_query_expansion=False,
        )
        assert searcher.has_vector_metadata is True


class TestMergeDuplicateCommits:
    """Tests for cross-project duplicate commit merging.

    A single P4 changelist that touches multiple subdir-scoped projects is
    indexed as distinct rows ('Code/12345', 'Gems/12345') sharing a raw id.
    These must collapse to one combined result with no data loss.
    """

    @pytest.fixture
    def searcher(self, mock_embedder, mock_vector_db):
        mock_metadata_db = Mock()
        mock_metadata_db.list_projects.return_value = []
        return Searcher(
            expert_name="TestExpert",
            embedder=mock_embedder,
            metadata_db=mock_metadata_db,
            vector_db=mock_vector_db,
            enable_query_expansion=False,
        )

    @pytest.fixture
    def mock_embedder(self):
        embedder = Mock()
        embedder.embed.return_value = [0.1] * 1024
        embedder.dimension = 1024
        return embedder

    @pytest.fixture
    def mock_vector_db(self):
        return Mock()

    @staticmethod
    def _commit(project, raw_id, files, diff, score, source="metadata",
                message="Add feature", author="john", ts=None):
        ts = ts or datetime(2024, 1, 15, 10, 30, 0)
        cl = Changelist(
            id=f"{project}/{raw_id}",
            expert_name="TestExpert",
            project_name=project,
            timestamp=ts,
            author=author,
            message=message,
            diff=diff,
            files=files,
        )
        return CommitResult(changelist=cl, similarity_score=score, source=source)

    def test_raw_id_strips_project_prefix(self):
        cl = Changelist(
            id="Code/12345", expert_name="E", project_name="Code",
            timestamp=datetime.now(), author="a", message="m", diff="d",
            files=["Code/x.py"],
        )
        assert Searcher._raw_changelist_id(cl) == "12345"

    def test_raw_id_unchanged_when_no_prefix(self):
        cl = Changelist(
            id="abcdef", expert_name="E", project_name="Code",
            timestamp=datetime.now(), author="a", message="m", diff="d",
            files=["Code/x.py"],
        )
        # id doesn't start with "Code/", so it is returned unchanged
        assert Searcher._raw_changelist_id(cl) == "abcdef"

    def test_cross_project_duplicates_merge(self, searcher):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        results = [
            self._commit("Code", "12345", ["Code/a.py"], "diff-code", 0.91, "metadata", ts=ts),
            self._commit("Gems", "12345", ["Gems/b.py"], "diff-gems", 0.74, "diff", ts=ts),
        ]
        merged = searcher._merge_duplicate_commits(results)

        assert len(merged) == 1
        cl = merged[0].changelist
        # Union of files, no loss
        assert cl.files == ["Code/a.py", "Gems/b.py"]
        # Both diffs preserved under labeled headers
        assert "diff-code" in cl.diff
        assert "diff-gems" in cl.diff
        assert "project: Code" in cl.diff
        assert "project: Gems" in cl.diff
        # Provenance in id/project_name
        assert cl.id == "Code+Gems/12345"
        assert cl.project_name == "Code+Gems"
        # Max score wins
        assert merged[0].similarity_score == 0.91

    def test_single_commit_passes_through_unchanged(self, searcher):
        results = [self._commit("Code", "999", ["Code/a.py"], "d", 0.5)]
        merged = searcher._merge_duplicate_commits(results)
        assert len(merged) == 1
        assert merged[0].changelist.id == "Code/999"
        assert merged[0].changelist.project_name == "Code"

    def test_distinct_raw_ids_not_merged(self, searcher):
        results = [
            self._commit("Code", "111", ["Code/a.py"], "d1", 0.9),
            self._commit("Code", "222", ["Code/b.py"], "d2", 0.8),
        ]
        merged = searcher._merge_duplicate_commits(results)
        assert len(merged) == 2

    def test_merge_preserves_highest_ranked_position(self, searcher):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        # Order: rank0 distinct, rank1 Code/12345, rank2 distinct, rank3 Gems/12345
        results = [
            self._commit("Code", "100", ["Code/x.py"], "dx", 0.95),
            self._commit("Code", "12345", ["Code/a.py"], "diff-code", 0.90, ts=ts),
            self._commit("Code", "200", ["Code/y.py"], "dy", 0.80),
            self._commit("Gems", "12345", ["Gems/b.py"], "diff-gems", 0.50, ts=ts),
        ]
        merged = searcher._merge_duplicate_commits(results)
        ids = [r.changelist.id for r in merged]
        # The merged entry keeps the Code/12345 position (index 1), Gems dropped from index 3
        assert ids == ["Code/100", "Code+Gems/12345", "Code/200"]

    def test_merge_combines_review_comments(self, searcher):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        c1 = self._commit("Code", "12345", ["Code/a.py"], "d1", 0.9, ts=ts)
        c2 = self._commit("Gems", "12345", ["Gems/b.py"], "d2", 0.8, ts=ts)
        c1.changelist.review_comments = "LGTM code"
        c2.changelist.review_comments = "LGTM gems"
        merged = searcher._merge_duplicate_commits([c1, c2])
        assert "LGTM code" in merged[0].changelist.review_comments
        assert "LGTM gems" in merged[0].changelist.review_comments

    def test_source_combined_when_members_differ(self, searcher):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        results = [
            self._commit("Code", "12345", ["Code/a.py"], "d1", 0.9, source="metadata", ts=ts),
            self._commit("Gems", "12345", ["Gems/b.py"], "d2", 0.8, source="diff", ts=ts),
        ]
        merged = searcher._merge_duplicate_commits(results)
        assert merged[0].source == "combined"
