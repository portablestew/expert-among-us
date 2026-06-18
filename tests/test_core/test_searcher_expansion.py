"""Tests for Progressive Centroid Query Expansion feature."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
from expert_among_us.core.searcher import Searcher
from expert_among_us.models.query import QueryParams, VectorSearchResult
from expert_among_us.models.query_result import CommitResult, FileChunkResult
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.file_chunk import FileChunk
from expert_among_us.embeddings.base import Embedder
from expert_among_us.db.metadata.base import MetadataDB
from expert_among_us.db.vector.base import VectorDB
from expert_among_us.reranking.base import Reranker


class TestQueryExpansion:
    """Test suite for Progressive Centroid Query Expansion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedder = Mock(spec=Embedder)
        self.mock_metadata_db = Mock(spec=MetadataDB)
        self.mock_vector_db = Mock(spec=VectorDB)
        self.mock_reranker = Mock(spec=Reranker)
        
        # Configure mock embedder
        self.mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        
        # Configure mock vector DB with search methods
        self.mock_vector_db.search_metadata.return_value = []
        self.mock_vector_db.search_diffs.return_value = []
        self.mock_vector_db.search_files.return_value = []
        
        # Create searcher with expansion enabled
        self.searcher = Searcher(
            expert_name="test_expert",
            embedder=self.mock_embedder,
            metadata_db=self.mock_metadata_db,
            vector_db=self.mock_vector_db,
            reranker=self.mock_reranker,
            enable_query_expansion=True,
            expansion_std_threshold=1.0,
            expansion_min_anchors=3
        )

    def create_mock_commit_result(self, commit_id: str, score: float, source: str = "metadata") -> CommitResult:
        """Create a mock CommitResult for testing."""
        mock_changelist = Mock(spec=Changelist)
        mock_changelist.id = commit_id
        mock_changelist.project_name = "test-project"
        mock_changelist.message = f"Commit {commit_id}"
        mock_changelist.author = "test_author"
        mock_changelist.timestamp = None
        mock_changelist.files = []
        mock_changelist.diff = ""
        
        return CommitResult(
            changelist=mock_changelist,
            similarity_score=score,
            source=source,
            embedding=[0.1, 0.2, 0.3]  # Add embedding for expansion tests
        )

    def create_mock_file_result(self, file_path: str, score: float) -> FileChunkResult:
        """Create a mock FileChunkResult for testing."""
        mock_file_chunk = Mock(spec=FileChunk)
        mock_file_chunk.file_path = file_path
        mock_file_chunk.content = f"Content for {file_path}"
        mock_file_chunk.line_start = 1
        mock_file_chunk.line_end = 10
        mock_file_chunk.revision_id = "abc123"
        
        return FileChunkResult(
            file_chunk=mock_file_chunk,
            similarity_score=score,
            source="file",
            embedding=[0.1, 0.2, 0.3]  # Add embedding for expansion tests
        )

    def test_select_expansion_anchors_statistical_threshold(self):
        """Test anchor selection using statistical threshold."""
        # Create mock results with varying scores
        results = [
            self.create_mock_commit_result("commit1", 0.9),
            self.create_mock_commit_result("commit2", 0.8),
            self.create_mock_commit_result("commit3", 0.7),
            self.create_mock_commit_result("commit4", 0.6),
            self.create_mock_commit_result("commit5", 0.3),  # Should be filtered out
            self.create_mock_commit_result("commit6", 0.2),  # Should be filtered out
        ]
        
        anchors = self.searcher._select_expansion_anchors(results)
        
        # Should select anchors above statistical threshold
        assert len(anchors) >= 3  # Minimum anchors guarantee
        assert all(anchor.similarity_score >= 0.2 for anchor in anchors)  # All should be above minimum

    def test_select_expansion_anchors_minimum_fallback(self):
        """Test anchor selection fallback to minimum count when threshold is too strict."""
        # Create mock results with very similar scores (low std deviation)
        results = [
            self.create_mock_commit_result("commit1", 0.5),
            self.create_mock_commit_result("commit2", 0.51),
            self.create_mock_commit_result("commit3", 0.49),
            self.create_mock_commit_result("commit4", 0.48),
        ]
        
        anchors = self.searcher._select_expansion_anchors(results)
        
        # Should fallback to minimum anchors even if statistical threshold is strict
        assert len(anchors) >= self.searcher.expansion_min_anchors
        assert len(anchors) <= len(results)

    def test_select_expansion_anchors_empty_results(self):
        """Test anchor selection with empty results."""
        anchors = self.searcher._select_expansion_anchors([])
        assert len(anchors) == 0

    def test_progressive_expansion_commits_separate_centroids(self):
        """Test progressive expansion with separate metadata/diff centroids."""
        # Create mock anchor results with different sources
        metadata_anchors = [
            self.create_mock_commit_result("meta1", 0.9, "metadata"),
            self.create_mock_commit_result("meta2", 0.8, "metadata"),
            self.create_mock_commit_result("meta3", 0.85, "combined"),
        ]
        
        diff_anchors = [
            self.create_mock_commit_result("diff1", 0.95, "diff"),
            self.create_mock_commit_result("diff2", 0.88, "diff"),
        ]
        
        all_anchors = metadata_anchors + diff_anchors
        
        # Mock vector DB responses
        mock_metadata_results = [
            VectorSearchResult(result_id="new_commit1", similarity_score=0.8, source="metadata", embedding=[0.1, 0.2, 0.3]),
            VectorSearchResult(result_id="new_commit2", similarity_score=0.7, source="metadata", embedding=[0.4, 0.5, 0.6]),
        ]
        
        mock_diff_results = [
            VectorSearchResult(result_id="new_commit3", similarity_score=0.85, source="diff", embedding=[0.7, 0.8, 0.9]),
        ]
        
        self.mock_vector_db.search_metadata.return_value = mock_metadata_results
        self.mock_vector_db.search_diffs.return_value = mock_diff_results
        
        # Mock metadata DB to return changelists
        def mock_get_changelists(ids):
            changelists = []
            for commit_id in ids:
                mock_changelist = Mock(spec=Changelist)
                mock_changelist.id = commit_id
                mock_changelist.project_name = "test-project"
                mock_changelist.message = f"New commit {commit_id}"
                mock_changelist.author = "test_author"
                mock_changelist.timestamp = None
                mock_changelist.files = []
                mock_changelist.diff = ""
                changelists.append(mock_changelist)
            return changelists
        
        self.mock_metadata_db.get_changelists_by_ids.side_effect = mock_get_changelists
        
        # Mock reranker
        self.mock_reranker.rerank.return_value = [(0, 0.9), (1, 0.85), (2, 0.8)]
        
        with patch.object(self.searcher, '_extract_embedding_vector', return_value=[0.1, 0.2, 0.3]):
            with patch.object(self.searcher, '_calculate_centroid', return_value=[0.5, 0.5, 0.5]):
                new_commits = self.searcher._progressive_expansion_commits("test query", all_anchors, 5)
        
        # Should perform searches for both metadata and diff centroids
        assert self.mock_vector_db.search_metadata.called
        assert self.mock_vector_db.search_diffs.called
        
        # Should return aggregated results
        assert len(new_commits) > 0

    def test_progressive_expansion_files(self):
        """Test progressive expansion for file chunks."""
        # Create mock file anchor results
        file_anchors = [
            self.create_mock_file_result("file1.py", 0.9),
            self.create_mock_file_result("file2.py", 0.85),
            self.create_mock_file_result("file3.py", 0.8),
        ]
        
        # Mock vector DB response
        mock_file_results = [
            VectorSearchResult(result_id="new_file1.py", similarity_score=0.82, source="file", embedding=[0.1, 0.2, 0.3]),
            VectorSearchResult(result_id="new_file2.py", similarity_score=0.78, source="file", embedding=[0.4, 0.5, 0.6]),
        ]
        
        self.mock_vector_db.search_files.return_value = mock_file_results
        
        # Mock metadata DB to return file chunks
        def mock_get_file_chunks(ids):
            chunks = []
            for chunk_id in ids:
                mock_chunk = Mock(spec=FileChunk)
                mock_chunk.file_path = chunk_id
                mock_chunk.content = f"Content for {chunk_id}"
                mock_chunk.line_start = 1
                mock_chunk.line_end = 10
                mock_chunk.revision_id = "abc123"
                chunks.append(mock_chunk)
            return chunks
        
        self.mock_metadata_db.get_file_chunks_by_ids.side_effect = mock_get_file_chunks
        
        # Mock reranker
        self.mock_reranker.rerank.return_value = [(0, 0.85), (1, 0.8)]
        
        with patch.object(self.searcher, '_extract_embedding_vector', return_value=[0.1, 0.2, 0.3]):
            with patch.object(self.searcher, '_calculate_centroid', return_value=[0.5, 0.5, 0.5]):
                new_files = self.searcher._progressive_expansion_files("test query", file_anchors, 3)
        
        # Should perform file search with centroid
        assert self.mock_vector_db.search_files.called
        
        # Should return results
        assert len(new_files) > 0

    def test_calculate_centroid_numpy_fallback(self):
        """Test centroid calculation with numpy fallback."""
        vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        centroid = self.searcher._calculate_centroid(vectors)
        
        # Should calculate average for each dimension
        expected = [4.0, 5.0, 6.0]  # (1+4+7)/3, (2+5+8)/3, (3+6+9)/3
        assert len(centroid) == len(expected)
        for i, val in enumerate(centroid):
            assert abs(val - expected[i]) < 1e-10

    def test_calculate_centroid_empty_vectors(self):
        """Test centroid calculation with empty vectors."""
        with pytest.raises(ValueError, match="No vectors provided to _calculate_centroid"):
            self.searcher._calculate_centroid([])

    def test_calculate_centroid_mismatched_dimensions(self):
        """Test centroid calculation with mismatched vector dimensions."""
        vectors = [
            [1.0, 2.0],
            [3.0, 4.0, 5.0]  # Different dimension
        ]
        
        # Test that mismatched dimensions raise an exception
        with pytest.raises(ValueError, match="Found 1 vectors with inconsistent dimensions"):
            self.searcher._calculate_centroid(vectors)

    def test_search_with_centroid_metadata(self):
        """Test search with metadata centroid."""
        centroid = [0.5, 0.5, 0.5]
        mock_results = [
            VectorSearchResult(result_id="test1", similarity_score=0.8, source="metadata")
        ]
        
        self.mock_vector_db.search_metadata.return_value = mock_results
        
        results = self.searcher._search_with_centroid("test query", centroid, 5, is_metadata=True)
        
        assert self.mock_vector_db.search_metadata.called
        assert len(results) == 1

    def test_search_with_centroid_diff(self):
        """Test search with diff centroid."""
        centroid = [0.5, 0.5, 0.5]
        mock_results = [
            VectorSearchResult(result_id="test1", similarity_score=0.8, source="diff")
        ]
        
        self.mock_vector_db.search_diffs.return_value = mock_results
        
        results = self.searcher._search_with_centroid("test query", centroid, 5, is_metadata=False)
        
        assert self.mock_vector_db.search_diffs.called
        assert len(results) == 1

    def test_search_with_centroid_files(self):
        """Test search with file centroid."""
        centroid = [0.5, 0.5, 0.5]
        mock_results = [
            VectorSearchResult(result_id="test.py", similarity_score=0.8, source="file")
        ]
        
        self.mock_vector_db.search_files.return_value = mock_results
        
        results = self.searcher._search_with_centroid("test query", centroid, 5, is_file=True)
        
        assert self.mock_vector_db.search_files.called
        assert len(results) == 1

    def test_search_with_centroid_exception_handling(self):
        """Test search with centroid exception handling."""
        centroid = [0.5, 0.5, 0.5]
        
        # Mock exception during search
        self.mock_vector_db.search_metadata.side_effect = Exception("Search failed")
        
        results = self.searcher._search_with_centroid("test query", centroid, 5, is_metadata=True)
        
        # Should return empty list on exception
        assert len(results) == 0

    def test_full_search_flow_with_expansion(self):
        """Test full search flow with query expansion enabled."""
        # Create test parameters
        params = QueryParams(
            prompt="test query",
            max_changes=5,
            max_file_chunks=5
        )
        
        # Mock initial search results
        mock_metadata_results = [
            VectorSearchResult(result_id="commit1", similarity_score=0.9, source="metadata", embedding=[0.1, 0.2, 0.3]),
            VectorSearchResult(result_id="commit2", similarity_score=0.8, source="metadata", embedding=[0.4, 0.5, 0.6]),
        ]
        
        mock_diff_results = [
            VectorSearchResult(result_id="commit1", similarity_score=0.85, source="diff", embedding=[0.7, 0.8, 0.9]),
        ]
        
        mock_file_results = [
            VectorSearchResult(result_id="file1.py", similarity_score=0.9, source="file", embedding=[0.1, 0.2, 0.3]),
        ]
        
        self.mock_vector_db.search_metadata.return_value = mock_metadata_results
        self.mock_vector_db.search_diffs.return_value = mock_diff_results
        self.mock_vector_db.search_files.return_value = mock_file_results
        
        # Mock metadata DB responses
        def mock_get_changelists(ids):
            changelists = []
            for commit_id in ids:
                mock_changelist = Mock(spec=Changelist)
                mock_changelist.id = commit_id
                mock_changelist.project_name = "test-project"
                mock_changelist.message = f"Commit {commit_id}"
                mock_changelist.author = "test_author"
                mock_changelist.timestamp = None
                mock_changelist.files = []
                mock_changelist.diff = ""
                changelists.append(mock_changelist)
            return changelists
        
        def mock_get_file_chunks(ids):
            chunks = []
            for chunk_id in ids:
                mock_chunk = Mock(spec=FileChunk)
                mock_chunk.file_path = chunk_id
                mock_chunk.content = f"Content for {chunk_id}"
                mock_chunk.line_start = 1
                mock_chunk.line_end = 10
                mock_chunk.revision_id = "abc123"
                chunks.append(mock_chunk)
            return chunks
        
        self.mock_metadata_db.get_changelists_by_ids.side_effect = mock_get_changelists
        self.mock_metadata_db.get_file_chunks_by_ids.side_effect = mock_get_file_chunks
        
        # Mock reranker
        self.mock_reranker.rerank.return_value = [(0, 0.95), (1, 0.85)]
        
        # Mock expansion results
        mock_expansion_results = [
            VectorSearchResult(result_id="expanded_commit1", similarity_score=0.75, source="metadata", embedding=[0.2, 0.3, 0.4]),
        ]
        
        self.mock_vector_db.search_metadata.return_value = mock_expansion_results
        
        # Mock metadata DB for expansion results
        def mock_get_expansion_changelists(ids):
            changelists = []
            for commit_id in ids:
                mock_changelist = Mock(spec=Changelist)
                mock_changelist.id = commit_id
                mock_changelist.project_name = "test-project"
                mock_changelist.message = f"Expanded commit {commit_id}"
                mock_changelist.author = "test_author"
                mock_changelist.timestamp = None
                mock_changelist.files = []
                mock_changelist.diff = ""
                changelists.append(mock_changelist)
            return changelists
        
        self.mock_metadata_db.get_changelists_by_ids.side_effect = lambda ids: mock_get_expansion_changelists(ids) if "expanded" in ids[0] else mock_get_changelists(ids)
        
        with patch.object(self.searcher, '_extract_embedding_vector', return_value=[0.1, 0.2, 0.3]):
            with patch.object(self.searcher, '_calculate_centroid', return_value=[0.5, 0.5, 0.5]):
                results = self.searcher.search(params)
        
        # Should complete search with expansion
        assert len(results) <= 10  # Combined limit of commits and files
        assert isinstance(results, list)

    def test_search_flow_without_expansion(self):
        """Test search flow with query expansion disabled."""
        # Create searcher with expansion disabled
        searcher_no_expansion = Searcher(
            expert_name="test_expert",
            embedder=self.mock_embedder,
            metadata_db=self.mock_metadata_db,
            vector_db=self.mock_vector_db,
            reranker=self.mock_reranker,
            enable_query_expansion=False
        )
        
        # Create test parameters
        params = QueryParams(
            prompt="test query",
            max_changes=5,
            max_file_chunks=5
        )
        
        # Mock search results
        mock_metadata_results = [
            VectorSearchResult(result_id="commit1", similarity_score=0.9, source="metadata"),
            VectorSearchResult(result_id="commit2", similarity_score=0.8, source="metadata"),
        ]
        
        mock_file_results = [
            VectorSearchResult(result_id="file1.py", similarity_score=0.9, source="file"),
        ]
        
        self.mock_vector_db.search_metadata.return_value = mock_metadata_results
        self.mock_vector_db.search_files.return_value = mock_file_results
        
        # Mock metadata DB responses
        def mock_get_changelists(ids):
            changelists = []
            for commit_id in ids:
                mock_changelist = Mock(spec=Changelist)
                mock_changelist.id = commit_id
                mock_changelist.project_name = "test-project"
                mock_changelist.message = f"Commit {commit_id}"
                mock_changelist.author = "test_author"
                mock_changelist.timestamp = None
                mock_changelist.files = []
                mock_changelist.diff = ""
                changelists.append(mock_changelist)
            return changelists
        
        def mock_get_file_chunks(ids):
            chunks = []
            for chunk_id in ids:
                mock_chunk = Mock(spec=FileChunk)
                mock_chunk.file_path = chunk_id
                mock_chunk.content = f"Content for {chunk_id}"
                mock_chunk.line_start = 1
                mock_chunk.line_end = 10
                mock_chunk.revision_id = "abc123"
                chunks.append(mock_chunk)
            return chunks
        
        self.mock_metadata_db.get_changelists_by_ids.side_effect = mock_get_changelists
        self.mock_metadata_db.get_file_chunks_by_ids.side_effect = mock_get_file_chunks
        
        # Mock reranker
        self.mock_reranker.rerank.return_value = [(0, 0.95), (1, 0.85)]
        
        results = searcher_no_expansion.search(params)
        
        # Should complete search without expansion
        assert len(results) <= 10  # Combined limit of commits and files
        assert isinstance(results, list)
        
        # Should not use expansion (expansion is disabled)
        # The methods still exist on the class but should not be used in the search flow
        assert not searcher_no_expansion.enable_query_expansion

    def test_edge_case_single_anchor(self):
        """Test behavior with only single anchor (below minimum)."""
        single_result = [self.create_mock_commit_result("single", 0.9, "metadata")]
        
        anchors = self.searcher._select_expansion_anchors(single_result)
        
        # Should still return the single anchor due to minimum fallback
        assert len(anchors) == 1
        assert anchors[0].get_id() == "single"

    def test_edge_case_large_quality_gap(self):
        """Test behavior with large quality gap between top and rest."""
        results = [
            self.create_mock_commit_result("top", 0.95, "metadata"),
            self.create_mock_commit_result("good1", 0.6, "metadata"),
            self.create_mock_commit_result("good2", 0.55, "metadata"),
            self.create_mock_commit_result("bad1", 0.2, "metadata"),  # Large gap
            self.create_mock_commit_result("bad2", 0.15, "metadata"),
        ]
        
        anchors = self.searcher._select_expansion_anchors(results)
        
        # Should select minimum anchors even with large quality gap
        assert len(anchors) >= self.searcher.expansion_min_anchors
        assert "top" in [a.get_id() for a in anchors]

    def test_deduplication_in_expansion(self):
        """Test that expansion results are properly deduplicated and don't crash."""
        # Create test parameters
        params = QueryParams(
            prompt="test query",
            max_changes=10,
            max_file_chunks=5
        )
        
        # Mock initial search results
        mock_metadata_results = [
            VectorSearchResult(result_id="original_commit1", similarity_score=0.9, source="metadata", embedding=[0.1, 0.2, 0.3]),
            VectorSearchResult(result_id="original_commit2", similarity_score=0.8, source="metadata", embedding=[0.4, 0.5, 0.6]),
        ]
        
        self.mock_vector_db.search_metadata.return_value = mock_metadata_results
        self.mock_vector_db.search_diffs.return_value = []
        self.mock_vector_db.search_files.return_value = []
        
        # Mock metadata DB responses for original results
        def mock_get_changelists(ids):
            changelists = []
            for commit_id in ids:
                mock_changelist = Mock(spec=Changelist)
                mock_changelist.id = commit_id
                mock_changelist.project_name = "test-project"
                mock_changelist.message = f"Original commit {commit_id}"
                mock_changelist.author = "test_author"
                mock_changelist.timestamp = None
                mock_changelist.files = []
                mock_changelist.diff = ""
                changelists.append(mock_changelist)
            return changelists
        
        self.mock_metadata_db.get_changelists_by_ids.side_effect = mock_get_changelists
        
        # Mock reranker
        self.mock_reranker.rerank.return_value = [(0, 0.95), (1, 0.85)]
        
        # Mock expansion results that include duplicates - this would previously cause a crash
        # because VectorSearchResult objects don't have get_id() method
        mock_expansion_results = [
            VectorSearchResult(result_id="original_commit1", similarity_score=0.75, source="metadata", embedding=[0.7, 0.8, 0.9]),  # Duplicate
            VectorSearchResult(result_id="new_commit1", similarity_score=0.65, source="metadata", embedding=[0.4, 0.5, 0.6]),       # New
        ]
        
        # Mock metadata DB for expansion results
        def mock_get_expansion_changelists(ids):
            changelists = []
            for commit_id in ids:
                mock_changelist = Mock(spec=Changelist)
                mock_changelist.id = commit_id
                mock_changelist.project_name = "test-project"
                mock_changelist.message = f"Expansion commit {commit_id}"
                mock_changelist.author = "test_author"
                mock_changelist.timestamp = None
                mock_changelist.files = []
                mock_changelist.diff = ""
                changelists.append(mock_changelist)
            return changelists
        
        # Override the metadata DB to handle expansion results
        self.mock_metadata_db.get_changelists_by_ids.side_effect = mock_get_expansion_changelists
        
        # Test that the search completes without crashing
        # This verifies that the deduplication fix works
        with patch.object(self.searcher, '_progressive_expansion_commits', return_value=mock_expansion_results):
            with patch.object(self.searcher, '_extract_embedding_vector', return_value=[0.1, 0.2, 0.3]):
                with patch.object(self.searcher, '_calculate_centroid', return_value=[0.5, 0.5, 0.5]):
                    # This should not crash due to the deduplication fix
                    results = self.searcher.search(params)
        
        # Verify that search completed successfully
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Verify that all results are proper QueryResult objects with get_id() method
        for result in results:
            assert hasattr(result, 'get_id'), f"Result {type(result)} missing get_id() method"
            assert callable(getattr(result, 'get_id')), f"get_id() not callable on {type(result)}"
            commit_id = result.get_id()
            assert isinstance(commit_id, str), f"get_id() should return string, got {type(commit_id)}"
            assert len(commit_id) > 0, f"get_id() should return non-empty string, got '{commit_id}'"
        
        print(f"✓ Deduplication test passed. Search completed without crash.")
        print(f"✓ Found {len(results)} results with proper get_id() methods")


if __name__ == "__main__":
    pytest.main([__file__])