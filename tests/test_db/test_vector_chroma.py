"""
Comprehensive tests for ChromaVectorDB covering:
- Collection initialization with specified dimension
- Vector insertion (single and batch)
- Vector search with similarity scores
- Vector deletion by IDs
- Count operations
- Connection closure
- Edge cases (empty collections, invalid queries)
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from expert_among_us.db.vector.chroma import ChromaVectorDB
from expert_among_us.db.vector.base import VectorSearchResult


@pytest.fixture
def temp_vector_db():
    """Fixture providing an initialized ChromaVectorDB instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        expert_name = "test_vector_expert"
        
        # Create ChromaDB with temporary directory
        db = ChromaVectorDB(expert_name)
        db.client = __import__('chromadb').PersistentClient(path=tmpdir)
        db.initialize(dimension=1024)
        
        yield db
        
        # Ensure proper cleanup order (critical on Windows)
        try:
            # First close the database properly
            db.close()
            
            # Force garbage collection to release file handles
            import gc
            gc.collect()
            
            # Give Windows time to release file locks
            import time
            time.sleep(0.2)
            
            # Additional cleanup - delete the client reference explicitly
            del db
            gc.collect()
        except Exception:
            pass


@pytest.fixture
def sample_vectors():
    """Fixture providing sample vectors for testing."""
    # Create normalized vectors of dimension 1024
    np.random.seed(42)
    vectors = []
    for i in range(5):
        vec = np.random.randn(1024)
        vec = vec / np.linalg.norm(vec)  # Normalize
        vectors.append(vec.tolist())
    return vectors


class TestCollectionInitialization:
    """Tests for collection initialization with specified dimension."""

    def test_db_initialization_with_dimension(self):
        """Verify that ChromaVectorDB can be initialized with a specific dimension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChromaVectorDB("test_expert")
            db.client = __import__('chromadb').PersistentClient(path=tmpdir)
            db.initialize(dimension=1024)
            assert db.metadata_collection is not None
            assert db.diff_collection is not None
            assert db.file_collection is not None
            db.close()
            
            # Clean up for Windows
            import gc
            import time
            del db
            gc.collect()
            time.sleep(0.2)

    def test_db_persists_directory(self):
        """Verify that database persists to the specified directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_name = "persist_test"
            
            # Create and add data
            db = ChromaVectorDB(expert_name)
            db.client = __import__('chromadb').PersistentClient(path=tmpdir)
            db.initialize(dimension=1024)
            
            vec = np.random.randn(1024).tolist()
            db.insert_vectors([("id_1", vec)])
            db.close()
            
            # Clean up first instance
            import gc
            import time
            del db
            gc.collect()
            time.sleep(0.2)

            # Reopen and verify data persists
            db2 = ChromaVectorDB(expert_name)
            db2.client = __import__('chromadb').PersistentClient(path=tmpdir)
            db2.initialize(dimension=1024)
            assert db2.count() == 1
            db2.close()
            
            # Clean up second instance
            del db2
            gc.collect()
            time.sleep(0.2)



class TestVectorInsertion:
    """Tests for vector insertion operations."""

    def test_insert_single_vector(self, temp_vector_db, sample_vectors):
        """Verify that a single vector can be inserted."""
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[0])])
        assert temp_vector_db.count() == 1

    def test_insert_batch_vectors(self, temp_vector_db, sample_vectors):
        """Verify that multiple vectors can be inserted in batch."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        
        temp_vector_db.insert_vectors(vectors_to_insert)
        assert temp_vector_db.count() == 5

    def test_insert_duplicate_id_overwrites(self, temp_vector_db, sample_vectors):
        """Verify that inserting with duplicate ID overwrites the previous vector."""
        temp_vector_db.insert_vectors([("duplicate_id", sample_vectors[0])])
        temp_vector_db.insert_vectors([("duplicate_id", sample_vectors[1])])
        
        assert temp_vector_db.count() == 1


    def test_insert_without_initialize_raises_error(self):
        """Verify that inserting without initialization raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChromaVectorDB("no_init_test")
            db.client = __import__('chromadb').PersistentClient(path=tmpdir)
            # Don't call initialize()
            
            vec = np.random.randn(1024).tolist()
            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.insert_vectors([("id_1", vec)])
            
            db.close()
            
            # Clean up for Windows
            import gc
            import time
            del db
            gc.collect()
            time.sleep(0.2)

    def test_insert_metadata_vectors(self, temp_vector_db, sample_vectors):
        """Verify metadata-specific insert works."""
        temp_vector_db.insert_metadata([("commit_1", sample_vectors[0])])
        assert temp_vector_db.count() == 1

    def test_insert_diff_vectors(self, temp_vector_db, sample_vectors):
        """Verify diff-specific insert works."""
        temp_vector_db.insert_diffs([("commit_1_chunk_0", sample_vectors[0])])
        assert temp_vector_db.count() == 1

    def test_insert_file_vectors(self, temp_vector_db, sample_vectors):
        """Verify file-specific insert works."""
        temp_vector_db.insert_files([("path/to/file.py:chunk_0", sample_vectors[0])])
        assert temp_vector_db.count() == 1


class TestVectorSearch:
    """Tests for vector search operations with similarity scores."""

    def test_search_returns_vector_search_results(self, temp_vector_db, sample_vectors):
        """Verify that search returns VectorSearchResult objects."""
        # Insert vectors
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(3)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        # Search with the first vector
        results = temp_vector_db.search(sample_vectors[0], top_k=2)
        
        assert len(results) > 0
        assert isinstance(results[0], VectorSearchResult)

    def test_search_result_contains_required_fields(self, temp_vector_db, sample_vectors):
        """Verify that search results contain changelist_id and similarity_score."""
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[0])])
        
        results = temp_vector_db.search(sample_vectors[0], top_k=1)
        
        assert len(results) == 1
        result = results[0]
        assert hasattr(result, 'result_id')
        assert hasattr(result, 'similarity_score')
        assert result.result_id == "vec_id_1"

    def test_search_with_limit(self, temp_vector_db, sample_vectors):
        """Verify that search top_k parameter works correctly."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        results = temp_vector_db.search(sample_vectors[0], top_k=2)
        assert len(results) <= 2

    def test_search_returns_closest_vectors(self, temp_vector_db, sample_vectors):
        """Verify that search returns vectors in order of similarity."""
        # Insert vectors
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        # Search with the first vector - it should be most similar to itself
        results = temp_vector_db.search(sample_vectors[0], top_k=5)
        
        # The first result should be the query vector itself or very similar
        assert results[0].result_id == "vec_id_0" or results[0].similarity_score > 0.99

    def test_search_similarity_ordering(self, temp_vector_db, sample_vectors):
        """Verify that search results are ordered by similarity score."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        results = temp_vector_db.search(sample_vectors[0], top_k=5)
        
        # Similarity scores should be in descending order (highest first)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_collection(self, temp_vector_db, sample_vectors):
        """Verify search behavior on empty collection."""
        results = temp_vector_db.search(sample_vectors[0], top_k=10)
        assert results == []

    def test_search_without_initialize_raises_error(self):
        """Verify that searching without initialization raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChromaVectorDB("no_init_search")
            db.client = __import__('chromadb').PersistentClient(path=tmpdir)
            # Don't call initialize()
            
            vec = np.random.randn(1024).tolist()
            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.search(vec, top_k=5)
            
            db.close()
            
            # Clean up for Windows
            import gc
            import time
            del db
            gc.collect()
            time.sleep(0.2)


class TestVectorDeletion:
    """Tests for vector deletion operations."""

    def test_delete_file_chunks(self, temp_vector_db, sample_vectors):
        """Verify file chunk deletion works correctly."""
        # Insert file chunk vectors
        file_chunk_ids = [("file:example.py:chunk_0", sample_vectors[0])]
        temp_vector_db.insert_files(file_chunk_ids)
        assert temp_vector_db.count() == 1
        
        # Delete file chunks
        temp_vector_db.delete_file_chunks(["file:example.py:chunk_0"])
        assert temp_vector_db.count() == 0

    def test_delete_file_chunks_multiple(self, temp_vector_db, sample_vectors):
        """Verify multiple file chunks can be deleted at once."""
        # Insert multiple file chunk vectors
        file_chunk_ids = [
            ("file:example.py:chunk_0", sample_vectors[0]),
            ("file:example.py:chunk_1", sample_vectors[1]),
            ("file:other.py:chunk_0", sample_vectors[2])
        ]
        temp_vector_db.insert_files(file_chunk_ids)
        assert temp_vector_db.count() == 3
        
        # Delete specific file chunks
        temp_vector_db.delete_file_chunks(["file:example.py:chunk_0", "file:example.py:chunk_1"])
        assert temp_vector_db.count() == 1

    def test_delete_file_chunks_nonexistent(self, temp_vector_db):
        """Verify that deleting non-existent file chunks doesn't raise error."""
        # Should not raise an exception
        temp_vector_db.delete_file_chunks(["file:nonexistent.py:chunk_0"])
        assert temp_vector_db.count() == 0

    def test_delete_file_chunks_empty_list(self, temp_vector_db, sample_vectors):
        """Verify delete with empty chunk ID list doesn't affect collection."""
        temp_vector_db.insert_files([("file:example.py:chunk_0", sample_vectors[0])])
        
        temp_vector_db.delete_file_chunks([])
        assert temp_vector_db.count() == 1



class TestCountOperations:
    """Tests for vector count operations."""

    def test_count_empty_collection(self, temp_vector_db):
        """Verify count returns 0 for empty collection."""
        assert temp_vector_db.count() == 0

    def test_count_without_collection(self):
        """Verify count returns 0 when collection is not initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChromaVectorDB("no_collection")
            db.client = __import__('chromadb').PersistentClient(path=tmpdir)
            # Don't call initialize()
            
            assert db.count() == 0
            db.close()
            
            # Clean up for Windows
            import gc
            import time
            del db
            gc.collect()
            time.sleep(0.2)


class TestConnectionManagement:
    """Tests for database connection management."""

    def test_close_connection(self):
        """Verify that database connection can be closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChromaVectorDB("close_test")
            db.client = __import__('chromadb').PersistentClient(path=tmpdir)
            db.initialize(dimension=1024)
            
            # Close should not raise an error
            db.close()
            
            # Clean up for Windows
            import gc
            import time
            del db
            gc.collect()
            time.sleep(0.2)

    def test_multiple_instances_same_directory(self):
        """Verify that multiple instances can access the same database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_name = "multi_instance"
            
            db1 = ChromaVectorDB(expert_name)
            db1.client = __import__('chromadb').PersistentClient(path=tmpdir)
            db1.initialize(dimension=1024)
            
            vec = np.random.randn(1024).tolist()
            db1.insert_vectors([("vec_id_1", vec)])
            db1.close()
            
            # Clean up first instance
            import gc
            import time
            del db1
            gc.collect()
            time.sleep(0.2)
            
            db2 = ChromaVectorDB(expert_name)
            db2.client = __import__('chromadb').PersistentClient(path=tmpdir)
            db2.initialize(dimension=1024)
            
            # Should be able to search and find the vector
            results = db2.search(vec, top_k=1)
            assert len(results) > 0
            db2.close()
            
            # Clean up second instance
            del db2
            gc.collect()
            time.sleep(0.2)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_search_with_single_vector_collection(self, temp_vector_db, sample_vectors):
        """Verify search works correctly with single vector in collection."""
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[0])])
        
        results = temp_vector_db.search(sample_vectors[0], top_k=5)
        
        assert len(results) == 1
        assert results[0].result_id == "vec_id_1"

    def test_large_batch_insertion(self, temp_vector_db):
        """Verify batch insertion works with large number of vectors."""
        num_vectors = 100
        np.random.seed(42)
        vectors_to_insert = [
            (f"vec_id_{i}", (np.random.randn(1024) / np.linalg.norm(np.random.randn(1024))).tolist())
            for i in range(num_vectors)
        ]
        
        temp_vector_db.insert_vectors(vectors_to_insert)
        assert temp_vector_db.count() == num_vectors

    def test_search_limit_exceeds_collection_size(self, temp_vector_db, sample_vectors):
        """Verify search with limit exceeding collection size returns all vectors."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(3)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        results = temp_vector_db.search(sample_vectors[0], top_k=100)
        assert len(results) == 3

    def test_vector_normalization_consistency(self, temp_vector_db):
        """Verify that vector operations handle normalization consistently."""
        # Create two identical vectors
        vec = [1.0] + [0.0] * 1023
        
        temp_vector_db.insert_vectors([
            ("vec_id_1", vec),
            ("vec_id_2", vec)
        ])
        
        # Search should find both as very similar
        results = temp_vector_db.search(vec, top_k=2)
        assert len(results) == 2
        # Both should have high similarity
        assert all(r.similarity_score > 0.99 for r in results)

    def test_insert_and_search_with_zero_vector(self, temp_vector_db):
        """Verify handling of zero vectors."""
        zero_vec = [0.0] * 1024
        
        # Inserting zero vector should not crash
        temp_vector_db.insert_vectors([("zero_vec", zero_vec)])
        assert temp_vector_db.count() == 1
        
        # Searching with zero vector should work
        results = temp_vector_db.search(zero_vec, top_k=1)
        assert len(results) <= 1

    def test_special_characters_in_ids(self, temp_vector_db, sample_vectors):
        """Verify vectors with special characters in ID are handled."""
        special_id = "vec:id/with-special_chars.123"
        temp_vector_db.insert_vectors([(special_id, sample_vectors[0])])
        assert temp_vector_db.count() == 1
        
        # Should be able to search and find it
        results = temp_vector_db.search(sample_vectors[0], top_k=1)
        assert len(results) == 1
        assert results[0].result_id == special_id

    def test_similarity_score_range(self, temp_vector_db, sample_vectors):
        """Verify that similarity scores are in valid range [0, 1]."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        results = temp_vector_db.search(sample_vectors[0], top_k=5)
        
        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0

    def test_empty_query_vector_list(self, temp_vector_db):
        """Verify behavior when searching with minimal valid vector."""
        # Insert a vector
        vec = np.random.randn(1024).tolist()
        temp_vector_db.insert_vectors([("vec_id_1", vec)])
        
        # Search should work with any valid 1024-d vector
        query_vec = [0.1] * 1024
        results = temp_vector_db.search(query_vec, top_k=1)
        assert len(results) == 1


class TestEmbeddingExtraction:
    """Tests for proper embedding extraction with include_embeddings parameter."""

    def test_search_returns_embeddings_when_requested(self, temp_vector_db, sample_vectors):
        """Verify that search returns embeddings when include_embeddings=True."""
        # Insert vectors
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(3)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        # Search with include_embeddings=True
        results = temp_vector_db.search(sample_vectors[0], top_k=2, include_embeddings=True)
        
        assert len(results) > 0
        # Verify embeddings are returned
        for result in results:
            assert result.embedding is not None
            assert isinstance(result.embedding, list)
            assert len(result.embedding) == 1024  # Expected dimension

    def test_search_returns_no_embeddings_when_not_requested(self, temp_vector_db, sample_vectors):
        """Verify that search returns None for embeddings when include_embeddings=False."""
        # Insert vectors
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[0])])
        
        # Search with include_embeddings=False (default)
        results = temp_vector_db.search(sample_vectors[0], top_k=1, include_embeddings=False)
        
        assert len(results) == 1
        assert results[0].embedding is None

    def test_search_metadata_returns_embeddings(self, temp_vector_db, sample_vectors):
        """Verify that search_metadata returns embeddings when requested."""
        # Insert metadata vectors
        temp_vector_db.insert_metadata([("commit_1", sample_vectors[0])])
        
        # Search metadata with include_embeddings=True
        results = temp_vector_db.search_metadata(sample_vectors[0], top_k=1, include_embeddings=True)
        
        assert len(results) == 1
        assert results[0].embedding is not None
        assert isinstance(results[0].embedding, list)
        assert len(results[0].embedding) == 1024

    def test_search_diffs_returns_embeddings(self, temp_vector_db, sample_vectors):
        """Verify that search_diffs returns embeddings when requested."""
        # Insert diff vectors
        temp_vector_db.insert_diffs([("commit_1_chunk_0", sample_vectors[0])])
        
        # Search diffs with include_embeddings=True
        results = temp_vector_db.search_diffs(sample_vectors[0], top_k=1, include_embeddings=True)
        
        assert len(results) == 1
        assert results[0].embedding is not None
        assert isinstance(results[0].embedding, list)
        assert len(results[0].embedding) == 1024

    def test_search_files_returns_embeddings(self, temp_vector_db, sample_vectors):
        """Verify that search_files returns embeddings when requested."""
        # Insert file vectors
        temp_vector_db.insert_files([("path/to/file.py:chunk_0", sample_vectors[0])])
        
        # Search files with include_embeddings=True
        results = temp_vector_db.search_files(sample_vectors[0], top_k=1, include_embeddings=True)
        
        assert len(results) == 1
        assert results[0].embedding is not None
        assert isinstance(results[0].embedding, list)
        assert len(results[0].embedding) == 1024

    def test_search_with_empty_results_and_embeddings(self, temp_vector_db, sample_vectors):
        """Verify that search handles empty results gracefully when include_embeddings=True."""
        # Search empty collection with include_embeddings=True
        results = temp_vector_db.search(sample_vectors[0], top_k=5, include_embeddings=True)
        
        assert results == []

    def test_search_with_partial_results_and_embeddings(self, temp_vector_db, sample_vectors):
        """Verify that search handles partial results correctly with embeddings."""
        # Insert only one vector
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[0])])
        
        # Search for more results than exist
        results = temp_vector_db.search(sample_vectors[0], top_k=5, include_embeddings=True)
        
        assert len(results) == 1
        assert results[0].embedding is not None
        assert isinstance(results[0].embedding, list)
        assert len(results[0].embedding) == 1024


class TestProjectMetadataInserts:
    """Tests for project metadata on vector inserts and where clause on searches."""

    def test_insert_metadata_with_project_metadata(self, temp_vector_db, sample_vectors):
        """Verify that insert_metadata attaches project metadata to vectors."""
        metadata = {"project": "payment-service"}
        temp_vector_db.insert_metadata([("commit_1", sample_vectors[0])], metadata=metadata)

        # Retrieve directly from collection to verify metadata
        result = temp_vector_db.metadata_collection.get(
            ids=["commit_1"], include=["metadatas"]
        )
        assert result["metadatas"][0]["project"] == "payment-service"

    def test_insert_diffs_with_project_metadata(self, temp_vector_db, sample_vectors):
        """Verify that insert_diffs attaches project metadata to vectors."""
        metadata = {"project": "user-service"}
        temp_vector_db.insert_diffs([("diff_1", sample_vectors[0])], metadata=metadata)

        result = temp_vector_db.diff_collection.get(
            ids=["diff_1"], include=["metadatas"]
        )
        assert result["metadatas"][0]["project"] == "user-service"

    def test_insert_files_with_project_metadata(self, temp_vector_db, sample_vectors):
        """Verify that insert_files attaches project metadata to vectors."""
        metadata = {"project": "shared-lib"}
        temp_vector_db.insert_files([("file_1", sample_vectors[0])], metadata=metadata)

        result = temp_vector_db.file_collection.get(
            ids=["file_1"], include=["metadatas"]
        )
        assert result["metadatas"][0]["project"] == "shared-lib"

    def test_insert_without_metadata(self, temp_vector_db, sample_vectors):
        """Verify that inserts work without metadata (legacy behavior)."""
        temp_vector_db.insert_metadata([("commit_no_meta", sample_vectors[0])])

        result = temp_vector_db.metadata_collection.get(
            ids=["commit_no_meta"], include=["metadatas"]
        )
        # Should still have a metadata dict (possibly empty or None project)
        assert result["metadatas"] is not None


class TestWhereClauseFiltering:
    """Tests for where clause filtering on ChromaDB searches."""

    def test_search_metadata_with_where_clause(self, temp_vector_db, sample_vectors):
        """Verify that search_metadata filters by project when where clause is provided."""
        # Insert vectors for two different projects
        temp_vector_db.insert_metadata(
            [("commit_proj_a", sample_vectors[0])], metadata={"project": "proj-a"}
        )
        temp_vector_db.insert_metadata(
            [("commit_proj_b", sample_vectors[1])], metadata={"project": "proj-b"}
        )

        # Search with where clause filtering to proj-a
        where = {"project": {"$in": ["proj-a"]}}
        results = temp_vector_db.search_metadata(sample_vectors[0], top_k=10, where=where)

        # Should only return proj-a result
        assert len(results) == 1
        assert results[0].result_id == "commit_proj_a"

    def test_search_diffs_with_where_clause(self, temp_vector_db, sample_vectors):
        """Verify that search_diffs filters by project when where clause is provided."""
        temp_vector_db.insert_diffs(
            [("diff_proj_a", sample_vectors[0])], metadata={"project": "proj-a"}
        )
        temp_vector_db.insert_diffs(
            [("diff_proj_b", sample_vectors[1])], metadata={"project": "proj-b"}
        )

        where = {"project": {"$in": ["proj-a"]}}
        results = temp_vector_db.search_diffs(sample_vectors[0], top_k=10, where=where)

        assert len(results) == 1
        assert results[0].result_id == "diff_proj_a"

    def test_search_files_with_where_clause(self, temp_vector_db, sample_vectors):
        """Verify that search_files filters by project when where clause is provided."""
        temp_vector_db.insert_files(
            [("file_proj_a", sample_vectors[0])], metadata={"project": "proj-a"}
        )
        temp_vector_db.insert_files(
            [("file_proj_b", sample_vectors[1])], metadata={"project": "proj-b"}
        )

        where = {"project": {"$in": ["proj-a"]}}
        results = temp_vector_db.search_files(sample_vectors[0], top_k=10, where=where)

        assert len(results) == 1
        assert results[0].result_id == "file_proj_a"

    def test_search_without_where_returns_all(self, temp_vector_db, sample_vectors):
        """Verify that search without where clause returns results from all projects."""
        temp_vector_db.insert_metadata(
            [("commit_a", sample_vectors[0])], metadata={"project": "proj-a"}
        )
        temp_vector_db.insert_metadata(
            [("commit_b", sample_vectors[1])], metadata={"project": "proj-b"}
        )

        # Search without where clause
        results = temp_vector_db.search_metadata(sample_vectors[0], top_k=10)

        # Should return both
        assert len(results) == 2

    def test_search_where_multiple_projects(self, temp_vector_db, sample_vectors):
        """Verify that where clause with multiple projects returns results from all listed."""
        temp_vector_db.insert_metadata(
            [("commit_a", sample_vectors[0])], metadata={"project": "proj-a"}
        )
        temp_vector_db.insert_metadata(
            [("commit_b", sample_vectors[1])], metadata={"project": "proj-b"}
        )
        temp_vector_db.insert_metadata(
            [("commit_c", sample_vectors[2])], metadata={"project": "proj-c"}
        )

        where = {"project": {"$in": ["proj-a", "proj-c"]}}
        results = temp_vector_db.search_metadata(sample_vectors[0], top_k=10, where=where)

        result_ids = {r.result_id for r in results}
        assert "commit_a" in result_ids
        assert "commit_c" in result_ids
        assert "commit_b" not in result_ids


class TestDeleteProjectVectors:
    """Tests for project vector deletion."""

    def test_delete_project_vectors_removes_from_all_collections(self, temp_vector_db, sample_vectors):
        """Verify that delete_project_vectors removes vectors from metadata, diffs, and files."""
        metadata = {"project": "to-delete"}
        temp_vector_db.insert_metadata([("m1", sample_vectors[0])], metadata=metadata)
        temp_vector_db.insert_diffs([("d1", sample_vectors[0])], metadata=metadata)
        temp_vector_db.insert_files([("f1", sample_vectors[0])], metadata=metadata)

        assert temp_vector_db.count() == 3

        temp_vector_db.delete_project_vectors("to-delete")

        assert temp_vector_db.count() == 0

    def test_delete_project_vectors_preserves_other_projects(self, temp_vector_db, sample_vectors):
        """Verify that deleting one project's vectors doesn't affect others."""
        temp_vector_db.insert_metadata(
            [("keep_m1", sample_vectors[0])], metadata={"project": "keep"}
        )
        temp_vector_db.insert_metadata(
            [("del_m1", sample_vectors[1])], metadata={"project": "delete-me"}
        )

        temp_vector_db.delete_project_vectors("delete-me")

        # keep project's vectors should still exist
        result = temp_vector_db.metadata_collection.get(ids=["keep_m1"])
        assert len(result["ids"]) == 1

        # deleted project's vectors should be gone
        result = temp_vector_db.metadata_collection.get(ids=["del_m1"])
        assert len(result["ids"]) == 0
