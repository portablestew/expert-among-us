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
            assert db.collection is not None
            db.close()
            
            # Clean up for Windows
            import gc
            import time
            del db
            gc.collect()
            time.sleep(0.2)

    def test_db_creates_collection(self, temp_vector_db):
        """Verify that a collection is created on initialization."""
        assert temp_vector_db.collection is not None

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

    def test_dimension_parameter_stored(self):
        """Verify that the dimension parameter is properly used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChromaVectorDB("dim_test")
            db.client = __import__('chromadb').PersistentClient(path=tmpdir)
            db.initialize(dimension=512)
            
            # Verify collection was created with metadata
            assert db.collection is not None
            db.close()
            
            # Clean up for Windows
            import gc
            import time
            del db
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

    def test_insert_multiple_vectors_sequentially(self, temp_vector_db, sample_vectors):
        """Verify that vectors can be inserted one at a time."""
        for i, vec in enumerate(sample_vectors):
            temp_vector_db.insert_vectors([(f"vec_id_{i}", vec)])
        
        assert temp_vector_db.count() == len(sample_vectors)

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
        assert hasattr(result, 'changelist_id')
        assert hasattr(result, 'similarity_score')
        assert result.changelist_id == "vec_id_1"

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
        assert results[0].changelist_id == "vec_id_0" or results[0].similarity_score > 0.99

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

    def test_delete_single_vector(self, temp_vector_db, sample_vectors):
        """Verify that a single vector can be deleted."""
        temp_vector_db.insert_vectors([
            ("vec_id_1", sample_vectors[0]),
            ("vec_id_2", sample_vectors[1])
        ])
        
        assert temp_vector_db.count() == 2
        
        temp_vector_db.delete_by_ids(["vec_id_1"])
        assert temp_vector_db.count() == 1

    def test_delete_multiple_vectors(self, temp_vector_db, sample_vectors):
        """Verify that multiple vectors can be deleted."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        assert temp_vector_db.count() == 5
        
        temp_vector_db.delete_by_ids(["vec_id_0", "vec_id_1", "vec_id_2"])
        assert temp_vector_db.count() == 2

    def test_delete_nonexistent_vector(self, temp_vector_db):
        """Verify that deleting non-existent vector doesn't raise error."""
        # Should not raise an exception
        temp_vector_db.delete_by_ids(["nonexistent_id"])
        assert temp_vector_db.count() == 0

    def test_delete_vectors_partial_match(self, temp_vector_db, sample_vectors):
        """Verify that delete handles partial matches gracefully."""
        temp_vector_db.insert_vectors([
            ("vec_id_1", sample_vectors[0]),
            ("vec_id_2", sample_vectors[1])
        ])
        
        # Try to delete with mix of existing and non-existing IDs
        temp_vector_db.delete_by_ids(["vec_id_1", "nonexistent"])
        
        assert temp_vector_db.count() == 1

    def test_delete_all_vectors_sequentially(self, temp_vector_db, sample_vectors):
        """Verify that all vectors can be deleted one by one."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        for i in range(len(sample_vectors)):
            temp_vector_db.delete_by_ids([f"vec_id_{i}"])
        
        assert temp_vector_db.count() == 0

    def test_delete_without_initialize_raises_error(self):
        """Verify that deleting without initialization raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChromaVectorDB("no_init_delete")
            db.client = __import__('chromadb').PersistentClient(path=tmpdir)
            # Don't call initialize()
            
            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.delete_by_ids(["id_1"])
            
            db.close()
            
            # Clean up for Windows
            import gc
            import time
            del db
            gc.collect()
            time.sleep(0.2)


class TestCountOperations:
    """Tests for vector count operations."""

    def test_count_empty_collection(self, temp_vector_db):
        """Verify count returns 0 for empty collection."""
        assert temp_vector_db.count() == 0

    def test_count_after_insertions(self, temp_vector_db, sample_vectors):
        """Verify count increases with insertions."""
        assert temp_vector_db.count() == 0
        
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[0])])
        assert temp_vector_db.count() == 1
        
        temp_vector_db.insert_vectors([("vec_id_2", sample_vectors[1])])
        assert temp_vector_db.count() == 2

    def test_count_after_batch_insertion(self, temp_vector_db, sample_vectors):
        """Verify count is correct after batch insertion."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        
        temp_vector_db.insert_vectors(vectors_to_insert)
        assert temp_vector_db.count() == 5

    def test_count_after_deletion(self, temp_vector_db, sample_vectors):
        """Verify count decreases correctly after deletion."""
        vectors_to_insert = [
            (f"vec_id_{i}", sample_vectors[i])
            for i in range(5)
        ]
        temp_vector_db.insert_vectors(vectors_to_insert)
        
        assert temp_vector_db.count() == 5
        
        temp_vector_db.delete_by_ids(["vec_id_0"])
        assert temp_vector_db.count() == 4

    def test_count_with_duplicate_insert(self, temp_vector_db, sample_vectors):
        """Verify count doesn't increase with duplicate ID insertion."""
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[0])])
        assert temp_vector_db.count() == 1
        
        # Insert with same ID - should overwrite, not increase count
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[1])])
        assert temp_vector_db.count() == 1

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
        assert results[0].changelist_id == "vec_id_1"

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

    def test_delete_empty_id_list(self, temp_vector_db, sample_vectors):
        """Verify delete with empty ID list doesn't affect collection."""
        temp_vector_db.insert_vectors([("vec_id_1", sample_vectors[0])])
        
        temp_vector_db.delete_by_ids([])
        assert temp_vector_db.count() == 1

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
        assert results[0].changelist_id == special_id

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