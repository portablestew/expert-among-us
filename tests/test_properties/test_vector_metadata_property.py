"""
Property-based test for vector metadata completeness.

Property 4: Vector Metadata Completeness — every vector inserted with a project
name has metadata {"project": project_name}.

**Validates: Requirements 4.1, 4.2**
"""

import chromadb
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from expert_among_us.db.vector.chroma import ChromaVectorDB


# --- Strategies ---

# Valid project names: starts with alphanumeric, then alphanumeric/hyphens/underscores
valid_name_start = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
)
valid_name_rest = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=0,
    max_size=30,
)


@st.composite
def valid_project_names(draw):
    """Generate valid project names matching [a-zA-Z0-9][a-zA-Z0-9_-]*."""
    start = draw(valid_name_start)
    rest = draw(valid_name_rest)
    return start + rest


# Strategy for generating a small list of vectors (1–5 vectors, dimension 64 for speed)
VECTOR_DIM = 64


@st.composite
def vector_batch(draw, min_size=1, max_size=5):
    """Generate a batch of (id, embedding) tuples."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    vectors = []
    for i in range(count):
        vec_id = draw(st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
            min_size=3,
            max_size=20,
        ))
        # Ensure unique IDs within a batch
        vec_id = f"{vec_id}_{i}"
        embedding = np.random.randn(VECTOR_DIM).tolist()
        vectors.append((vec_id, embedding))
    return vectors


# --- Shared fixture: single EphemeralClient reused across all examples ---

def _make_ephemeral_db():
    """Create a ChromaVectorDB backed by an EphemeralClient (in-memory, fast)."""
    db = ChromaVectorDB("test_expert")
    db.client = chromadb.EphemeralClient(
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    db.initialize(dimension=VECTOR_DIM)
    return db


def _clear_collections(db: ChromaVectorDB):
    """Delete all documents from all collections so the DB can be reused."""
    for coll in [db.metadata_collection, db.diff_collection, db.file_collection]:
        if coll is not None:
            # Get all IDs and delete them
            existing = coll.get()
            if existing["ids"]:
                coll.delete(ids=existing["ids"])


# --- Property 4: Vector Metadata Completeness ---

class TestProperty4VectorMetadataCompleteness:
    """
    Property 4: Vector Metadata Completeness

    For any vector inserted during indexing with a project name, the vector's
    ChromaDB metadata should contain a "project" key whose value equals the
    project name.

    **Validates: Requirements 4.1, 4.2**
    """

    def setup_method(self):
        """Create a shared ephemeral ChromaDB for all hypothesis examples."""
        self.db = _make_ephemeral_db()

    def teardown_method(self):
        """Close the shared DB after each test method."""
        self.db.close()

    @given(
        project_name=valid_project_names(),
        vectors=vector_batch(),
    )
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_insert_metadata_stores_project_metadata(self, project_name, vectors):
        """Vectors inserted via insert_metadata with project metadata have correct metadata stored."""
        _clear_collections(self.db)

        metadata = {"project": project_name}
        self.db.insert_metadata(vectors, metadata=metadata)

        ids = [v[0] for v in vectors]
        result = self.db.metadata_collection.get(ids=ids, include=["metadatas"])

        assert len(result["ids"]) == len(vectors)
        for meta in result["metadatas"]:
            assert meta is not None
            assert "project" in meta
            assert meta["project"] == project_name

    @given(
        project_name=valid_project_names(),
        vectors=vector_batch(),
    )
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_insert_diffs_stores_project_metadata(self, project_name, vectors):
        """Vectors inserted via insert_diffs with project metadata have correct metadata stored."""
        _clear_collections(self.db)

        metadata = {"project": project_name}
        self.db.insert_diffs(vectors, metadata=metadata)

        ids = [v[0] for v in vectors]
        result = self.db.diff_collection.get(ids=ids, include=["metadatas"])

        assert len(result["ids"]) == len(vectors)
        for meta in result["metadatas"]:
            assert meta is not None
            assert "project" in meta
            assert meta["project"] == project_name

    @given(
        project_name=valid_project_names(),
        vectors=vector_batch(),
    )
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_insert_files_stores_project_metadata(self, project_name, vectors):
        """Vectors inserted via insert_files with project metadata have correct metadata stored."""
        _clear_collections(self.db)

        metadata = {"project": project_name}
        self.db.insert_files(vectors, metadata=metadata)

        ids = [v[0] for v in vectors]
        result = self.db.file_collection.get(ids=ids, include=["metadatas"])

        assert len(result["ids"]) == len(vectors)
        for meta in result["metadatas"]:
            assert meta is not None
            assert "project" in meta
            assert meta["project"] == project_name

    @given(
        project_name=valid_project_names(),
        vectors=vector_batch(),
    )
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_insert_without_metadata_stores_no_project_key(self, project_name, vectors):
        """Vectors inserted without metadata (metadata=None) have no project key in metadata."""
        _clear_collections(self.db)

        self.db.insert_metadata(vectors, metadata=None)

        ids = [v[0] for v in vectors]
        result = self.db.metadata_collection.get(ids=ids, include=["metadatas"])

        assert len(result["ids"]) == len(vectors)
        for meta in result["metadatas"]:
            if meta is not None:
                assert "project" not in meta
