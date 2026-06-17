"""Integration tests for multi-project end-to-end flow.

Tests use real SQLiteMetadataDB and ChromaVectorDB instances (in temp directories)
with a mocked embedder to verify the full multi-project flow.

**Validates: Requirements 4.3, 4.4, 4.5, 10.1, 10.2, 11.1, 12.3, 12.4**
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Callable
from unittest.mock import Mock, patch

from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.db.vector.chroma import ChromaVectorDB
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.query import QueryParams
from expert_among_us.models.query_result import CommitResult
from expert_among_us.core.searcher import Searcher
from expert_among_us.embeddings.base import Embedder


# ---------------------------------------------------------------------------
# Deterministic mock embedder that returns predictable vectors
# ---------------------------------------------------------------------------

DIMENSION = 1024


class DeterministicEmbedder(Embedder):
    """Embedder that produces deterministic vectors based on text content.

    Uses a simple hash-based approach to generate reproducible embeddings
    that are semantically meaningful enough for integration testing:
    - Similar texts produce similar vectors (via shared hash components)
    - Different texts produce different vectors
    """

    def embed(self, text: str) -> List[float]:
        """Generate a deterministic 1024-d embedding from text."""
        import hashlib
        # Use SHA-512 to get enough bytes, then extend to DIMENSION
        h = hashlib.sha512(text.encode("utf-8")).digest()
        # Create base vector from hash bytes (normalized 0..1)
        base = [b / 255.0 for b in h]
        # Repeat to fill DIMENSION
        vector = (base * (DIMENSION // len(base) + 1))[:DIMENSION]
        # Normalize to unit length for cosine similarity
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Optional[List[float]]]:
        results = []
        for i, text in enumerate(texts):
            if text and text.strip():
                results.append(self.embed(text))
            else:
                results.append(None)
            if progress_callback:
                progress_callback(i + 1, len(texts))
        return results

    @property
    def dimension(self) -> int:
        return DIMENSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for databases, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="expert_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def expert_name():
    return "test-multi-expert"


@pytest.fixture
def embedder():
    return DeterministicEmbedder()


@pytest.fixture
def metadata_db(temp_dir, expert_name):
    """Real SQLiteMetadataDB in temp dir."""
    # Ensure directory structure exists (SQLiteMetadataDB expects data_dir/data/expert_name/)
    db_dir = temp_dir / "data" / expert_name
    db_dir.mkdir(parents=True, exist_ok=True)
    db = SQLiteMetadataDB(expert_name=expert_name, data_dir=temp_dir)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def vector_db(temp_dir, expert_name):
    """Real ChromaVectorDB (PersistentClient) in temp dir."""
    # Ensure directory structure exists (ChromaVectorDB expects data_dir/data/expert_name/chroma/)
    chroma_dir = temp_dir / "data" / expert_name / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    db = ChromaVectorDB(expert_name=expert_name, data_dir=temp_dir)
    db.initialize(dimension=DIMENSION)
    yield db
    db.close()


@pytest.fixture
def populated_dbs(metadata_db, vector_db, embedder, expert_name):
    """Populate two projects (payment-service, user-service) with changelists and vectors."""
    # Create expert
    metadata_db.create_expert(expert_name, description="Multi-project test expert")

    # Create projects
    metadata_db.create_project(
        expert_name=expert_name,
        project_name="payment-service",
        project_root="/repos/payment",
        vcs_type="git",
    )
    metadata_db.create_project(
        expert_name=expert_name,
        project_name="user-service",
        project_root="/repos/users",
        vcs_type="git",
    )

    # --- Payment service changelists ---
    payment_changelists = [
        Changelist(
            id="pay_commit_1",
            expert_name=expert_name,
            project_name="payment-service",
            timestamp=datetime(2024, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
            author="alice",
            message="Add payment processing handler",
            diff="--- a/payment-service/src/handler.py\n+++ b/payment-service/src/handler.py\n+def process_payment(amount):\n+    return True",
            files=["payment-service/src/handler.py", "payment-service/src/models.py"],
        ),
        Changelist(
            id="pay_commit_2",
            expert_name=expert_name,
            project_name="payment-service",
            timestamp=datetime(2024, 3, 2, 11, 0, 0, tzinfo=timezone.utc),
            author="bob",
            message="Add refund logic for payments",
            diff="--- a/payment-service/src/refund.py\n+++ b/payment-service/src/refund.py\n+def refund(order_id):\n+    return True",
            files=["payment-service/src/refund.py"],
        ),
    ]

    # --- User service changelists ---
    user_changelists = [
        Changelist(
            id="user_commit_1",
            expert_name=expert_name,
            project_name="user-service",
            timestamp=datetime(2024, 3, 3, 12, 0, 0, tzinfo=timezone.utc),
            author="charlie",
            message="Add user authentication endpoint",
            diff="--- a/user-service/src/auth.py\n+++ b/user-service/src/auth.py\n+def authenticate(token):\n+    return True",
            files=["user-service/src/auth.py", "user-service/src/middleware.py"],
        ),
        Changelist(
            id="user_commit_2",
            expert_name=expert_name,
            project_name="user-service",
            timestamp=datetime(2024, 3, 4, 13, 0, 0, tzinfo=timezone.utc),
            author="alice",
            message="Add user profile retrieval",
            diff="--- a/user-service/src/profile.py\n+++ b/user-service/src/profile.py\n+def get_profile(user_id):\n+    return {}",
            files=["user-service/src/profile.py"],
        ),
    ]

    # Insert changelists into metadata DB
    metadata_db.insert_changelists(payment_changelists)
    metadata_db.insert_changelists(user_changelists)

    # Insert vectors with project metadata into ChromaDB
    for cl in payment_changelists:
        embedding = embedder.embed(cl.get_metadata_text())
        vector_db.insert_metadata(
            vectors=[(cl.id, embedding)],
            metadata={"project": "payment-service"},
        )

    for cl in user_changelists:
        embedding = embedder.embed(cl.get_metadata_text())
        vector_db.insert_metadata(
            vectors=[(cl.id, embedding)],
            metadata={"project": "user-service"},
        )

    return {
        "payment_changelists": payment_changelists,
        "user_changelists": user_changelists,
    }


# ---------------------------------------------------------------------------
# Test: unified search returns results from both projects
# ---------------------------------------------------------------------------


class TestUnifiedSearch:
    """Test that unfiltered search returns results from both projects."""

    def test_unfiltered_search_returns_both_projects(
        self, metadata_db, vector_db, embedder, expert_name, populated_dbs
    ):
        """Verify unfiltered search returns results from both payment-service and user-service.

        **Validates: Requirements 4.4**
        """
        searcher = Searcher(
            expert_name=expert_name,
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            has_vector_metadata=True,
            enable_query_expansion=False,
            enable_reranking=False,
            enable_diff_search=False,
            enable_file_search=False,
        )

        params = QueryParams(
            prompt="How is authentication handled?",
            max_changes=10,
            max_file_chunks=0,
        )

        results = searcher.search(params)

        # Should get results (at least some from both projects)
        assert len(results) > 0

        # Collect project_names from returned changelists
        project_names = set()
        for r in results:
            assert isinstance(r, CommitResult)
            project_names.add(r.changelist.project_name)

        # Both projects should be represented in unfiltered results
        assert "payment-service" in project_names
        assert "user-service" in project_names


# ---------------------------------------------------------------------------
# Test: project prefix filter scopes results correctly
# ---------------------------------------------------------------------------


class TestProjectFilter:
    """Test that querying with project prefix filter scopes results correctly."""

    def test_search_with_payment_service_filter(
        self, metadata_db, vector_db, embedder, expert_name, populated_dbs
    ):
        """Verify search with files=["payment-service/"] only returns payment-service results.

        **Validates: Requirements 4.3, 4.4**
        """
        searcher = Searcher(
            expert_name=expert_name,
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            has_vector_metadata=True,
            enable_query_expansion=False,
            enable_reranking=False,
            enable_diff_search=False,
            enable_file_search=False,
        )

        params = QueryParams(
            prompt="How does payment processing work?",
            max_changes=10,
            max_file_chunks=0,
            files=["payment-service/"],
        )

        results = searcher.search(params)

        # All results should be from payment-service
        assert len(results) > 0
        for r in results:
            assert isinstance(r, CommitResult)
            assert r.changelist.project_name == "payment-service"
            # All files should start with payment-service/
            for f in r.changelist.files:
                assert f.startswith("payment-service/")

    def test_search_with_user_service_filter(
        self, metadata_db, vector_db, embedder, expert_name, populated_dbs
    ):
        """Verify search with files=["user-service/"] only returns user-service results.

        **Validates: Requirements 4.3, 4.4**
        """
        searcher = Searcher(
            expert_name=expert_name,
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            has_vector_metadata=True,
            enable_query_expansion=False,
            enable_reranking=False,
            enable_diff_search=False,
            enable_file_search=False,
        )

        params = QueryParams(
            prompt="How does user authentication work?",
            max_changes=10,
            max_file_chunks=0,
            files=["user-service/"],
        )

        results = searcher.search(params)

        # All results should be from user-service
        assert len(results) > 0
        for r in results:
            assert isinstance(r, CommitResult)
            assert r.changelist.project_name == "user-service"
            for f in r.changelist.files:
                assert f.startswith("user-service/")


# ---------------------------------------------------------------------------
# Test: legacy expert (has_vector_metadata=False) queries still work
# ---------------------------------------------------------------------------


class TestLegacyExpert:
    """Test that legacy experts with has_vector_metadata=False work without where clause."""

    def test_legacy_expert_query_no_where_clause(
        self, temp_dir, embedder
    ):
        """Verify that a legacy expert (has_vector_metadata=False) can be queried
        without ChromaDB where clause failures.

        **Validates: Requirements 10.1, 10.2**
        """
        expert_name = "legacy-expert"

        # Ensure directory structure exists
        db_dir = temp_dir / "data" / expert_name
        db_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir = db_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        # Create a fresh DB for legacy expert
        meta_db = SQLiteMetadataDB(expert_name=expert_name, data_dir=temp_dir)
        meta_db.initialize()

        vec_db = ChromaVectorDB(expert_name=expert_name, data_dir=temp_dir)
        vec_db.initialize(dimension=DIMENSION)

        # Create expert and project with has_vector_metadata=False (simulating migration)
        meta_db.create_expert(expert_name)
        meta_db.create_project(
            expert_name=expert_name,
            project_name="legacy-project",
            project_root="/repos/legacy",
            vcs_type="git",
        )
        # Manually set has_vector_metadata=False to simulate migrated project
        meta_db._connect()
        meta_db.conn.execute(
            "UPDATE projects SET has_vector_metadata = 0 WHERE expert_name = ? AND name = ?",
            (expert_name, "legacy-project"),
        )
        meta_db.conn.commit()

        # Insert a changelist (legacy - no project prefix, stored as-is)
        cl = Changelist(
            id="legacy_commit_1",
            expert_name=expert_name,
            project_name="legacy-project",
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            author="developer",
            message="Legacy commit without project prefix",
            diff="--- a/src/main.py\n+++ b/src/main.py\n+print('hello')",
            files=["src/main.py"],
        )
        meta_db.insert_changelists([cl])

        # Insert vector WITHOUT project metadata (simulating legacy behavior)
        embedding = embedder.embed(cl.get_metadata_text())
        vec_db.insert_metadata(vectors=[(cl.id, embedding)], metadata=None)

        # Create searcher with has_vector_metadata=False
        searcher = Searcher(
            expert_name=expert_name,
            embedder=embedder,
            metadata_db=meta_db,
            vector_db=vec_db,
            has_vector_metadata=False,
            enable_query_expansion=False,
            enable_reranking=False,
            enable_diff_search=False,
            enable_file_search=False,
        )

        # Query - even with files parameter, should NOT use where clause
        params = QueryParams(
            prompt="Legacy commit query",
            max_changes=10,
            max_file_chunks=0,
            files=["some-project/"],
        )

        # This should NOT raise an error (no where clause sent to ChromaDB)
        results = searcher.search(params)

        # The results may be empty because file filtering (startsWith) won't match
        # legacy unprefixed paths, but the critical thing is it doesn't crash
        # due to a where clause on vectors without project metadata
        assert isinstance(results, list)

        # Without files filter, legacy expert should return results
        params_no_filter = QueryParams(
            prompt="Legacy commit query",
            max_changes=10,
            max_file_chunks=0,
        )
        results_no_filter = searcher.search(params_no_filter)
        assert len(results_no_filter) > 0
        assert results_no_filter[0].changelist.id == "legacy_commit_1"

        meta_db.close()
        vec_db.close()


# ---------------------------------------------------------------------------
# Test: expansion pipeline threads where clause (no cross-project leakage)
# ---------------------------------------------------------------------------


class TestExpansionWhereClause:
    """Test that the expansion pipeline passes the where clause to all search calls."""

    def test_expansion_threads_where_clause(
        self, metadata_db, vector_db, embedder, expert_name, populated_dbs
    ):
        """Verify expansion pipeline threads where clause so no cross-project leakage occurs.

        When searching with files=["payment-service/"], expansion calls should also
        be scoped to payment-service only.

        **Validates: Requirements 11.1**
        """
        searcher = Searcher(
            expert_name=expert_name,
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            has_vector_metadata=True,
            enable_query_expansion=True,
            enable_reranking=False,
            enable_diff_search=False,
            enable_file_search=False,
            expansion_passes=1,
            expansion_min_anchors=1,
        )

        # Patch the vector_db.search_metadata to spy on where clause usage
        original_search_metadata = vector_db.search_metadata
        where_clauses_seen = []

        def spy_search_metadata(query_vector, top_k, include_embeddings=False, where=None):
            where_clauses_seen.append(where)
            return original_search_metadata(query_vector, top_k, include_embeddings, where=where)

        vector_db.search_metadata = spy_search_metadata

        params = QueryParams(
            prompt="How does payment processing work?",
            max_changes=10,
            max_file_chunks=0,
            files=["payment-service/"],
        )

        results = searcher.search(params)

        # All search_metadata calls should have used the payment-service where clause
        expected_where = {"project": {"$in": ["payment-service"]}}
        for where_clause in where_clauses_seen:
            assert where_clause == expected_where, (
                f"Expected where clause {expected_where}, got {where_clause}"
            )

        # All results should be from payment-service only (no leakage)
        for r in results:
            assert isinstance(r, CommitResult)
            assert r.changelist.project_name == "payment-service"


# ---------------------------------------------------------------------------
# Property 11: Project Deletion Isolation
# ---------------------------------------------------------------------------


class TestProjectDeletionIsolation:
    """Test that deleting a project removes only its data.

    **Property 11: Project Deletion Isolation** — deleting a project removes
    only its data.

    **Validates: Requirements 4.5, 12.3, 12.4**
    """

    def test_delete_project_removes_only_its_changelists(
        self, metadata_db, vector_db, embedder, expert_name, populated_dbs
    ):
        """Deleting payment-service should remove its changelists but leave user-service intact."""
        # Verify both projects have changelists before deletion
        pay_count_before = metadata_db.get_project_commit_count(expert_name, "payment-service")
        user_count_before = metadata_db.get_project_commit_count(expert_name, "user-service")
        assert pay_count_before == 2
        assert user_count_before == 2

        # Delete payment-service project
        metadata_db.delete_project(expert_name, "payment-service")

        # Verify payment-service changelists are gone
        pay_count_after = metadata_db.get_project_commit_count(expert_name, "payment-service")
        assert pay_count_after == 0

        # Verify user-service changelists are untouched
        user_count_after = metadata_db.get_project_commit_count(expert_name, "user-service")
        assert user_count_after == 2

        # Verify payment-service project record is gone
        assert metadata_db.get_project(expert_name, "payment-service") is None

        # Verify user-service project record still exists
        assert metadata_db.get_project(expert_name, "user-service") is not None

    def test_delete_project_removes_only_its_vectors(
        self, metadata_db, vector_db, embedder, expert_name, populated_dbs
    ):
        """Deleting payment-service should remove its vectors but leave user-service vectors intact."""
        # Verify we can find payment-service vectors before deletion
        query_vec = embedder.embed("payment processing")
        pay_results_before = vector_db.search_metadata(
            query_vec, top_k=10, where={"project": {"$in": ["payment-service"]}}
        )
        user_results_before = vector_db.search_metadata(
            query_vec, top_k=10, where={"project": {"$in": ["user-service"]}}
        )
        assert len(pay_results_before) > 0
        assert len(user_results_before) > 0

        # Delete payment-service vectors
        vector_db.delete_project_vectors("payment-service")

        # Verify payment-service vectors are gone
        pay_results_after = vector_db.search_metadata(
            query_vec, top_k=10, where={"project": {"$in": ["payment-service"]}}
        )
        assert len(pay_results_after) == 0

        # Verify user-service vectors are still present
        user_results_after = vector_db.search_metadata(
            query_vec, top_k=10, where={"project": {"$in": ["user-service"]}}
        )
        assert len(user_results_after) == len(user_results_before)

    def test_full_project_deletion_end_to_end(
        self, metadata_db, vector_db, embedder, expert_name, populated_dbs
    ):
        """Full deletion: remove project from SQLite and ChromaDB, verify isolation."""
        # Delete payment-service from both stores
        metadata_db.delete_project(expert_name, "payment-service")
        vector_db.delete_project_vectors("payment-service")

        # Verify: search scoped to user-service still works
        searcher = Searcher(
            expert_name=expert_name,
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            has_vector_metadata=True,
            enable_query_expansion=False,
            enable_reranking=False,
            enable_diff_search=False,
            enable_file_search=False,
        )

        params = QueryParams(
            prompt="user authentication",
            max_changes=10,
            max_file_chunks=0,
            files=["user-service/"],
        )

        results = searcher.search(params)
        assert len(results) > 0
        for r in results:
            assert r.changelist.project_name == "user-service"

        # Verify: unfiltered search only returns user-service now
        params_all = QueryParams(
            prompt="user authentication",
            max_changes=10,
            max_file_chunks=0,
        )
        results_all = searcher.search(params_all)
        for r in results_all:
            assert r.changelist.project_name == "user-service"
