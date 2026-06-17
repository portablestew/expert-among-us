"""
Property-based tests for project indexing independence and populate idempotency.

Tests Properties 10 and 12 from the design document using Hypothesis.

**Validates: Requirements 9.2, 9.3, 9.7, 10.3, 12.1, 12.2, 13.2, 14.1, 14.2, 14.3**
"""

import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.models.changelist import Changelist


# --- Helpers ---

_shared_tmpdir = None
_shared_db = None


def _get_shared_db():
    """Get the shared DB instance for tests that use the new schema."""
    global _shared_tmpdir, _shared_db
    if _shared_db is None:
        _shared_tmpdir = tempfile.mkdtemp(prefix="migration_prop_test_")
        data_dir = Path(_shared_tmpdir)
        expert_name = "prop_test_expert"
        db_path = data_dir / "data" / expert_name / "metadata.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _shared_db = SQLiteMetadataDB(expert_name, data_dir=data_dir)
        _shared_db.initialize()
        _shared_db.create_expert(expert_name)
    return _shared_db


def _reset_db(db):
    """Truncate project-related tables to reset state between examples."""
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM changelist_files")
    cursor.execute("DELETE FROM file_chunks")
    cursor.execute("DELETE FROM file_contents")
    cursor.execute("DELETE FROM changelists")
    cursor.execute("DELETE FROM projects")
    db.conn.commit()


# --- Strategies ---

valid_name_start = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
)
valid_name_rest = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=0,
    max_size=15,
)


@st.composite
def valid_project_names(draw):
    """Generate valid project names matching [a-zA-Z0-9][a-zA-Z0-9_-]*."""
    start = draw(valid_name_start)
    rest = draw(valid_name_rest)
    return start + rest


# Strategy for commit hashes (hex strings)
commit_hashes = st.text(
    alphabet="0123456789abcdef",
    min_size=7,
    max_size=40,
)

# Strategy for workspace paths
project_roots = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789/_-.",
    min_size=1,
    max_size=80,
).map(lambda s: "/" + s)

# Strategy for VCS types
vcs_types = st.sampled_from(["git", "p4"])

# Strategy for changelist counts (keep small for speed)
changelist_counts = st.integers(min_value=0, max_value=5)


@st.composite
def changelist_dicts(draw, expert_name: str = "test-expert"):
    """Generate changelist dicts for old-schema DB insertion."""
    cl_id = draw(st.text(
        alphabet="0123456789abcdef",
        min_size=8,
        max_size=12,
    ))
    author = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz",
        min_size=3,
        max_size=15,
    ))
    message = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ",
        min_size=5,
        max_size=50,
    ))
    return {
        "id": cl_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "author": author,
        "message": message,
        "diff": b"diff content",
        "files": "src/main.py,src/utils.py",
    }


# --- Property 10: Project Indexing Independence ---

class TestProperty10ProjectIndexingIndependence:
    """
    Property 10: Project Indexing Independence

    For any expert with multiple projects, indexing one project (updating its
    last_processed_commit_hash) should not modify any other project's
    last_processed_commit_hash, first_processed_commit_hash, or last_indexed_at fields.

    **Validates: Requirements 12.1, 12.2, 13.2**
    """

    @given(
        project_a=valid_project_names(),
        project_b=valid_project_names(),
        commit_hash_a=commit_hashes,
        workspace_a=project_roots,
        workspace_b=project_roots,
        vcs_a=vcs_types,
        vcs_b=vcs_types,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_updating_project_a_does_not_modify_project_b(
        self, project_a, project_b, commit_hash_a, workspace_a, workspace_b, vcs_a, vcs_b
    ):
        """Updating project A's last_processed_commit_hash leaves project B unchanged."""
        assume(project_a != project_b)

        db = _get_shared_db()
        _reset_db(db)

        # Create two projects
        db.create_project("prop_test_expert", project_a, workspace_a, vcs_a)
        db.create_project("prop_test_expert", project_b, workspace_b, vcs_b)

        # Record project B's initial state
        proj_b_before = db.get_project("prop_test_expert", project_b)

        # Update project A's indexing state
        db.update_project_last_processed("prop_test_expert", project_a, commit_hash_a)

        # Verify project B is unchanged
        proj_b_after = db.get_project("prop_test_expert", project_b)

        assert proj_b_after["last_processed_commit_hash"] == proj_b_before["last_processed_commit_hash"]
        assert proj_b_after["first_processed_commit_hash"] == proj_b_before["first_processed_commit_hash"]
        assert proj_b_after["last_indexed_at"] == proj_b_before["last_indexed_at"]

    @given(
        project_a=valid_project_names(),
        project_b=valid_project_names(),
        commit_hash_a1=commit_hashes,
        commit_hash_a2=commit_hashes,
        commit_hash_b=commit_hashes,
        workspace_a=project_roots,
        workspace_b=project_roots,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_updates_to_a_still_leave_b_independent(
        self, project_a, project_b, commit_hash_a1, commit_hash_a2, commit_hash_b,
        workspace_a, workspace_b
    ):
        """Multiple updates to project A don't affect project B, even after B is also updated."""
        assume(project_a != project_b)

        db = _get_shared_db()
        _reset_db(db)

        db.create_project("prop_test_expert", project_a, workspace_a, "git")
        db.create_project("prop_test_expert", project_b, workspace_b, "git")

        # Update project B once
        db.update_project_last_processed("prop_test_expert", project_b, commit_hash_b)
        proj_b_after_own_update = db.get_project("prop_test_expert", project_b)

        # Update project A multiple times
        db.update_project_last_processed("prop_test_expert", project_a, commit_hash_a1)
        db.update_project_last_processed("prop_test_expert", project_a, commit_hash_a2)

        # Project B should still match its state after its own update
        proj_b_final = db.get_project("prop_test_expert", project_b)
        assert proj_b_final["last_processed_commit_hash"] == commit_hash_b
        assert proj_b_final["first_processed_commit_hash"] == proj_b_after_own_update["first_processed_commit_hash"]


# --- Property 12: Populate Idempotency ---

class TestProperty12PopulateIdempotency:
    """
    Property 12: Populate Idempotency

    For any fully-indexed project with no new commits, running populate again
    (inserting the same changelists) should produce no new rows in SQLite and
    no changes to any project's indexing state.

    **Validates: Requirements 14.1, 14.2, 14.3**
    """

    @given(
        project_name=valid_project_names(),
        workspace=project_roots,
        vcs_type=vcs_types,
        num_changelists=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_reinserting_same_changelists_produces_no_new_rows(
        self, project_name, workspace, vcs_type, num_changelists
    ):
        """INSERT OR REPLACE semantics means re-inserting same changelists is a no-op."""
        db = _get_shared_db()
        _reset_db(db)

        db.create_project("prop_test_expert", project_name, workspace, vcs_type)

        # Create changelists
        changelists = []
        for i in range(num_changelists):
            cl = Changelist(
                id=f"cl-{project_name}-{i}",
                expert_name="prop_test_expert",
                project_name=project_name,
                timestamp=datetime.now(timezone.utc),
                author="test-author",
                message=f"Commit {i}",
                diff=f"--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
                files=[f"{project_name}/file_{i}.py"],
            )
            changelists.append(cl)

        # First insert
        db.insert_changelists(changelists)
        count_after_first = db.get_project_commit_count("prop_test_expert", project_name)

        # Record state after first insert
        project_state_before = db.get_project("prop_test_expert", project_name)

        # Re-insert same changelists (simulates re-running populate with no new commits)
        db.insert_changelists(changelists)
        count_after_second = db.get_project_commit_count("prop_test_expert", project_name)

        # No new rows created
        assert count_after_second == count_after_first
        assert count_after_second == num_changelists

        # Project indexing state unchanged
        project_state_after = db.get_project("prop_test_expert", project_name)
        assert project_state_after["last_processed_commit_hash"] == project_state_before["last_processed_commit_hash"]
        assert project_state_after["first_processed_commit_hash"] == project_state_before["first_processed_commit_hash"]

    @given(
        project_name=valid_project_names(),
        workspace=project_roots,
        commit_hash=commit_hashes,
        num_changelists=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_update_project_last_processed_is_idempotent_with_same_hash(
        self, project_name, workspace, commit_hash, num_changelists
    ):
        """Calling update_project_last_processed with the same hash is effectively a no-op."""
        db = _get_shared_db()
        _reset_db(db)

        db.create_project("prop_test_expert", project_name, workspace, "git")

        # Insert changelists and update state
        changelists = []
        for i in range(num_changelists):
            cl = Changelist(
                id=f"cl-{project_name}-{i}",
                expert_name="prop_test_expert",
                project_name=project_name,
                timestamp=datetime.now(timezone.utc),
                author="test-author",
                message=f"Commit {i}",
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b",
                files=[f"{project_name}/f_{i}.py"],
            )
            changelists.append(cl)

        db.insert_changelists(changelists)
        db.update_project_last_processed("prop_test_expert", project_name, commit_hash)

        # Record state
        state_before = db.get_project("prop_test_expert", project_name)

        # Call update again with same hash
        db.update_project_last_processed("prop_test_expert", project_name, commit_hash)

        state_after = db.get_project("prop_test_expert", project_name)

        # Core fields unchanged
        assert state_after["last_processed_commit_hash"] == state_before["last_processed_commit_hash"]
        assert state_after["first_processed_commit_hash"] == state_before["first_processed_commit_hash"]

        # Commit count unchanged
        count = db.get_project_commit_count("prop_test_expert", project_name)
        assert count == num_changelists
