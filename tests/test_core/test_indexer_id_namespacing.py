"""Tests that the indexer namespaces commit storage ids by project.

Commit identity (a P4 CL number or git hash) is only unique *within* a single
repository. When one expert indexes multiple projects that draw from the same
Perforce depot (or the same git history), a cross-cutting changelist appears
under more than one project with the *same raw id*. Without namespacing, the
shared id collides across:

  - ``changelists.id`` (SQLite PRIMARY KEY, written with INSERT OR REPLACE)
  - the metadata vector id (== raw commit id)
  - the diff vector ids (``{raw}_chunk_{n}``)

...causing the last-indexed project to clobber the others. These tests pin the
fix: the indexer rewrites ``commit.id`` to ``"{project}/{raw}"`` for storage,
while preserving the raw id as the VCS resume token.
"""

import datetime
from unittest.mock import Mock

import pytest

from expert_among_us.core.indexer import Indexer
from expert_among_us.models.changelist import Changelist
from expert_among_us.config.settings import Settings


BASE_TIME = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)


@pytest.fixture
def mock_metadata_db():
    db = Mock()
    db.get_last_processed_commit_hash.return_value = None
    db.get_project_commit_count.return_value = 0
    db.get_commit_count.return_value = 0
    return db


@pytest.fixture
def mock_vector_db():
    return Mock()


@pytest.fixture
def mock_embedder():
    embedder = Mock()
    # One vector per embed_batch call (single metadata text / single diff chunk).
    embedder.embed_batch.return_value = [[0.1, 0.2]]
    embedder.dimension = 2
    return embedder


def _make_vcs():
    vcs = Mock()
    vcs.get_total_commit_count.return_value = 1
    vcs.get_commit_position.return_value = (0, 1)
    return vcs


def _make_indexer(project_name, tmp_path, mock_metadata_db, mock_vector_db, mock_embedder, vcs):
    # Index diffs, skip file-content chunking to keep the test focused.
    settings = Settings(embed_file_chunks=False, embed_diffs=True)
    expert_config = {"name": "TestExpert", "project_root": str(tmp_path)}
    project_config = {
        "name": project_name,
        "expert_name": "TestExpert",
        "project_root": str(tmp_path),
        "vcs_type": "p4",
        "has_vector_metadata": True,
    }
    return Indexer(
        expert_config=expert_config,
        vcs=vcs,
        metadata_db=mock_metadata_db,
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        settings=settings,
        project_config=project_config,
    )


def _commit(raw_id):
    return Changelist(
        id=raw_id,
        expert_name="TestExpert",
        project_name="",  # set by the indexer
        timestamp=BASE_TIME,
        author="tester",
        message="Cross-cutting change",
        diff="--- a/a.cpp\n+++ b/a.cpp\n+int x = 1;",
        files=["a.cpp"],
    )


def test_stored_changelist_id_is_project_namespaced(
    tmp_path, mock_metadata_db, mock_vector_db, mock_embedder
):
    """The changelist stored in SQLite carries the ``{project}/{raw}`` id."""
    vcs = _make_vcs()
    vcs.get_commits_after.side_effect = [[_commit("12345")], []]

    indexer = _make_indexer(
        "Code", tmp_path, mock_metadata_db, mock_vector_db, mock_embedder, vcs
    )
    indexer.index_unified(batch_size=10)

    assert mock_metadata_db.insert_changelists.called
    stored = mock_metadata_db.insert_changelists.call_args[0][0][0]
    assert stored.id == "Code/12345"
    assert stored.project_name == "Code"


def test_vector_ids_are_project_namespaced(
    tmp_path, mock_metadata_db, mock_vector_db, mock_embedder
):
    """Metadata and diff vector ids share the namespaced commit id."""
    vcs = _make_vcs()
    vcs.get_commits_after.side_effect = [[_commit("12345")], []]

    indexer = _make_indexer(
        "Code", tmp_path, mock_metadata_db, mock_vector_db, mock_embedder, vcs
    )
    indexer.index_unified(batch_size=10)

    # Metadata vector id == storage id
    meta_vectors = mock_vector_db.insert_metadata.call_args[0][0]
    assert meta_vectors[0][0] == "Code/12345"
    assert mock_vector_db.insert_metadata.call_args.kwargs["metadata"] == {"project": "Code"}

    # Diff vector ids are "{storage_id}_chunk_{n}" so stripping "_chunk_" round-trips
    diff_vectors = mock_vector_db.insert_diffs.call_args[0][0]
    assert diff_vectors[0][0] == "Code/12345_chunk_0"
    assert diff_vectors[0][0].split("_chunk_")[0] == "Code/12345"


def test_resume_token_remains_raw_id(
    tmp_path, mock_metadata_db, mock_vector_db, mock_embedder
):
    """Per-project indexing state stores the raw VCS id, not the storage id."""
    vcs = _make_vcs()
    vcs.get_commits_after.side_effect = [[_commit("12345")], []]

    indexer = _make_indexer(
        "Code", tmp_path, mock_metadata_db, mock_vector_db, mock_embedder, vcs
    )
    indexer.index_unified(batch_size=10)

    assert mock_metadata_db.update_project_last_processed.called
    expert_arg, project_arg, hash_arg = (
        mock_metadata_db.update_project_last_processed.call_args[0]
    )
    assert expert_arg == "TestExpert"
    assert project_arg == "Code"
    assert hash_arg == "12345"  # raw id, so the next run can resume from VCS


def test_same_raw_id_in_two_projects_does_not_collide(
    tmp_path, mock_metadata_db, mock_vector_db, mock_embedder
):
    """Two projects sharing CL 12345 produce distinct stored ids (no clobber)."""
    # Project "Code"
    vcs_code = _make_vcs()
    vcs_code.get_commits_after.side_effect = [[_commit("12345")], []]
    _make_indexer(
        "Code", tmp_path, mock_metadata_db, mock_vector_db, mock_embedder, vcs_code
    ).index_unified(batch_size=10)

    # Project "Gems" (same raw CL number)
    vcs_gems = _make_vcs()
    vcs_gems.get_commits_after.side_effect = [[_commit("12345")], []]
    _make_indexer(
        "Gems", tmp_path, mock_metadata_db, mock_vector_db, mock_embedder, vcs_gems
    ).index_unified(batch_size=10)

    stored_ids = {
        call[0][0][0].id for call in mock_metadata_db.insert_changelists.call_args_list
    }
    assert stored_ids == {"Code/12345", "Gems/12345"}

    vector_ids = {
        call[0][0][0][0] for call in mock_vector_db.insert_metadata.call_args_list
    }
    assert vector_ids == {"Code/12345", "Gems/12345"}


def test_to_stored_id_format():
    assert Indexer.to_stored_id("Code", "12345") == "Code/12345"
    assert Indexer.to_stored_id("payment-service", "abc123") == "payment-service/abc123"
