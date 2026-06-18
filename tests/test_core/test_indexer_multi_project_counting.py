"""Regression tests for per-project commit counting in the indexer.

Bug: ``index_unified`` seeded its running ``total_commits`` counter from an
expert-wide commit count (``get_commit_count``) while clamping the loop to a
project-scoped ``max_commits``/``total_available``. For the second and later
projects of a multi-project expert, this made the loop terminate early — the
project stopped after ``total_available - (other projects' commits)`` commits —
and the final "skipped" stat then equaled the other projects' commit count.

These tests pin that a project indexes its FULL history regardless of how many
commits other projects have already contributed to the expert.
"""

import datetime
from unittest.mock import Mock

import pytest

from expert_among_us.core.indexer import Indexer
from expert_among_us.models.changelist import Changelist
from expert_among_us.config.settings import Settings


def _make_commits(n):
    base = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    return [
        Changelist(
            id=f"c{i}",
            expert_name="TestExpert",
            project_name="",  # assigned by the indexer
            timestamp=base + datetime.timedelta(days=i),
            author="tester",
            message=f"Commit c{i}",
            diff=f"--- a/f{i}.py\n+++ b/f{i}.py\n+x = {i}",
            files=[f"f{i}.py"],
        )
        for i in range(n)
    ]


class _MockVCS:
    """Mock VCS that hands out fresh copies and counts only its own commits."""

    def __init__(self, commits):
        self._commits = commits
        self._ids = [c.id for c in commits]

    def get_total_commit_count(self, project_root):
        return len(self._commits)

    def get_commits_after(self, project_root, after_hash, batch_size, progress_callback=None):
        if after_hash is None:
            start = 0
        else:
            start = (self._ids.index(after_hash) + 1) if after_hash in self._ids else 0
        batch = [c.model_copy(deep=True) for c in self._commits[start : start + batch_size]]
        if progress_callback and batch:
            progress_callback(len(batch), len(batch))
        return batch

    def get_commit_position(self, commit_id):
        if commit_id is None:
            return (0, len(self._commits))
        try:
            return (self._ids.index(commit_id) + 1, len(self._commits))
        except ValueError:
            return (0, len(self._commits))

    def get_tracked_files_at_commit(self, project_root, revision_id):
        return []

    def get_files_content_at_commit(self, project_root, file_paths, commit_hash, progress_callback=None):
        return {fp: f"# {commit_hash}\nprint('ok')\n" for fp in file_paths}


@pytest.fixture
def mock_vector_db():
    return Mock()


@pytest.fixture
def mock_embedder():
    embedder = Mock()
    embedder.embed_batch.side_effect = lambda texts, progress_callback=None: [
        [float(i)] for i, _ in enumerate(texts)
    ]
    embedder.dimension = 1
    return embedder


def _make_indexer(vcs, metadata_db, mock_vector_db, mock_embedder, tmp_path, max_commits=60000):
    settings = Settings(embed_file_chunks=False, embed_diffs=False)
    expert_config = {"name": "TestExpert", "project_root": str(tmp_path)}
    project_config = {
        "name": "second-project",
        "expert_name": "TestExpert",
        "project_root": str(tmp_path),
        "vcs_type": "p4",
        "has_vector_metadata": True,
    }
    return Indexer(
        expert_config=expert_config,
        vcs=vcs,
        metadata_db=metadata_db,
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        settings=settings,
        max_commits=max_commits,
        project_config=project_config,
    )


def test_second_project_indexes_full_history(
    mock_vector_db, mock_embedder, tmp_path
):
    """A project indexes all its commits even when the expert already has many."""
    commits = _make_commits(5)
    vcs = _MockVCS(commits)

    metadata_db = Mock()
    metadata_db.get_last_processed_commit_hash.return_value = None
    # Expert already has 757 commits from a *different* project ...
    metadata_db.get_commit_count.return_value = 757
    # ... but THIS project has none yet.
    metadata_db.get_project_commit_count.return_value = 0

    indexer = _make_indexer(vcs, metadata_db, mock_vector_db, mock_embedder, tmp_path)
    more_remain = indexer.index_unified(batch_size=2)

    # All 5 of this project's commits should be stored (no early termination).
    assert metadata_db.insert_changelists.call_count == 5
    stored_ids = [
        c[0][0][0].id for c in metadata_db.insert_changelists.call_args_list
    ]
    assert stored_ids == [f"second-project/c{i}" for i in range(5)]

    # Loop ended by exhausting the VCS, not by hitting a miscomputed cap.
    assert more_remain is False
    # Resume token is the raw id of the final commit.
    assert metadata_db.update_project_last_processed.call_args_list[-1][0][2] == "c4"


def test_already_indexed_other_project_does_not_short_circuit(
    mock_vector_db, mock_embedder, tmp_path
):
    """A huge expert-wide count must not trip the 'already indexed' early return."""
    commits = _make_commits(3)
    vcs = _MockVCS(commits)

    metadata_db = Mock()
    metadata_db.get_last_processed_commit_hash.return_value = None
    # Expert-wide count exceeds max_commits, but it's all from other projects.
    metadata_db.get_commit_count.return_value = 100
    metadata_db.get_project_commit_count.return_value = 0

    indexer = _make_indexer(
        vcs, metadata_db, mock_vector_db, mock_embedder, tmp_path, max_commits=50
    )
    indexer.index_unified(batch_size=2)

    # Despite expert-wide count (100) > max_commits (50), this project still indexes.
    assert metadata_db.insert_changelists.call_count == 3
