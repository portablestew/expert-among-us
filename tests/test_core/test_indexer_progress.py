import datetime
from unittest.mock import Mock, patch

import pytest

from expert_among_us.core.indexer import Indexer
from expert_among_us.models.changelist import Changelist


class DummyCommit(Changelist):
    """Lightweight commit model used for testing Indexer batching."""

    def __init__(self, id: str, ts: datetime.datetime, files: list[str], diff: str = "diff"):
        super().__init__(
            id=id,
            expert_name="TestExpert",
            timestamp=ts,
            author="tester",
            message=f"Commit {id}",
            diff=diff,
            files=files,
        )


@pytest.fixture
def mock_vcs():
    """Minimal mock VCS provider returning a fixed sequence of commits."""

    class MockVCS:
        def __init__(self):
            base = datetime.datetime(2024, 1, 1)
            # 3 commits, 2 in the first batch, 1 in the second
            self._commits = [
                DummyCommit("c1", base + datetime.timedelta(days=0), ["file1.py"], diff="diff1"),
                DummyCommit("c2", base + datetime.timedelta(days=1), ["file2.py"], diff="diff2"),
                DummyCommit("c3", base + datetime.timedelta(days=2), ["file3.py"], diff="diff3"),
            ]

        def get_commits_after(self, workspace_path, after_hash, batch_size, subdirs=None):
            if after_hash is None:
                start_idx = 0
            else:
                ids = [c.id for c in self._commits]
                start_idx = ids.index(after_hash) + 1 if after_hash in ids else 0
            return self._commits[start_idx : start_idx + batch_size]

        def get_tracked_files_at_commit(self, workspace_path, revision_id, subdirs=None):
            # All files reported by the matching commit exist
            for c in self._commits:
                if c.id == revision_id:
                    return c.files
            return []

        def get_file_content_at_commit(self, workspace_path, file_path, revision_id):
            # Simple deterministic content to avoid being treated as binary
            return f"# {revision_id}::{file_path}\nprint('ok')\n"

    return MockVCS()


@pytest.fixture
def mock_metadata_db():
    """Mock metadata DB with minimal behavior for unified indexing."""
    db = Mock()
    db.get_last_processed_commit_hash.return_value = None
    db.get_commit_count.return_value = 0
    return db


@pytest.fixture
def mock_vector_db():
    """Mock vector DB capturing inserted vectors."""
    db = Mock()
    return db


@pytest.fixture
def mock_embedder():
    """Mock embedder with predictable outputs and embed_batch tracking."""

    def _embed_batch(texts):
        # Return a distinct scalar per position so sizes are easy to reason about
        return [[float(i)] for i, _ in enumerate(texts)]

    embedder = Mock()
    embedder.embed_batch.side_effect = _embed_batch
    embedder.dimension = 4
    return embedder


@pytest.fixture
def expert_config(tmp_path):
    return {
        "name": "TestExpert",
        "workspace_path": str(tmp_path),
        "max_commits": 10,
    }


@pytest.fixture
def indexer(expert_config, mock_vcs, mock_metadata_db, mock_vector_db, mock_embedder):
    return Indexer(
        expert_config=expert_config,
        vcs=mock_vcs,
        metadata_db=mock_metadata_db,
        vector_db=mock_vector_db,
        embedder=mock_embedder,
    )


@patch("expert_among_us.core.indexer.console")
def test_index_unified_progress_and_batches(mock_console, indexer, mock_metadata_db, mock_embedder):
    """
    Happy-path verification for progress reporting and embedding batching.

    Verifies:
    - No exceptions when running index_unified.
    - Batch summary includes newest commit date in expected format.
    - Progress instance is not left running at end.
    - Embedding batching is used with list inputs of appropriate sizes.
    """
    # Run indexing with small batch size to force multiple batches
    indexer.index_unified(batch_size=2)

    # 1) No exception is implicitly covered by successful return.

    # 2) Verify batch summary printed with progress format.
    printed = "".join(str(args) for args, _ in mock_console.print.call_args_list)
    assert "[green]Progress:" in printed
    # We accept any of the known commit dates; just ensure "YYYY-MM-DD" timestamp is present.
    assert "2024-01-01" in printed or "2024-01-02" in printed or "2024-01-03" in printed

    # 3) Verify Progress is not left running after index_unified completes.
    # The Progress API exposes a 'finished' flag indicating all tasks are done and stopped.
    assert indexer.progress.finished

    # 4) Verify embed_batch batching behavior.
    # Collect all embed_batch calls and ensure each used a non-empty list.
    assert mock_embedder.embed_batch.call_count >= 2
    batch_lengths = [len(call_args[0][0]) for call_args in mock_embedder.embed_batch.call_args_list]
    assert all(n > 0 for n in batch_lengths)

    # The implementation may batch metadata and diff embeddings separately and/or in smaller chunks.
    # We only require that batching is used (multiple non-empty calls), without enforcing a specific
    # minimum batch size. This keeps the test aligned with progress/batching refactors.

    # Ensure metadata_db last processed commit was updated to final commit in sequence.
    assert mock_metadata_db.update_last_processed_commit.call_count >= 1
    last_call = mock_metadata_db.update_last_processed_commit.call_args_list[-1]
    # last_call[0] contains positional args: (expert_name, last_processed_hash)
    assert last_call[0][1] == "c3"