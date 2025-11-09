import pytest
from unittest.mock import Mock


class DummyProgress:
    """
    Test-only no-op progress object to avoid Rich LiveError from Live/Progress.

    This implementation matches the Progress API usage patterns in the codebase
    closely enough that it can safely stand in for Rich's Progress.
    """

    def __init__(self, *args, **kwargs):
        # Accept arbitrary args/kwargs so it can replace Progress(...)
        self.finished = False

    # Methods used in tests / implementation
    def add_task(self, *args, **kwargs):
        # Return a stable dummy task id
        return "task-id"

    def update(self, *args, **kwargs):
        # No-op
        pass

    def advance(self, *args, **kwargs):
        # No-op
        pass

    def start(self):
        # Mirror Progress.start(); mark as running
        self.finished = False

    def stop(self):
        # Mark as finished
        self.finished = True

    # Context manager support, since Rich Progress is used that way sometimes
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


@pytest.fixture(autouse=True)
def dummy_progress(monkeypatch):
    """
    Globally patch expert_among_us.utils.progress.create_progress_bar for tests.

    We hook the factory so that any code calling create_progress_bar(...)
    (e.g. PromptGenerator.generate_prompts) receives a DummyProgress instance,
    preventing Rich Live/Console LiveError during pytest runs.
    """
    try:
        from expert_among_us.utils import progress as progress_utils
    except Exception:
        # If the module cannot be imported in some context, do nothing.
        yield
        return

    if hasattr(progress_utils, "create_progress_bar"):
        # Patch the factory used by PromptGenerator and others so tests never
        # construct a real Rich Progress/Live instance.
        def _create_progress_bar(description: str = "Processing", total=None):
            prog = DummyProgress()
            # Return (progress, task_id) as expected by callers.
            return prog, "task-id"

        monkeypatch.setattr(progress_utils, "create_progress_bar", _create_progress_bar)

    # ALSO patch the Progress symbol itself so any direct Progress(...) calls
    # (as used inside expert_among_us.utils.progress.create_progress_bar) resolve
    # to DummyProgress instead of Rich's Progress, preventing LiveError.
    if hasattr(progress_utils, "Progress"):
        monkeypatch.setattr(progress_utils, "Progress", DummyProgress)

    yield