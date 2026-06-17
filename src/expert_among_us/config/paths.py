"""Data directory resolution for Expert Among Us.

This module is the single source of truth for where expert data lives on disk.
All call sites that need a base data directory should go through
:func:`resolve_data_dir` so the precedence rules stay consistent.

Resolution precedence (Option A - single resolved directory):

1. An explicit path (e.g. the ``--data-dir`` CLI flag or MCP ``--data-dir``).
2. A workspace-local ``./.expert-among-us`` directory, *if it already exists*.
3. The global ``~/.expert-among-us`` directory.

The workspace-local candidate is gated on existence so we never scatter new
``.expert-among-us`` directories into arbitrary working directories. To create
a workspace-local store, bootstrap it once with an explicit
``--data-dir ./.expert-among-us`` (or by creating the directory), after which
it is auto-detected.
"""

from pathlib import Path
from typing import Optional

# Name of the data directory in both the workspace (cwd) and home locations.
DATA_DIR_NAME = ".expert-among-us"


def home_data_dir() -> Path:
    """Return the global (home) data directory: ``~/.expert-among-us``."""
    return Path.home() / DATA_DIR_NAME


def workspace_data_dir() -> Path:
    """Return the workspace-local candidate: ``<cwd>/.expert-among-us``.

    This does not check for existence; use :func:`resolve_data_dir` for the
    gated resolution logic.
    """
    return Path.cwd() / DATA_DIR_NAME


def resolve_data_dir(explicit: Optional[Path] = None) -> Path:
    """Resolve the base data directory using the precedence chain.

    Args:
        explicit: An explicitly provided path (e.g. from ``--data-dir``).
            When supplied it always wins and is expanded/resolved to an
            absolute path.

    Returns:
        The resolved base data directory. Experts are stored under
        ``<resolved>/data/<expert_name>/``.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    workspace_candidate = workspace_data_dir()
    if workspace_candidate.is_dir():
        return workspace_candidate.resolve()

    return home_data_dir()
