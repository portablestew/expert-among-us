"""Auto-detection module for version control systems.

This module provides functionality to automatically detect which VCS
is in use in a given workspace and return an appropriate provider instance.
"""

from typing import Optional

from expert_among_us.vcs.base import VCSProvider
from expert_among_us.vcs.git import Git


# Registry of available VCS providers to check during auto-detection
# Providers are checked in order, so put more common ones first
VCS_PROVIDERS: list[type[VCSProvider]] = [
    Git,
    # Future providers can be added here (e.g., PerforceProvider)
]


def detect_vcs(workspace_path: str, debug_logger=None) -> Optional[VCSProvider]:
    """Automatically detect which VCS is in use and return a provider instance.
    
    This function tries each registered VCS provider's detect() method in order
    and returns an instance of the first provider that successfully detects
    the VCS in the given workspace.
    
    Args:
        workspace_path: Path to the workspace directory to check
        
    Returns:
        An instance of the detected VCS provider, or None if no VCS is detected
        
    Example:
        >>> vcs = detect_vcs("/path/to/my/project")
        >>> if vcs:
        ...     commits = vcs.get_commits_after("/path/to/my/project", after_hash=None, batch_size=10)
        >>> else:
        ...     print("No VCS detected")
    """
    # Validate path is not empty or whitespace-only
    if not workspace_path or not workspace_path.strip():
        return None

    for provider_class in VCS_PROVIDERS:
        if provider_class.detect(workspace_path):
            # Instantiate provider without threading debug_logger; providers consult DebugLogger directly.
            try:
                # New-style providers ignore debug_logger and use DebugLogger.is_enabled() internally.
                return provider_class()
            except TypeError:
                # Fallback: in case of legacy providers with different signatures.
                return provider_class()
     
    return None