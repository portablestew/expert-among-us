"""Auto-detection module for version control systems.

This module provides functionality to automatically detect which VCS
is in use in a given workspace and return an appropriate provider instance.
"""

from typing import Optional

from expert_among_us.vcs.base import VCSProvider
from expert_among_us.vcs.git import Git
from expert_among_us.vcs.perforce import Perforce


# Registry of available VCS providers to check during auto-detection
# Providers are checked in order, so put more common ones first
VCS_PROVIDERS: list[type[VCSProvider]] = [
    Git,
    Perforce,
]


def detect_vcs(project_root: str, settings) -> Optional[VCSProvider]:
    """Automatically detect which VCS is in use and return a provider instance.
    
    This function tries each registered VCS provider's detect() method in order
    and returns an instance of the first provider that successfully detects
    the VCS in the given workspace.
    
    Args:
        project_root: Path to the project root directory to check
        settings: Settings instance (required)
        
    Returns:
        An instance of the detected VCS provider, or None if no VCS is detected
        
    Example:
        >>> from expert_among_us.config.settings import Settings
        >>> settings = Settings()
        >>> vcs = detect_vcs("/path/to/my/project", settings)
        >>> if vcs:
        ...     commits = vcs.get_commits_after("/path/to/my/project", after_hash=None, batch_size=10)
        >>> else:
        ...     print("No VCS detected")
    """
    # Validate path is not empty or whitespace-only
    if not project_root or not project_root.strip():
        return None

    for provider_class in VCS_PROVIDERS:
        if provider_class.detect(project_root):
            # Instantiate provider with settings
            return provider_class(settings)
     
    return None