"""Core operations for Expert Among Us API.

This module provides the main operations for interacting with experts:
- query_expert: Search for similar changes in an expert's history
- list_experts: List all available experts
- import_expert: Import an expert via symlink
"""

import sys
from pathlib import Path
from typing import List, Optional

from expert_among_us.core.searcher import Searcher, SearchResult
from expert_among_us.models.query import QueryParams
from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB

from .context import ExpertContext
from .exceptions import ExpertNotFoundError, ExpertAlreadyExistsError, InvalidExpertError
from .models import ExpertInfo


def query_expert(
    expert_name: str,
    prompt: str,
    max_changes: int,
    max_file_chunks: int,
    users: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    search_scope: str = "all",
    min_score: float = 0.1,
    relative_threshold: float = 0.3,
    data_dir: Optional[Path] = None,
    embedding_provider: str = "local",
    enable_reranking: bool = True,  # NEW PARAMETER
) -> List[SearchResult]:
    """Query expert for similar changes.
    
    Searches the expert's indexed commit history for changes similar to the
    provided prompt using semantic search with embeddings.
    
    Args:
        expert_name: Name of the expert to query
        prompt: Search query describing what to look for
        max_changes: Maximum number of changelists to return
        max_file_chunks: Maximum number of file chunks to return
        users: Optional list of authors to filter results by
        files: Optional list of file paths to filter results by
        search_scope: Search scope - "metadata", "diffs", "files", or "all" (default)
        min_score: Minimum similarity score threshold 0.0-1.0 (default: 0.1)
        relative_threshold: Relative threshold from top score 0.0-1.0 (default: 0.3)
        data_dir: Optional custom data directory path
        embedding_provider: Embedding provider - "local" or "bedrock" (default: "local")
        enable_reranking: Whether to enable cross-encoder reranking (default: True)
        
    Returns:
        List of SearchResult objects ordered by similarity score (highest first).
        Each result contains the changelist and similarity score.
        
    Raises:
        ExpertNotFoundError: If the expert does not exist
        ValueError: If search parameters are invalid
        
    Example:
        ```python
        results = query_expert(
            expert_name="MyExpert",
            prompt="How to add a new feature?",
            max_changes=10,
            users=["john", "jane"],
            min_score=0.3
        )
        
        for result in results:
            print(f"Score: {result.similarity_score:.3f}")
            print(f"Message: {result.changelist.message}")
        ```
    """
    # Validate search_scope
    search_scope_lower = search_scope.lower()
    if search_scope_lower not in ("metadata", "diffs", "files", "all"):
        raise ValueError(f"Invalid search_scope: {search_scope}. Must be 'metadata', 'diffs', 'files', or 'all'")
    
    # Initialize context
    ctx = ExpertContext(
        expert_name=expert_name,
        data_dir=data_dir,
        embedding_provider=embedding_provider
    )
    
    try:
        # Check expert exists
        if not ctx.metadata_db.exists():
            raise ExpertNotFoundError(expert_name)
        
        # Verify expert exists in database
        expert_info = ctx.metadata_db.get_expert(expert_name)
        if not expert_info:
            raise ExpertNotFoundError(expert_name)
        
        # Initialize vector DB
        ctx.vector_db.initialize(
            dimension=ctx.embedder.dimension,
            require_exists=True
        )
        
        # Determine search scope
        enable_metadata_search = search_scope_lower in ("metadata", "all")
        enable_diff_search = search_scope_lower in ("diffs", "all")
        enable_file_search = search_scope_lower in ("files", "all")
        
        # Create reranker if enabled
        from expert_among_us.config.settings import Settings
        settings = Settings(
            embedding_provider=embedding_provider,
            enable_reranking=enable_reranking
        )

        reranker = None
        if enable_reranking:
            from expert_among_us.reranking.factory import create_reranker
            reranker = create_reranker(settings)

        # Create searcher
        searcher = Searcher(
            expert_name=expert_name,
            embedder=ctx.embedder,
            metadata_db=ctx.metadata_db,
            vector_db=ctx.vector_db,
            reranker=reranker,  # NEW
            enable_metadata_search=enable_metadata_search,
            enable_diff_search=enable_diff_search,
            enable_file_search=enable_file_search,
            enable_reranking=enable_reranking,  # NEW
            min_similarity_score=min_score,
            relative_threshold=relative_threshold
        )
        
        # Create query parameters
        params = QueryParams(
            prompt=prompt,
            max_changes=max_changes,
            max_file_chunks=max_file_chunks,
            users=users,
            files=files,
            amogus=False
        )
        
        # Execute search
        results = searcher.search(params)
        
        # Clean up searcher
        searcher.close()
        
        return results
        
    finally:
        ctx.close()


def list_experts(
    data_dir: Optional[Path] = None
) -> List[ExpertInfo]:
    """List all available experts.
    
    Scans the data directory for expert databases and collects metadata
    about each expert including commit counts and time ranges.
    
    Args:
        data_dir: Optional custom data directory path. If not provided,
                  uses default ~/.expert-among-us
        
    Returns:
        List of ExpertInfo objects containing metadata about each expert.
        Returns empty list if no experts are found.
        
    Raises:
        ValueError: If data directory path is invalid
        
    Example:
        ```python
        experts = list_experts()
        
        for expert in experts:
            print(f"Name: {expert.name}")
            print(f"Commits: {expert.commit_count}")
            print(f"Last indexed: {expert.last_indexed_at}")
        ```
    """
    # Determine data directory
    if data_dir is None:
        data_dir = Path.home() / ".expert-among-us"
    
    experts_dir = data_dir / "data"
    
    # Return empty list if directory doesn't exist
    if not experts_dir.exists():
        return []
    
    # Find all expert databases
    expert_dirs = [
        d for d in experts_dir.iterdir() 
        if d.is_dir() and (d / "metadata.db").exists()
    ]
    
    # Collect expert information
    experts_info = []
    for expert_dir in expert_dirs:
        expert_name = expert_dir.name
        metadata_db = SQLiteMetadataDB(expert_name, data_dir=data_dir)
        
        try:
            # Get expert details
            experts = metadata_db.list_all_experts()
            if experts:
                expert = experts[0]  # Should only be one expert per database
                commit_count = metadata_db.get_commit_count(expert_name)
                
                experts_info.append(ExpertInfo(
                    name=expert['name'],
                    vcs_type=expert['vcs_type'],
                    workspace_path=expert['workspace_path'],
                    subdirs=expert['subdirs'],
                    commit_count=commit_count,
                    last_indexed_at=expert['last_indexed_at'],
                    last_processed_commit_hash=expert['last_processed_commit_hash'],
                    first_processed_commit_hash=expert['first_processed_commit_hash']
                ))
        finally:
            metadata_db.close()
    
    return experts_info


def import_expert(
    source_path: Path,
    data_dir: Optional[Path] = None
) -> str:
    """Import an expert via symlink.
    
    Creates a symlink in the data directory pointing to an external expert
    directory. This allows experts to be stored outside the default data
    directory while still being accessible.
    
    Args:
        source_path: Path to the expert directory to import. Must contain
                     a valid metadata.db file.
        data_dir: Optional custom data directory path. If not provided,
                  uses default ~/.expert-among-us
        
    Returns:
        The expert name that was imported (derived from source directory name)
        
    Raises:
        InvalidExpertError: If source directory doesn't contain metadata.db
        ExpertAlreadyExistsError: If an expert with the same name already exists
        PermissionError: On Windows if symlink permissions are missing
        FileNotFoundError: If source_path doesn't exist
        
    Example:
        ```python
        # Import an expert from external storage
        expert_name = import_expert(Path("/external/storage/MyExpert"))
        print(f"Imported: {expert_name}")
        ```
        
    Note:
        On Windows, creating symlinks requires either:
        1. Administrator privileges, OR
        2. Developer Mode enabled (Settings > System > For Developers)
    """
    # Resolve source path
    source_path = source_path.resolve()
    
    # Check if source exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    
    if not source_path.is_dir():
        raise ValueError(f"Source path is not a directory: {source_path}")
    
    # Determine data directory
    if data_dir is None:
        data_dir = Path.home() / ".expert-among-us"
    
    # Ensure data directory exists
    experts_dir = data_dir / "data"
    experts_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate source has metadata.db
    metadata_db_path = source_path / "metadata.db"
    if not metadata_db_path.exists():
        raise InvalidExpertError(
            expert_name=source_path.name,
            reason="No metadata.db found",
            message=(
                f"Invalid expert directory: {source_path}\n"
                f"No metadata.db found. The source must be a valid expert directory."
            )
        )
    
    # Extract expert name from source directory name
    expert_name = source_path.name
    target_path = experts_dir / expert_name
    
    # Check if target already exists (directory or symlink)
    if target_path.exists() or target_path.is_symlink():
        existing_path = target_path.resolve() if target_path.is_symlink() else target_path
        raise ExpertAlreadyExistsError(
            expert_name=expert_name,
            message=(
                f"Expert '{expert_name}' already exists in data directory.\n"
                f"Target path: {target_path}\n"
                f"Existing path: {existing_path}"
            )
        )
    
    # Create symlink
    try:
        target_path.symlink_to(source_path, target_is_directory=True)
    except OSError as e:
        # Handle Windows permission errors
        if sys.platform == "win32" and "privilege" in str(e).lower():
            raise PermissionError(
                f"Failed to create symlink: {e}\n"
                "On Windows, creating symlinks requires:\n"
                "  1. Administrator privileges, OR\n"
                "  2. Developer Mode enabled (Settings > System > For Developers)"
            )
        raise
    
    return expert_name