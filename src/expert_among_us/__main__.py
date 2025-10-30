"""CLI entry point for Expert Among Us."""

import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from expert_among_us import __version__
from expert_among_us.config.settings import Settings
from expert_among_us.models.query import QueryParams
from expert_among_us.models.expert import ExpertConfig
from expert_among_us.core.searcher import Searcher
from expert_among_us.embeddings.bedrock import BedrockEmbedder
from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.db.vector.chroma import ChromaVectorDB
from expert_among_us.vcs.detector import detect_vcs
from expert_among_us.llm.factory import create_llm_provider
from expert_among_us.utils.progress import (
    log_error,
    log_info,
    log_success,
    create_progress_bar,
    update_progress
)
from expert_among_us.utils.truncate import truncate_diff_for_embedding

console = Console()


def create_embedder(provider: str, settings: Settings):
    """Factory function to create embedder based on provider.
    
    Args:
        provider: "local" for Jina Code embeddings or "bedrock" for AWS Titan
        settings: Settings instance with configuration
        
    Returns:
        Embedder instance
    """
    if provider == "local":
        from expert_among_us.embeddings.local import JinaCodeEmbedder
        return JinaCodeEmbedder(
            model_id=settings.local_embedding_model,
            dimension=settings.local_embedding_dimension
        )
    elif provider == "bedrock":
        return BedrockEmbedder(model_id=settings.embedding_model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def _process_changelist_batch(
    changelists,
    expert_config: ExpertConfig,
    embedder,
    metadata_db,
    vector_db,
    max_embedding_tokens: int
) -> None:
    """Helper function to process a batch of changelists: generate embeddings and store.
    
    Args:
        changelists: List of changelists to process in batch
        expert_config: Expert configuration
        embedder: Embedder instance
        metadata_db: Metadata database instance
        vector_db: Vector database instance
        max_embedding_tokens: Maximum tokens for embedding
    """
    if not changelists:
        return
    
    # Prepare texts for batch embedding
    metadata_texts = []
    diff_texts = []
    diff_indices = []  # Track which changelists have diffs
    
    for i, changelist in enumerate(changelists):
        # Prepare metadata text
        if expert_config.embed_metadata:
            metadata_text = changelist.get_metadata_text()
            metadata_text_truncated, _ = truncate_diff_for_embedding(
                metadata_text,
                max_bytes=expert_config.max_embedding_text_size,
                max_tokens=max_embedding_tokens
            )
            metadata_texts.append(metadata_text_truncated)
        
        # Prepare diff text
        if expert_config.embed_diffs and changelist.diff:
            diff_truncated, _ = truncate_diff_for_embedding(
                changelist.diff,
                max_bytes=expert_config.max_embedding_text_size,
                max_tokens=max_embedding_tokens
            )
            diff_texts.append(diff_truncated)
            diff_indices.append(i)
    
    # Generate metadata embeddings in batch
    if metadata_texts and expert_config.embed_metadata:
        metadata_embeddings = embedder.embed_batch(metadata_texts)
        for i, embedding in enumerate(metadata_embeddings):
            changelists[i].metadata_embedding = embedding
    
    # Generate diff embeddings in batch
    if diff_texts and expert_config.embed_diffs:
        diff_embeddings = embedder.embed_batch(diff_texts)
        for i, embedding in enumerate(diff_embeddings):
            changelist_idx = diff_indices[i]
            changelists[changelist_idx].diff_embedding = embedding
    
    # Clear GPU cache after batch processing
    if hasattr(embedder, 'device') and embedder.device == "cuda" and hasattr(embedder, 'torch'):
        embedder.torch.cuda.empty_cache()
    
    # Store all changelists in metadata database
    metadata_db.insert_changelists(changelists)
    
    # Store all embeddings in vector database
    metadata_vectors = []
    diff_vectors = []
    
    for changelist in changelists:
        if changelist.metadata_embedding:
            metadata_vectors.append((changelist.id, changelist.metadata_embedding))
        if changelist.diff_embedding:
            diff_vectors.append((f"{changelist.id}_diff", changelist.diff_embedding))
    
    if metadata_vectors:
        vector_db.insert_vectors(metadata_vectors)
    if diff_vectors:
        vector_db.insert_vectors(diff_vectors)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging for all Bedrock API calls')
@click.option('--data-dir', type=click.Path(path_type=Path), help='Base directory for expert data storage (default: ~/.expert-among-us)')
@click.option('--llm-provider', type=click.Choice(['openai', 'openrouter', 'local', 'bedrock', 'claude-code']), help='LLM provider for AI recommendations (required for prompt command)')
@click.option('--embedding-provider', type=click.Choice(['local', 'bedrock']), default='local', help='Embedding provider: local=Jina Code, bedrock=AWS Titan (default: local)')
@click.version_option(version=__version__)
@click.pass_context
def main(ctx, debug: bool, data_dir: Optional[Path], llm_provider: str, embedding_provider: str) -> None:
    """Expert Among Us - Queryable expert from commit history using LLM and embeddings.
    
    Create experts from git repositories, search commit history, and get
    AI-powered recommendations based on historical development patterns.
    
    Global Options:
        --llm-provider: Choose LLM provider (required for prompt command)
            - openai: OpenAI API (requires OPENAI_API_KEY)
            - openrouter: OpenRouter API (requires OPENROUTER_API_KEY)
            - local: Local LLM server (requires LOCAL_LLM_BASE_URL)
            - bedrock: AWS Bedrock (requires AWS credentials)
            - claude-code: Claude Code CLI (requires claude CLI)
        --embedding-provider: Choose embedding provider (local or bedrock)
    """
    from expert_among_us.utils.debug import DebugLogger
    
    # Store global options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['data_dir'] = data_dir.expanduser().resolve() if data_dir else None
    ctx.obj['llm_provider'] = llm_provider
    ctx.obj['embedding_provider'] = embedding_provider
    
    # Initialize debug logger if enabled
    if debug:
        DebugLogger.configure(enabled=True)


@main.command()
@click.argument("workspace", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("expert_name", type=str)
@click.argument("subdirs", nargs=-1, type=str)
@click.option("--max-commits", default=10000, type=int, help="Maximum commits to index")
@click.option("--vcs-type", type=click.Choice(["git", "p4"]), default="git", help="Version control system type")
@click.option("--batch-size", default=10, type=int, help="Number of commits to process per batch (default: 10)")
@click.pass_context
def populate(
    ctx,
    workspace: Path,
    expert_name: str,
    subdirs: tuple[str, ...],
    max_commits: int,
    vcs_type: str,
    batch_size: int,
) -> None:
    """Build or update an expert index from a repository.
    
    WORKSPACE: Path to the repository root
    
    EXPERT_NAME: Unique name for this expert
    
    SUBDIRS: Optional subdirectories to filter (e.g., src/main/ src/resources/)
    
    Examples:
    
        \b
        # Index entire repository
        $ expert-among-us populate /path/to/repo MyExpert
        
        \b
        # Index specific subdirectories
        $ expert-among-us populate /path/to/repo MyExpert src/main/ src/resources/
        
        \b
        # Use Bedrock embeddings (global flag)
        $ expert-among-us --embedding-provider bedrock populate /path/to/repo MyExpert
    """
    log_info(f"Populating expert index '{expert_name}' from workspace: {workspace}")
    if subdirs:
        log_info(f"Filtering to subdirectories: {', '.join(subdirs)}")
    
    try:
        # Get global options from context
        data_dir = ctx.obj.get('data_dir')
        embedding_provider = ctx.obj.get('embedding_provider')
        
        # Step 1: Initialize components and settings
        # Only pass data_dir if it's not None to allow default_factory to work
        settings_kwargs = {'embedding_provider': embedding_provider}
        if data_dir is not None:
            settings_kwargs['data_dir'] = data_dir
        
        settings = Settings(**settings_kwargs)
        
        log_info("Initializing components...")
        log_info(f"Using embedding provider: {embedding_provider}")
        if data_dir:
            log_info(f"Using data directory: {data_dir}")
        
        # Initialize embedder
        embedder = create_embedder(embedding_provider, settings)
        
        # Determine max tokens based on provider
        max_embedding_tokens = (
            settings.local_embedding_max_tokens
            if embedding_provider == "local"
            else settings.bedrock_embedding_max_tokens
        )
        
        # Create expert configuration
        expert_config = ExpertConfig(
            name=expert_name,
            workspace_path=workspace,
            subdirs=list(subdirs) if subdirs else [],
            vcs_type=vcs_type,
            max_commits=max_commits,
            data_dir=data_dir or Path.home() / ".expert-among-us"
        )
        
        # Ensure storage directories exist
        expert_config.ensure_storage_exists()
        
        # Initialize databases
        metadata_db = SQLiteMetadataDB(expert_name, data_dir=data_dir)
        metadata_db.initialize()
        vector_db = ChromaVectorDB(expert_name, data_dir=data_dir)
        vector_db.initialize(dimension=embedder.dimension)
        
        # Step 2: Detect and initialize VCS provider
        log_info(f"Detecting version control system in {workspace}...")
        vcs_provider = detect_vcs(str(workspace))
        
        if vcs_provider is None:
            log_error(f"No supported VCS detected in {workspace}")
            log_error("Supported VCS: Git")
            sys.exit(1)
        
        log_success(f"Detected VCS: {vcs_provider.__class__.__name__}")
        
        # Step 3: Check if expert already exists and get commit time boundaries
        existing_expert = metadata_db.get_expert(expert_name)
        last_commit_time = None
        first_commit_time = None
        
        if existing_expert:
            log_info(f"Updating existing expert '{expert_name}'")
            last_commit_time = existing_expert.get('last_commit_time')
            first_commit_time = existing_expert.get('first_commit_time')
            if last_commit_time:
                log_info(f"Last indexed commit: {last_commit_time}")
            if first_commit_time:
                log_info(f"First indexed commit: {first_commit_time}")
        else:
            log_info(f"Creating new expert '{expert_name}'")
            # Store initial expert configuration
            metadata_db.create_expert(expert_config.name, str(expert_config.workspace_path), expert_config.subdirs, expert_config.vcs_type)
        
        subdirs_list = list(subdirs) if subdirs else None
        total_processed = 0
        
        # Step 4: Phase 1 - Process newer commits (if last_commit_time exists)
        if last_commit_time is not None:
            log_info("Phase 1: Processing newer commits...")
            
            newer_changelists = vcs_provider.get_commits(
                workspace_path=str(workspace),
                subdirs=subdirs_list,
                max_commits=max_commits,
                since=last_commit_time
            )
            
            if newer_changelists:
                log_success(f"Found {len(newer_changelists)} newer commits to process")
                
                progress, task_id = create_progress_bar(
                    description="Processing newer commits",
                    total=len(newer_changelists)
                )
                progress.start()
                
                # Track the maximum (newest) timestamp processed so far
                current_max_timestamp = last_commit_time
                
                # Batch processing
                batch = []
                
                try:
                    for changelist in newer_changelists:
                        # Skip if already processed (handles interrupted runs)
                        existing = metadata_db.get_changelist(changelist.id)
                        if existing:
                            # Still update timestamp to advance past this commit
                            if changelist.timestamp > current_max_timestamp:
                                current_max_timestamp = changelist.timestamp
                                metadata_db.update_expert_last_commit(expert_config.name, current_max_timestamp)
                            update_progress(
                                progress,
                                task_id,
                                advance=1,
                                description=f"Skipping {changelist.id[:12]}..."
                            )
                            continue
                        
                        # Add to batch
                        batch.append(changelist)
                        
                        # Process batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            _process_changelist_batch(batch, expert_config, embedder, metadata_db, vector_db, max_embedding_tokens)
                            total_processed += len(batch)
                            
                            # Update timestamps
                            for cl in batch:
                                if cl.timestamp > current_max_timestamp:
                                    current_max_timestamp = cl.timestamp
                                    metadata_db.update_expert_last_commit(expert_config.name, current_max_timestamp)
                            
                            update_progress(
                                progress,
                                task_id,
                                advance=len(batch),
                                description=f"Processed batch of {len(batch)} commits..."
                            )
                            batch = []
                    
                    # Process remaining partial batch
                    if batch:
                        _process_changelist_batch(batch, expert_config, embedder, metadata_db, vector_db, max_embedding_tokens)
                        total_processed += len(batch)
                        
                        # Update timestamps
                        for cl in batch:
                            if cl.timestamp > current_max_timestamp:
                                current_max_timestamp = cl.timestamp
                                metadata_db.update_expert_last_commit(expert_config.name, current_max_timestamp)
                        
                        update_progress(
                            progress,
                            task_id,
                            advance=len(batch),
                            description=f"Processed final batch of {len(batch)} commits..."
                        )
                finally:
                    progress.stop()
                
                log_info(f"Phase 1 complete - indexed up to {current_max_timestamp.isoformat()}")
            else:
                log_info("No newer commits found")
        
        # Step 5: Phase 2 - Process older commits
        oldest_commit_time = None
        
        if first_commit_time is not None:
            # Existing expert with first_commit_time: fetch commits before it
            log_info("Phase 2: Processing older commits...")
            
            older_changelists = vcs_provider.get_commits_before(
                workspace_path=str(workspace),
                before=first_commit_time,
                subdirs=subdirs_list,
                limit=max_commits
            )
            
            if older_changelists:
                log_success(f"Found {len(older_changelists)} older commits to process")
                
                progress, task_id = create_progress_bar(
                    description="Processing older commits",
                    total=len(older_changelists)
                )
                progress.start()
                
                # Track the minimum (oldest) timestamp processed so far
                current_min_timestamp = first_commit_time
                
                # Batch processing
                batch = []
                
                try:
                    for changelist in older_changelists:
                        # Skip if already processed (handles interrupted runs)
                        existing = metadata_db.get_changelist(changelist.id)
                        if existing:
                            # Still update timestamp to advance past this commit
                            if changelist.timestamp < current_min_timestamp:
                                current_min_timestamp = changelist.timestamp
                                metadata_db.update_expert_first_commit(expert_config.name, current_min_timestamp)
                            update_progress(
                                progress,
                                task_id,
                                advance=1,
                                description=f"Skipping {changelist.id[:12]}..."
                            )
                            continue
                        
                        # Add to batch
                        batch.append(changelist)
                        
                        # Process batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            _process_changelist_batch(batch, expert_config, embedder, metadata_db, vector_db, max_embedding_tokens)
                            total_processed += len(batch)
                            
                            # Update timestamps
                            for cl in batch:
                                if cl.timestamp < current_min_timestamp:
                                    current_min_timestamp = cl.timestamp
                                    metadata_db.update_expert_first_commit(expert_config.name, current_min_timestamp)
                            
                            update_progress(
                                progress,
                                task_id,
                                advance=len(batch),
                                description=f"Processed batch of {len(batch)} commits..."
                            )
                            batch = []
                    
                    # Process remaining partial batch
                    if batch:
                        _process_changelist_batch(batch, expert_config, embedder, metadata_db, vector_db, max_embedding_tokens)
                        total_processed += len(batch)
                        
                        # Update timestamps
                        for cl in batch:
                            if cl.timestamp < current_min_timestamp:
                                current_min_timestamp = cl.timestamp
                                metadata_db.update_expert_first_commit(expert_config.name, current_min_timestamp)
                        
                        update_progress(
                            progress,
                            task_id,
                            advance=len(batch),
                            description=f"Processed final batch of {len(batch)} commits..."
                        )
                finally:
                    progress.stop()
                
                log_info(f"Phase 2 complete - indexed back to {current_min_timestamp.isoformat()}")
            else:
                log_info("No older commits found")
        else:
            # First run: fetch recent commits
            log_info("Phase 2: Initial indexing - fetching recent commits...")
            
            recent_changelists = vcs_provider.get_commits(
                workspace_path=str(workspace),
                subdirs=subdirs_list,
                max_commits=max_commits,
                since=None
            )
            
            if recent_changelists:
                log_success(f"Found {len(recent_changelists)} commits to process")
                
                progress, task_id = create_progress_bar(
                    description="Processing recent commits",
                    total=len(recent_changelists)
                )
                progress.start()
                
                # Track both max (newest) and min (oldest) timestamps
                current_max_timestamp = None
                current_min_timestamp = None
                
                # Batch processing
                batch = []
                
                try:
                    for changelist in recent_changelists:
                        # Skip if already processed (handles interrupted runs)
                        existing = metadata_db.get_changelist(changelist.id)
                        if existing:
                            # Still update timestamps to advance past this commit
                            if current_max_timestamp is None or changelist.timestamp > current_max_timestamp:
                                current_max_timestamp = changelist.timestamp
                                metadata_db.update_expert_last_commit(expert_config.name, current_max_timestamp)
                            
                            if current_min_timestamp is None or changelist.timestamp < current_min_timestamp:
                                current_min_timestamp = changelist.timestamp
                                metadata_db.update_expert_first_commit(expert_config.name, current_min_timestamp)
                            
                            update_progress(
                                progress,
                                task_id,
                                advance=1,
                                description=f"Skipping {changelist.id[:12]}..."
                            )
                            continue
                        
                        # Add to batch
                        batch.append(changelist)
                        
                        # Process batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            _process_changelist_batch(batch, expert_config, embedder, metadata_db, vector_db, max_embedding_tokens)
                            total_processed += len(batch)
                            
                            # Update timestamps
                            for cl in batch:
                                if current_max_timestamp is None or cl.timestamp > current_max_timestamp:
                                    current_max_timestamp = cl.timestamp
                                    metadata_db.update_expert_last_commit(expert_config.name, current_max_timestamp)
                                
                                if current_min_timestamp is None or cl.timestamp < current_min_timestamp:
                                    current_min_timestamp = cl.timestamp
                                    metadata_db.update_expert_first_commit(expert_config.name, current_min_timestamp)
                            
                            update_progress(
                                progress,
                                task_id,
                                advance=len(batch),
                                description=f"Processed batch of {len(batch)} commits..."
                            )
                            batch = []
                    
                    # Process remaining partial batch
                    if batch:
                        _process_changelist_batch(batch, expert_config, embedder, metadata_db, vector_db, max_embedding_tokens)
                        total_processed += len(batch)
                        
                        # Update timestamps
                        for cl in batch:
                            if current_max_timestamp is None or cl.timestamp > current_max_timestamp:
                                current_max_timestamp = cl.timestamp
                                metadata_db.update_expert_last_commit(expert_config.name, current_max_timestamp)
                            
                            if current_min_timestamp is None or cl.timestamp < current_min_timestamp:
                                current_min_timestamp = cl.timestamp
                                metadata_db.update_expert_first_commit(expert_config.name, current_min_timestamp)
                        
                        update_progress(
                            progress,
                            task_id,
                            advance=len(batch),
                            description=f"Processed final batch of {len(batch)} commits..."
                        )
                finally:
                    progress.stop()
                
                log_info(f"Initial indexing complete - processed {len(recent_changelists)} commits")
            else:
                log_info("No commits found")
        
        # Update expert index time
        metadata_db.update_expert_index_time(expert_config.name, datetime.now(timezone.utc))
        
        # Step 6: Report results
        if total_processed > 0:
            log_success(f"Indexing complete!")
            log_info(f"Successfully processed: {total_processed} commits")
        else:
            log_info("No new commits to index")
        
        # Cleanup
        metadata_db.close()
        vector_db.close()
        
    except Exception as e:
        log_error(f"Failed to populate expert: {str(e)}")
        raise  # Always reraise to show full stack trace


@main.command()
@click.argument("expert_name", type=str)
@click.argument("prompt", type=str)
@click.option("--max-changes", default=10, type=int, help="Maximum results to return")
@click.option("--users", type=str, help="Filter by authors (comma-separated)")
@click.option("--files", type=str, help="Filter by files (comma-separated)")
@click.option("--output", type=click.Path(path_type=Path), help="Output JSON file")
@click.option("--search-scope",
              type=click.Choice(["both", "metadata", "diffs"], case_sensitive=False),
              default="both",
              help="Search scope: both (default), metadata only, or diffs only")
@click.option("--min-score",
              type=float,
              default=0.1,
              help="Minimum similarity score threshold (0.0-1.0, default: 0.1)")
@click.option("--relative-threshold",
              type=float,
              default=0.3,
              help="Relative score threshold as percentage drop from top result (0.0-1.0, default: 0.3)")
@click.pass_context
def query(
    ctx,
    expert_name: str,
    prompt: str,
    max_changes: int,
    users: Optional[str],
    files: Optional[str],
    output: Optional[Path],
    search_scope: str,
    min_score: float,
    relative_threshold: float,
) -> None:
    """Search for similar changes in the expert's history.
    
    EXPERT_NAME: Name of the expert to query
    
    PROMPT: Search query (e.g., "How to add a new arrow type?")
    
    Options:
        --search-scope: Control which embeddings to search
            - both: Search both metadata and diff embeddings (default)
            - metadata: Search only metadata embeddings
            - diffs: Search only diff embeddings
        
        --min-score: Absolute minimum similarity score threshold (default: 0.1)
            - Applied first, filters out results below this absolute threshold
            - Range: 0.0 (include all) to 1.0 (only perfect matches)
        
        --relative-threshold: Relative threshold as percentage drop from top result (default: 0.3)
            - Applied second, after absolute threshold
            - Filters results based on percentage of top score
            - Calculation: cutoff = top_score * (1 - relative_threshold)
            - Range: 0.0 (no relative filtering) to 1.0 (filters everything)
            - Makes filtering adaptive to query quality
    
    Filter Order:
        Both thresholds are applied sequentially:
        1. First: Apply --min-score (absolute threshold)
        2. Second: Apply --relative-threshold (percentage of top score)
    
    Examples:
    
        \b
        # Basic query (searches both metadata and diffs)
        $ expert-among-us query MyExpert "How to add new feature?"
        
        \b
        # Use Bedrock embeddings (global flag)
        $ expert-among-us --embedding-provider bedrock query MyExpert "How to..."
        
        \b
        # Search only metadata embeddings
        $ expert-among-us query MyExpert "What changed?" --search-scope metadata
        
        \b
        # Search only diff embeddings
        $ expert-among-us query MyExpert "Code changes" --search-scope diffs
        
        \b
        # Strict filtering (only results very close to top)
        $ expert-among-us query MyExpert "fixed nullptr" --relative-threshold 0.1
        
        \b
        # Lenient filtering (allow wider range)
        $ expert-among-us query MyExpert "fixed nullptr" --relative-threshold 0.5
        
        \b
        # Combined filtering
        $ expert-among-us query MyExpert "fixed nullptr" --min-score 0.2 --relative-threshold 0.3
        
        \b
        # Only absolute threshold
        $ expert-among-us query MyExpert "Bug fix" --min-score 0.5 --relative-threshold 1.0
        
        \b
        # Only relative threshold
        $ expert-among-us query MyExpert "Bug fix" --min-score 0.0 --relative-threshold 0.2
        
        \b
        # With filters and output
        $ expert-among-us query MyExpert "Bug fix needed" \\
            --users john,jane --files src/main.py --output results.json
    """
    log_info(f"Querying similar changes in '{expert_name}'")
    log_info(f"Query: {prompt}")
    
    try:
        # Get global options from context
        data_dir = ctx.obj.get('data_dir')
        embedding_provider = ctx.obj.get('embedding_provider')
        
        # Parse filters
        user_list = [u.strip() for u in users.split(",")] if users else None
        file_list = [f.strip() for f in files.split(",")] if files else None
        
        if user_list:
            log_info(f"Filtering by users: {', '.join(user_list)}")
        if file_list:
            log_info(f"Filtering by files: {', '.join(file_list)}")
        
        # Create query parameters
        params = QueryParams(
            prompt=prompt,
            max_changes=max_changes,
            users=user_list,
            files=file_list,
            amogus=False
        )
        
        # Initialize components and settings
        # Only pass data_dir if it's not None to allow default_factory to work
        settings_kwargs = {'embedding_provider': embedding_provider}
        if data_dir is not None:
            settings_kwargs['data_dir'] = data_dir
        
        settings = Settings(**settings_kwargs)
        
        log_info("Initializing search components...")
        log_info(f"Using embedding provider: {embedding_provider}")
        if data_dir:
            log_info(f"Using data directory: {data_dir}")
        embedder = create_embedder(embedding_provider, settings)
        metadata_db = SQLiteMetadataDB(expert_name, data_dir=data_dir)
        
        # Check if expert database exists before proceeding
        if not metadata_db.exists():
            log_error(f"Expert '{expert_name}' does not exist. Please run 'populate' command first to create the expert index.")
            sys.exit(1)
        
        vector_db = ChromaVectorDB(expert_name, data_dir=data_dir)
        vector_db.initialize(dimension=embedder.dimension, require_exists=True)
        
        # Determine search scope
        search_scope_lower = search_scope.lower()
        enable_metadata_search = search_scope_lower in ("both", "metadata")
        enable_diff_search = search_scope_lower in ("both", "diffs")
        
        log_info(f"Search scope: {search_scope} (metadata={enable_metadata_search}, diffs={enable_diff_search})")
        
        # Create searcher
        searcher = Searcher(
            expert_name=expert_name,
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            enable_metadata_search=enable_metadata_search,
            enable_diff_search=enable_diff_search,
            min_similarity_score=min_score,
            relative_threshold=relative_threshold
        )
        
        # Perform search
        results = searcher.search(params)
        
        # Display results
        if results:
            log_success(f"Found {len(results)} matching changes")
            
            # Create rich table
            table = Table(title=f"Search Results for '{expert_name}'")
            table.add_column("ID", style="cyan", no_wrap=True, width=12)
            table.add_column("Author", style="green")
            table.add_column("Score", style="yellow", justify="right")
            table.add_column("Source", style="magenta")
            table.add_column("Timestamp", style="blue", width=19)
            table.add_column("Message", style="white")
            
            for result in results:
                table.add_row(
                    result.changelist.id[:12],
                    result.changelist.author,
                    f"{result.similarity_score:.3f}",
                    result.source,
                    result.changelist.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    result.changelist.message[:60] + "..." if len(result.changelist.message) > 60 else result.changelist.message
                )
            
            console.print(table)
            
            # Output to JSON if requested
            if output:
                output_data = {
                    "query": prompt,
                    "expert": expert_name,
                    "filters": {
                        "users": user_list,
                        "files": file_list
                    },
                    "results": [
                        {
                            "id": r.changelist.id,
                            "author": r.changelist.author,
                            "timestamp": r.changelist.timestamp.isoformat(),
                            "message": r.changelist.message,
                            "files": r.changelist.files,
                            "diff": r.changelist.diff,
                            "similarity_score": r.similarity_score,
                            "source": r.source
                        }
                        for r in results
                    ]
                }
                
                output.parent.mkdir(parents=True, exist_ok=True)
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                log_success(f"Results saved to {output}")
        else:
            log_info("No matching changes found")
        
        # Cleanup
        searcher.close()
        
    except Exception as e:
        log_error(f"Search failed: {str(e)}")
        raise  # Always reraise to show full stack trace


@main.command()
@click.argument("expert_name", type=str)
@click.argument("prompt", type=str)
@click.option("--max-changes", default=10, type=int, help="Maximum context changes to use")
@click.option("--users", type=str, help="Filter by authors (comma-separated)")
@click.option("--files", type=str, help="Filter by files (comma-separated)")
@click.option("--amogus", is_flag=True, help="Enable Among Us mode")
@click.option("--temperature", default=0.7, type=float, help="LLM temperature (0.0-1.0)")
@click.pass_context
def prompt(
    ctx,
    expert_name: str,
    prompt: str,
    max_changes: int,
    users: Optional[str],
    files: Optional[str],
    amogus: bool,
    temperature: float,
) -> None:
    """Get AI recommendations based on expert's historical patterns.
    
    EXPERT_NAME: Name of the expert to query
    
    PROMPT: Question or task description
    
    Examples:
    
        \b
        # Get recommendations
        $ expert-among-us prompt MyExpert "How should I implement X?"
        
        \b
        # Use OpenAI LLM provider
        $ expert-among-us --llm-provider openai prompt MyExpert "Implement X"
        
        \b
        # Use Claude Code LLM with Bedrock embeddings (global flags)
        $ expert-among-us --llm-provider claude-code --embedding-provider bedrock prompt MyExpert "Implement X"
        
        \b
        # With Among Us mode (occasionally gives bad advice)
        $ expert-among-us prompt MyExpert "Fix the bug" --amogus
        
        \b
        # With debug logging (global flag)
        $ expert-among-us --debug prompt MyExpert "Add feature"
    """
    import asyncio
    from expert_among_us.core.promptgen import PromptGenerator
    from expert_among_us.core.conversation import ConversationBuilder
    from expert_among_us.utils.debug import DebugLogger
    
    log_info(f"Generating AI recommendations for '{expert_name}'")
    log_info(f"Query: {prompt}")
    
    if amogus:
        log_info("🎮 Among Us mode enabled - beware of sabotage!")
    
    # Get global options from context
    debug = ctx.obj.get('debug', False)
    data_dir = ctx.obj.get('data_dir')
    llm_provider = ctx.obj.get('llm_provider')
    embedding_provider = ctx.obj.get('embedding_provider')
    
    try:
        # Step 1: Show debug logging info if enabled (already configured in main)
        if debug:
            log_info(f"Debug logging enabled: {DebugLogger._log_dir}")
        
        # Step 2: Parse filters
        user_list = [u.strip() for u in users.split(",")] if users else None
        file_list = [f.strip() for f in files.split(",")] if files else None
        
        if user_list:
            log_info(f"Filtering by users: {', '.join(user_list)}")
        if file_list:
            log_info(f"Filtering by files: {', '.join(file_list)}")
        
        # Step 3: Initialize components and settings
        # Only pass data_dir if it's not None to allow default_factory to work
        settings_kwargs = {
            'llm_provider': llm_provider,
            'embedding_provider': embedding_provider
        }
        if data_dir is not None:
            settings_kwargs['data_dir'] = data_dir
        
        settings = Settings(**settings_kwargs)
        
        log_info("Initializing components...")
        log_info(f"Using LLM provider: {llm_provider}")
        log_info(f"Using embedding provider: {embedding_provider}")
        if data_dir:
            log_info(f"Using data directory: {data_dir}")
        embedder = create_embedder(embedding_provider, settings)
        metadata_db = SQLiteMetadataDB(expert_name, data_dir=data_dir)
        
        # Check if expert database exists before proceeding
        if not metadata_db.exists():
            log_error(f"Expert '{expert_name}' does not exist. Please run 'populate' command first to create the expert index.")
            sys.exit(1)
        
        vector_db = ChromaVectorDB(expert_name, data_dir=data_dir)
        vector_db.initialize(dimension=embedder.dimension, require_exists=True)
        
        # Step 4: Verify expert exists
        expert_info = metadata_db.get_expert(expert_name)
        if not expert_info:
            log_error(f"Expert '{expert_name}' not found. Run 'populate' first.")
            sys.exit(1)
        
        # Step 5: Create searcher and find relevant examples
        log_info("Searching for relevant examples...")
        searcher = Searcher(
            expert_name=expert_name,
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db
        )
        
        params = QueryParams(
            prompt=prompt,
            max_changes=max_changes,
            users=user_list,
            files=file_list,
            amogus=False
        )
        
        search_results = searcher.search(params)
        
        if not search_results:
            log_error("No relevant examples found")
            searcher.close()
            sys.exit(1)
        
        log_success(f"Found {len(search_results)} relevant examples")
        
        # Step 6: Initialize LLM
        llm = create_llm_provider(settings, debug=debug)
        
        # Step 7: Generate prompts for changelists (with progress)
        prompt_gen = PromptGenerator(
            llm_provider=llm,
            metadata_db=metadata_db,
            model=settings.promptgen_model,
            max_diff_chars=2000
        )
        
        # Extract changelists from search results
        changelists = [result.changelist for result in search_results]
        
        # Generate prompts using the PromptGenerator method (includes progress bar and cache logging)
        prompt_results = prompt_gen.generate_prompts(changelists)
        
        # Step 8: Build conversation
        log_info("Building conversation context...")
        conv_builder = ConversationBuilder(prompt_generator=prompt_gen)
        system_prompt, messages = conv_builder.build_conversation(
            changelists=changelists,
            user_prompt=prompt,
            amogus=amogus
        )
        
        log_info(f"Using {len(changelists)} examples for context")
        console.print()
        
        # Step 9: Stream response
        async def stream_response():
            console.print("[bold cyan]Expert Response:[/bold cyan]\n")
            
            full_response = ""
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            cache_read = 0
            cache_creation = 0
            
            async for chunk in llm.stream(
                messages=messages,
                model=settings.expert_model,
                system=system_prompt,
                max_tokens=4096,
                temperature=temperature,
            ):
                if chunk.delta:
                    console.print(chunk.delta, end="")
                    full_response += chunk.delta
                
                if chunk.usage:
                    input_tokens = chunk.usage.input_tokens
                    output_tokens = chunk.usage.output_tokens
                    total_tokens = chunk.usage.total_tokens
                    cache_read = chunk.usage.cache_read_tokens
                    cache_creation = chunk.usage.cache_creation_tokens
            
            console.print("\n")
            
            # Show token usage and cache metrics only when debug is enabled
            if debug:
                # Use Bedrock's totalTokens for the actual API cost
                debug_msg = f"Tokens used: input={input_tokens}, output={output_tokens}, total={total_tokens}"
                
                # Always show cache metrics when prompt caching is enabled
                debug_msg += f" | Cache: read={cache_read}, creation={cache_creation}"
                
                console.print(f"[dim]{debug_msg}[/dim]")
                console.print(f"[dim]Debug logs written to: {DebugLogger._log_dir}[/dim]")
        
        asyncio.run(stream_response())
        
        # Cleanup
        searcher.close()
        
    except KeyboardInterrupt:
        log_info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        log_error(f"Failed to generate recommendations: {str(e)}")
        if debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()