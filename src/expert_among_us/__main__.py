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
    log_warning,
    create_progress_bar,
    update_progress
)
from expert_among_us.utils.truncate import truncate_diff_for_embedding

console = Console()


def create_embedder(provider: str, settings: Settings, compile_model: bool = True):
    """Factory function to create embedder based on provider.
    
    Args:
        provider: "local" for Jina Code embeddings or "bedrock" for AWS Titan
        settings: Settings instance with configuration
        compile_model: Whether to use torch.compile for local embedder (default: True)
        
    Returns:
        Embedder instance
    """
    if provider == "local":
        from expert_among_us.embeddings.local import JinaCodeEmbedder
        return JinaCodeEmbedder(
            model_id=settings.local_embedding_model,
            dimension=settings.local_embedding_dimension,
            compile_model=compile_model
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
    max_embedding_tokens: int,
    debug: bool = False
) -> None:
    """Helper function to process a batch of changelists: generate embeddings and store.
    
    Args:
        changelists: List of changelists to process in batch
        expert_config: Expert configuration
        embedder: Embedder instance
        metadata_db: Metadata database instance
        vector_db: Vector database instance
        max_embedding_tokens: Maximum tokens for embedding
        debug: Enable debug logging
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
                max_bytes=expert_config.max_metadata_embedding_size,
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
        metadata_start = time.perf_counter()
        metadata_embeddings = embedder.embed_batch(metadata_texts)
        metadata_embed_time = time.perf_counter() - metadata_start
        if debug:
            total_metadata_bytes = sum(len(text.encode('utf-8')) for text in metadata_texts)
            log_info(f"[DEBUG] Metadata embedding took {metadata_embed_time:.3f}s for {len(metadata_texts)} items ({total_metadata_bytes:,} bytes total)")
        for i, embedding in enumerate(metadata_embeddings):
            changelists[i].metadata_embedding = embedding
    
    # Generate diff embeddings in batch (WITH CHUNKING)
    if diff_texts and expert_config.embed_diffs:
        from expert_among_us.utils.chunking import chunk_text
        
        # Collect all chunks from all diffs
        all_chunks = []
        chunk_metadata = []  # (changelist_id, chunk_index)
        
        for i, diff_text in enumerate(diff_texts):
            changelist_idx = diff_indices[i]
            changelist = changelists[changelist_idx]
            
            # Split diff into chunks
            chunks = chunk_text(diff_text, chunk_size=expert_config.diff_chunk_size_bytes)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append((changelist.id, chunk_idx))
        
        # Embed all chunks in one batch
        diff_start = time.perf_counter()
        chunk_embeddings = embedder.embed_batch(all_chunks)
        diff_embed_time = time.perf_counter() - diff_start
        
        if debug:
            total_bytes = sum(len(c.encode('utf-8')) for c in all_chunks)
            log_info(f"[DEBUG] Chunked diff embedding: {len(all_chunks)} chunks "
                    f"from {len(diff_texts)} diffs in {diff_embed_time:.3f}s "
                    f"({total_bytes:,} bytes total)")
    
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
    
    # Store chunk embeddings with IDs like {commit_id}_diff_chunk_0
    if expert_config.embed_diffs and 'chunk_embeddings' in locals():
        chunk_vectors = [
            (f"{chunk_metadata[i][0]}_diff_chunk_{chunk_metadata[i][1]}", emb)
            for i, emb in enumerate(chunk_embeddings)
        ]
        if chunk_vectors:
            vector_db.insert_vectors(chunk_vectors)


def _process_commits_incrementally(
    vcs_provider,
    workspace_path: str,
    subdirs_list: Optional[list[str]],
    batch_size: int,
    expert_config,
    embedder,
    metadata_db,
    vector_db,
    max_embedding_tokens: int,
    is_newer: bool,
    boundary_hash: Optional[str],
    max_commits: int,
    track_both_boundaries: bool = False,
    debug: bool = False,
) -> tuple[int, Optional[datetime], Optional[str], Optional[datetime], Optional[str]]:
    """Process commits incrementally using topological traversal.
    
    Uses Git's default topological ordering with --no-merges to traverse all non-merge commits:
    - Phase 1 (newer): Moves boundary forward from last_commit_hash toward HEAD
    - Phase 2 (older): Moves boundary backward from first_commit_hash toward repo root
    
    After each batch, the boundary is updated based on Git's topological ordering (parent-child relationships).
    
    Args:
        vcs_provider: VCS provider instance
        workspace_path: Path to repository
        subdirs_list: Optional list of subdirectories to filter
        batch_size: Maximum commits per batch (also triggers embedding)
        expert_config: Expert configuration
        embedder: Embedder instance
        metadata_db: Metadata database instance
        vector_db: Vector database instance
        max_embedding_tokens: Maximum tokens for embedding
        is_newer: True for newer commits (after boundary), False for older (before boundary)
        boundary_hash: The boundary commit hash (last_commit_hash or first_commit_hash)
        max_commits: Maximum total commits to process
        track_both_boundaries: If True, track both boundaries (for initial indexing)
        
    Returns:
        Tuple of (total_processed, new_boundary_time, new_boundary_hash, optional_second_boundary_time, optional_second_boundary_hash)
    """
    total_processed = 0
    new_boundary_time = None
    new_boundary_hash = boundary_hash
    second_boundary_time = None
    second_boundary_hash = None
    
    # Track if we've set the last_commit during initial indexing
    last_commit_set = False
    
    # Moving boundary: updated after each batch to track progress
    current_boundary_hash = boundary_hash
    current_boundary_time = None
    
    # Cycle detection: track recent boundaries to detect infinite loops
    boundary_history = []
    max_boundary_history = 10
    
    # Counter for display purposes
    total_examined = 0
    
    while total_processed < max_commits:
        # Fetch one page of commits using the moving boundary
        # Boundary advances after each batch for continuous pagination
        print(f"\r{total_examined} commits - paging...    {current_boundary_time if current_boundary_time else ''}", end='', flush=True)
        if is_newer:
            if current_boundary_hash is None:
                # Initial indexing - fetch recent commits
                page = vcs_provider.get_commits_page(
                    workspace_path=workspace_path,
                    subdirs=subdirs_list,
                    page_size=batch_size,
                    since_hash=None,
                    debug=debug
                )
            else:
                # Fetch newer commits after current boundary
                page = vcs_provider.get_commits_page(
                    workspace_path=workspace_path,
                    subdirs=subdirs_list,
                    page_size=batch_size,
                    since_hash=current_boundary_hash,
                    debug=debug
                )
        else:
            # Fetch older commits before current boundary
            page = vcs_provider.get_commits_page_before(
                workspace_path=workspace_path,
                subdirs=subdirs_list,
                page_size=batch_size,
                before_hash=current_boundary_hash,
                debug=debug
            )
        
        # If no more commits, we're done
        if not page:
            print()  # Newline after progress
            break
        
        # Filter out already-indexed commits
        new_commits = []
        for changelist in page:
            total_examined += 1
            print(f"\r{total_examined} commits - fetching...  {changelist.timestamp}", end='', flush=True)
            
            existing = metadata_db.get_changelist(changelist.id)
            if existing:
                if debug:
                    log_info(f"\n[DEBUG] Found existing commit {changelist.id[:8]} at {changelist.timestamp}")
                # Skip already-indexed commits
                continue
            
            new_commits.append(changelist)
        
        # If no new commits in this page, advance the boundary and continue
        # This handles cases where merge commits pull in already-indexed commits
        if not new_commits:
            # Update boundary to skip past these already-indexed commits
            if page:
                # Use the first commit in the page to advance the boundary
                # Phase 1: page[0] is newest (git log returns newest-first)
                # Phase 2: page[0] is oldest (git log --reverse returns oldest-first)
                last_commit = page[0]
                current_boundary_hash = last_commit.id
                current_boundary_time = last_commit.timestamp
                
                # Check for boundary cycles (indicates we're stuck between branches)
                if current_boundary_hash in boundary_history:
                    if debug:
                        log_info(f"\n[DEBUG] Cycle detected: {current_boundary_hash[:8]} seen before in boundary history")
                        log_info(f"[DEBUG] Boundary history: {[h[:8] for h in boundary_history]}")
                    print()  # Newline after progress
                    break  # Exit the loop to prevent infinite cycling
                
                # Add to boundary history (ring buffer)
                boundary_history.append(current_boundary_hash)
                if len(boundary_history) > max_boundary_history:
                    boundary_history.pop(0)  # Remove oldest entry
                
                if debug:
                    direction = "forward" if is_newer else "backward"
                    phase = "Phase 1" if is_newer else "Phase 2"
                    commit_info = f"{current_boundary_hash[:8]} (timestamp: {last_commit.timestamp}, message: {last_commit.message[:50]})"
                    log_info(f"\n[DEBUG] {phase}: All commits in batch already indexed, advancing boundary {direction} to {commit_info}")
                continue  # Try next batch
            else:
                # No more commits available
                print()  # Newline after progress
                break
        
        # Embed the batch
        print(f"\r{total_examined} commits - embedding... {current_boundary_time if current_boundary_time else ''}", end='', flush=True)
        _process_changelist_batch(new_commits, expert_config, embedder, metadata_db, vector_db, max_embedding_tokens, debug)
        total_processed += len(new_commits)
        
        # Update boundaries based on Git's topological ordering (ignoring timestamps)
        if track_both_boundaries:
            # Initial indexing with git log --reverse goes from HEAD backwards
            # Batch 1 has newest commits, later batches have progressively older commits
            
            # Only set last_commit from FIRST batch (newest commits, closest to HEAD)
            if new_boundary_hash is None:
                newest_in_batch = new_commits[-1]
                new_boundary_time = newest_in_batch.timestamp
                new_boundary_hash = newest_in_batch.id
                last_commit_set = True  # Mark that we've set it
            
            # Update first_commit with EVERY batch (tracks oldest commit seen so far)
            oldest_in_batch = new_commits[0]
            second_boundary_time = oldest_in_batch.timestamp
            second_boundary_hash = oldest_in_batch.id
        elif is_newer:
            # Phase 1: track newest (first in list from git log, which returns newest-first)
            newest_commit = new_commits[0]
            new_boundary_time = newest_commit.timestamp
            new_boundary_hash = newest_commit.id
        else:
            # Phase 2: track oldest (first in list from git log --reverse)
            oldest_commit = new_commits[0]
            new_boundary_time = oldest_commit.timestamp
            new_boundary_hash = oldest_commit.id
        
        # Save boundaries incrementally for resumability
        if new_boundary_time and new_boundary_hash:
            if track_both_boundaries:
                # Initial indexing: Update last_commit only on first batch, first_commit on every batch
                if last_commit_set:
                    metadata_db.update_expert_last_commit(expert_config.name, new_boundary_time, new_boundary_hash)
                    if debug:
                        log_info(f"[DEBUG] Saved last_commit: {new_boundary_hash[:8]} at {new_boundary_time}")
                    last_commit_set = False  # Only write it once
                if second_boundary_time and second_boundary_hash:
                    metadata_db.update_expert_first_commit(expert_config.name, second_boundary_time, second_boundary_hash)
                    if debug:
                        log_info(f"[DEBUG] Saved first_commit: {second_boundary_hash[:8]} at {second_boundary_time}")
                elif debug:
                    log_info(f"[DEBUG] WARNING: second_boundary not set! second_time={second_boundary_time}, second_hash={second_boundary_hash}")
            elif is_newer:
                # Phase 1: Write new last_commit after every batch
                metadata_db.update_expert_last_commit(expert_config.name, new_boundary_time, new_boundary_hash)
                if debug:
                    log_info(f"[DEBUG] Phase 1 saved last_commit: {new_boundary_hash[:8]} at {new_boundary_time}")
            else:
                # Phase 2: Write first_commit after every batch
                metadata_db.update_expert_first_commit(expert_config.name, new_boundary_time, new_boundary_hash)
                if debug:
                    log_info(f"[DEBUG] Phase 2 saved first_commit: {new_boundary_hash[:8]} at {new_boundary_time}")
        
        # Update moving boundary for next iteration using raw page data (not filtered)
        # This ensures boundary follows Git's topological order exactly
        # Phase 1: git log returns newest-first, so use [0] for newest
        # Phase 2: git log --reverse returns oldest-first, so use [0] for oldest
        current_boundary_hash = page[0].id
        current_boundary_time = page[0].timestamp
        
        if debug:
            direction = "forward" if is_newer else "backward"
            phase = "Phase 1" if is_newer else "Phase 2"
            commit_info = f"{current_boundary_hash[:8]} (timestamp: {page[0].timestamp}, message: {page[0].message[:50]})"
            log_info(f"[DEBUG] {phase}: Moving boundary {direction} to {commit_info}")
        
        print(f"\r{total_examined} commits - batch done.  ", end='', flush=True)
        
        # Check if we've reached max_commits
        if total_processed >= max_commits:
            print()  # Newline after progress
            break
    
    return total_processed, new_boundary_time, new_boundary_hash, second_boundary_time, second_boundary_hash


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging for all Bedrock API calls')
@click.option('--data-dir', type=click.Path(path_type=Path), help='Base directory for expert data storage (default: ~/.expert-among-us)')
@click.option('--llm-provider', type=click.Choice(['auto', 'openai', 'openrouter', 'ollama', 'bedrock', 'claude-code']), default='auto', help='LLM provider for AI recommendations (auto-detects by default)')
@click.option('--base-url-override', type=str, help='Override base URL for OpenAI-compatible providers (openai, openrouter, ollama)')
@click.option('--embedding-provider', type=click.Choice(['local', 'bedrock']), default='local', help='Embedding provider: local=Jina Code, bedrock=AWS Titan (default: local)')
@click.option('--expert-model', type=str, help='Override default expert model for the selected provider')
@click.option('--promptgen-model', type=str, help='Override default promptgen model for the selected provider')
@click.version_option(version=__version__)
@click.pass_context
def main(ctx, debug: bool, data_dir: Optional[Path], llm_provider: str, base_url_override: Optional[str], embedding_provider: str, expert_model: Optional[str], promptgen_model: Optional[str]) -> None:
    """Expert Among Us - Queryable expert from commit history using LLM and embeddings.
    
    Create experts from git repositories, search commit history, and get
    AI-powered recommendations based on historical development patterns.
    
    Global Options:
        --llm-provider: Choose LLM provider (auto-detects by default)
            - auto: Auto-detect available provider (default)
            - openai: OpenAI API (requires OPENAI_API_KEY)
            - openrouter: OpenRouter API (requires OPENROUTER_API_KEY)
            - ollama: Ollama LLM server (default: http://127.0.0.1:11434/v1)
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
    ctx.obj['base_url_override'] = base_url_override
    ctx.obj['embedding_provider'] = embedding_provider
    ctx.obj['expert_model'] = expert_model
    ctx.obj['promptgen_model'] = promptgen_model
    
    # Initialize debug logger if enabled
    if debug:
        DebugLogger.configure(enabled=True)


@main.command()
@click.argument("workspace", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("expert_name", type=str)
@click.argument("subdirs", nargs=-1, type=str)
@click.option("--max-commits", default=10000, type=int, help="Maximum commits to index")
@click.option("--vcs-type", type=click.Choice(["git", "p4"]), default="git", help="Version control system type")
@click.option("--batch-size", default=50, type=int, help="Maximum commits per embedding batch")
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
        debug = ctx.obj.get('debug', False)
        
        # Step 1: Initialize components and settings
        settings_kwargs = {'embedding_provider': embedding_provider}
        if data_dir is not None:
            settings_kwargs['data_dir'] = data_dir
        
        settings = Settings(**settings_kwargs)
        
        log_info("Initializing components...")
        log_info(f"Using embedding provider: {embedding_provider}")
        if data_dir:
            log_info(f"Using data directory: {data_dir}")
        
        # Initialize embedder (torch.compile always enabled)
        embedder = create_embedder(embedding_provider, settings, compile_model=True)
        
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
        last_commit_hash = None
        first_commit_time = None
        first_commit_hash = None
        
        if existing_expert:
            log_info(f"Updating existing expert '{expert_name}'")
            last_commit_time = existing_expert.get('last_commit_time')
            last_commit_hash = existing_expert.get('last_commit_hash')
            first_commit_time = existing_expert.get('first_commit_time')
            first_commit_hash = existing_expert.get('first_commit_hash')
            if last_commit_time:
                hash_display = f" ({last_commit_hash[:8]})" if last_commit_hash else ""
                log_info(f"Last indexed commit: {last_commit_time}{hash_display}")
            if first_commit_time:
                hash_display = f" ({first_commit_hash[:8]})" if first_commit_hash else ""
                log_info(f"First indexed commit: {first_commit_time}{hash_display}")
        else:
            log_info(f"Creating new expert '{expert_name}'")
            metadata_db.create_expert(expert_config.name, str(expert_config.workspace_path), expert_config.subdirs, expert_config.vcs_type)
        
        subdirs_list = list(subdirs) if subdirs else None
        total_processed = 0
        
        # Step 4: Phase 1 - Process newer commits (if last_commit_hash exists)
        if last_commit_hash is not None:
            log_info("Phase 1: Processing newer commits...")
            
            phase1_processed, new_last_time, new_last_hash, _, _ = _process_commits_incrementally(
                vcs_provider=vcs_provider,
                workspace_path=str(workspace),
                subdirs_list=subdirs_list,
                batch_size=batch_size,
                expert_config=expert_config,
                embedder=embedder,
                metadata_db=metadata_db,
                vector_db=vector_db,
                max_embedding_tokens=max_embedding_tokens,
                is_newer=True,
                boundary_hash=last_commit_hash,
                max_commits=max_commits,
                track_both_boundaries=False,
                debug=debug
            )
            
            total_processed += phase1_processed
            
            if phase1_processed > 0 and new_last_hash:
                # Save the boundary at the end of Phase 1
                metadata_db.update_expert_last_commit(expert_config.name, new_last_time, new_last_hash)
                log_info(f"Phase 1 complete - indexed up to {new_last_time.isoformat()}")
            else:
                log_info("No newer commits found")
        
        # Step 5: Phase 2 - Process older commits
        if first_commit_time is not None:
            log_info("Phase 2: Processing older commits...")
            
            phase2_processed, new_first_time, new_first_hash, _, _ = _process_commits_incrementally(
                vcs_provider=vcs_provider,
                workspace_path=str(workspace),
                subdirs_list=subdirs_list,
                batch_size=batch_size,
                expert_config=expert_config,
                embedder=embedder,
                metadata_db=metadata_db,
                vector_db=vector_db,
                max_embedding_tokens=max_embedding_tokens,
                is_newer=False,
                boundary_hash=first_commit_hash,
                max_commits=max_commits,
                track_both_boundaries=False,
                debug=debug
            )
            
            total_processed += phase2_processed
            
            if phase2_processed > 0 and new_first_hash:
                # Save the boundary at the end of Phase 2
                metadata_db.update_expert_first_commit(expert_config.name, new_first_time, new_first_hash)
                log_info(f"Phase 2 complete - indexed back to {new_first_time.isoformat()}")
            else:
                log_info("No older commits found")
        else:
            # Initial indexing - fetch and process recent commits going backwards from HEAD
            log_info("Phase 2: Initial indexing - fetching recent commits...")
            
            phase2_processed, min_time, min_hash, max_time, max_hash = _process_commits_incrementally(
                vcs_provider=vcs_provider,
                workspace_path=str(workspace),
                subdirs_list=subdirs_list,
                batch_size=batch_size,
                expert_config=expert_config,
                embedder=embedder,
                metadata_db=metadata_db,
                vector_db=vector_db,
                max_embedding_tokens=max_embedding_tokens,
                is_newer=False,  # Use Phase 2 logic to go backwards from HEAD
                boundary_hash=None,
                max_commits=max_commits,
                track_both_boundaries=True,
                debug=debug
            )
            
            total_processed += phase2_processed
            
            if phase2_processed > 0 and max_hash and min_hash:
                # Save both boundaries at the end of initial indexing
                metadata_db.update_expert_last_commit(expert_config.name, max_time, max_hash)
                metadata_db.update_expert_first_commit(expert_config.name, min_time, min_hash)
                log_info(f"Initial indexing complete - processed {phase2_processed} commits")
                log_info(f"Time range: {min_time.isoformat()} to {max_time.isoformat()}")
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
        raise


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
            
            # Get debug flag from context
            debug = ctx.obj.get('debug', False)
            
            # Create rich table
            table = Table(title=f"Search Results for '{expert_name}'")
            table.add_column("ID", style="cyan", no_wrap=True, width=12)
            table.add_column("Author", style="green")
            table.add_column("Score", style="yellow", justify="right")
            table.add_column("Source", style="magenta")
            
            # Add ChromaDB ID column when debug is enabled
            if debug:
                table.add_column("ChromaDB ID", style="dim cyan", overflow="fold", max_width=40)
            
            table.add_column("Timestamp", style="blue", width=19)
            table.add_column("Message", style="white")
            
            for result in results:
                row_data = [
                    result.changelist.id[:12],
                    result.changelist.author,
                    f"{result.similarity_score:.3f}",
                    result.source,
                ]
                
                # Add ChromaDB ID if debug is enabled
                if debug:
                    chroma_id_display = result.chroma_id if result.chroma_id else "N/A"
                    row_data.append(chroma_id_display)
                
                row_data.extend([
                    result.changelist.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    result.changelist.message[:60] + "..." if len(result.changelist.message) > 60 else result.changelist.message
                ])
                
                table.add_row(*row_data)
            
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
        # Use Ollama LLM with Bedrock embeddings (global flags)
        $ expert-among-us --llm-provider ollama --embedding-provider bedrock prompt MyExpert "Implement X"
        
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
            'embedding_provider': embedding_provider,
            'base_url_override': ctx.obj.get('base_url_override'),
            'expert_model_override': ctx.obj.get('expert_model'),
            'promptgen_model_override': ctx.obj.get('promptgen_model'),
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
            max_diff_chars=settings.max_diff_chars_for_llm
        )
        
        # Extract changelists from search results
        changelists = [result.changelist for result in search_results]
        
        # Generate prompts using the PromptGenerator method (includes progress bar and cache logging)
        prompt_results = prompt_gen.generate_prompts(changelists)
        
        # Step 8: Build conversation
        log_info("Building conversation context...")
        conv_builder = ConversationBuilder(
            prompt_generator=prompt_gen,
            max_diff_chars=settings.max_diff_chars_for_llm
        )
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