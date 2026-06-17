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
from expert_among_us.config.paths import resolve_data_dir
from expert_among_us.models.expert import ExpertConfig
from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.db.vector.chroma import ChromaVectorDB
from expert_among_us.vcs.detector import detect_vcs
from expert_among_us.embeddings.factory import create_embedder
from expert_among_us.utils.progress import (
    log_error,
    log_info,
    log_success,
    log_warning
)
from expert_among_us.utils.truncate import truncate_diff_for_embedding
# IMPORTANT: Use stderr=True to avoid corrupting MCP stdio protocol when running as MCP server
console = Console(stderr=True)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging for all Bedrock API calls')
@click.option('--data-dir', type=click.Path(path_type=Path), help='Base directory for expert data storage (default: ./.expert-among-us if present, else ~/.expert-among-us)')
@click.option('--llm-provider', type=click.Choice(['auto', 'openai', 'openrouter', 'ollama', 'bedrock', 'claude-code', 'kiro-cli']), default='auto', help='LLM provider for AI recommendations (auto-detects by default)')
@click.option('--base-url-override', type=str, help='Override base URL for OpenAI-compatible providers (openai, openrouter, ollama)')
@click.option('--embedding-provider', type=click.Choice(['local', 'bedrock']), default='local', help='Embedding provider: local=Jina Code, bedrock=AWS Titan (default: local)')
@click.option('--expert-model', type=str, help='Override default expert model for the selected provider')
@click.option('--promptgen-model', type=str, help='Override default promptgen model for the selected provider')
@click.option('--no-reranking', is_flag=True, help='Disable cross-encoder reranking (enabled by default)')
@click.option('--gpu-memory-multiplier', type=float, default=1.0, help='GPU memory scaling factor (0.5=half, 2.0=double). Affects embedding and reranking batch sizes.')
@click.version_option(version=__version__)
@click.pass_context
def main(ctx, debug: bool, data_dir: Optional[Path], llm_provider: str, base_url_override: Optional[str], embedding_provider: str, expert_model: Optional[str], promptgen_model: Optional[str], no_reranking: bool, gpu_memory_multiplier: float) -> None:
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
    ctx.obj['enable_reranking'] = not no_reranking
    ctx.obj['gpu_memory_multiplier'] = gpu_memory_multiplier
    
    # Initialize debug logger if enabled
    if debug:
        DebugLogger.configure(enabled=True)


@main.command()
@click.argument("expert_name", type=str)
@click.argument("project_root_arg", required=False, type=str)
@click.option("--project", type=str, required=True, help="Project name within the expert (required)")
@click.option("--max-commits", default=60000, type=int, help="Maximum commits to index")
@click.option("--max-batches", type=int, help="Maximum batches to run (returns exit code 2 if more remain)")
@click.option("--batch-size", default=1000, type=int, help="Maximum commits per embedding batch")
@click.option("--start-at", type=str, help="Start indexing from this specific commit hash (use with --max-commits to test specific commits)")
@click.option("--index-scope",
              type=click.Choice(["metadata", "diffs", "files", "all"], case_sensitive=False),
              default="all",
              help="Index scope: metadata (messages+files only), diffs (metadata+diffs, no file content), or all (everything, default)")
@click.option("--allowed-extensions", type=str, help="Comma-separated list of allowed file extensions (e.g., '.cpp,.h,.py')")
@click.option("--compact-diffs", is_flag=True, help="Reduce diff size by removing context (trades search quality for cost)")
@click.option("--custom-sanitize-pattern", type=str, help="Custom regex pattern to remove from text before embedding")
@click.pass_context
def populate(
    ctx,
    expert_name: str,
    project_root_arg: Optional[str],
    project: str,
    max_commits: int,
    max_batches: Optional[int],
    batch_size: int,
    start_at: Optional[str],
    index_scope: str,
    allowed_extensions: Optional[str],
    compact_diffs: bool,
    custom_sanitize_pattern: Optional[str],
) -> None:
    """Build or update an expert index from a repository.
    
    The VCS type (Git, Perforce, etc.) is automatically detected from the project root.
    
    EXPERT_NAME: Unique name for this expert
    
    [PROJECT_ROOT]: Optional path to the project root (the directory to index).
    Omit it to update an existing project (the root is looked up from the project).
    
    Examples:
    
        \b
        # Create new expert with a project (VCS auto-detected)
        $ expert-among-us populate MyExpert /path/to/repo --project my-project
        
        \b
        # Update existing project - project root optional (looks up from project)
        $ expert-among-us populate MyExpert --project my-project
        
        \b
        # Multi-project expert: add second project
        $ expert-among-us populate MyExpert /path/to/other-repo --project other-service
        
        \b
        # Use Bedrock embeddings (global flag)
        $ expert-among-us --embedding-provider bedrock populate MyExpert /path/to/repo --project my-project
        
        \b
        # Test specific commit (useful for debugging sanitization)
        $ expert-among-us populate MyExpert /path/to/repo --project my-project --start-at abc123def --max-commits 1
    """
    import re
    
    # Get global options from context
    data_dir = ctx.obj.get('data_dir')
    embedding_provider = ctx.obj.get('embedding_provider')
    
    # Validate project name using the same regex as ProjectConfig
    project_name_pattern = re.compile(r"^[a-zA-Z0-9_-][a-zA-Z0-9_-]*$")
    if not project_name_pattern.match(project):
        log_error(f"Invalid project name: '{project}'")
        log_error("Project name must contain only alphanumeric characters, hyphens, and underscores")
        sys.exit(1)
    
    # Parse the optional project root argument
    project_root: Optional[Path] = None
    
    if project_root_arg:
        potential_root = Path(project_root_arg)
        if not potential_root.exists() or not potential_root.is_dir():
            log_error(f"Invalid project root path: {potential_root}")
            sys.exit(1)
        project_root = potential_root
    
    # Step 0: Handle optional project root - look up from existing project if not provided
    if project_root is None:
        log_info(f"Looking up project root for existing project '{project}' in expert '{expert_name}'...")
        metadata_db_temp = SQLiteMetadataDB(expert_name, data_dir=data_dir)
        if metadata_db_temp.exists():
            metadata_db_temp.initialize()
            existing_project = metadata_db_temp.get_project(expert_name, project)
        else:
            existing_project = None
        metadata_db_temp.close()
        
        if existing_project:
            project_root = Path(existing_project['project_root'])
            log_info(f"Found project root: {project_root}")
        else:
            log_error(f"Project '{project}' does not exist in expert '{expert_name}'. Project root is required for new projects.")
            log_error(f"Usage: expert-among-us populate {expert_name} <project_root> --project {project}")
            sys.exit(1)
    else:
        # Project root provided - allow create-if-missing semantics.
        # We only enforce a mismatch check if an existing project record is found.
        metadata_db_temp = SQLiteMetadataDB(expert_name, data_dir=data_dir)
        if metadata_db_temp.exists():
            metadata_db_temp.initialize()
            existing_project = metadata_db_temp.get_project(expert_name, project)
        else:
            existing_project = None
        metadata_db_temp.close()
        
        if existing_project:
            existing_root = Path(existing_project['project_root'])
            if existing_root != project_root:
                log_warning("Project root mismatch detected!")
                log_warning(f"  Stored project root: {existing_root}")
                log_warning(f"  Provided project root: {project_root}")
                log_error("Cannot change project root for existing project. Use the original path or create a new project.")
                sys.exit(1)
    
    log_info(f"Populating expert '{expert_name}', project '{project}' from project root: {project_root}")
    
    try:
        # Get global options from context
        data_dir = ctx.obj.get('data_dir')
        embedding_provider = ctx.obj.get('embedding_provider')
        debug = ctx.obj.get('debug', False)
        
        # Step 1: Initialize components and settings
        settings_kwargs = {'embedding_provider': embedding_provider}
        if data_dir is not None:
            settings_kwargs['data_dir'] = data_dir
        
        # Configure GPU memory multiplier
        gpu_memory_multiplier = ctx.obj.get('gpu_memory_multiplier', 1.0)
        settings_kwargs['gpu_memory_multiplier'] = gpu_memory_multiplier
        
        # Configure indexing scope based on CLI option
        if index_scope == "metadata":
            settings_kwargs['embed_file_chunks'] = False
            settings_kwargs['embed_diffs'] = False
            log_info("Index scope: metadata only -- commit messages + file lists")
        elif index_scope == "diffs":
            settings_kwargs['embed_file_chunks'] = False
            settings_kwargs['embed_diffs'] = True
            log_info("Index scope: diffs -- metadata + diffs, no file content")
        elif index_scope == "files":
            settings_kwargs['embed_file_chunks'] = True
            settings_kwargs['embed_diffs'] = False
            log_info("Index scope: files -- metadata + files, no diff content")
        else:  # "all"
            settings_kwargs['embed_file_chunks'] = True
            settings_kwargs['embed_diffs'] = True
            log_info("Index scope: all -- files + diffs + metadata")
        
        # Configure file extension filtering
        if allowed_extensions:
            extensions_list = [ext.strip() for ext in allowed_extensions.split(',')]
            settings_kwargs['allowed_file_extensions'] = extensions_list
            log_info(f"Filtering to extensions: {', '.join(extensions_list)}")
        
        # Configure compact diff setting
        if compact_diffs:
            settings_kwargs['compact_diffs'] = True
        
        # Configure custom sanitization pattern
        if custom_sanitize_pattern:
            settings_kwargs['custom_sanitization_patterns'] = [custom_sanitize_pattern]
            log_info(f"Custom sanitization pattern: {custom_sanitize_pattern[:50]}...")
        
        settings = Settings(**settings_kwargs)
        
        # Log compact diff setting after settings object is created
        if compact_diffs:
            log_info(f"Compact diffs enabled: no context, max {settings.compact_diff_max_line_bytes} bytes per change line")
        
        log_info("Initializing components...")
        log_info(f"Using embedding provider: {embedding_provider}")
        if data_dir:
            log_info(f"Using data directory: {data_dir}")
        
        # Step 2: Detect and initialize VCS provider (moved earlier)
        log_info(f"Detecting version control system in {project_root}...")
 
        # Detect VCS directly; Git will consult DebugLogger.is_enabled() for its own debug output.
        vcs_provider = detect_vcs(str(project_root), settings)
        
        if vcs_provider is None:
            log_error(f"No supported VCS detected in {project_root}")
            log_error("Supported VCS: Git, Perforce")
            sys.exit(1)
        
        log_success(f"Detected VCS: {vcs_provider.__class__.__name__}")
        
        # Auto-detect VCS type from provider class name
        # Map class names to expected vcs_type values
        vcs_class_name = vcs_provider.__class__.__name__
        vcs_type_map = {
            "Git": "git",
            "Perforce": "p4"
        }
        vcs_type = vcs_type_map.get(vcs_class_name, vcs_class_name.lower())
        
        # Initialize embedder using settings (provider/model come from Settings)
        embedder = create_embedder(settings)
        
        # Create expert configuration (logical grouping only - no project/vcs fields)
        expert_config = ExpertConfig(
            name=expert_name,
            data_dir=resolve_data_dir(data_dir)
        )
        
        # Ensure storage directories exist
        expert_config.ensure_storage_exists()
        
        # Initialize databases
        metadata_db = SQLiteMetadataDB(expert_name, data_dir=data_dir)
        metadata_db.initialize()
        vector_db = ChromaVectorDB(expert_name, data_dir=data_dir)
        vector_db.initialize(dimension=embedder.dimension)
        
        # Step 3: Get or create expert, then create project if needed
        existing_expert = metadata_db.get_expert(expert_name)
        
        if existing_expert:
            log_info(f"Using existing expert '{expert_name}'")
        else:
            log_info(f"Creating new expert '{expert_name}'")
            metadata_db.create_expert(expert_name)
        
        # Create project if it doesn't already exist
        existing_project = metadata_db.get_project(expert_name, project)
        if existing_project:
            log_info(f"Using existing project '{project}'")
            last_processed = existing_project.get('last_processed_commit_hash')
            if last_processed:
                log_info(f"Last processed commit: {last_processed[:8]}")
        else:
            log_info(f"Creating new project '{project}' in expert '{expert_name}'")
            metadata_db.create_project(expert_name, project, str(project_root), vcs_type)
        
        # Build project_config dict for the Indexer
        project_config = {
            'name': project,
            'expert_name': expert_name,
            'project_root': str(project_root),
            'vcs_type': vcs_type,
            'has_vector_metadata': True,
        }
        
        total_processed = 0
        
        # Step 4: Use unified indexing (processes both files and commits together)
        log_info("Starting unified indexing...")
        try:
            # Initialize unified indexer with explicit dependency injection
            from expert_among_us.core.indexer import Indexer

            indexer = Indexer(
                expert_config={**expert_config.model_dump(), 'project_root': str(project_root)},
                vcs=vcs_provider,
                metadata_db=metadata_db,
                vector_db=vector_db,
                embedder=embedder,
                settings=settings,
                max_commits=max_commits,
                project_config=project_config,
            )

            # Run unified indexing
            more_remain = indexer.index_unified(batch_size=batch_size, start_after=start_at, max_batches=max_batches)
            log_success("Unified indexing complete!")
            
            # Get total processed commits for reporting (per-project)
            total_processed = metadata_db.get_project_commit_count(expert_name, project)
            
        except Exception as e:
            log_error(f"Unified indexing failed: {str(e)}")
            log_info("Indexing may have been interrupted - try again to resume")
            raise  # Always show full stack trace
        
        # Update expert index time
        metadata_db.update_expert_index_time(expert_config.name, datetime.now(timezone.utc))
        
        # Step 7: Report results
        if total_processed > 0:
            log_success(f"Indexing complete!")
            log_info(f"Successfully processed: {total_processed} commits")
            
            if not more_remain:
                # Calculate and report skipped commits
                total_available = vcs_provider.get_total_commit_count(
                    project_root=str(project_root),
                )
                skipped_commit_count = total_available - total_processed if isinstance(total_available, int) else 0
                if skipped_commit_count > 0:
                    log_info(f"  (skipped {skipped_commit_count} commits with no indexable text content, e.g. binary-only)")
        else:
            log_info("No commits were processed")
        
        # Cleanup
        metadata_db.close()
        vector_db.close()
        
        # Exit with code 2 if more commits remain (batch limit reached)
        if more_remain:
            log_info("More commits available. Run populate again to continue.")
            sys.exit(2)
        
    except Exception as e:
        log_error(f"Failed to populate expert: {str(e)}")
        raise


@main.command()
@click.argument("expert_name", type=str)
@click.argument("prompt", type=str)
@click.option("--max-changes", default=20, type=int, help="Maximum changelist results to return")
@click.option("--max-file-chunks", default=10, type=int, help="Maximum file chunk results to return")
@click.option("--users", type=str, help="Filter by authors (comma-separated)")
@click.option("--files", type=str, help="Filter by files (comma-separated)")
@click.option("--output", type=click.Path(path_type=Path), help="Output JSON file")
@click.option("--search-scope",
              type=click.Choice(["metadata", "diffs", "files", "all"], case_sensitive=False),
              default="all",
              help="Search scope: all (default), metadata only, diffs only, or files only")
@click.option("--min-score",
              type=float,
              default=0.1,
              help="Minimum similarity score threshold (0.0-1.0, default: 0.1)")
@click.option("--relative-threshold",
              type=float,
              default=0.8,
              help="Relative score threshold as percentage drop from top result (0.0-1.0, default: 0.3)")
@click.option(
    '--expansion-candidate-multiplier',
    type=int,
    default=5,
    help='Multiplier for candidate retrieval during expansion (default: 5)'
)
@click.option(
    '--expansion-passes',
    type=int,
    default=1,
    help='Number of expansion iterations/passes (default: 1)'
)
@click.pass_context
def query(
    ctx,
    expert_name: str,
    prompt: str,
    max_changes: int,
    max_file_chunks: int,
    users: Optional[str],
    files: Optional[str],
    output: Optional[Path],
    search_scope: str,
    min_score: float,
    relative_threshold: float,
    expansion_candidate_multiplier: int,
    expansion_passes: int,
) -> None:
    """Search for similar changes in the expert's history.
    
    EXPERT_NAME: Name of the expert to query
    
    PROMPT: Search query (e.g., "How to add a new arrow type?")
    
    Options:
        --search-scope: Control which embeddings to search
            - all: Search metadata, diffs, and file content (default)
            - metadata: Search only metadata embeddings
            - diffs: Search only diff embeddings
            - files: Search only file content embeddings
        
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
        # Basic query (searches all sources)
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
        # Search only file content
        $ expert-among-us query MyExpert "function implementation" --search-scope files
        
        \b
        # Search everything (metadata, diffs, and files)
        $ expert-among-us query MyExpert "authentication logic" --search-scope all
        
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
    from expert_among_us.api import query_expert
    from expert_among_us.models.query_result import CommitResult, FileChunkResult
    
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
        
        log_info("Initializing search components...")
        log_info(f"Using embedding provider: {embedding_provider}")
        if data_dir:
            log_info(f"Using data directory: {data_dir}")
        
        # Call API function
        enable_reranking = ctx.obj.get('enable_reranking', True)
        gpu_memory_multiplier = ctx.obj.get('gpu_memory_multiplier', 1.0)
        results = query_expert(
            expert_name=expert_name,
            prompt=prompt,
            max_changes=max_changes,
            max_file_chunks=max_file_chunks,
            users=user_list,
            files=file_list,
            search_scope=search_scope,
            min_score=min_score,
            relative_threshold=relative_threshold,
            expansion_candidate_multiplier=expansion_candidate_multiplier,
            expansion_passes=expansion_passes,
            data_dir=data_dir,
            embedding_provider=embedding_provider,
            llm_provider="auto",
            enable_multiprocessing=True,
            enable_reranking=enable_reranking,
            gpu_memory_multiplier=gpu_memory_multiplier,
        )
        
        # Display results
        if results:
            log_success(f"Found {len(results)} matching changes")
            
            # Get debug flag from context
            debug = ctx.obj.get('debug', False)
            
            # Create rich table
            table = Table(title=f"Search Results for '{expert_name}'")
            table.add_column("ID", style="cyan", no_wrap=True, width=16)
            table.add_column("Project", style="dim cyan")
            table.add_column("Source", style="magenta")
            table.add_column("Author", style="green")

            # Add ChromaDB ID column when debug is enabled
            if debug:
                table.add_column("ChromaDB ID", style="dim cyan", overflow="fold", max_width=40)

            table.add_column("Message", style="white")

            # Show dual scores in debug mode, single score otherwise
            if debug:
                table.add_column("Search", style="dim yellow", justify="right")   # NEW: Original vector search score
                table.add_column("Rerank", style="yellow", justify="right")       # NEW: Reranked score
            else:
                table.add_column("Score", style="yellow", justify="right")
            
            table.add_column("Timestamp", style="blue", width=19)
            
            for result in results:
                # Use common interface instead of assuming Changelist
                row_data = []

                if isinstance(result, FileChunkResult):
                    row_data.append(Path(result.get_file_path()).name[:16])
                else:
                    row_data.append(result.get_id()[:16])

                # Extract project name from result
                if isinstance(result, CommitResult):
                    row_data.append(result.changelist.project_name)
                elif isinstance(result, FileChunkResult):
                    # File paths are prefixed: "project_name/rest/of/path"
                    parts = result.get_file_path().split("/", 1)
                    row_data.append(parts[0] if len(parts) > 1 else "")
                else:
                    row_data.append("")

                row_data.append(result.source)

                if result.get_author():
                    row_data.append(result.get_author())
                elif isinstance(result, FileChunkResult):
                    row_data.append(f"{result.get_file_path()}:{result.get_line_range()[0]}-{result.get_line_range()[1]}")
                else:
                    row_data.append("N/A")

                # Add ChromaDB ID if debug is enabled
                if debug:
                    chroma_id_display = result.chroma_id if result.chroma_id else "N/A"
                    row_data.append(chroma_id_display)
                
                row_data.append(result.get_preview_text(max_len=100));

                # Score display: dual scores in debug, single score otherwise
                if debug:
                    # Show both search and reranked scores
                    search_score = result.search_similarity_score or result.similarity_score
                    row_data.append(f"{search_score:.3f}")
                    row_data.append(f"{result.similarity_score:.3f}")
                else:
                    # Show only final score (after reranking if applied)
                    row_data.append(f"{result.similarity_score:.3f}")
                
                ts = result.get_timestamp()
                if ts:
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(result, FileChunkResult):
                    ts_str = "HEAD (indexed)"
                else:
                    ts_str = "N/A"
                row_data.append(ts_str)

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
                        (
                            {
                                "id": r.changelist.id,
                                "type": "commit",
                                "author": r.changelist.author,
                                "timestamp": r.changelist.timestamp.isoformat(),
                                "message": r.changelist.message,
                                "files": r.changelist.files,
                                "diff": r.changelist.diff,
                                "similarity_score": r.similarity_score,
                                "source": r.source,
                            }
                            if isinstance(r, CommitResult)
                            else {
                                "id": r.get_id(),
                                "type": "file_chunk",
                                "file_path": r.get_file_path(),
                                "line_range": list(r.get_line_range()),
                                "content": r.get_content(),
                                "revision_id": r.get_revision_id(),
                                "similarity_score": r.similarity_score,
                                "source": r.source,
                            }
                        )
                        for r in results
                    ]
                }
                
                output.parent.mkdir(parents=True, exist_ok=True)
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                log_success(f"Results saved to {output}")
        else:
            log_info("No matching changes found")
        
    except Exception as e:
        log_error(f"Search failed: {str(e)}")
        raise  # Always reraise to show full stack trace


@main.command()
@click.argument("expert_name", type=str)
@click.argument("prompt", type=str)
@click.option("--max-changes", default=20, type=int, help="Maximum changelist results to use as context")
@click.option("--max-file-chunks", default=10, type=int, help="Maximum file chunk results to use as context")
@click.option("--users", type=str, help="Filter by authors (comma-separated)")
@click.option("--files", type=str, help="Filter by files (comma-separated)")
@click.option("--amogus", is_flag=True, help="ඞ")
@click.option("--impostor", is_flag=True, default=False,
              help="Generate prompts and use user-assistant pairs (old behavior)")
@click.option("--temperature", default=0.7, type=float, help="LLM temperature (0.0-1.0)")
@click.option(
    '--expansion-candidate-multiplier',
    type=int,
    default=5,
    help='Multiplier for candidate retrieval during expansion (default: 5)'
)
@click.option(
    '--expansion-passes',
    type=int,
    default=1,
    help='Number of expansion iterations/passes (default: 1)'
)
@click.pass_context
def prompt(
    ctx,
    expert_name: str,
    prompt: str,
    max_changes: int,
    max_file_chunks: int,
    users: Optional[str],
    files: Optional[str],
    amogus: bool,
    impostor: bool,
    temperature: float,
    expansion_candidate_multiplier: int,
    expansion_passes: int,
) -> None:
    """Get AI recommendations based on expert's historical patterns.
    
    EXPERT_NAME: Name of the expert to query
    
    PROMPT: Question or task description
    
    Note: File content is automatically searched and included when relevant,
    providing current codebase context alongside historical commit patterns.
    
    Examples:
    
        \b
        # Get recommendations (default mode - faster, cheaper)
        $ expert-among-us prompt MyExpert "How should I implement X?"
        
        \b
        # Use impostor mode (generate prompts, user-assistant pairs)
        $ expert-among-us prompt MyExpert "Implement X" --impostor
        
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
    from expert_among_us.api import prompt_expert_stream
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
    
    # Show debug logging info if enabled
    if debug:
        log_info(f"Debug logging enabled: {DebugLogger._log_dir}")
    
    # Parse filters
    user_list = [u.strip() for u in users.split(",")] if users else None
    file_list = [f.strip() for f in files.split(",")] if files else None
    
    if user_list:
        log_info(f"Filtering by users: {', '.join(user_list)}")
    if file_list:
        log_info(f"Filtering by files: {', '.join(file_list)}")

    # Auto-detect LLM provider early if needed (before any initialization)
    # The detect_llm_provider() function prints "Auto-detecting LLM provider: <provider>"
    if llm_provider == "auto":
        from expert_among_us.llm.auto_detect import detect_llm_provider
        llm_provider = detect_llm_provider()
    
    # Only print "Using LLM provider" if it wasn't auto-detected (which already printed it)
    if ctx.obj.get('llm_provider') != "auto":
        log_info(f"Using LLM provider: {llm_provider}")
    if data_dir:
        log_info(f"Using data directory: {data_dir}")
    log_info(f"Using embedding provider: {embedding_provider}")
    
    async def stream_and_display():
        """Stream response and display to console."""
        full_response = ""
        final_usage = None  # Usage only in final chunk
        first_chunk = True  # Track if this is the first chunk
        
        try:
            enable_reranking = ctx.obj.get('enable_reranking', True)
            gpu_memory_multiplier = ctx.obj.get('gpu_memory_multiplier', 1.0)
            async for chunk in prompt_expert_stream(
                expert_name=expert_name,
                prompt=prompt,
                max_changes=max_changes,
                max_file_chunks=max_file_chunks,
                users=user_list,
                files=file_list,
                amogus=amogus,
                impostor=impostor,
                temperature=temperature,
                data_dir=data_dir,
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                enable_multiprocessing=True,
                enable_reranking=enable_reranking,
                expansion_candidate_multiplier=expansion_candidate_multiplier,
                expansion_passes=expansion_passes,
                gpu_memory_multiplier=gpu_memory_multiplier,
            ):
                if chunk.delta:
                    # Print "Expert Response:" header on first chunk with content
                    if first_chunk:
                        console.print("[bold cyan]Expert Response:[/bold cyan]\n")
                        first_chunk = False
                    console.print(chunk.delta, end="")
                    full_response += chunk.delta
                
                if chunk.usage:
                    # Usage is only present in the final chunk
                    # Already accumulated by the LLM provider
                    final_usage = chunk.usage
            
            console.print("\n")
            
            # Show token usage and cache metrics only when debug is enabled
            if debug and final_usage:
                debug_msg = f"Tokens used: input={final_usage.input_tokens}, output={final_usage.output_tokens}, total={final_usage.total_tokens}"
                debug_msg += f" | Cache: read={final_usage.cache_read_tokens}, creation={final_usage.cache_creation_tokens}"
                console.print(f"[dim]{debug_msg}[/dim]")
                console.print(f"[dim]Debug logs written to: {DebugLogger._log_dir}[/dim]")
                
        except ValueError as e:
            log_error(str(e))
            sys.exit(1)
    
    try:
        asyncio.run(stream_and_display())
        
    except KeyboardInterrupt:
        log_info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        log_error(f"Failed to generate recommendations: {str(e)}")
        raise  # Always show full stack trace


@main.command("list")
@click.pass_context
def list_experts(ctx) -> None:
    """List all available experts and their information.
    
    Displays a table showing each expert's name, project root, VCS type,
    number of commits indexed, last indexed time, and commit time range.
    
    Examples:
    
        \b
        # List all experts
        $ expert-among-us list
        
        \b
        # List experts from custom data directory
        $ expert-among-us --data-dir /mnt/data/experts list
    """
    from expert_among_us.api import list_experts
    
    try:
        # Get data directory from context
        data_dir = ctx.obj.get('data_dir')
        
        # Use the API function to get expert info
        experts_info = list_experts(data_dir=data_dir)
        
        # Display message if no experts found
        if not experts_info:
            log_info("No experts found.")
            log_info("Run 'expert-among-us populate <expert_name> <project_root> --project <name>' to create your first expert.")
            return
        
        # Create Rich table
        table = Table(title="Available Experts")
        table.add_column("Name", style="cyan bold")
        table.add_column("Projects", style="green")
        table.add_column("Total Commits", style="yellow", justify="right")
        table.add_column("Last Indexed", style="blue")
        table.add_column("Commit Range", style="magenta")
        
        for expert in experts_info:
            # Format last indexed (UTC with ISO-8601 format)
            if expert.last_indexed_at:
                last_indexed = expert.last_indexed_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                last_indexed = "Never"
            
            # Format commit range spanning all projects
            first_hash = None
            last_hash = None
            for proj in expert.projects:
                if proj.first_processed_commit_hash and first_hash is None:
                    first_hash = proj.first_processed_commit_hash
                if proj.last_processed_commit_hash:
                    last_hash = proj.last_processed_commit_hash
            if first_hash and last_hash:
                commit_range = f"Range: {first_hash[:8]} → {last_hash[:8]}"
            elif last_hash:
                commit_range = f"Only: {last_hash[:8]}"
            else:
                commit_range = "No commits processed"
            
            # Format projects summary
            if expert.projects:
                proj_names = ", ".join(
                    f"{p.name} ({p.vcs_type})" for p in expert.projects
                )
            else:
                proj_names = "(none)"
            
            table.add_row(
                expert.name,
                proj_names,
                str(expert.total_commit_count),
                last_indexed,
                commit_range
            )
        
        console.print(table)
        
    except Exception as e:
        log_error(f"Failed to list experts: {str(e)}")
        raise

@main.command(name="import")
@click.argument("source_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.pass_context
def import_(ctx, source_path: Path) -> None:
    """Import an expert from an external directory via symlink.
    
    Creates a symlink in the data directory pointing to the source expert directory.
    The source directory must contain a valid expert (metadata.db).
    
    SOURCE_PATH: Path to the expert directory to import
    
    Examples:
    
        \b
        # Import an expert from external storage
        $ expert-among-us import /external/storage/MyExpert
        
        \b
        # Import from a shared network location
        $ expert-among-us import ~/shared/experts/TeamExpert
        
        \b
        # Import with custom data directory
        $ expert-among-us --data-dir /custom/location import /external/MyExpert
    """
    from expert_among_us.api import import_expert
    
    try:
        # Get data directory from context
        data_dir = ctx.obj.get('data_dir')
        
        # Get expert name for logging
        expert_name = source_path.name
        log_info(f"Importing expert '{expert_name}' from {source_path}...")
        
        # Call API function
        imported_name = import_expert(
            source_path=source_path,
            data_dir=data_dir
        )
        
        log_success(f"Successfully imported expert '{imported_name}'")
        
        # Calculate paths for informational logging
        data_dir = resolve_data_dir(data_dir)
        target_path = data_dir / "data" / imported_name
        log_info(f"Symlink created: {target_path} -> {source_path.resolve()}")
        
    except Exception as e:
        log_error(f"Failed to import expert: {str(e)}")
        raise



@main.command()
@click.pass_context
def prewarm(ctx) -> None:
    """Download and initialize embedding models, then exit.
    
    Use this to ensure all models are cached locally before first real use.
    Avoids a ~60s delay on the first MCP query or populate run.
    
    Examples:
    
        \b
        # Pre-download models
        $ expert-among-us prewarm
        
        \b
        # With Bedrock embeddings (no local model needed)
        $ expert-among-us --embedding-provider bedrock prewarm
    """
    import time
    from expert_among_us.config.settings import Settings
    from expert_among_us.embeddings.factory import create_embedder
    from expert_among_us.reranking.factory import create_reranker

    embedding_provider = ctx.obj.get('embedding_provider', 'local')
    gpu_memory_multiplier = ctx.obj.get('gpu_memory_multiplier', 1.0)

    settings = Settings(
        embedding_provider=embedding_provider,
        gpu_memory_multiplier=gpu_memory_multiplier,
    )

    log_info("Pre-warming embedding and reranking models...")
    start = time.time()

    try:
        embedder = create_embedder(settings)
        log_success(f"Embedder ready: {embedder.__class__.__name__} (dim={embedder.dimension})")
    except Exception as e:
        log_error(f"Embedder failed: {e}")

    try:
        reranker = create_reranker(settings)
        log_success(f"Reranker ready: {reranker.__class__.__name__}")
    except Exception as e:
        log_error(f"Reranker failed: {e}")

    elapsed = time.time() - start
    log_success(f"Prewarm complete in {elapsed:.1f}s")


@main.command()
@click.option('--impostor', is_flag=True, help='Enable impostor mode for all queries')
@click.option('--amogus', is_flag=True, hidden=True)
@click.option('--max-response-tokens', type=int, default=4096, help='Maximum tokens for expert response (default: 4096)')
@click.option('--prompt-timeout-seconds', type=int, default=None, help='Maximum seconds for expert-prompt operations (default: no timeout)')
@click.pass_context
def mcp(ctx, impostor: bool, amogus: bool, max_response_tokens: int, prompt_timeout_seconds: Optional[int]) -> None:
    """Start the MCP (Model Context Protocol) server.
    
    Exposes expert-among-us tools to MCP clients like Claude Desktop or Kiro.
    
    Examples:
    
        \b
        # Start MCP server
        $ expert-among-us mcp
        
        \b
        # With impostor mode and debug logging
        $ expert-among-us --debug mcp --impostor
        
        \b
        # With custom data directory
        $ expert-among-us --data-dir /shared/experts mcp
    """
    import asyncio
    from expert_among_us.__mcp__ import start_server
    
    asyncio.run(start_server(
        llm_provider=ctx.obj.get('llm_provider', 'auto'),
        data_dir=ctx.obj.get('data_dir'),
        embedding_provider=ctx.obj.get('embedding_provider', 'local'),
        impostor_mode=impostor,
        amogus_mode=amogus,
        max_response_tokens=max_response_tokens,
        prompt_timeout_seconds=prompt_timeout_seconds,
        debug=ctx.obj.get('debug', False),
    ))


if __name__ == "__main__":
    main()

