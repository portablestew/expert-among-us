"""MCP Server for Expert Among Us.

This module implements a Model Context Protocol (MCP) server that exposes
Expert Among Us functionality to MCP clients like Claude Desktop.

The server provides four tools with dynamically generated descriptions:
- list: List all available experts with metadata
- import: Import external experts via symlink
- query: Search commit history (returns raw data)
- prompt: Get AI-powered recommendations (returns synthesized insights)

Tool descriptions automatically include the current list of available experts
and usage guidance, making them immediately visible to users without needing
to call additional tools.

Run with: expert-among-us mcp
"""

import asyncio
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, List

# Global variable to store LLM provider choice
_llm_provider = "auto"

# Global variable to store embedding provider choice
_embedding_provider = "local"

# Global variable to store data directory
_data_dir: Optional[Path] = None

# Global variables to store impostor and amogus modes
_impostor_mode = False
_amogus_mode = False

# Global variable to store max response tokens
_max_response_tokens = 4096

# Global variable to store prompt timeout in seconds
_prompt_timeout_seconds: Optional[int] = None

# Global variable to store debug flag
_debug = False

from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

# Import API functions
from expert_among_us.api import (
    list_experts,
    import_expert,
    query_expert,
    prompt_expert_stream,
    ExpertNotFoundError,
    ExpertAlreadyExistsError,
    InvalidExpertError,
    NoResultsError,
)
from expert_among_us.models.query_result import CommitResult, FileChunkResult
from expert_among_us import __version__


# Initialize MCP server
server = Server("expert-among-us")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools with dynamic expert information."""
    
    # Generate dynamic expert list for tool descriptions
    try:
        experts = list_experts(data_dir=_data_dir)
        
        if not experts:
            expert_list = "\n\n**Currently Available Experts:** None yet. Create one using the CLI 'populate' command."
        else:
            lines = ["\n\n**Currently Available Experts:**"]
            for expert in experts:
                lines.append(f"- Name: **{expert.name}** -- {expert.total_commit_count} commits")
                # Show commit hash range spanning all projects
                first_hash = None
                last_hash = None
                for proj in expert.projects:
                    if proj.first_processed_commit_hash and first_hash is None:
                        first_hash = proj.first_processed_commit_hash
                    if proj.last_processed_commit_hash:
                        last_hash = proj.last_processed_commit_hash
                if first_hash and last_hash:
                    lines.append(f"  spanning {first_hash[:8]} to {last_hash[:8]}")
                # Show per-project details
                for proj in expert.projects:
                    lines.append(f"  \u2022 {proj.name} ({proj.vcs_type}) - {proj.project_root} -- {proj.commit_count} commits")
            expert_list = "\n".join(lines)
    except Exception as e:
        expert_list = f"\n\n**Currently Available Experts:** Error loading: {str(e)}"
    
    return [
        Tool(
            name="experts-list",
            description=(
                "List all indexed experts with their metadata (commit counts, time ranges, workspace paths). "
                "Use this to discover available experts or check when they were last updated.\n\n"
                "**What are experts?** Each expert indexes a repository's complete version history, "
                "capturing not just WHAT the code does, but WHY decisions were made and HOW solutions "
                "evolved over time. Historical context includes bug fix patterns, performance optimizations, "
                "test cases discovered through fixes, and developer decision-making. Experts provide insights "
                "not available to naive code analysis.\n\n"
                "NOTE: There is normally no reason to call the 'list' tool; the available experts are listed below--"
                f"{expert_list}"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="experts-import",
            description=(
                "Import an expert from an external directory by creating a symlink. "
                "Useful for accessing team-shared experts or experts stored on external/network drives. "
                "The source directory must contain a valid expert (metadata.db file)."
                f"{expert_list}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Absolute path to the expert directory to import (must contain metadata.db)"
                    }
                },
                "required": ["source_path"]
            }
        ),
        Tool(
            name="expert-query",
            description=(
                "The 'expert-query' tool searches an expert's commit history using semantic similarity. "
                "It returns COMPLETE raw commit details (messages, diffs, files, authors) for your own analysis.\n\n"
                "TRADEOFF: This returns full commit content which can consume significant context window tokens. "
                "Prefer using 'expert-prompt' instead for AI-synthesized insights at lower cost. "
                "Use this tool to bypass AI summarization for: detailed code review, custom analysis, or manual pattern extraction."
                f"{expert_list}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "expert_name": {
                        "type": "string",
                        "description": "Name of the expert to query"
                    },
                    "prompt": {
                        "type": "string",
                        "description":
                            "Natural language question or task description. "
                            "Prompt semantics should be in the form of a task or commit message."
                            "Generally, it is best to pass the user's core prompt verbatim, stripping references to this tool and future instructions."
                    },
                    "max_changes": {
                        "type": "integer",
                        "description": "Optional: Maximum context changes to use (default=30)",
                        "default": 30
                    },
                    "users": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by commit authors"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by file paths"
                    },
                    "search_scope": {
                        "type": "string",
                        "enum": ["metadata", "diffs", "files", "all"],
                        "description": "Optional search scope: 'all' (default), 'metadata' only, 'diffs' only, or 'files' only",
                        "default": "all"
                    }
                },
                "required": ["expert_name", "prompt"]
            }
        ),
        Tool(
            name="expert-prompt",
            description=(
                "Get AI-powered recommendations that impersonate an expert based on their historical commit patterns. "
                "**Always use 'expert-prompt'** when asked questions about the content of these repositories:\n"
                f"{expert_list}"
                "\n\n"
                "Call 'expert-prompt' to answer natural language questions about the code base: locate matching files and find new insights. "
                "ALWAYS consult the expert FIRST by using this tool. Results are based on historical context and contain deep high-level insights."
                "Validate results: historical context may not be up-to-date. Use the information returned to guide narrower file searches.\n\n"
                "**Prefer defaults:** Parameters like max_changes, users, files are already tuned for optimal results. "
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "expert_name": {
                        "type": "string",
                        "description": "Name of the expert to query"
                    },
                    "prompt": {
                        "type": "string",
                        "description":
                            "Natural language question or task description. "
                            "Prompt semantics should be in the form of a task or commit message."
                            "Generally, it is best to pass the user's core prompt verbatim, stripping references to this tool and future instructions."
                    },
                    "max_changes": {
                        "type": "integer",
                        "description": "Optional: Maximum context changes to use (default=30)",
                        "default": 30
                    },
                    "users": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by commit authors"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by file paths"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Optional: LLM temperature for generation (0.0-1.0, default: 0.7)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["expert_name", "prompt"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "experts-list":
            return await handle_list()
        elif name == "experts-import":
            return await handle_import(**arguments)
        elif name == "expert-query":
            return await handle_query(**arguments)
        elif name == "expert-prompt":
            return await handle_prompt(**arguments)
        else:
            return [TextContent(
                type="text",
                text=f"✗ Unknown tool: {name}"
            )]
    except Exception as e:
        # Catch any unexpected errors
        return [TextContent(
            type="text",
            text=f"✗ Unexpected error: {str(e)}"
        )]


async def handle_list() -> list[TextContent]:
    """Handle list tool - list all experts."""
    try:
        experts = list_experts(data_dir=_data_dir)
        
        if not experts:
            return [TextContent(
                type="text",
                text="No experts found. Create one with the 'populate' command."
            )]
        
        # Format as markdown with per-project details
        lines = ["# Available Experts\n"]
        for expert in experts:
            lines.append(f"## {expert.name}")
            lines.append(f"- **Total Commits**: {expert.total_commit_count}")
            if expert.description:
                lines.append(f"- **Description**: {expert.description}")
            if expert.last_indexed_at:
                lines.append(f"- **Last Indexed**: {expert.last_indexed_at.isoformat()}")
            
            # Show commit hash range spanning all projects
            first_hash = None
            last_hash = None
            for proj in expert.projects:
                if proj.first_processed_commit_hash and first_hash is None:
                    first_hash = proj.first_processed_commit_hash
                if proj.last_processed_commit_hash:
                    last_hash = proj.last_processed_commit_hash
            if first_hash and last_hash:
                lines.append(
                    f"- **Commit Range**: {first_hash[:8]} to "
                    f"{last_hash[:8]}"
                )
            
            # Show per-project details
            if expert.projects:
                lines.append("")
                lines.append("**Projects:**")
                for proj in expert.projects:
                    proj_line = f"  \u2022 **{proj.name}** ({proj.vcs_type}) - {proj.project_root} -- {proj.commit_count} commits"
                    lines.append(proj_line)
                    if proj.first_processed_commit_hash and proj.last_processed_commit_hash:
                        lines.append(f"    spanning {proj.first_processed_commit_hash[:8]} to {proj.last_processed_commit_hash[:8]}")
            
            lines.append("")
        
        return [TextContent(
            type="text",
            text="\n".join(lines)
        )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"✗ Error listing experts: {str(e)}"
        )]


async def handle_import(source_path: str) -> list[TextContent]:
    """Handle import tool - import expert via symlink."""
    try:
        expert_name = import_expert(
            source_path=Path(source_path),
            data_dir=_data_dir
        )
        
        return [TextContent(
            type="text",
            text=f"✓ Successfully imported expert '{expert_name}' from {source_path}"
        )]
    
    except ExpertAlreadyExistsError as e:
        return [TextContent(
            type="text",
            text=f"✗ Error: {str(e)}"
        )]
    
    except InvalidExpertError as e:
        return [TextContent(
            type="text",
            text=f"✗ Invalid expert directory: {str(e)}"
        )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"✗ Error importing expert: {str(e)}"
        )]


async def handle_query(
    expert_name: str,
    prompt: str,
    max_changes: int = 30,
    users: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    search_scope: str = "all"
) -> list[TextContent]:
    """Handle query tool - search commit history."""
    try:
        results = query_expert(
            expert_name=expert_name,
            prompt=prompt,
            max_changes=math.floor(max_changes * 2 / 3),
            max_file_chunks=math.ceil(max_changes / 3),
            users=users,
            files=files,
            search_scope=search_scope,
            data_dir=_data_dir,
            embedding_provider=_embedding_provider,
            llm_provider=_llm_provider,
            enable_multiprocessing=False,
        )
        
        if not results:
            return [TextContent(
                type="text",
                text=f"No matching commits found for query: {prompt}"
            )]
        
        # Format results as detailed markdown
        lines = [f"# Query Results: {prompt}\n"]
        lines.append(f"Found {len(results)} matching commits\n")
        
        for i, result in enumerate(results, 1):
            if isinstance(result, CommitResult):
                cl = result.changelist
                lines.append(f"## {i}. Commit {cl.id[:12]} (Score: {result.similarity_score:.3f})")
                lines.append(f"**Author**: {cl.author}")
                lines.append(f"**Date**: {cl.timestamp.isoformat()}")
                lines.append(f"**Files**: {', '.join(cl.files)}")
                lines.append(f"\n**Message**:\n```\n{cl.message}\n```")
                
                if cl.diff:
                    # Truncate very long diffs
                    diff_preview = cl.diff[:10000]
                    if len(cl.diff) > 10000:
                        diff_preview += f"\n... (truncated, {len(cl.diff)} total chars)"
                    lines.append(f"\n**Diff**:\n```diff\n{diff_preview}\n```")
            
            elif isinstance(result, FileChunkResult):
                fc = result.file_chunk
                lines.append(f"## {i}. File {fc.file_path} (Score: {result.similarity_score:.3f})")
                lines.append(f"**Lines**: {fc.line_start}-{fc.line_end}")
                lines.append(f"**Revision**: {fc.revision_id[:12]}")
                
                content_preview = fc.content[:10000]
                if len(fc.content) > 10000:
                    content_preview += f"\n... (truncated, {len(fc.content)} total chars)"
                
                # Detect language for syntax highlighting
                import os
                _, ext = os.path.splitext(fc.file_path)
                lang = ext.lstrip('.') if ext else ''
                
                lines.append(f"\n**Content**:\n```{lang}\n{content_preview}\n```")
            
            lines.append("\n---\n")
        
        return [TextContent(
            type="text",
            text="\n".join(lines)
        )]
    
    except ExpertNotFoundError:
        return [TextContent(
            type="text",
            text=f"✗ Expert '{expert_name}' not found. Use 'list' to see available experts."
        )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"✗ Error querying expert: {str(e)}"
        )]


async def handle_prompt(
    expert_name: str,
    prompt: str,
    max_changes: int = 30,
    users: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    temperature: float = 0.7
) -> list[TextContent]:
    """Handle prompt tool - get AI recommendations."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Calculate deadline at request start (if timeout is configured)
    deadline = time.time() + _prompt_timeout_seconds if _prompt_timeout_seconds else None
    if deadline:
        logger.info(f"[PROMPT] Timeout configured: {_prompt_timeout_seconds}s (deadline: {deadline:.2f})")
    
    try:
        logger.info(f"[PROMPT] Starting prompt request for expert '{expert_name}'")
        logger.debug(f"[PROMPT] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        logger.debug(f"[PROMPT] Params: max_changes={max_changes}, impostor={_impostor_mode}, temperature={temperature}")
        
        # Accumulate streaming response
        full_response = ""
        chunk_count = 0
        first_chunk_time = None
        timed_out = False
        
        stream_start = time.time()
        logger.debug(f"[PROMPT] Initiating stream at +{stream_start - start_time:.2f}s")
        
        async for chunk in prompt_expert_stream(
            expert_name=expert_name,
            prompt=prompt,
            max_changes=math.floor(max_changes * 2 / 3),
            max_file_chunks=math.ceil(max_changes / 3),
            users=users,
            files=files,
            amogus=_amogus_mode,
            impostor=_impostor_mode,
            temperature=temperature,
            max_expert_response_tokens=_max_response_tokens,
            data_dir=_data_dir,
            embedding_provider=_embedding_provider,
            llm_provider=_llm_provider,
            enable_multiprocessing=False,
        ):
            if chunk.delta:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttft = first_chunk_time - stream_start
                    logger.info(f"[PROMPT] First token received at +{first_chunk_time - start_time:.2f}s (TTFT: {ttft:.2f}s)")
                
                full_response += chunk.delta
                chunk_count += 1
                
                # Log progress every 10 chunks
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.debug(f"[PROMPT] Received {chunk_count} chunks, {len(full_response)} chars at +{elapsed:.2f}s")
            
            # Check deadline AFTER processing chunk (keeps final chunk that exceeded timeout)
            if deadline and time.time() >= deadline:
                elapsed = time.time() - start_time
                full_response += f"\n\n[Response truncated: timeout reached after {elapsed:.1f}s]"
                timed_out = True
                logger.warning(f"[PROMPT] Timeout reached at +{elapsed:.2f}s after {chunk_count} chunks")
                break
        
        total_time = time.time() - start_time
        status = "timed out" if timed_out else "completed"
        logger.info(f"[PROMPT] {status.capitalize()} in {total_time:.2f}s - received {chunk_count} chunks, {len(full_response)} chars")
        
        return [TextContent(
            type="text",
            text=full_response
        )]
    
    except ExpertNotFoundError:
        return [TextContent(
            type="text",
            text=f"✗ Expert '{expert_name}' not found. Use 'list' to see available experts."
        )]
    
    except NoResultsError:
        return [TextContent(
            type="text",
            text=(
                f"✗ No relevant commits found for: {prompt}\n\n"
                "Try:\n"
                "- Broader search terms\n"
                "- Different keywords\n"
                "- Removing filters"
            )
        )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"✗ Error getting AI recommendations: {str(e)}"
        )]


async def start_server(
    llm_provider: str = "auto",
    data_dir: Optional[Path] = None,
    embedding_provider: str = "local",
    impostor_mode: bool = False,
    amogus_mode: bool = False,
    max_response_tokens: int = 4096,
    prompt_timeout_seconds: Optional[int] = None,
    debug: bool = False,
):
    """Start the MCP server with the given configuration."""
    global _llm_provider, _data_dir, _embedding_provider
    global _impostor_mode, _amogus_mode
    global _max_response_tokens, _prompt_timeout_seconds, _debug
    _llm_provider = llm_provider
    _data_dir = data_dir
    _embedding_provider = embedding_provider
    _impostor_mode = impostor_mode
    _amogus_mode = amogus_mode
    _max_response_tokens = max_response_tokens
    _prompt_timeout_seconds = prompt_timeout_seconds
    _debug = debug

    # Get PID for log file name
    pid = os.getpid()
    
    # Configure logging to both stderr AND file when debug is enabled
    log_level = logging.DEBUG if _debug else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    
    # Add file handler for debug mode
    if _debug:
        log_dir = Path.home() / ".expert-among-us" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"mcp-{pid}.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    
    if _debug:
        logger.debug(f"Debug logging enabled - writing to mcp-{pid}.log")
        logger.debug(f"Process ID: {pid}")
    
    logger.info("Starting Expert Among Us MCP Server...")
    logger.info("Initializing MCP server...")
    
    # Initialize settings with selected providers and optional data_dir
    from expert_among_us.config.settings import Settings
    from expert_among_us.embeddings.factory import create_embedder
    from expert_among_us.reranking.factory import create_reranker
    
    settings_kwargs: dict[str, Any] = {
        "embedding_provider": _embedding_provider,
        "llm_provider": _llm_provider,
        "enable_multiprocessing": False,  # Disable multiprocessing in MCP context to prevent hanging
    }
    if _data_dir:
        settings_kwargs["data_dir"] = _data_dir
    settings = Settings(**settings_kwargs)
    
    # Warm up embeddings, mainly for the local provider
    logger.info("Warming up local embedding model (this may take ~60s on first run)...")
    warmup_start = time.time()
    try:
        # Force embedder and reranker model initializations
        create_embedder(settings)
        create_reranker(settings)

        warmup_time = time.time() - warmup_start
        logger.info(f"Local transformer models ready (took {warmup_time:.1f}s)")
    except Exception as e:
        logger.warning(f"Failed to warm up local embedder: {e}")
    
    # Pre-warm LLM provider to avoid first-call delays
    logger.info(f"Pre-warming {settings.llm_provider} LLM provider...")
    llm_warmup_start = time.time()
    try:
        from expert_among_us.llm.factory import create_llm_provider
        create_llm_provider(settings)
        llm_warmup_time = time.time() - llm_warmup_start
        logger.info(f"LLM provider ready (took {llm_warmup_time:.1f}s)")
    except Exception as e:
        logger.warning(f"Failed to pre-warm LLM provider: {e}")
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server ready and listening for requests")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="expert-among-us",
                server_version=__version__,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(start_server())

