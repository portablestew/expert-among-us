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

Run with: python -m expert_among_us.__mcp__
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, List

# Global variable to store LLM provider choice
_llm_provider = "auto"

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


# Initialize MCP server
server = Server("expert-among-us")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools with dynamic expert information."""
    
    # Generate dynamic expert list for tool descriptions
    try:
        experts = list_experts(data_dir=None)
        
        if not experts:
            expert_list = "\n\n**Currently Available Experts:** None yet. Create one using the CLI 'populate' command."
        else:
            lines = ["\n\n**Currently Available Experts:**"]
            for expert in experts:
                lines.append(f"- Name: **{expert.name}** -- Path: {expert.workspace_path} -- {expert.commit_count} {expert.vcs_type} commits")
                if expert.first_commit_time and expert.last_commit_time:
                    lines.append(f" spanning {expert.first_commit_time.date()} to {expert.last_commit_time.date()}")
            expert_list = "\n".join(lines)
    except Exception as e:
        expert_list = f"\n\n**Currently Available Experts:** Error loading: {str(e)}"
    
    return [
        Tool(
            name="list",
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
            name="import",
            description=(
                "Import an expert from an external directory by creating a symlink. "
                "Useful for accessing team-shared experts or experts stored on external/network drives. "
                "The source directory must contain a valid expert (metadata.db file).\n\n"
                "**Team collaboration:** Import experts created by colleagues to leverage their project knowledge without re-indexing the same repository."
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
            name="query",
            description=(
                "The 'query' command searches an expert's commit history using semantic similarity. "
                "It returns COMPLETE raw commit details (messages, diffs, files, authors) for your own analysis.\n\n"
                "TRADEOFF: This returns full commit content which can consume significant context window tokens. "
                "Prefer using 'prompt' instead for AI-synthesized insights at lower cost.\n\n"
                "**When to use this over 'prompt':**\n"
                "- You need complete commit details for custom analysis\n"
                "- Detailed code review or pattern extraction\n"
                "- The LLM's synthesis isn't sufficient for your needs\n\n"
                "Best for: Custom analysis, detailed code review, pattern extraction, when you need complete commit details rather than AI summaries."
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
                        "description": "Natural language description of what to search for. Semantic search works best with descriptive phrases rather than bare keywords (e.g., 'how authentication was implemented', 'bug fixes for memory leaks', 'performance optimization in caching layer')"
                    },
                    "max_changes": {
                        "type": "integer",
                        "description": "Maximum number of commits to return (default: 15)",
                        "default": 15
                    },
                    "users": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by commit authors (e.g., ['alice', 'bob'])"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by file paths (e.g., ['src/auth/', 'tests/'])"
                    },
                    "search_scope": {
                        "type": "string",
                        "enum": ["both", "metadata", "diffs"],
                        "description": "Search scope: 'both' (default), 'metadata' only, or 'diffs' only",
                        "default": "both"
                    }
                },
                "required": ["expert_name", "prompt"]
            }
        ),
        Tool(
            name="prompt",
            description=(
                "Get AI-powered recommendations that impersonate an expert based on their historical commit patterns.\n"
                "**Always use expert-among-us 'prompt'** when asked questions about code potentially matching these repositories:\n"
                f"{expert_list}"
                "\n\n"
                "Call 'prompt' to answer natural language questions and discover insights about the code base. "
                "ALWAYS consult the expert FIRST using the 'prompt' tool to discover relevant context BEFORE reading code. "
                "Topics include: established implementation patterns; evolutionary history; abandoned approaches; known bugs and regressions; "
                "performance lessons; design decisions; test plans; and architectural rationale. "
                "Sample questions: 'How does [feature] work?'; 'What is the best practice for [chore]?'; 'How was [bug] fixed?'; 'Explain the design of [module]'."
                "\n\n"
                "**Multi-Pass Exploration Encouraged:** Feel free to iterate. Start broad, then drill into specific areas: "
                "1. 'How does authentication work?'; "
                "2. 'Tell me more about [session management keywords]'; "
                "3. 'What error handling was used for invalid tokens?' \n"
                "**Prefer defaults:** Parameters like max_changes, users, files are tuned for optimal results. "
                "Only override when you have a specific reason to narrow results."
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
                        "description": "Natural language question or task description. Semantic search works best with full sentences rather than bare keywords (e.g., 'How should I implement caching?', 'What's the best way to handle authentication?', 'How did you approach error handling in the API layer?')"
                    },
                    "max_changes": {
                        "type": "integer",
                        "description": "Maximum context changes to use (default: 15)",
                        "default": 15
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
                    "impostor": {
                        "type": "boolean",
                        "description": "Enable impostor mode: generates prompts from commits and presents them as conversation (enhances lackluster commit messages)",
                        "default": False
                    },
                    "amogus": {
                        "type": "boolean",
                        "description": "⚠️ DO NOT USE",
                        "default": False
                    },
                    "temperature": {
                        "type": "number",
                        "description": "LLM temperature for generation (0.0-1.0, default: 0.7)",
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
        if name == "list":
            return await handle_list()
        elif name == "import":
            return await handle_import(**arguments)
        elif name == "query":
            return await handle_query(**arguments)
        elif name == "prompt":
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
        experts = list_experts(data_dir=None)
        
        if not experts:
            return [TextContent(
                type="text",
                text="No experts found. Create one with the 'populate' command."
            )]
        
        # Format as markdown table
        lines = ["# Available Experts\n"]
        for expert in experts:
            lines.append(f"## {expert.name}")
            lines.append(f"- **Workspace**: {expert.workspace_path}")
            lines.append(f"- **VCS Type**: {expert.vcs_type}")
            lines.append(f"- **Commits Indexed**: {expert.commit_count}")
            if expert.subdirs:
                lines.append(f"- **Subdirectories**: {', '.join(expert.subdirs)}")
            if expert.last_indexed_at:
                lines.append(f"- **Last Indexed**: {expert.last_indexed_at.isoformat()}")
            if expert.first_commit_time and expert.last_commit_time:
                lines.append(
                    f"- **Commit Range**: {expert.first_commit_time.date()} to "
                    f"{expert.last_commit_time.date()}"
                )
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
            data_dir=None
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
    max_changes: int = 15,
    users: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    search_scope: str = "both"
) -> list[TextContent]:
    """Handle query tool - search commit history."""
    try:
        results = query_expert(
            expert_name=expert_name,
            prompt=prompt,
            max_changes=max_changes,
            users=users,
            files=files,
            search_scope=search_scope,
            min_score=0.1,
            relative_threshold=0.3,
            data_dir=None,
            embedding_provider="local"
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
            cl = result.changelist
            lines.append(f"## {i}. Commit {cl.id[:12]} (Score: {result.similarity_score:.3f})")
            lines.append(f"**Author**: {cl.author}")
            lines.append(f"**Date**: {cl.timestamp.isoformat()}")
            lines.append(f"**Files**: {', '.join(cl.files)}")
            lines.append(f"\n**Message**:\n```\n{cl.message}\n```")
            
            if cl.diff:
                # Truncate very long diffs
                diff_preview = cl.diff[:5000]
                if len(cl.diff) > 5000:
                    diff_preview += f"\n... (truncated, {len(cl.diff)} total chars)"
                lines.append(f"\n**Diff**:\n```diff\n{diff_preview}\n```")
            
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
    max_changes: int = 15,
    users: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    impostor: bool = False,
    amogus: bool = False,
    temperature: float = 0.7
) -> list[TextContent]:
    """Handle prompt tool - get AI recommendations."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        logger.info(f"[PROMPT] Starting prompt request for expert '{expert_name}'")
        logger.debug(f"[PROMPT] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        logger.debug(f"[PROMPT] Params: max_changes={max_changes}, impostor={impostor}, temperature={temperature}")
        
        # Accumulate streaming response
        full_response = ""
        chunk_count = 0
        first_chunk_time = None
        
        stream_start = time.time()
        logger.debug(f"[PROMPT] Initiating stream at +{stream_start - start_time:.2f}s")
        
        async for chunk in prompt_expert_stream(
            expert_name=expert_name,
            prompt=prompt,
            max_changes=max_changes,
            users=users,
            files=files,
            amogus=amogus,
            impostor=impostor,
            temperature=temperature,
            data_dir=None,
            embedding_provider="local",
            llm_provider=_llm_provider
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
        
        total_time = time.time() - start_time
        logger.info(f"[PROMPT] Completed in {total_time:.2f}s - received {chunk_count} chunks, {len(full_response)} chars")
        
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


async def main():
    """Main entry point for MCP server."""
    global _llm_provider
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Expert Among Us MCP Server")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--llm-provider',
                       type=str,
                       choices=['auto', 'openai', 'openrouter', 'ollama', 'bedrock', 'claude-code'],
                       default='auto',
                       help='LLM provider for AI recommendations (auto-detects by default)')
    args = parser.parse_args()
    
    # Store LLM provider choice in global variable
    _llm_provider = args.llm_provider
    
    # Get PID for log file name
    pid = os.getpid()
    
    # Configure logging to both stderr AND file when --debug is enabled
    log_level = logging.DEBUG if args.debug else logging.INFO
    handlers = [logging.StreamHandler(sys.stderr)]
    
    # Add file handler for debug mode
    if args.debug:
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
    
    if args.debug:
        logger.debug(f"Debug logging enabled - writing to mcp-{pid}.log")
        logger.debug(f"Process ID: {pid}")
    
    logger.info("Starting Expert Among Us MCP Server...")
    logger.info("Initializing MCP server...")
    
    # Warm up embedding model to avoid cold start timeout
    logger.info("Warming up embedding model (this may take ~60s on first run)...")
    from expert_among_us.embeddings.local import JinaCodeEmbedder
    from expert_among_us.config.settings import Settings
    warmup_start = time.time()
    settings = Settings(embedding_provider='local')
    embedder = JinaCodeEmbedder(
        model_id=settings.local_embedding_model,
        dimension=settings.local_embedding_dimension,
        compile_model=True
    )
    _ = embedder.dimension  # Force model load
    warmup_time = time.time() - warmup_start
    logger.info(f"Embedding model ready (took {warmup_time:.1f}s)")
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server ready and listening for requests")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="expert-among-us",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())