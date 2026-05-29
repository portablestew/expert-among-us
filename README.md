# Expert Among Us ඞ

MCP which indexes commit history, then uses secondary inference to form a queryable "expert".

## Why?

SDEs with practitioner-level knowledge of sprawling legacy codebases have instincts and thought processes not well represented in the final code structure. High-level documentation also fails to fully capture their experience (if such documentation even exists). A software expert's experience and instincts are often implied in their raw commit messages and code review comment discussions.

### Beyond Semantic File Searches

Traditional semantic code search tools focus on finding relevant files and functions based on current codebase state.
Expert Among Us goes deeper by **indexing your repository's entire commit history** in addition to current state.
This unlocks a trove of unintended expert documentation, all written in natural language specifically to communicate intent to humans.
By performing semantic search on commit messages alongside code diffs, Expert Among Us can:
- Better match natural language queries to the developer's original explanation of their changes
- Surface relevant context even when the code itself uses technical jargon or domain-specific terminology
- Find conceptual matches (e.g., "authentication" finds commits about "login", "security", "user verification")
- Understand intent beyond just matching function names or variable names in code

This makes searches like "How do I handle authentication?" far more accurate than searching code alone, since developers naturally describe these concepts clearly in their commit messages.
The result is key contextual insights that naive file search cannot provide:
- **Historical Context:** Understand not just what the code does, but why decisions were made and how solutions evolved over time
- **Hidden Insights:** Discover patterns in bug fixes, regressions, performance optimizations, and architectural changes that aren't visible in the final code
- **Thought Processes:** Capture the reasoning behind technical decisions through commit messages and diff patterns, even when formal documentation is lacking
- **Test Cases & Edge Cases:** Learn from past bug fixes and edge case handling that shaped the current implementation
- **Future Plans:** Identify intended directions and planned improvements mentioned in commit messages but not yet implemented
- **Evolution Patterns:** See how similar problems were solved across different parts of the codebase over time

### Key Differences from Traditional Search Approaches

| Static File Search | Traditional Semantic Search | Expert Among Us |
|--------------------|----------------------------|-----------------|
| Keyword/regex matching | Searches current file contents | Searches historical commit patterns |
| Shows matching code lines | Shows what code does | Shows why and how code evolved |
| No context understanding | Static snapshot | Temporal context and progression |
| Fast but literal | File-level relevance | Change-level insights |
| No semantic understanding | Limited to current codebase | Captures individual expert's style |
| Documentation-independent | Documentation-dependent | Can generate deep documentation |
| File paths and code only | Code semantics | Natural language text |
| No authorship context | No authorship context | Preserves expert's decision-making patterns |

### Case Study Validation

A **[blind comparative analysis](case-studies/summary.md)** of the expert-among-us MCP was conducted on the [OpenRA game engine](https://github.com/OpenRA/OpenRA/), comparing outcomes with and without the MCP across four technical scenarios. The analysis was performed without prior knowledge of expert-among-us or its purpose, including stripping the tool description from the conversation history. This provides an unbiased (albeit AI-generated) evaluation.

**Key Findings:**
- Completed all scenarios with roughly **20% fewer actions** overall, and context sizes comparable to non-MCP completion
- Successfully identified regressions and key patterns that standard exploration missed
- Provided historical context and design rationale not available through code inspection alone

The case study demonstrates measurable efficiency gains and qualitative improvements in debugging and architecture understanding. For the detailed comparison, see [case-studies/summary.md](case-studies/summary.md) and the [raw conversation files](case-studies/OpenRA/).

### Synthetic Commit Context

Not all commit messages are created equal. Fortunately, transformer LLMs are excellent at filling in the blanks.
When run with `--impostor` mode, Expert Among Us generates additional commit message content.
This is presented as an ordered chain of user prompt -> assistant response entries, where the user is the generated prompts, and the real commits are the assistant responses.
The actual user prompt is the final message. Effectively, a conversation is presented as if the LLM has authored all commits by itself. The AI acts as an impostor of the human experts.

## Overview

Expert Among Us creates a queryable "expert" from your repository's commit history using AI-powered semantic search and vector embeddings.
It combines your complete commit history with the current codebase state, enabling insights not possible with either approach alone.
It helps you understand development patterns, find relevant changes, and get AI-powered recommendations based on historical code changes.

### Key Capabilities

- **Semantic Search**: Find commits by meaning, not just keywords, using vector embeddings
- **Dual Indexing**: Seamlessly combines full commit history with current codebase state for comprehensive insights
- **AI-Powered Reranking**: Cross-encoder reranking dramatically improves search result relevance
- **Smart Text Sanitization**: Automatically removes high-entropy patterns (API keys, UUIDs, binary data) to improve search quality
- **Metadata Extraction**: Index commit messages, authors, files, and code diffs
- **Vector Embeddings**: Supports local (GPU-accelerated) or cloud (AWS Bedrock) embedding models
- **Flexible Filtering**: Search by author, files, or time period
- **Version Control Support**: Works with Git and Perforce repositories
- **Commit Enhancement**: Optionally adds LLM-generated analysis of a commit to its context

### Search Quality Features

Expert Among Us includes several features that significantly improve search quality and relevance:

**Cross-Encoder Reranking**
- Uses modern cross-encoder models to re-rank search results
- Provides dramatically better relevance than vector search alone
- Works seamlessly with all search scopes (metadata, diffs, files)

**Smart Text Sanitization**
- Automatically removes high-entropy patterns like API keys, UUIDs, and binary data
- Preserves semantic meaning while reducing noise in embeddings
- Improves search quality by focusing on meaningful code patterns

**Dual-Source Indexing**
- Indexes both historical commit patterns and current file content
- Seamlessly combines insights from development history with present-day code structure
- Enables queries that span both "how we got here" and "what's here now"

## Installation

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### End-Users (Recommended)

Install as a persistent tool with GPU support (NVIDIA CUDA):

```bash
uv tool install expert-among-us --index pytorch-cu130=https://download.pytorch.org/whl/cu130
```

If you don't have an NVIDIA GPU, install without the index flag (CPU-only):

```bash
uv tool install expert-among-us
```

After installation, `expert-among-us` is available directly on your PATH:

```bash
expert-among-us --help
```

#### Updating

```bash
uv tool upgrade expert-among-us --upgrade
```

This retains the CUDA index setting from the original install. The `--upgrade` flag forces a fresh check against PyPI.

#### GPU Notes

- **Prerequisites:** NVIDIA GPU with compatible drivers (`nvidia-smi` should show your GPU). A separate CUDA toolkit install is not required — the PyTorch wheel bundles its own CUDA runtime.
- **Performance:** GPU embeddings run ~8x faster (~0.5s vs ~4s per commit batch).
- **Why not `uvx`?** `uvx` resolves dependencies from PyPI, which only ships CPU-only PyTorch wheels on Windows and macOS. The `--index` flag on `uv tool install` directs torch to the CUDA wheel index.

### Local Development

For contributors working from a clone:

```bash
# Linux/macOS
./install.sh

# Windows (PowerShell)
.\install.ps1
```

These scripts install uv if needed, then run `uv sync` to set up the environment with CUDA-enabled PyTorch (configured via `[tool.uv.sources]` in `pyproject.toml`). Use `uv run` to execute commands:

```bash
uv run expert-among-us --help
```

## Quick Start

### 1. Index a Repository

Create an expert index from your git repository:

```bash
# Index entire repository (uses local embeddings by default)
expert-among-us populate MyExpert /path/to/repo

# Use AWS Bedrock embeddings instead
expert-among-us --embedding-provider bedrock populate MyExpert /path/to/repo

# Index specific subdirectories only
expert-among-us populate MyExpert /path/to/repo src/main/ src/resources/

# Limit the number of commits to index
expert-among-us populate MyExpert /path/to/repo --max-commits 5000
```

**Note**: On first run with local embeddings, the Jina Code model (~1.2GB) will be downloaded automatically. This is a one-time download.

The first indexing will take some time depending on repository size. Subsequent runs are incremental and only process new commits.

### 2. Search for Similar Changes

Find commits similar to your query:

```bash
# Basic search
expert-among-us query MyExpert "How to add a new feature?"

# Search with filters
expert-among-us query MyExpert "Bug fix for memory leak" \
    --users john,jane \
    --files src/main.py,src/utils.py \
    --max-changes 20

# Save results to JSON
expert-among-us query MyExpert "API endpoint implementation" \
    --output results.json
```

**Important**: Use the same `--embedding-provider` for querying as you used during indexing.

### 3. Get AI Recommendations

Get AI-powered recommendations that impersonate the expert based on their historical commit patterns:

```bash
# Get recommendations (auto-detects LLM provider)
expert-among-us prompt MyExpert "How should I implement authentication?"

# With filters for specific context
expert-among-us prompt MyExpert "How to handle errors?" \
    --users alice,bob \
    --files src/handlers/

# With improved commit message context
expert-among-us prompt MyExpert "Add caching" --impostor

# With debug logging to inspect API calls
expert-among-us --debug prompt MyExpert "Optimize queries"
```

**How It Works:**
1. Auto-detects available LLM provider (or use explicit `--llm-provider`)
2. Searches for relevant commits using semantic similarity
3. Generates conversational prompts from historical diffs
4. Builds a conversation showing the expert's past work
5. Streams an AI response impersonating the expert's style

## CLI Command Reference

> **Note:** All examples below assume `expert-among-us` is installed via `uv tool install`. When running from a local clone, prefix with `uv run`.

### `populate` - Index Repository

Create or update an expert index from a repository.

```bash
expert-among-us populate EXPERT_NAME [WORKSPACE] [SUBDIRS...] [OPTIONS]
```

**Arguments:**
- `EXPERT_NAME`: Unique name for this expert (used to identify the index)
- `WORKSPACE`: Path to the repository root directory (required for new experts, optional for updates)
- `SUBDIRS`: Optional subdirectories to filter (e.g., `src/main/ src/resources/`)

**Options:**
- `--max-commits INTEGER`: Maximum number of commits to index (default: 60000)
- `--max-batches INTEGER`: Maximum batches to run (returns exit code 2 if more remain)
- `--batch-size INTEGER`: Maximum commits per embedding batch (default: 1000)
- `--start-at TEXT`: Start indexing from a specific commit hash
- `--index-scope [metadata|diffs|files|all]`: What to index (default: all)
- `--allowed-extensions TEXT`: Comma-separated list of allowed file extensions
- `--compact-diffs`: Reduce diff size by removing context (trades search quality for cost)
- `--custom-sanitize-pattern TEXT`: Custom regex pattern to remove from text before embedding

**Global Options (before command):**
- `--embedding-provider [local|bedrock]`: Embedding provider (default: local)
- `--data-dir PATH`: Base directory for expert data storage (default: ~/.expert-among-us)
- `--gpu-memory-multiplier FLOAT`: GPU memory scaling factor (default: 1.0)
- `--debug`: Enable debug logging

**Examples:**
```bash
# Index entire repository with local embeddings (default)
expert-among-us populate AppExpert ~/projects/myapp

# Index with AWS Bedrock embeddings
expert-among-us --embedding-provider bedrock populate AppExpert ~/projects/myapp

# Index only backend code
expert-among-us populate BackendExpert ~/projects/myapp src/backend/ src/api/

# Update existing expert (workspace looked up automatically)
expert-among-us populate AppExpert

# Use custom data directory
expert-among-us --data-dir /mnt/data/experts populate AppExpert ~/projects/myapp
```

### `list` - List Available Experts

Display all indexed experts and their metadata.

```bash
expert-among-us list
```

### `import` - Import Expert via Symlink

Import an expert from an external directory by creating a symlink.

```bash
expert-among-us import SOURCE_PATH
```

### `query` - Search History

Search for commits similar to your query using semantic search.

```bash
expert-among-us query EXPERT_NAME PROMPT [OPTIONS]
```

**Arguments:**
- `EXPERT_NAME`: Name of the expert to query
- `PROMPT`: Search query describing what you're looking for

**Options:**
- `--max-changes INTEGER`: Maximum changelist results (default: 20)
- `--max-file-chunks INTEGER`: Maximum file chunk results (default: 10)
- `--users TEXT`: Filter by commit authors (comma-separated)
- `--files TEXT`: Filter by file paths (comma-separated)
- `--search-scope [all|metadata|diffs|files]`: Search scope (default: all)
- `--no-reranking`: Disable cross-encoder reranking (faster but less accurate)
- `--min-score FLOAT`: Minimum similarity score threshold (default: 0.1)
- `--relative-threshold FLOAT`: Relative score threshold as fractional drop from top result (default: 0.8)
- `--expansion-candidate-multiplier INTEGER`: Multiplier for candidate retrieval during expansion (default: 5)
- `--expansion-passes INTEGER`: Number of expansion iterations (default: 1)
- `--output PATH`: Save results to JSON file

**Examples:**
```bash
# Find commits about authentication
expert-among-us query AppExpert "authentication implementation"

# Search with author filter
expert-among-us query AppExpert "database optimization" --users alice,bob

# Search only current file content
expert-among-us query AppExpert "function implementation" --search-scope files

# Strict filtering
expert-among-us query AppExpert "exact pattern" --min-score 0.3 --relative-threshold 0.2
```

### `prompt` - AI Recommendations

Get AI-powered recommendations that impersonate the expert based on their historical commit patterns.

```bash
expert-among-us [GLOBAL OPTIONS] prompt EXPERT_NAME PROMPT [OPTIONS]
```

**Global Options (must come before command):**
- `--llm-provider [auto|openai|openrouter|ollama|bedrock|claude-code|kiro-cli]`: LLM provider (auto-detects by default)
- `--base-url-override TEXT`: Override base URL for OpenAI-compatible providers
- `--expert-model TEXT`: Override default expert model
- `--promptgen-model TEXT`: Override default promptgen model
- `--debug`: Enable debug logging

**Arguments:**
- `EXPERT_NAME`: Name of the expert to query
- `PROMPT`: Question or task description for the AI

**Options:**
- `--max-changes INTEGER`: Maximum context changes to use (default: 20)
- `--users TEXT`: Filter by commit authors (comma-separated)
- `--files TEXT`: Filter by file paths (comma-separated)
- `--impostor`: Generate synthetic prompts for each commit (improves poor commit messages)
- `--amogus`: Enable Among Us mode
- `--temperature FLOAT`: LLM temperature (0.0–1.0, default: 0.7)

**Examples:**
```bash
# Basic usage (auto-detects provider)
expert-among-us prompt AppExpert "How to implement caching?"

# Explicitly specify OpenAI
expert-among-us --llm-provider openai prompt AppExpert "How to implement caching?"

# With impostor mode for better context
expert-among-us prompt AppExpert "Add caching" --impostor

# With debug logging
expert-among-us --debug prompt AppExpert "Optimize queries"
```

## Configuration

### Storage Location

By default, expert indexes are stored in: `~/.expert-among-us/data/`

Customize with the `--data-dir` global option. Always use the same `--data-dir` for all operations on the same expert.

Each expert creates:
- **ChromaDB**: Vector embeddings (`{data-dir}/data/{expert-name}/chroma/`)
- **SQLite**: Metadata (`{data-dir}/data/{expert-name}/metadata.db`)
- **Debug Logs**: API call logs when `--debug` is enabled (`{data-dir}/logs/`)

### Embedding Models

Expert Among Us supports two embedding providers. Use `--embedding-provider` to switch.
**Important**: You must use the same provider for both indexing and querying.

**Local (Default):**
- **Model**: `jinaai/jina-code-embeddings-0.5b` (`code2code` task)
- **Dimension**: 512 (Matryoshka truncation from 896)
- **Max tokens**: 32,768
- **Download**: ~1.2GB (one-time, automatic)
- **Advantages**: No API costs, works offline, GPU-accelerated

**AWS Bedrock:**
- **Model**: `amazon.titan-embed-text-v2:0`
- **Dimension**: 1024
- **Max tokens**: 8,000
- **Requirements**: AWS credentials and Bedrock access

### LLM Providers

Expert Among Us supports multiple LLM providers for prompt generation. By default, it **auto-detects** an available provider.

#### Auto-Detection Order

1. **Environment Variables** (must be exactly one):
   - `AWS_ACCESS_KEY_ID` → AWS Bedrock
   - `OPENROUTER_API_KEY` → OpenRouter
   - `OPENAI_API_KEY` → OpenAI
2. **AWS Default Credentials** → Bedrock
3. **Claude Code CLI** (`claude` on PATH) → Claude Code
4. **Ollama Server** (localhost:11434) → Ollama

#### Provider Setup

| Provider | Required | Notes |
|----------|----------|-------|
| `openai` | `OPENAI_API_KEY` | [Get key](https://platform.openai.com/api-keys) |
| `openrouter` | `OPENROUTER_API_KEY` | [Get key](https://openrouter.ai/settings/keys) — free models available |
| `ollama` | Ollama running locally | Default: `http://127.0.0.1:11434/v1` |
| `bedrock` | AWS credentials | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| `claude-code` | `claude` CLI on PATH | [Install](https://www.claude.com/product/claude-code) |
| `kiro-cli` | Kiro IDE running | Used within Kiro sessions |

**Default Models (Bedrock):**
- **Prompt Generation**: `us.amazon.nova-lite-v1:0`
- **Expert Analysis**: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`

Use `--promptgen-model` and `--expert-model` to override.

## MCP Integration

Expert Among Us provides a fully implemented MCP (Model Context Protocol) server, allowing AI assistants to query your codebase history directly.

### Available MCP Tools

1. **experts-list** - List all available experts with metadata
2. **experts-import** - Import external experts via symlink
3. **expert-query** - Get raw commit details for manual analysis
4. **expert-prompt** - Get AI-powered recommendations based on expert's historical patterns

### Starting the MCP Server

```bash
expert-among-us mcp

# With options
expert-among-us --debug mcp --impostor

# From a local clone
uv run expert-among-us mcp
```

### MCP Server CLI Arguments

- `--data-dir`: Custom data directory location
- `--impostor`: Enable impostor mode for all queries
- `--debug`: Enable debug logging
- `--llm-provider`: Choose LLM provider
- `--embedding-provider`: Choose embedding provider (default: local)
- `--max-response-tokens`: Maximum tokens for expert response (default: 4096)
- `--prompt-timeout-seconds`: Maximum seconds for expert-prompt operations (default: no timeout)

### Configuration for MCP Clients

#### Example: Claude Desktop (Linux/macOS)

Config: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "expert-among-us": {
      "command": "expert-among-us",
      "args": ["mcp"],
      "timeout": 120,
      "alwaysAllow": ["experts-list", "expert-prompt", "expert-query"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

#### Example: Claude Desktop (Windows)

Config: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "expert-among-us": {
      "command": "expert-among-us",
      "args": ["--debug", "mcp"],
      "timeout": 120,
      "alwaysAllow": ["experts-list", "expert-prompt", "expert-query"],
      "env": {
        "AWS_PROFILE": "your-profile-here"
      }
    }
  }
}
```

#### Example: With Impostor Mode

```json
{
  "mcpServers": {
    "expert-among-us": {
      "command": "expert-among-us",
      "args": ["mcp", "--impostor"],
      "timeout": 120,
      "alwaysAllow": ["experts-list", "expert-prompt", "expert-query"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

#### Example: Kiro MCP Configuration

`.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "expert-among-us": {
      "command": "expert-among-us",
      "args": ["--data-dir", "/path/to/shared/experts", "mcp"],
      "timeout": 120,
      "autoApprove": ["experts-list", "expert-query", "expert-prompt"],
      "env": {
        "AWS_PROFILE": "your-profile-here"
      }
    }
  }
}
```

**Important Notes:**
- These examples assume `expert-among-us` is installed via `uv tool install` (see [Installation](#installation))
- Set required environment variables in the `env` section
- Restart your MCP client after updating the configuration

## Development

### Running Tests

```bash
uv run pytest

# With coverage
uv run pytest --cov=expert_among_us --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

## License

MIT License - see LICENSE file for details
