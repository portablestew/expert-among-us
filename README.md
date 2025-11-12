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
- Completed all scenarios with roughly **20% fewer actions** overall, and context sizes comparble to non-MCP completion
- Successfully identified regressions and key patterns that standard exploration missed
- Provided historical context and design rationale not available through code inspection alone

The case study demonstrates measurable efficiency gains and qualitative improvements in debugging and architecture understanding. For the detailed comparison, see [case-studies/summary.md]() and the [raw conversation files](case-studies/OpenRA/).

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
- **Version Control Support**: Works with Git repositories (Perforce support planned)
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

### Quick Install (Recommended)

Use the provided installation scripts to automate the setup process:

#### CPU-Only Installation

**Linux/macOS:**
```bash
./install-cpu.sh
```

**Windows (PowerShell):**
```powershell
.\install-cpu.ps1
```

#### GPU Installation (NVIDIA GPUs)

For 10-20x faster local embeddings with NVIDIA GPU support:

**Linux/macOS:**
```bash
./install-gpu.sh
```

**Windows (PowerShell):**
```powershell
.\install-gpu.ps1
```

The GPU installation scripts will:
- Install Python dependencies
- Install CUDA-enabled PyTorch
- Verify GPU detection
- Provide usage instructions

**Performance Impact:**
- **With GPU**: ~0.5s per commit embedding
- **CPU only**: ~4s per commit embedding

### Quick Run Scripts

For convenience, you can use the provided run scripts instead of activating the virtual environment:

**Linux/macOS:**
```bash
./run.sh --help
./run.sh populate /path/to/repo MyExpert
./run.sh query /path/to/repo MyExpert "your question"
```

**Windows (PowerShell):**
```powershell
.\run.ps1 --help
.\run.ps1 populate /path/to/repo MyExpert
.\run.ps1 query /path/to/repo MyExpert "your question"
```

These scripts automatically use the virtual environment without activating it, leaving no side effects after execution. They work with both CPU and GPU installations.

**Why not use `uv run`?** The `uv run` command resyncs the environment to the lock file, which would revert to CPU-only PyTorch. To preserve GPU PyTorch, use the run scripts which automatically use the venv without resyncing.

## Quick Start

### 1. Index a Repository

Create an expert index from your git repository:

```bash
# Index entire repository (uses local embeddings by default)
./run.sh populate /path/to/repo MyExpert

# Use AWS Bedrock embeddings instead
./run.sh populate /path/to/repo MyExpert --embedding-provider bedrock

# Index specific subdirectories only
./run.sh populate /path/to/repo MyExpert src/main/ src/resources/

# Limit the number of commits to index
./run.sh populate /path/to/repo MyExpert --max-commits 5000
```

**Note**: On first run with local embeddings, the Jina Code model (~1.2GB) will be downloaded automatically. This is a one-time download.

The first indexing will take some time depending on repository size. Subsequent runs are incremental and only process new commits.

### 2. Search for Similar Changes

Find commits similar to your query:

```bash
# Basic search (uses local embeddings by default)
./run.sh query /path/to/repo MyExpert "How to add a new feature?"

# Use same embedding provider as during indexing
./run.sh query /path/to/repo MyExpert "How to add a new feature?" \
    --embedding-provider bedrock

# Search with filters
./run.sh query /path/to/repo MyExpert "Bug fix for memory leak" \
    --users john,jane \
    --files src/main.py,src/utils.py \
    --max-changes 20

# Save results to JSON
./run.sh query /path/to/repo MyExpert "API endpoint implementation" \
    --output results.json
```

**Important**: Use the same `--embedding-provider` for querying as you used during indexing.

### 3. Get AI Recommendations

Get AI-powered recommendations that impersonate the expert based on their historical commit patterns:

```bash
# Get recommendations (auto-detects provider by default)
./run.sh prompt /path/to/repo MyExpert "How should I implement authentication?"

# Explicitly specify provider if needed
./run.sh --llm-provider openai prompt /path/to/repo MyExpert "How should I implement authentication?"

# With filters for specific context
./run.sh prompt /path/to/repo MyExpert "How to handle errors?" \
    --users alice,bob \
    --files src/handlers/

# With improved commit message context
./run.sh prompt /path/to/repo MyExpert "Add caching" \
    --impostor

# With debug logging to inspect API calls
./run.sh --debug prompt /path/to/repo MyExpert "Optimize queries"
```

**How It Works:**
1. Auto-detects available LLM provider (or use explicit `--llm-provider`)
2. Searches for relevant commits using semantic similarity
3. Generates conversational prompts from historical diffs
4. Builds a conversation showing the expert's past work
5. Streams an AI response impersonating the expert's style

## CLI Command Reference

### `populate` - Index Repository

Create or update an expert index from a repository.

```bash
./run.sh populate WORKSPACE EXPERT_NAME [SUBDIRS...] [OPTIONS]
```

**Arguments:**
- `WORKSPACE`: Path to the repository root directory
- `EXPERT_NAME`: Unique name for this expert (used to identify the index)
- `SUBDIRS`: Optional subdirectories to filter (e.g., `src/main/ src/resources/`)

**Options:**
- `--max-commits INTEGER`: Maximum number of commits to index (default: 10000)
- `--vcs-type [git|p4]`: Version control system type (default: git)
- `--embedding-provider [local|bedrock]`: Embedding provider (default: local)
- `--data-dir PATH`: Base directory for expert data storage (default: ~/.expert-among-us)

**Examples:**
```bash
# Index entire repository with local embeddings (default)
./run.sh populate ~/projects/myapp "AppExpert"

# Index with AWS Bedrock embeddings
./run.sh populate ~/projects/myapp "AppExpert" --embedding-provider bedrock

# Index only backend code
./run.sh populate ~/projects/myapp "BackendExpert" src/backend/ src/api/

# Limit indexing to recent commits
./run.sh populate ~/projects/myapp "RecentExpert" --max-commits 1000

# Use custom data directory
./run.sh --data-dir /mnt/data/experts populate ~/projects/myapp "AppExpert"
```

### `list` - List Available Experts

Display all indexed experts and their metadata.

**Options:**
- `--data-dir PATH`: Optional base directory for expert data storage (default: ~/.expert-among-us)

**Examples:**
```bash
# List all experts
./run.sh list

# List from custom directory
./run.sh --data-dir /mnt/data/experts list
```

### `import` - Import Expert via Symlink

Import an expert from an external directory by creating a symlink.

**Arguments:**
- `SOURCE_PATH`: Path to the expert directory to import (must contain metadata.db)

**Examples:**
```bash
# Import from external storage
./run.sh import /external/storage/MyExpert

# Import from network location
./run.sh import ~/shared/experts/TeamExpert

# Import with custom data directory
./run.sh --data-dir /custom/location import /external/MyExpert
```

**Notes:**
- Source directory must contain a valid expert (metadata.db file)
- Expert name is extracted from the source directory name
- Fails if an expert with the same name already exists
- On Windows, requires Administrator privileges or Developer Mode enabled

### `query` - Search History

Search for commits similar to your query using semantic search.

```bash
./run.sh query WORKSPACE EXPERT_NAME PROMPT [OPTIONS]
```

**Arguments:**
- `WORKSPACE`: Path to the repository root directory
- `EXPERT_NAME`: Name of the expert to query
- `PROMPT`: Search query describing what you're looking for

**Options:**
- `--max-changes INTEGER`: Maximum number of results to return (default: 10)
- `--users TEXT`: Filter by commit authors (comma-separated, e.g., "john,jane")
- `--files TEXT`: Filter by file paths (comma-separated, e.g., "src/main.py,src/utils.py")
- `--search-scope [all|metadata|diffs|files]`: Search scope - "all" (default), "metadata" only, "diffs" only, or "files" only
- `--no-reranking`: Disable cross-encoder reranking (faster but less accurate)
- `--min-score FLOAT`: Minimum similarity score threshold (default: 0.1)
- `--relative-threshold FLOAT`: Relative score threshold as fractional drop from top result (default: 0.8)
- `--output PATH`: Save results to JSON file
- `--embedding-provider [local|bedrock]`: Embedding provider - must match what was used during indexing (default: local)
- `--data-dir PATH`: Base directory for expert data storage (default: ~/.expert-among-us)

**Examples:**
```bash
# Find commits about authentication
./run.sh query ~/projects/myapp "AppExpert" "authentication implementation"

# Search with author filter
./run.sh query ~/projects/myapp "AppExpert" "database optimization" \
    --users alice,bob

# Search specific files and save results
./run.sh query ~/projects/myapp "AppExpert" "error handling" \
    --files src/handlers/ \
    --output search-results.json \
    --max-changes 20

# Search only current file content
./run.sh query ~/projects/myapp "AppExpert" "function implementation" \
    --search-scope files

# Search only commit history
./run.sh query ~/projects/myapp "AppExpert" "why was this changed?" \
    --search-scope metadata

# Faster search without reranking
./run.sh query ~/projects/myapp "AppExpert" "quick search" \
    --no-reranking

# Strict filtering with high similarity threshold
./run.sh query ~/projects/myapp "AppExpert" "exact pattern" \
    --min-score 0.3 --relative-threshold 0.2

# Query with custom data directory
./run.sh --data-dir /mnt/data/experts query ~/projects/myapp "AppExpert" "feature implementation"
```

### `prompt` - AI Recommendations

Get AI-powered recommendations that impersonate the expert based on their historical commit patterns.
The system uses relevant past commits as examples to generate responses that match the expert's coding style and approach.

```bash
./run.sh [--llm-provider PROVIDER] prompt WORKSPACE EXPERT_NAME PROMPT [OPTIONS]
```

**Global Options (must come before command):**
- `--llm-provider [auto|openai|openrouter|ollama|bedrock|claude-code]`: LLM provider to use (auto-detects by default)
- `--base-url-override TEXT`: Override base URL for OpenAI-compatible providers (openai, openrouter, ollama)
- `--debug`: Enable debug logging of API calls
- `--embedding-provider [local|bedrock]`: Embedding provider - must match what was used during indexing (default: local)
- `--data-dir PATH`: Base directory for expert data storage (default: ~/.expert-among-us)

**Arguments:**
- `WORKSPACE`: Path to the repository root directory
- `EXPERT_NAME`: Name of the expert to query
- `PROMPT`: Question or task description for the AI

**Options:**
- `--max-changes INTEGER`: Maximum context changes to use (default: 15)
- `--users TEXT`: Filter by commit authors (comma-separated)
- `--files TEXT`: Filter by file paths (comma-separated)
- `--impostor`: Generates a user prompt for each commit, and places commit content in "assistant" messages
  - Enhances lackluster commit messages by adding a feasible thought process
  - Tricks the LLM into thinking it created all the code, but it is an AI impostor
- `--amogus`: Enable Among Us mode (the LLM expert is your crewmate.. or are they?)
- `--temperature FLOAT`: LLM temperature for generation (0.0-1.0, default: 0.7) -- if the provider supports it

**LLM Provider Selection:**
By default, the system auto-detects an available provider. You can explicitly specify a provider with `--llm-provider`. Each provider requires specific environment variables:
- `auto`: Auto-detect available provider (default)
- `openai`: Requires `OPENAI_API_KEY`
- `openrouter`: Requires `OPENROUTER_API_KEY`
- `ollama`: Uses default endpoint at `http://127.0.0.1:11434/v1` (override with `--base-url-override`)
- `bedrock`: Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
- `claude-code`: Requires Claude Code CLI to be installed

**Base URL Override:**
Use `--base-url-override` to use a different API endpoint, for example a custom proxy, router, or localhost port.
It works with any OpenAI-compatible provider (openai, openrouter, ollama).

**Debug Logging:**
When `--debug` is enabled, all API requests and responses are logged to JSON files in `~/.expert-among-us/logs/` for troubleshooting and analysis.

**Examples:**
```bash
# Basic usage (auto-detects provider)
./run.sh prompt ~/projects/myapp "AppExpert" "How to implement caching?"

# Explicitly specify OpenAI
./run.sh --llm-provider openai prompt ~/projects/myapp "AppExpert" "How to implement caching?"

# With filters (auto-detects provider)
./run.sh prompt ~/projects/myapp "AppExpert" "Optimize database queries" \
    --users alice,bob \
    --files src/db/

# Using debug mode
./run.sh --debug prompt ~/projects/myapp "AppExpert" "Add error handling"

# With improved commit message context
./run.sh prompt ~/projects/myapp "AppExpert" "Add caching" \
    --impostor

# With Among Us mode (don't use)
./run.sh prompt ~/projects/myapp "AppExpert" "Implement authentication" \
    --amogus
```

## Configuration

### Storage Location

By default, expert indexes are stored in: `~/.expert-among-us/data/`

You can customize the storage location using the `--data-dir` global option:

```bash
# Use custom data directory
./run.sh --data-dir /mnt/data/experts populate /path/to/repo MyExpert

# Query from custom location (must match where data was indexed)
./run.sh --data-dir /mnt/data/experts query /path/to/repo MyExpert "search query"
```

**Important**: Always use the same `--data-dir` value for all operations on the same expert.

Each expert creates two databases within the data directory:
- **ChromaDB**: Vector embeddings for semantic search (`{data-dir}/data/{expert-name}/chroma/`)
- **SQLite**: Metadata (commit info, authors, files, diffs) (`{data-dir}/data/{expert-name}/metadata.db`)
- **Debug Logs**: API call logs when `--debug` is enabled (`{data-dir}/logs/`)

### Embedding Models

Expert Among Us supports two embedding providers. Use `--embedding-provider` flag to switch between them.
**Important**: You must use the same provider for both indexing and querying, as different providers produce incompatible embeddings.

**Local (Default):**
- **Model**: `jinaai/jina-code-embeddings-0.5b` (`code2code` task, optimized for code similarity)
- **Dimension**: 512 (Matryoshka truncation from 896)
- **Max tokens**: 32,768
- **Download**: ~1.2GB (one-time, automatic on first run)
- **Advantages**: No API costs, works offline, fast CPU inference, no credentials needed

**AWS Bedrock:**
- **Model**: `amazon.titan-embed-text-v2:0`
- **Dimension**: 1024
- **Max tokens**: 8,000
- **Advantages**: Managed service with high availability, no local storage needed
- **Requirements**: AWS credentials and Bedrock access

**Usage Examples:**
```bash
# Use local embeddings (default)
./run.sh populate /path/to/repo MyExpert

# Use AWS Bedrock embeddings
./run.sh populate /path/to/repo MyExpert --embedding-provider bedrock

# Query must use same provider as indexing
./run.sh query /path/to/repo MyExpert "query" --embedding-provider bedrock
```

### LLM Providers

Expert Among Us supports multiple LLM providers for prompt generation and recommendations. By default, it **auto-detects** an available provider, or you can explicitly specify one with `--llm-provider`.

#### Auto-Detection (Default)

When you run `prompt` without specifying `--llm-provider`, the system automatically detects available providers in this order:

1. **Environment Variables** (must be exactly one):
   - `AWS_ACCESS_KEY_ID` → Uses AWS Bedrock
   - `OPENROUTER_API_KEY` → Uses OpenRouter
   - `OPENAI_API_KEY` → Uses OpenAI
   - If multiple are set, you must explicitly specify with `--llm-provider`

2. **AWS Default Credentials**: Checks for boto3 default profile → Uses Bedrock

3. **Claude Code CLI**: Checks if `claude` command is on PATH → Uses Claude Code

4. **Ollama Server**: Checks if Ollama is running on `localhost:11434` → Uses Ollama

5. **Error**: If none found, shows error message with setup instructions

#### OpenAI Provider

Use OpenAI's GPT models for AI recommendations:

**Required Environment Variables:**
- `OPENAI_API_KEY`: (required) Your OpenAI API key
- Create one at https://platform.openai.com/api-keys -- Add credit before using

**Example Configuration:**
```bash
# Basic OpenAI setup
export OPENAI_API_KEY=sk-proj-...

./run.sh --llm-provider openai prompt ~/projects/myapp "AppExpert" "How to implement auth?"
```

#### OpenRouter Provider

Use OpenRouter to access multiple LLM providers through a single API:

**Required Environment Variables:**
- `OPENROUTER_API_KEY`: (required) Your OpenRouter API key
- Get one for free at https://openrouter.ai/settings/keys -- Improved rate limits with one-time $10

**Base URL:** `https://openrouter.ai/api/v1`

**Supported Models:**
- Use `--promptgen-model` and `--expert-model` to select the LLM models to invoke
- Suitable free models are used by default: meta-llama/llama-3.3-70b-instruct:free, minimax/minimax-m2:free
- See more available models at [openrouter.ai/models](https://openrouter.ai/models)

**Example Configuration:**
```bash
# Basic OpenRouter setup
export OPENROUTER_API_KEY=sk-or-v1-...

./run.sh --llm-provider openrouter prompt ~/projects/myapp "AppExpert" "Optimize database queries"
```

#### Ollama Provider

Use Ollama for local LLM inference with an OpenAI-compatible API:

**Installation:**

1. Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)

2. Install the default models from settings.py:
   ```bash
   # Install the expert model (for main recommendations)
   ollama pull deepseek-coder-v2:16b
   
   # Install the promptgen model (for generating prompts from diffs)
   ollama pull qwen2.5-coder:7b
   ```

3. Start the Ollama server (usually starts automatically after installation):
   ```bash
   ollama serve
   ```

**Configuration:**

**Default Endpoint:** `http://127.0.0.1:11434/v1` (no configuration needed)

**Override Endpoint:** Use `--base-url-override` flag if Ollama is running on a different host/port

**Example Configurations:**

```bash
# Use default endpoint (recommended)
./run.sh --llm-provider ollama prompt ~/projects/myapp "AppExpert" "Add caching layer"

# Use custom endpoint (if Ollama is on a different host)
./run.sh --llm-provider ollama --base-url-override http://192.168.1.100:11434/v1 \
    prompt ~/projects/myapp "AppExpert" "Add caching layer"

# Use with specific models
./run.sh --llm-provider ollama \
    --expert-model devstral:24b \
    --promptgen-model llama3:8b \
    prompt ~/projects/myapp "AppExpert" "Implement authentication"

# Override endpoint for any OpenAI-compatible local LLM
./run.sh --llm-provider openai --base-url-override http://localhost:8080/v1 \
    prompt ~/projects/myapp "AppExpert" "Add feature"
```

**Supported Models:** See available models at [ollama.com/library](https://ollama.com/library)

#### AWS Bedrock Provider

Use AWS Bedrock's managed LLM services:

**Requirements:**
- Credentials from a billing-enabled AWS credentials and Bedrock access
- Additional steps in the AWS Bedrock console may be required to opt into certain models

**Required Environment Variables:**
AWS credentials are required (configured via AWS CLI or environment variables):
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_REGION`: AWS region (e.g., `us-east-1`)

**Example Configuration:**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

./run.sh --llm-provider bedrock prompt ~/projects/myapp "AppExpert" "Implement retry logic"
```

**Default Models:**
- **Prompt Generation**: `us.amazon.nova-lite-v1:0`
- **Expert Analysis**: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Use `--promptgen-model` and `--expert-model` to select diferent LLM models

#### Claude Code Provider

Use a Claude Pro/Max subscription for inference via Anthropic's CLI interface:

**Requirements:**
- Claude Code CLI must be installed and configured separately
- The `claude` command must be available in your system PATH
- https://www.claude.com/product/claude-code

**Usage:**
```bash
# Confirm Claude CLI is available
which claude

./run.sh --llm-provider claude-code prompt /path/to/repo MyExpert "your question"
```

**Note:** This provider is primarily for users who already have the Claude Code CLI set up. Most users should prefer the other LLM providers.

### Limits and Defaults

- **Max commits**: 10,000 (configurable with `--max-commits`)
- **Max diff size**: 100KB (larger diffs are truncated)
- **Max embedding text**: 30KB
- **AWS Region**: us-east-1 (configurable via `AWS_REGION` environment variable)

### Filtering Options

When indexing with subdirectories:
```bash
./run.sh populate /path/to/repo MyExpert src/main/ src/resources/
```

Only commits affecting files in those subdirectories will be indexed.

When searching, filter by:
- **Authors**: `--users john,jane` (comma-separated list)
- **Files**: `--files src/main.py,tests/` (comma-separated paths/patterns)
- **Max results**: `--max-changes 20` (number of results to return)

## Architecture

Expert Among Us uses a sophisticated multi-layered approach:

1. **Dual-Source Indexing**: Seamlessly combines full commit history with current codebase state
2. **AI-Powered Search**: Cross-encoder reranking dramatically improves result relevance
3. **Smart Text Processing**: Automatic sanitization removes noise while preserving meaning
4. **Multi-Collection Vector Database**: Separate collections for metadata, diffs, and files for optimal performance
5. **Metadata Filtering**: Fast filtering by author, files, and timestamps using SQLite
6. **Incremental Updates**: Only new commits are processed on subsequent runs
7. **Diff Processing**: Code diffs are embedded for semantic code change search

## MCP Integration

Expert Among Us provides a fully implemented MCP (Model Context Protocol) server, allowing AI assistants like Claude Desktop to query your codebase history directly. The MCP server gives AI assistants access to your expert indexes through four powerful tools.

### Available MCP Tools

1. **list** - List all available experts with metadata (commit counts, time ranges, workspace paths)
2. **import** - Import external experts via symlink (useful for team-shared or network-stored experts)
3. **query** - Get raw commit details for manual analysis (complete messages, diffs, files, authors)
4. **prompt** - Get AI-powered recommendations based on expert's historical patterns (recommended)

### Setup Instructions

#### Starting the MCP Server

Use the provided run scripts to start the MCP server:

**Linux/macOS:**
```bash
./run-mcp.sh
```

**Windows (PowerShell):**
```powershell
.\run-mcp.ps1
```

**Auto-Installation Feature:**
The run scripts automatically check for the virtual environment. If `.venv` doesn't exist, they will run the GPU installation script (`install-gpu.sh` or `install-gpu.ps1`) to set up dependencies automatically.

### Configuration for MCP Clients

Add Expert Among Us to your MCP client configuration. Use **absolute paths** for the command.

#### Example: Claude Desktop (Linux/macOS) + OpenAI

Configuration file location: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "expert-among-us": {
      "command": "/absolute/path/to/expert-among-us/run-mcp.sh",
      "timeout": 120,
      "alwaysAllow":["list","prompt","query"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

#### Example: Claude Desktop (Windows) + AWS profile + debug logs

Configuration file location: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "expert-among-us": {
      "command": "powershell -file C:\\absolute\\path\\to\\expert-among-us\\run-mcp.ps1 --debug",
      "timeout": 120,
      "alwaysAllow":["list","prompt","query"],
      "env": {
        "AWS_PROFILE": "your-profile-here"
      }
    }
  }
}
```

**Important Notes:**
- Always use absolute paths in the `command` field
- Set required environment variables in the `env` section
- Add `--debug` to the command to troubleshoot a problem (writes logs to data-dir)
- Restart your MCP client after updating the configuration

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=expert_among_us --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## License

MIT License - see LICENSE file for details
