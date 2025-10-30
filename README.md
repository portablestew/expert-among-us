# expert-among-us

MCP which indexes change history, then uses secondary inference to form a queryable "expert"

## Why?

SDEs with practitioner-level knowledge of sprawling legacy codebases have instincts and thought processes not well represented in the final code structure. High-level documentation also fails to fully capture their experience (if such documentation even exists). A software expert's experience and instincts are often implied in their raw commit messages and code review comment discussions.

Transformer LLMs are excellent at filling in the blank, even when the "blank" is a complex analysis topic. By presenting an ordered chain of sample prompt->response entries representing a human expert's knowledge, then a final prompt->[?] entry, an LLM is placed in position to impersonate the human expert in its response. In effect, the assistant becomes a trained impostor.

## Overview

Expert Among Us creates a queryable "expert" from your repository's commit history using AI-powered semantic search and vector embeddings. It helps you understand development patterns, find relevant changes, and get AI-powered recommendations based on historical code changes.

### Key Capabilities

- **Semantic Search**: Find commits by meaning, not just keywords, using vector embeddings
- **Metadata Extraction**: Index commit messages, authors, files, and code diffs
- **Vector Embeddings**: Local (Jina Code) or cloud (AWS Bedrock Titan) embedding models
- **Flexible Filtering**: Search by author, files, or time period
- **Multiple Storage**: Combines ChromaDB for vectors and SQLite for metadata
- **Version Control Support**: Works with Git repositories (Perforce support planned)

### Embedding Providers

Expert Among Us supports two embedding providers:

#### Local Embeddings (Default)
- **Model**: Jina Code Embeddings v0.5b
- **Dimension**: 512 (Matryoshka truncation)
- **Task**: `code2code` - optimized for code similarity
- **Advantages**:
  - No API costs or rate limits
  - Works offline after initial model download
  - Fast inference on CPU
  - Specialized for code embeddings
- **First Run**: Downloads model (~1.2GB) automatically
- **Requirements**: No AWS credentials needed

#### AWS Bedrock (Cloud)
- **Model**: Amazon Titan Embed Text v2
- **Dimension**: 1024
- **Advantages**:
  - Managed service with high availability
  - No local model storage needed
  - Consistent results across environments
- **Requirements**: AWS credentials and Bedrock access

**Switching Providers**: Use the `--embedding-provider` flag with `local` or `bedrock`:

```bash
# Use local embeddings (default)
./run.sh populate /path/to/repo MyExpert

# Use AWS Bedrock embeddings
./run.sh populate /path/to/repo MyExpert --embedding-provider bedrock
```

**Note**: You must use the same embedding provider for both indexing (`populate`) and querying (`query`/`prompt`), as different providers produce incompatible embeddings.

## Requirements

- **Python**: 3.12 or higher
- **Git**: For repository access
- **uv**: Package manager (>=0.1)

### For Local Embeddings (Default)
- **No additional setup required** - model downloads automatically on first run
- **Storage**: ~1.2GB for Jina Code model
- **Dependencies**: PyTorch and sentence-transformers (installed automatically)

### For AWS Bedrock Embeddings (Optional)
- **AWS Credentials**: Required for Bedrock API access

Export your AWS credentials as environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1  # or your preferred region
```

Or configure AWS CLI with:
```bash
aws configure
```

## Installation

### Quick Install (Recommended)

Use the provided installation scripts to automate the setup process:

#### CPU-Only Installation

**Linux/macOS:**
```bash
chmod +x install-cpu.sh
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
chmod +x install-gpu.sh
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

**Important Usage Note**: After GPU setup, you can use the run scripts which automatically use the virtual environment:

**Linux/macOS:**
```bash
./run.sh populate /path/to/repo MyExpert
```

**Windows:**
```powershell
.\run.ps1 populate /path/to/repo MyExpert
```

**Why not use `uv run`?** The `uv run` command resyncs the environment to the lock file, which would revert to CPU-only PyTorch. To preserve GPU PyTorch, use the run scripts which automatically use the venv without resyncing.

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
    --max-changes 15

# Save results to JSON
./run.sh query /path/to/repo MyExpert "API endpoint implementation" \
    --output results.json
```

**Important**: Use the same `--embedding-provider` for querying as you used during indexing.

### 3. Get AI Recommendations

Get AI-powered recommendations that impersonate the expert based on their historical commit patterns:

```bash
# Get recommendations based on expert's patterns (must specify provider)
./run.sh prompt /path/to/repo MyExpert "How should I implement authentication?" \
    --llm-provider openai

# With filters for specific context
./run.sh prompt /path/to/repo MyExpert "How to handle errors?" \
    --llm-provider openrouter \
    --users alice,bob \
    --files src/handlers/

# With Among Us mode (occasionally gives intentionally incorrect advice)
./run.sh prompt /path/to/repo MyExpert "Add caching" \
    --llm-provider local \
    --amogus

# With debug logging to inspect API calls
./run.sh prompt /path/to/repo MyExpert "Optimize queries" \
    --llm-provider openai \
    --debug
```

**How It Works:**
1. Searches for relevant commits using semantic similarity
2. Generates conversational prompts from historical diffs
3. Builds a conversation showing the expert's past work
4. Streams an AI response impersonating the expert's style

**Important:** The `--llm-provider` argument is **required**. If you don't specify it, or if the required environment variables for the chosen provider are missing, you'll receive an error message.

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

# Query with custom data directory
./run.sh --data-dir /mnt/data/experts query ~/projects/myapp "AppExpert" "feature implementation"
```

### `prompt` - AI Recommendations

Get AI-powered recommendations that impersonate the expert based on their historical commit patterns. The system uses relevant past commits as examples to generate responses that match the expert's coding style and approach.

```bash
./run.sh prompt WORKSPACE EXPERT_NAME PROMPT [OPTIONS]
```

**Arguments:**
- `WORKSPACE`: Path to the repository root directory
- `EXPERT_NAME`: Name of the expert to query
- `PROMPT`: Question or task description for the AI

**Options:**
- `--llm-provider [openai|openrouter|local|bedrock|claude-code]`: **(REQUIRED)** LLM provider to use for generating recommendations
- `--max-changes INTEGER`: Maximum context changes to use (default: 10)
- `--users TEXT`: Filter by commit authors (comma-separated)
- `--files TEXT`: Filter by file paths (comma-separated)
- `--amogus`: Enable Among Us mode (task is performed as a crewmate)
- `--debug`: Enable debug logging of API calls
- `--temperature FLOAT`: LLM temperature for generation (0.0-1.0, default: 0.7)
- `--embedding-provider [local|bedrock]`: Embedding provider - must match what was used during indexing (default: local)
- `--data-dir PATH`: Base directory for expert data storage (default: ~/.expert-among-us)

**LLM Provider Selection:**
The `--llm-provider` argument is **required** for all prompt commands. Each provider requires specific environment variables:
- `openai`: Requires `OPENAI_API_KEY`
- `openrouter`: Requires `OPENROUTER_API_KEY`
- `local`: Requires `LOCAL_LLM_BASE_URL`
- `bedrock`: Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
- `claude-code`: Requires Claude Code CLI to be installed

If the required environment variables are missing, the command will fail with a clear error message.

**Debug Logging:**
When `--debug` is enabled, all API requests and responses are logged to JSON files in `~/.expert-among-us/logs/` for troubleshooting and analysis.

**Examples:**
```bash
# Basic usage with OpenAI
./run.sh prompt ~/projects/myapp "AppExpert" "How to implement caching?" \
    --llm-provider openai

# With OpenRouter and filters
./run.sh prompt ~/projects/myapp "AppExpert" "Optimize database queries" \
    --llm-provider openrouter \
    --users alice,bob \
    --files src/db/

# Using local LLM with debug mode
./run.sh prompt ~/projects/myapp "AppExpert" "Add error handling" \
    --llm-provider local \
    --debug

# With Among Us mode for training
./run.sh prompt ~/projects/myapp "AppExpert" "Implement authentication" \
    --llm-provider openai \
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

**Local (Default):**
- **Model**: `jinaai/jina-code-embeddings-0.5b`
- **Dimension**: 512 (Matryoshka truncation from 896)
- **Max tokens**: 32,768
- **Download size**: ~1.2GB (one-time)

**AWS Bedrock:**
- **Model**: `amazon.titan-embed-text-v2:0`
- **Dimension**: 1024
- **Max tokens**: 8,000

### LLM Providers

Expert Among Us supports multiple LLM providers for prompt generation and recommendations. **You must explicitly specify which provider to use** via the `--llm-provider` CLI argument when running the `prompt` command.

#### Provider Selection

The `--llm-provider` argument is **required** to specify which LLM provider to use:

```bash
./run.sh prompt /path/to/repo MyExpert "your question" --llm-provider [provider]
```

**Available providers:**
- `openai` - OpenAI GPT models
- `openrouter` - OpenRouter (access multiple LLM providers)
- `local` - Local OpenAI-compatible LLM server
- `bedrock` - AWS Bedrock managed LLMs
- `claude-code` - Anthropic Claude Code CLI interface

**Note:** Each provider requires specific environment variables to be set. If the required credentials are missing, the command will fail with a clear error message indicating which environment variable needs to be configured.

#### OpenAI Provider

Use OpenAI's GPT models for AI recommendations:

**Usage:**
```bash
./run.sh prompt /path/to/repo MyExpert "your question" --llm-provider openai
```

**Required Environment Variables:**
- `OPENAI_API_KEY`: (required) Your OpenAI API key

**Optional Environment Variables:**
- `OPENAI_BASE_URL`: Custom base URL (default: `https://api.openai.com/v1`)
- `OPENAI_MODEL`: Model to use (default: `gpt-4`)

**Supported Models:**
- `gpt-4` - Most capable model
- `gpt-4-turbo` - Faster GPT-4 variant
- `gpt-3.5-turbo` - Fast and cost-effective

**Example Configuration:**
```bash
# Basic OpenAI setup
export OPENAI_API_KEY=sk-proj-...
export OPENAI_MODEL=gpt-4-turbo

./run.sh prompt ~/projects/myapp "AppExpert" "How to implement auth?" --llm-provider openai

# With custom base URL (for proxies)
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://your-proxy.com/v1

./run.sh prompt ~/projects/myapp "AppExpert" "How to implement auth?" --llm-provider openai
```

**Error Handling:**
If `OPENAI_API_KEY` is not set, you'll receive a fatal error:
```
Error: OPENAI_API_KEY environment variable is required for OpenAI provider
```

#### OpenRouter Provider

Use OpenRouter to access multiple LLM providers through a single API:

**Usage:**
```bash
./run.sh prompt /path/to/repo MyExpert "your question" --llm-provider openrouter
```

**Required Environment Variables:**
- `OPENROUTER_API_KEY`: (required) Your OpenRouter API key

**Optional Environment Variables:**
- `OPENROUTER_MODEL`: Model to use (default: `anthropic/claude-3-opus`)
- `OPENROUTER_APP_NAME`: Application name for OpenRouter tracking
- `OPENROUTER_SITE_URL`: Site URL for OpenRouter tracking

**Base URL:** `https://openrouter.ai/api/v1`

**Supported Models (examples):**
- `anthropic/claude-3-opus` - Claude's most capable model
- `anthropic/claude-3-sonnet` - Balanced performance and speed
- `openai/gpt-4` - OpenAI GPT-4 via OpenRouter
- `openai/gpt-4-turbo` - Faster GPT-4 variant
- `google/gemini-pro` - Google's Gemini model
- Many more available at [openrouter.ai/models](https://openrouter.ai/models)

**Example Configuration:**
```bash
# Basic OpenRouter setup
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_MODEL=anthropic/claude-3-opus

./run.sh prompt ~/projects/myapp "AppExpert" "Optimize database queries" --llm-provider openrouter

# With tracking metadata
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_MODEL=openai/gpt-4-turbo
export OPENROUTER_APP_NAME="Expert Among Us"
export OPENROUTER_SITE_URL="https://github.com/yourorg/expert-among-us"

./run.sh prompt ~/projects/myapp "AppExpert" "Optimize database queries" --llm-provider openrouter
```

**Error Handling:**
If `OPENROUTER_API_KEY` is not set, you'll receive a fatal error:
```
Error: OPENROUTER_API_KEY environment variable is required for OpenRouter provider
```

#### Local LLM Server

Use any OpenAI-compatible local LLM server:

**Usage:**
```bash
./run.sh prompt /path/to/repo MyExpert "your question" --llm-provider local
```

**Required Environment Variables:**
- `LOCAL_LLM_BASE_URL`: (required) Base URL of your local LLM server

**Optional Environment Variables:**
- `LOCAL_LLM_MODEL`: Model to use (default: `llama3`)

**No API Key Required** - Local servers typically don't require authentication.

**Supported Local Servers:**
- **Ollama**: `http://localhost:11434/v1`
- **llama.cpp**: `http://localhost:8080/v1`
- **LocalAI**: `http://localhost:8080/v1`
- **LM Studio**: `http://localhost:1234/v1`
- **text-generation-webui**: `http://localhost:5000/v1`

**Example Configurations:**

```bash
# Ollama (default port 11434)
export LOCAL_LLM_BASE_URL=http://localhost:11434/v1
export LOCAL_LLM_MODEL=llama3.1

./run.sh prompt ~/projects/myapp "AppExpert" "Add caching layer" --llm-provider local

# llama.cpp server (default port 8080)
export LOCAL_LLM_BASE_URL=http://localhost:8080/v1
export LOCAL_LLM_MODEL=llama-3-8b

./run.sh prompt ~/projects/myapp "AppExpert" "Add caching layer" --llm-provider local

# LM Studio (default port 1234)
export LOCAL_LLM_BASE_URL=http://localhost:1234/v1
export LOCAL_LLM_MODEL=llama-3-8b-instruct

./run.sh prompt ~/projects/myapp "AppExpert" "Add caching layer" --llm-provider local

# Custom local server
export LOCAL_LLM_BASE_URL=http://192.168.1.100:8000/v1
export LOCAL_LLM_MODEL=mixtral-8x7b

./run.sh prompt ~/projects/myapp "AppExpert" "Add caching layer" --llm-provider local
```

**Error Handling:**
If `LOCAL_LLM_BASE_URL` is not set, you'll receive a fatal error:
```
Error: LOCAL_LLM_BASE_URL environment variable is required for local provider
```

#### AWS Bedrock Provider

Use AWS Bedrock's managed LLM services:

**Usage:**
```bash
./run.sh prompt /path/to/repo MyExpert "your question" --llm-provider bedrock
```

**Required Environment Variables:**
AWS credentials are required (configured via AWS CLI or environment variables):
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_REGION`: AWS region (e.g., `us-east-1`)

**Models:**
- **Prompt Generation**: `us.amazon.nova-lite-v1:0`
- **Impostor Mode**: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`

**Example Configuration:**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

./run.sh prompt ~/projects/myapp "AppExpert" "Implement retry logic" --llm-provider bedrock
```

**Requirements:** AWS credentials and Bedrock access (see Requirements section above)

**Error Handling:**
If AWS credentials are not configured, you'll receive a fatal error indicating missing AWS configuration.

#### Claude Code CLI

Use Anthropic's Claude Code CLI interface:

**Usage:**
```bash
./run.sh prompt /path/to/repo MyExpert "your question" --llm-provider claude-code
```

**Requirements:**
- Claude Code CLI must be installed and configured separately
- The `claude` command must be available in your system PATH

**Note:** This provider is primarily for users who already have the Claude Code CLI set up. Most users should prefer the OpenAI, OpenRouter, or Local LLM providers.

#### Complete Configuration Examples

**Example 1: OpenAI with GPT-4**
```bash
# .env file
OPENAI_API_KEY=sk-proj-abc123...
OPENAI_MODEL=gpt-4-turbo

# Usage
./run.sh prompt ~/projects/myapp "AppExpert" "How to implement auth?" --llm-provider openai
```

**Example 2: OpenRouter with Claude**
```bash
# .env file
OPENROUTER_API_KEY=sk-or-v1-xyz789...
OPENROUTER_MODEL=anthropic/claude-3-opus
OPENROUTER_APP_NAME=Expert Among Us
OPENROUTER_SITE_URL=https://github.com/myorg/myproject

# Usage
./run.sh prompt ~/projects/myapp "AppExpert" "How to implement auth?" --llm-provider openrouter
```

**Example 3: Local Ollama Server**
```bash
# .env file
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=llama3.1

# Usage
./run.sh prompt ~/projects/myapp "AppExpert" "How to implement auth?" --llm-provider local
```

**Example 4: AWS Bedrock**
```bash
# .env file
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=secret...
AWS_REGION=us-east-1

# Usage
./run.sh prompt ~/projects/myapp "AppExpert" "How to implement auth?" --llm-provider bedrock
```

**Important Notes:**
- The `--llm-provider` argument is **required** for all `prompt` commands
- Each provider requires its specific environment variables to be set
- There is no automatic fallback between providers - you must explicitly choose which provider to use
- If required environment variables are missing, the command will fail immediately with a clear error message

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

Expert Among Us uses a hybrid approach:

1. **Vector Search**: Semantic similarity using AWS Bedrock embeddings and ChromaDB
2. **Metadata Filtering**: Fast filtering by author, files, and timestamps using SQLite
3. **Incremental Updates**: Only new commits are processed on subsequent runs
4. **Diff Processing**: Code diffs are embedded for semantic code change search

## MCP Integration (Planned)

Expert Among Us can be used as an MCP (Model Context Protocol) server, allowing AI assistants to query your codebase history:

```python
# MCP tools available:
- expert-among-us/populate
- expert-among-us/query  
- expert-among-us/prompt
```

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
