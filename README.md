# expert-among-us

MCP which indexes change history, then uses secondary inference to form a queryable "expert"

## Why Expert Among Us?

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
python -m expert_among_us populate /path/to/repo MyExpert

# Use AWS Bedrock embeddings
python -m expert_among_us populate /path/to/repo MyExpert --embedding-provider bedrock
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

### Basic Installation

1. Install uv package manager:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install project dependencies:
   ```bash
   uv sync
   ```

**Note**: Requires Python 3.12+ and uv >=0.1. The `uv sync` command automatically creates and manages the `.venv` directory.

### GPU Acceleration Setup (Windows with NVIDIA GPU)

For significantly faster local embeddings (10-20x speedup), install GPU-enabled PyTorch:

```bash
# 1. After uv sync, activate the virtual environment
.venv\Scripts\activate

# 2. Install GPU-enabled PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
nvidia-smi  # Verify NVIDIA drivers are working
```

**Performance Impact:**
- **With GPU**: ~0.5s per commit embedding
- **CPU only**: ~4s per commit embedding

**Important Usage Note**: After GPU setup, you must **activate the virtual environment** before running commands:

```bash
# Activate venv (do this each time you open a new terminal)
.venv\Scripts\activate

# Then run commands directly (NOT with uv run)
python -m expert_among_us populate /path/to/repo MyExpert
python -m expert_among_us query /path/to/repo MyExpert "search query"
python -m expert_among_us prompt /path/to/repo MyExpert "question"
```

**Why not use `uv run`?** The `uv run` command resyncs the environment to the lock file, which would revert to CPU-only PyTorch. To preserve GPU PyTorch, always activate the venv and use `python -m expert_among_us` instead.

**For CPU-only systems** or if you prefer simpler workflow (without GPU), you can use `uv run` throughout the documentation - it will work but be slower.

## Quick Start

### 1. Index a Repository

Create an expert index from your git repository:

```bash
# Index entire repository (uses local embeddings by default)
python -m expert_among_us populate /path/to/repo MyExpert

# Use AWS Bedrock embeddings instead
python -m expert_among_us populate /path/to/repo MyExpert --embedding-provider bedrock

# Index specific subdirectories only
python -m expert_among_us populate /path/to/repo MyExpert src/main/ src/resources/

# Limit the number of commits to index
python -m expert_among_us populate /path/to/repo MyExpert --max-commits 5000
```

**Note**: On first run with local embeddings, the Jina Code model (~1.2GB) will be downloaded automatically. This is a one-time download.

The first indexing will take some time depending on repository size. Subsequent runs are incremental and only process new commits.

### 2. Search for Similar Changes

Find commits similar to your query:

```bash
# Basic search (uses local embeddings by default)
python -m expert_among_us query /path/to/repo MyExpert "How to add a new feature?"

# Use same embedding provider as during indexing
python -m expert_among_us query /path/to/repo MyExpert "How to add a new feature?" \
    --embedding-provider bedrock

# Search with filters
python -m expert_among_us query /path/to/repo MyExpert "Bug fix for memory leak" \
    --users john,jane \
    --files src/main.py,src/utils.py \
    --max-changes 15

# Save results to JSON
python -m expert_among_us query /path/to/repo MyExpert "API endpoint implementation" \
    --output results.json
```

**Important**: Use the same `--embedding-provider` for querying as you used during indexing.

### 3. Get AI Recommendations

Get AI-powered recommendations that impersonate the expert based on their historical commit patterns:

```bash
# Get recommendations based on expert's patterns
python -m expert_among_us prompt /path/to/repo MyExpert "How should I implement authentication?"

# With filters for specific context
python -m expert_among_us prompt /path/to/repo MyExpert "How to handle errors?" \
    --users alice,bob \
    --files src/handlers/

# With Among Us mode (occasionally gives intentionally incorrect advice)
python -m expert_among_us prompt /path/to/repo MyExpert "Add caching" --amogus

# With debug logging to inspect API calls
python -m expert_among_us prompt /path/to/repo MyExpert "Optimize queries" --debug
```

**How It Works:**
1. Searches for relevant commits using semantic similarity
2. Generates conversational prompts from historical diffs
3. Builds a conversation showing the expert's past work
4. Streams an AI response impersonating the expert's style

## CLI Command Reference

### `populate` - Index Repository

Create or update an expert index from a repository.

```bash
python -m expert_among_us populate WORKSPACE EXPERT_NAME [SUBDIRS...] [OPTIONS]
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
python -m expert_among_us populate ~/projects/myapp "AppExpert"

# Index with AWS Bedrock embeddings
python -m expert_among_us populate ~/projects/myapp "AppExpert" --embedding-provider bedrock

# Index only backend code
python -m expert_among_us populate ~/projects/myapp "BackendExpert" src/backend/ src/api/

# Limit indexing to recent commits
python -m expert_among_us populate ~/projects/myapp "RecentExpert" --max-commits 1000

# Use custom data directory
python -m expert_among_us --data-dir /mnt/data/experts populate ~/projects/myapp "AppExpert"
```

### `query` - Search History

Search for commits similar to your query using semantic search.

```bash
python -m expert_among_us query WORKSPACE EXPERT_NAME PROMPT [OPTIONS]
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
python -m expert_among_us query ~/projects/myapp "AppExpert" "authentication implementation"

# Search with author filter
python -m expert_among_us query ~/projects/myapp "AppExpert" "database optimization" \
    --users alice,bob

# Search specific files and save results
python -m expert_among_us query ~/projects/myapp "AppExpert" "error handling" \
    --files src/handlers/ \
    --output search-results.json \
    --max-changes 20

# Query with custom data directory
python -m expert_among_us --data-dir /mnt/data/experts query ~/projects/myapp "AppExpert" "feature implementation"
```

### `prompt` - AI Recommendations

Get AI-powered recommendations that impersonate the expert based on their historical commit patterns. The system uses relevant past commits as examples to generate responses that match the expert's coding style and approach.

```bash
python -m expert_among_us prompt WORKSPACE EXPERT_NAME PROMPT [OPTIONS]
```

**Arguments:**
- `WORKSPACE`: Path to the repository root directory
- `EXPERT_NAME`: Name of the expert to query
- `PROMPT`: Question or task description for the AI

**Options:**
- `--max-changes INTEGER`: Maximum context changes to use (default: 10)
- `--users TEXT`: Filter by commit authors (comma-separated)
- `--files TEXT`: Filter by file paths (comma-separated)
- `--amogus`: Enable Among Us mode (occasionally gives subtly incorrect advice)
- `--debug`: Enable debug logging of API calls
- `--temperature FLOAT`: LLM temperature for generation (0.0-1.0, default: 0.7)
- `--embedding-provider [local|bedrock]`: Embedding provider - must match what was used during indexing (default: local)
- `--data-dir PATH`: Base directory for expert data storage (default: ~/.expert-among-us)

**Among Us Mode:**
When `--amogus` is enabled, the AI will occasionally (about 20% of the time) give subtly incorrect advice that sounds plausible but contains bugs or anti-patterns. This mode is inspired by the "Among Us" game and is useful for:
- Training developers to catch mistakes
- Testing code review skills
- Adding unpredictability to recommendations

**Debug Logging:**
When `--debug` is enabled, all API requests and responses are logged to JSON files in `~/.expert-among-us/logs/` for troubleshooting and analysis.

## Configuration

### Storage Location

By default, expert indexes are stored in: `~/.expert-among-us/data/`

You can customize the storage location using the `--data-dir` global option:

```bash
# Use custom data directory
python -m expert_among_us --data-dir /mnt/data/experts populate /path/to/repo MyExpert

# Query from custom location (must match where data was indexed)
python -m expert_among_us --data-dir /mnt/data/experts query /path/to/repo MyExpert "search query"
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

### LLM Models (AWS Bedrock)

For prompt generation and recommendations:
- **Prompt Generation**: `us.amazon.nova-lite-v1:0`
- **Impostor Mode**: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`

### Limits and Defaults

- **Max commits**: 10,000 (configurable with `--max-commits`)
- **Max diff size**: 100KB (larger diffs are truncated)
- **Max embedding text**: 30KB
- **AWS Region**: us-east-1 (configurable via `AWS_REGION` environment variable)

### Filtering Options

When indexing with subdirectories:
```bash
python -m expert_among_us populate /path/to/repo MyExpert src/main/ src/resources/
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
