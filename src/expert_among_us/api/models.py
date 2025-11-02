"""Request and response models for Expert Among Us API."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ExpertInfo:
    """Information about an expert.
    
    Attributes:
        name: Unique name of the expert
        vcs_type: Version control system type (e.g., "git")
        workspace_path: Path to the repository workspace
        subdirs: List of subdirectories being indexed (empty for all)
        commit_count: Number of commits indexed
        last_indexed_at: Timestamp of last indexing operation
        first_commit_time: Timestamp of oldest indexed commit
        last_commit_time: Timestamp of newest indexed commit
    """
    name: str
    vcs_type: str
    workspace_path: str
    subdirs: List[str]
    commit_count: int
    last_indexed_at: Optional[datetime]
    first_commit_time: Optional[datetime]
    last_commit_time: Optional[datetime]


@dataclass
class PromptResponse:
    """Complete response from prompt generation.
    
    Attributes:
        content: Full text content of the response
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/response
        total_tokens: Total tokens used (input + output)
        cache_read_tokens: Number of tokens read from cache
        cache_creation_tokens: Number of tokens written to cache
    """
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class QueryRequest:
    """Request parameters for querying an expert.
    
    Attributes:
        expert_name: Name of the expert to query
        prompt: Search query text
        max_changes: Maximum number of results to return
        users: Optional list of authors to filter by
        files: Optional list of file paths to filter by
        search_scope: Search scope ("both", "metadata", or "diffs")
        min_score: Minimum similarity score threshold (0.0-1.0)
        relative_threshold: Relative threshold from top score (0.0-1.0)
        data_dir: Optional custom data directory
        embedding_provider: Embedding provider to use
    """
    expert_name: str
    prompt: str
    max_changes: int = 15
    users: Optional[List[str]] = None
    files: Optional[List[str]] = None
    search_scope: str = "both"
    min_score: float = 0.1
    relative_threshold: float = 0.3
    data_dir: Optional[Path] = None
    embedding_provider: str = "local"


@dataclass
class PromptRequest:
    """Request parameters for generating AI recommendations.
    
    Attributes:
        expert_name: Name of the expert to query
        prompt: User's question or task description
        max_changes: Maximum context changes to use
        users: Optional list of authors to filter by
        files: Optional list of file paths to filter by
        amogus: Enable Among Us mode (occasionally bad advice)
        impostor: Use impostor mode (generate prompts for examples)
        temperature: LLM temperature (0.0-1.0)
        data_dir: Optional custom data directory
        embedding_provider: Embedding provider to use
        llm_provider: LLM provider to use
    """
    expert_name: str
    prompt: str
    max_changes: int = 15
    users: Optional[List[str]] = None
    files: Optional[List[str]] = None
    amogus: bool = False
    impostor: bool = False
    temperature: float = 0.7
    data_dir: Optional[Path] = None
    embedding_provider: str = "local"
    llm_provider: str = "auto"


@dataclass
class ImportRequest:
    """Request parameters for importing an expert.
    
    Attributes:
        source_path: Path to the expert directory to import
        data_dir: Optional custom data directory
    """
    source_path: Path
    data_dir: Optional[Path] = None