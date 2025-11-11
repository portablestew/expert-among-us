"""Query result abstractions for type-safe handling of search results.

This module provides a type-safe abstraction over heterogeneous query results,
replacing the duck-typed SearchResult pattern with proper polymorphism.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal

from .changelist import Changelist
from .file_chunk import FileChunk


# Type aliases for result classification
ResultSource = Literal["metadata", "diff", "file", "combined"]
ResultType = Literal["commit", "file_chunk"]


class QueryResultBase(ABC):
    """Base class for all query result types.
    
    Provides common fields and stable interface for CLI, MCP, and API consumers.
    Implementations are lightweight wrappers around underlying models.
    
    Note: similarity_score, source, and chroma_id are provided by concrete implementations
    as dataclass fields.
    """
    
    @abstractmethod
    def get_id(self) -> str:
        """Stable identifier: commit hash, file:chunk id, etc."""
        pass
    
    @abstractmethod
    def get_result_type(self) -> ResultType:
        """Semantic kind of result for branching logic."""
        pass
    
    @abstractmethod
    def get_display_title(self) -> str:
        """Short human-readable label for tables/headings."""
        pass
    
    @abstractmethod
    def get_preview_text(self, max_len: int = 200) -> str:
        """Short preview body: message or content snippet."""
        pass
    
    @abstractmethod
    def get_timestamp(self) -> Optional[datetime]:
        """Logical timestamp if available (commits), else None."""
        pass
    
    @abstractmethod
    def get_author(self) -> Optional[str]:
        """Author/owner if available (commits), else None."""
        pass


@dataclass
class CommitResult(QueryResultBase):
    """Query result representing a commit/changelist."""
    changelist: Changelist
    similarity_score: float
    source: ResultSource
    chroma_id: Optional[str] = None
    search_similarity_score: Optional[float] = None
    
    def get_id(self) -> str:
        return self.changelist.id
    
    def get_result_type(self) -> ResultType:
        return "commit"
    
    def get_display_title(self) -> str:
        return self.changelist.message
    
    def get_preview_text(self, max_len: int = 200) -> str:
        msg = self.changelist.message or ""
        return (msg[:max_len - 3] + "...") if len(msg) > max_len else msg
    
    def get_timestamp(self) -> Optional[datetime]:
        return self.changelist.timestamp
    
    def get_author(self) -> Optional[str]:
        return self.changelist.author
    
    # Commit-specific accessors for advanced consumers
    
    def get_files(self) -> list[str]:
        """Get list of files affected by this commit."""
        return self.changelist.files
    
    def get_diff(self) -> str:
        """Get full diff content for this commit."""
        return self.changelist.diff


@dataclass
class FileChunkResult(QueryResultBase):
    """Query result representing a file content chunk (current codebase snapshot)."""
    file_chunk: FileChunk
    similarity_score: float
    source: ResultSource
    chroma_id: Optional[str] = None
    search_similarity_score: Optional[float] = None
    
    def get_id(self) -> str:
        # Use ChromaDB ID for uniqueness across file+revision+location
        return self.file_chunk.get_chroma_id()
    
    def get_result_type(self) -> ResultType:
        return "file_chunk"
    
    def get_display_title(self) -> str:
        return f"{self.file_chunk.file_path} (lines {self.file_chunk.line_start}-{self.file_chunk.line_end})"
    
    def get_preview_text(self, max_len: int = 200) -> str:
        content = self.file_chunk.content or ""
        return (content[:max_len - 3] + "...") if len(content) > max_len else content
    
    def get_timestamp(self) -> Optional[datetime]:
        # FileChunk currently has no timestamp
        return None
    
    def get_author(self) -> Optional[str]:
        # FileChunk currently has no author
        return None
    
    # File-specific accessors
    
    def get_file_path(self) -> str:
        """Get file path for this chunk."""
        return self.file_chunk.file_path
    
    def get_line_range(self) -> tuple[int, int]:
        """Get (start_line, end_line) range for this chunk."""
        return (self.file_chunk.line_start, self.file_chunk.line_end)
    
    def get_content(self) -> str:
        """Get full content text for this chunk."""
        return self.file_chunk.content
    
    def get_revision_id(self) -> str:
        """Get revision identifier (commit hash) this chunk was indexed from."""
        return self.file_chunk.revision_id


# Type alias for consumers handling any result type
QueryResult = CommitResult | FileChunkResult