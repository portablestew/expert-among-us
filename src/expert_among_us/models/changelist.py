"""Changelist data model representing a single commit/changelist with metadata and embeddings."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config.settings import TITAN_EMBEDDING_DIMENSION


class Changelist(BaseModel):
    """Represents a single commit/changelist with all metadata and embeddings.
    
    This model stores commit information including the diff, metadata, and optional
    embeddings for similarity search. It supports both metadata embeddings (commit
    message + files + review comments) and diff embeddings (actual code changes).
    
    Attributes:
        id: Commit hash or CL number (primary key)
        expert_name: Which expert this belongs to
        timestamp: When the commit was made
        author: Commit author
        message: Commit message
        diff: Full diff (stored up to max_diff_size)
        files: List of affected file paths
        review_comments: Optional code review comments
        metadata_embedding: Optional embedding of message+files+comments (1024D)
        diff_embedding: Optional embedding of code changes (1024D)
        generated_prompt: Optional cached LLM-generated prompt
    """

    id: str = Field(..., description="Commit hash or CL number (primary key)")
    expert_name: str = Field(..., description="Expert this changelist belongs to")
    timestamp: datetime = Field(..., description="When the commit was made")
    author: str = Field(..., description="Commit author")
    message: str = Field(..., description="Commit message")
    diff: str = Field(..., description="Full diff content")
    files: List[str] = Field(..., description="List of affected file paths")
    review_comments: Optional[str] = Field(
        None, description="Code review comments (if available)"
    )
    metadata_embedding: Optional[List[float]] = Field(
        None, description=f"Embedding of message+files+comments ({TITAN_EMBEDDING_DIMENSION}D)"
    )
    diff_embedding: Optional[List[float]] = Field(
        None, description=f"Embedding of code changes ({TITAN_EMBEDDING_DIMENSION}D)"
    )
    generated_prompt: Optional[str] = Field(
        None, description="Cached LLM-generated prompt"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "abc123def456",
                "expert_name": "MyExpert",
                "timestamp": "2024-01-15T10:30:00Z",
                "author": "portablestew",
                "message": "Add new arrow type support",
                "diff": "diff --git a/Arrow.java...",
                "files": ["src/Arrow.java", "src/ArrowRegistry.java"],
                "review_comments": "LGTM, nice implementation",
            }
        }
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate that commit ID is non-empty."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Commit ID cannot be empty")
        return v.strip()

    @field_validator("message", "diff", "author")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Validate that required string fields are non-empty."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("files")
    @classmethod
    def validate_files(cls, v: List[str]) -> List[str]:
        """Validate that files list is non-empty."""
        if not v or len(v) == 0:
            raise ValueError("Files list cannot be empty")
        return v

    @field_validator("metadata_embedding", "diff_embedding")
    @classmethod
    def validate_embedding_dimension(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate that embeddings, if present, match the expected dimension."""
        if v is not None and len(v) != TITAN_EMBEDDING_DIMENSION:
            raise ValueError(f"Embedding must be {TITAN_EMBEDDING_DIMENSION} dimensions, got {len(v)}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage.
        
        Returns:
            Dictionary representation with all fields
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Changelist":
        """Create from database row.
        
        Args:
            data: Dictionary with changelist data
            
        Returns:
            Changelist instance
        """
        return cls.model_validate(data)

    def get_metadata_text(self) -> str:
        """Combine message + files + comments for embedding.
        
        This creates the text that will be used for metadata embeddings,
        combining the commit message, file list, and review comments into
        a single string suitable for embedding generation.
        
        Returns:
            Combined text string for metadata embedding
        """
        parts = [
            f"Commit Message: {self.message}",
            f"Files: {', '.join(self.files)}",
        ]
        
        if self.review_comments:
            parts.append(f"Review Comments: {self.review_comments}")
        
        return "\n".join(parts)

    def get_truncated_diff(self, max_size: int = 100000) -> str:
        """Return diff truncated to max size.
        
        Args:
            max_size: Maximum size in bytes (default 100KB)
            
        Returns:
            Diff content, truncated if necessary
        """
        if len(self.diff.encode("utf-8")) <= max_size:
            return self.diff
        
        # Truncate to max_size bytes
        encoded = self.diff.encode("utf-8")[:max_size]
        
        # Decode, ignoring any incomplete characters at the end
        truncated = encoded.decode("utf-8", errors="ignore")
        
        return truncated + "\n\n[TRUNCATED - content exceeded max size]"