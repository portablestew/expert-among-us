"""Query parameter models for searching and querying experts."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryParams(BaseModel):
    """Parameters for searching/querying an expert.
    
    This model defines all parameters for performing searches against an expert's
    indexed commit history, including filters, limits, and special modes.
    
    Attributes:
        prompt: User's query string
        max_changes: Maximum results to return (1-100)
        users: Optional filter by commit authors
        files: Optional filter by file paths (OR logic)
        amogus: Among Us mode for impostor responses
        temperature: LLM temperature for generation (0.0-1.0)
    """

    prompt: str = Field(..., description="User's query")
    max_changes: int = Field(
        default=10, ge=1, le=100, description="Max changelist results to return"
    )
    max_file_chunks: int = Field(
        default=10, ge=1, le=100, description="Max file chunk results to return"
    )
    users: Optional[List[str]] = Field(
        None, description="Filter by authors (AND logic)"
    )
    files: Optional[List[str]] = Field(
        None, description="Filter by file paths (OR logic)"
    )
    amogus: bool = Field(
        default=False, description="Among Us mode for impostor"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="LLM temperature"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "What changes are needed to add a new arrow type?",
                "max_changes": 10,
                "users": ["portablestew"],
                "files": ["src/Arrow.java", "src/ArrowRegistry.java"],
            }
        }
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate that prompt is non-empty.
        
        Args:
            v: Prompt to validate
            
        Returns:
            Stripped prompt string
            
        Raises:
            ValueError: If prompt is empty
        """
        if not v or len(v.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class VectorSearchResult(BaseModel):
    """Result from vector similarity search.
    
    This model represents search results from any vector collection (commits or files).
    
    Semantic Note: The 'result_id' field serves dual purposes:
    - For commit/diff results: Contains the changelist ID (commit hash)
    - For file results: Contains the file path (used as grouping key)
    
    The 'chroma_id' field provides unique chunk-level identification for all result types.
    
    Examples:
        Commit result:
            VectorSearchResult(
                result_id="abc123def456",  # commit hash
                similarity_score=0.87,
                source="metadata",
                chroma_id="abc123def456"
            )
        
        File chunk result:
            VectorSearchResult(
                result_id="src/utils/searcher.py",  # file path
                similarity_score=0.92,
                source="file",
                chroma_id="file:src/utils/searcher.py:chunk_0"
            )

    Attributes:
        result_id: Identifier for the result (changelist ID or file path)
        similarity_score: Cosine similarity score (0.0-1.0)
        source: Which collection matched ('metadata', 'diff', or 'file')
        chroma_id: Optional ChromaDB ID (for chunk-level identification)
    """

    result_id: str = Field(
        ...,
        description="Identifier: changelist ID for commits, file path for files"
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity (0-1)"
    )
    source: Literal["metadata", "diff", "file"] = Field(
        default="metadata", description="Which collection matched"
    )
    chroma_id: Optional[str] = Field(
        default=None,
        description="ChromaDB ID for chunk-level identification"
    )

    # Helper methods for type-safe access
    def is_commit_result(self) -> bool:
        """Check if this is a commit/diff result.
        
        Returns:
            True if this result is from metadata or diff collection
        """
        return self.source in ("metadata", "diff")

    def is_file_result(self) -> bool:
        """Check if this is a file chunk result.
        
        Returns:
            True if this result is from file collection
        """
        return self.source == "file"

    def get_commit_id(self) -> str:
        """Get commit ID (only valid for commit results).
        
        Returns:
            The changelist ID for commit/diff results
            
        Raises:
            ValueError: If called on a file result
        """
        if not self.is_commit_result():
            raise ValueError(f"Cannot get commit_id from {self.source} result")
        return self.result_id

    def get_file_path(self) -> str:
        """Get file path (only valid for file results).
        
        Returns:
            The file path for file results
            
        Raises:
            ValueError: If called on a commit result
        """
        if not self.is_file_result():
            raise ValueError(f"Cannot get file_path from {self.source} result")
        return self.result_id


    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "changelist_id": "abc123def456",
                "similarity_score": 0.87,
                "source": "metadata",
            }
        }
    )

    def __lt__(self, other: "VectorSearchResult") -> bool:
        """Enable sorting by similarity score (descending).
        
        Args:
            other: Other result to compare with
            
        Returns:
            True if this result has higher similarity score
        """
        return self.similarity_score > other.similarity_score

    def __eq__(self, other: object) -> bool:
        """Check equality based on changelist_id and source.
        
        Args:
            other: Other result to compare with
            
        Returns:
            True if changelist_id and source match
        """
        if not isinstance(other, VectorSearchResult):
            return NotImplemented
        return (
            self.result_id == other.result_id
            and self.source == other.source
        )

    def __hash__(self) -> int:
        """Hash based on result_id and source.

        Returns:
            Hash value for use in sets/dicts
        """
        return hash((self.result_id, self.source))