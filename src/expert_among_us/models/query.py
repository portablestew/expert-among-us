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
        default=10, ge=1, le=100, description="Max results to return"
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
    
    Represents a single match from vector database search, including the
    changelist ID, similarity score, and which collection matched.
    
    Attributes:
        changelist_id: ID of matching changelist
        similarity_score: Cosine similarity score (0.0-1.0)
        source: Which collection matched ('metadata' or 'diff')
        chroma_id: Optional ChromaDB ID (for debugging chunk-level matching)
    """

    changelist_id: str = Field(..., description="ID of matching changelist")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity (0-1)"
    )
    source: Literal["metadata", "diff"] = Field(
        default="metadata", description="Which collection matched"
    )
    chroma_id: Optional[str] = Field(
        default=None, description="ChromaDB ID (for debugging chunked diffs)"
    )

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
            self.changelist_id == other.changelist_id
            and self.source == other.source
        )

    def __hash__(self) -> int:
        """Hash based on changelist_id and source.
        
        Returns:
            Hash value for use in sets/dicts
        """
        return hash((self.changelist_id, self.source))