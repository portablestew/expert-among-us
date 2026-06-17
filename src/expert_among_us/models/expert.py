"""Expert and project configuration models."""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Regex for valid project/expert names: alphanumeric, hyphens, and underscores (can start with any)
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-][a-zA-Z0-9_-]*$")


def _validate_identifier_name(v: str, field_label: str) -> str:
    """Validate that a name matches [a-zA-Z0-9][a-zA-Z0-9_-]*.

    Rejects empty strings, strings with path separators, and strings
    that don't match the required pattern.

    Args:
        v: Name to validate
        field_label: Human-readable label for error messages (e.g. "Expert name")

    Returns:
        Validated name (stripped of leading/trailing whitespace)

    Raises:
        ValueError: If name is invalid
    """
    if not v or not v.strip():
        raise ValueError(f"{field_label} cannot be empty")

    v = v.strip()

    # Reject path separators explicitly for clear error messages
    if "/" in v or "\\" in v:
        raise ValueError(
            f"{field_label} must not contain path separators"
        )

    if not _NAME_PATTERN.match(v):
        raise ValueError(
            f"{field_label} must contain only alphanumeric characters, hyphens, and underscores"
        )

    return v


class ProjectConfig(BaseModel):
    """Configuration for a project within an expert.

    A project represents a physical repository link within an expert.
    Multiple projects can share a single expert's ChromaDB vector space,
    enabling unified semantic search across related repositories.

    Attributes:
        name: Project identifier within the expert (unique per expert)
        expert_name: Parent expert name
        project_root: Path to the project root (the indexed directory)
        vcs_type: Version control system type ('git' or 'p4')
        last_indexed_at: Last successful index time for this project
        last_processed_commit_hash: Most recent commit hash indexed
        first_processed_commit_hash: Oldest commit hash indexed (for display)
        has_vector_metadata: Whether ChromaDB vectors have project metadata
        created_at: When the project was first created
    """

    name: str = Field(..., description="Project identifier within expert")
    expert_name: str = Field(..., description="Parent expert name")
    project_root: Path = Field(..., description="Path to the project root")
    vcs_type: Literal["git", "p4"] = Field(
        default="git", description="Version control system type"
    )
    last_indexed_at: Optional[datetime] = Field(
        None, description="Last successful index time"
    )
    last_processed_commit_hash: Optional[str] = Field(
        None, description="Most recent commit hash indexed"
    )
    first_processed_commit_hash: Optional[str] = Field(
        None, description="Oldest commit hash indexed (for display)"
    )
    has_vector_metadata: bool = Field(
        default=True,
        description="Whether ChromaDB vectors carry project metadata",
    )
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the project was created",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "payment-service",
                "expert_name": "my-team",
                "project_root": "/repos/payment",
                "vcs_type": "git",
            }
        }
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that project name is safe for path prefixing.

        Project names must contain only alphanumeric characters, hyphens,
        and underscores since they are used as path prefixes in the virtual
        unified namespace.

        Args:
            v: Name to validate

        Returns:
            Validated name

        Raises:
            ValueError: If name is invalid
        """
        return _validate_identifier_name(v, "Project name")

    @field_validator("expert_name")
    @classmethod
    def validate_expert_name(cls, v: str) -> str:
        """Validate that parent expert name is valid.

        Args:
            v: Expert name to validate

        Returns:
            Validated expert name

        Raises:
            ValueError: If name is invalid
        """
        return _validate_identifier_name(v, "Expert name")

    @field_validator("project_root")
    @classmethod
    def validate_project_root(cls, v: Path) -> Path:
        """Validate that the project root exists and is a directory.

        Args:
            v: Project root path to validate

        Returns:
            Validated path

        Raises:
            ValueError: If path does not exist or is not a directory
        """
        if not v.exists():
            raise ValueError(f"Project root does not exist: {v}")

        if not v.is_dir():
            raise ValueError(f"Project root is not a directory: {v}")

        return v


class ExpertConfig(BaseModel):
    """Configuration for an expert (logical grouping of projects).

    An expert owns a ChromaDB vector space and contains one or more projects.
    The project_root and vcs_type are stored per-project in ProjectConfig
    rather than on the expert.

    Attributes:
        name: Expert identifier (unique, used as directory name)
        description: Optional human-readable description of the expert
        data_dir: Base directory for expert data storage
        created_at: When the expert was first created
        last_indexed_at: Last successful index time (aggregate across projects)
        max_metadata_embedding_size: Maximum bytes for metadata embeddings (20KB default)
        max_embedding_text_size: Maximum bytes for diff before chunking (100KB default)
        file_chunk_size_bytes: File content chunk size (4KB default)
        diff_chunk_size_bytes: Chunk size for diff embeddings (4KB default)
        embed_diffs: Whether to create diff embeddings
        embed_metadata: Whether to create metadata embeddings
    """

    name: str = Field(..., description="Expert identifier (unique)")
    description: Optional[str] = Field(
        None, description="Human-readable description of the expert"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".expert-among-us",
        description="Base directory for expert data storage",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the expert was created",
    )
    last_indexed_at: Optional[datetime] = Field(
        None, description="Last successful index time"
    )
    max_metadata_embedding_size: int = Field(
        default=20000, ge=1, description="Max bytes for metadata embeddings (20KB)"
    )
    max_embedding_text_size: int = Field(
        default=100000, ge=1, description="Max bytes for diff before chunking (100KB)"
    )
    file_chunk_size_bytes: int = Field(
        default=4096, ge=1, description="File content chunk size (4KB)"
    )
    diff_chunk_size_bytes: int = Field(
        default=4096, ge=1, description="Chunk size for diff embeddings (4KB)"
    )
    embed_diffs: bool = Field(
        default=True, description="Whether to create diff embeddings"
    )
    embed_metadata: bool = Field(
        default=True, description="Whether to create metadata embeddings"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "MyExpert",
                "description": "Team microservices expert",
            }
        }
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that expert name is valid for use as a filename.

        Expert names must contain only alphanumeric characters, hyphens,
        and underscores as they are used as directory names in the storage
        structure.

        Args:
            v: Name to validate

        Returns:
            Validated name

        Raises:
            ValueError: If name is invalid
        """
        return _validate_identifier_name(v, "Expert name")

    def get_storage_dir(self) -> Path:
        """Get the storage directory for this expert.

        Returns the base directory where all data for this expert is stored,
        including metadata database and vector database.

        Returns:
            Path to storage directory: {data_dir}/data/{name}/
        """
        storage_dir = self.data_dir / "data" / self.name
        return storage_dir

    def get_metadata_db_path(self) -> Path:
        """Get the path to the SQLite metadata database.

        Returns:
            Path to metadata.db file
        """
        return self.get_storage_dir() / "metadata.db"

    def get_vector_db_path(self) -> Path:
        """Get the path to the ChromaDB vector database directory.

        Returns:
            Path to chroma/ directory
        """
        return self.get_storage_dir() / "chroma"

    def ensure_storage_exists(self) -> None:
        """Create storage directories if they don't exist.

        This creates the full storage structure for the expert, including
        the base directory and subdirectories for databases.
        """
        storage_dir = self.get_storage_dir()
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Create vector DB directory
        vector_dir = self.get_vector_db_path()
        vector_dir.mkdir(parents=True, exist_ok=True)