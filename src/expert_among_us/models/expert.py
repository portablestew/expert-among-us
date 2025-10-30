"""Expert configuration model representing an indexed repository."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExpertConfig(BaseModel):
    """Configuration for an expert (indexed repository).
    
    This model defines all settings and metadata for an expert, including
    the repository location, VCS type, indexing parameters, and tracking
    information for incremental updates.
    
    Attributes:
        name: Expert identifier (unique, used as directory name)
        workspace_path: Path to repository root
        subdirs: Optional subdirectory filters for indexing
        vcs_type: Version control system type ('git' or 'p4')
        data_dir: Base directory for expert data storage
        created_at: When the expert was first created
        last_indexed_at: Last successful index time
        last_commit_time: Most recent commit indexed
        max_commits: Maximum commits to index
        max_diff_size: Maximum diff size in bytes (100KB default)
        max_embedding_text_size: Maximum text per embedding (~8K tokens, 30KB default)
        embed_diffs: Whether to create diff embeddings
        embed_metadata: Whether to create metadata embeddings
    """

    name: str = Field(..., description="Expert identifier (unique)")
    workspace_path: Path = Field(..., description="Path to repository")
    subdirs: List[str] = Field(
        default_factory=list, description="Optional subdir filters"
    )
    vcs_type: Literal["git", "p4"] = Field(
        default="git", description="Version control system type"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".expert-among-us",
        description="Base directory for expert data storage"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the expert was created",
    )
    last_indexed_at: Optional[datetime] = Field(
        None, description="Last successful index time"
    )
    last_commit_time: Optional[datetime] = Field(
        None, description="Most recent commit indexed"
    )
    max_commits: int = Field(
        default=10000, ge=1, description="Max commits to index"
    )
    max_diff_size: int = Field(
        default=100000, ge=1, description="Max diff size in bytes (100KB)"
    )
    max_embedding_text_size: int = Field(
        default=30000, ge=1, description="Max text per embedding (~8K tokens)"
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
                "workspace_path": "/home/user/projects/myrepo",
                "subdirs": ["src/main/", "src/resources/"],
                "vcs_type": "git",
                "max_commits": 10000,
            }
        }
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that expert name is valid for use as a filename.
        
        Expert names must be alphanumeric with optional hyphens and underscores,
        as they are used as directory names in the storage structure.
        
        Args:
            v: Name to validate
            
        Returns:
            Validated name
            
        Raises:
            ValueError: If name is invalid
        """
        if not v or not v.strip():
            raise ValueError("Expert name cannot be empty")
        
        # Remove valid separators and check if remaining chars are alphanumeric
        sanitized = v.replace("_", "").replace("-", "")
        if not sanitized.isalnum():
            raise ValueError(
                "Expert name must be alphanumeric with hyphens/underscores only"
            )
        
        return v.strip()

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace(cls, v: Path) -> Path:
        """Validate that workspace path exists.
        
        Args:
            v: Workspace path to validate
            
        Returns:
            Validated path
            
        Raises:
            ValueError: If path does not exist
        """
        if not v.exists():
            raise ValueError(f"Workspace path does not exist: {v}")
        
        if not v.is_dir():
            raise ValueError(f"Workspace path is not a directory: {v}")
        
        return v

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