"""FileChunk data model representing a chunk of file content with location tracking."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FileChunk:
    """Represents a chunk of a file's content with location tracking.
    
    This model stores chunks of file content from a specific revision, enabling
    semantic search across the current state of the codebase. Each chunk
    tracks its position within the original file for context retrieval.
    
    Attributes:
        file_path: Relative path to the file in the repository
        chunk_index: Zero-based index of this chunk within the file
        content: The actual text content of this chunk
        line_start: Starting line number in the original file (1-based)
        line_end: Ending line number in the original file (1-based, inclusive)
        revision_id: Revision identifier when this content was indexed (e.g. commit hash or changelist ID)
    """
    
    file_path: str
    chunk_index: int
    content: str
    line_start: int
    line_end: int
    revision_id: str
    
    def get_chroma_id(self) -> str:
        """Generate ChromaDB ID for this chunk.
        
        Creates a unique identifier for storing/retrieving this chunk
        in the vector database. Format: "file:<path>:chunk_<index>"
        
        Returns:
            Unique string identifier for ChromaDB
        """
        return f"file:{self.file_path}:chunk_{self.chunk_index}"