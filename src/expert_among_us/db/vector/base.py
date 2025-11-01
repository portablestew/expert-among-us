"""Abstract base class for vector database implementations.

This module defines the VectorDB interface that all vector storage backends
must implement, along with the VectorSearchResult dataclass for search results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...config.settings import TITAN_EMBEDDING_DIMENSION


@dataclass
class VectorSearchResult:
    """Result from vector similarity search.
    
    Attributes:
        changelist_id: Unique identifier of the matching changelist
        similarity_score: Cosine similarity score (0-1, higher is more similar)
        chroma_id: Optional ChromaDB ID for debugging chunk-level matching
    """
    changelist_id: str
    similarity_score: float
    chroma_id: str | None = None


class VectorDB(ABC):
    """Abstract base class for vector database operations.
    
    Implementations must provide vector storage and similarity search capabilities.
    The vector database stores only changelist IDs and embedding vectors - all
    metadata is stored in the metadata database.
    
    Note: The same changelist_id may appear multiple times in search results when
    searching across both metadata and diff collections. Callers should handle
    deduplication and score merging as needed.
    """

    @abstractmethod
    def initialize(self, dimension: int) -> None:
        """Initialize vector collection with specified dimension.
        
        Creates the vector collection/index with the appropriate dimension
        for the embedding model being used. Should be idempotent.
        
        Args:
            dimension: Dimensionality of the embedding vectors (e.g., {TITAN_EMBEDDING_DIMENSION} for Titan)
        """
        pass

    @abstractmethod
    def insert_vectors(
        self,
        vectors: list[tuple[str, list[float]]]
    ) -> None:
        """Insert vectors in a batch operation.
        
        Stores changelist IDs and their corresponding embedding vectors.
        Only IDs and embeddings are stored - no metadata.
        
        Args:
            vectors: List of tuples, each containing:
                - changelist_id (str): Unique identifier for the changelist
                - embedding (list[float]): Embedding vector
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int
    ) -> list[VectorSearchResult]:
        """Search for similar vectors and return results sorted by similarity.
        
        Performs k-nearest neighbor search using cosine similarity and returns
        the top_k most similar vectors.
        
        Args:
            query_vector: Query embedding vector to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of VectorSearchResult objects, sorted by similarity_score
            (highest/most similar first). May contain duplicate changelist_ids
            when searching across multiple collections.
            
        Note:
            No metadata filtering is performed at this layer. All filtering
            (by author, files, etc.) happens in the metadata database after
            retrieving changelist IDs from vector search.
        """
        pass

    @abstractmethod
    def delete_by_ids(self, changelist_ids: list[str]) -> None:
        """Delete vectors by changelist IDs.
        
        Removes all vectors associated with the specified changelist IDs.
        Should handle missing IDs gracefully (no error if ID doesn't exist).
        
        Args:
            changelist_ids: List of changelist IDs whose vectors should be deleted
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total number of vectors in the collection.
        
        Returns:
            Total count of stored vectors
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection and release resources.
        
        Should be called when done with the database to ensure proper cleanup.
        """
        pass