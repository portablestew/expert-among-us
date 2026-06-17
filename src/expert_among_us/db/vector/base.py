"""Abstract base class for vector database implementations.

This module defines the VectorDB interface that all vector storage backends
must implement. The VectorSearchResult model is imported from the models module.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ...models.query import VectorSearchResult
from ...config.settings import TITAN_EMBEDDING_DIMENSION


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
    def insert_metadata(
        self,
        vectors: list[tuple[str, list[float]]],
        metadata: Optional[dict] = None
    ) -> None:
        """Insert commit metadata vectors.
        
        Stores metadata embeddings (commit descriptions, messages, authors).
        
        Args:
            vectors: List of tuples, each containing:
                - changelist_id (str): Unique identifier for the commit
                - embedding (list[float]): Metadata embedding vector
            metadata: Optional dict of metadata to attach to each vector
                (e.g., {"project": "payment-service"})
        """
        pass

    @abstractmethod
    def insert_diffs(
        self,
        vectors: list[tuple[str, list[float]]],
        metadata: Optional[dict] = None
    ) -> None:
        """Insert diff chunk vectors.
        
        Stores embeddings for code change diffs, potentially chunked for large diffs.
        
        Args:
            vectors: List of tuples, each containing:
                - vector_id (str): Identifier for the diff chunk (e.g., commit_hash_chunk_N)
                - embedding (list[float]): Diff embedding vector
            metadata: Optional dict of metadata to attach to each vector
                (e.g., {"project": "payment-service"})
        """
        pass

    @abstractmethod
    def insert_files(
        self,
        vectors: list[tuple[str, list[float]]],
        metadata: Optional[dict] = None
    ) -> None:
        """Insert file content chunk vectors.
        
        Stores embeddings for file content at HEAD, chunked for large files.
        
        Args:
            vectors: List of tuples, each containing:
                - chunk_id (str): Identifier for the file chunk (e.g., file:path:chunk_N)
                - embedding (list[float]): File content embedding vector
            metadata: Optional dict of metadata to attach to each vector
                (e.g., {"project": "payment-service"})
        """
        pass


    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        include_embeddings: bool = False
    ) -> list[VectorSearchResult]:
        """Search for similar vectors and return results sorted by similarity.
        
        Performs k-nearest neighbor search using cosine similarity and returns
        the top_k most similar vectors.
        
        Args:
            query_vector: Query embedding vector to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of VectorSearchResult objects, sorted by similarity_score
            (highest/most similar first). May contain duplicate result_ids
            when searching across multiple collections.
            
        Note:
            No metadata filtering is performed at this layer. All filtering
            (by author, files, etc.) happens in the metadata database after
            retrieving changelist IDs from vector search.
        """
        pass

    @abstractmethod
    def delete_project_vectors(self, project_name: str) -> None:
        """Delete all vectors belonging to a project from all collections.
        
        Removes vectors from metadata, diff, and file collections where
        the project metadata matches the given project name.
        
        Args:
            project_name: Name of the project whose vectors should be deleted
        """
        pass

    @abstractmethod
    def delete_file_chunks(
        self,
        chunk_ids: list[str]
    ) -> None:
        """Delete file chunk vectors from the files collection.
        
        Args:
            chunk_ids: List of file chunk IDs to delete (format: file:{path}:chunk_{n})
        """
        pass

    @abstractmethod
    def search_metadata(
        self,
        query_vector: list[float],
        top_k: int,
        include_embeddings: bool = False,
        where: Optional[dict] = None
    ) -> list[VectorSearchResult]:
        """Search metadata collection for similar vectors.
        
        Searches commit metadata (messages, authors, descriptions) for similar vectors.
        
        Args:
            query_vector: Query embedding vector to search for
            top_k: Maximum number of results to return
            include_embeddings: Whether to include embedding vectors in results
            where: Optional ChromaDB where clause for filtering
                (e.g., {"project": {"$in": ["payment-service", "user-service"]}})
                When None, no filtering is applied.
            
        Returns:
            List of VectorSearchResult objects sorted by similarity score
        """
        pass

    @abstractmethod
    def search_diffs(
        self,
        query_vector: list[float],
        top_k: int,
        include_embeddings: bool = False,
        where: Optional[dict] = None
    ) -> list[VectorSearchResult]:
        """Search diff collection for similar vectors.
        
        Searches code change diffs for similar vectors.
        
        Args:
            query_vector: Query embedding vector to search for
            top_k: Maximum number of results to return
            include_embeddings: Whether to include embedding vectors in results
            where: Optional ChromaDB where clause for filtering
                (e.g., {"project": {"$in": ["payment-service", "user-service"]}})
                When None, no filtering is applied.
            
        Returns:
            List of VectorSearchResult objects sorted by similarity score
        """
        pass

    @abstractmethod
    def search_files(
        self,
        query_vector: list[float],
        top_k: int,
        include_embeddings: bool = False,
        where: Optional[dict] = None
    ) -> list[VectorSearchResult]:
        """Search file collection for similar vectors.
        
        Searches file content chunks for similar vectors.
        
        Args:
            query_vector: Query embedding vector to search for
            top_k: Maximum number of results to return
            include_embeddings: Whether to include embedding vectors in results
            where: Optional ChromaDB where clause for filtering
                (e.g., {"project": {"$in": ["payment-service", "user-service"]}})
                When None, no filtering is applied.
            
        Returns:
            List of VectorSearchResult objects sorted by similarity score
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