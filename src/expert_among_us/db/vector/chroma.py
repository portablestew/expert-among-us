import os
import chromadb
from pathlib import Path
from typing import Optional
from expert_among_us.db.vector.base import VectorDB, VectorSearchResult

class ChromaVectorDB(VectorDB):
    def __init__(self, expert_name: str, data_dir: Optional[Path] = None):
        base_dir = data_dir or Path.home() / ".expert-among-us"
        self.storage_path = Path(base_dir / "data" / expert_name / "chroma")
        self.client = None
        self.metadata_collection = None
        self.diff_collection = None
        self.file_collection = None
        self.expert_name = expert_name
    
    def exists(self) -> bool:
        """Check if the ChromaDB storage directory exists."""
        return self.storage_path.exists()
    
    def _ensure_client(self, require_exists: bool = False):
        """Ensure the ChromaDB client is initialized.
        
        Args:
            require_exists: If True, raise an error if the database doesn't exist
        """
        if self.client is None:
            if require_exists and not self.exists():
                raise FileNotFoundError(
                    f"Expert database '{self.expert_name}' does not exist. "
                    f"Please run 'populate' command first to create the expert index."
                )
            
            self.client = chromadb.PersistentClient(
                path=str(self.storage_path),
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
    
    def initialize(self, dimension: int, require_exists: bool = False) -> None:
        """Initialize vector collections with specified dimension.
        
        Args:
            dimension: The dimension of the vectors
            require_exists: If True, raise an error if the database doesn't exist
        """
        self._ensure_client(require_exists=require_exists)
        
        # Create three separate collections for different content types
        self.metadata_collection = self.client.get_or_create_collection(
            name=f"{self.expert_name}_metadata",
            metadata={"dimension": dimension, "hnsw:space": "cosine"}
        )
        self.diff_collection = self.client.get_or_create_collection(
            name=f"{self.expert_name}_diffs",
            metadata={"dimension": dimension, "hnsw:space": "cosine"}
        )
        self.file_collection = self.client.get_or_create_collection(
            name=f"{self.expert_name}_files",
            metadata={"dimension": dimension, "hnsw:space": "cosine"}
        )
    
    def insert_vectors(self, vectors: list[tuple[str, list[float]]]) -> None:
        """Insert or update vectors in a batch operation (idempotent)."""
        self._ensure_client()
        if not self.metadata_collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        ids = [v[0] for v in vectors]
        embeddings = [v[1] for v in vectors]
        # Use upsert instead of add to make re-indexing idempotent
        # This will update existing embeddings or insert new ones
        self.metadata_collection.upsert(
            embeddings=embeddings,
            ids=ids
        )
    
    def insert_metadata(self, vectors: list[tuple[str, list[float]]]) -> None:
        """Insert commit metadata vectors."""
        if not self.metadata_collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        ids = [v[0] for v in vectors]
        embeddings = [v[1] for v in vectors]
        self.metadata_collection.upsert(embeddings=embeddings, ids=ids)

    def insert_diffs(self, vectors: list[tuple[str, list[float]]]) -> None:
        """Insert diff chunk vectors."""
        if not self.diff_collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        ids = [v[0] for v in vectors]
        embeddings = [v[1] for v in vectors]
        self.diff_collection.upsert(embeddings=embeddings, ids=ids)

    def insert_files(self, vectors: list[tuple[str, list[float]]]) -> None:
        """Insert file content chunk vectors."""
        if not self.file_collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        ids = [v[0] for v in vectors]
        embeddings = [v[1] for v in vectors]
        self.file_collection.upsert(embeddings=embeddings, ids=ids)
        
    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search for similar vectors and return results sorted by similarity."""
        self._ensure_client(require_exists=True)
        if not self.metadata_collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        results = self.metadata_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        return self._parse_results(results)
    
    def search_metadata(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search metadata collection - NO FILTERING NEEDED."""
        self._ensure_client(require_exists=True)
        if not self.metadata_collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        results = self.metadata_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k  # Direct top_k, no multiplier
        )
        
        return self._parse_results(results)
    
    def search_diffs(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search diffs collection - NO FILTERING NEEDED."""
        self._ensure_client(require_exists=True)
        if not self.diff_collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        results = self.diff_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k  # Direct top_k, no multiplier
        )
        
        # Extract commit hash from chunk IDs
        vector_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result_id = results['ids'][0][i]
                # ID format: {commit_hash}_chunk_{n}
                changelist_id = result_id.split('_chunk_')[0] if '_chunk_' in result_id else result_id
                distance = results['distances'][0][i] if results['distances'] else 0.0
                similarity = max(0.0, min(1.0, 1.0 - distance))
                
                vector_results.append(VectorSearchResult(
                    result_id=changelist_id,  # Commit hash for diff results
                    similarity_score=similarity,
                    source="diff",
                    chroma_id=result_id
                ))
        
        return vector_results
    
    def search_files(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search files collection - NO FILTERING NEEDED."""
        self._ensure_client(require_exists=True)
        if not self.file_collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        results = self.file_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k  # Direct top_k, no multiplier
        )
        
        # Extract file path from chunk IDs
        vector_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result_id = results['ids'][0][i]
                # ID format: {file_path}:chunk_{n}
                file_path = result_id.split(':chunk_')[0] if ':chunk_' in result_id else result_id
                distance = results['distances'][0][i] if results['distances'] else 0.0
                similarity = max(0.0, min(1.0, 1.0 - distance))
                
                vector_results.append(VectorSearchResult(
                    result_id=file_path,  # File path for file results (used as result grouping key)
                    similarity_score=similarity,
                    source="file",
                    chroma_id=result_id
                ))
        
        return vector_results
    
    def _parse_results(self, results) -> list[VectorSearchResult]:
        """Helper to convert ChromaDB results to VectorSearchResult."""
        vector_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if results['distances'] else 0.0
                similarity = max(0.0, min(1.0, 1.0 - distance))
                vector_results.append(VectorSearchResult(
                    result_id=results['ids'][0][i],  # Can be commit hash or chunk ID depending on collection
                    similarity_score=similarity,
                    source="metadata"  # Default to metadata for generic search
                ))
        return vector_results
    
    def delete_file_chunks(self, chunk_ids: list[str]) -> None:
        """Delete file chunk vectors from the files collection."""
        if not chunk_ids:
            return
            
        if not self.file_collection:
            raise RuntimeError("File collection not initialized. Call initialize() first.")
        
        try:
            self.file_collection.delete(ids=chunk_ids)
        except Exception:
            # Handle missing IDs gracefully
            pass
    
    def count(self) -> int:
        """Get total number of vectors across all collections."""
        total = 0
        if self.metadata_collection:
            total += self.metadata_collection.count()
        if self.diff_collection:
            total += self.diff_collection.count()
        if self.file_collection:
            total += self.file_collection.count()
        return total
    
    def close(self) -> None:
        """Close database connection and release resources."""
        # Clear collection references
        self.metadata_collection = None
        self.diff_collection = None
        self.file_collection = None
        
        # Properly shut down ChromaDB client to release file handles (critical on Windows)
        if self.client is not None:
            try:
                # Stop the producer thread if it exists
                if hasattr(self.client, '_producer') and self.client._producer:
                    self.client._producer.stop()
                
                # Clear the system cache to release file handles
                if hasattr(self.client, 'clear_system_cache'):
                    self.client.clear_system_cache()
                
                # Reset internal server to release resources
                if hasattr(self.client, '_server'):
                    self.client._server = None
            except Exception:
                # Ignore cleanup errors
                pass
            finally:
                self.client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False
