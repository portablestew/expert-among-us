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
        self.collection = None
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
        """Initialize vector collection with specified dimension.
        
        Args:
            dimension: The dimension of the vectors
            require_exists: If True, raise an error if the database doesn't exist
        """
        self._ensure_client(require_exists=require_exists)
        self.collection = self.client.get_or_create_collection(
            name=f"expert_{self.expert_name}",
            metadata={"dimension": dimension, "hnsw:space": "cosine"}
        )
    
    def insert_vectors(self, vectors: list[tuple[str, list[float]]]) -> None:
        """Insert or update vectors in a batch operation (idempotent)."""
        self._ensure_client()
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        ids = [v[0] for v in vectors]
        embeddings = [v[1] for v in vectors]
        # Use upsert instead of add to make re-indexing idempotent
        # This will update existing embeddings or insert new ones
        self.collection.upsert(
            embeddings=embeddings,
            ids=ids
        )
        
    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search for similar vectors and return results sorted by similarity."""
        self._ensure_client(require_exists=True)
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        vector_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                # Convert distance to similarity score (ChromaDB returns cosine distance)
                # Cosine distance = 1 - cosine_similarity, ranges from 0 (identical) to 2 (opposite)
                # We convert back to similarity and clamp to [0, 1] range
                distance = results['distances'][0][i] if results['distances'] else 0.0
                similarity = max(0.0, min(1.0, 1.0 - distance))
                vector_results.append(VectorSearchResult(
                    changelist_id=results['ids'][0][i],
                    similarity_score=similarity
                ))
        return vector_results
    
    def search_metadata(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search for similar metadata embeddings and return results sorted by similarity.
        
        Searches the collection and filters for IDs that don't end with '_diff',
        which represent metadata embeddings.
        
        Args:
            query_vector: Query embedding vector to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of VectorSearchResult objects with metadata embeddings only
        """
        self._ensure_client(require_exists=True)
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        # Request more results to account for filtering
        # We need to fetch enough to get top_k metadata results after filtering out diffs
        fetch_count = top_k * 2
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_count
        )
        
        vector_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result_id = results['ids'][0][i]
                
                # Filter: only include IDs that don't end with '_diff'
                if result_id.endswith('_diff'):
                    continue
                
                # Convert distance to similarity score
                distance = results['distances'][0][i] if results['distances'] else 0.0
                similarity = max(0.0, min(1.0, 1.0 - distance))
                
                vector_results.append(VectorSearchResult(
                    changelist_id=result_id,
                    similarity_score=similarity
                ))
                
                # Stop once we have enough metadata results
                if len(vector_results) >= top_k:
                    break
        
        return vector_results
    
    def search_diffs(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search for similar diff embeddings and return results sorted by similarity.
        
        Searches the collection and filters for IDs that end with '_diff',
        which represent diff embeddings. The '_diff' suffix is stripped from
        the changelist_id in the results.
        
        Args:
            query_vector: Query embedding vector to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of VectorSearchResult objects with diff embeddings only,
            with changelist_id having the '_diff' suffix removed and chroma_id preserved
        """
        self._ensure_client(require_exists=True)
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        # Request more results to account for filtering and multiple chunks per commit
        # We need to fetch enough to get top_k diff results after filtering out metadata
        fetch_count = top_k * 5
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_count
        )
        
        vector_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result_id = results['ids'][0][i]
                
                # Filter: include IDs ending with '_diff' or containing '_diff_chunk_'
                is_diff = result_id.endswith('_diff')
                is_chunk = '_diff_chunk_' in result_id
                
                if not (is_diff or is_chunk):
                    continue
                
                # Convert distance to similarity score
                distance = results['distances'][0][i] if results['distances'] else 0.0
                similarity = max(0.0, min(1.0, 1.0 - distance))
                
                # Extract changelist_id from chunk IDs
                if is_chunk:
                    changelist_id = result_id.split('_diff_chunk_')[0]
                else:
                    changelist_id = result_id[:-5]  # Remove '_diff'
                
                vector_results.append(VectorSearchResult(
                    changelist_id=changelist_id,
                    similarity_score=similarity,
                    chroma_id=result_id  # Preserve full ChromaDB ID for debugging
                ))
                
                # Stop once we have enough diff results
                if len(vector_results) >= top_k:
                    break
        
        return vector_results
    
    def delete_by_ids(self, changelist_ids: list[str]) -> None:
        """Delete vectors by changelist IDs."""
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call initialize() first.")
        
        try:
            self.collection.delete(ids=changelist_ids)
        except Exception:
            # Handle missing IDs gracefully
            pass
    
    def count(self) -> int:
        """Get total number of vectors in the collection."""
        if not self.collection:
            return 0
        return self.collection.count()
    
    def close(self) -> None:
        """Close database connection and release resources."""
        # Clear collection reference
        self.collection = None
        
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