from abc import ABC, abstractmethod
from typing import List, Tuple

class Reranker(ABC):
    """Abstract base class for result reranking."""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rank
            top_k: Return only top K results (optional)
            
        Returns:
            List of (document_index, score) sorted by score descending
        """
        pass