from abc import ABC, abstractmethod
from typing import List, Optional, Callable


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        
    @abstractmethod
    def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently
        
        Returns None for empty/whitespace-only texts to avoid polluting vector space
        with meaningless embeddings.
        
        Args:
            texts: List of input texts to embed
            progress_callback: Optional callback(current, total) called after each batch
            
        Returns:
            List of embeddings (same length as input), with None for empty texts
        """
        
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension"""