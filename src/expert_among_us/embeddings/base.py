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
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of input texts to embed
            progress_callback: Optional callback(current, total) called after each batch
        """
        
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension"""