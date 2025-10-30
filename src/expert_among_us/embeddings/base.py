from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension"""