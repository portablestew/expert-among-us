from typing import List, Tuple
import torch
from sentence_transformers import CrossEncoder
from .base import Reranker
from expert_among_us.utils.progress import console
from expert_among_us.utils.symbols import SYMBOLS

class LocalCrossEncoderReranker(Reranker):
    """Local cross-encoder reranker using sentence-transformers.
    
    Implements chunked reranking with max pooling for long documents.
    Documents exceeding the token limit are split into chunks, each chunk
    is scored independently, and the maximum score is used.
    """
    
    def __init__(
        self,
        model_id: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        batch_size: int = 32,
        max_chunk_chars: int = 2000  # ~512 tokens
    ):
        """Initialize cross-encoder reranker.
        
        Args:
            model_id: HuggingFace model identifier
            batch_size: Batch size for GPU inference
            max_chunk_chars: Maximum characters per chunk (~512 tokens)
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_chunk_chars = max_chunk_chars
        
        # Auto-detect device (same pattern as JinaCodeEmbedder)
        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available else "cpu"
        
        # Load cross-encoder model
        self.model = CrossEncoder(model_id, device=self.device)
        
        # Log device
        if self.device == "cuda":
            device_name = torch.cuda.get_device_name(0)
            console.print(f"{SYMBOLS['success']} Cross-encoder reranker using GPU: {device_name}")
        else:
            console.print(f"{SYMBOLS['info']} Cross-encoder reranker using CPU (slower)")
    
    def _chunk_document(self, document: str) -> List[str]:
        """Split document into chunks for reranking.
        
        Args:
            document: Text to chunk
            
        Returns:
            List of text chunks, each <= max_chunk_chars
        """
        if len(document) <= self.max_chunk_chars:
            return [document]
        
        chunks = []
        for i in range(0, len(document), self.max_chunk_chars):
            chunks.append(document[i:i + self.max_chunk_chars])
        
        return chunks
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """Rerank documents using cross-encoder scores with max pooling.
        
        Long documents are split into chunks, each chunk is scored,
        and the maximum score across chunks is used as the document score.
        """
        if not documents:
            return []
        
        # Split documents into chunks and track mappings
        all_pairs = []
        chunk_to_doc: List[int] = []  # Maps chunk index -> document index
        
        for doc_idx, doc in enumerate(documents):
            chunks = self._chunk_document(doc)
            for chunk in chunks:
                all_pairs.append([query, chunk])
                chunk_to_doc.append(doc_idx)
        
        # Batch predict all chunks (uses GPU if available)
        chunk_scores = self.model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False  # Avoid conflict with rich Progress
        )
        
        # Max pooling: aggregate chunk scores by document
        doc_scores = {}
        for chunk_idx, score in enumerate(chunk_scores):
            doc_idx = chunk_to_doc[chunk_idx]
            
            # Keep maximum score across all chunks
            if doc_idx not in doc_scores or score > doc_scores[doc_idx]:
                doc_scores[doc_idx] = score
        
        # Sort by score descending
        ranked = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to top_k if specified
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked