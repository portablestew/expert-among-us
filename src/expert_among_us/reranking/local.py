from typing import List, Tuple
from .base import Reranker
from expert_among_us.utils.progress import console
from expert_among_us.utils.symbols import SYMBOLS
from ..utils.debug import DebugLogger
from ..utils.batching import build_token_batches, estimate_tokens, calculate_max_batch_tokens

class LocalCrossEncoderReranker(Reranker):
    """Local cross-encoder reranker using sentence-transformers.
    
    Implements chunked reranking with max pooling for long documents.
    Documents exceeding the token limit are split into chunks, each chunk
    is scored independently, and the maximum score is used.
    """
    
    def __init__(
        self,
        model_id: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        tokens_per_gb: int = 1024,
        max_chunk_chars: int = 2000  # ~512 tokens
    ):
        """Initialize cross-encoder reranker.
        
        Args:
            model_id: HuggingFace model identifier
            tokens_per_gb: Maximum tokens per batch per GB of GPU memory
            max_chunk_chars: Maximum characters per chunk (~512 tokens)
        """
        self.model_id = model_id
        self.max_chunk_chars = max_chunk_chars
        
        # Use shared utility for device detection and batch token calculation
        self.max_batch_tokens, self.device, torch = calculate_max_batch_tokens(
            tokens_per_gb,
            device_name="reranker"
        )
        
        if DebugLogger.is_enabled():
            console.print("[DEBUG] Loading sentence_transformers...")
        
        # Lazy import sentence_transformers (heavy import ~3.7s)
        from sentence_transformers import CrossEncoder
        
        # Load cross-encoder model
        self.model = CrossEncoder(model_id, device=self.device)
        
        # Store torch reference for later use
        self.torch = torch
        
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
        """Rerank documents with token-aware batching and max pooling.
        
        Long documents are split into chunks, each chunk is scored,
        and the maximum score across chunks is used as the document score.
        Batching is done dynamically based on estimated token counts.
        """
        if not documents:
            return []
        
        # Build query-document pairs
        all_pairs = []
        chunk_to_doc: List[int] = []  # Maps chunk index -> document index
        
        for doc_idx, doc in enumerate(documents):
            chunks = self._chunk_document(doc)
            for chunk in chunks:
                all_pairs.append((query, chunk))  # Tuples for batching
                chunk_to_doc.append(doc_idx)
        
        # Build token-aware batches using shared utility
        batches = build_token_batches(
            all_pairs,
            self.max_batch_tokens,
            item_token_estimator=lambda pair: estimate_tokens(pair[0]) + estimate_tokens(pair[1])
        )
        
        # Score all batches
        all_scores = []
        with self.torch.no_grad():
            for batch_indices in batches:
                batch_pairs = [all_pairs[i] for i in batch_indices]
                
                # Score batch
                batch_scores = self.model.predict(
                    batch_pairs,
                    batch_size=len(batch_pairs),
                    show_progress_bar=False  # Avoid conflict with rich Progress
                )
                
                all_scores.extend(batch_scores)
        
        # Max pooling: aggregate chunk scores by document
        doc_scores = {}
        for chunk_idx, score in enumerate(all_scores):
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