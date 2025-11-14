from typing import List, Tuple
from .base import Reranker
from expert_among_us.utils.progress import console
from expert_among_us.utils.symbols import SYMBOLS
from ..utils.debug import DebugLogger

class LocalCrossEncoderReranker(Reranker):
    """Local cross-encoder reranker using sentence-transformers.
    
    Implements chunked reranking with max pooling for long documents.
    Documents exceeding the token limit are split into chunks, each chunk
    is scored independently, and the maximum score is used.
    """
    
    def __init__(
        self,
        model_id: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        batch_per_gb: int = 1.0,
        max_chunk_chars: int = 2000  # ~512 tokens
    ):
        """Initialize cross-encoder reranker.
        
        Args:
            model_id: HuggingFace model identifier
            batch_per_gb: Batch size per GB of GPU memory for inference
            max_chunk_chars: Maximum characters per chunk (~512 tokens)
        """
        # Log loading message in debug mode
        if DebugLogger.is_enabled():
            console.print("[DEBUG] Loading PyTorch...")
        
        # Lazy import torch (heavy import ~2.4s)
        import torch
        
        if DebugLogger.is_enabled():
            console.print("[DEBUG] Loading sentence_transformers...")
        
        # Lazy import sentence_transformers (heavy import ~3.7s)
        from sentence_transformers import CrossEncoder
        
        self.model_id = model_id
        self.max_chunk_chars = max_chunk_chars
        
        # Auto-detect device (same pattern as JinaCodeEmbedder)
        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available else "cpu"
        
        # Calculate actual batch size based on GPU memory
        if self.device == "cuda":
            # Get total GPU memory in GB
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / (1024**3)
            self.batch_size = round(batch_per_gb * total_memory_gb)
            if DebugLogger.is_enabled():
                console.print(f"[DEBUG] GPU memory: {total_memory_gb:.2f} GB, batch_per_gb: {batch_per_gb}, calculated batch_size: {self.batch_size}")
        else:
            # For CPU, use number of cores * batch_per_gb
            import os
            cpu_cores = os.cpu_count() or 1
            self.batch_size = round(batch_per_gb * cpu_cores)
            if DebugLogger.is_enabled():
                console.print(f"[DEBUG] CPU cores: {cpu_cores}, batch_per_gb: {batch_per_gb}, calculated batch_size: {self.batch_size}")
        
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
        # Use no_grad to prevent gradient accumulation during inference
        with self.torch.no_grad():
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