from .base import Embedder
from typing import List
from ..utils.debug import DebugLogger
from ..utils.symbols import SYMBOLS
from ..utils.progress import console


class JinaCodeEmbedder(Embedder):
    """Local embeddings using Jina Code model with code2code task."""
    
    def __init__(self, model_id: str, dimension: int = 512, compile_model: bool = True):
        """Initialize the Jina Code embedder.
        
        Args:
            model_id: Hugging Face model identifier
            dimension: Target dimension for Matryoshka truncation
            compile_model: Whether to use torch.compile for optimization (default: True)
        """
        # Log loading message in debug mode
        if DebugLogger.is_enabled():
            console.print("[DEBUG] Loading PyTorch...")
        
        # Lazy import torch (heavy import ~2.4s)
        import torch
        
        if DebugLogger.is_enabled():
            console.print("[DEBUG] Loading sentence_transformers...")
        
        # Lazy import sentence_transformers (heavy import ~3.7s)
        from sentence_transformers import SentenceTransformer
        
        self.model_id = model_id
        self._dimension = dimension
        self.task = "code2code"
        self.task_prefix = "Represent this code for retrieving similar code: "
        
        # Auto-detect GPU availability with diagnostics
        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available else "cpu"
        
        # Detailed GPU diagnostics
        if not cuda_available:
            console.print(f"{SYMBOLS['warning']}  GPU not detected - using CPU")
            console.print(f"   PyTorch version: {torch.__version__}")
            console.print(f"   CUDA available: {cuda_available}")
            console.print(f"   CUDA built: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
            if hasattr(torch.cuda, 'device_count'):
                console.print(f"   GPU count: {torch.cuda.device_count()}")
            console.print("\n   To enable GPU (run after uv sync):")
            console.print("   1. Check NVIDIA drivers: nvidia-smi")
            console.print("   2. Install PyTorch with CUDA: ./install-gpu.sh or ./install-gpu.ps1")
            console.print("   3. Restart your terminal or reactivate venv")
            console.print()
        
        # Load model with trust_remote_code for FlashAttention2 and fp16 for GPU optimization
        self.model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
            device=self.device,
            model_kwargs={'dtype': torch.float16}
        )
        
        # Compile model for faster inference (PyTorch 2.0+) if enabled
        compilation_enabled = False
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                compilation_enabled = True
            except Exception as e:
                # torch.compile may not be available or may fail
                if DebugLogger.is_enabled():
                    console.print(f"[DEBUG] torch.compile not available or failed: {e}")
        else:
            if DebugLogger.is_enabled():
                console.print("[DEBUG] torch.compile disabled by user")
        
        # Log device being used
        if self.device == "cuda":
            device_name = torch.cuda.get_device_name(0)
            compile_status = " (compiled)" if compilation_enabled else " (not compiled)"
            console.print(f"{SYMBOLS['success']} Local embeddings using GPU: {device_name}{compile_status}")
        else:
            console.print(f"{SYMBOLS['info']} Local embeddings using CPU (this will be slower)")
            console.print(f"{SYMBOLS['info']} Using fp16 precision (auto-casts on CPU)")
        
        # Store torch reference for later use
        self.torch = torch
        
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text using code2code task.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        # Add task prefix
        prefixed_text = self.task_prefix + text
        
        # Log request if debug enabled
        request_id = None
        if DebugLogger.is_enabled():
            request = {
                "model_id": self.model_id,
                "task": self.task,
                "dimension": self._dimension,
                "text_length": len(text),
                "prefixed_text_length": len(prefixed_text)
            }
            request_id = DebugLogger.log_request(
                "local",
                request,
                category="embedding"
            )
        
        # Generate embedding with no_grad to prevent gradient accumulation
        # Disable internal tqdm bar to avoid conflicts with rich Progress/indexer output
        with self.torch.no_grad():
            embedding = self.model.encode(
                prefixed_text,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()
        
        # Truncate to target dimension (Matryoshka)
        embedding = embedding[:self._dimension]
        
        # Log response if debug enabled
        if DebugLogger.is_enabled():
            DebugLogger.log_response(
                "local",
                {
                    "embedding": embedding,
                    "response_metadata": {
                        "model_id": self.model_id,
                        "embedding_dimension": len(embedding),
                        "task": self.task
                    }
                },
                request_id,
                category="embedding"
            )
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Add task prefix to all texts
        prefixed_texts = [self.task_prefix + text for text in texts]
        
        # Log request if debug enabled
        request_id = None
        if DebugLogger.is_enabled():
            request = {
                "model_id": self.model_id,
                "task": self.task,
                "dimension": self._dimension,
                "batch_size": len(texts),
                "text_lengths": [len(text) for text in texts]
            }
            request_id = DebugLogger.log_request(
                "local",
                request,
                category="embedding"
            )
        
        # Generate embeddings in batch with no_grad to prevent gradient accumulation.
        # We RE-ENABLE the internal tqdm progress bar here because batch embedding
        # progress is critical for UX. To keep output readable, rely on tqdm's
        # single-line behavior and avoid overlapping rich Progress for this section.
        with self.torch.no_grad():
            embeddings = self.model.encode(
                prefixed_texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=12,
            )
        
        # Truncate to target dimension (Matryoshka) and convert to lists
        embeddings_list = [emb[:self._dimension].tolist() for emb in embeddings]
        
        # Log response if debug enabled
        if DebugLogger.is_enabled():
            DebugLogger.log_response(
                "local",
                {
                    "embeddings_count": len(embeddings_list),
                    "response_metadata": {
                        "model_id": self.model_id,
                        "embedding_dimension": self._dimension,
                        "task": self.task
                    }
                },
                request_id,
                category="embedding"
            )
        
        return embeddings_list
    
    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension