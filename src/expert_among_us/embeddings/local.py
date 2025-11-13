from .base import Embedder
from typing import List
from ..utils.debug import DebugLogger
from ..utils.symbols import SYMBOLS
from ..utils.progress import console


class JinaCodeEmbedder(Embedder):
    """Local embeddings using Jina Code model with code2code task."""
    
    def __init__(self, model_id: str, dimension: int = 512, compile_model: bool = True, batch_size: int = 12, enable_multiprocessing: bool = True):
        """Initialize the Jina Code embedder.
        
        Args:
            model_id: Hugging Face model identifier
            dimension: Target dimension for Matryoshka truncation
            compile_model: Whether to use torch.compile for optimization (default: True)
            batch_size: Batch size for GPU inference (default: 12)
            enable_multiprocessing: Whether to enable multiprocessing (default: True)
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
        self.batch_size = batch_size
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
        
        # Start multi-process pool for CPU/GPU/multi-GPU support
        # Automatically handles single GPU, multi-GPU, and multi-CPU scenarios
        
        # Start multi-process pool only if enabled in settings
        if enable_multiprocessing:
            try:
                self.pool = self.model.start_multi_process_pool()
            except Exception as e:
                # If multiprocessing fails, fall back to single-process mode
                self.pool = None
                if DebugLogger.is_enabled():
                    console.print(f"[DEBUG] Multiprocessing failed ({type(e).__name__}), using single-process mode")
        else:
            self.pool = None
            if DebugLogger.is_enabled():
                console.print(f"[DEBUG] Multiprocessing disabled in settings")
        
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
            gpu_count = torch.cuda.device_count() if enable_multiprocessing else 1
            # List all GPUs that will be used (single or multi-GPU)
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            gpu_list = ", ".join(gpu_names)
            compile_status = " (compiled)" if compilation_enabled else " (not compiled)"
            console.print(f"{SYMBOLS['success']} Local embeddings using {gpu_count} GPUs: {gpu_list}{compile_status}")
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
                "batch_size": self.batch_size,
                "text_lengths": [len(text) for text in texts]
            }
            request_id = DebugLogger.log_request(
                "local",
                request,
                category="embedding"
            )
        
        # Generate embeddings in batch using multi-process pool.
        # Automatically handles CPU/multi-CPU, single GPU, and multi-GPU scenarios.
        # We RE-ENABLE the internal tqdm progress bar here because batch embedding
        # progress is critical for UX. To keep output readable, rely on tqdm's
        # single-line behavior and avoid overlapping rich Progress for this section.
        
        if self.pool is not None:
            # Use multi-process pool if available
            embeddings = self.model.encode_multi_process(
                prefixed_texts,
                pool=self.pool,
                batch_size=self.batch_size,
                show_progress_bar=True
            )
        else:
            # Fallback to single-process mode
            if DebugLogger.is_enabled():
                console.print(f"[DEBUG] Using single-process encoding (multiprocessing unavailable)")
            embeddings = self.model.encode(
                prefixed_texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=self.batch_size
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
    
    def __del__(self):
        """Cleanup multi-process pool when object is destroyed."""
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                if DebugLogger.is_enabled():
                    console.print(f"[DEBUG] Stopping multi-process pool...")
                self.model.stop_multi_process_pool(self.pool)
                if DebugLogger.is_enabled():
                    console.print(f"[DEBUG] Multi-process pool stopped successfully")
            except Exception as e:
                # Ignore cleanup errors during shutdown
                if DebugLogger.is_enabled():
                    console.print(f"[DEBUG] Error stopping pool (ignored): {e}")
                pass
    
    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension