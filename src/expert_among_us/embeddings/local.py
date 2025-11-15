from .base import Embedder
from typing import List, Optional, Callable
from ..utils.debug import DebugLogger
from ..utils.symbols import SYMBOLS
from ..utils.progress import console
from ..utils.batching import build_token_batches, estimate_tokens, calculate_max_batch_tokens
import gc

torch_import = None
sentence_transformers_import = None

class JinaCodeEmbedder(Embedder):
    """Local embeddings using Jina Code model with code2code task."""
    
    def __init__(self, model_id: str, dimension: int = 512, compile_model: bool = True, tokens_per_gb: int = 2048, enable_multiprocessing: bool = True):
        """Initialize the Jina Code embedder.
        
        Args:
            model_id: Hugging Face model identifier
            dimension: Target dimension for Matryoshka truncation
            compile_model: Whether to use torch.compile for optimization (default: True)
            tokens_per_gb: Maximum tokens per batch per GB of GPU memory (default: 2048)
            enable_multiprocessing: Whether to enable multiprocessing (default: True)
        """
        self.model_id = model_id
        self._dimension = dimension
        self.task = "code2code"
        self.task_prefix = "Represent this code for retrieving similar code: "
        
        # Use shared utility for device detection and batch token calculation
        self.max_batch_tokens, self.device, torch = calculate_max_batch_tokens(
            tokens_per_gb,
            device_name="embedder"
        )
        
        # Get GPU count for diagnostics and multiprocessing
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if DebugLogger.is_enabled():
            console.print("[DEBUG] Loading sentence_transformers...")
        
        # Lazy import sentence_transformers (heavy import ~3.7s)
        global sentence_transformers_import
        if sentence_transformers_import is None:
            sentence_transformers_import = __import__("sentence_transformers")
        sentence_transformers = sentence_transformers_import
        
        # Detailed GPU diagnostics
        if gpu_count == 0:
            console.print(f"{SYMBOLS['warning']}  GPU not detected - using CPU")
            console.print(f"   PyTorch version: {torch.__version__}")
            console.print(f"   CUDA GPUs available: {gpu_count}")
            console.print(f"   CUDA built: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
            console.print("\n   To enable GPU (run after uv sync):")
            console.print("   1. Check NVIDIA drivers: nvidia-smi")
            console.print("   2. Install PyTorch with CUDA: ./install-gpu.sh or ./install-gpu.ps1")
            console.print("   3. Restart your terminal or reactivate venv")
            console.print()
        
        # Load model with trust_remote_code for FlashAttention2 and fp16 for GPU optimization
        self.model = sentence_transformers.SentenceTransformer(
            model_id,
            trust_remote_code=True,
            device=self.device,
            model_kwargs={'dtype': torch.float16}
        )
        
        # Start multi-process pool for CPU/GPU/multi-GPU support
        # Automatically handles single GPU, multi-GPU, and multi-CPU scenarios
        
        # Start multi-process pool only if enabled in settings
        if enable_multiprocessing and gpu_count > 1:
            try:
                self.pool = self.model.start_multi_process_pool([f"cuda:{i}" for i in range(gpu_count)])
                console.print(f"{SYMBOLS['info']} Multiprocessing enabled: {gpu_count} GPUs")
            except Exception as e:
                # If multiprocessing fails, fall back to single-process mode
                self.pool = None
                if DebugLogger.is_enabled():
                    console.print(f"[DEBUG] Multiprocessing failed ({type(e).__name__}), using single-process mode")
        else:
            self.pool = None
        
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
            # List all GPUs that will be used (single or multi-GPU)
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            gpu_list = ", ".join(gpu_names)
            compile_status = " (compiled)" if compilation_enabled else " (not compiled)"
            console.print(f"{SYMBOLS['success']} Local embeddings using {gpu_count} GPUs: {gpu_list}{compile_status}")
        else:
            console.print(f"{SYMBOLS['info']} Local embeddings using CPU (this will be slower)")
        
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
    
    def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[float]]:
        """Generate embeddings with token-aware dynamic batching.
        
        Batches are built dynamically based on estimated token counts to
        prevent OOM errors with variable-length inputs.
        
        Args:
            texts: List of input texts to embed
            progress_callback: Optional callback(current, total) called after each batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Add task prefix to all texts
        prefixed_texts = [self.task_prefix + text for text in texts]
        
        # Log request if debug enabled
        request_id = None
        if DebugLogger.is_enabled():
            request = {
                "model_id": self.model_id,
                "task": self.task,
                "dimension": self._dimension,
                "max_batch_tokens": self.max_batch_tokens,
                "text_count": len(texts)
            }
            request_id = DebugLogger.log_request(
                "local",
                request,
                category="embedding"
            )
        
        # Build token-aware batches using shared utility
        batches = build_token_batches(
            prefixed_texts,
            self.max_batch_tokens,
            item_token_estimator=estimate_tokens
        )
        
        # Process each batch
        embeddings_list = []
        total_texts = len(prefixed_texts)
        processed = 0
        
        with self.torch.no_grad():
            for batch_indices in batches:
                batch_texts = [prefixed_texts[i] for i in batch_indices]
                
                # Encode batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    pool=self.pool,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Truncate to target dimension and convert to lists
                for emb in batch_embeddings:
                    embeddings_list.append(emb[:self._dimension].tolist())
                
                # Update progress
                processed += len(batch_texts)
                if progress_callback:
                    progress_callback(processed, total_texts)
                
                # Clear GPU memory
                del batch_embeddings
                gc.collect()
                if self.device == "cuda":
                    self.torch.cuda.empty_cache()
        
        # Log response if debug enabled
        if DebugLogger.is_enabled():
            DebugLogger.log_response(
                "local",
                {
                    "embeddings_count": len(embeddings_list),
                    "batches_used": len(batches),
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