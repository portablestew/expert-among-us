from .base import Embedder
from typing import List, Optional, Callable
from ..utils.debug import DebugLogger
from ..utils.symbols import SYMBOLS
from ..utils.progress import console
import gc

torch_import = None
sentence_transformers_import = None

class JinaCodeEmbedder(Embedder):
    """Local embeddings using Jina Code model with code2code task."""
    
    def __init__(self, model_id: str, dimension: int = 512, compile_model: bool = True, batch_per_gb: int = 1.0, enable_multiprocessing: bool = True):
        """Initialize the Jina Code embedder.
        
        Args:
            model_id: Hugging Face model identifier
            dimension: Target dimension for Matryoshka truncation
            compile_model: Whether to use torch.compile for optimization (default: True)
            batch_per_gb: Batch size per GB of GPU memory for inference (default: 1.0)
            enable_multiprocessing: Whether to enable multiprocessing (default: True)
        """
        # Log loading message in debug mode
        if DebugLogger.is_enabled():
            console.print("[DEBUG] Loading PyTorch...")
        
        # Lazy import torch (heavy import ~2.4s)
        global torch_import
        if torch_import is None:
            torch_import = __import__("torch")
        torch = torch_import
        
        if DebugLogger.is_enabled():
            console.print("[DEBUG] Loading sentence_transformers...")
        
        # Lazy import sentence_transformers (heavy import ~3.7s)
        global sentence_transformers_import
        if sentence_transformers_import is None:
            sentence_transformers_import = __import__("sentence_transformers")
        sentence_transformers = sentence_transformers_import
        
        self.model_id = model_id
        self._dimension = dimension
        self.task = "code2code"
        self.task_prefix = "Represent this code for retrieving similar code: "
        
        # Auto-detect GPU availability with diagnostics
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.device = "cuda" if gpu_count > 0 else "cpu"
        
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
        """Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts to embed
            progress_callback: Optional callback(current, total) called after each job batch
            
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
        
        embeddings_list = []
        total_texts = len(prefixed_texts)
        processed = 0
        
        # Process texts in small batches to avoid them hanging around in GPU memory
        with self.torch.no_grad():
            job_batch_size = self.batch_size * 8
            for i in range(0, len(prefixed_texts), job_batch_size):
                batch = prefixed_texts[i:i + job_batch_size]
                
                # Perform embedding jobs
                embeddings = self.model.encode(
                    batch,
                    pool=self.pool,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=not progress_callback  # Use built-in if no callback
                )
                
                # Truncate to target dimension (Matryoshka) and convert to lists
                embeddings_list.extend([emb[:self._dimension].tolist() for emb in embeddings])
                
                # Update progress after each job batch
                processed += len(batch)
                if progress_callback:
                    progress_callback(processed, total_texts)
                
                del embeddings
                gc.collect()
                if self.device == "cuda":
                    self.torch.cuda.empty_cache()
        
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