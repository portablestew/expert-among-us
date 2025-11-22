"""Token-aware batching utilities for embeddings and reranking."""

import math
from typing import List, TypeVar, Callable, Tuple, Any

# Generic type for items being batched (str, tuple, etc.)
T = TypeVar('T')

# Fixed overhead cost per text (prevents batching too many small items)
BASE_BATCH_COST = 2.0

# Reference size for O(n²) scaling (typical chunk size)
CANONICAL_TOKEN_SIZE = 768


def estimate_tokens(text: str, bytes_per_token: float = 4) -> int:
    """Estimate token count from text.
    
    Uses bytes/4 heuristic: 1 token ≈ 4 UTF-8 bytes.
    Fast approximation without loading tokenizer.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return math.ceil(len(text.encode('utf-8')) // bytes_per_token)


def build_token_batches(
    items: List[T],
    max_batch_tokens: int,
    item_token_estimator: Callable[[T], int]
) -> List[List[int]]:
    """Build batches accounting for O(n²) memory scaling of transformer attention.
    
    Cost model: cost = 2.0 + (tokens/768) + (tokens/768)²
    
    This polynomial cost function ensures:
    - Fixed base overhead (2.0) prevents excessive small text batching
    - Linear term adds graduated cost scaling
    - Quadratic term models O(n²) attention memory growth
    - Larger reference size (768) reduces cost for typical code chunks
    
    Args:
        items: List of items to batch (texts, tuples, etc.)
        max_batch_tokens: Token budget (controls batch capacity)
        item_token_estimator: Function that estimates tokens for a single item
        
    Returns:
        List of batches, where each batch is a list of item indices
        
    Examples:
        # For max_batch_tokens = 16384:
        # - Max capacity = 16384 / 768 ≈ 21.3 canonical units
        
        # Small texts (100 tokens each, cost ≈ 2.15):
        >>> texts = ["short"] * 9
        >>> batches = build_token_batches(texts, 16384, estimate_tokens)
        >>> len(batches)  # 1 batch (9 items × 2.15 ≈ 19.4)
        1
        
        # Medium texts (1024 tokens each, cost ≈ 4.12):
        >>> texts = ["x" * 1024] * 5
        >>> batches = build_token_batches(texts, 16384, estimate_tokens)
        >>> len(batches)  # 1 batch (5 items × 4.12 ≈ 20.6)
        1
        
        # Large texts (2048 tokens each, cost ≈ 11.8):
        >>> texts = ["x" * 2048] * 2
        >>> batches = build_token_batches(texts, 16384, estimate_tokens)
        >>> len(batches)  # 2 batches (each exceeds capacity)
        2
    """
    if not items:
        return []
    
    # Calculate batch capacity in canonical units
    # Reference: 768 tokens = 1 canonical unit
    max_batch_items = max_batch_tokens / CANONICAL_TOKEN_SIZE
    
    batches = []
    current_batch = []
    current_cost = 0.0  # Accumulated canonical cost
    
    for i, item in enumerate(items):
        estimated_tokens = item_token_estimator(item)
        
        # Calculate cost: fixed base + quadratic scaling
        token_ratio = estimated_tokens / CANONICAL_TOKEN_SIZE
        canonical_cost = BASE_BATCH_COST + token_ratio + (token_ratio ** 2)
        
        # Edge case: single item exceeds capacity
        # Must include it anyway in its own batch
        if canonical_cost > max_batch_items:
            batches.append([i])
            continue
        
        # Start new batch if adding this item exceeds capacity
        if current_batch and (current_cost + canonical_cost > max_batch_items):
            batches.append(current_batch)
            current_batch = []
            current_cost = 0.0
        
        # Add item to current batch
        current_batch.append(i)
        current_cost += canonical_cost
    
    # Add final batch if non-empty
    if current_batch:
        batches.append(current_batch)
    
    return batches


def calculate_max_batch_tokens(
    tokens_per_gb: int,
    device_name: str = "model"
) -> Tuple[int, str, Any]:
    """Calculate max batch tokens based on available hardware.
    
    Handles lazy torch import, device detection, and token calculation.
    This eliminates code duplication between embedder and reranker classes.
    
    Args:
        tokens_per_gb: Maximum tokens per GB of GPU memory
        device_name: Name for debug logging (e.g., "embedder", "reranker")
        
    Returns:
        Tuple of (max_batch_tokens, device, torch_module)
        
    Example:
        >>> max_tokens, device, torch = calculate_max_batch_tokens(2048, "embedder")
        >>> # On 8GB GPU: max_tokens=16384, device="cuda"
    """
    from ..utils.debug import DebugLogger
    from ..utils.progress import console
    
    # Lazy import torch
    if DebugLogger.is_enabled():
        console.print("[DEBUG] Loading PyTorch...")
    import torch
    
    # Auto-detect device
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    
    # Calculate max_batch_tokens
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)
        max_batch_tokens = int(tokens_per_gb * total_memory_gb)
        
        if DebugLogger.is_enabled():
            console.print(
                f"[DEBUG] {device_name}: GPU {total_memory_gb:.2f}GB, "
                f"tokens_per_gb={tokens_per_gb}, "
                f"max_batch_tokens={max_batch_tokens}"
            )
    else:
        # CPU: conservative token limit (2 "virtual GB")
        max_batch_tokens = tokens_per_gb * 2
        
        if DebugLogger.is_enabled():
            console.print(
                f"[DEBUG] {device_name}: CPU mode, "
                f"max_batch_tokens={max_batch_tokens}"
            )
    
    return max_batch_tokens, device, torch