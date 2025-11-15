from typing import Optional
from .base import Reranker
from .local import LocalCrossEncoderReranker

def create_reranker(settings) -> Optional[Reranker]:
    """Create reranker instance based on settings.
    
    Args:
        settings: Settings object with reranking configuration
        
    Returns:
        Reranker instance or None if reranking disabled
    """
    if not settings.enable_reranking:
        return None
    
    if settings.reranking_provider == "local":
        return LocalCrossEncoderReranker(
            model_id=settings.cross_encoder_model,
            tokens_per_gb=settings.reranking_tokens_per_gb
        )
    
    raise ValueError(f"Unknown reranking provider: {settings.reranking_provider}")