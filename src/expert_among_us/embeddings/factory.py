from typing import Literal

from .base import Embedder
from .bedrock import BedrockEmbedder
from .local import JinaCodeEmbedder
from ..config.settings import Settings


def create_embedder(settings: Settings) -> Embedder:
    """
    Create an Embedder implementation based on settings.

    Provider is selected via:
      - settings.embedding_provider: "local" or "bedrock"
      - settings.embedding_model: model identifier for the provider
        (for bedrock this is the Bedrock modelId, for local this may be ignored
        or used by JinaCodeEmbedder depending on implementation).
    """
    provider: Literal["local", "bedrock"] = settings.embedding_provider  # type: ignore[assignment]

    if provider == "local":
        # JinaCodeEmbedder is responsible for using the configured local model.
        # We pass settings so it can use local_embedding_model / related config.
        return JinaCodeEmbedder(
            model_id=settings.local_embedding_model,
            dimension=settings.local_embedding_dimension,
            batch_size=settings.embedding_batch_size,
        )

    if provider == "bedrock":
        # Use the configured Bedrock embedding model ID.
        # Default comes from DEFAULT_EMBEDDING_MODEL_ID in settings.
        return BedrockEmbedder(model_id=settings.embedding_model)

    raise ValueError(
        f"Unknown embedding provider: {settings.embedding_provider}. "
        f"Expected one of: 'local', 'bedrock'."
    )