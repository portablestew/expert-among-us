"""Context manager for expert operations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from expert_among_us.config.settings import Settings
from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.db.vector.chroma import ChromaVectorDB
from expert_among_us.embeddings.base import Embedder
from expert_among_us.llm.base import LLMProvider


@dataclass
class ExpertContext:
    """Manages all dependencies for expert operations.
    
    This context manager provides lazy-initialized access to all components
    needed for expert operations (embedders, databases, LLM providers, etc.).
    Components are only created when accessed for the first time.
    
    Example:
        ```python
        ctx = ExpertContext(
            expert_name="MyExpert",
            data_dir=Path("/custom/path"),
            embedding_provider="local"
        )
        
        try:
            # Components are initialized on first access
            if not ctx.metadata_db.exists():
                raise ValueError("Expert does not exist")
            
            # Use the components
            results = searcher.search(params)
        finally:
            # Clean up resources
            ctx.close()
        ```
    
    Attributes:
        expert_name: Name of the expert
        data_dir: Optional custom data directory
        embedding_provider: Embedding provider ("local" or "bedrock")
        llm_provider: LLM provider ("auto", "openai", etc.)
        debug: Enable debug logging
    """
    
    expert_name: str
    data_dir: Optional[Path] = None
    embedding_provider: str = "local"
    llm_provider: str = "auto"
    debug: bool = False
    
    # Lazy-initialized components (use field with init=False)
    _settings: Optional[Settings] = field(default=None, init=False, repr=False)
    _embedder: Optional[Embedder] = field(default=None, init=False, repr=False)
    _metadata_db: Optional[SQLiteMetadataDB] = field(default=None, init=False, repr=False)
    _vector_db: Optional[ChromaVectorDB] = field(default=None, init=False, repr=False)
    _llm: Optional[LLMProvider] = field(default=None, init=False, repr=False)
    
    @property
    def settings(self) -> Settings:
        """Lazy-load settings.
        
        Returns:
            Settings instance with configuration for this expert
        """
        if self._settings is None:
            settings_kwargs = {
                'embedding_provider': self.embedding_provider,
                'llm_provider': self.llm_provider,
            }
            if self.data_dir is not None:
                settings_kwargs['data_dir'] = self.data_dir
            
            self._settings = Settings(**settings_kwargs)
        return self._settings
    
    @property
    def embedder(self) -> Embedder:
        """Lazy-load embedder.
        
        Returns:
            Embedder instance for the configured provider
        """
        if self._embedder is None:
            # Use the shared embeddings factory to avoid duplicating provider logic
            from expert_among_us.embeddings.factory import create_embedder
            self._embedder = create_embedder(self.settings)
        return self._embedder
    
    @property
    def metadata_db(self) -> SQLiteMetadataDB:
        """Lazy-load metadata database.
        
        Returns:
            SQLiteMetadataDB instance for this expert
        """
        if self._metadata_db is None:
            self._metadata_db = SQLiteMetadataDB(
                self.expert_name,
                data_dir=self.data_dir
            )
        return self._metadata_db
    
    @property
    def vector_db(self) -> ChromaVectorDB:
        """Lazy-load vector database.
        
        Returns:
            ChromaVectorDB instance for this expert
        """
        if self._vector_db is None:
            self._vector_db = ChromaVectorDB(
                self.expert_name,
                data_dir=self.data_dir
            )
        return self._vector_db
    
    @property
    def llm(self) -> LLMProvider:
        """Lazy-load LLM provider.
        
        Returns:
            LLM provider instance for the configured provider
        """
        if self._llm is None:
            from expert_among_us.llm.factory import create_llm_provider
            self._llm = create_llm_provider(self.settings, debug=self.debug)
        return self._llm
    
    def close(self) -> None:
        """Clean up resources.
        
        Closes all initialized database connections and releases resources.
        Should be called when the context is no longer needed.
        """
        if self._metadata_db is not None:
            self._metadata_db.close()
            self._metadata_db = None
        
        if self._vector_db is not None:
            self._vector_db.close()
            self._vector_db = None
    
    def __enter__(self):
        """Context manager entry.
        
        Returns:
            Self for use in with statement
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
            
        Returns:
            False to propagate any exception
        """
        self.close()
        return False