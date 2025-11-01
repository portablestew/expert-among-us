"""Configuration management for Expert Among Us (stub for Phase 1)."""

from pathlib import Path
from typing import Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


# Model identifiers
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Provider-specific model defaults
PROVIDER_MODEL_DEFAULTS = {
    "bedrock": {
        "expert": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "promptgen": "us.amazon.nova-lite-v1:0",
    },
    "openai": {
        "expert": "gpt-5",
        "promptgen": "gpt-5-mini",
    },
    "openrouter": {
        "expert": "minimax/minimax-m2:free",
        "promptgen": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "ollama": {
        "expert": "deepseek-coder-v2:16b",
        "promptgen": "qwen2.5-coder:7b",
    },
    "claude-code": {
        "expert": "claude-sonnet-4-5",
        "promptgen": "claude-haiku-4-5",
    },
}

# Embedding model characteristics - Bedrock Titan
TITAN_EMBEDDING_DIMENSION = 1024  # Titan embedding vector dimension
TITAN_MAX_EMBEDDING_TOKENS = 8000  # Titan embedding model token limit

# Embedding model characteristics - Local Jina Code
JINA_CODE_MODEL_ID = "jinaai/jina-code-embeddings-0.5b"
JINA_CODE_DIMENSION = 512  # Matryoshka truncation from 896
JINA_CODE_MAX_TOKENS = 32768  # Jina Code model token limit


class Settings(BaseSettings):
    """Application settings (stub for Phase 1).
    
    This is a placeholder that will be fully implemented in later phases
    with proper configuration loading from files and environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )
    
    # Storage base directory
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".expert-among-us")
    
    # AWS settings
    aws_region: str = "us-west-2"
    aws_profile: Optional[str] = None
    
    # Provider settings
    embedding_provider: str = "local"  # "local" or "bedrock"
    llm_provider: str = "auto"  # "auto", "openai", "openrouter", "ollama", "bedrock", "claude-code"
    base_url_override: Optional[str] = None  # Override base URL for OpenAI-compatible providers
    
    # LLM provider settings
    bedrock_enable_caching: bool = True
    claude_code_model: str = "claude-sonnet-4-5"
    claude_code_cli_path: str = "claude"
    claude_code_session_dir: Optional[Path] = None
    
    # OpenAI-compatible API keys
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key")
    )
    openrouter_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "openrouter_api_key")
    )
    
    # OpenAI-compatible endpoints
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    
    # Bedrock region (alias for aws_region for LLM factory)
    @property
    def bedrock_region(self) -> str:
        """Get Bedrock region (uses aws_region)."""
        return self.aws_region
    
    # Model settings
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_ID
    
    # Model overrides (set via CLI)
    expert_model_override: Optional[str] = None
    promptgen_model_override: Optional[str] = None
    
    @property
    def expert_model(self) -> str:
        """Get expert model for current provider.
        
        Returns the CLI override if set, otherwise the provider default.
        
        Raises:
            ValueError: If no LLM provider is configured
            KeyError: If provider is not recognized
        """
        if self.expert_model_override:
            return self.expert_model_override
        
        if not self.llm_provider:
            raise ValueError("No LLM provider configured. Use --llm-provider.")
        
        if self.llm_provider not in PROVIDER_MODEL_DEFAULTS:
            raise KeyError(f"Unknown provider: {self.llm_provider}")
        
        return PROVIDER_MODEL_DEFAULTS[self.llm_provider]["expert"]
    
    @property
    def promptgen_model(self) -> str:
        """Get promptgen model for current provider.
        
        Returns the CLI override if set, otherwise the provider default.
        
        Raises:
            ValueError: If no LLM provider is configured
            KeyError: If provider is not recognized
        """
        if self.promptgen_model_override:
            return self.promptgen_model_override
        
        if not self.llm_provider:
            raise ValueError("No LLM provider configured. Use --llm-provider.")
        
        if self.llm_provider not in PROVIDER_MODEL_DEFAULTS:
            raise KeyError(f"Unknown provider: {self.llm_provider}")
        
        return PROVIDER_MODEL_DEFAULTS[self.llm_provider]["promptgen"]
    
    # Local embedding settings
    local_embedding_model: str = JINA_CODE_MODEL_ID
    local_embedding_dimension: int = JINA_CODE_DIMENSION
    local_embedding_max_tokens: int = JINA_CODE_MAX_TOKENS
    
    # Bedrock embedding settings
    bedrock_embedding_dimension: int = TITAN_EMBEDDING_DIMENSION
    bedrock_embedding_max_tokens: int = TITAN_MAX_EMBEDDING_TOKENS
    
    # Limits
    max_commits: int = 50000
    max_metadata_embedding_size_bytes: int = 20000  # Maximum bytes for metadata embeddings (20KB)
    max_embedding_text_size_bytes: int = 100000  # Maximum bytes for diff before chunking (100KB)
    diff_chunk_size_bytes: int = 8192  # Chunk size for diff embeddings (8KB)
    max_tokens_impostor: int = 8000
    max_tokens_prompt_gen: int = 1000
    max_diff_chars_for_llm: int = 80000  # Maximum diff characters to send to LLM (80KB)
    
    # Indexing
    embed_diffs: bool = True
    embed_metadata: bool = True
    
    # Debug settings
    debug: bool = False
    
    @property
    def storage_dir(self) -> Path:
        """Get the storage directory."""
        return self.data_dir / "data"
    
    @property
    def debug_log_dir(self) -> Path:
        """Get the debug log directory."""
        return self.data_dir / "logs"
    