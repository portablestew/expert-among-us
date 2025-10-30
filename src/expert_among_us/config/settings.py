"""Configuration management for Expert Among Us (stub for Phase 1)."""

from pathlib import Path
from typing import Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


# Model identifiers - centralized source of truth for all model IDs
CLAUDE_SONNET_MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
NOVA_LITE_MODEL_ID = "us.amazon.nova-lite-v1:0"
TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

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
    llm_provider: Optional[str] = None  # "openai", "openrouter", "local", "bedrock", "claude-code"
    
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
    local_llm_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LOCAL_LLM_BASE_URL", "local_llm_base_url")
    )
    
    # Bedrock region (alias for aws_region for LLM factory)
    @property
    def bedrock_region(self) -> str:
        """Get Bedrock region (uses aws_region)."""
        return self.aws_region
    
    # OpenRouter settings
    openrouter_app_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_APP_NAME", "openrouter_app_name")
    )
    openrouter_site_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_SITE_URL", "openrouter_site_url")
    )
    
    # Model settings - use centralized constants (Bedrock defaults)
    embedding_model: str = TITAN_EMBEDDING_MODEL_ID
    _promptgen_model: str = NOVA_LITE_MODEL_ID
    _expert_model: str = CLAUDE_SONNET_MODEL_ID
    
    # Provider-specific model defaults
    openai_expert_model: str = "gpt-5"
    openai_promptgen_model: str = "gpt-5-mini"
    claude_code_promptgen_model: str = "haiku-4-5"
    
    @property
    def promptgen_model(self) -> str:
        """Get promptgen model based on provider.
        
        Returns provider-specific default model for prompt generation.
        """
        if self.llm_provider == "claude-code":
            return self.claude_code_promptgen_model
        elif self.llm_provider in ("openai", "openrouter", "local"):
            return self.openai_promptgen_model
        else:
            # Bedrock or unspecified - use Bedrock default
            return self._promptgen_model
    
    @property
    def expert_model(self) -> str:
        """Get expert model based on provider.
        
        Returns provider-specific default model for expert responses.
        """
        if self.llm_provider in ("openai", "openrouter", "local"):
            return self.openai_expert_model
        else:
            # Bedrock, claude-code, or unspecified - use Bedrock default
            return self._expert_model
    
    # Local embedding settings
    local_embedding_model: str = JINA_CODE_MODEL_ID
    local_embedding_dimension: int = JINA_CODE_DIMENSION
    local_embedding_max_tokens: int = JINA_CODE_MAX_TOKENS
    
    # Bedrock embedding settings
    bedrock_embedding_dimension: int = TITAN_EMBEDDING_DIMENSION
    bedrock_embedding_max_tokens: int = TITAN_MAX_EMBEDDING_TOKENS
    
    # Limits
    max_commits: int = 10000
    max_diff_size_bytes: int = 100000
    max_embedding_text_size_bytes: int = 30000
    max_tokens_impostor: int = 8000
    max_tokens_prompt_gen: int = 1000
    
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
    
    @property
    def prompt_generation_model(self) -> str:
        """Backward compatibility alias (deprecated)."""
        return self.promptgen_model
    
    @property
    def impostor_model(self) -> str:
        """Backward compatibility alias (deprecated)."""
        return self.expert_model