"""Configuration management for Expert Among Us (stub for Phase 1)."""

from pathlib import Path
from typing import Optional


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


class Settings:
    """Application settings (stub for Phase 1).
    
    This is a placeholder that will be fully implemented in later phases
    with proper configuration loading from files and environment variables.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        llm_provider: str = "bedrock",
        embedding_provider: str = "local"
    ) -> None:
        """Initialize settings with defaults.
        
        Args:
            data_dir: Base directory for expert data storage (default: ~/.expert-among-us)
            llm_provider: LLM provider to use (default: "bedrock")
            embedding_provider: Embedding provider to use (default: "local")
        """
        # Storage base directory
        self.data_dir: Path = data_dir or Path.home() / ".expert-among-us"
        
        # AWS settings
        self.aws_region: str = "us-west-2"
        self.aws_profile: Optional[str] = None
        
        # Provider settings
        self.embedding_provider: str = embedding_provider  # "local" or "bedrock"
        self.llm_provider: str = llm_provider  # "bedrock" or other future providers
        
        # LLM provider settings
        self.bedrock_enable_caching: bool = True
        self.claude_code_model: str = "claude-sonnet-4-5"
        self.claude_code_cli_path: str = "claude"
        self.claude_code_session_dir: Optional[Path] = None
        
        # Model settings - use centralized constants
        self.embedding_model: str = TITAN_EMBEDDING_MODEL_ID
        self.promptgen_model: str = NOVA_LITE_MODEL_ID
        self.expert_model: str = CLAUDE_SONNET_MODEL_ID
        
        # Backward compatibility aliases (deprecated)
        self.prompt_generation_model: str = self.promptgen_model
        self.impostor_model: str = self.expert_model
        
        # Local embedding settings
        self.local_embedding_model: str = JINA_CODE_MODEL_ID
        self.local_embedding_dimension: int = JINA_CODE_DIMENSION
        self.local_embedding_max_tokens: int = JINA_CODE_MAX_TOKENS
        
        # Bedrock embedding settings
        self.bedrock_embedding_dimension: int = TITAN_EMBEDDING_DIMENSION
        self.bedrock_embedding_max_tokens: int = TITAN_MAX_EMBEDDING_TOKENS
        
        # Limits
        self.max_commits: int = 10000
        self.max_diff_size_bytes: int = 100000
        self.max_embedding_text_size_bytes: int = 30000
        self.max_tokens_impostor: int = 8000
        self.max_tokens_prompt_gen: int = 1000
        
        # Indexing
        self.embed_diffs: bool = True
        self.embed_metadata: bool = True
        
        # Storage
        self.storage_dir: Path = self.data_dir / "data"
        
        # Debug settings
        self.debug: bool = False
        self.debug_log_dir: Path = self.data_dir / "logs"