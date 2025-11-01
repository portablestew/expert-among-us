"""LLM provider factory for Expert Among Us."""

import shutil
from typing import Optional, Dict

from .base import LLMProvider
from .bedrock import BedrockLLM
from .claude_code import ClaudeCodeLLM
from .openai_compatible import OpenAICompatibleLLM
from ..config.settings import Settings


def create_llm_provider(settings: Settings, debug: bool = False) -> LLMProvider:
    """Create an LLM provider based on settings configuration.
    
    Supports auto-detection (default) or explicit provider selection:
    - "auto": Auto-detect available provider (default)
    - "openai": OpenAI API
    - "openrouter": OpenRouter API
    - "ollama": Ollama LLM server
    - "bedrock": AWS Bedrock
    - "claude-code": Claude Code CLI
    
    Args:
        settings: Application settings
        debug: Enable debug logging (default: False)
        
    Returns:
        Configured LLM provider instance
        
    Raises:
        ValueError: If no provider specified or required configuration missing
    """
    provider = settings.llm_provider
    
    # Handle auto-detection
    if provider == "auto":
        from .auto_detect import detect_llm_provider
        provider = detect_llm_provider()
        # Update settings object so properties can access the detected provider
        settings.llm_provider = provider
    
    if not provider:
        raise ValueError(
            "No LLM provider specified. Please use the --llm-provider argument.\n"
            "Available providers:\n"
            "  --llm-provider auto        (auto-detect, default)\n"
            "  --llm-provider openai      (requires OPENAI_API_KEY)\n"
            "  --llm-provider openrouter  (requires OPENROUTER_API_KEY)\n"
            "  --llm-provider ollama      (default: http://127.0.0.1:11434/v1)\n"
            "  --llm-provider bedrock     (requires AWS credentials)\n"
            "  --llm-provider claude-code (requires claude CLI)"
        )
    
    # Handle each provider with validation
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI provider requires OPENAI_API_KEY environment variable to be set.\n"
                "Please set your OpenAI API key:\n"
                "  export OPENAI_API_KEY=your-key-here"
            )
        # Use base_url_override if provided, otherwise use default (None = OpenAI default)
        base_url = settings.base_url_override if settings.base_url_override else None
        return OpenAICompatibleLLM(
            api_key=settings.openai_api_key,
            model="not-used",  # Model specified per-call
            base_url=base_url,
            debug=debug,
        )
    
    elif provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                "OpenRouter provider requires OPENROUTER_API_KEY environment variable to be set.\n"
                "Please set your OpenRouter API key:\n"
                "  export OPENROUTER_API_KEY=your-key-here"
            )
        # Use base_url_override if provided, otherwise use OpenRouter default
        base_url = settings.base_url_override if settings.base_url_override else "https://openrouter.ai/api/v1"
        return OpenAICompatibleLLM(
            api_key=settings.openrouter_api_key,
            model="not-used",  # Model specified per-call
            base_url=base_url,
            debug=debug,
        )
    
    elif provider == "ollama":
        # Use base_url_override if provided, otherwise use Ollama default
        base_url = settings.base_url_override if settings.base_url_override else settings.ollama_base_url
        return OpenAICompatibleLLM(
            api_key="ollama",  # Ollama doesn't require real API keys
            model="not-used",  # Model specified per-call
            base_url=base_url,
            debug=debug,
        )
    
    elif provider == "bedrock":
        if not settings.bedrock_region:
            raise ValueError(
                "Bedrock provider requires AWS region to be configured.\n"
                "Please set the AWS_REGION environment variable or --aws-region flag:\n"
                "  export AWS_REGION=us-west-2"
            )
        return BedrockLLM(
            region_name=settings.bedrock_region,
            profile_name=settings.aws_profile,
            enable_caching=settings.bedrock_enable_caching,
        )
    
    elif provider == "claude-code":
        # Check if claude CLI is available
        if not shutil.which(settings.claude_code_cli_path):
            raise ValueError(
                f"Claude Code CLI not found at '{settings.claude_code_cli_path}'.\n"
                "Please install the Claude CLI:\n"
                "  https://www.anthropic.com/claude/download"
            )
        return ClaudeCodeLLM(
            cli_path=settings.claude_code_cli_path,
            project_dir=settings.claude_code_session_dir,
        )
    
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}\n"
            "Valid providers: openai, openrouter, ollama, bedrock, claude-code"
        )