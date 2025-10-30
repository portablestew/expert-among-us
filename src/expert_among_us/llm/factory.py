"""LLM provider factory for Expert Among Us."""

import shutil
from typing import Optional, Dict

from .base import LLMProvider
from .bedrock import BedrockLLM
from .claude_code import ClaudeCodeLLM
from .openai_compatible import OpenAICompatibleLLM
from ..config.settings import Settings


def create_llm_provider(settings: Settings, debug: bool = False) -> LLMProvider:
    """Create an LLM provider based on explicit settings configuration.
    
    Requires an explicit --llm-provider argument to be set. Supports:
    - "openai": OpenAI API
    - "openrouter": OpenRouter API
    - "local": Local LLM server
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
    # Require explicit provider specification
    provider = settings.llm_provider
    if not provider:
        raise ValueError(
            "No LLM provider specified. Please use the --llm-provider argument.\n"
            "Available providers:\n"
            "  --llm-provider openai      (requires OPENAI_API_KEY)\n"
            "  --llm-provider openrouter  (requires OPENROUTER_API_KEY)\n"
            "  --llm-provider local       (requires LOCAL_LLM_BASE_URL)\n"
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
        return OpenAICompatibleLLM(
            api_key=settings.openai_api_key,
            model=settings.openai_expert_model,
            debug=debug,
        )
    
    elif provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                "OpenRouter provider requires OPENROUTER_API_KEY environment variable to be set.\n"
                "Please set your OpenRouter API key:\n"
                "  export OPENROUTER_API_KEY=your-key-here"
            )
        extra_headers: Dict[str, str] = {}
        if settings.openrouter_site_url:
            extra_headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            extra_headers["X-Title"] = settings.openrouter_app_name
        
        return OpenAICompatibleLLM(
            api_key=settings.openrouter_api_key,
            model=settings.openai_expert_model,
            base_url="https://openrouter.ai/api/v1",
            extra_headers=extra_headers if extra_headers else None,
            debug=debug,
        )
    
    elif provider == "local":
        if not settings.local_llm_base_url:
            raise ValueError(
                "Local LLM provider requires LOCAL_LLM_BASE_URL environment variable to be set.\n"
                "Please set your local LLM server URL:\n"
                "  export LOCAL_LLM_BASE_URL=http://localhost:1234/v1"
            )
        return OpenAICompatibleLLM(
            api_key="local",  # Local LLMs often don't require real API keys
            model=settings.openai_expert_model,
            base_url=settings.local_llm_base_url,
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
            "Valid providers: openai, openrouter, local, bedrock, claude-code"
        )