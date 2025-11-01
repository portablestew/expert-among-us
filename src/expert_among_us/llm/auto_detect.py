"""Auto-detection logic for LLM providers."""

import os
import shutil
from typing import List

import httpx


def detect_llm_provider() -> str:
    """Auto-detect available LLM provider using waterfall approach.
    
    Detection order:
    1. Environment variable-based providers (must be exactly one)
       - OPENAI_API_KEY -> openai
       - OPENROUTER_API_KEY -> openrouter
       - AWS_ACCESS_KEY_ID -> bedrock
    2. AWS Bedrock via boto3 default credentials
    3. Claude Code CLI on PATH
    4. Ollama running on localhost:11434
    5. Fail with error
    
    Returns:
        Provider name string
        
    Raises:
        ValueError: If no provider can be detected or multiple env-based providers exist
    """
    from ..utils.progress import log_info, console
    
    # Log that auto-detection is starting (without newline)
    console.print(f"[blue]ℹ[/blue] Auto-detecting LLM provider: ", end="")
    
    # Step 1: Check environment variables (must be exactly one)
    detected_env_providers: List[str] = []
    
    if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"):
        detected_env_providers.append("bedrock")
    if os.getenv("OPENROUTER_API_KEY"):
        detected_env_providers.append("openrouter")
    if os.getenv("OPENAI_API_KEY"):
        detected_env_providers.append("openai")
    
    if len(detected_env_providers) > 1:
        providers_str = ", ".join(detected_env_providers)
        raise ValueError(
            f"Multiple LLM providers detected via environment variables: {providers_str}. "
            f"Please specify one with --llm-provider. See README for details."
        )
    
    if len(detected_env_providers) == 1:
        provider = detected_env_providers[0]
        console.print(provider)
        return provider
    
    # Step 2: Check for default AWS profile
    if _check_aws_credentials():
        console.print("bedrock")
        return "bedrock"
    
    # Step 3: Check for claude CLI
    if shutil.which("claude"):
        console.print("claude-code")
        return "claude-code"
    
    # Step 4: Check for Ollama
    if _check_ollama_running():
        console.print("ollama")
        return "ollama"
    
    # Step 5: No provider found
    raise ValueError(
        "No LLM provider detected. Please configure a provider or specify with --llm-provider. "
        "See README for setup instructions."
    )


def _check_aws_credentials() -> bool:
    """Check if AWS credentials are available via boto3 default profile."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ProfileNotFound
        
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except (NoCredentialsError, ProfileNotFound):
        return False


def _check_ollama_running() -> bool:
    """Check if Ollama is running on default port."""
    try:
        response = httpx.get("http://127.0.0.1:11434/", timeout=2.0)
        return response.status_code == 200 and "Ollama is running" in response.text
    except (httpx.RequestError, httpx.TimeoutException):
        return False