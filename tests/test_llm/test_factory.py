"""Tests for LLM provider factory function."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from expert_among_us.llm.factory import create_llm_provider
from expert_among_us.llm.base import LLMProvider
from expert_among_us.llm.openai_compatible import OpenAICompatibleLLM
from expert_among_us.llm.bedrock import BedrockLLM
from expert_among_us.llm.claude_code import ClaudeCodeLLM
from expert_among_us.config.settings import Settings


@pytest.fixture
def base_settings():
    """Create base settings with no provider configured."""
    settings = Settings(
        expert_model="gpt-4",
        openai_api_key=None,
        openrouter_api_key=None,
        local_llm_base_url=None,
        aws_profile=None,
        llm_provider=None
    )
    return settings


class TestOpenAIConfiguration:
    """Tests for OpenAI provider configuration with explicit selection."""
    
    @patch("expert_among_us.llm.factory.OpenAICompatibleLLM")
    def test_factory_creates_openai_with_explicit_provider(self, mock_openai_class, base_settings):
        """Test factory creates OpenAI provider when explicitly selected."""
        base_settings.llm_provider = "openai"
        base_settings.openai_api_key = "sk-test-key"
        
        provider = create_llm_provider(base_settings)
        
        # Verify OpenAICompatibleLLM was called with correct parameters
        mock_openai_class.assert_called_once_with(
            api_key="sk-test-key",
            model="not-used",  # Model specified per-call
            base_url=None,
            debug=False
        )
        assert provider == mock_openai_class.return_value
    
    @patch("expert_among_us.llm.factory.OpenAICompatibleLLM")
    def test_factory_with_openai_debug_enabled(self, mock_openai_class, base_settings):
        """Test factory propagates debug parameter for OpenAI."""
        base_settings.llm_provider = "openai"
        base_settings.openai_api_key = "sk-test-key"
        
        provider = create_llm_provider(base_settings, debug=True)
        
        mock_openai_class.assert_called_once_with(
            api_key="sk-test-key",
            model="not-used",  # Model specified per-call
            base_url=None,
            debug=True
        )
    
    @patch("expert_among_us.llm.factory.OpenAICompatibleLLM")
    def test_factory_with_openai_default_model(self, mock_openai_class, base_settings):
        """Test factory uses default_model if available."""
        base_settings.llm_provider = "openai"
        base_settings.openai_api_key = "sk-test-key"
        base_settings.default_model = "gpt-4-turbo"
        
        provider = create_llm_provider(base_settings)
        
        mock_openai_class.assert_called_once_with(
            api_key="sk-test-key",
            model="not-used",  # Model specified per-call
            base_url=None,
            debug=False
        )
    
    def test_openai_provider_raises_error_when_api_key_missing(self, base_settings):
        """Test OpenAI provider raises ValueError when API key is missing."""
        base_settings.llm_provider = "openai"
        base_settings.openai_api_key = None
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(base_settings)
        
        error_msg = str(exc_info.value)
        assert "OPENAI_API_KEY" in error_msg
        assert "environment variable" in error_msg


class TestOpenRouterConfiguration:
    """Tests for OpenRouter provider configuration with explicit selection."""
    
    @patch("expert_among_us.llm.factory.OpenAICompatibleLLM")
    def test_factory_with_openrouter_debug_enabled(self, mock_openai_class, base_settings):
        """Test factory propagates debug parameter for OpenRouter."""
        base_settings.llm_provider = "openrouter"
        base_settings.openrouter_api_key = "sk-or-test-key"
        
        provider = create_llm_provider(base_settings, debug=True)
        
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["debug"] is True
    
    def test_openrouter_provider_raises_error_when_api_key_missing(self, base_settings):
        """Test OpenRouter provider raises ValueError when API key is missing."""
        base_settings.llm_provider = "openrouter"
        base_settings.openrouter_api_key = None
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(base_settings)
        
        error_msg = str(exc_info.value)
        assert "OPENROUTER_API_KEY" in error_msg
        assert "environment variable" in error_msg


# TestLocalLLMConfiguration class removed - "local" provider no longer exists.
# Use "ollama" provider instead for local LLM testing.


class TestBedrockConfiguration:
    """Tests for Bedrock provider configuration with explicit selection."""
    
    @patch("expert_among_us.llm.factory.BedrockLLM")
    def test_factory_creates_bedrock_with_explicit_provider(self, mock_bedrock_class, base_settings):
        """Test factory creates Bedrock provider when explicitly selected."""
        base_settings.llm_provider = "bedrock"
        base_settings.aws_region = "us-west-2"
        
        provider = create_llm_provider(base_settings)
        
        # Verify BedrockLLM was called with correct parameters
        mock_bedrock_class.assert_called_once_with(
            region_name="us-west-2",
            profile_name=None,
            enable_caching=True
        )
        assert provider == mock_bedrock_class.return_value
        assert isinstance(mock_bedrock_class.return_value, type(mock_bedrock_class.return_value))
    
    @patch("expert_among_us.llm.factory.BedrockLLM")
    def test_factory_with_bedrock_custom_region(self, mock_bedrock_class, base_settings):
        """Test factory with Bedrock custom region."""
        base_settings.llm_provider = "bedrock"
        base_settings.aws_region = "us-east-1"
        
        provider = create_llm_provider(base_settings)
        
        mock_bedrock_class.assert_called_once_with(
            region_name="us-east-1",
            profile_name=None,
            enable_caching=True
        )
    
    @patch("expert_among_us.llm.factory.BedrockLLM")
    def test_factory_with_bedrock_profile(self, mock_bedrock_class, base_settings):
        """Test factory with Bedrock profile."""
        base_settings.llm_provider = "bedrock"
        base_settings.aws_region = "us-west-2"
        base_settings.aws_profile = "my-profile"
        
        provider = create_llm_provider(base_settings)
        
        mock_bedrock_class.assert_called_once_with(
            region_name="us-west-2",
            profile_name="my-profile",
            enable_caching=True
        )
    
    @patch("expert_among_us.llm.factory.BedrockLLM")
    def test_factory_with_bedrock_caching_disabled(self, mock_bedrock_class, base_settings):
        """Test factory with Bedrock caching disabled."""
        base_settings.llm_provider = "bedrock"
        base_settings.aws_region = "us-west-2"
        base_settings.bedrock_enable_caching = False
        
        provider = create_llm_provider(base_settings)
        
        mock_bedrock_class.assert_called_once_with(
            region_name="us-west-2",
            profile_name=None,
            enable_caching=False
        )
    
    @patch("expert_among_us.llm.factory.BedrockLLM")
    def test_factory_bedrock_ignores_debug(self, mock_bedrock_class, base_settings):
        """Test that debug parameter is not passed to Bedrock (it doesn't support it in __init__)."""
        base_settings.llm_provider = "bedrock"
        base_settings.aws_region = "us-west-2"
        
        provider = create_llm_provider(base_settings, debug=True)
        
        # BedrockLLM doesn't accept debug in __init__, only in methods
        call_kwargs = mock_bedrock_class.call_args[1]
        assert "debug" not in call_kwargs
    
    def test_bedrock_provider_raises_error_when_region_missing(self, base_settings):
        """Test Bedrock provider raises ValueError when region is missing."""
        base_settings.llm_provider = "bedrock"
        base_settings.aws_region = None
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(base_settings)
        
        error_msg = str(exc_info.value)
        assert "AWS region" in error_msg or "AWS_REGION" in error_msg


class TestClaudeCodeConfiguration:
    """Tests for Claude Code CLI provider configuration with explicit selection."""
    
    @patch("expert_among_us.llm.factory.shutil.which")
    @patch("expert_among_us.llm.factory.ClaudeCodeLLM")
    def test_factory_creates_claude_code_with_explicit_provider(self, mock_claude_class, mock_which, base_settings):
        """Test factory creates Claude Code provider when explicitly selected."""
        mock_which.return_value = "/usr/local/bin/claude"
        base_settings.llm_provider = "claude-code"
        
        provider = create_llm_provider(base_settings)
        
        # Verify ClaudeCodeLLM was called with correct parameters
        mock_claude_class.assert_called_once_with(
            cli_path="claude",
            project_dir=None
        )
        assert provider == mock_claude_class.return_value
    
    @patch("expert_among_us.llm.factory.shutil.which")
    @patch("expert_among_us.llm.factory.ClaudeCodeLLM")
    def test_factory_with_claude_code_custom_cli(self, mock_claude_class, mock_which, base_settings):
        """Test factory with custom Claude CLI path."""
        mock_which.return_value = "/custom/path/claude"
        base_settings.llm_provider = "claude-code"
        base_settings.claude_code_cli_path = "/custom/path/claude"
        
        provider = create_llm_provider(base_settings)
        
        mock_claude_class.assert_called_once_with(
            cli_path="/custom/path/claude",
            project_dir=None
        )
    
    @patch("expert_among_us.llm.factory.shutil.which")
    @patch("expert_among_us.llm.factory.ClaudeCodeLLM")
    def test_factory_with_claude_code_session_dir(self, mock_claude_class, mock_which, base_settings, tmp_path):
        """Test factory with custom Claude Code session directory."""
        mock_which.return_value = "/usr/local/bin/claude"
        base_settings.llm_provider = "claude-code"
        session_dir = tmp_path / "sessions"
        base_settings.claude_code_session_dir = session_dir
        
        provider = create_llm_provider(base_settings)
        
        mock_claude_class.assert_called_once_with(
            cli_path="claude",
            project_dir=session_dir
        )
    
    @patch("expert_among_us.llm.factory.shutil.which")
    @patch("expert_among_us.llm.factory.ClaudeCodeLLM")
    def test_factory_claude_code_ignores_debug(self, mock_claude_class, mock_which, base_settings):
        """Test that debug parameter is not passed to Claude Code (it doesn't support it in __init__)."""
        mock_which.return_value = "/usr/local/bin/claude"
        base_settings.llm_provider = "claude-code"
        
        provider = create_llm_provider(base_settings, debug=True)
        
        # ClaudeCodeLLM doesn't accept debug in __init__, only in methods
        call_kwargs = mock_claude_class.call_args[1]
        assert "debug" not in call_kwargs
    
    @patch("expert_among_us.llm.factory.shutil.which")
    def test_claude_code_provider_raises_error_when_cli_not_found(self, mock_which, base_settings):
        """Test Claude Code provider raises ValueError when CLI is not found."""
        mock_which.return_value = None
        base_settings.llm_provider = "claude-code"
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(base_settings)
        
        error_msg = str(exc_info.value)
        assert "Claude Code CLI not found" in error_msg
        assert "claude" in error_msg
    
class TestErrorCases:
    """Tests for error handling with explicit provider selection."""
    
    def test_factory_raises_error_when_no_provider_specified(self, base_settings):
        """Test factory raises ValueError when no provider is specified."""
        base_settings.llm_provider = None
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(base_settings)
        
        error_msg = str(exc_info.value)
        assert "No LLM provider specified" in error_msg
        assert "--llm-provider" in error_msg
        assert "openai" in error_msg
        assert "openrouter" in error_msg
        assert "ollama" in error_msg
        assert "bedrock" in error_msg
        assert "claude-code" in error_msg
    
    def test_factory_raises_error_for_unknown_provider(self, base_settings):
        """Test factory raises ValueError for unknown provider."""
        base_settings.llm_provider = "unknown-provider"
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(base_settings)
        
        error_msg = str(exc_info.value)
        assert "Unknown LLM provider" in error_msg
        assert "unknown-provider" in error_msg


class TestDebugParameterPropagation:
    """Tests for debug parameter propagation to providers."""
    
    @patch("expert_among_us.llm.factory.OpenAICompatibleLLM")
    def test_debug_propagates_to_openai(self, mock_openai, base_settings):
        """Test debug parameter propagates to OpenAI provider."""
        base_settings.llm_provider = "openai"
        base_settings.openai_api_key = "sk-test"
        
        create_llm_provider(base_settings, debug=True)
        
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["debug"] is True
    
    @patch("expert_among_us.llm.factory.OpenAICompatibleLLM")
    def test_debug_false_by_default(self, mock_openai, base_settings):
        """Test debug is False by default."""
        base_settings.llm_provider = "openai"
        base_settings.openai_api_key = "sk-test"
        
        create_llm_provider(base_settings)
        
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["debug"] is False
    
    @patch("expert_among_us.llm.factory.OpenAICompatibleLLM")
    def test_debug_propagates_to_openrouter(self, mock_openai, base_settings):
        """Test debug parameter propagates to OpenRouter provider."""
        base_settings.llm_provider = "openrouter"
        base_settings.openrouter_api_key = "sk-or-test"
        
        create_llm_provider(base_settings, debug=True)
        
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["debug"] is True
    
    # test_debug_propagates_to_local_llm removed - "local" provider no longer exists