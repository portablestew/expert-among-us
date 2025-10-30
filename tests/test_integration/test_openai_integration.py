"""Integration tests for OpenAI-compatible LLM providers.

Tests the integration between Settings, Factory, and OpenAI-compatible providers.
Focuses on end-to-end flows with mocked external API calls.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path

from expert_among_us.config.settings import Settings
from expert_among_us.llm.factory import create_llm_provider
from expert_among_us.llm.openai_compatible import OpenAICompatibleLLM
from expert_among_us.llm.base import (
    Message,
    LLMResponse,
    StreamChunk,
    UsageMetrics,
    LLMError,
    LLMRateLimitError,
    LLMInvalidRequestError,
)
from openai import APIError, RateLimitError, AuthenticationError


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client to avoid real API calls."""
    with patch("expert_among_us.llm.openai_compatible.OpenAI") as mock:
        yield mock


@pytest.fixture
def base_settings():
    """Create base settings for testing."""
    return Settings(
        data_dir=Path("/tmp/test-expert"),
        aws_region="us-west-2",
        embedding_provider="local",
        llm_provider="openai"
    )


@pytest.fixture
def mock_successful_response():
    """Create a mock successful OpenAI response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-4"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 70
    return mock_response


@pytest.fixture
def mock_streaming_chunks():
    """Create mock streaming chunks."""
    chunks = [
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)]
        ),
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content=" world"), finish_reason=None)]
        ),
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content="!"), finish_reason=None)]
        ),
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content=None), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)
        )
    ]
    return chunks


class TestFactoryIntegration:
    """Tests for factory integration with settings and explicit provider selection."""
    
    def test_factory_creates_openai_provider_with_explicit_selection(self, mock_openai_client, base_settings):
        """Test factory creates OpenAI provider when explicitly selected."""
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        
        provider = create_llm_provider(settings, debug=False)
        
        assert isinstance(provider, OpenAICompatibleLLM)
        assert provider.model == settings.expert_model
        mock_openai_client.assert_called_once()
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-key"
        assert "base_url" not in call_kwargs
    
    def test_factory_creates_openrouter_provider_with_explicit_selection(self, mock_openai_client, base_settings):
        """Test factory creates OpenRouter provider when explicitly selected."""
        settings = base_settings.model_copy(update={
            "llm_provider": "openrouter",
            "openrouter_api_key": "sk-or-test-key",
            "openrouter_app_name": "Expert Among Us",
            "openrouter_site_url": "https://example.com"
        })
        
        provider = create_llm_provider(settings, debug=False)
        
        assert isinstance(provider, OpenAICompatibleLLM)
        mock_openai_client.assert_called_once()
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "sk-or-test-key"
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert "default_headers" in call_kwargs
        assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://example.com"
        assert call_kwargs["default_headers"]["X-Title"] == "Expert Among Us"
    
    def test_factory_creates_local_llm_provider_with_explicit_selection(self, mock_openai_client, base_settings):
        """Test factory creates local LLM provider when explicitly selected."""
        settings = base_settings.model_copy(update={
            "llm_provider": "local",
            "local_llm_base_url": "http://localhost:8000/v1"
        })
        
        provider = create_llm_provider(settings, debug=False)
        
        assert isinstance(provider, OpenAICompatibleLLM)
        mock_openai_client.assert_called_once()
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "local"
        assert call_kwargs["base_url"] == "http://localhost:8000/v1"
    
    def test_factory_raises_error_without_provider_selection(self, base_settings):
        """Test factory raises ValueError when no provider is selected."""
        settings = base_settings.model_copy(update={"llm_provider": None})
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(settings, debug=False)
        
        assert "No LLM provider specified" in str(exc_info.value)
        assert "--llm-provider" in str(exc_info.value)
    
    def test_openai_provider_raises_error_when_api_key_missing(self, base_settings):
        """Test OpenAI provider raises error when API key is missing."""
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": None
        })
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(settings, debug=False)
        
        assert "OPENAI_API_KEY" in str(exc_info.value)
    
    def test_openrouter_provider_raises_error_when_api_key_missing(self, base_settings):
        """Test OpenRouter provider raises error when API key is missing."""
        settings = base_settings.model_copy(update={
            "llm_provider": "openrouter",
            "openrouter_api_key": None
        })
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(settings, debug=False)
        
        assert "OPENROUTER_API_KEY" in str(exc_info.value)
    
    def test_local_provider_raises_error_when_base_url_missing(self, base_settings):
        """Test local provider raises error when base URL is missing."""
        settings = base_settings.model_copy(update={
            "llm_provider": "local",
            "local_llm_base_url": None
        })
        
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(settings, debug=False)
        
        assert "LOCAL_LLM_BASE_URL" in str(exc_info.value)
    
    def test_factory_with_debug_enabled(self, mock_openai_client, base_settings):
        """Test factory creates provider with debug enabled."""
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        
        provider = create_llm_provider(settings, debug=True)
        
        assert isinstance(provider, OpenAICompatibleLLM)
        assert provider.debug is True


class TestEndToEndProviderFlow:
    """Tests for end-to-end provider flows with mocking."""
    
    def test_openai_generate_flow(self, mock_openai_client, base_settings, mock_successful_response):
        """Test complete flow: Settings → Factory → Provider → Generate."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        # Configure settings
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        
        # Create provider through factory
        provider = create_llm_provider(settings)
        
        # Generate response
        messages = [Message(role="user", content="Hello, world!")]
        response = provider.generate(
            messages=messages,
            model="gpt-4",
            max_tokens=100,
            temperature=0.7
        )
        
        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response"
        assert response.model == "gpt-4"
        assert response.stop_reason == "stop"
        assert response.usage.input_tokens == 50
        assert response.usage.output_tokens == 20
        
        # Verify API was called correctly
        mock_client_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.7
    
    @pytest.mark.asyncio
    async def test_openai_stream_flow(self, mock_openai_client, base_settings, mock_streaming_chunks):
        """Test complete flow: Settings → Factory → Provider → Stream."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = iter(mock_streaming_chunks)
        
        # Configure settings
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        
        # Create provider through factory
        provider = create_llm_provider(settings)
        
        # Stream response
        messages = [Message(role="user", content="Tell me a story")]
        chunks = []
        async for chunk in provider.stream(
            messages=messages,
            model="gpt-4",
            max_tokens=200,
            temperature=0.8
        ):
            chunks.append(chunk)
        
        # Verify chunks
        assert len(chunks) == 4
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[2].delta == "!"
        assert chunks[3].delta == ""
        assert chunks[3].stop_reason == "stop"
        assert chunks[3].usage.input_tokens == 50
        
        # Verify API was called correctly
        mock_client_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
    
    def test_openrouter_with_headers_flow(self, mock_openai_client, base_settings, mock_successful_response):
        """Test OpenRouter flow with custom headers."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        # Configure settings with OpenRouter
        settings = base_settings.model_copy(update={
            "llm_provider": "openrouter",
            "openrouter_api_key": "sk-or-key",
            "openrouter_app_name": "Test App",
            "openrouter_site_url": "https://test.com"
        })
        
        # Create provider and generate
        provider = create_llm_provider(settings)
        messages = [Message(role="user", content="Test")]
        response = provider.generate(messages=messages, model="anthropic/claude-3-opus")
        
        # Verify headers were configured
        call_kwargs = mock_openai_client.call_args[1]
        assert "default_headers" in call_kwargs
        headers = call_kwargs["default_headers"]
        assert headers["HTTP-Referer"] == "https://test.com"
        assert headers["X-Title"] == "Test App"
        
        # Verify response
        assert response.content == "This is a test response"
    
    def test_local_llm_flow(self, mock_openai_client, base_settings, mock_successful_response):
        """Test local LLM flow."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        # Configure settings for local LLM
        settings = base_settings.model_copy(update={
            "llm_provider": "local",
            "local_llm_base_url": "http://localhost:11434/v1"
        })
        
        # Create provider and generate
        provider = create_llm_provider(settings)
        messages = [Message(role="user", content="Test local model")]
        response = provider.generate(messages=messages, model="llama-3-70b")
        
        # Verify configuration
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "local"
        assert call_kwargs["base_url"] == "http://localhost:11434/v1"
        
        # Verify response
        assert response.content == "This is a test response"
    
    def test_generate_with_system_prompt_flow(self, mock_openai_client, base_settings, mock_successful_response):
        """Test generate with system prompt through full flow."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Generate with system prompt
        messages = [Message(role="user", content="Hello")]
        response = provider.generate(
            messages=messages,
            model="gpt-4",
            system="You are a helpful assistant"
        )
        
        # Verify system prompt was added
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        api_messages = call_kwargs["messages"]
        assert api_messages[0]["role"] == "system"
        assert api_messages[0]["content"] == "You are a helpful assistant"
        assert api_messages[1]["role"] == "user"


class TestErrorHandlingIntegration:
    """Tests for error handling across the full chain."""
    
    def test_rate_limit_error_propagation(self, mock_openai_client, base_settings):
        """Test rate limit error propagates correctly through the chain."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        
        # Create a mock response with request attribute
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        
        mock_client_instance.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=mock_response, body=None
        )
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Attempt generation
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMRateLimitError) as exc_info:
            provider.generate(messages=messages, model="gpt-4")
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_authentication_error_propagation(self, mock_openai_client, base_settings):
        """Test authentication error propagates correctly."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        
        # Create a mock response with request attribute
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        
        mock_client_instance.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key", response=mock_response, body=None
        )
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "invalid-key"
        })
        provider = create_llm_provider(settings)
        
        # Attempt generation
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            provider.generate(messages=messages, model="gpt-4")
        
        assert "Authentication failed" in str(exc_info.value)
    
    def test_api_error_propagation(self, mock_openai_client, base_settings):
        """Test generic API error propagates correctly."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.side_effect = APIError(
            "Service unavailable", request=None, body=None
        )
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Attempt generation
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMError) as exc_info:
            provider.generate(messages=messages, model="gpt-4")
        
        assert "API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stream_error_propagation(self, mock_openai_client, base_settings):
        """Test error propagates correctly during streaming."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        
        # Create a mock response with request attribute
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        
        mock_client_instance.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=mock_response, body=None
        )
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Attempt streaming
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMRateLimitError):
            async for _ in provider.stream(messages=messages, model="gpt-4"):
                pass


class TestSettingsIntegration:
    """Tests for settings integration with providers."""
    
    def test_settings_from_environment_openai(self, mock_openai_client, monkeypatch):
        """Test loading OpenAI settings from environment variables."""
        # Set environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        
        # Load settings (should pick up env var)
        settings = Settings(
            data_dir=Path("/tmp/test"),
            llm_provider="openai"
        )
        
        # Create provider
        provider = create_llm_provider(settings)
        
        # Verify correct API key was used
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "sk-env-key"
    
    def test_settings_from_environment_openrouter(self, mock_openai_client, monkeypatch):
        """Test loading OpenRouter settings from environment variables."""
        # Set environment variables
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        monkeypatch.setenv("OPENROUTER_APP_NAME", "Test From Env")
        monkeypatch.setenv("OPENROUTER_SITE_URL", "https://env-test.com")
        
        # Load settings
        settings = Settings(
            data_dir=Path("/tmp/test"),
            llm_provider="openrouter"
        )
        
        # Create provider
        provider = create_llm_provider(settings)
        
        # Verify correct configuration
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "sk-or-env-key"
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["default_headers"]["X-Title"] == "Test From Env"
        assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://env-test.com"
    
    def test_settings_from_environment_local_llm(self, mock_openai_client, monkeypatch):
        """Test loading local LLM settings from environment variables."""
        # Set environment variable
        monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://env-host:8080/v1")
        
        # Load settings
        settings = Settings(
            data_dir=Path("/tmp/test"),
            llm_provider="local"
        )
        
        # Create provider
        provider = create_llm_provider(settings)
        
        # Verify configuration
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "local"
        assert call_kwargs["base_url"] == "http://env-host:8080/v1"


class TestMessageFlowIntegration:
    """Tests for message flow through the full stack."""
    
    def test_multi_turn_conversation_flow(self, mock_openai_client, base_settings, mock_successful_response):
        """Test multi-turn conversation through the stack."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Multi-turn conversation
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi! How can I help?"),
            Message(role="user", content="Tell me about AI")
        ]
        
        response = provider.generate(messages=messages, model="gpt-4")
        
        # Verify all messages were sent
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        api_messages = call_kwargs["messages"]
        assert len(api_messages) == 3
        assert api_messages[0]["role"] == "user"
        assert api_messages[1]["role"] == "assistant"
        assert api_messages[2]["role"] == "user"
    
    def test_empty_messages_handling(self, mock_openai_client, base_settings, mock_successful_response):
        """Test handling of empty message list."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Empty messages
        messages = []
        response = provider.generate(messages=messages, model="gpt-4")
        
        # Should still work
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == []
    
    def test_special_characters_in_messages(self, mock_openai_client, base_settings, mock_successful_response):
        """Test messages with special characters flow through correctly."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Messages with special characters
        messages = [
            Message(role="user", content="Test with <>&\"' and emoji 🎉")
        ]
        
        response = provider.generate(messages=messages, model="gpt-4")
        
        # Verify special characters were preserved
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        api_messages = call_kwargs["messages"]
        assert "🎉" in api_messages[0]["content"]
        assert "<>&\"'" in api_messages[0]["content"]


class TestDebugLoggingIntegration:
    """Tests for debug logging integration across components."""
    
    @patch("expert_among_us.llm.openai_compatible.DebugLogger")
    def test_debug_logging_through_factory(self, mock_debug_logger, mock_openai_client, base_settings, mock_successful_response):
        """Test debug logging works when enabled through factory."""
        # Enable debug in DebugLogger
        mock_debug_logger.is_enabled.return_value = True
        
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_successful_response.model_dump.return_value = {"test": "data"}
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        # Create provider with debug enabled
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings, debug=True)
        
        # Generate response
        messages = [Message(role="user", content="Test")]
        response = provider.generate(messages=messages, model="gpt-4")
        
        # Verify debug logging was called
        assert mock_debug_logger.log_request.called
        assert mock_debug_logger.log_response.called
    
    @patch("expert_among_us.llm.openai_compatible.DebugLogger")
    def test_no_debug_logging_when_disabled(self, mock_debug_logger, mock_openai_client, base_settings, mock_successful_response):
        """Test debug logging is not called when disabled."""
        # Disable debug
        mock_debug_logger.is_enabled.return_value = False
        
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        # Create provider without debug
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings, debug=False)
        
        # Generate response
        messages = [Message(role="user", content="Test")]
        response = provider.generate(messages=messages, model="gpt-4")
        
        # Verify debug logging was not called
        assert not mock_debug_logger.log_request.called
        assert not mock_debug_logger.log_response.called


class TestRealisticUsagePatterns:
    """Tests for realistic usage patterns."""
    
    def test_prompt_generation_workflow(self, mock_openai_client, base_settings, mock_successful_response):
        """Test realistic prompt generation workflow."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Simulate prompt generation for a commit
        messages = [
            Message(
                role="user",
                content="Generate a one-sentence prompt for this commit:\n\n"
                       "Author: alice@example.com\n"
                       "Date: 2024-01-15\n"
                       "Message: Add authentication support\n"
                       "Files: auth.py, middleware.py\n"
                       "Diff: +def authenticate(user): return True"
            )
        ]
        
        response = provider.generate(
            messages=messages,
            model="gpt-4",
            max_tokens=150,
            temperature=0.7
        )
        
        assert response.content == "This is a test response"
        assert response.usage.input_tokens > 0
    
    @pytest.mark.asyncio
    async def test_streaming_expert_response_workflow(self, mock_openai_client, base_settings, mock_streaming_chunks):
        """Test realistic streaming workflow for expert responses."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = iter(mock_streaming_chunks)
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Simulate expert response streaming
        system_prompt = "You are an expert software developer providing guidance."
        messages = [
            Message(role="user", content="Expert 1: Add authentication"),
            Message(role="assistant", content="I added JWT-based auth"),
            Message(role="user", content="How should I implement OAuth2?")
        ]
        
        # Stream response
        full_response = ""
        final_usage = None
        async for chunk in provider.stream(
            messages=messages,
            model="gpt-4",
            system=system_prompt,
            max_tokens=4096,
            temperature=0.7
        ):
            if chunk.delta:
                full_response += chunk.delta
            if chunk.usage:
                final_usage = chunk.usage
        
        # Verify streaming worked correctly
        assert full_response == "Hello world!"
        assert final_usage is not None
        assert final_usage.total_tokens == 70
    
    def test_model_parameter_override(self, mock_openai_client, base_settings, mock_successful_response):
        """Test that model parameter can override default model."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        # Create provider with default model
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Generate with different model
        messages = [Message(role="user", content="Test")]
        response = provider.generate(
            messages=messages,
            model="gpt-3.5-turbo"  # Override default
        )
        
        # Verify correct model was used
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-3.5-turbo"
    
    def test_temperature_and_token_variations(self, mock_openai_client, base_settings, mock_successful_response):
        """Test different temperature and token configurations."""
        # Setup
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_successful_response
        
        settings = base_settings.model_copy(update={
            "llm_provider": "openai",
            "openai_api_key": "sk-test-key"
        })
        provider = create_llm_provider(settings)
        
        # Test different configurations
        test_cases = [
            {"max_tokens": 100, "temperature": 0.0},  # Deterministic
            {"max_tokens": 4096, "temperature": 1.0},  # Default
            {"max_tokens": 2048, "temperature": 2.0},  # Creative
        ]
        
        for config in test_cases:
            messages = [Message(role="user", content="Test")]
            response = provider.generate(
                messages=messages,
                model="gpt-4",
                **config
            )
            
            # Verify parameters were set correctly
            call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
            assert call_kwargs["max_tokens"] == config["max_tokens"]
            assert call_kwargs["temperature"] == config["temperature"]