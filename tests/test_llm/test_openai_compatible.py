"""Tests for OpenAICompatibleLLM implementation."""

import pytest
from unittest.mock import MagicMock, patch, Mock
from openai import APIError, RateLimitError, APIConnectionError, AuthenticationError

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


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch("expert_among_us.llm.openai_compatible.OpenAI") as mock:
        yield mock


@pytest.fixture
def openai_llm(mock_openai_client):
    """Create OpenAICompatibleLLM instance with mocked client."""
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance
    
    llm = OpenAICompatibleLLM(
        api_key="test-api-key",
        model="gpt-4"
    )
    return llm


class TestOpenAICompatibleLLMInit:
    """Tests for OpenAICompatibleLLM initialization."""
    
    def test_init_minimal_config(self, mock_openai_client):
        """Test initialization with minimal configuration."""
        llm = OpenAICompatibleLLM(
            api_key="test-key",
            model="gpt-4"
        )
        
        mock_openai_client.assert_called_once()
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["max_retries"] == 3
        assert "base_url" not in call_kwargs
        assert "organization" not in call_kwargs
        assert llm.model == "gpt-4"
        assert llm.debug is False
    
    def test_init_openai_config(self, mock_openai_client):
        """Test initialization with OpenAI-specific configuration."""
        llm = OpenAICompatibleLLM(
            api_key="sk-test-key",
            model="gpt-4-turbo",
            organization="org-123",
            timeout=60.0,
            max_retries=5
        )
        
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-key"
        assert call_kwargs["organization"] == "org-123"
        assert call_kwargs["timeout"] == 60.0
        assert call_kwargs["max_retries"] == 5
        assert llm.model == "gpt-4-turbo"
    
    def test_init_openrouter_config(self, mock_openai_client):
        """Test initialization with OpenRouter configuration."""
        extra_headers = {
            "HTTP-Referer": "https://example.com",
            "X-Title": "My App"
        }
        
        llm = OpenAICompatibleLLM(
            api_key="sk-or-test",
            model="anthropic/claude-3-opus",
            base_url="https://openrouter.ai/api/v1",
            extra_headers=extra_headers
        )
        
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "sk-or-test"
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["default_headers"] == extra_headers
        assert llm.model == "anthropic/claude-3-opus"
    
    def test_init_local_llm_config(self, mock_openai_client):
        """Test initialization with local LLM configuration."""
        llm = OpenAICompatibleLLM(
            api_key="local",
            model="llama-3-70b",
            base_url="http://localhost:8000/v1",
            timeout=120.0
        )
        
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "local"
        assert call_kwargs["base_url"] == "http://localhost:8000/v1"
        assert call_kwargs["timeout"] == 120.0
        assert llm.model == "llama-3-70b"
    
    def test_init_with_debug(self, mock_openai_client):
        """Test initialization with debug enabled."""
        llm = OpenAICompatibleLLM(
            api_key="test-key",
            model="gpt-4",
            debug=True
        )
        
        assert llm.debug is True


class TestMessageFormatConversion:
    """Tests for message format conversion."""
    
    def test_to_openai_format_single_message(self, openai_llm):
        """Test converting single message to OpenAI format."""
        messages = [Message(role="user", content="Hello")]
        
        result = openai_llm._to_openai_format(messages)
        
        assert result == [
            {"role": "user", "content": "Hello"}
        ]
    
    def test_to_openai_format_multiple_messages(self, openai_llm):
        """Test converting multiple messages to OpenAI format."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?")
        ]
        
        result = openai_llm._to_openai_format(messages)
        
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    
    def test_to_openai_format_empty_list(self, openai_llm):
        """Test converting empty message list."""
        messages = []
        
        result = openai_llm._to_openai_format(messages)
        
        assert result == []


class TestGenerate:
    """Tests for synchronous generate method."""
    
    def test_generate_success(self, openai_llm):
        """Test successful generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        response = openai_llm.generate(
            messages=messages,
            model="gpt-4"
        )
        
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a response"
        assert response.model == "gpt-4"
        assert response.stop_reason == "stop"
        assert isinstance(response.usage, UsageMetrics)
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        assert response.usage.cache_read_tokens == 0
        assert response.usage.cache_creation_tokens == 0
    
    def test_generate_with_system_prompt(self, openai_llm):
        """Test generation with system prompt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        response = openai_llm.generate(
            messages=messages,
            model="gpt-4",
            system="You are a helpful assistant"
        )
        
        # Verify system prompt was included
        call_args = openai_llm.client.chat.completions.create.call_args
        request = call_args[1]
        assert request["messages"][0] == {"role": "system", "content": "You are a helpful assistant"}
        assert request["messages"][1] == {"role": "user", "content": "Hello"}
    
    def test_generate_with_custom_params(self, openai_llm):
        """Test generation with custom parameters."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        response = openai_llm.generate(
            messages=messages,
            model="gpt-4-turbo",
            max_tokens=2048,
            temperature=0.7
        )
        
        # Verify parameters were passed correctly
        call_args = openai_llm.client.chat.completions.create.call_args
        request = call_args[1]
        assert request["model"] == "gpt-4-turbo"
        assert request["max_tokens"] == 2048
        assert request["temperature"] == 0.7
    
    def test_generate_with_cache_metrics(self, openai_llm):
        """Test generation with cache metrics (OpenAI prompt caching)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Cached response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.usage.prompt_tokens_details = {"cached_tokens": 100}
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        response = openai_llm.generate(
            messages=messages,
            model="gpt-4"
        )
        
        assert response.usage.cache_read_tokens == 100
    
    def test_generate_empty_content(self, openai_llm):
        """Test generation with empty content response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 5
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        response = openai_llm.generate(
            messages=messages,
            model="gpt-4"
        )
        
        assert response.content == ""


class TestStream:
    """Tests for streaming generate method."""
    
    @pytest.mark.asyncio
    async def test_stream_success(self, openai_llm):
        """Test successful streaming."""
        # Mock streaming chunks
        mock_chunks = []
        
        # Content chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock(content="Hello")
        chunk1.choices[0].finish_reason = None
        mock_chunks.append(chunk1)
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock(content=" world")
        chunk2.choices[0].finish_reason = None
        mock_chunks.append(chunk2)
        
        # Final chunk with stop reason
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock(content=None)
        chunk3.choices[0].finish_reason = "stop"
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 5
        chunk3.usage.completion_tokens = 10
        chunk3.usage.total_tokens = 15
        mock_chunks.append(chunk3)
        
        openai_llm.client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in openai_llm.stream(
            messages=messages,
            model="gpt-4"
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[0].stop_reason is None
        assert chunks[0].usage is None
        
        assert chunks[1].delta == " world"
        
        assert chunks[2].delta == ""
        assert chunks[2].stop_reason == "stop"
        assert isinstance(chunks[2].usage, UsageMetrics)
        assert chunks[2].usage.input_tokens == 5
        assert chunks[2].usage.output_tokens == 10
        assert chunks[2].usage.total_tokens == 15
    
    @pytest.mark.asyncio
    async def test_stream_with_cache_metrics(self, openai_llm):
        """Test streaming with cache metrics."""
        mock_chunks = []
        
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock(content="Test")
        chunk1.choices[0].finish_reason = None
        mock_chunks.append(chunk1)
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock(content=None)
        chunk2.choices[0].finish_reason = "stop"
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 5
        chunk2.usage.completion_tokens = 10
        chunk2.usage.total_tokens = 15
        chunk2.usage.prompt_tokens_details = {"cached_tokens": 100}
        mock_chunks.append(chunk2)
        
        openai_llm.client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in openai_llm.stream(
            messages=messages,
            model="gpt-4"
        ):
            chunks.append(chunk)
        
        assert chunks[-1].usage.cache_read_tokens == 100
    
    @pytest.mark.asyncio
    async def test_stream_with_system_prompt(self, openai_llm):
        """Test streaming with system prompt."""
        mock_chunks = [
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="Test"), finish_reason=None)]
            ),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content=None), finish_reason="stop")],
                usage=MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
            )
        ]
        
        openai_llm.client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in openai_llm.stream(
            messages=messages,
            model="gpt-4",
            system="You are helpful"
        ):
            chunks.append(chunk)
        
        # Verify system prompt was included
        call_args = openai_llm.client.chat.completions.create.call_args
        request = call_args[1]
        assert request["messages"][0] == {"role": "system", "content": "You are helpful"}
        assert request["stream"] is True
    
    @pytest.mark.asyncio
    async def test_stream_with_custom_params(self, openai_llm):
        """Test streaming with custom parameters."""
        mock_chunks = [
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="Test"), finish_reason=None)]
            ),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content=None), finish_reason="max_tokens")],
                usage=MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
            )
        ]
        
        openai_llm.client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in openai_llm.stream(
            messages=messages,
            model="gpt-4-turbo",
            max_tokens=1024,
            temperature=0.3
        ):
            chunks.append(chunk)
        
        # Verify parameters
        call_args = openai_llm.client.chat.completions.create.call_args
        request = call_args[1]
        assert request["model"] == "gpt-4-turbo"
        assert request["max_tokens"] == 1024
        assert request["temperature"] == 0.3


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_handle_rate_limit_error(self, openai_llm):
        """Test handling of rate limit errors."""
        error = RateLimitError("Rate limit exceeded", response=None, body=None)
        
        result = openai_llm._handle_error(error)
        
        assert isinstance(result, LLMRateLimitError)
        assert "Rate limit exceeded" in str(result)
    
    def test_handle_authentication_error(self, openai_llm):
        """Test handling of authentication errors."""
        error = AuthenticationError("Invalid API key", response=None, body=None)
        
        result = openai_llm._handle_error(error)
        
        assert isinstance(result, LLMInvalidRequestError)
        assert "Authentication failed" in str(result)
    
    def test_handle_connection_error(self, openai_llm):
        """Test handling of connection errors."""
        error = APIConnectionError(request=None)
        
        result = openai_llm._handle_error(error)
        
        assert isinstance(result, LLMError)
        assert "Connection error" in str(result)
    
    def test_handle_api_error(self, openai_llm):
        """Test handling of generic API errors."""
        error = APIError("API error occurred", request=None, body=None)
        
        result = openai_llm._handle_error(error)
        
        assert isinstance(result, LLMError)
        assert "API error" in str(result)
    
    def test_handle_unexpected_error(self, openai_llm):
        """Test handling of unexpected errors."""
        error = ValueError("Unexpected error")
        
        result = openai_llm._handle_error(error)
        
        assert isinstance(result, LLMError)
        assert "Unexpected error" in str(result)
    
    def test_generate_raises_rate_limit_error(self, openai_llm):
        """Test generate method raises rate limit error."""
        openai_llm.client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=None, body=None
        )
        
        messages = [Message(role="user", content="Hello")]
        with pytest.raises(LLMRateLimitError):
            openai_llm.generate(messages=messages, model="gpt-4")
    
    def test_generate_raises_invalid_request_error(self, openai_llm):
        """Test generate method raises invalid request error."""
        openai_llm.client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key", response=None, body=None
        )
        
        messages = [Message(role="user", content="Hello")]
        with pytest.raises(LLMInvalidRequestError):
            openai_llm.generate(messages=messages, model="gpt-4")
    
    def test_generate_raises_connection_error(self, openai_llm):
        """Test generate method raises connection error."""
        openai_llm.client.chat.completions.create.side_effect = APIConnectionError(
            request=None
        )
        
        messages = [Message(role="user", content="Hello")]
        with pytest.raises(LLMError):
            openai_llm.generate(messages=messages, model="gpt-4")
    
    @pytest.mark.asyncio
    async def test_stream_raises_rate_limit_error(self, openai_llm):
        """Test stream method raises rate limit error."""
        openai_llm.client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=None, body=None
        )
        
        messages = [Message(role="user", content="Hello")]
        with pytest.raises(LLMRateLimitError):
            async for _ in openai_llm.stream(messages=messages, model="gpt-4"):
                pass


class TestDebugLogging:
    """Tests for debug logging integration."""
    
    @patch("expert_among_us.llm.openai_compatible.DebugLogger")
    def test_generate_logs_when_debug_enabled(self, mock_debug_logger, openai_llm):
        """Test that generate logs when debug is enabled."""
        mock_debug_logger.is_enabled.return_value = True
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        mock_response.model_dump.return_value = {"test": "data"}
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        openai_llm.generate(messages=messages, model="gpt-4")
        
        # Verify logging was called
        assert mock_debug_logger.log_request.called
        assert mock_debug_logger.log_response.called
    
    @patch("expert_among_us.llm.openai_compatible.DebugLogger")
    def test_generate_logs_with_instance_debug(self, mock_debug_logger, mock_openai_client):
        """Test that generate logs when instance debug is enabled."""
        mock_debug_logger.is_enabled.return_value = False
        
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        
        llm = OpenAICompatibleLLM(
            api_key="test-key",
            model="gpt-4",
            debug=True
        )
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        mock_response.model_dump.return_value = {"test": "data"}
        
        llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        llm.generate(messages=messages, model="gpt-4")
        
        # Verify logging was called even when global debug is disabled
        assert mock_debug_logger.log_request.called
        assert mock_debug_logger.log_response.called
    
    @patch("expert_among_us.llm.openai_compatible.DebugLogger")
    def test_generate_no_logs_when_debug_disabled(self, mock_debug_logger, openai_llm):
        """Test that generate doesn't log when debug is disabled."""
        mock_debug_logger.is_enabled.return_value = False
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        openai_llm.generate(messages=messages, model="gpt-4")
        
        # Verify logging was not called
        assert not mock_debug_logger.log_request.called
        assert not mock_debug_logger.log_response.called
    
    @pytest.mark.asyncio
    @patch("expert_among_us.llm.openai_compatible.DebugLogger")
    async def test_stream_logs_when_debug_enabled(self, mock_debug_logger, openai_llm):
        """Test that stream logs when debug is enabled."""
        mock_debug_logger.is_enabled.return_value = True
        
        mock_chunks = [
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="Test"), finish_reason=None)],
                model_dump=lambda: {"chunk": 1}
            ),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content=None), finish_reason="stop")],
                usage=MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15),
                model_dump=lambda: {"chunk": 2}
            )
        ]
        
        openai_llm.client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [Message(role="user", content="Hello")]
        async for _ in openai_llm.stream(messages=messages, model="gpt-4"):
            pass
        
        # Verify logging was called
        assert mock_debug_logger.log_request.called
        assert mock_debug_logger.log_response.called


class TestOpenRouterIntegration:
    """Tests for OpenRouter-specific features."""
    
    def test_openrouter_headers_configuration(self, mock_openai_client):
        """Test OpenRouter-specific headers are properly configured."""
        extra_headers = {
            "HTTP-Referer": "https://myapp.com",
            "X-Title": "Expert Among Us"
        }
        
        llm = OpenAICompatibleLLM(
            api_key="sk-or-v1-test",
            model="anthropic/claude-3-opus",
            base_url="https://openrouter.ai/api/v1",
            extra_headers=extra_headers
        )
        
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["default_headers"] == extra_headers
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    
    def test_openrouter_without_headers(self, mock_openai_client):
        """Test OpenRouter configuration without optional headers."""
        llm = OpenAICompatibleLLM(
            api_key="sk-or-v1-test",
            model="anthropic/claude-3-opus",
            base_url="https://openrouter.ai/api/v1"
        )
        
        call_kwargs = mock_openai_client.call_args[1]
        assert "default_headers" not in call_kwargs
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"


class TestUsageMetricsExtraction:
    """Tests for usage metrics extraction."""
    
    def test_usage_metrics_basic(self, openai_llm):
        """Test basic usage metrics extraction."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        response = openai_llm.generate(messages=messages, model="gpt-4")
        
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150
        assert response.usage.cache_read_tokens == 0
        assert response.usage.cache_creation_tokens == 0
    
    def test_usage_metrics_with_cache(self, openai_llm):
        """Test usage metrics extraction with cache."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Cached response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens_details = {"cached_tokens": 80}
        
        openai_llm.client.chat.completions.create.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        response = openai_llm.generate(messages=messages, model="gpt-4")
        
        assert response.usage.cache_read_tokens == 80
        assert response.usage.cache_creation_tokens == 0
    
    @pytest.mark.asyncio
    async def test_usage_metrics_stream_basic(self, openai_llm):
        """Test basic usage metrics extraction from stream."""
        mock_chunks = [
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="Test"), finish_reason=None)]
            ),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content=None), finish_reason="stop")],
                usage=MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            )
        ]
        
        openai_llm.client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in openai_llm.stream(messages=messages, model="gpt-4"):
            chunks.append(chunk)
        
        assert chunks[-1].usage.input_tokens == 100
        assert chunks[-1].usage.output_tokens == 50
        assert chunks[-1].usage.total_tokens == 150