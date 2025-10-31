"""Tests for BedrockLLM implementation."""

import pytest
from unittest.mock import MagicMock, patch, call

from botocore.exceptions import ClientError

from expert_among_us.llm.bedrock import BedrockLLM
from expert_among_us.llm.base import (
    Message,
    LLMResponse,
    StreamChunk,
    UsageMetrics,
    LLMError,
    LLMRateLimitError,
    LLMInvalidRequestError,
)
from expert_among_us.config.settings import PROVIDER_MODEL_DEFAULTS

# Extract model IDs from settings
CLAUDE_SONNET_MODEL_ID = PROVIDER_MODEL_DEFAULTS["bedrock"]["expert"]
NOVA_LITE_MODEL_ID = PROVIDER_MODEL_DEFAULTS["bedrock"]["promptgen"]


@pytest.fixture
def mock_boto3():
    """Mock boto3 for testing."""
    with patch("expert_among_us.llm.bedrock.boto3") as mock:
        yield mock


@pytest.fixture
def bedrock_llm(mock_boto3):
    """Create BedrockLLM instance with mocked boto3."""
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client
    
    llm = BedrockLLM(region_name="us-west-2")
    return llm


class TestBedrockLLMInit:
    """Tests for BedrockLLM initialization."""
    
    def test_init_default_region(self, mock_boto3):
        """Test initialization with default region."""
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        
        llm = BedrockLLM()
        
        mock_boto3.Session.assert_called_once_with(
            region_name="us-west-2",
            profile_name=None
        )
        mock_session.client.assert_called_once_with("bedrock-runtime")
        assert llm.enable_caching is True
    
    def test_init_custom_region(self, mock_boto3):
        """Test initialization with custom region."""
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        
        llm = BedrockLLM(region_name="us-east-1")
        
        mock_boto3.Session.assert_called_once_with(
            region_name="us-east-1",
            profile_name=None
        )
    
    def test_init_with_profile(self, mock_boto3):
        """Test initialization with AWS profile."""
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        
        llm = BedrockLLM(profile_name="my-profile")
        
        mock_boto3.Session.assert_called_once_with(
            region_name="us-west-2",
            profile_name="my-profile"
        )
    
    def test_init_with_caching_disabled(self, mock_boto3):
        """Test initialization with caching disabled."""
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        
        llm = BedrockLLM(enable_caching=False)
        
        assert llm.enable_caching is False
    
    def test_model_constants(self):
        """Test model identifier constants are available from settings."""
        assert CLAUDE_SONNET_MODEL_ID == "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert NOVA_LITE_MODEL_ID == "us.amazon.nova-lite-v1:0"


class TestMessageFormatConversion:
    """Tests for message format conversion."""
    
    def test_to_bedrock_format_single_message(self, bedrock_llm):
        """Test converting single message to Bedrock format."""
        messages = [Message(role="user", content="Hello")]
        
        result = bedrock_llm._to_bedrock_format(messages)
        
        assert result == [
            {"role": "user", "content": [{"text": "Hello"}]}
        ]
    
    def test_to_bedrock_format_multiple_messages(self, bedrock_llm):
        """Test converting multiple messages to Bedrock format."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?")
        ]
        
        result = bedrock_llm._to_bedrock_format(messages)
        
        assert result == [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there"}]},
            {"role": "user", "content": [{"text": "How are you?"}]}
        ]
    
    def test_to_bedrock_format_empty_list(self, bedrock_llm):
        """Test converting empty message list."""
        messages = []
        
        result = bedrock_llm._to_bedrock_format(messages)
        
        assert result == []


class TestGenerate:
    """Tests for synchronous generate method."""
    
    def test_generate_success(self, bedrock_llm):
        """Test successful generation."""
        bedrock_llm.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "This is a response"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30
            }
        }
        
        messages = [Message(role="user", content="Hello")]
        response = bedrock_llm.generate(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID
        )
        
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a response"
        assert response.model == CLAUDE_SONNET_MODEL_ID
        assert response.stop_reason == "end_turn"
        assert isinstance(response.usage, UsageMetrics)
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        assert response.usage.cache_read_tokens == 0
        assert response.usage.cache_creation_tokens == 0
    
    def test_generate_with_cache_metrics(self, bedrock_llm):
        """Test generation with cache metrics."""
        bedrock_llm.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Cached response"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30,
                "cacheReadInputTokens": 100,
                "cacheWriteInputTokens": 50
            }
        }
        
        messages = [Message(role="user", content="Hello")]
        response = bedrock_llm.generate(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID
        )
        
        assert response.usage.cache_read_tokens == 100
        assert response.usage.cache_creation_tokens == 50
    
    def test_generate_with_system_prompt(self, bedrock_llm):
        """Test generation with system prompt."""
        bedrock_llm.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Response"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 10}
        }
        
        messages = [Message(role="user", content="Hello")]
        response = bedrock_llm.generate(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID,
            system="You are a helpful assistant"
        )
        
        # Verify system prompt was included in request
        call_args = bedrock_llm.client.converse.call_args
        assert call_args[1]["system"][0]["text"] == "You are a helpful assistant"
    
    def test_generate_with_custom_params(self, bedrock_llm):
        """Test generation with custom parameters."""
        bedrock_llm.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Response"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 10}
        }
        
        messages = [Message(role="user", content="Hello")]
        response = bedrock_llm.generate(
            messages=messages,
            model=NOVA_LITE_MODEL_ID,
            max_tokens=2048,
            temperature=0.5
        )
        
        # Verify parameters were passed correctly
        call_args = bedrock_llm.client.converse.call_args
        assert call_args[1]["inferenceConfig"]["maxTokens"] == 2048
        assert call_args[1]["inferenceConfig"]["temperature"] == 0.5
    
    def test_generate_nova_lite_model(self, bedrock_llm):
        """Test generation with Nova Lite model."""
        bedrock_llm.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Nova response"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 10}
        }
        
        messages = [Message(role="user", content="Test")]
        response = bedrock_llm.generate(
            messages=messages,
            model=NOVA_LITE_MODEL_ID
        )
        
        assert response.model == NOVA_LITE_MODEL_ID
        assert response.content == "Nova response"


class TestStream:
    """Tests for streaming generate method."""
    
    @pytest.mark.asyncio
    async def test_stream_success(self, bedrock_llm):
        """Test successful streaming."""
        # Mock streaming events
        mock_events = [
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"delta": {"text": " world"}}},
            {
                "metadata": {
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 5,
                        "outputTokens": 10,
                        "totalTokens": 15
                    }
                }
            }
        ]
        
        bedrock_llm.client.converse_stream.return_value = {
            "stream": iter(mock_events)
        }
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in bedrock_llm.stream(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[0].stop_reason is None
        assert chunks[0].usage is None
        
        assert chunks[1].delta == " world"
        
        assert chunks[2].delta == ""
        assert chunks[2].stop_reason == "end_turn"
        assert isinstance(chunks[2].usage, UsageMetrics)
        assert chunks[2].usage.input_tokens == 5
        assert chunks[2].usage.output_tokens == 10
        assert chunks[2].usage.total_tokens == 15
    
    @pytest.mark.asyncio
    async def test_stream_with_cache_metrics(self, bedrock_llm):
        """Test streaming with cache metrics."""
        mock_events = [
            {"contentBlockDelta": {"delta": {"text": "Cached"}}},
            {
                "metadata": {
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 5,
                        "outputTokens": 10,
                        "totalTokens": 15,
                        "cacheReadInputTokens": 100,
                        "cacheWriteInputTokens": 50
                    }
                }
            }
        ]
        
        bedrock_llm.client.converse_stream.return_value = {
            "stream": iter(mock_events)
        }
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in bedrock_llm.stream(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID
        ):
            chunks.append(chunk)
        
        assert chunks[1].usage.cache_read_tokens == 100
        assert chunks[1].usage.cache_creation_tokens == 50
    
    @pytest.mark.asyncio
    async def test_stream_with_system_prompt(self, bedrock_llm):
        """Test streaming with system prompt."""
        mock_events = [
            {"contentBlockDelta": {"delta": {"text": "Response"}}},
            {
                "metadata": {
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 5, "outputTokens": 10}
                }
            }
        ]
        
        bedrock_llm.client.converse_stream.return_value = {
            "stream": iter(mock_events)
        }
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in bedrock_llm.stream(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID,
            system="You are helpful"
        ):
            chunks.append(chunk)
        
        # Verify system prompt was included
        call_args = bedrock_llm.client.converse_stream.call_args
        assert call_args[1]["system"][0]["text"] == "You are helpful"
    
    @pytest.mark.asyncio
    async def test_stream_with_custom_params(self, bedrock_llm):
        """Test streaming with custom parameters."""
        mock_events = [
            {"contentBlockDelta": {"delta": {"text": "Test"}}},
            {
                "metadata": {
                    "stopReason": "max_tokens",
                    "usage": {"inputTokens": 5, "outputTokens": 10}
                }
            }
        ]
        
        bedrock_llm.client.converse_stream.return_value = {
            "stream": iter(mock_events)
        }
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in bedrock_llm.stream(
            messages=messages,
            model=NOVA_LITE_MODEL_ID,
            max_tokens=1024,
            temperature=0.3
        ):
            chunks.append(chunk)
        
        # Verify parameters
        call_args = bedrock_llm.client.converse_stream.call_args
        assert call_args[1]["inferenceConfig"]["maxTokens"] == 1024
        assert call_args[1]["inferenceConfig"]["temperature"] == 0.3


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_handle_throttling_exception(self, bedrock_llm):
        """Test handling of rate limit errors."""
        error = ClientError(
            {
                "Error": {
                    "Code": "ThrottlingException",
                    "Message": "Rate limit exceeded"
                }
            },
            "converse"
        )
        
        with pytest.raises(LLMRateLimitError) as exc_info:
            bedrock_llm._handle_error(error)
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_handle_validation_exception(self, bedrock_llm):
        """Test handling of validation errors."""
        error = ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Invalid parameter"
                }
            },
            "converse"
        )
        
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            bedrock_llm._handle_error(error)
        
        assert "Invalid request" in str(exc_info.value)
    
    def test_handle_generic_error(self, bedrock_llm):
        """Test handling of generic errors."""
        error = ClientError(
            {
                "Error": {
                    "Code": "ServiceException",
                    "Message": "Service error"
                }
            },
            "converse"
        )
        
        with pytest.raises(LLMError) as exc_info:
            bedrock_llm._handle_error(error)
        
        assert "ServiceException" in str(exc_info.value)
    
    def test_generate_raises_rate_limit_error(self, bedrock_llm):
        """Test generate method raises rate limit error."""
        bedrock_llm.client.converse.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ThrottlingException",
                    "Message": "Rate limit"
                }
            },
            "converse"
        )
        
        messages = [Message(role="user", content="Hello")]
        with pytest.raises(LLMRateLimitError):
            bedrock_llm.generate(
                messages=messages,
                model=CLAUDE_SONNET_MODEL_ID
            )
    
    def test_generate_raises_invalid_request_error(self, bedrock_llm):
        """Test generate method raises invalid request error."""
        bedrock_llm.client.converse.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Invalid"
                }
            },
            "converse"
        )
        
        messages = [Message(role="user", content="Hello")]
        with pytest.raises(LLMInvalidRequestError):
            bedrock_llm.generate(
                messages=messages,
                model=CLAUDE_SONNET_MODEL_ID
            )
    
    @pytest.mark.asyncio
    async def test_stream_raises_rate_limit_error(self, bedrock_llm):
        """Test stream method raises rate limit error."""
        bedrock_llm.client.converse_stream.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ThrottlingException",
                    "Message": "Rate limit"
                }
            },
            "converse_stream"
        )
        
        messages = [Message(role="user", content="Hello")]
        with pytest.raises(LLMRateLimitError):
            async for _ in bedrock_llm.stream(
                messages=messages,
                model=CLAUDE_SONNET_MODEL_ID
            ):
                pass


class TestDebugLogging:
    """Tests for debug logging integration."""
    
    @patch("expert_among_us.llm.bedrock.DebugLogger")
    def test_generate_logs_when_debug_enabled(self, mock_debug_logger, bedrock_llm):
        """Test that generate logs when debug is enabled."""
        mock_debug_logger.is_enabled.return_value = True
        
        bedrock_llm.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Response"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 10}
        }
        
        messages = [Message(role="user", content="Hello")]
        bedrock_llm.generate(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID
        )
        
        # Verify logging was called
        assert mock_debug_logger.log_request.called
        assert mock_debug_logger.log_response.called
    
    @patch("expert_among_us.llm.bedrock.DebugLogger")
    def test_generate_no_logs_when_debug_disabled(self, mock_debug_logger, bedrock_llm):
        """Test that generate doesn't log when debug is disabled."""
        mock_debug_logger.is_enabled.return_value = False
        
        bedrock_llm.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Response"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 10}
        }
        
        messages = [Message(role="user", content="Hello")]
        bedrock_llm.generate(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID
        )
        
        # Verify logging was not called
        assert not mock_debug_logger.log_request.called
        assert not mock_debug_logger.log_response.called
    
    @pytest.mark.asyncio
    @patch("expert_among_us.llm.bedrock.DebugLogger")
    async def test_stream_logs_when_debug_enabled(self, mock_debug_logger, bedrock_llm):
        """Test that stream logs when debug is enabled."""
        mock_debug_logger.is_enabled.return_value = True
        
        mock_events = [
            {"contentBlockDelta": {"delta": {"text": "Test"}}},
            {
                "metadata": {
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 5, "outputTokens": 10}
                }
            }
        ]
        
        bedrock_llm.client.converse_stream.return_value = {
            "stream": iter(mock_events)
        }
        
        messages = [Message(role="user", content="Hello")]
        async for _ in bedrock_llm.stream(
            messages=messages,
            model=CLAUDE_SONNET_MODEL_ID
        ):
            pass
        
        # Verify logging was called
        assert mock_debug_logger.log_request.called
        assert mock_debug_logger.log_response.called