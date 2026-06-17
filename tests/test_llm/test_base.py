"""Tests for LLM base abstractions."""

import pytest
from typing import AsyncIterator, List

from expert_among_us.llm.base import (
    Message,
    LLMResponse,
    StreamChunk,
    UsageMetrics,
    LLMProvider,
    LLMError,
    LLMRateLimitError,
    LLMInvalidRequestError,
)


class TestMessage:
    """Tests for Message dataclass."""
    
    def test_message_creation(self):
        """Test creating a Message with valid data."""
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"


class TestUsageMetrics:
    """Tests for UsageMetrics dataclass."""
    
    def test_usage_metrics_creation(self):
        """Test creating UsageMetrics with required fields."""
        metrics = UsageMetrics(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30
        )
        assert metrics.input_tokens == 10
        assert metrics.output_tokens == 20
        assert metrics.total_tokens == 30
        assert metrics.cache_read_tokens == 0
        assert metrics.cache_creation_tokens == 0
    
    def test_usage_metrics_with_cache(self):
        """Test UsageMetrics with cache metrics."""
        metrics = UsageMetrics(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cache_read_tokens=100,
            cache_creation_tokens=50
        )
        assert metrics.cache_read_tokens == 100
        assert metrics.cache_creation_tokens == 50


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""
    
    def test_llm_response_creation(self):
        """Test creating an LLMResponse with valid data."""
        response = LLMResponse(
            content="Generated text",
            model="test-model",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=10, output_tokens=20, total_tokens=30)
        )
        assert response.content == "Generated text"
        assert response.model == "test-model"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""
    
    def test_stream_chunk_basic(self):
        """Test creating a basic StreamChunk."""
        chunk = StreamChunk(delta="Hello")
        assert chunk.delta == "Hello"
        assert chunk.stop_reason is None
        assert chunk.usage is None
    
    def test_stream_chunk_with_stop_reason(self):
        """Test StreamChunk with stop_reason (final chunk)."""
        chunk = StreamChunk(
            delta="",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=15, output_tokens=25, total_tokens=40)
        )
        assert chunk.delta == ""
        assert chunk.stop_reason == "end_turn"
        assert chunk.usage is not None


class TestLLMProvider:
    """Tests for LLMProvider abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()


class TestMockImplementation:
    """Tests for mock LLMProvider implementation."""
    
    class MockLLMProvider(LLMProvider):
        """Mock implementation for testing."""
        
        def generate(
            self,
            messages: List[Message],
            model: str,
            max_tokens: int = 4096,
            temperature: float = 1.0,
            system: str = None,
            debug_category: str = "expert",
        ) -> LLMResponse:
            """Mock generate method."""
            return LLMResponse(
                content=f"Mock response to: {messages[-1].content}",
                model=model,
                stop_reason="end_turn",
                usage=UsageMetrics(input_tokens=10, output_tokens=20, total_tokens=30)
            )
        
        async def stream(
            self,
            messages: List[Message],
            model: str,
            max_tokens: int = 4096,
            temperature: float = 1.0,
            system: str = None,
            debug_category: str = "expert",
        ) -> AsyncIterator[StreamChunk]:
            """Mock stream method."""
            chunks = ["Mock", " streaming", " response"]
            for chunk in chunks:
                yield StreamChunk(delta=chunk)
            yield StreamChunk(
                delta="",
                stop_reason="end_turn",
                usage=UsageMetrics(input_tokens=10, output_tokens=15, total_tokens=25)
            )
    
    def test_mock_provider_instantiation(self):
        """Test that mock provider can be instantiated."""
        provider = self.MockLLMProvider()
        assert isinstance(provider, LLMProvider)
    
    def test_mock_generate(self):
        """Test mock generate method."""
        provider = self.MockLLMProvider()
        messages = [Message(role="user", content="Test prompt")]
        
        response = provider.generate(
            messages=messages,
            model="test-model"
        )
        
        assert isinstance(response, LLMResponse)
        assert "Test prompt" in response.content
        assert response.model == "test-model"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens > 0
    
    @pytest.mark.asyncio
    async def test_mock_stream(self):
        """Test mock stream method."""
        provider = self.MockLLMProvider()
        messages = [Message(role="user", content="Stream test")]
        
        chunks = []
        async for chunk in provider.stream(
            messages=messages,
            model="test-model"
        ):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert all(isinstance(c, StreamChunk) for c in chunks)
        
        final_chunk = chunks[-1]
        assert final_chunk.stop_reason == "end_turn"
        assert final_chunk.usage is not None


class TestLLMExceptions:
    """Tests for LLM exception hierarchy."""
    
    def test_llm_error_base(self):
        """Test LLMError base exception."""
        error = LLMError("Base error")
        assert isinstance(error, Exception)
        assert str(error) == "Base error"
    
    def test_rate_limit_error(self):
        """Test LLMRateLimitError exception."""
        error = LLMRateLimitError("Rate limit exceeded")
        assert isinstance(error, LLMError)
        assert "Rate limit" in str(error)
    
    def test_invalid_request_error(self):
        """Test LLMInvalidRequestError exception."""
        error = LLMInvalidRequestError("Invalid parameters")
        assert isinstance(error, LLMError)
        assert "Invalid" in str(error)


class TestNotImplementedErrors:
    """Tests for NotImplementedError messages."""
    
    def test_generate_not_implemented_message(self):
        """Test that generate raises NotImplementedError with class name."""
        
        class PartialProvider(LLMProvider):
            """Provider that calls parent generate."""
            def generate(self, messages, model, max_tokens=4096,
                        temperature=1.0, system=None, debug_category="expert"):
                return super().generate(messages, model, max_tokens, temperature, system, debug_category)
            
            async def stream(self, messages, model, max_tokens=4096,
                           temperature=1.0, system=None, debug_category="expert"):
                yield StreamChunk(delta="test")
        
        provider = PartialProvider()
        
        with pytest.raises(NotImplementedError) as exc_info:
            provider.generate([], "model")
        
        assert "PartialProvider" in str(exc_info.value)
        assert "generate()" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stream_not_implemented_message(self):
        """Test that stream raises NotImplementedError with class name."""
        
        class PartialProvider(LLMProvider):
            """Provider that calls parent stream."""
            def generate(self, messages, model, max_tokens=4096,
                        temperature=1.0, system=None, debug_category="expert"):
                return LLMResponse(
                    content="test",
                    model=model,
                    stop_reason="end_turn",
                    usage=UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2)
                )
            
            async def stream(self, messages, model, max_tokens=4096,
                           temperature=1.0, system=None, debug_category="expert"):
                async for chunk in super().stream(messages, model, max_tokens, temperature, system, debug_category):
                    yield chunk
        
        provider = PartialProvider()
        
        with pytest.raises(NotImplementedError) as exc_info:
            async for _ in provider.stream([], "model"):
                pass
        
        assert "PartialProvider" in str(exc_info.value)
        assert "stream()" in str(exc_info.value)
