"""Abstract base classes and dataclasses for LLM providers."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional
from pydantic.dataclasses import dataclass


@dataclass
class Message:
    """Single message in a conversation.
    
    Attributes:
        role: Message role - "user", "assistant", or "system"
        content: Message content text
    """
    role: str
    content: str


@dataclass
class UsageMetrics:
    """Token usage and caching metrics for LLM responses.
    
    Attributes:
        input_tokens: Number of input tokens processed
        output_tokens: Number of output tokens generated
        total_tokens: Total tokens (input + output)
        cache_read_tokens: Number of tokens read from cache (0 if not cached)
        cache_creation_tokens: Number of tokens written to cache (0 if not caching)
    """
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class LLMResponse:
    """Complete response from an LLM.
    
    Attributes:
        content: Generated text content
        model: Model identifier used for generation
        stop_reason: Reason generation stopped - "end_turn", "max_tokens", or "stop_sequence"
        usage: Token usage and caching metrics
    """
    content: str
    model: str
    stop_reason: str
    usage: UsageMetrics


@dataclass
class StreamChunk:
    """Streaming response chunk from an LLM.
    
    Attributes:
        delta: Incremental text content for this chunk
        stop_reason: Optional reason generation stopped (present in final chunk)
        usage: Optional token usage and caching metrics (present in final chunk)
    """
    delta: str
    stop_reason: Optional[str] = None
    usage: Optional[UsageMetrics] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    Provides a consistent interface for interacting with different LLM backends.
    Implementations should handle provider-specific details like authentication,
    API formatting, and error handling.
    
    Provider-specific configuration (e.g., caching, region) should be handled
    in the constructor, not in method parameters. This ensures perfect encapsulation
    and makes consumer code provider-agnostic.
    """
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> LLMResponse:
        """Generate a complete response from the LLM.
        
        Args:
            messages: Conversation history as a list of Message objects
            model: Model identifier to use for generation
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-2.0 (default: 1.0)
            system: Optional system prompt to set behavior
            debug_category: Category for debug logging (default: "expert")
            
        Returns:
            LLMResponse containing generated content and metadata
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate() method"
        )
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks from the LLM.
        
        Args:
            messages: Conversation history as a list of Message objects
            model: Model identifier to use for generation
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-2.0 (default: 1.0)
            system: Optional system prompt to set behavior
            debug_category: Category for debug logging (default: "expert")
            
        Yields:
            StreamChunk objects containing incremental content
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stream() method"
        )
        # Make this a proper async generator
        yield  # pragma: no cover


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded when calling LLM API."""
    pass


class LLMInvalidRequestError(LLMError):
    """Invalid request parameters for LLM API."""
    pass