"""OpenAI-compatible LLM provider implementation."""

from typing import AsyncIterator, List, Optional, Dict
import openai
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError

from .base import (
    LLMProvider,
    Message,
    LLMResponse,
    StreamChunk,
    UsageMetrics,
    LLMError,
    LLMRateLimitError,
    LLMInvalidRequestError,
)
from ..utils.debug import DebugLogger


class OpenAICompatibleLLM(LLMProvider):
    """OpenAI-compatible LLM provider implementation.
    
    Supports any OpenAI-compatible API including OpenAI, OpenRouter, and other
    providers that follow the OpenAI API format.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        """Initialize OpenAI-compatible client.
        
        Args:
            api_key: API key for authentication
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            base_url: Custom base URL (defaults to OpenAI's standard URL)
            organization: Organization ID
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts (default: 3)
            extra_headers: Additional HTTP headers (for OpenRouter support)
            debug: Enable debug logging (default: False)
        """
        self.model = model
        self.debug = debug
        
        # Build client configuration
        client_kwargs = {
            "api_key": api_key,
            "max_retries": max_retries,
        }
        
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        if timeout:
            client_kwargs["timeout"] = timeout
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
        
        self.client = OpenAI(**client_kwargs)
    
    def _to_openai_format(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert internal Message objects to OpenAI API format.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of dicts with "role" and "content" keys
        """
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role,
                "content": msg.content
            })
        return formatted
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Map OpenAI exceptions to custom LLMError types.
        
        Args:
            error: Exception from OpenAI client
            
        Returns:
            Appropriate LLMError subclass instance
        """
        if isinstance(error, RateLimitError):
            return LLMRateLimitError(f"Rate limit exceeded: {str(error)}")
        elif isinstance(error, AuthenticationError):
            return LLMInvalidRequestError(f"Authentication failed: {str(error)}")
        elif isinstance(error, APIConnectionError):
            return LLMError(f"Connection error: {str(error)}")
        elif isinstance(error, APIError):
            return LLMError(f"API error: {str(error)}")
        else:
            return LLMError(f"Unexpected error: {str(error)}")
    
    def generate(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> LLMResponse:
        """Generate complete response using OpenAI API.
        
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
            LLMRateLimitError: If rate limit exceeded
            LLMInvalidRequestError: If request parameters invalid
            LLMError: For other OpenAI API errors
        """
        # Convert messages to OpenAI format
        formatted_messages = self._to_openai_format(messages)
        
        # Add system message if provided
        if system:
            formatted_messages.insert(0, {"role": "system", "content": system})
        
        request = {
            "model": model or self.model,
            "messages": formatted_messages,
            "max_completion_tokens": max_tokens,
        }
        
        # Log request if debug enabled
        request_id = None
        if DebugLogger.is_enabled() or self.debug:
            request_id = DebugLogger.log_request("openai_generate", request, category=debug_category)
        
        try:
            response = self.client.chat.completions.create(**request)
            
            # Log response if debug enabled
            if DebugLogger.is_enabled() or self.debug:
                response_data = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
                DebugLogger.log_response("openai_generate", response_data, request_id, category=debug_category)
            
            # Extract content
            content = response.choices[0].message.content or ""
            
            # Extract usage metrics
            usage = response.usage
            usage_metrics = UsageMetrics(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cache_read_tokens=getattr(usage, 'prompt_tokens_details', {}).get('cached_tokens', 0) if hasattr(usage, 'prompt_tokens_details') else 0,
                cache_creation_tokens=0,
            )
            
            return LLMResponse(
                content=content,
                model=response.model,
                stop_reason=response.choices[0].finish_reason or "stop",
                usage=usage_metrics
            )
        except Exception as e:
            raise self._handle_error(e)
    
    async def stream(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> AsyncIterator[StreamChunk]:
        """Stream response using OpenAI API.
        
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
            LLMRateLimitError: If rate limit exceeded
            LLMInvalidRequestError: If request parameters invalid
            LLMError: For other OpenAI API errors
        """
        # Convert messages to OpenAI format
        formatted_messages = self._to_openai_format(messages)
        
        # Add system message if provided
        if system:
            formatted_messages.insert(0, {"role": "system", "content": system})
        
        request = {
            "model": model or self.model,
            "messages": formatted_messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        
        # Log request if debug enabled
        request_id = None
        if DebugLogger.is_enabled() or self.debug:
            request_id = DebugLogger.log_request("openai_stream", request, category=debug_category)
        
        try:
            stream = self.client.chat.completions.create(**request)
            
            # Collect all chunks for debug logging
            all_chunks = []
            final_usage = None
            final_stop_reason = None
            
            for chunk in stream:
                if DebugLogger.is_enabled() or self.debug:
                    chunk_data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk.dict()
                    all_chunks.append(chunk_data)
                
                # Process delta content
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Extract content delta
                    if choice.delta and choice.delta.content:
                        yield StreamChunk(delta=choice.delta.content)
                    
                    # Extract stop reason
                    if choice.finish_reason:
                        final_stop_reason = choice.finish_reason
                
                # Extract usage from final chunk (if available)
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage = chunk.usage
            
            # Yield final chunk with usage metrics
            if final_usage or final_stop_reason:
                usage_metrics = None
                if final_usage:
                    usage_metrics = UsageMetrics(
                        input_tokens=final_usage.prompt_tokens,
                        output_tokens=final_usage.completion_tokens,
                        total_tokens=final_usage.total_tokens,
                        cache_read_tokens=getattr(final_usage, 'prompt_tokens_details', {}).get('cached_tokens', 0) if hasattr(final_usage, 'prompt_tokens_details') else 0,
                        cache_creation_tokens=0,
                    )
                
                yield StreamChunk(
                    delta="",
                    stop_reason=final_stop_reason,
                    usage=usage_metrics
                )
            
            # Log all chunks if debug enabled
            if DebugLogger.is_enabled() or self.debug:
                DebugLogger.log_response("openai_stream", {"chunks": all_chunks}, request_id, category=debug_category)
                
        except Exception as e:
            raise self._handle_error(e)