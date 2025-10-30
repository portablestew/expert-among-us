"""AWS Bedrock LLM provider implementation using Converse API."""

import boto3
from botocore.exceptions import ClientError
from typing import AsyncIterator, List, Optional, Dict, Any

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


class BedrockLLM(LLMProvider):
    """AWS Bedrock implementation using Converse API.
    
    Supports Claude Sonnet and Nova Lite models with both synchronous
    and streaming generation capabilities.
    """
    
    def __init__(
        self,
        region_name: str = "us-west-2",
        profile_name: Optional[str] = None,
        enable_caching: bool = True,
    ):
        """Initialize Bedrock client.
        
        Args:
            region_name: AWS region (default: us-west-2)
            profile_name: AWS profile name (optional)
            enable_caching: Enable prompt caching for system and conversation history (default: True)
        """
        session = boto3.Session(
            region_name=region_name,
            profile_name=profile_name
        )
        self.client = session.client("bedrock-runtime")
        self.enable_caching = enable_caching
    
    def _to_bedrock_format(
        self,
        messages: List[Message],
        enable_caching: bool = False
    ) -> List[Dict[str, Any]]:
        """Convert Message objects to Bedrock Converse API format.
        
        Args:
            messages: List of Message objects
            enable_caching: If True, add a cache point after the last historical message
            
        Returns:
            List of messages in Bedrock format
        """
        formatted = []
        for i, msg in enumerate(messages):
            content_blocks = [{"text": msg.content}]
            
            # Add cache point only after the last historical message (second-to-last in the list)
            # This caches the entire conversation history while keeping the final user message uncached
            # Combined with the system prompt cache point, this uses only 2 of the 4 allowed cache points
            if enable_caching and i == len(messages) - 2:
                content_blocks.append({"cachePoint": {"type": "default"}})
            
            message_dict = {"role": msg.role, "content": content_blocks}
            formatted.append(message_dict)
        
        return formatted
    
    def _handle_error(self, error: ClientError) -> None:
        """Map Bedrock errors to LLMError types.
        
        Args:
            error: Boto3 ClientError
            
        Raises:
            LLMRateLimitError: If rate limit exceeded
            LLMInvalidRequestError: If request parameters invalid
            LLMError: For other Bedrock errors
        """
        code = error.response["Error"]["Code"]
        msg = error.response["Error"]["Message"]
        
        if code == "ThrottlingException":
            raise LLMRateLimitError(f"Rate limit exceeded: {msg}")
        elif code == "ValidationException":
            raise LLMInvalidRequestError(f"Invalid request: {msg}")
        else:
            raise LLMError(f"Bedrock error ({code}): {msg}")
    
    def generate(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> LLMResponse:
        """Generate complete response using Converse API.
        
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
            LLMError: For other Bedrock errors
        """
        request = {
            "modelId": model,
            "messages": self._to_bedrock_format(messages, enable_caching=self.enable_caching),
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        if system:
            system_blocks = [{"text": system}]
            # Add cache point as a separate block after the text when caching is enabled
            if self.enable_caching:
                system_blocks.append({"cachePoint": {"type": "default"}})
            request["system"] = system_blocks
        
        # Log request if debug enabled
        request_id = None
        if DebugLogger.is_enabled():
            request_id = DebugLogger.log_request("converse", request, category=debug_category)
        
        try:
            response = self.client.converse(**request)
            
            # Log response if debug enabled
            if DebugLogger.is_enabled():
                DebugLogger.log_response("converse", response, request_id, category=debug_category)
            
            # Extract usage metrics
            usage = response["usage"]
            usage_metrics = UsageMetrics(
                input_tokens=usage["inputTokens"],
                output_tokens=usage["outputTokens"],
                total_tokens=usage.get("totalTokens", usage["inputTokens"] + usage["outputTokens"]),
                cache_read_tokens=usage.get("cacheReadInputTokens", 0),
                cache_creation_tokens=usage.get("cacheWriteInputTokens", 0),
            )
            
            return LLMResponse(
                content=response["output"]["message"]["content"][0]["text"],
                model=model,
                stop_reason=response["stopReason"],
                usage=usage_metrics
            )
        except ClientError as e:
            self._handle_error(e)
    
    async def stream(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> AsyncIterator[StreamChunk]:
        """Stream response using Converse Stream API.
        
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
            LLMError: For other Bedrock errors
        """
        request = {
            "modelId": model,
            "messages": self._to_bedrock_format(messages, enable_caching=self.enable_caching),
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        if system:
            system_blocks = [{"text": system}]
            # Add cache point as a separate block after the text when caching is enabled
            if self.enable_caching:
                system_blocks.append({"cachePoint": {"type": "default"}})
            request["system"] = system_blocks
        
        # Log request if debug enabled
        request_id = None
        if DebugLogger.is_enabled():
            request_id = DebugLogger.log_request("converse_stream", request, category=debug_category)
        
        try:
            response = self.client.converse_stream(**request)
            stream = response["stream"]
            
            # Collect all events for debug logging
            all_events = []
            
            for event in stream:
                if DebugLogger.is_enabled():
                    all_events.append(event)
                
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]["text"]
                    yield StreamChunk(delta=delta)
                
                elif "metadata" in event:
                    # Final event with usage
                    metadata = event["metadata"]
                    usage_data = metadata.get("usage", {})
                    
                    # Create UsageMetrics object
                    usage_metrics = UsageMetrics(
                        input_tokens=usage_data.get("inputTokens", 0),
                        output_tokens=usage_data.get("outputTokens", 0),
                        total_tokens=usage_data.get("totalTokens", 0),
                        cache_read_tokens=usage_data.get("cacheReadInputTokens", 0),
                        cache_creation_tokens=usage_data.get("cacheWriteInputTokens", 0),
                    )
                    
                    yield StreamChunk(
                        delta="",
                        stop_reason=metadata.get("stopReason"),
                        usage=usage_metrics
                    )
            
            # Log all events if debug enabled
            if DebugLogger.is_enabled():
                DebugLogger.log_response("converse_stream", {"events": all_events}, request_id, category=debug_category)
                
        except ClientError as e:
            self._handle_error(e)