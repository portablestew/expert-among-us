"""Streaming interface for AI-powered expert recommendations.

This module provides async streaming generators for getting AI recommendations
based on an expert's historical patterns. It supports both streaming and
non-streaming modes for different use cases.
"""

from typing import AsyncIterator, Optional, List
from pathlib import Path

from expert_among_us.llm.base import StreamChunk
from .context import ExpertContext
from .exceptions import ExpertNotFoundError, NoResultsError
from .models import PromptResponse


# IMPORTANT: Usage Metrics Pattern
#
# LLM providers (e.g., BedrockLLM.stream()) accumulate usage metrics internally
# and only emit them in the FINAL chunk. The metrics represent the TOTAL for the
# entire request, not incremental values. Therefore:
#
# - DO: Capture the last chunk's usage (overwrite pattern: final_usage = chunk.usage)
# - DON'T: Accumulate across chunks (e.g., total += chunk.usage.total_tokens)
#
# See: src/expert_among_us/llm/bedrock.py:248 (metadata event with final usage)


async def prompt_expert_stream(
    expert_name: str,
    prompt: str,
    max_changes: int = 15,
    users: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    amogus: bool = False,
    impostor: bool = False,
    temperature: float = 0.7,
    data_dir: Optional[Path] = None,
    embedding_provider: str = "local",
    llm_provider: str = "auto",
) -> AsyncIterator[StreamChunk]:
    """Stream AI recommendations from expert.
    
    This is an async generator that yields StreamChunk objects as the
    LLM generates the response. The final chunk contains usage metrics.
    
    Args:
        expert_name: Name of the expert to query
        prompt: User's question or task
        max_changes: Maximum context changes
        users: Filter by authors
        files: Filter by file paths
        amogus: Enable Among Us mode
        impostor: Use impostor mode (generate prompts)
        temperature: LLM temperature
        data_dir: Custom data directory
        embedding_provider: Embedding provider
        llm_provider: LLM provider
        
    Yields:
        StreamChunk objects with incremental content
        
    Raises:
        ExpertNotFoundError: If expert doesn't exist
        NoResultsError: If no relevant examples found
        
    Note:
        Usage metrics are only present in the final chunk from the LLM provider.
        They are already accumulated internally by the provider (not incremental).
        
    Example:
        ```python
        async for chunk in prompt_expert_stream("MyExpert", "Add feature"):
            if chunk.delta:
                print(chunk.delta, end="")
            if chunk.usage:
                print(f"Tokens: {chunk.usage.total_tokens}")
        ```
    """
    from expert_among_us.core.searcher import Searcher
    from expert_among_us.core.promptgen import PromptGenerator
    from expert_among_us.core.conversation import ConversationBuilder
    from expert_among_us.llm.factory import create_llm_provider
    from expert_among_us.models.query import QueryParams
    
    ctx = ExpertContext(
        expert_name=expert_name,
        data_dir=data_dir,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider
    )
    
    try:
        # Check expert exists
        if not ctx.metadata_db.exists():
            raise ExpertNotFoundError(expert_name)
        
        ctx.vector_db.initialize(
            dimension=ctx.embedder.dimension,
            require_exists=True
        )
        
        # Search for relevant examples
        searcher = Searcher(
            expert_name=expert_name,
            embedder=ctx.embedder,
            metadata_db=ctx.metadata_db,
            vector_db=ctx.vector_db
        )
        
        params = QueryParams(
            prompt=prompt,
            max_changes=max_changes,
            users=users,
            files=files,
            amogus=False
        )
        
        search_results = searcher.search(params)
        
        if not search_results:
            raise NoResultsError("No relevant examples found for this query")
        
        # Generate prompts if impostor mode
        changelists = [result.changelist for result in search_results]
        
        if impostor:
            llm = create_llm_provider(ctx.settings)
            prompt_gen = PromptGenerator(
                llm_provider=llm,
                metadata_db=ctx.metadata_db,
                model=ctx.settings.promptgen_model,
                max_diff_chars=ctx.settings.max_diff_chars_for_promptgen
            )
            prompt_gen.generate_prompts(changelists)
        else:
            prompt_gen = None
        
        # Build conversation
        conv_builder = ConversationBuilder(
            prompt_generator=prompt_gen,
            max_diff_chars=ctx.settings.max_diff_chars_for_llm
        )
        system_prompt, messages = conv_builder.build_conversation(
            changelists=changelists,
            user_prompt=prompt,
            amogus=amogus,
            impostor=impostor
        )
        
        # Stream response
        llm = create_llm_provider(ctx.settings)
        async for chunk in llm.stream(
            messages=messages,
            model=ctx.settings.expert_model,
            system=system_prompt,
            max_tokens=4096,
            temperature=temperature,
        ):
            yield chunk
            
    finally:
        ctx.close()


async def prompt_expert(
    expert_name: str,
    prompt: str,
    **kwargs
) -> PromptResponse:
    """Non-streaming version that returns complete response.
    
    This is a convenience wrapper around prompt_expert_stream() that
    accumulates all chunks and returns a complete PromptResponse object.
    Use this when you need the full response at once rather than streaming.
    
    Args:
        expert_name: Name of expert
        prompt: User's question
        **kwargs: Same as prompt_expert_stream (max_changes, users, files,
                 amogus, impostor, temperature, data_dir, embedding_provider,
                 llm_provider)
        
    Returns:
        PromptResponse with complete content and metrics
        
    Raises:
        ExpertNotFoundError: If expert doesn't exist
        NoResultsError: If no relevant examples found
        
    Note:
        Usage metrics are only present in the final chunk from the LLM provider.
        They are already accumulated internally by the provider (not incremental).
        
    Example:
        ```python
        response = await prompt_expert("MyExpert", "Add feature")
        print(response.content)
        print(f"Tokens used: {response.total_tokens}")
        ```
    """
    content = ""
    final_usage = None
    
    async for chunk in prompt_expert_stream(expert_name, prompt, **kwargs):
        if chunk.delta:
            content += chunk.delta
        if chunk.usage:
            # Usage is only in the final chunk, already accumulated by provider
            final_usage = chunk.usage
    
    return PromptResponse(
        content=content,
        input_tokens=final_usage.input_tokens if final_usage else 0,
        output_tokens=final_usage.output_tokens if final_usage else 0,
        total_tokens=final_usage.total_tokens if final_usage else 0,
        cache_read_tokens=final_usage.cache_read_tokens if final_usage else 0,
        cache_creation_tokens=final_usage.cache_creation_tokens if final_usage else 0,
    )