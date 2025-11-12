"""Public API for Expert Among Us library usage.

This module provides the public API for using Expert Among Us as a library,
enabling programmatic access to expert querying, AI recommendations, and
expert management.

Available exports:
    - Context: ExpertContext for managing dependencies
    - Operations: query_expert, list_experts, import_expert
    - Streaming: prompt_expert, prompt_expert_stream (Phase 3)
    - Exceptions: Custom exceptions for error handling
    - Models: Request/response models

Example:
    ```python
    from expert_among_us.api import (
        query_expert,
        list_experts,
        prompt_expert_stream,
        ExpertNotFoundError
    )
    
    # Query an expert
    try:
        results = query_expert("MyExpert", "How to add feature?", enable_multiprocessing=True)
        for result in results:
            print(f"Score: {result.similarity_score:.3f}")
            print(f"Message: {result.changelist.message}")
    except ExpertNotFoundError as e:
        print(f"Error: {e}")
    
    # Stream AI recommendations
    async for chunk in prompt_expert_stream("MyExpert", "Implement X", enable_multiprocessing=True):
        if chunk.delta:
            print(chunk.delta, end="")
    
    # List all experts
    experts = list_experts()
    for expert in experts:
        print(f"{expert.name}: {expert.commit_count} commits")
    ```
"""

# Context manager
from .context import ExpertContext

# Operations (Phase 2)
from .operations import (
    query_expert,
    list_experts,
    import_expert,
)

# Streaming (Phase 3)
from .streaming import (
    prompt_expert,
    prompt_expert_stream,
)

# Exceptions
from .exceptions import (
    ExpertAPIError,
    ExpertNotFoundError,
    ExpertAlreadyExistsError,
    InvalidExpertError,
    NoResultsError,
)

# Models
from .models import (
    ExpertInfo,
    PromptResponse,
    QueryRequest,
    PromptRequest,
    ImportRequest,
)

__all__ = [
    # Context
    'ExpertContext',
    
    # Operations
    'query_expert',
    'list_experts',
    'import_expert',
    
    # Streaming
    'prompt_expert',
    'prompt_expert_stream',
    
    # Exceptions
    'ExpertAPIError',
    'ExpertNotFoundError',
    'ExpertAlreadyExistsError',
    'InvalidExpertError',
    'NoResultsError',
    
    # Models
    'ExpertInfo',
    'PromptResponse',
    'QueryRequest',
    'PromptRequest',
    'ImportRequest',
]

# Version info
__version__ = '0.1.0-phase3'