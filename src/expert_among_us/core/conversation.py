"""Conversation builder for constructing LLM conversation chains from search results.

This module builds complete conversation contexts from changelists, formatting them
as user-assistant message pairs with generated prompts and code changes. Supports
both normal mode and Among Us mode (where the AI occasionally gives bad advice).
"""

from typing import List, Tuple
from expert_among_us.models.changelist import Changelist
from expert_among_us.llm.base import Message
from expert_among_us.core.promptgen import PromptGenerator
from expert_among_us.utils.truncate import truncate_to_bytes


class ConversationBuilder:
    """Builds conversation chains from changelists for LLM prompting.
    
    Constructs a conversation by:
    1. Creating user-assistant pairs from each changelist
    2. Using generated prompts as user messages
    3. Formatting changelists as assistant responses
    4. Appending the final user prompt
    5. Building appropriate system prompt
    
    Attributes:
        prompt_generator: Optional PromptGenerator for creating user prompts
    """
    
    # System prompt for normal mode
    NORMAL_SYSTEM_PROMPT = """You are an expert software developer providing recommendations based on historical commit patterns. 
Analyze the provided code changes and respond with practical, actionable advice for the current request.
Focus on architectural patterns, potential pitfalls, implementation approaches, that have worked in this codebase.
Include a list of the top most relevant *existing* files that provide context for the current request.
Also report relevant testing strategies or regression traps, such as changes that needed a follow-up fix later.
When appropriate, explain the side effect of observed past mistakes or anti-patterns.
If the commit analysis is inconclusive or not relevant to the request, include that assessment in your response."""
    
    # System prompt for Among Us mode
    AMOGUS_SYSTEM_PROMPT = NORMAL_SYSTEM_PROMPT + """
Incorporate references to the game 'Among Us' in your responses. Your workplace is a dangerous spacecraft in deep space.
Occasionally present advice on how to sabotage the codebase or mislead other developers, but make it tounge-in-cheek while still providing useful information."""
    
    def __init__(self, prompt_generator: PromptGenerator):
        """Initialize conversation builder with prompt generator.
        
        Args:
            prompt_generator: PromptGenerator for creating user prompts from diffs
        """
        self.prompt_generator = prompt_generator
    
    def build_conversation(
        self,
        changelists: List[Changelist],
        user_prompt: str,
        amogus: bool = False,
    ) -> Tuple[str, List[Message]]:
        """Build complete conversation from changelists and user prompt.
        
        Creates a conversation by:
        1. Sorting changelists chronologically
        2. For each changelist:
           - Generate user prompt (or use cached)
           - Format changelist as assistant response
        3. Append final user prompt
        4. Build appropriate system prompt
        
        Args:
            changelists: List of changelists to include as context
            user_prompt: Final user prompt/question
            amogus: Enable Among Us mode (occasional bad advice)
            
        Returns:
            Tuple of (system_prompt, messages) where messages is chronologically ordered
            
        Raises:
            ValueError: If changelists list is empty
        """
        if not changelists:
            raise ValueError("Cannot build conversation with empty changelists")
        
        # Sort changelists chronologically
        sorted_changelists = sorted(changelists, key=lambda cl: cl.timestamp)
        
        # Build messages list
        messages: List[Message] = []
        
        for changelist in sorted_changelists:
            # Get or generate prompt for this changelist
            if changelist.generated_prompt:
                generated_prompt = changelist.generated_prompt
            else:
                # Generate using prompt generator
                generated_prompt = self.prompt_generator._generate_single_prompt(changelist)
                changelist.generated_prompt = generated_prompt
            
            # Add user message with generated prompt
            messages.append(Message(role="user", content=generated_prompt))
            
            # Add assistant message with formatted changelist
            formatted_changelist = self._format_changelist_as_assistant(changelist)
            messages.append(Message(role="assistant", content=formatted_changelist))
        
        # Add final user prompt
        messages.append(Message(role="user", content=user_prompt))
        
        # Build system prompt
        system_prompt = self._build_system_prompt(amogus)
        
        return system_prompt, messages
    
    def _build_system_prompt(self, amogus: bool) -> str:
        """Build system prompt based on mode.
        
        Args:
            amogus: If True, use Among Us mode system prompt
            
        Returns:
            System prompt string
        """
        if amogus:
            return self.AMOGUS_SYSTEM_PROMPT
        else:
            return self.NORMAL_SYSTEM_PROMPT
    
    def _format_changelist_as_assistant(self, changelist: Changelist) -> str:
        """Format changelist as assistant message.
        
        Formats as:
        ```
        Commit: {message}
        Files: {file1}, {file2}, ...
        Changes:
        {truncated diff to ~3000 chars}
        ```
        
        Args:
            changelist: Changelist to format
            
        Returns:
            Formatted string for assistant message
        """
        # Truncate diff to ~3000 chars
        diff = changelist.diff
        max_diff_chars = 3000
        
        if len(diff) > max_diff_chars:
            diff, was_truncated = truncate_to_bytes(diff, max_diff_chars)
            if was_truncated:
                diff += "\n\n[... diff truncated for brevity ...]"
        
        # Format file list (show up to 10 files)
        file_list = ", ".join(changelist.files[:10])
        if len(changelist.files) > 10:
            file_list += f" (and {len(changelist.files) - 10} more)"
        
        # Build formatted string
        parts = [
            f"Commit: {changelist.message}",
            f"Files: {file_list}",
            f"Changes:\n{diff}",
        ]
        
        return "\n".join(parts)