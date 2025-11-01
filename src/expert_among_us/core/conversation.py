"""Conversation builder for constructing LLM conversation chains from search results.

This module builds complete conversation contexts from changelists, formatting them
as user-assistant message pairs with generated prompts and code changes. Supports
both normal mode and Among Us mode (where the AI occasionally gives bad advice).
"""

from typing import List, Tuple, Optional
from expert_among_us.models.changelist import Changelist
from expert_among_us.llm.base import Message
from expert_among_us.core.promptgen import PromptGenerator
from expert_among_us.utils.truncate import truncate_to_bytes


class ConversationBuilder:
    """Builds conversation chains from changelists for LLM prompting.
    
    Supports two modes:
    1. Default mode (impostor=False): All commits as user messages, no prompt generation
    2. Impostor mode (impostor=True): User-assistant pairs with generated prompts
    
    Attributes:
        prompt_generator: Optional PromptGenerator for creating user prompts (None in default mode)
        max_diff_chars: Maximum diff characters to include
    """
    
    # System prompt for normal mode
    NORMAL_SYSTEM_PROMPT = """You are an expert software developer providing recommendations based on historical commit patterns. 
Analyze the provided code changes and respond with practical, actionable advice for the current request.
DO NOT USE any external tools, APIs, searches, or other resources; only the messages in this conversation.
Focus on architectural patterns, potential pitfalls, implementation approaches, that have worked in this codebase.
Include a list of the top most relevant *existing* files that provide context for the current request.
Also report relevant testing strategies or regression traps, such as changes that needed a follow-up fix later.
When appropriate, explain the side effect of observed past mistakes or anti-patterns.
If the commit analysis is inconclusive or not relevant to the request, include that assessment in your response."""
    
    # System prompt for Among Us mode
    AMOGUS_SYSTEM_PROMPT = """Incorporate references to the game 'Among Us' in your responses. Your workplace is a dangerous spacecraft in deep space.
Occasionally present advice on how to sabotage the codebase or mislead other developers, but make it tounge-in-cheek while still providing useful information."""
    
    def __init__(self, prompt_generator: Optional[PromptGenerator], max_diff_chars: int):
        """Initialize conversation builder.
        
        Args:
            prompt_generator: Optional PromptGenerator for creating user prompts.
                             None when impostor=False (skip prompt generation).
            max_diff_chars: Maximum diff characters to include in conversation
        """
        self.prompt_generator = prompt_generator
        self.max_diff_chars = max_diff_chars
    
    def build_conversation(
        self,
        changelists: List[Changelist],
        user_prompt: str,
        amogus: bool = False,
        impostor: bool = False,
    ) -> Tuple[str, List[Message]]:
        """Build complete conversation from changelists and user prompt.
        
        Supports two modes:
        
        1. Default mode (impostor=False):
           - Skip prompt generation
           - All commits as user messages
           - Faster and cheaper
        
        2. Impostor mode (impostor=True):
           - Generate AI prompts for each commit
           - Create user-assistant pairs
           - User = generated prompt, Assistant = commit
        
        Args:
            changelists: List of changelists to include as context
            user_prompt: Final user prompt/question
            amogus: Enable Among Us mode (occasional bad advice)
            impostor: If True, generate prompts and use user-assistant pairs.
                     If False (default), skip prompts and use all user messages.
            
        Returns:
            Tuple of (system_prompt, messages) where messages is chronologically ordered
            
        Raises:
            ValueError: If changelists list is empty
            ValueError: If impostor=True but prompt_generator is None
        """
        if not changelists:
            raise ValueError("Cannot build conversation with empty changelists")
        
        # Sort changelists chronologically
        sorted_changelists = sorted(changelists, key=lambda cl: cl.timestamp)
        
        # Build messages list
        messages: List[Message] = []
        
        if impostor:
            # Impostor mode: Generate prompts and create user-assistant pairs
            if self.prompt_generator is None:
                raise ValueError("prompt_generator required when impostor=True")
            
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
        else:
            # Default mode: All commits as user messages, no prompts
            for changelist in sorted_changelists:
                # Format and add as user message
                formatted_changelist = self._format_changelist_as_user(changelist)
                messages.append(Message(role="user", content=formatted_changelist))
        
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
            return self.NORMAL_SYSTEM_PROMPT + "\n\n" + self.AMOGUS_SYSTEM_PROMPT
        else:
            return self.NORMAL_SYSTEM_PROMPT
    
    def _format_changelist_as_assistant(self, changelist: Changelist) -> str:
        """Format changelist as assistant message.
        
        Formats as:
        ```
        Commit: {message}
        Files: {file1}, {file2}, ...
        Changes:
        {truncated diff}
        ```
        
        Args:
            changelist: Changelist to format
            
        Returns:
            Formatted string for assistant message
        """
        # Truncate diff to max_diff_chars
        diff = changelist.diff
        
        if len(diff) > self.max_diff_chars:
            diff, was_truncated = truncate_to_bytes(diff, self.max_diff_chars)
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
    
    def _format_changelist_as_user(self, changelist: Changelist) -> str:
        """Format changelist as user message (default mode).
        
        Uses same format as assistant messages to maintain consistency:
        ```
        Commit: {message}
        Files: {file1}, {file2}, ...
        Changes:
        {truncated diff}
        ```
        
        Args:
            changelist: Changelist to format
            
        Returns:
            Formatted string for user message
        """
        # Reuse assistant formatting logic for consistency
        return self._format_changelist_as_assistant(changelist)