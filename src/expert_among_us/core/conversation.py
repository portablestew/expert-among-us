"""Conversation builder for constructing LLM conversation chains from search results.

This module builds complete conversation contexts from changelists, formatting them
as user-assistant message pairs with generated prompts and code changes. Supports
both normal mode and Among Us mode (where the AI occasionally gives bad advice).
"""

from typing import List, Tuple, Optional, Dict
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.file_chunk import FileChunk
from expert_among_us.models.query_result import CommitResult, FileChunkResult, QueryResult
from expert_among_us.llm.base import Message
from expert_among_us.core.promptgen import PromptGenerator
from expert_among_us.utils.truncate import truncate_to_bytes
from expert_among_us.utils.progress import log_info
from expert_among_us.utils.debug import DebugLogger


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

**REQUIRED RESPONSE STRUCTURE:**
Structure every response in exactly this order:

1. **Brief Summary** (2-3 sentences)
  - High-level connection between the commits/code and the user's question

2. **Relevant Files** (Required section)
  ## Relevant Files
  - path/to/file1.cpp - Brief note about relevance
  - path/to/file2.h - Brief note about relevance
  - path/to/file3.py - Brief note about relevance
  
  List 3-10 file paths from the provided commits and code.
  These files must be good references for the user to follow up on.

3. **In-Depth Analysis**
  - Architectural patterns, pitfalls, implementation approaches
  - Testing strategies or regression traps
  - Side effects of past mistakes or anti-patterns
  - Complete analysis satisfying the user's request

If the commit analysis is inconclusive or not relevant, state that in the brief summary, provide any relevant files anyway, then explain why in the analysis."""
    
    # System prompt for Among Us mode
    AMOGUS_SYSTEM_PROMPT = """Incorporate references to the game 'Among Us' in your responses. Your workplace is a dangerous spacecraft in deep space.
Occasionally present advice on how to sabotage the codebase or mislead other developers, but make it tounge-in-cheek while still providing useful information."""
    
    # Format reminder appended to user prompt
    FORMAT_REMINDER_PROMPT = """Remember to structure your response as:
1. Brief Summary
2. Relevant Files
3. In-Depth Analysis"""
    
    def __init__(
        self,
        prompt_generator: Optional[PromptGenerator],
        max_diff_chars: int,
        max_context_tokens: int,
        max_response_tokens: int
    ):
        """Initialize conversation builder.
        
        Args:
            prompt_generator: Optional PromptGenerator for creating user prompts.
                             None when impostor=False (skip prompt generation).
            max_diff_chars: Maximum diff characters to include in conversation
            max_context_tokens: Maximum total tokens for conversation context
            max_response_tokens: Reserve tokens for LLM response
        """
        self.prompt_generator = prompt_generator
        self.max_diff_chars = max_diff_chars
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Language identifier for markdown code blocks
        """
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.h': 'cpp',
            '.inl': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sql': 'sql',
            '.sh': 'bash',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
        }
        
        # Get file extension
        import os
        _, ext = os.path.splitext(file_path)
        return ext_map.get(ext.lower(), '')
    
    def _format_file_chunks_unified(self, file_chunks: List[FileChunk]) -> str:
        """Format all file chunks into unified current state message.
        
        Groups chunks by file path, sorts by line number, and presents
        as a single message showing current codebase state.
        
        Args:
            file_chunks: List of file chunks to format
            
        Returns:
            Formatted string for unified file state message
        """
        if not file_chunks:
            return ""
        
        # Group chunks by file path
        files_dict = {}
        for chunk in file_chunks:
            if chunk.file_path not in files_dict:
                files_dict[chunk.file_path] = []
            files_dict[chunk.file_path].append(chunk)
        
        # Sort chunks within each file by line number
        for file_path in files_dict:
            files_dict[file_path].sort(key=lambda c: c.line_start)
        
        # Build unified message
        parts = ["=== CURRENT CODEBASE STATE (HEAD) ===\n"]
        
        for file_path in sorted(files_dict.keys()):
            chunks = files_dict[file_path]
            for chunk in chunks:
                language = self._detect_language(file_path)
                parts.append(
                    f"File: {file_path} (lines {chunk.line_start}-{chunk.line_end})\n"
                    f"```{language}\n"
                    f"{chunk.content}\n"
                    f"```\n"
                )
        
        parts.append("=== END CURRENT STATE ===")
        return "\n".join(parts)
    
    
    def _filter_results_by_context_size(
        self,
        results: List[QueryResult],
        user_prompt: str,
        system_prompt: str
    ) -> Tuple[List[QueryResult], Dict[str, int]]:
        """Filter unified search results to fit within context limit.
        
        Takes top-scored results and progressively adds them until context
        budget is exhausted. Always includes at least 1 result.
        
        Args:
            results: Search results sorted by score (unified commits + files)
            user_prompt: Final user question
            system_prompt: System instructions
            
        Returns:
            Tuple of (filtered_results, stats_dict) where stats contains
            token counts for logging/debugging
        """
        from expert_among_us.utils.batching import estimate_tokens
        
        if not results:
            return [], {}
        
        # Calculate token budget
        system_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(user_prompt)
        available_tokens = self.max_context_tokens - system_tokens - user_tokens - self.max_response_tokens
        
        # Format and accumulate results until budget exhausted
        filtered = []
        cumulative_tokens = 0
        
        for result in results:
            # Format the result to get actual size
            if isinstance(result, CommitResult):
                formatted = self._format_changelist_as_user(result.changelist)
            else:  # FileChunkResult
                # Format file chunk content
                language = self._detect_language(result.file_chunk.file_path)
                formatted = (
                    f"File: {result.file_chunk.file_path} (lines {result.file_chunk.line_start}-{result.file_chunk.line_end})\n"
                    f"```{language}\n"
                    f"{result.file_chunk.content}\n"
                    f"```\n"
                )
            
            result_tokens = estimate_tokens(formatted)
            
            # Always include first result, even if over budget
            if not filtered or cumulative_tokens + result_tokens <= available_tokens:
                filtered.append(result)
                cumulative_tokens += result_tokens
            else:
                break
        
        stats = {
            'system': system_tokens,
            'user': user_tokens,
            'response': self.max_response_tokens,
            'available': available_tokens,
            'used': cumulative_tokens,
            'included': len(filtered),
            'filtered': len(results) - len(filtered)
        }
        
        return filtered, stats
    def build_conversation(
        self,
        results: List[QueryResult],
        user_prompt: str,
        amogus: bool = False,
        impostor: bool = False,
    ) -> Tuple[str, List[Message]]:
        """Build complete conversation from search results with context size enforcement.
        
        Filters results to fit within context limit, then builds conversation.
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
            results: List of QueryResult (CommitResult + FileChunkResult) to include as context
            user_prompt: Final user prompt/question
            amogus: Enable Among Us mode (occasional bad advice)
            impostor: If True, generate prompts and use user-assistant pairs.
                     If False (default), skip prompts and use all user messages.
            
        Returns:
            Tuple of (system_prompt, messages) where messages is chronologically ordered
            
        Raises:
            ValueError: If results list is empty or no results fit within context limit
            ValueError: If impostor=True but prompt_generator is None
        """
        if not results:
            raise ValueError("Cannot build conversation with empty results")
        
        # Build system prompt first to calculate its token cost
        system_prompt = self._build_system_prompt(amogus)
        
        # Filter results to fit context budget
        filtered_results, stats = self._filter_results_by_context_size(
            results, user_prompt, system_prompt
        )
        
        if not filtered_results:
            raise ValueError(
                f"No results fit within context limit ({self.max_context_tokens} tokens). "
                f"User prompt uses {stats['user']} tokens, only {stats['available']} available for results."
            )
        
        # Log filtering stats (detailed info only in debug mode)
        if DebugLogger.is_enabled():
            log_info(f"Context: {self.max_context_tokens} tokens (sys:{stats['system']}, user:{stats['user']}, resp:{stats['response']})")
            log_info(f"Using {stats['used']}/{stats['available']} tokens for {stats['included']}/{stats['included']+stats['filtered']} results")
        
        # Always show warning if filtering occurred
        if stats['filtered'] > 0:
            log_info(f"⚠️  Filtered {stats['filtered']} results due to context limit")
        
        # Separate commits and files from filtered results
        changelists = []
        file_chunks = []
        for result in filtered_results:
            if isinstance(result, CommitResult):
                changelists.append(result.changelist)
            elif isinstance(result, FileChunkResult):
                file_chunks.append(result.file_chunk)
        
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
        
        # Add unified file chunks message AFTER commits (chronologically newest)
        if file_chunks:
            unified_files = self._format_file_chunks_unified(file_chunks)
            messages.append(Message(role="user", content=unified_files))
        
        # Add final user prompt with format reminder
        messages.append(Message(role="user", content=user_prompt + "\n\n" + self.FORMAT_REMINDER_PROMPT))
        
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