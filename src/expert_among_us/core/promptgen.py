"""Prompt generation engine for creating plausible user prompts from changelists.

This module generates task-oriented prompts from code changes using LLM,
with SQLite caching to avoid redundant generation calls.
"""

from typing import Dict, List
import warnings

from expert_among_us.models.changelist import Changelist
from expert_among_us.db.metadata.base import MetadataDB
from expert_among_us.llm.base import LLMProvider, Message, LLMError
from expert_among_us.utils.truncate import truncate_to_bytes
from expert_among_us.utils.progress import log_info, log_warning, create_progress_bar, update_progress


class PromptGenerator:
    """Generates plausible user prompts from code changes using LLM with caching.
    
    Uses Nova Lite to infer what task, feature, or bugfix request might have
    led to a given code change. Results are cached in the metadata database
    to avoid redundant LLM calls.
    
    Attributes:
        llm: LLM provider for generating prompts
        metadata_db: Database for caching generated prompts
        model: Model identifier to use (typically Nova Lite)
        max_diff_chars: Maximum diff characters to send to LLM
    """
    
    # System prompt template for prompt generation
    SYSTEM_PROMPT = """You are an expert at inferring developer intent from code changes.

Given a code change (commit message and diff), generate a concise, realistic prompt that a developer might have written to request these changes.

Rules:
- Write 2-3 sentences maximum
- Focus on the task or feature being requested, not implementation details
- Sound natural and conversational, like a real developer would write
- Focus on the 'what' and 'why', not the 'how'
- Don't mention technical terms like "git diff", "commit", or "code change"

Example:
Input: "Add null checks to user service" + diff showing null validation
Output: "Add proper null handling to the user service to prevent crashes when processing user data."
"""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        metadata_db: MetadataDB,
        model: str,
        max_diff_chars: int = 2000,
    ):
        """Initialize prompt generator with dependencies.
        
        Args:
            llm_provider: LLM provider for generating prompts
            metadata_db: Metadata database for caching
            model: Model identifier (e.g., us.amazon.nova-lite-v1:0)
            max_diff_chars: Maximum diff characters for LLM (default: 2000)
        """
        self.llm = llm_provider
        self.metadata_db = metadata_db
        self.model = model
        self.max_diff_chars = max_diff_chars
    
    def generate_prompts(
        self,
        changelists: List[Changelist],
    ) -> Dict[str, str]:
        """Generate prompts for multiple changelists with caching.
        
        Processes changelists in batch, checking cache first and only
        generating for cache misses. Results are saved incrementally.
        
        Args:
            changelists: List of changelists to generate prompts for
            
        Returns:
            Dictionary mapping changelist IDs to generated prompts
        """
        results = {}
        cache_hits = 0
        cache_misses = 0
        errors = 0
        
        # Create progress bar for prompt generation
        progress, task_id = create_progress_bar(
            description="Generating prompts",
            total=len(changelists)
        )
        progress.start()
        
        try:
            for i, changelist in enumerate(changelists, 1):
                # Check cache first
                cached_prompt = self.metadata_db.get_generated_prompt(changelist.id)
                
                if cached_prompt:
                    results[changelist.id] = cached_prompt
                    cache_hits += 1
                    update_progress(
                        progress,
                        task_id,
                        advance=1,
                        description=f"Loading from cache ({i}/{len(changelists)}): {changelist.id[:8]}..."
                    )
                    continue
                
                # Generate new prompt
                try:
                    update_progress(
                        progress,
                        task_id,
                        advance=0,  # Don't advance yet, just update description
                        description=f"Generating with LLM ({i}/{len(changelists)}): {changelist.id[:8]}..."
                    )
                    
                    prompt = self._generate_single_prompt(changelist)
                    results[changelist.id] = prompt
                    
                    # Cache immediately (incremental caching)
                    self.metadata_db.cache_generated_prompt(changelist.id, prompt)
                    changelist.generated_prompt = prompt
                    cache_misses += 1
                    
                    # Now advance after successful generation
                    update_progress(progress, task_id, advance=1)
                        
                except LLMError as e:
                    log_warning(f"Failed to generate prompt for {changelist.id}: {str(e)}")
                    errors += 1
                    update_progress(progress, task_id, advance=1)
                    # Skip this changelist and continue with others
                    continue
        finally:
            progress.stop()
        
        # Final summary
        log_info(
            f"Prompt generation complete: {cache_hits} cached, "
            f"{cache_misses} generated, {errors} errors"
        )
        
        return results
    
    def _generate_single_prompt(self, changelist: Changelist) -> str:
        """Generate prompt for a single changelist using LLM.
        
        Args:
            changelist: Changelist to generate prompt for
            
        Returns:
            Generated prompt text
            
        Raises:
            LLMError: If LLM generation fails
        """
        # Build the request with truncated diff
        request_text = self._build_prompt_request(changelist)
        
        # Generate using LLM
        messages = [Message(role="user", content=request_text)]
        
        response = self.llm.generate(
            messages=messages,
            model=self.model,
            max_tokens=500,  # Prompts should be brief
            temperature=0.7,  # Some variety
            system=self.SYSTEM_PROMPT,
            debug_category="promptgen",
        )
        
        # Clean up and return
        prompt = response.content.strip()
        
        # Remove any quotes that LLM might add
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1]
        if prompt.startswith("'") and prompt.endswith("'"):
            prompt = prompt[1:-1]
        
        return prompt
    
    def _build_prompt_request(self, changelist: Changelist) -> str:
        """Build LLM request from changelist data.
        
        Combines commit message, files changed, and truncated diff into
        a format suitable for the LLM to infer developer intent.
        
        Args:
            changelist: Changelist to build request for
            
        Returns:
            Formatted request text for LLM
        """
        # Truncate diff to max_diff_chars
        diff = changelist.diff
        if len(diff) > self.max_diff_chars:
            diff, was_truncated = truncate_to_bytes(diff, self.max_diff_chars)
            if was_truncated:
                diff += "\n[... truncated for brevity ...]"
        
        # Build request text
        parts = [
            f"Commit Message: {changelist.message}",
            f"Files Changed: {', '.join(changelist.files[:10])}",  # Limit files shown
        ]
        
        if len(changelist.files) > 10:
            parts[-1] += f" (and {len(changelist.files) - 10} more)"
        
        parts.append(f"\nCode Changes:\n{diff}")
        
        return "\n".join(parts)