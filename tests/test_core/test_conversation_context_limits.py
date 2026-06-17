"""Tests for conversation context size limiting."""

import pytest
from datetime import datetime, timezone
from expert_among_us.core.conversation import ConversationBuilder
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.file_chunk import FileChunk
from expert_among_us.models.query_result import CommitResult, FileChunkResult
from expert_among_us.llm.base import Message


def create_test_changelist(id: str, message: str, diff_size: int = 1000) -> Changelist:
    """Create a test changelist with specified diff size."""
    diff = "+" + ("x" * diff_size)  # Simple diff content
    return Changelist(
        id=id,
        expert_name="TestExpert",
        project_name="test-project",
        message=message,
        author="test@example.com",
        timestamp=datetime.now(timezone.utc),
        files=["test.py"],
        diff=diff
    )


def create_test_file_chunk(file_path: str, content_size: int = 1000) -> FileChunk:
    """Create a test file chunk with specified content size."""
    content = "x" * content_size
    return FileChunk(
        file_path=file_path,
        content=content,
        line_start=1,
        line_end=10,
        revision_id="abc123",
        chunk_index=0
    )


class TestConversationContextLimits:
    """Test suite for conversation context size limiting."""
    
    def test_filter_results_within_budget(self):
        """Test that results fitting within budget are all included."""
        # Create small results that should all fit
        results = [
            CommitResult(
                changelist=create_test_changelist("1", "Small change 1", diff_size=100),
                similarity_score=0.9,
                source="metadata"
            ),
            CommitResult(
                changelist=create_test_changelist("2", "Small change 2", diff_size=100),
                similarity_score=0.8,
                source="metadata"
            ),
        ]
        
        builder = ConversationBuilder(
            prompt_generator=None,
            max_diff_chars=10000,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        system_prompt = builder._build_system_prompt(amogus=False)
        user_prompt = "Test question"
        
        filtered, stats = builder._filter_results_by_context_size(
            results, user_prompt, system_prompt
        )
        
        # All results should be included
        assert len(filtered) == 2
        assert stats['included'] == 2
        assert stats['filtered'] == 0
        assert stats['used'] < stats['available']
    
    def test_filter_results_exceeds_budget(self):
        """Test that results exceeding budget are filtered."""
        # Create large results that won't all fit
        results = [
            CommitResult(
                changelist=create_test_changelist("1", "Large change " + ("x" * 1000), diff_size=50000),
                similarity_score=0.9,
                source="metadata"
            ),
            CommitResult(
                changelist=create_test_changelist("2", "Large change " + ("y" * 1000), diff_size=50000),
                similarity_score=0.8,
                source="metadata"
            ),
            CommitResult(
                changelist=create_test_changelist("3", "Large change " + ("z" * 1000), diff_size=50000),
                similarity_score=0.7,
                source="metadata"
            ),
        ]
        
        builder = ConversationBuilder(
            prompt_generator=None,
            max_diff_chars=100000,
            max_context_tokens=10000,  # Very small budget
            max_response_tokens=1000
        )
        
        system_prompt = builder._build_system_prompt(amogus=False)
        user_prompt = "Test question"
        
        filtered, stats = builder._filter_results_by_context_size(
            results, user_prompt, system_prompt
        )
        
        # Not all results should be included
        assert len(filtered) < len(results)
        assert stats['included'] >= 1  # At least one included
        assert stats['filtered'] > 0
        # Top result should always be included (sorted by score)
        assert filtered[0].changelist.id == "1"
    
    def test_filter_always_includes_one(self):
        """Test that at least one result is always included even if over budget."""
        # Create one huge result that exceeds budget
        results = [
            CommitResult(
                changelist=create_test_changelist("1", "Huge change", diff_size=100000),
                similarity_score=0.9,
                source="metadata"
            ),
        ]
        
        builder = ConversationBuilder(
            prompt_generator=None,
            max_diff_chars=200000,
            max_context_tokens=1000,  # Very small budget
            max_response_tokens=500
        )
        
        system_prompt = builder._build_system_prompt(amogus=False)
        user_prompt = "Test question"
        
        filtered, stats = builder._filter_results_by_context_size(
            results, user_prompt, system_prompt
        )
        
        # First result should always be included
        assert len(filtered) == 1
        assert stats['included'] == 1
        assert filtered[0].changelist.id == "1"
    
    def test_filter_empty_results(self):
        """Test behavior with empty results list."""
        builder = ConversationBuilder(
            prompt_generator=None,
            max_diff_chars=10000,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        system_prompt = builder._build_system_prompt(amogus=False)
        user_prompt = "Test question"
        
        filtered, stats = builder._filter_results_by_context_size(
            [], user_prompt, system_prompt
        )
        
        assert len(filtered) == 0
        assert stats == {}
    
    def test_filter_mixed_commit_and_file_results(self):
        """Test filtering with both commit and file chunk results."""
        results = [
            CommitResult(
                changelist=create_test_changelist("1", "Commit 1", diff_size=1000),
                similarity_score=0.9,
                source="metadata"
            ),
            FileChunkResult(
                file_chunk=create_test_file_chunk("test.py", content_size=1000),
                similarity_score=0.85,
                source="file"
            ),
            CommitResult(
                changelist=create_test_changelist("2", "Commit 2", diff_size=1000),
                similarity_score=0.8,
                source="diff"
            ),
        ]
        
        builder = ConversationBuilder(
            prompt_generator=None,
            max_diff_chars=10000,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        system_prompt = builder._build_system_prompt(amogus=False)
        user_prompt = "Test question"
        
        filtered, stats = builder._filter_results_by_context_size(
            results, user_prompt, system_prompt
        )
        
        # All results should fit
        assert len(filtered) == 3
        assert stats['included'] == 3
        # Check that both types are preserved
        has_commit = any(isinstance(r, CommitResult) for r in filtered)
        has_file = any(isinstance(r, FileChunkResult) for r in filtered)
        assert has_commit and has_file
    
    def test_build_conversation_with_filtering(self):
        """Test end-to-end conversation building with context filtering."""
        results = [
            CommitResult(
                changelist=create_test_changelist("1", "Change 1", diff_size=500),
                similarity_score=0.9,
                source="metadata"
            ),
            CommitResult(
                changelist=create_test_changelist("2", "Change 2", diff_size=500),
                similarity_score=0.8,
                source="metadata"
            ),
        ]
        
        builder = ConversationBuilder(
            prompt_generator=None,
            max_diff_chars=10000,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        system_prompt, messages = builder.build_conversation(
            results=results,
            user_prompt="How to implement feature X?",
            amogus=False,
            impostor=False
        )
        
        # Check that conversation was built
        assert system_prompt
        assert len(messages) > 0
        # Last message should be user prompt
        assert messages[-1].role == "user"
        assert "How to implement feature X?" in messages[-1].content
        assert "Remember to structure your response" in messages[-1].content
    
    def test_build_conversation_raises_when_none_fit(self):
        """Test that build_conversation always includes at least one result."""
        # Create results with huge user prompt that leaves no room
        results = [
            CommitResult(
                changelist=create_test_changelist("1", "Change", diff_size=1000),
                similarity_score=0.9,
                source="metadata"
            ),
        ]
        
        builder = ConversationBuilder(
            prompt_generator=None,
            max_diff_chars=10000,
            max_context_tokens=1000,  # Very small
            max_response_tokens=500
        )
        
        # Create a huge user prompt that consumes most of budget
        huge_prompt = "x" * 10000
        
        # Since at least one result is always included, this should succeed
        system_prompt, messages = builder.build_conversation(
            results=results,
            user_prompt=huge_prompt,
            amogus=False,
            impostor=False
        )
        
        # Verify at least one result was included (commit message + final prompt)
        assert len(messages) >= 2