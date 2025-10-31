"""Integration tests for the full prompt command flow."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path

from expert_among_us.models.changelist import Changelist
from expert_among_us.models.query import QueryParams
from expert_among_us.core.searcher import SearchResult
from expert_among_us.llm.base import Message, LLMResponse, StreamChunk, UsageMetrics
from expert_among_us.core.promptgen import PromptGenerator
from expert_among_us.core.conversation import ConversationBuilder

# Test constants for max_diff_chars
TEST_MAX_DIFF_CHARS_PROMPT = 2000
TEST_MAX_DIFF_CHARS_CONV = 3000


@pytest.fixture
def sample_changelists():
    """Create sample changelists for testing."""
    return [
        Changelist(
            id="commit1",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="alice@example.com",
            message="Add authentication",
            diff="diff --git a/auth.py b/auth.py\n+def authenticate(user):\n+    return True",
            files=["auth.py"],
        ),
        Changelist(
            id="commit2",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 16, 11, 0, 0),
            author="bob@example.com",
            message="Add validation",
            diff="diff --git a/validate.py b/validate.py\n+def validate(data):\n+    return data is not None",
            files=["validate.py"],
        ),
        Changelist(
            id="commit3",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 17, 12, 0, 0),
            author="charlie@example.com",
            message="Add error handling",
            diff="diff --git a/errors.py b/errors.py\n+try:\n+    process()\n+except Exception as e:\n+    log(e)",
            files=["errors.py"],
        ),
    ]


@pytest.fixture
def mock_searcher(sample_changelists):
    """Create a mock searcher that returns sample results."""
    mock = Mock()
    
    # Create SearchResult objects
    search_results = [
        SearchResult(
            changelist=cl,
            similarity_score=0.9 - (i * 0.1),
            source="metadata"
        )
        for i, cl in enumerate(sample_changelists)
    ]
    
    mock.search.return_value = search_results
    mock.close.return_value = None
    
    return mock


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    mock = Mock()
    
    # Mock generate method for prompt generation
    mock.generate.return_value = LLMResponse(
        content="Generated prompt for commit",
        model="us.amazon.nova-lite-v1:0",
        stop_reason="end_turn",
        usage=UsageMetrics(input_tokens=50, output_tokens=15, total_tokens=65)
    )
    
    # Mock stream method for final response
    async def mock_stream(*args, **kwargs):
        chunks = [
            StreamChunk(delta="This "),
            StreamChunk(delta="is "),
            StreamChunk(delta="a "),
            StreamChunk(delta="test "),
            StreamChunk(delta="response."),
            StreamChunk(
                delta="",
                stop_reason="end_turn",
                usage=UsageMetrics(input_tokens=200, output_tokens=50, total_tokens=250)
            )
        ]
        for chunk in chunks:
            yield chunk
    
    mock.stream = mock_stream
    
    return mock


@pytest.fixture
def mock_metadata_db():
    """Create a mock metadata database."""
    mock = Mock()
    mock.get_generated_prompt.return_value = None  # Force generation
    mock.cache_generated_prompt.return_value = None
    mock.get_expert.return_value = {"name": "TestExpert", "workspace": "/test/path"}
    return mock


class TestPromptCommandIntegration:
    """Integration tests for the prompt command flow."""
    
    @pytest.mark.asyncio
    async def test_full_prompt_flow_without_cache(
        self, mock_searcher, mock_llm, mock_metadata_db, sample_changelists
    ):
        """Test complete flow: search → prompt gen → conversation → LLM streaming."""
        
        # Step 1: Search for relevant changelists (mocked)
        params = QueryParams(
            prompt="How should I implement authentication?",
            max_changes=3,
            users=None,
            files=None,
            amogus=False
        )
        
        search_results = mock_searcher.search(params)
        assert len(search_results) == 3
        
        # Step 2: Generate prompts for changelists
        prompt_gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="us.amazon.nova-lite-v1:0",
            max_diff_chars=TEST_MAX_DIFF_CHARS_PROMPT
        )
        
        changelists = [r.changelist for r in search_results]
        prompt_results = prompt_gen.generate_prompts(changelists)
        
        # Verify prompts were generated
        assert len(prompt_results) == 3
        assert all(cl_id in prompt_results for cl_id in ["commit1", "commit2", "commit3"])
        
        # Verify LLM was called for each changelist
        assert mock_llm.generate.call_count == 3
        
        # Step 3: Build conversation context
        conv_builder = ConversationBuilder(prompt_generator=prompt_gen, max_diff_chars=TEST_MAX_DIFF_CHARS_CONV)
        system_prompt, messages = conv_builder.build_conversation(
            changelists=changelists,
            user_prompt="How should I implement authentication?",
            amogus=False
        )
        
        # Verify conversation structure
        assert len(messages) == 7  # 3 pairs + final user message
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[-1].role == "user"
        assert messages[-1].content == "How should I implement authentication?"
        
        # Verify system prompt
        assert "expert software developer" in system_prompt
        assert "impostor" not in system_prompt.lower()
        
        # Step 4: Stream LLM response
        full_response = ""
        final_usage = None
        
        async for chunk in mock_llm.stream(
            messages=messages,
            model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            system=system_prompt,
            max_tokens=4096,
            temperature=0.7
        ):
            if chunk.delta:
                full_response += chunk.delta
            if chunk.usage:
                final_usage = chunk.usage
        
        # Verify streaming worked
        assert full_response == "This is a test response."
        assert final_usage is not None
        assert final_usage.input_tokens == 200
        assert final_usage.output_tokens == 50
    
    @pytest.mark.asyncio
    async def test_full_prompt_flow_with_cache(
        self, mock_searcher, mock_llm, mock_metadata_db, sample_changelists
    ):
        """Test flow when prompts are already cached."""
        
        # Configure cache hits
        mock_metadata_db.get_generated_prompt.side_effect = [
            "Cached prompt 1",
            "Cached prompt 2",
            "Cached prompt 3"
        ]
        
        # Get search results
        params = QueryParams(
            prompt="Test query",
            max_changes=3,
            users=None,
            files=None,
            amogus=False
        )
        search_results = mock_searcher.search(params)
        
        # Generate prompts (should use cache)
        prompt_gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="us.amazon.nova-lite-v1:0",
            max_diff_chars=TEST_MAX_DIFF_CHARS_PROMPT
        )
        
        changelists = [r.changelist for r in search_results]
        prompt_results = prompt_gen.generate_prompts(changelists)
        
        # Verify cache was used (LLM not called)
        assert mock_llm.generate.call_count == 0
        assert len(prompt_results) == 3
        
        # Verify cached prompts were used
        assert prompt_results["commit1"] == "Cached prompt 1"
        assert prompt_results["commit2"] == "Cached prompt 2"
        assert prompt_results["commit3"] == "Cached prompt 3"
    
    @pytest.mark.asyncio
    async def test_prompt_flow_with_amogus_mode(
        self, mock_searcher, mock_llm, mock_metadata_db, sample_changelists
    ):
        """Test flow with Among Us mode enabled."""
        
        # Get search results
        search_results = mock_searcher.search(QueryParams(
            prompt="Test",
            max_changes=2,
            users=None,
            files=None,
            amogus=True
        ))
        
        # Generate prompts
        prompt_gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="us.amazon.nova-lite-v1:0",
            max_diff_chars=TEST_MAX_DIFF_CHARS_PROMPT
        )
        
        changelists = [r.changelist for r in search_results[:2]]
        prompt_gen.generate_prompts(changelists)
        
        # Build conversation with amogus=True
        conv_builder = ConversationBuilder(prompt_generator=prompt_gen, max_diff_chars=TEST_MAX_DIFF_CHARS_CONV)
        system_prompt, messages = conv_builder.build_conversation(
            changelists=changelists,
            user_prompt="Test query",
            amogus=True
        )
        
        # Verify Among Us elements in system prompt
        assert "Among Us" in system_prompt
        assert "sabotage" in system_prompt
        assert "mislead" in system_prompt
    
    @pytest.mark.asyncio
    async def test_prompt_flow_with_filters(
        self, mock_searcher, mock_llm, mock_metadata_db
    ):
        """Test flow with user and file filters."""
        
        # Create filtered search
        params = QueryParams(
            prompt="Test query",
            max_changes=5,
            users=["alice@example.com"],
            files=["auth.py"],
            amogus=False
        )
        
        search_results = mock_searcher.search(params)
        
        # Verify filters were passed
        mock_searcher.search.assert_called_once_with(params)
        assert params.users == ["alice@example.com"]
        assert params.files == ["auth.py"]
    
    @pytest.mark.asyncio
    async def test_prompt_flow_handles_empty_results(
        self, mock_llm, mock_metadata_db
    ):
        """Test flow handles empty search results gracefully."""
        
        # Create searcher that returns no results
        empty_searcher = Mock()
        empty_searcher.search.return_value = []
        
        params = QueryParams(
            prompt="Non-existent feature",
            max_changes=10,
            users=None,
            files=None,
            amogus=False
        )
        
        results = empty_searcher.search(params)
        
        # Should return empty list
        assert results == []
        
        # ConversationBuilder should raise error on empty changelists
        conv_builder = ConversationBuilder(None, TEST_MAX_DIFF_CHARS_CONV)
        with pytest.raises(ValueError, match="Cannot build conversation with empty changelists"):
            conv_builder.build_conversation(
                changelists=[],
                user_prompt="Test",
                amogus=False
            )


class TestPromptFlowEdgeCases:
    """Edge case tests for the prompt flow."""
    
    @pytest.mark.asyncio
    async def test_very_long_diff(self, mock_llm, mock_metadata_db):
        """Test handling of very long diffs."""
        
        long_diff = "diff --git a/file.py b/file.py\n" + ("+" + "x" * 100 + "\n") * 100
        
        changelist = Changelist(
            id="long123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Long change",
            diff=long_diff,
            files=["file.py"],
        )
        
        # Generate prompt (should truncate diff)
        prompt_gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="test-model",
            max_diff_chars=500
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Generated prompt",
            model="test-model",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=50, output_tokens=10, total_tokens=60)
        )
        
        prompt = prompt_gen._generate_single_prompt(changelist)
        
        # Should succeed without error
        assert prompt == "Generated prompt"
        
        # Verify truncation happened
        call_args = mock_llm.generate.call_args
        messages = call_args.kwargs["messages"]
        request_text = messages[0].content
        
        assert len(request_text) < len(long_diff)
        assert "truncated" in request_text.lower()
    
    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, mock_llm, mock_metadata_db):
        """Test handling of special characters in prompts."""
        
        changelist = Changelist(
            id="special123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Add support for <>&\"'",
            diff="diff --git a/file.py b/file.py\n+special = '<>&\"'",
            files=["file.py"],
        )
        
        prompt_gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="test-model",
            max_diff_chars=TEST_MAX_DIFF_CHARS_PROMPT
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Handle special characters properly",
            model="test-model",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=50, output_tokens=10, total_tokens=60)
        )
        
        # Should handle without errors
        prompt = prompt_gen._generate_single_prompt(changelist)
        assert prompt == "Handle special characters properly"
    
    @pytest.mark.asyncio
    async def test_unicode_in_diff(self, mock_llm, mock_metadata_db):
        """Test handling of Unicode characters in diffs."""
        
        changelist = Changelist(
            id="unicode123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Add emoji support 🎉",
            diff="diff --git a/file.py b/file.py\n+message = 'Hello 世界 🌍'",
            files=["file.py"],
        )
        
        prompt_gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="test-model",
            max_diff_chars=TEST_MAX_DIFF_CHARS_PROMPT
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Add internationalization support",
            model="test-model",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=50, output_tokens=10, total_tokens=60)
        )
        
        # Should handle Unicode properly
        prompt = prompt_gen._generate_single_prompt(changelist)
        assert prompt == "Add internationalization support"
    
    def test_conversation_with_many_changelists(self, mock_metadata_db):
        """Test conversation building with many changelists."""
        
        # Create 20 changelists with incrementing minutes
        many_changelists = [
            Changelist(
                id=f"commit{i}",
                expert_name="TestExpert",
                timestamp=datetime(2024, 1, 15, 10, i, 0),
                author="test@example.com",
                message=f"Change {i}",
                diff=f"diff {i}",
                files=[f"file{i}.py"],
                generated_prompt=f"Prompt {i}"
            )
            for i in range(20)
        ]
        
        conv_builder = ConversationBuilder(None, TEST_MAX_DIFF_CHARS_CONV)
        system_prompt, messages = conv_builder.build_conversation(
            changelists=many_changelists,
            user_prompt="Test query",
            amogus=False
        )
        
        # Should have 41 messages: 20 pairs + final user
        assert len(messages) == 41
        
        # Verify chronological ordering
        for i in range(20):
            assert f"Change {i}" in messages[i * 2 + 1].content

