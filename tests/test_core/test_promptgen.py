"""Tests for the PromptGenerator class."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, call

from expert_among_us.core.promptgen import PromptGenerator
from expert_among_us.models.changelist import Changelist
from expert_among_us.llm.base import Message, LLMResponse, UsageMetrics, LLMError, LLMRateLimitError
from expert_among_us.db.metadata.base import MetadataDB

# Test constant for max_diff_chars
TEST_MAX_DIFF_CHARS = 2000


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return Mock()


@pytest.fixture
def mock_metadata_db():
    """Create a mock metadata database."""
    return Mock(spec=MetadataDB)


@pytest.fixture
def prompt_generator(mock_llm, mock_metadata_db):
    """Create a PromptGenerator instance with mocks."""
    return PromptGenerator(
        llm_provider=mock_llm,
        metadata_db=mock_metadata_db,
        model="us.amazon.nova-lite-v1:0",
        max_diff_chars=TEST_MAX_DIFF_CHARS,
    )


@pytest.fixture
def sample_changelist():
    """Create a sample changelist for testing."""
    return Changelist(
        id="abc123",
        expert_name="TestExpert",
        timestamp=datetime(2024, 1, 15, 10, 30),
        author="test.user@example.com",
        message="Add error handling to user service",
        diff="""diff --git a/service.py b/service.py
index 1234567..abcdefg 100644
--- a/service.py
+++ b/service.py
@@ -10,6 +10,10 @@ def process_user(user_id):
     user = get_user(user_id)
+    if user is None:
+        raise ValueError("User not found")
+    
     return user.process()
""",
        files=["service.py"],
    )


@pytest.fixture
def sample_changelists():
    """Create multiple sample changelists."""
    return [
        Changelist(
            id=f"commit{i}",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 30),
            author="test.user@example.com",
            message=f"Commit message {i}",
            diff=f"diff --git a/file{i}.py b/file{i}.py\n+new line {i}",
            files=[f"file{i}.py"],
        )
        for i in range(3)
    ]


class TestPromptGeneratorInit:
    """Tests for PromptGenerator initialization."""
    
    def test_init_with_required_params(self, mock_llm, mock_metadata_db):
        """Test initialization with required parameters."""
        gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="test-model",
            max_diff_chars=TEST_MAX_DIFF_CHARS,
        )
        
        assert gen.llm is mock_llm
        assert gen.metadata_db is mock_metadata_db
        assert gen.model == "test-model"
        assert gen.max_diff_chars == TEST_MAX_DIFF_CHARS
    
    def test_init_with_custom_max_diff_chars(self, mock_llm, mock_metadata_db):
        """Test initialization with custom max_diff_chars."""
        gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="test-model",
            max_diff_chars=5000,
        )
        
        assert gen.max_diff_chars == 5000


class TestSinglePromptGeneration:
    """Tests for generating prompts for a single changelist."""
    
    def test_generate_single_prompt_success(
        self, prompt_generator, mock_llm, sample_changelist
    ):
        """Test successful single prompt generation."""
        mock_llm.generate.return_value = LLMResponse(
            content="Add error handling for user not found cases",
            model="us.amazon.nova-lite-v1:0",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=100, output_tokens=20, total_tokens=120),
        )
        
        prompt = prompt_generator._generate_single_prompt(sample_changelist)
        
        assert prompt == "Add error handling for user not found cases"
        assert mock_llm.generate.called
        
        # Check the call arguments
        call_args = mock_llm.generate.call_args
        assert call_args.kwargs["model"] == "us.amazon.nova-lite-v1:0"
        assert call_args.kwargs["max_tokens"] == 500
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["system"] == PromptGenerator.SYSTEM_PROMPT
    
    def test_generate_single_prompt_removes_quotes(
        self, prompt_generator, mock_llm, sample_changelist
    ):
        """Test that generated prompts have quotes removed."""
        mock_llm.generate.return_value = LLMResponse(
            content='"Add error handling for user service"',
            model="us.amazon.nova-lite-v1:0",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=100, output_tokens=20, total_tokens=120),
        )
        
        prompt = prompt_generator._generate_single_prompt(sample_changelist)
        
        assert prompt == "Add error handling for user service"
        assert not prompt.startswith('"')
        assert not prompt.endswith('"')
    
    def test_generate_single_prompt_removes_single_quotes(
        self, prompt_generator, mock_llm, sample_changelist
    ):
        """Test that generated prompts have single quotes removed."""
        mock_llm.generate.return_value = LLMResponse(
            content="'Add error handling for user service'",
            model="us.amazon.nova-lite-v1:0",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=100, output_tokens=20, total_tokens=120),
        )
        
        prompt = prompt_generator._generate_single_prompt(sample_changelist)
        
        assert prompt == "Add error handling for user service"


class TestBuildPromptRequest:
    """Tests for building LLM requests from changelists."""
    
    def test_build_request_includes_all_fields(
        self, prompt_generator, sample_changelist
    ):
        """Test that request includes commit message, files, and diff."""
        request = prompt_generator._build_prompt_request(sample_changelist)
        
        assert "Commit Message: Add error handling to user service" in request
        assert "Files Changed: service.py" in request
        assert "Code Changes:" in request
        assert "diff --git a/service.py" in request
    
    def test_build_request_truncates_long_diff(self, prompt_generator):
        """Test that long diffs are truncated."""
        long_changelist = Changelist(
            id="long123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 30),
            author="test.user@example.com",
            message="Long diff commit",
            diff="x" * 5000,  # Longer than max_diff_chars
            files=["file.py"],
        )
        
        request = prompt_generator._build_prompt_request(long_changelist)
        
        # Should be truncated and have truncation marker
        assert len(request) < 5000
        assert "[... truncated for brevity ...]" in request
    
    def test_build_request_limits_files_shown(self, prompt_generator):
        """Test that only first 10 files are shown."""
        many_files_changelist = Changelist(
            id="files123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 30),
            author="test.user@example.com",
            message="Many files commit",
            diff="diff content",
            files=[f"file{i}.py" for i in range(20)],
        )
        
        request = prompt_generator._build_prompt_request(many_files_changelist)
        
        # Should show first 10 files and indicate more
        assert "(and 10 more)" in request


class TestBatchProcessing:
    """Tests for batch prompt generation."""
    
    def test_generate_prompts_all_cache_hits(
        self, prompt_generator, mock_metadata_db, sample_changelists
    ):
        """Test batch generation when all prompts are cached."""
        # Mock all cache hits
        mock_metadata_db.get_generated_prompt.side_effect = [
            "Cached prompt 0",
            "Cached prompt 1",
            "Cached prompt 2",
        ]
        
        results = prompt_generator.generate_prompts(sample_changelists)
        
        assert len(results) == 3
        assert results["commit0"] == "Cached prompt 0"
        assert results["commit1"] == "Cached prompt 1"
        assert results["commit2"] == "Cached prompt 2"
        
        # Should not call LLM
        assert not prompt_generator.llm.generate.called
    
    def test_generate_prompts_all_cache_misses(
        self, prompt_generator, mock_llm, mock_metadata_db, sample_changelists
    ):
        """Test batch generation when no prompts are cached."""
        # Mock all cache misses
        mock_metadata_db.get_generated_prompt.return_value = None
        
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            LLMResponse(
                content=f"Generated prompt {i}",
                model="us.amazon.nova-lite-v1:0",
                stop_reason="end_turn",
                usage=UsageMetrics(input_tokens=100, output_tokens=20, total_tokens=120),
            )
            for i in range(3)
        ]
        
        results = prompt_generator.generate_prompts(sample_changelists)
        
        assert len(results) == 3
        assert results["commit0"] == "Generated prompt 0"
        assert results["commit1"] == "Generated prompt 1"
        assert results["commit2"] == "Generated prompt 2"
        
        # Should call LLM 3 times
        assert mock_llm.generate.call_count == 3
        
        # Should cache all results
        assert mock_metadata_db.cache_generated_prompt.call_count == 3
    
    def test_generate_prompts_mixed_cache_hits_and_misses(
        self, prompt_generator, mock_llm, mock_metadata_db, sample_changelists
    ):
        """Test batch generation with mixed cache hits and misses."""
        # Mock cache: hit, miss, hit
        mock_metadata_db.get_generated_prompt.side_effect = [
            "Cached prompt 0",
            None,
            "Cached prompt 2",
        ]
        
        # Mock LLM response for the miss
        mock_llm.generate.return_value = LLMResponse(
            content="Generated prompt 1",
            model="us.amazon.nova-lite-v1:0",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=100, output_tokens=20, total_tokens=120),
        )
        
        results = prompt_generator.generate_prompts(sample_changelists)
        
        assert len(results) == 3
        assert results["commit0"] == "Cached prompt 0"
        assert results["commit1"] == "Generated prompt 1"
        assert results["commit2"] == "Cached prompt 2"
        
        # Should call LLM only once (for cache miss)
        assert mock_llm.generate.call_count == 1
        
        # Should cache only the new result
        assert mock_metadata_db.cache_generated_prompt.call_count == 1
    
    def test_generate_prompts_updates_changelist_objects(
        self, prompt_generator, mock_llm, mock_metadata_db, sample_changelists
    ):
        """Test that changelist objects are updated with generated prompts."""
        mock_metadata_db.get_generated_prompt.return_value = None
        
        mock_llm.generate.side_effect = [
            LLMResponse(
                content=f"Prompt {i}",
                model="us.amazon.nova-lite-v1:0",
                stop_reason="end_turn",
                usage=UsageMetrics(input_tokens=100, output_tokens=20, total_tokens=120),
            )
            for i in range(3)
        ]
        
        prompt_generator.generate_prompts(sample_changelists)
        
        # Check that changelist objects were updated
        assert sample_changelists[0].generated_prompt == "Prompt 0"
        assert sample_changelists[1].generated_prompt == "Prompt 1"
        assert sample_changelists[2].generated_prompt == "Prompt 2"



class TestDiffTruncation:
    """Tests for diff truncation."""
    
    def test_diff_not_truncated_when_small(self, prompt_generator):
        """Test that small diffs are not truncated."""
        small_changelist = Changelist(
            id="small123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 30),
            author="test.user@example.com",
            message="Small change",
            diff="x" * 100,  # Small diff
            files=["file.py"],
        )
        
        request = prompt_generator._build_prompt_request(small_changelist)
        
        # Should not have truncation marker
        assert "[... truncated for brevity ...]" not in request
        assert "x" * 100 in request
    
    def test_diff_truncated_when_large(self, prompt_generator):
        """Test that large diffs are truncated."""
        large_changelist = Changelist(
            id="large123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 30),
            author="test.user@example.com",
            message="Large change",
            diff="x" * 10000,  # Large diff
            files=["file.py"],
        )
        
        request = prompt_generator._build_prompt_request(large_changelist)
        
        # Should be truncated
        assert "[... truncated for brevity ...]" in request
        # Should not contain full original diff
        assert "x" * 10000 not in request
    
    def test_diff_truncation_respects_max_diff_chars(
        self, mock_llm, mock_metadata_db
    ):
        """Test that diff truncation respects the configured max_diff_chars."""
        gen = PromptGenerator(
            llm_provider=mock_llm,
            metadata_db=mock_metadata_db,
            model="test-model",
            max_diff_chars=500,  # Smaller limit
        )
        
        changelist = Changelist(
            id="test123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 30),
            author="test.user@example.com",
            message="Test",
            diff="x" * 1000,  # Exceeds limit
            files=["file.py"],
        )
        
        request = gen._build_prompt_request(changelist)
        
        # Should be truncated at 500 chars (plus some overhead for message/files)
        diff_section = request.split("Code Changes:")[1]
        assert len(diff_section) < 1000


class TestIncrementalCaching:
    """Tests for incremental caching behavior."""
    
    def test_prompts_cached_immediately_after_generation(
        self, prompt_generator, mock_llm, mock_metadata_db, sample_changelists
    ):
        """Test that generated prompts are cached immediately."""
        mock_metadata_db.get_generated_prompt.return_value = None
        
        mock_llm.generate.side_effect = [
            LLMResponse(
                content=f"Prompt {i}",
                model="us.amazon.nova-lite-v1:0",
                stop_reason="end_turn",
                usage=UsageMetrics(input_tokens=100, output_tokens=20, total_tokens=120),
            )
            for i in range(3)
        ]
        
        prompt_generator.generate_prompts(sample_changelists)
        
        # Verify each prompt was cached immediately after generation
        cache_calls = mock_metadata_db.cache_generated_prompt.call_args_list
        assert len(cache_calls) == 3
        
        # Check order - should be called in sequence
        assert cache_calls[0] == call("commit0", "Prompt 0")
        assert cache_calls[1] == call("commit1", "Prompt 1")
        assert cache_calls[2] == call("commit2", "Prompt 2")


class TestProgressReporting:
    """Tests for progress reporting during batch generation."""
    
    def test_progress_reported_for_large_batches(
        self, prompt_generator, mock_llm, mock_metadata_db
    ):
        """Test that progress is reported for large batches."""
        # Create a batch of 10 changelists
        large_batch = [
            Changelist(
                id=f"commit{i}",
                expert_name="TestExpert",
                timestamp=datetime(2024, 1, 15, 10, 30),
                author="test.user@example.com",
                message=f"Message {i}",
                diff=f"diff {i}",
                files=[f"file{i}.py"],
            )
            for i in range(10)
        ]
        
        mock_metadata_db.get_generated_prompt.return_value = None
        
        mock_llm.generate.return_value = LLMResponse(
            content="Test prompt",
            model="us.amazon.nova-lite-v1:0",
            stop_reason="end_turn",
            usage=UsageMetrics(input_tokens=100, output_tokens=20, total_tokens=120),
        )
        
        # Should complete without errors (progress messages logged)
        results = prompt_generator.generate_prompts(large_batch)
        
        assert len(results) == 10