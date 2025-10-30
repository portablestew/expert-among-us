"""Edge case tests for PromptGenerator."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from expert_among_us.core.promptgen import PromptGenerator
from expert_among_us.models.changelist import Changelist
from expert_among_us.llm.base import LLMResponse, LLMError, LLMRateLimitError


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return Mock()


@pytest.fixture
def mock_metadata_db():
    """Create a mock metadata database."""
    return Mock()


@pytest.fixture
def prompt_generator(mock_llm, mock_metadata_db):
    """Create a PromptGenerator instance."""
    return PromptGenerator(
        llm_provider=mock_llm,
        metadata_db=mock_metadata_db,
        model="test-model",
        max_diff_chars=2000
    )


class TestEmptyAndNullInputs:
    """Test handling of minimal valid inputs."""
    
    def test_minimal_commit_message(self, prompt_generator, mock_llm):
        """Test handling of changelist with minimal commit message."""
        changelist = Changelist(
            id="minimal123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="x",  # Minimal but valid message
            diff="diff --git a/file.py b/file.py\n+new line",
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Add new functionality",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        # Should work with minimal message
        prompt = prompt_generator._generate_single_prompt(changelist)
        assert prompt == "Add new functionality"
    
    def test_minimal_diff(self, prompt_generator, mock_llm):
        """Test handling of changelist with minimal diff."""
        changelist = Changelist(
            id="mindiff123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Commit with minimal diff",
            diff="x",  # Minimal but valid diff
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Update configuration",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        # Should handle minimal diff gracefully
        prompt = prompt_generator._generate_single_prompt(changelist)
        assert prompt == "Update configuration"
    
    def test_single_file_changed(self, prompt_generator, mock_llm):
        """Test handling of changelist with single file."""
        changelist = Changelist(
            id="singlefile123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Commit message",
            diff="diff content",
            files=["file.py"],  # Single file
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Update metadata",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        request = prompt_generator._build_prompt_request(changelist)
        assert "Files Changed:" in request
        assert "file.py" in request


class TestExtremeInputs:
    """Test handling of extreme input values."""
    
    def test_extremely_long_commit_message(self, prompt_generator, mock_llm):
        """Test handling of very long commit messages."""
        long_message = "Fix bug " * 1000  # Very long message
        
        changelist = Changelist(
            id="longmsg123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message=long_message,
            diff="diff content",
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Fix multiple bugs",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        # Should handle long messages
        prompt = prompt_generator._generate_single_prompt(changelist)
        assert prompt == "Fix multiple bugs"
    
    def test_very_many_files(self, prompt_generator):
        """Test handling of commits with many files."""
        many_files = [f"file{i}.py" for i in range(1000)]
        
        changelist = Changelist(
            id="manyfiles123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Update many files",
            diff="diff content",
            files=many_files,
        )
        
        request = prompt_generator._build_prompt_request(changelist)
        
        # Should limit files shown
        assert "(and 990 more)" in request
        assert "file0.py" in request
        assert "file9.py" in request
    
    def test_diff_exactly_at_limit(self, prompt_generator):
        """Test diff that is exactly at the character limit."""
        # Create diff that's exactly max_diff_chars
        exact_diff = "x" * 2000
        
        changelist = Changelist(
            id="exact123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Exact limit change",
            diff=exact_diff,
            files=["file.py"],
        )
        
        request = prompt_generator._build_prompt_request(changelist)
        
        # Should not truncate at exact limit
        # (truncation happens slightly above due to overhead)
        assert exact_diff[:1900] in request


class TestSpecialCharactersAndEncoding:
    """Test handling of special characters and encoding issues."""
    
    def test_newlines_in_commit_message(self, prompt_generator, mock_llm):
        """Test handling of multi-line commit messages."""
        multiline_message = "First line\n\nSecond paragraph\n\nThird paragraph"
        
        changelist = Changelist(
            id="multiline123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message=multiline_message,
            diff="diff content",
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Update with multiple changes",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        # Should handle newlines properly
        prompt = prompt_generator._generate_single_prompt(changelist)
        assert prompt == "Update with multiple changes"
    
    def test_tabs_and_special_whitespace(self, prompt_generator):
        """Test handling of tabs and special whitespace."""
        changelist = Changelist(
            id="whitespace123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Commit\twith\ttabs",
            diff="diff --git a/file.py\n+\tindented\tline",
            files=["file.py"],
        )
        
        request = prompt_generator._build_prompt_request(changelist)
        
        # Should preserve tabs in request
        assert "\t" in request
    
    def test_emoji_and_unicode(self, prompt_generator, mock_llm):
        """Test handling of emoji and Unicode characters."""
        changelist = Changelist(
            id="emoji123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Add emoji support 🎉✨🚀",
            diff="diff --git a/file.py\n+message = '你好世界 🌍'",
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="Add internationalization and emoji support",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        # Should handle Unicode properly
        prompt = prompt_generator._generate_single_prompt(changelist)
        assert prompt == "Add internationalization and emoji support"
    
    def test_control_characters(self, prompt_generator):
        """Test handling of control characters in input."""
        changelist = Changelist(
            id="control123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Commit\x00with\x01control\x02chars",
            diff="diff content",
            files=["file.py"],
        )
        
        # Should not crash with control characters
        request = prompt_generator._build_prompt_request(changelist)
        assert "Commit Message:" in request


class TestBatchProcessingEdgeCases:
    """Test edge cases in batch processing."""
    
    def test_empty_changelist_batch(self, prompt_generator):
        """Test generating prompts for empty list."""
        results = prompt_generator.generate_prompts([])
        
        # Should return empty dict
        assert results == {}
    
    def test_single_changelist_batch(self, prompt_generator, mock_llm, mock_metadata_db):
        """Test batch with single changelist."""
        mock_metadata_db.get_generated_prompt.return_value = None
        
        mock_llm.generate.return_value = LLMResponse(
            content="Single prompt",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        changelist = Changelist(
            id="single123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Single change",
            diff="diff content",
            files=["file.py"],
        )
        
        results = prompt_generator.generate_prompts([changelist])
        
        assert len(results) == 1
        assert results["single123"] == "Single prompt"
    
    def test_all_failures_in_batch(self, prompt_generator, mock_llm, mock_metadata_db):
        """Test batch where all LLM calls fail."""
        mock_metadata_db.get_generated_prompt.return_value = None
        mock_llm.generate.side_effect = LLMError("API error")
        
        changelists = [
            Changelist(
                id=f"fail{i}",
                expert_name="TestExpert",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                author="test@example.com",
                message=f"Change {i}",
                diff="diff content",
                files=["file.py"],
            )
            for i in range(3)
        ]
        
        results = prompt_generator.generate_prompts(changelists)
        
        # Should return empty dict when all fail
        assert len(results) == 0


class TestQuoteHandling:
    """Test handling of quotes in generated prompts."""
    
    def test_double_quotes_at_start_and_end(self, prompt_generator, mock_llm):
        """Test removal of double quotes from generated prompts."""
        changelist = Changelist(
            id="quotes123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Add feature",
            diff="diff content",
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content='"Add authentication to the API"',
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        prompt = prompt_generator._generate_single_prompt(changelist)
        
        # Should strip quotes
        assert prompt == "Add authentication to the API"
        assert not prompt.startswith('"')
        assert not prompt.endswith('"')
    
    def test_single_quotes_at_start_and_end(self, prompt_generator, mock_llm):
        """Test removal of single quotes from generated prompts."""
        changelist = Changelist(
            id="singles123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Add feature",
            diff="diff content",
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content="'Implement user validation'",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        prompt = prompt_generator._generate_single_prompt(changelist)
        
        # Should strip single quotes
        assert prompt == "Implement user validation"
    
    def test_quotes_in_middle_preserved(self, prompt_generator, mock_llm):
        """Test that quotes in the middle of text are preserved."""
        changelist = Changelist(
            id="middle123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Add feature",
            diff="diff content",
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content='Add support for "special" characters',
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        prompt = prompt_generator._generate_single_prompt(changelist)
        
        # Should preserve internal quotes
        assert prompt == 'Add support for "special" characters'
    
    def test_mixed_quotes(self, prompt_generator, mock_llm):
        """Test handling of mixed quote types."""
        changelist = Changelist(
            id="mixed123",
            expert_name="TestExpert",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            author="test@example.com",
            message="Add feature",
            diff="diff content",
            files=["file.py"],
        )
        
        mock_llm.generate.return_value = LLMResponse(
            content='"Add \'nested\' quote handling"',
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        prompt = prompt_generator._generate_single_prompt(changelist)
        
        # Should only strip outer quotes
        assert prompt == "Add 'nested' quote handling"


class TestConcurrentGenerationScenarios:
    """Test scenarios that might occur with concurrent generation."""
    
    def test_duplicate_changelist_ids(self, prompt_generator, mock_llm, mock_metadata_db):
        """Test handling of duplicate changelist IDs in batch."""
        mock_metadata_db.get_generated_prompt.return_value = None
        
        mock_llm.generate.return_value = LLMResponse(
            content="Generated prompt",
            model="test-model",
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 10}
        )
        
        # Create changelists with same ID (edge case, shouldn't happen normally)
        changelists = [
            Changelist(
                id="duplicate",
                expert_name="TestExpert",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                author="test@example.com",
                message=f"Change {i}",
                diff="diff content",
                files=["file.py"],
            )
            for i in range(2)
        ]
        
        results = prompt_generator.generate_prompts(changelists)
        
        # Last one should win
        assert len(results) == 1
        assert "duplicate" in results