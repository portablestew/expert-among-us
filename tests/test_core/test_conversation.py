"""Tests for conversation builder."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
import pytest

from expert_among_us.core.conversation import ConversationBuilder
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.query_result import CommitResult
from expert_among_us.llm.base import Message, LLMResponse

# Test constant for max_diff_chars
TEST_MAX_DIFF_CHARS = 3000


class TestConversationBuilder:
    """Test suite for ConversationBuilder class."""
    
    @pytest.fixture
    def mock_prompt_generator(self):
        """Create mock prompt generator."""
        mock_gen = MagicMock()
        mock_gen._generate_single_prompt.return_value = "Generated prompt text"
        return mock_gen
    
    @pytest.fixture
    def sample_changelist(self):
        """Create sample changelist for testing."""
        return Changelist(
            id="abc123",
            expert_name="TestExpert",
            project_name="test-project",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            author="test_author",
            message="Add feature X",
            diff="diff --git a/file.py\n+new line\n-old line",
            files=["file.py"],
            generated_prompt="Can you add feature X?"
        )
    
    @pytest.fixture
    def sample_changelists_multiple(self):
        """Create multiple changelists with different timestamps."""
        return [
            Changelist(
                id="abc123",
                expert_name="TestExpert",
                project_name="test-project",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                author="test_author",
                message="Add feature X",
                diff="diff --git a/file1.py\n+new line",
                files=["file1.py"],
                generated_prompt="Add feature X"
            ),
            Changelist(
                id="def456",
                expert_name="TestExpert",
                project_name="test-project",
                timestamp=datetime(2024, 1, 16, 14, 20, 0),
                author="test_author",
                message="Fix bug Y",
                diff="diff --git a/file2.py\n-bug\n+fix",
                files=["file2.py"],
                generated_prompt="Fix bug Y"
            ),
            Changelist(
                id="ghi789",
                expert_name="TestExpert",
                project_name="test-project",
                timestamp=datetime(2024, 1, 14, 9, 0, 0),  # Earlier than abc123
                author="test_author",
                message="Initial commit",
                diff="diff --git a/file0.py\n+initial",
                files=["file0.py"],
                generated_prompt="Create initial structure"
            ),
        ]
    
    def test_format_changelist_multiple_files(self, mock_prompt_generator):
        """Test changelist formatting with multiple files."""
        builder = ConversationBuilder(
            mock_prompt_generator,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        # Create changelist with 22 files (more than the 20-file limit)
        files = [f"file{i}.py" for i in range(22)]
        changelist = Changelist(
            id="abc123",
            expert_name="TestExpert",
            project_name="test-project",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            author="test_author",
            message="Update many files",
            diff="diff --git a/file.py\n+change",
            files=files,
        )
        
        formatted = builder._format_changelist_as_assistant(changelist)
        
        # Should show first 20 files plus "(and 2 more)"
        assert "file0.py" in formatted
        assert "file19.py" in formatted
        assert "(and 2 more)" in formatted
    
    def test_format_changelist_truncates_large_diff(self, mock_prompt_generator):
        """Test that large diffs are truncated."""
        builder = ConversationBuilder(
            mock_prompt_generator,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        # Create changelist with large diff (>3000 chars)
        large_diff = "diff --git a/file.py\n" + ("+" + "x" * 100 + "\n") * 50
        changelist = Changelist(
            id="abc123",
            expert_name="TestExpert",
            project_name="test-project",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            author="test_author",
            message="Large change",
            diff=large_diff,
            files=["file.py"],
        )
        
        formatted = builder._format_changelist_as_assistant(changelist)
        
        # Should be truncated
        assert len(formatted) < len(large_diff) + 200  # Some overhead for format
        assert "truncated" in formatted.lower()
    
    def test_build_conversation_single_changelist(self, mock_prompt_generator, sample_changelist):
        """Test building conversation with single changelist."""
        builder = ConversationBuilder(
            mock_prompt_generator,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        system_prompt, messages = builder.build_conversation(
            results=[CommitResult(changelist=sample_changelist, similarity_score=0.9, source="metadata")],
            user_prompt="How should I implement feature Z?",
            amogus=False,
            impostor=True
        )
        
        # Should have 3 messages: user (generated), assistant (changelist), user (final)
        assert len(messages) == 3
        
        # Check message roles and order
        assert messages[0].role == "user"
        assert messages[0].content == "Can you add feature X?"
        
        assert messages[1].role == "assistant"
        assert "Commit: Add feature X" in messages[1].content
        
        assert messages[2].role == "user"
        assert "How should I implement feature Z?" in messages[2].content
        assert "Remember to structure your response" in messages[2].content
        
        # Check system prompt
        assert "expert software developer" in system_prompt
        assert "impostor" not in system_prompt.lower()
    
    def test_build_conversation_multiple_changelists(self, mock_prompt_generator, sample_changelists_multiple):
        """Test building conversation with multiple changelists."""
        builder = ConversationBuilder(
            mock_prompt_generator,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        results = [
            CommitResult(changelist=cl, similarity_score=0.9 - i*0.1, source="metadata")
            for i, cl in enumerate(sample_changelists_multiple)
        ]
        system_prompt, messages = builder.build_conversation(
            results=results,
            user_prompt="Final question",
            amogus=False,
            impostor=True
        )
        
        # Should have 7 messages: 3 pairs (user+assistant) + final user
        assert len(messages) == 7
        
        # Verify chronological ordering (ghi789 earliest, then abc123, then def456)
        assert "Initial commit" in messages[1].content  # ghi789 assistant
        assert "Add feature X" in messages[3].content   # abc123 assistant
        assert "Fix bug Y" in messages[5].content       # def456 assistant
        
        # Verify all user messages are prompts
        assert messages[0].role == "user"
        assert messages[2].role == "user"
        assert messages[4].role == "user"
        
        # Verify all assistant messages have formatted changelists
        assert messages[1].role == "assistant"
        assert messages[3].role == "assistant"
        assert messages[5].role == "assistant"
        
        # Final message should be user prompt
        assert messages[6].role == "user"
        assert "Final question" in messages[6].content
        assert "Remember to structure your response" in messages[6].content
    
    def test_build_conversation_chronological_ordering(self, mock_prompt_generator):
        """Test that changelists are sorted chronologically."""
        builder = ConversationBuilder(
            mock_prompt_generator,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        # Create changelists in non-chronological order
        changelists = [
            Changelist(
                id="newest",
                expert_name="TestExpert",
                project_name="test-project",
                timestamp=datetime(2024, 1, 17, 10, 0, 0),
                author="author",
                message="Newest",
                diff="diff",
                files=["file.py"],
                generated_prompt="Newest prompt"
            ),
            Changelist(
                id="oldest",
                expert_name="TestExpert",
                project_name="test-project",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                author="author",
                message="Oldest",
                diff="diff",
                files=["file.py"],
                generated_prompt="Oldest prompt"
            ),
            Changelist(
                id="middle",
                expert_name="TestExpert",
                project_name="test-project",
                timestamp=datetime(2024, 1, 16, 10, 0, 0),
                author="author",
                message="Middle",
                diff="diff",
                files=["file.py"],
                generated_prompt="Middle prompt"
            ),
        ]
        
        results = [
            CommitResult(changelist=cl, similarity_score=0.9, source="metadata")
            for cl in changelists
        ]
        system_prompt, messages = builder.build_conversation(
            results=results,
            user_prompt="Test",
            amogus=False,
            impostor=True
        )
        
        # Verify chronological order in messages
        assert "Oldest" in messages[1].content
        assert "Middle" in messages[3].content
        assert "Newest" in messages[5].content
    
    def test_build_conversation_generates_missing_prompts(self, mock_prompt_generator):
        """Test that missing prompts are generated."""
        builder = ConversationBuilder(
            mock_prompt_generator,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        # Changelist without generated_prompt
        changelist = Changelist(
            id="abc123",
            expert_name="TestExpert",
            project_name="test-project",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            author="test_author",
            message="Add feature",
            diff="diff --git a/file.py\n+new",
            files=["file.py"],
            # No generated_prompt
        )
        
        system_prompt, messages = builder.build_conversation(
            results=[CommitResult(changelist=changelist, similarity_score=0.9, source="metadata")],
            user_prompt="Test prompt",
            amogus=False,
            impostor=True
        )
        
        # Should call _generate_single_prompt
        mock_prompt_generator._generate_single_prompt.assert_called_once_with(changelist)
        
        # Should have generated prompt in message
        assert messages[0].content == "Generated prompt text"
    
    def test_build_conversation_uses_cached_prompts(self, mock_prompt_generator, sample_changelist):
        """Test that cached prompts are used."""
        builder = ConversationBuilder(
            mock_prompt_generator,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        # Changelist already has generated_prompt
        system_prompt, messages = builder.build_conversation(
            results=[CommitResult(changelist=sample_changelist, similarity_score=0.9, source="metadata")],
            user_prompt="Test",
            amogus=False,
            impostor=True
        )
        
        # Should NOT call _generate_single_prompt
        mock_prompt_generator._generate_single_prompt.assert_not_called()
        
        # Should use cached prompt
        assert messages[0].content == "Can you add feature X?"
    
    def test_build_conversation_empty_changelists_raises_error(self, mock_prompt_generator):
        """Test that empty changelists raises ValueError."""
        builder = ConversationBuilder(
            mock_prompt_generator,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        with pytest.raises(ValueError, match="Cannot build conversation with empty results"):
            builder.build_conversation(
                results=[],
                user_prompt="Test",
                amogus=False
            )
    
    def test_build_conversation_default_mode_all_user_messages(
        self, mock_prompt_generator, sample_changelists_multiple
    ):
        """Test default mode with impostor=False creates alternating user-assistant pairs."""
        builder = ConversationBuilder(
            None,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )  # No generator needed
        
        results = [
            CommitResult(changelist=cl, similarity_score=0.9 - i*0.1, source="metadata")
            for i, cl in enumerate(sample_changelists_multiple)
        ]
        system_prompt, messages = builder.build_conversation(
            results=results,
            user_prompt="Final question",
            amogus=False,
            impostor=False  # Default mode
        )
        
        # Should have 7 messages: 3 commits × (user + assistant) + final user prompt
        assert len(messages) == 7
        
        # Verify alternating pattern: user, assistant, user, assistant, user, assistant, user
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
        assert messages[3].role == "assistant"
        assert messages[4].role == "user"
        assert messages[5].role == "assistant"
        assert messages[6].role == "user"
        
        # User messages are "Show me commit <id>" references
        assert "Show me commit" in messages[0].content
        assert "Show me commit" in messages[2].content
        assert "Show me commit" in messages[4].content
        
        # Assistant messages contain commit content (chronological order)
        assert "Initial commit" in messages[1].content  # ghi789
        assert "Add feature X" in messages[3].content   # abc123
        assert "Fix bug Y" in messages[5].content       # def456
        
        # Final message is the user prompt
        assert "Final question" in messages[6].content
        assert "Remember to structure your response" in messages[6].content
        
        # Verify no prompts were generated (mock not called)
        mock_prompt_generator._generate_single_prompt.assert_not_called()
    
    
    def test_default_mode_raises_if_generator_none_and_impostor_true(
        self, sample_changelist
    ):
        """Test that impostor=True requires prompt_generator."""
        builder = ConversationBuilder(
            None,
            TEST_MAX_DIFF_CHARS,
            max_context_tokens=120000,
            max_response_tokens=4096
        )
        
        with pytest.raises(ValueError, match="prompt_generator required when impostor=True"):
            builder.build_conversation(
                results=[CommitResult(changelist=sample_changelist, similarity_score=0.9, source="metadata")],
                user_prompt="Test",
                impostor=True
            )