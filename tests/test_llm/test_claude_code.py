"""Tests for ClaudeCodeLLM implementation."""

import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, call
from datetime import datetime, timezone

from expert_among_us.llm.claude_code import ClaudeCodeLLM
from expert_among_us.llm.base import (
    Message,
    LLMResponse,
    StreamChunk,
    UsageMetrics,
    LLMError,
    LLMRateLimitError,
    LLMInvalidRequestError,
)


@pytest.fixture
def mock_shutil_which():
    """Mock shutil.which to simulate CLI availability."""
    with patch("expert_among_us.llm.claude_code.shutil.which") as mock:
        mock.return_value = "/usr/local/bin/claude"
        yield mock


@pytest.fixture
def claude_llm(mock_shutil_which, tmp_path):
    """Create ClaudeCodeLLM instance with mocked CLI check."""
    llm = ClaudeCodeLLM(
        cli_path="claude",
        session_dir=tmp_path / "sessions"
    )
    return llm


class TestClaudeCodeLLMInit:
    """Tests for ClaudeCodeLLM initialization."""
    
    def test_init_default_params(self, mock_shutil_which):
        """Test initialization with default parameters."""
        llm = ClaudeCodeLLM()
        
        mock_shutil_which.assert_called_once_with("claude")
        assert llm.cli_path == "claude"
        assert llm.session_dir == Path.home() / ".claude" / "projects" / "expert-among-us"
        assert llm.session_dir.exists()
    
    def test_init_custom_cli_path(self, mock_shutil_which):
        """Test initialization with custom CLI path."""
        mock_shutil_which.return_value = "/custom/path/claude"
        
        llm = ClaudeCodeLLM(cli_path="/custom/path/claude")
        
        mock_shutil_which.assert_called_once_with("/custom/path/claude")
        assert llm.cli_path == "/custom/path/claude"
    
    def test_init_custom_session_dir(self, mock_shutil_which, tmp_path):
        """Test initialization with custom session directory."""
        custom_dir = tmp_path / "custom_sessions"
        
        llm = ClaudeCodeLLM(project_dir=custom_dir)
        
        assert llm.session_dir == custom_dir
        assert custom_dir.exists()
    
    def test_init_cli_not_found(self):
        """Test initialization fails when CLI not found."""
        with patch("expert_among_us.llm.claude_code.shutil.which", return_value=None):
            with pytest.raises(LLMError) as exc_info:
                ClaudeCodeLLM(cli_path="nonexistent")
            
            assert "Claude CLI not found" in str(exc_info.value)
            assert "nonexistent" in str(exc_info.value)


class TestSessionFileCreation:
    """Tests for session file creation and format."""
    
    def test_create_session_id(self, claude_llm):
        """Test session ID generation."""
        session_id = claude_llm._create_session_id()
        
        # Should be a valid UUID string
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID format: 8-4-4-4-12
        assert session_id.count("-") == 4
    
    def test_create_session_file_single_message(self, claude_llm):
        """Test creating session file with single message."""
        messages = [Message(role="user", content="Hello")]
        session_id = "test-session-123"
        
        session_file = claude_llm._create_session_file(session_id, messages)
        
        assert session_file.exists()
        assert session_file.name == f"{session_id}.jsonl"
        
        # Read and verify JSONL format
        with open(session_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2  # Summary + 1 message
        
        # Verify summary event
        summary = json.loads(lines[0])
        assert summary["type"] == "summary"
        assert "summary" in summary
        assert "leafUuid" in summary
        
        # Verify message event
        message = json.loads(lines[1])
        assert message["type"] == "user"
        assert message["message"]["role"] == "user"
        assert message["message"]["content"] == "Hello"
        assert message["sessionId"] == session_id
        assert message["parentUuid"] is None
    
    def test_create_session_file_multi_turn(self, claude_llm):
        """Test creating session file with multi-turn conversation."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?")
        ]
        session_id = "test-session-456"
        
        session_file = claude_llm._create_session_file(session_id, messages)
        
        assert session_file.exists()
        
        # Read and verify JSONL format
        with open(session_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 4  # Summary + 3 messages
        
        # Parse all events
        events = [json.loads(line) for line in lines]
        
        # Verify event chain
        summary = events[0]
        user1 = events[1]
        assistant = events[2]
        user2 = events[3]
        
        assert summary["type"] == "summary"
        assert user1["type"] == "user"
        assert assistant["type"] == "assistant"
        assert user2["type"] == "user"
        
        # Verify parent-child relationships
        assert user1["parentUuid"] is None
        assert assistant["parentUuid"] == user1["uuid"]
        assert user2["parentUuid"] == assistant["uuid"]
    
    def test_create_session_file_with_system_prompt(self, claude_llm):
        """Test creating session file with system prompt."""
        messages = [Message(role="user", content="Test")]
        session_id = "test-session-789"
        system_prompt = "You are a helpful assistant"
        
        session_file = claude_llm._create_session_file(session_id, messages, system=system_prompt)
        
        assert session_file.exists()
        
        # Read summary line
        with open(session_file, 'r') as f:
            summary_line = f.readline()
        
        summary = json.loads(summary_line)
        assert system_prompt[:50] in summary["summary"]
    
    def test_session_file_event_structure(self, claude_llm):
        """Test that session file events have required fields."""
        messages = [
            Message(role="user", content="Test"),
            Message(role="assistant", content="Response")
        ]
        session_id = "test-structure"
        
        session_file = claude_llm._create_session_file(session_id, messages)
        
        with open(session_file, 'r') as f:
            lines = f.readlines()
        
        # Check user event structure
        user_event = json.loads(lines[1])
        assert "parentUuid" in user_event
        assert "isSidechain" in user_event
        assert "sessionId" in user_event
        assert "uuid" in user_event
        assert "timestamp" in user_event
        assert "message" in user_event
        assert user_event["message"]["role"] == "user"
        
        # Check assistant event structure
        assistant_event = json.loads(lines[2])
        assert "parentUuid" in assistant_event
        assert "sessionId" in assistant_event
        assert "uuid" in assistant_event
        assert "message" in assistant_event
        assert assistant_event["message"]["role"] == "assistant"
        assert isinstance(assistant_event["message"]["content"], list)


class TestGenerate:
    """Tests for synchronous generate method."""
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    def test_generate_success(self, mock_run, claude_llm):
        """Test successful generation."""
        mock_response = {
            "content": [{"type": "text", "text": "This is a response"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_response)
        )
        
        messages = [Message(role="user", content="Hello")]
        response = claude_llm.generate(
            messages=messages,
            model="claude-sonnet-4-5"
        )
        
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a response"
        assert response.model == "claude-sonnet-4-5"
        assert response.stop_reason == "end_turn"
        assert isinstance(response.usage, UsageMetrics)
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        
        # Verify CLI command
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert "--resume" in cmd
        assert "--print" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    def test_generate_with_cache_metrics(self, mock_run, claude_llm):
        """Test generation with cache metrics."""
        mock_response = {
            "content": [{"type": "text", "text": "Cached response"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 50
            }
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_response)
        )
        
        messages = [Message(role="user", content="Hello")]
        response = claude_llm.generate(
            messages=messages,
            model="claude-sonnet-4-5"
        )
        
        assert response.usage.cache_read_tokens == 100
        assert response.usage.cache_creation_tokens == 50
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    def test_generate_multi_turn_conversation(self, mock_run, claude_llm):
        """Test generation with multi-turn conversation."""
        mock_response = {
            "content": [{"type": "text", "text": "Follow-up response"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 30, "output_tokens": 15}
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_response)
        )
        
        messages = [
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Follow-up question")
        ]
        response = claude_llm.generate(
            messages=messages,
            model="claude-sonnet-4-5"
        )
        
        assert response.content == "Follow-up response"
        # Verify session file was created with all messages
        mock_run.assert_called_once()
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    def test_generate_session_cleanup(self, mock_run, claude_llm):
        """Test that session files are cleaned up after generation."""
        mock_response = {
            "content": [{"type": "text", "text": "Response"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10}
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_response)
        )
        
        messages = [Message(role="user", content="Test")]
        claude_llm.generate(messages=messages, model="claude-sonnet-4-5")
        
        # Verify no session files remain
        session_files = list(claude_llm.session_dir.glob("*.jsonl"))
        assert len(session_files) == 0
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    def test_generate_invalid_json_response(self, mock_run, claude_llm):
        """Test handling of invalid JSON response."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Not valid JSON"
        )
        
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMError) as exc_info:
            claude_llm.generate(messages=messages, model="claude-sonnet-4-5")
        
        assert "Failed to parse Claude response" in str(exc_info.value)


class TestStream:
    """Tests for streaming generate method."""
    
    @pytest.mark.asyncio
    @patch("expert_among_us.llm.claude_code.subprocess.Popen")
    async def test_stream_success(self, mock_popen, claude_llm):
        """Test successful streaming."""
        # Mock streaming events
        mock_events = [
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n',
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}\n',
            '{"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n',
            '{"type":"message_end","usage":{"input_tokens":5,"output_tokens":10}}\n'
        ]
        
        mock_process = MagicMock()
        mock_process.stdout = iter(mock_events)
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in claude_llm.stream(
            messages=messages,
            model="claude-sonnet-4-5"
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[0].stop_reason is None
        assert chunks[0].usage is None
        
        assert chunks[1].delta == " world"
        
        assert chunks[2].delta == ""
        assert chunks[2].stop_reason == "end_turn"
        assert isinstance(chunks[2].usage, UsageMetrics)
        assert chunks[2].usage.input_tokens == 5
        assert chunks[2].usage.output_tokens == 10
    
    @pytest.mark.asyncio
    @patch("expert_among_us.llm.claude_code.subprocess.Popen")
    async def test_stream_with_cache_metrics(self, mock_popen, claude_llm):
        """Test streaming with cache metrics."""
        mock_events = [
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Test"}}\n',
            '{"type":"message_end","usage":{"input_tokens":5,"output_tokens":10,"cache_read_input_tokens":100,"cache_creation_input_tokens":50}}\n'
        ]
        
        mock_process = MagicMock()
        mock_process.stdout = iter(mock_events)
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in claude_llm.stream(
            messages=messages,
            model="claude-sonnet-4-5"
        ):
            chunks.append(chunk)
        
        assert chunks[-1].usage.cache_read_tokens == 100
        assert chunks[-1].usage.cache_creation_tokens == 50
    
    @pytest.mark.asyncio
    @patch("expert_among_us.llm.claude_code.subprocess.Popen")
    async def test_stream_session_cleanup(self, mock_popen, claude_llm):
        """Test that session files are cleaned up after streaming."""
        mock_events = [
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Test"}}\n',
            '{"type":"message_end","usage":{"input_tokens":5,"output_tokens":10}}\n'
        ]
        
        mock_process = MagicMock()
        mock_process.stdout = iter(mock_events)
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        messages = [Message(role="user", content="Test")]
        async for chunk in claude_llm.stream(messages=messages, model="claude-sonnet-4-5"):
            pass
        
        # Verify no session files remain
        session_files = list(claude_llm.session_dir.glob("*.jsonl"))
        assert len(session_files) == 0


class TestErrorHandling:
    """Tests for error handling."""
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    def test_generate_rate_limit_error(self, mock_run, claude_llm):
        """Test handling of rate limit errors."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["claude"],
            stderr="Error: rate limit exceeded"
        )
        
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMRateLimitError) as exc_info:
            claude_llm.generate(messages=messages, model="claude-sonnet-4-5")
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    def test_generate_invalid_request_error(self, mock_run, claude_llm):
        """Test handling of invalid request errors."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["claude"],
            stderr="Error: invalid request parameters"
        )
        
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            claude_llm.generate(messages=messages, model="claude-sonnet-4-5")
        
        assert "Invalid request" in str(exc_info.value)
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    def test_generate_generic_error(self, mock_run, claude_llm):
        """Test handling of generic errors."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["claude"],
            stderr="Some other error"
        )
        
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMError) as exc_info:
            claude_llm.generate(messages=messages, model="claude-sonnet-4-5")
        
        assert "Claude CLI error" in str(exc_info.value)
        assert "code 1" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch("expert_among_us.llm.claude_code.subprocess.Popen")
    async def test_stream_subprocess_error(self, mock_popen, claude_llm):
        """Test streaming handles subprocess errors."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_process.stderr.read.return_value = "rate limit error"
        mock_popen.return_value = mock_process
        
        messages = [Message(role="user", content="Test")]
        with pytest.raises(LLMRateLimitError):
            async for _ in claude_llm.stream(messages=messages, model="claude-sonnet-4-5"):
                pass


class TestDebugLogging:
    """Tests for debug logging integration."""
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    @patch("expert_among_us.llm.claude_code.DebugLogger")
    def test_generate_logs_when_debug_enabled(self, mock_debug_logger, mock_run, claude_llm):
        """Test that generate logs when debug is enabled."""
        mock_debug_logger.is_enabled.return_value = True
        
        mock_response = {
            "content": [{"type": "text", "text": "Response"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10}
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_response)
        )
        
        messages = [Message(role="user", content="Test")]
        claude_llm.generate(messages=messages, model="claude-sonnet-4-5")
        
        # Verify logging was called
        assert mock_debug_logger.log_request.called
        assert mock_debug_logger.log_response.called
    
    @patch("expert_among_us.llm.claude_code.subprocess.run")
    @patch("expert_among_us.llm.claude_code.DebugLogger")
    def test_generate_no_logs_when_debug_disabled(self, mock_debug_logger, mock_run, claude_llm):
        """Test that generate doesn't log when debug is disabled."""
        mock_debug_logger.is_enabled.return_value = False
        
        mock_response = {
            "content": [{"type": "text", "text": "Response"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10}
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_response)
        )
        
        messages = [Message(role="user", content="Test")]
        claude_llm.generate(messages=messages, model="claude-sonnet-4-5")
        
        # Verify logging was not called
        assert not mock_debug_logger.log_request.called
        assert not mock_debug_logger.log_response.called