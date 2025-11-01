"""Tests for LLM provider auto-detection."""

import pytest
from unittest.mock import patch, MagicMock
import os

from expert_among_us.llm.auto_detect import detect_llm_provider, _check_aws_credentials, _check_ollama_running


class TestEnvironmentVariableDetection:
    """Tests for environment variable-based provider detection."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_detects_openai_from_env_var(self):
        """Test detection of OpenAI from OPENAI_API_KEY."""
        provider = detect_llm_provider()
        
        assert provider == "openai"
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True)
    def test_detects_openrouter_from_env_var(self):
        """Test detection of OpenRouter from OPENROUTER_API_KEY."""
        provider = detect_llm_provider()
        
        assert provider == "openrouter"
    
    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "AKIATEST"}, clear=True)
    def test_detects_bedrock_from_env_var(self):
        """Test detection of Bedrock from AWS_ACCESS_KEY_ID."""
        provider = detect_llm_provider()
        
        assert provider == "bedrock"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "OPENROUTER_API_KEY": "sk-or-test"}, clear=True)
    def test_fails_with_multiple_env_vars(self):
        """Test failure when multiple API keys are present."""
        with pytest.raises(ValueError) as exc_info:
            detect_llm_provider()
        
        error_msg = str(exc_info.value)
        assert "Multiple LLM providers detected" in error_msg
        assert "openai" in error_msg
        assert "openrouter" in error_msg
        assert "--llm-provider" in error_msg
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "AWS_ACCESS_KEY_ID": "AKIATEST"}, clear=True)
    @patch("expert_among_us.utils.progress.log_info")
    def test_fails_with_openai_and_aws(self, mock_log):
        """Test failure when both OpenAI and AWS keys are present."""
        with pytest.raises(ValueError) as exc_info:
            detect_llm_provider()
        
        error_msg = str(exc_info.value)
        assert "Multiple LLM providers detected" in error_msg
        assert "bedrock" in error_msg or "openai" in error_msg
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or", "AWS_ACCESS_KEY_ID": "AKIA"}, clear=True)
    @patch("expert_among_us.utils.progress.log_info")
    def test_fails_with_openrouter_and_aws(self, mock_log):
        """Test failure when both OpenRouter and AWS keys are present."""
        with pytest.raises(ValueError) as exc_info:
            detect_llm_provider()
        
        error_msg = str(exc_info.value)
        assert "Multiple LLM providers detected" in error_msg
        assert "bedrock" in error_msg or "openrouter" in error_msg


class TestAWSCredentialsDetection:
    """Tests for AWS credentials detection via boto3."""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("expert_among_us.llm.auto_detect._check_aws_credentials")
    @patch("expert_among_us.utils.progress.log_info")
    def test_detects_aws_default_credentials(self, mock_log, mock_check_aws):
        """Test detection of AWS default credentials."""
        mock_check_aws.return_value = True
        
        provider = detect_llm_provider()
        
        assert provider == "bedrock"
        mock_check_aws.assert_called_once()
    
    @patch("boto3.Session")
    def test_check_aws_credentials_returns_true_when_available(self, mock_session_class):
        """Test _check_aws_credentials returns True when credentials exist."""
        mock_session = MagicMock()
        mock_credentials = MagicMock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        
        result = _check_aws_credentials()
        
        assert result is True
    
    @patch("boto3.Session")
    def test_check_aws_credentials_returns_false_when_none(self, mock_session_class):
        """Test _check_aws_credentials returns False when credentials are None."""
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        mock_session_class.return_value = mock_session
        
        result = _check_aws_credentials()
        
        assert result is False
    
    @patch("boto3.Session")
    def test_check_aws_credentials_handles_no_credentials_error(self, mock_session_class):
        """Test _check_aws_credentials handles NoCredentialsError."""
        from botocore.exceptions import NoCredentialsError
        
        mock_session = MagicMock()
        mock_session.get_credentials.side_effect = NoCredentialsError()
        mock_session_class.return_value = mock_session
        
        result = _check_aws_credentials()
        
        assert result is False
    
    @patch("boto3.Session")
    def test_check_aws_credentials_handles_profile_not_found(self, mock_session_class):
        """Test _check_aws_credentials handles ProfileNotFound."""
        from botocore.exceptions import ProfileNotFound
        
        mock_session = MagicMock()
        mock_session.get_credentials.side_effect = ProfileNotFound(profile="default")
        mock_session_class.return_value = mock_session
        
        result = _check_aws_credentials()
        
        assert result is False


class TestClaudeCLIDetection:
    """Tests for Claude Code CLI detection."""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("expert_among_us.llm.auto_detect._check_aws_credentials")
    @patch("expert_among_us.llm.auto_detect.shutil.which")
    @patch("expert_among_us.utils.progress.log_info")
    def test_detects_claude_cli(self, mock_log, mock_which, mock_check_aws):
        """Test detection of Claude CLI when it's on PATH."""
        mock_check_aws.return_value = False
        mock_which.return_value = "/usr/local/bin/claude"
        
        provider = detect_llm_provider()
        
        assert provider == "claude-code"
        mock_which.assert_called_once_with("claude")
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("expert_among_us.llm.auto_detect._check_aws_credentials")
    @patch("expert_among_us.llm.auto_detect.shutil.which")
    @patch("expert_among_us.llm.auto_detect._check_ollama_running")
    def test_skips_claude_when_not_found(self, mock_ollama, mock_which, mock_check_aws):
        """Test skips Claude CLI when not found."""
        mock_check_aws.return_value = False
        mock_which.return_value = None
        mock_ollama.return_value = True
        
        provider = detect_llm_provider()
        
        assert provider == "ollama"


class TestOllamaDetection:
    """Tests for Ollama server detection."""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("expert_among_us.llm.auto_detect._check_aws_credentials")
    @patch("expert_among_us.llm.auto_detect.shutil.which")
    @patch("expert_among_us.llm.auto_detect._check_ollama_running")
    @patch("expert_among_us.utils.progress.log_info")
    def test_detects_ollama_when_running(self, mock_log, mock_ollama, mock_which, mock_check_aws):
        """Test detection of Ollama when server is running."""
        mock_check_aws.return_value = False
        mock_which.return_value = None
        mock_ollama.return_value = True
        
        provider = detect_llm_provider()
        
        assert provider == "ollama"
        mock_ollama.assert_called_once()
    
    @patch("httpx.get")
    def test_check_ollama_returns_true_when_running(self, mock_get):
        """Test _check_ollama_running returns True when Ollama is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Ollama is running"
        mock_get.return_value = mock_response
        
        result = _check_ollama_running()
        
        assert result is True
        mock_get.assert_called_once_with("http://127.0.0.1:11434/", timeout=2.0)
    
    @patch("httpx.get")
    def test_check_ollama_returns_false_when_not_running(self, mock_get):
        """Test _check_ollama_running returns False when server not responding."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = _check_ollama_running()
        
        assert result is False
    
    @patch("httpx.get")
    def test_check_ollama_returns_false_on_connection_error(self, mock_get):
        """Test _check_ollama_running returns False on connection error."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        
        result = _check_ollama_running()
        
        assert result is False
    
    @patch("httpx.get")
    def test_check_ollama_returns_false_on_timeout(self, mock_get):
        """Test _check_ollama_running returns False on timeout."""
        import httpx
        mock_get.side_effect = httpx.TimeoutException("Timeout")
        
        result = _check_ollama_running()
        
        assert result is False


class TestNoProviderFound:
    """Tests for when no provider is detected."""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("expert_among_us.llm.auto_detect._check_aws_credentials")
    @patch("expert_among_us.llm.auto_detect.shutil.which")
    @patch("expert_among_us.llm.auto_detect._check_ollama_running")
    def test_raises_error_when_no_provider_found(self, mock_ollama, mock_which, mock_check_aws):
        """Test raises error when no provider is detected."""
        mock_check_aws.return_value = False
        mock_which.return_value = None
        mock_ollama.return_value = False
        
        with pytest.raises(ValueError) as exc_info:
            detect_llm_provider()
        
        error_msg = str(exc_info.value)
        assert "No LLM provider detected" in error_msg
        assert "--llm-provider" in error_msg
        assert "README" in error_msg


class TestDetectionPriority:
    """Tests for detection priority order."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch("expert_among_us.llm.auto_detect._check_aws_credentials")
    @patch("expert_among_us.llm.auto_detect.shutil.which")
    @patch("expert_among_us.llm.auto_detect._check_ollama_running")
    @patch("expert_among_us.utils.progress.log_info")
    def test_env_var_takes_priority_over_all(self, mock_log, mock_ollama, mock_which, mock_check_aws):
        """Test environment variable takes priority over other detection methods."""
        # Even though AWS, Claude, and Ollama are available, env var wins
        mock_check_aws.return_value = True
        mock_which.return_value = "/usr/local/bin/claude"
        mock_ollama.return_value = True
        
        provider = detect_llm_provider()
        
        assert provider == "openai"
        # Should not check other methods when env var is found
        mock_check_aws.assert_not_called()
        mock_which.assert_not_called()
        mock_ollama.assert_not_called()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("expert_among_us.llm.auto_detect._check_aws_credentials")
    @patch("expert_among_us.llm.auto_detect.shutil.which")
    @patch("expert_among_us.llm.auto_detect._check_ollama_running")
    @patch("expert_among_us.utils.progress.log_info")
    def test_aws_takes_priority_over_claude_and_ollama(self, mock_log, mock_ollama, mock_which, mock_check_aws):
        """Test AWS credentials take priority over Claude and Ollama."""
        mock_check_aws.return_value = True
        mock_which.return_value = "/usr/local/bin/claude"
        mock_ollama.return_value = True
        
        provider = detect_llm_provider()
        
        assert provider == "bedrock"
        # Should not check Claude or Ollama when AWS is found
        mock_which.assert_not_called()
        mock_ollama.assert_not_called()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("expert_among_us.llm.auto_detect._check_aws_credentials")
    @patch("expert_among_us.llm.auto_detect.shutil.which")
    @patch("expert_among_us.llm.auto_detect._check_ollama_running")
    @patch("expert_among_us.utils.progress.log_info")
    def test_claude_takes_priority_over_ollama(self, mock_log, mock_ollama, mock_which, mock_check_aws):
        """Test Claude CLI takes priority over Ollama."""
        mock_check_aws.return_value = False
        mock_which.return_value = "/usr/local/bin/claude"
        mock_ollama.return_value = True
        
        provider = detect_llm_provider()
        
        assert provider == "claude-code"
        # Should not check Ollama when Claude is found
        mock_ollama.assert_not_called()