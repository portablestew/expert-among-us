"""Tests for debug logging infrastructure."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from expert_among_us.utils.debug import DebugLogger


@pytest.fixture
def temp_log_dir():
    """Create a temporary log directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_debug_logger():
    """Reset DebugLogger state before and after each test."""
    # Reset before test
    DebugLogger.configure(enabled=False, log_dir=None)
    yield
    # Reset after test
    DebugLogger.configure(enabled=False, log_dir=None)


def test_debug_logger_disabled_by_default():
    """Test that debug logger is disabled by default."""
    assert DebugLogger.is_enabled() is False


def test_debug_logger_configure_enabled(temp_log_dir):
    """Test configuring debug logger to be enabled."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    assert DebugLogger.is_enabled() is True


def test_debug_logger_configure_creates_directory(temp_log_dir):
    """Test that configure creates the log directory if it doesn't exist."""
    log_dir = temp_log_dir / "nested" / "logs"
    assert not log_dir.exists()
    
    DebugLogger.configure(enabled=True, log_dir=log_dir)
    
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_debug_logger_uses_default_directory():
    """Test that default log directory is ~/.expert-among-us/logs."""
    DebugLogger.configure(enabled=True)
    
    expected_dir = Path.home() / ".expert-among-us" / "logs"
    assert DebugLogger._log_dir == expected_dir


def test_log_request_when_disabled(temp_log_dir):
    """Test that no files are created when debug mode is disabled."""
    DebugLogger.configure(enabled=False, log_dir=temp_log_dir)
    
    DebugLogger.log_request("test_operation", {"key": "value"})
    
    # No files should be created (check recursively for subdirectories)
    files = list(temp_log_dir.glob("**/*.json"))
    assert len(files) == 0


def test_log_response_when_disabled(temp_log_dir):
    """Test that no files are created when debug mode is disabled."""
    DebugLogger.configure(enabled=False, log_dir=temp_log_dir)
    
    DebugLogger.log_response("test_operation", {"key": "value"})
    
    # No files should be created (check recursively for subdirectories)
    files = list(temp_log_dir.glob("**/*.json"))
    assert len(files) == 0


def test_log_request_creates_file(temp_log_dir):
    """Test that log_request creates a file with correct naming in subdirectory."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    DebugLogger.log_request("converse", {"model": "test"})
    
    # Files are now in subdirectories by category
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    assert len(files) == 1
    assert files[0].name.startswith("bedrock_")
    assert files[0].name.endswith("_request.json")
    assert files[0].suffix == ".json"
    # Should be in a category subdirectory (default is "general")
    assert files[0].parent.name in ["general", "embedding", "promptgen", "expert"]


def test_log_response_creates_file(temp_log_dir):
    """Test that log_response creates a file with correct naming in subdirectory."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    DebugLogger.log_response("converse", {"output": "test"})
    
    # Files are now in subdirectories by category
    files = list(temp_log_dir.glob("**/bedrock_*_response.json"))
    assert len(files) == 1
    assert files[0].name.startswith("bedrock_")
    assert files[0].name.endswith("_response.json")
    assert files[0].suffix == ".json"
    # Should be in a category subdirectory (default is "general")
    assert files[0].parent.name in ["general", "embedding", "promptgen", "expert"]


def test_log_request_json_content(temp_log_dir):
    """Test that log_request writes valid JSON with correct structure."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    test_payload = {"model": "test-model", "messages": ["hello"]}
    DebugLogger.log_request("converse", test_payload)
    
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    assert len(files) == 1
    
    with open(files[0], "r", encoding="utf-8") as f:
        content = json.load(f)
    
    assert content["type"] == "request"
    assert content["operation"] == "converse"
    assert content["payload"] == test_payload
    assert "timestamp" in content
    
    # Validate timestamp is ISO 8601 format
    datetime.fromisoformat(content["timestamp"])


def test_log_response_json_content(temp_log_dir):
    """Test that log_response writes valid JSON with correct structure."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    test_payload = {"output": "test output", "usage": {"tokens": 100}}
    DebugLogger.log_response("converse", test_payload)
    
    files = list(temp_log_dir.glob("**/bedrock_*_response.json"))
    assert len(files) == 1
    
    with open(files[0], "r", encoding="utf-8") as f:
        content = json.load(f)
    
    assert content["type"] == "response"
    assert content["operation"] == "converse"
    assert content["payload"] == test_payload
    assert "timestamp" in content
    
    # Validate timestamp is ISO 8601 format
    datetime.fromisoformat(content["timestamp"])


def test_json_formatted_with_indent(temp_log_dir):
    """Test that JSON is formatted with indent=2 for readability."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    test_payload = {"key": "value", "nested": {"data": "test"}}
    DebugLogger.log_request("test", test_payload)
    
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    with open(files[0], "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check for proper indentation (should have newlines and spaces)
    assert "\n" in content
    assert "  " in content  # 2-space indentation


def test_datetime_serialization(temp_log_dir):
    """Test that datetime objects are properly serialized."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    test_payload = {
        "timestamp": datetime(2024, 1, 1, 12, 0, 0),
        "data": "test"
    }
    DebugLogger.log_request("test", test_payload)
    
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    with open(files[0], "r", encoding="utf-8") as f:
        content = json.load(f)
    
    # Should serialize datetime as ISO string
    assert "2024-01-01" in content["payload"]["timestamp"]


def test_path_serialization(temp_log_dir):
    """Test that Path objects are properly serialized."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    test_payload = {
        "path": Path("/test/path"),
        "data": "test"
    }
    DebugLogger.log_request("test", test_payload)
    
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    with open(files[0], "r", encoding="utf-8") as f:
        content = json.load(f)
    
    # Should serialize Path as string
    assert isinstance(content["payload"]["path"], str)
    assert "test" in content["payload"]["path"]
    assert "path" in content["payload"]["path"]


def test_multiple_log_files_unique_names(temp_log_dir):
    """Test that multiple logs create unique filenames."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    # Log multiple times
    for i in range(3):
        DebugLogger.log_request("test", {"index": i})
    
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    assert len(files) == 3
    
    # All filenames should be unique
    filenames = [f.name for f in files]
    assert len(filenames) == len(set(filenames))


def test_request_and_response_separate_files(temp_log_dir):
    """Test that requests and responses are logged to separate files."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    DebugLogger.log_request("test", {"input": "data"})
    DebugLogger.log_response("test", {"output": "data"})
    
    request_files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    response_files = list(temp_log_dir.glob("**/bedrock_*_response.json"))
    
    assert len(request_files) == 1
    assert len(response_files) == 1


def test_filename_contains_iso8601_timestamp(temp_log_dir):
    """Test that filename contains ISO 8601 formatted timestamp."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    DebugLogger.log_request("test", {"data": "test"})
    
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    filename = files[0].stem  # Get filename without extension
    
    # Should have format: bedrock_<timestamp>_<uuid>_request
    parts = filename.split("_")
    assert len(parts) == 4  # bedrock, timestamp, uuid (4 chars), request
    
    # Should end with request
    assert parts[-1] == "request"
    
    # Timestamp should be in format like 20251029T054015Z
    timestamp = parts[1]
    assert timestamp.endswith('Z')
    assert 'T' in timestamp
    assert len(timestamp) == 16  # YYYYMMDDTHHMMSSs + Z


def test_handles_complex_nested_data(temp_log_dir):
    """Test logging complex nested data structures."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    complex_payload = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"}
        ],
        "config": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "nested": {
                "deep": {
                    "value": "test"
                }
            }
        }
    }
    
    DebugLogger.log_request("converse", complex_payload)
    
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    with open(files[0], "r", encoding="utf-8") as f:
        content = json.load(f)
    
    assert content["payload"] == complex_payload


def test_silent_failure_on_write_error(temp_log_dir):
    """Test that write errors don't raise exceptions (silent failure)."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    # Make directory read-only to cause write failure
    temp_log_dir.chmod(0o444)
    
    try:
        # Should not raise exception even though write fails
        DebugLogger.log_request("test", {"data": "test"})
    except Exception as e:
        pytest.fail(f"log_request should not raise exception: {e}")
    finally:
        # Restore permissions for cleanup
        temp_log_dir.chmod(0o755)


def test_configure_multiple_times(temp_log_dir):
    """Test that configure can be called multiple times."""
    # First configuration
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    assert DebugLogger.is_enabled() is True
    
    # Disable
    DebugLogger.configure(enabled=False)
    assert DebugLogger.is_enabled() is False
    
    # Re-enable
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    assert DebugLogger.is_enabled() is True


def test_thread_safety_multiple_logs(temp_log_dir):
    """Test that multiple log calls are thread-safe."""
    import threading
    
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    def log_multiple():
        for i in range(5):
            DebugLogger.log_request("test", {"thread_id": threading.current_thread().ident, "index": i})
    
    threads = [threading.Thread(target=log_multiple) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should have 15 files total (3 threads * 5 logs each)
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    assert len(files) == 15
    
    # All files should be valid JSON
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            json.load(file)


def test_is_enabled_returns_bool():
    """Test that is_enabled returns a boolean."""
    result = DebugLogger.is_enabled()
    assert isinstance(result, bool)


def test_empty_payload(temp_log_dir):
    """Test logging empty payload."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    DebugLogger.log_request("test", {})
    
    files = list(temp_log_dir.glob("**/bedrock_*_request.json"))
    with open(files[0], "r", encoding="utf-8") as f:
        content = json.load(f)
    
    assert content["payload"] == {}


def test_category_subdirectories(temp_log_dir):
    """Test that logs are organized into category subdirectories."""
    DebugLogger.configure(enabled=True, log_dir=temp_log_dir)
    
    # Log to different categories
    DebugLogger.log_request("converse", {"data": "expert"}, category="expert")
    DebugLogger.log_request("embed", {"data": "embedding"}, category="embedding")
    DebugLogger.log_request("generate", {"data": "promptgen"}, category="promptgen")
    DebugLogger.log_request("other", {"data": "general"}, category="general")
    
    # Check that each category subdirectory exists
    assert (temp_log_dir / "expert").exists()
    assert (temp_log_dir / "embedding").exists()
    assert (temp_log_dir / "promptgen").exists()
    assert (temp_log_dir / "general").exists()
    
    # Check that each category has exactly one log file
    expert_files = list((temp_log_dir / "expert").glob("bedrock_*_request.json"))
    embedding_files = list((temp_log_dir / "embedding").glob("bedrock_*_request.json"))
    promptgen_files = list((temp_log_dir / "promptgen").glob("bedrock_*_request.json"))
    general_files = list((temp_log_dir / "general").glob("bedrock_*_request.json"))
    
    assert len(expert_files) == 1
    assert len(embedding_files) == 1
    assert len(promptgen_files) == 1
    assert len(general_files) == 1
    
    # Verify content is correctly categorized
    with open(expert_files[0], "r", encoding="utf-8") as f:
        content = json.load(f)
        assert content["payload"]["data"] == "expert"
    
    with open(embedding_files[0], "r", encoding="utf-8") as f:
        content = json.load(f)
        assert content["payload"]["data"] == "embedding"