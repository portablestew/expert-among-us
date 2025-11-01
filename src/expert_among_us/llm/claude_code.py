"""Claude Code CLI LLM provider implementation with session-based conversations."""

import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import AsyncIterator, List, Optional, Dict, Any

from .base import (
    LLMProvider,
    Message,
    LLMResponse,
    StreamChunk,
    UsageMetrics,
    LLMError,
    LLMRateLimitError,
    LLMInvalidRequestError,
)
from ..utils.debug import DebugLogger


class ClaudeCodeLLM(LLMProvider):
    """Claude Code CLI implementation with session-based multi-turn conversations.
    
    Uses the Claude CLI's session file mechanism with the --resume flag to support
    multi-turn conversations. Session files are created in JSONL format and cleaned
    up after use. Emulates Claude's project directory structure.
    """
    
    def __init__(
        self,
        cli_path: str = "claude",
        project_dir: Optional[Path] = None,
    ):
        """Initialize Claude Code CLI provider.
        
        Args:
            cli_path: Path to claude CLI executable (default: "claude")
            project_dir: Project directory to run Claude from (default: current working directory)
            
        Raises:
            LLMError: If claude CLI is not found
        """
        # Check if claude CLI exists and get full path
        full_cli_path = shutil.which(cli_path)
        if not full_cli_path:
            raise LLMError(
                f"Claude CLI not found at '{cli_path}'. "
                "Please install the Claude CLI from https://github.com/anthropics/claude-cli"
            )
        
        # Store full path for Windows compatibility with subprocess
        self.cli_path = full_cli_path
        # Store project directory where we'll run Claude from
        self.project_dir = project_dir or Path.cwd()
        # Create session directory using Claude's naming convention
        self.session_dir = self._get_claude_session_dir(self.project_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_claude_session_dir(self, project_path: Path) -> Path:
        r"""Get Claude-style session directory for a project path.
        
        Claude sanitizes paths by removing the drive colon and replacing
        path separators with dashes. For example:
        c:\Dev\github\expert-among-us -> c--Dev-github-expert-among-us
        
        Args:
            project_path: Absolute path to project directory
            
        Returns:
            Path to session directory in ~/.claude/projects/
        """
        # Convert to absolute path string and normalize to lowercase
        abs_path = str(project_path.resolve()).lower()
        
        # Replace drive colon with dash, then replace path separators with dashes
        # This creates the double-dash pattern: c: -> c-, then \dev -> -dev = c--dev-...
        sanitized = abs_path.replace(':', '-').replace('\\', '-').replace('/', '-')
        
        return Path.home() / ".claude" / "projects" / sanitized
    
    def _create_session_id(self) -> str:
        """Generate a unique session ID.
        
        Returns:
            UUID string for session ID
        """
        return str(uuid.uuid4())
    
    def _create_session_file(
        self,
        session_id: str,
        messages: List[Message],
        system: Optional[str] = None
    ) -> Path:
        """Create a session file with conversation history.
        
        Args:
            session_id: Unique session identifier
            messages: Conversation history
            system: Optional system prompt
            
        Returns:
            Path to created session file
        """
        session_file = self.session_dir / f"{session_id}.jsonl"
        
        # Build session events with realistic timestamps
        events = []
        base_time = datetime.now(timezone.utc)
        
        # Add summary event (required first line) - matches Claude's actual format
        events.append({
            "type": "summary",
            "summary": "Claude Code CLI Session Started",
            "leafUuid": str(uuid.uuid4())
        })
        
        # Add file-history-snapshot event (required after summary)
        # This appears to be necessary for Claude to recognize the session properly
        first_msg_uuid = str(uuid.uuid4())
        events.append({
            "type": "file-history-snapshot",
            "messageId": first_msg_uuid,
            "snapshot": {
                "messageId": first_msg_uuid,
                "trackedFileBackups": {},
                "timestamp": base_time.isoformat()
            },
            "isSnapshotUpdate": False
        })
        
        # Add message events with incremental timestamps
        parent_uuid = None
        time_offset = 0
        for i, msg in enumerate(messages):
            msg_uuid = str(uuid.uuid4())
            # Increment timestamp by 1-3 seconds per message for realism
            time_offset += (1 + i % 3)
            timestamp = (base_time + timedelta(seconds=time_offset)).isoformat()
            
            if msg.role == "user":
                event = {
                    "parentUuid": parent_uuid,
                    "isSidechain": False,
                    "userType": "external",
                    "cwd": str(self.project_dir),
                    "sessionId": session_id,
                    "version": "2.0.29",
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": msg.content
                    },
                    "uuid": msg_uuid,
                    "timestamp": timestamp,
                    "thinkingMetadata": {
                        "level": "high",
                        "disabled": False,
                        "triggers": []
                    }
                }
            else:  # assistant
                event = {
                    "parentUuid": parent_uuid,
                    "isSidechain": False,
                    "userType": "external",
                    "cwd": str(self.project_dir),
                    "sessionId": session_id,
                    "version": "2.0.29",
                    "type": "assistant",
                    "message": {
                        "model": "claude-sonnet-4-5",
                        "id": f"msg_{uuid.uuid4().hex[:20]}",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": msg.content}],
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0
                        }
                    },
                    "uuid": msg_uuid,
                    "timestamp": timestamp
                }
            
            events.append(event)
            parent_uuid = msg_uuid
        
        # Write session file
        self._write_session_events(session_file, events)
        
        return session_file
    
    def _write_session_events(self, session_file: Path, events: List[Dict[str, Any]]) -> None:
        """Write session events to JSONL file.
        
        Args:
            session_file: Path to session file
            events: List of event dictionaries to write
        """
        with open(session_file, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
    
    def _parse_json_response(self, output: str) -> Dict[str, Any]:
        """Parse JSON response from claude CLI.
        
        Args:
            output: Raw output from CLI
            
        Returns:
            Parsed JSON response
            
        Raises:
            LLMError: If response cannot be parsed
        """
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse Claude response: {e}\nOutput: {output[:200]}")
    
    def _parse_stream_event(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a streaming event line.
        
        Args:
            line: Single line from streaming output
            
        Returns:
            Parsed event dictionary or None if not valid JSON
        """
        line = line.strip()
        if not line:
            return None
        
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    
    def _extract_usage_metrics(self, response: Dict[str, Any]) -> UsageMetrics:
        """Extract usage metrics from response.
        
        Args:
            response: Parsed response from CLI
            
        Returns:
            UsageMetrics object
        """
        usage = response.get("usage", {})
        return UsageMetrics(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        )
    
    def _handle_subprocess_error(self, e: subprocess.CalledProcessError) -> None:
        """Handle subprocess errors and convert to appropriate LLM exceptions.
        
        Args:
            e: CalledProcessError from subprocess
            
        Raises:
            LLMRateLimitError: If rate limit exceeded
            LLMInvalidRequestError: If invalid request
            LLMError: For other errors
        """
        error_output = e.stderr if e.stderr else str(e)
        
        # Check for rate limit
        if "rate limit" in error_output.lower() or "429" in error_output:
            raise LLMRateLimitError(f"Rate limit exceeded: {error_output}")
        
        # Check for validation errors
        if "invalid" in error_output.lower() or "400" in error_output:
            raise LLMInvalidRequestError(f"Invalid request: {error_output}")
        
        # Generic error
        raise LLMError(f"Claude CLI error (code {e.returncode}): {error_output}")
    
    def generate(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> LLMResponse:
        """Generate complete response using Claude CLI.
        
        Args:
            messages: Conversation history as a list of Message objects
            model: Model identifier (passed to CLI but may be ignored)
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-2.0 (default: 1.0)
            system: Optional system prompt to set behavior
            debug_category: Category for debug logging (default: "expert")
            
        Returns:
            LLMResponse containing generated content and metadata
            
        Raises:
            LLMRateLimitError: If rate limit exceeded
            LLMInvalidRequestError: If request parameters invalid
            LLMError: For other Claude CLI errors
        """
        session_id = self._create_session_id()
        session_file = None
        
        try:
            # Separate conversation history from current user prompt
            # Session file contains all messages EXCEPT the last user message
            # The last user message is passed as CLI argument
            history_messages = []
            current_prompt = ""
            
            if messages:
                # Find the last user message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].role == "user":
                        current_prompt = messages[i].content
                        history_messages = messages[:i]
                        break
            
            # Build command - only use --resume if we have conversation history
            # Add safeguard flags to prevent unwanted file access and MCP interactions
            cmd = [
                self.cli_path,
                "--print",
                "--output-format", "json",
                "--model", model,  # Specify the model to use
                "--allowed-tools", "",  # Disable all tools to prevent file access
                "--strict-mcp-config",  # Only use explicit MCP config, ignore defaults
            ]
            
            # Create session file and add --resume flag only if we have conversation history
            if history_messages:
                session_file = self._create_session_file(session_id, history_messages, system)
                # Insert --resume after claude command but before other flags
                cmd.insert(1, "--resume")
                cmd.insert(2, session_id)
            # For single-turn (no history), don't create session file or use --resume
            
            # Add system prompt if provided
            # Replace newlines with spaces to prevent command line parsing issues
            if system:
                system_clean = system.replace('\n', ' ').replace('\r', '')
                cmd.extend(["--system-prompt", system_clean])
            
            # Don't append prompt to command - will pass via stdin instead
            # This avoids Windows command-line length limits (~8191 chars)
            
            # Log request if debug enabled - use raw session file content
            request_id = None
            if DebugLogger.is_enabled():
                # Read raw session file content for debugging (if exists)
                session_events = []
                if session_file:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_events = [json.loads(line) for line in f if line.strip()]
                
                request_data = {
                    "session_id": session_id if session_file else None,
                    "current_prompt": current_prompt,
                    "session_file": str(session_file) if session_file else None,
                    "session_events": session_events,
                    "command": cmd,
                    "is_single_turn": not bool(session_file)
                }
                request_id = DebugLogger.log_request("claude_code", request_data, category=debug_category)
                
                # Copy session file to debug directory for manual testing (if exists)
                if DebugLogger._log_dir and session_file:
                    category_dir = DebugLogger._log_dir / debug_category
                    category_dir.mkdir(parents=True, exist_ok=True)
                    now = datetime.now(timezone.utc)
                    timestamp = now.strftime('%Y%m%dT%H%M%SZ')
                    short_request_id = request_id[:4]
                    session_copy = category_dir / f"claude_code_{timestamp}_{short_request_id}_session.jsonl"
                    shutil.copy2(session_file, session_copy)
            
            # Execute CLI command from project directory
            try:
                # Log encoding diagnostics and session info
                if DebugLogger.is_enabled():
                    import locale
                    import sys
                    DebugLogger.log_request("encoding_diagnostic", {
                        "subprocess_default_encoding": locale.getpreferredencoding(),
                        "stdout_encoding": sys.stdout.encoding,
                        "stderr_encoding": sys.stderr.encoding,
                        "file_system_encoding": sys.getfilesystemencoding(),
                        "session_file_exists": session_file.exists() if session_file else False,
                        "history_message_count": len(history_messages),
                        "has_current_prompt": bool(current_prompt)
                    }, category=debug_category)
                
                result = subprocess.run(
                    cmd,
                    input=current_prompt,  # Pass prompt via stdin to avoid command-line length limits
                    capture_output=True,
                    text=True,
                    encoding='utf-8',  # Explicit UTF-8 to handle Claude's output on Windows
                    errors='replace',  # Replace invalid sequences instead of crashing
                    check=True,
                    cwd=str(self.project_dir)
                )
            except UnicodeDecodeError as e:
                # Capture encoding error details
                error_msg = (
                    f"UTF-8 encoding error in Claude CLI output: {e}\n"
                    f"This suggests Claude output UTF-8 but Python used {e.encoding} encoding.\n"
                    f"Problematic byte: 0x{e.object[e.start:e.start+1].hex()} at position {e.start}"
                )
                if DebugLogger.is_enabled():
                    DebugLogger.log_response("encoding_error", {
                        "error": str(e),
                        "encoding": e.encoding,
                        "position": e.start,
                        "byte": e.object[e.start:e.start+1].hex() if e.start < len(e.object) else None
                    }, request_id, category=debug_category)
                raise LLMError(error_msg)
            except subprocess.CalledProcessError as e:
                # Log additional session diagnostics on error
                if DebugLogger.is_enabled() and session_file and session_file.exists():
                    DebugLogger.log_response("session_error_diagnostic", {
                        "session_id": session_id,
                        "session_file": str(session_file),
                        "session_dir": str(self.session_dir),
                        "command": cmd,
                        "stderr": e.stderr,
                        "returncode": e.returncode
                    }, request_id, category=debug_category)
                self._handle_subprocess_error(e)
            
            # Parse response
            response = self._parse_json_response(result.stdout)
            
            # Log response if debug enabled
            if DebugLogger.is_enabled():
                DebugLogger.log_response("claude_code", response, request_id, category=debug_category)
            
            # Extract content - handle both API format (content) and CLI format (result)
            content = ""
            
            # Try API format first (content field with structured blocks)
            if "content" in response:
                if isinstance(response["content"], list):
                    for block in response["content"]:
                        if block.get("type") == "text":
                            content += block.get("text", "")
                elif isinstance(response["content"], str):
                    content = response["content"]
            
            # Fall back to CLI format (result field with plain text)
            if not content and "result" in response:
                content = response["result"]
            
            # Validate that we got non-empty content
            if not content or content.strip() == "":
                error_details = {
                    "response_keys": list(response.keys()),
                    "has_content": "content" in response,
                    "has_result": "result" in response,
                    "session_id": session_id if session_file else None,
                    "is_single_turn": not bool(session_file)
                }
                raise LLMError(
                    f"Claude Code returned empty response. This usually indicates a session setup issue. "
                    f"Details: {error_details}"
                )
            
            # Extract usage metrics
            usage_metrics = self._extract_usage_metrics(response)
            
            return LLMResponse(
                content=content,
                model=response.get("model", model),
                stop_reason=response.get("stop_reason", "end_turn"),
                usage=usage_metrics
            )
            
        finally:
            # Clean up session file (unless debug is enabled)
            if session_file and session_file.exists() and not DebugLogger.is_enabled():
                try:
                    session_file.unlink()
                except Exception:
                    pass  # Best effort cleanup
    
    async def stream(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> AsyncIterator[StreamChunk]:
        """Stream response using Claude CLI.
        
        Args:
            messages: Conversation history as a list of Message objects
            model: Model identifier (passed to CLI but may be ignored)
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-2.0 (default: 1.0)
            system: Optional system prompt to set behavior
            debug_category: Category for debug logging (default: "expert")
            
        Yields:
            StreamChunk objects containing incremental content
            
        Raises:
            LLMRateLimitError: If rate limit exceeded
            LLMInvalidRequestError: If request parameters invalid
            LLMError: For other Claude CLI errors
        """
        from ..utils.debug import DebugLogger
        import sys
        
        session_id = self._create_session_id()
        session_file = None
        
        try:
            # Separate conversation history from current user prompt
            # Session file contains all messages EXCEPT the last user message
            # The last user message is passed as CLI argument
            history_messages = []
            current_prompt = ""
            
            if messages:
                # Find the last user message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].role == "user":
                        current_prompt = messages[i].content
                        history_messages = messages[:i]
                        break
            
            # Build command - only use --resume if we have conversation history
            # Add safeguard flags to prevent unwanted file access and MCP interactions
            cmd = [
                self.cli_path,
                "--print",
                "--verbose",
                "--output-format", "stream-json",
                "--model", model,  # Specify the model to use
                "--allowed-tools", "",  # Disable all tools to prevent file access
                "--strict-mcp-config",  # Only use explicit MCP config, ignore defaults
            ]
            
            # Create session file and add --resume flag only if we have conversation history
            if history_messages:
                session_file = self._create_session_file(session_id, history_messages, system)
                # Insert --resume after claude command but before other flags
                cmd.insert(1, "--resume")
                cmd.insert(2, session_id)
            # For single-turn (no history), don't create session file or use --resume
            
            # Add system prompt if provided
            # Replace newlines with spaces to prevent command line parsing issues
            if system:
                system_clean = system.replace('\n', ' ').replace('\r', '')
                cmd.extend(["--system-prompt", system_clean])
            
            # Don't append prompt to command - will pass via stdin instead
            # This avoids Windows command-line length limits (~8191 chars)
            
            # Log request if debug enabled - use raw session file content
            request_id = None
            all_events = []  # Always collect events for error extraction
            if DebugLogger.is_enabled():
                # Read raw session file content for debugging (if exists)
                session_events = []
                if session_file:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_events = [json.loads(line) for line in f if line.strip()]
                
                request_data = {
                    "session_id": session_id if session_file else None,
                    "current_prompt": current_prompt,
                    "session_file": str(session_file) if session_file else None,
                    "session_events": session_events,
                    "command": cmd,
                    "is_single_turn": not bool(session_file)
                }
                request_id = DebugLogger.log_request("claude_code_stream", request_data, category=debug_category)
                
                # Copy session file to debug directory for manual testing (if exists)
                if DebugLogger._log_dir and session_file:
                    category_dir = DebugLogger._log_dir / debug_category
                    category_dir.mkdir(parents=True, exist_ok=True)
                    now = datetime.now(timezone.utc)
                    timestamp = now.strftime('%Y%m%dT%H%M%SZ')
                    short_request_id = request_id[:4]
                    session_copy = category_dir / f"claude_code_stream_{timestamp}_{short_request_id}_session.jsonl"
                    shutil.copy2(session_file, session_copy)
            
            # Execute CLI command with streaming from project directory
            try:
                # Log encoding diagnostics
                if DebugLogger.is_enabled():
                    import locale
                    import sys
                    DebugLogger.log_request("stream_encoding_diagnostic", {
                        "subprocess_default_encoding": locale.getpreferredencoding(),
                        "stdout_encoding": sys.stdout.encoding,
                        "stderr_encoding": sys.stderr.encoding,
                        "file_system_encoding": sys.getfilesystemencoding()
                    }, category=debug_category)
                
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',  # Explicit UTF-8 to handle Claude's output on Windows
                    errors='replace',  # Replace invalid sequences instead of crashing
                    bufsize=1,  # Line buffered
                    universal_newlines=True,
                    cwd=str(self.project_dir)
                )
                
                # Write prompt to stdin and close it
                process.stdin.write(current_prompt)
                process.stdin.close()
                
                # Read streaming output
                final_usage = None
                final_stop_reason = None
                raw_lines = []  # Collect all raw lines for debugging
                total_content = ""  # Track total content for validation
                
                for line in process.stdout:
                    try:
                        # Collect raw line for debugging
                        if DebugLogger.is_enabled():
                            raw_lines.append(line.strip())
                    except UnicodeDecodeError as e:
                        # Capture encoding error details in streaming
                        error_msg = (
                            f"UTF-8 encoding error in Claude CLI stream: {e}\n"
                            f"This suggests Claude output UTF-8 but Python used {e.encoding} encoding.\n"
                            f"Problematic byte: 0x{e.object[e.start:e.start+1].hex()} at position {e.start}"
                        )
                        if DebugLogger.is_enabled():
                            DebugLogger.log_response("stream_encoding_error", {
                                "error": str(e),
                                "encoding": e.encoding,
                                "position": e.start,
                                "byte": e.object[e.start:e.start+1].hex() if e.start < len(e.object) else None,
                                "collected_events": len(all_events)
                            }, request_id, category=debug_category)
                        raise LLMError(error_msg)
                    
                    event = self._parse_stream_event(line)
                    if not event:
                        continue
                    
                    # Always collect events for error extraction
                    all_events.append(event)
                    
                    event_type = event.get("type")
                    
                    # Content delta events (streaming format)
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                total_content += text
                                yield StreamChunk(delta=text)
                    
                    # Assistant message event (contains full response in message.content)
                    elif event_type == "assistant":
                        message = event.get("message", {})
                        content = message.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        total_content += text
                                        yield StreamChunk(delta=text)
                        elif isinstance(content, str) and content:
                            total_content += content
                            yield StreamChunk(delta=content)
                    
                    # Result event (contains accurate usage metrics but duplicate text)
                    # Don't yield text here - it's already in the assistant event
                    # Instead, capture the complete usage metrics for final reporting
                    elif event_type == "result":
                        # Extract accurate usage from result event
                        result_usage = event.get("usage", {})
                        if result_usage:
                            final_usage = result_usage
                        # Extract stop reason if present
                        if not final_stop_reason and "subtype" in event:
                            final_stop_reason = event.get("subtype", "end_turn")
                    
                    # Message delta events (alternative format)
                    elif event_type == "message_delta":
                        delta = event.get("delta", {})
                        if "stop_reason" in delta:
                            final_stop_reason = delta["stop_reason"]
                        if "usage" in delta:
                            final_usage = delta["usage"]
                    
                    # Message stop event
                    elif event_type == "message_stop":
                        pass  # Just marks end of stream
                    
                    # Usage metadata (final event)
                    elif event_type == "message_end" or "usage" in event:
                        if "usage" in event:
                            final_usage = event["usage"]
                        if "stop_reason" in event:
                            final_stop_reason = event["stop_reason"]
                
                # Wait for process to complete
                process.wait()
                
                if process.returncode != 0:
                    # Try to extract user-friendly error from the last event
                    error_msg = None
                    if all_events:
                        last_event = all_events[-1]
                        if isinstance(last_event, dict) and last_event.get("is_error"):
                            # Prefer the "result" field for cleaner error messages
                            error_msg = last_event.get("result")
                            if not error_msg:
                                # Fallback to message or full dict
                                error_msg = last_event.get("message", str(last_event))
                    
                    # Use stderr if no error extracted from events
                    if not error_msg:
                        stderr_output = process.stderr.read() if process.stderr else ""
                        error_msg = stderr_output if stderr_output else "Unknown error"
                    
                    raise subprocess.CalledProcessError(
                        process.returncode,
                        cmd,
                        stderr=str(error_msg)
                    )
                
                # Validate that we got non-empty content
                if not total_content or total_content.strip() == "":
                    error_details = {
                        "num_events": len(all_events),
                        "event_types": [e.get("type") for e in all_events if isinstance(e, dict)],
                        "session_id": session_id if session_file else None,
                        "is_single_turn": not bool(session_file)
                    }
                    raise LLMError(
                        f"Claude Code returned empty streaming response. This usually indicates a session setup issue. "
                        f"Details: {error_details}"
                    )
                
                # Yield final chunk with usage
                if final_usage or final_stop_reason:
                    usage_metrics = None
                    if final_usage:
                        usage_metrics = UsageMetrics(
                            input_tokens=final_usage.get("input_tokens", 0),
                            output_tokens=final_usage.get("output_tokens", 0),
                            total_tokens=final_usage.get("input_tokens", 0) + final_usage.get("output_tokens", 0),
                            cache_read_tokens=final_usage.get("cache_read_input_tokens", 0),
                            cache_creation_tokens=final_usage.get("cache_creation_input_tokens", 0),
                        )
                    
                    yield StreamChunk(
                        delta="",
                        stop_reason=final_stop_reason,
                        usage=usage_metrics
                    )
                
                # Log all events if debug enabled
                if DebugLogger.is_enabled():
                    response_data = {
                        "events": all_events,
                        "raw_event_lines": raw_lines,  # Raw JSON lines from stdout
                        "num_events": len(all_events),
                        "num_raw_lines": len(raw_lines),
                        "event_types": [e.get("type") for e in all_events if isinstance(e, dict)]
                    }
                    DebugLogger.log_response("claude_code_stream", response_data, request_id, category=debug_category)
                
            except subprocess.CalledProcessError as e:
                self._handle_subprocess_error(e)
                
        finally:
            # Clean up session file (unless debug is enabled)
            if session_file and session_file.exists() and not DebugLogger.is_enabled():
                try:
                    session_file.unlink()
                except Exception:
                    pass  # Best effort cleanup