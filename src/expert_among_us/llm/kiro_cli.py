"""Kiro CLI LLM provider implementation with session-based conversations.

Uses kiro-cli's session file mechanism with --resume-id to support multi-turn
conversations. Session files are fabricated in Kiro's JSONL format, the CLI is
spawned via a PTY (so it sees isTTY=true and honors --resume-id), and the
response is polled from the session JSONL file.

Requires: pywinpty (Windows) or pexpect (Linux/macOS)
"""

import json
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, List, Optional

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


class KiroCliLLM(LLMProvider):
    """Kiro CLI implementation with session-based multi-turn conversations.

    Uses fabricated session files and kiro-cli's --resume-id flag to inject
    conversation history. A PTY is used to keep stdin as a real terminal so
    kiro-cli operates in interactive mode (required for --resume-id to work).
    """

    # Default kiro-cli path
    DEFAULT_CLI_PATH = "kiro-cli"

    def __init__(
        self,
        cli_path: Optional[str] = None,
        sessions_dir: Optional[Path] = None,
        poll_interval: float = 0.5,
        response_timeout: float = 300.0,
        shutdown_timeout: float = 10.0,
    ):
        """Initialize Kiro CLI provider.

        Args:
            cli_path: Path to kiro-cli executable. Defaults to "kiro-cli" (found via PATH).
            sessions_dir: Directory for session files. Defaults to ~/.kiro/sessions/cli/
            poll_interval: Seconds between jsonl polls (default: 0.5)
            response_timeout: Max seconds to wait for response (default: 300)
            shutdown_timeout: Max seconds to wait for /quit to take effect (default: 10)

        Raises:
            LLMError: If kiro-cli is not found or PTY library unavailable
        """
        # Resolve CLI path — verify it exists on PATH
        # On Windows, shutil.which may return uppercase .EXE extension which
        # breaks Toolbox shims (they're case-sensitive). Lowercase the path.
        cli_path = cli_path or self.DEFAULT_CLI_PATH
        full_path = shutil.which(cli_path)
        if not full_path:
            raise LLMError(
                f"kiro-cli not found at '{cli_path}'. "
                "Please ensure kiro-cli is installed and on your PATH."
            )
        if platform.system() == "Windows":
            full_path = full_path.lower()
        self.cli_path = full_path

        # Verify PTY library is available
        self._pty_backend = self._detect_pty_backend()

        # Session storage
        self.sessions_dir = sessions_dir or (Path.home() / ".kiro" / "sessions" / "cli")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Timing
        self.poll_interval = poll_interval
        self.response_timeout = response_timeout
        self.shutdown_timeout = shutdown_timeout

    @staticmethod
    def _detect_pty_backend() -> str:
        """Detect available PTY backend.

        Returns:
            "winpty" or "pexpect"

        Raises:
            LLMError: If no PTY library is available
        """
        if platform.system() == "Windows":
            try:
                import winpty  # noqa: F401
                return "winpty"
            except ImportError:
                raise LLMError(
                    "pywinpty is required on Windows for kiro-cli provider. "
                    "Install with: pip install pywinpty"
                )
        else:
            try:
                import pexpect  # noqa: F401
                return "pexpect"
            except ImportError:
                raise LLMError(
                    "pexpect is required on Linux/macOS for kiro-cli provider. "
                    "Install with: pip install pexpect"
                )

    def _create_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())

    def _create_session_files(
        self,
        session_id: str,
        messages: List[Message],
        system: Optional[str] = None,
    ) -> tuple[Path, Path]:
        """Create session .json and .jsonl files with conversation history.

        The last user message is NOT included in the files — it will be sent
        via stdin to the running kiro-cli process.

        Args:
            session_id: Unique session identifier
            messages: Full conversation history (excluding the last user message)
            system: Optional system prompt (baked into first exchange)

        Returns:
            Tuple of (json_path, jsonl_path)
        """
        json_path = self.sessions_dir / f"{session_id}.json"
        jsonl_path = self.sessions_dir / f"{session_id}.jsonl"

        # Build JSONL content
        jsonl_lines = []
        base_timestamp = int(time.time()) - len(messages) * 2  # Spread timestamps

        # If system prompt provided, inject as first user/assistant exchange
        effective_messages = []
        if system:
            effective_messages.append(Message(role="user", content=system))
            effective_messages.append(Message(role="assistant", content="Understood. I will follow these instructions."))
        effective_messages.extend(messages)

        for i, msg in enumerate(effective_messages):
            msg_id = str(uuid.uuid4())
            timestamp = base_timestamp + i * 2

            if msg.role == "user":
                entry = {
                    "version": "v1",
                    "kind": "Prompt",
                    "data": {
                        "message_id": msg_id,
                        "content": [{"kind": "text", "data": msg.content}],
                        "meta": {"timestamp": timestamp},
                    },
                }
            elif msg.role == "assistant":
                entry = {
                    "version": "v1",
                    "kind": "AssistantMessage",
                    "data": {
                        "message_id": msg_id,
                        "content": [{"kind": "text", "data": msg.content}],
                    },
                }
            else:
                # Skip system messages (already handled above)
                continue

            jsonl_lines.append(json.dumps(entry, ensure_ascii=False))

        # Write JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines))
            if jsonl_lines:
                f.write("\n")

        # Write minimal session JSON
        session_meta = {
            "session_id": session_id,
            "cwd": os.getcwd(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "title": "Expert code analysis session",
            "session_created_reason": "subagent",
            "session_state": {
                "version": "v1",
                "agent_name": "kiro_default",
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session_meta, f, ensure_ascii=False)

        return json_path, jsonl_path

    def _cleanup_session(self, session_id: str) -> None:
        """Remove session files (.json, .jsonl, .lock).

        Args:
            session_id: Session ID to clean up
        """
        for suffix in (".json", ".jsonl", ".lock"):
            path = self.sessions_dir / f"{session_id}{suffix}"
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass  # Best effort

    def _spawn_pty(self, cmd: str):
        """Spawn a process with a PTY.

        Args:
            cmd: Command string to execute

        Returns:
            PTY process object (winpty.PtyProcess or pexpect.spawn)
        """
        if self._pty_backend == "winpty":
            from winpty import PtyProcess
            return PtyProcess.spawn(cmd)
        else:
            import pexpect
            # pexpect.spawn takes command and args separately
            parts = cmd.split(" ", 1)
            if len(parts) == 2:
                return pexpect.spawn(parts[0], parts[1].split(), encoding="utf-8", timeout=None)
            return pexpect.spawn(parts[0], encoding="utf-8", timeout=None)

    def _pty_write(self, proc, text: str) -> None:
        """Write text to PTY stdin.

        Args:
            proc: PTY process object
            text: Text to write
        """
        if self._pty_backend == "winpty":
            proc.write(text)
        else:
            proc.sendline(text.rstrip("\r\n"))

    def _pty_is_alive(self, proc) -> bool:
        """Check if PTY process is still running.

        Args:
            proc: PTY process object

        Returns:
            True if process is alive
        """
        if self._pty_backend == "winpty":
            return proc.isalive()
        else:
            return proc.isalive()

    def _pty_read(self, proc) -> str:
        """Non-blocking read from PTY.

        Args:
            proc: PTY process object

        Returns:
            Available output text, or empty string
        """
        if self._pty_backend == "winpty":
            try:
                return proc.read(4096)
            except Exception:
                return ""
        else:
            try:
                # pexpect non-blocking read
                import pexpect
                proc.expect([pexpect.TIMEOUT, pexpect.EOF], timeout=0.1)
                return proc.before or ""
            except Exception:
                return ""

    def _pty_terminate(self, proc) -> None:
        """Force terminate PTY process.

        Args:
            proc: PTY process object
        """
        if self._pty_backend == "winpty":
            proc.terminate()
        else:
            proc.terminate(force=True)

    def _pty_exit_status(self, proc) -> Optional[int]:
        """Get exit status of PTY process.

        Args:
            proc: PTY process object

        Returns:
            Exit code or None if still running
        """
        if self._pty_backend == "winpty":
            return proc.exitstatus
        else:
            return proc.exitstatus

    def _poll_jsonl_for_response(self, jsonl_path: Path, initial_size: int) -> Optional[str]:
        """Check jsonl file for a new AssistantMessage.

        Args:
            jsonl_path: Path to the session JSONL file
            initial_size: File size before the request was sent

        Returns:
            Response text if found, None otherwise
        """
        if not jsonl_path.exists():
            return None

        current_size = jsonl_path.stat().st_size
        if current_size <= initial_size:
            return None

        with open(jsonl_path, "r", encoding="utf-8") as f:
            f.seek(initial_size)
            new_content = f.read()

        for line in new_content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("kind") == "AssistantMessage":
                    content_items = obj.get("data", {}).get("content", [])
                    for item in content_items:
                        if item.get("kind") == "text":
                            return item.get("data", "")
            except json.JSONDecodeError:
                continue

        return None

    def generate(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> LLMResponse:
        """Generate a complete response using Kiro CLI.

        Args:
            messages: Conversation history as a list of Message objects
            model: Model identifier (passed via --model flag)
            max_tokens: Maximum tokens to generate (not directly supported, ignored)
            temperature: Sampling temperature (not directly supported, ignored)
            system: Optional system prompt (baked into session history)
            debug_category: Category for debug logging (default: "expert")

        Returns:
            LLMResponse containing generated content and metadata

        Raises:
            LLMError: For Kiro CLI errors or timeouts
        """
        session_id = self._create_session_id()

        try:
            # Separate history from current prompt
            history_messages = []
            current_prompt = ""

            if messages:
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].role == "user":
                        current_prompt = messages[i].content
                        history_messages = list(messages[:i])
                        break

            if not current_prompt:
                raise LLMInvalidRequestError("No user message found in messages list")

            # Create session files with history
            json_path, jsonl_path = self._create_session_files(
                session_id, history_messages, system
            )

            # Record initial jsonl size (to detect new content)
            initial_size = jsonl_path.stat().st_size

            # Build command
            cmd = (
                f'{self.cli_path} chat --resume-id {session_id}'
                f' --model {model}'
                f' --trust-tools='
                f' --agent-engine v2'
            )

            # Log request if debug enabled
            request_id = None
            if DebugLogger.is_enabled():
                request_data = {
                    "session_id": session_id,
                    "current_prompt": current_prompt,
                    "history_count": len(history_messages),
                    "system_prompt": system[:200] if system else None,
                    "command": cmd,
                    "model": model,
                }
                request_id = DebugLogger.log_request(
                    "kiro_cli", request_data, category=debug_category
                )

            # Spawn PTY process
            proc = self._spawn_pty(cmd)

            # Immediately write prompt via stdin
            self._pty_write(proc, current_prompt + "\r\n")

            # Poll jsonl for response
            deadline = time.time() + self.response_timeout
            response_text = None
            pty_output = ""  # Collect PTY output for diagnostics

            while time.time() < deadline:
                # Drain PTY output to prevent blocking
                if self._pty_is_alive(proc):
                    chunk = self._pty_read(proc)
                    if chunk:
                        pty_output += chunk

                # Check for response in jsonl
                response_text = self._poll_jsonl_for_response(jsonl_path, initial_size)
                if response_text is not None:
                    break

                # Check if process died unexpectedly
                if not self._pty_is_alive(proc):
                    # Drain any remaining output
                    chunk = self._pty_read(proc)
                    if chunk:
                        pty_output += chunk
                    # One final check of jsonl
                    response_text = self._poll_jsonl_for_response(jsonl_path, initial_size)
                    if response_text is not None:
                        break
                    exit_code = self._pty_exit_status(proc)
                    raise LLMError(
                        f"kiro-cli exited unexpectedly with code {exit_code}. "
                        f"Session: {session_id}\n"
                        f"PTY output: {pty_output.strip()[-500:]}"
                    )

                time.sleep(self.poll_interval)

            # Shutdown the process
            self._shutdown_process(proc)

            # Handle timeout
            if response_text is None:
                raise LLMError(
                    f"Timed out waiting for kiro-cli response after {self.response_timeout}s. "
                    f"Session: {session_id}\n"
                    f"PTY output: {pty_output.strip()[-500:]}"
                )

            # Validate response
            if not response_text.strip():
                raise LLMError(
                    f"kiro-cli returned empty response. Session: {session_id}"
                )

            # Log response if debug enabled
            if DebugLogger.is_enabled():
                DebugLogger.log_response(
                    "kiro_cli",
                    {"content": response_text, "session_id": session_id},
                    request_id,
                    category=debug_category,
                )

            return LLMResponse(
                content=response_text,
                model=model,
                stop_reason="end_turn",
                usage=UsageMetrics(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                ),
            )

        finally:
            # Clean up session files (unless debug is enabled)
            if not DebugLogger.is_enabled():
                self._cleanup_session(session_id)

    def _shutdown_process(self, proc) -> None:
        """Gracefully shut down the kiro-cli process.

        Sends /quit, waits for exit, then force terminates if needed.

        Args:
            proc: PTY process object
        """
        if not self._pty_is_alive(proc):
            return

        # Send /quit
        try:
            self._pty_write(proc, "/quit\r\n")
        except (EOFError, OSError):
            return

        # Wait for graceful exit
        deadline = time.time() + self.shutdown_timeout
        while self._pty_is_alive(proc) and time.time() < deadline:
            time.sleep(0.2)

        # Force terminate if still alive
        if self._pty_is_alive(proc):
            self._pty_terminate(proc)

    async def stream(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from Kiro CLI.

        Since kiro-cli doesn't support true streaming output, this yields the
        complete response as a single chunk once available.

        Args:
            messages: Conversation history as a list of Message objects
            model: Model identifier (passed via --model flag)
            max_tokens: Maximum tokens to generate (ignored)
            temperature: Sampling temperature (ignored)
            system: Optional system prompt (baked into session history)
            debug_category: Category for debug logging (default: "expert")

        Yields:
            StreamChunk with the complete response
        """
        # Use generate() and yield as single chunk
        response = self.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            debug_category=debug_category,
        )

        yield StreamChunk(
            delta=response.content,
            stop_reason=response.stop_reason,
            usage=response.usage,
        )
