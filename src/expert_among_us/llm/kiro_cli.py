"""Kiro CLI LLM provider (v3 engine, headless).

Each call fabricates a v3 session containing the conversation history, then runs
``kiro-cli chat --no-interactive --resume-id <id> --output-format stream-json``
with the current prompt on stdin. Replies are parsed from the JSON Lines event
stream on stdout, and the session is deleted afterwards.

kiro-cli runs from a private workspace directory with a tool-less agent, so the
model cannot touch the filesystem and the caller's workspace config is not loaded.
"""

import asyncio
import hashlib
import json
import platform
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, List, Optional, Tuple

from .base import (
    LLMError,
    LLMInvalidRequestError,
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    UsageMetrics,
)
from ..utils.debug import DebugLogger


AGENT_NAME = "expert-among-us"

# Tool-less agent: the expert only answers from the supplied context.
AGENT_CONFIG = {
    "name": AGENT_NAME,
    "description": "Tool-less agent used by expert-among-us",
    "prompt": None,
    "mcpServers": {},
    "tools": [],
    "allowedTools": [],
    "resources": [],
    "hooks": {},
    "includeMcpJson": False,
}

_NO_USAGE = UsageMetrics(input_tokens=0, output_tokens=0, total_tokens=0)


class KiroCliLLM(LLMProvider):
    """LLM provider that drives kiro-cli's v3 engine in headless mode."""

    DEFAULT_CLI_PATH = "kiro-cli"

    def __init__(
        self,
        cli_path: Optional[str] = None,
        workspace_dir: Optional[Path] = None,
        response_timeout: float = 300.0,
    ):
        """Initialize Kiro CLI provider.

        Args:
            cli_path: kiro-cli executable name or path (default: "kiro-cli" on PATH)
            workspace_dir: Private working directory for kiro-cli
                (default: ~/.expert-among-us/kiro-cli)
            response_timeout: Max seconds to wait for a reply (default: 300)

        Raises:
            LLMError: If kiro-cli is not found
        """
        cli_path = cli_path or self.DEFAULT_CLI_PATH
        full_path = shutil.which(cli_path)
        if not full_path:
            raise LLMError(
                f"kiro-cli not found at '{cli_path}'. "
                "Please ensure kiro-cli is installed and on your PATH."
            )
        # shutil.which may return an uppercase .EXE on Windows, which breaks
        # case-sensitive Toolbox shims.
        if platform.system() == "Windows":
            full_path = full_path.lower()
        self.cli_path = full_path
        self.response_timeout = response_timeout

        self.workspace_dir = Path(
            workspace_dir or Path.home() / ".expert-among-us" / "kiro-cli"
        ).resolve()
        self._write_agent_config()

        # kiro-cli finds sessions by ID in any workspace folder, but files them
        # under sha256(lowercased forward-slash cwd)[:16]; match that layout.
        workspace_key = hashlib.sha256(
            self.workspace_dir.as_posix().lower().encode("utf-8")
        ).hexdigest()[:16]
        kiro_home = Path.home() / ".kiro"
        self.sessions_dir = kiro_home / "sessions" / workspace_key
        self.leases_dir = kiro_home / "session-leases"

    def _write_agent_config(self) -> None:
        """Write the tool-less agent config into the private workspace."""
        agents_dir = self.workspace_dir / ".kiro" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        (agents_dir / f"{AGENT_NAME}.json").write_text(
            json.dumps(AGENT_CONFIG, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Session files
    # ------------------------------------------------------------------

    @staticmethod
    def _split_prompt(messages: List[Message]) -> Tuple[List[Message], str]:
        """Split messages into (history, current prompt) at the last user message."""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                return list(messages[:i]), messages[i].content
        raise LLMInvalidRequestError("No user message found in messages list")

    def _create_session(
        self, history: List[Message], system: Optional[str]
    ) -> Tuple[str, Path]:
        """Write a v3 session (session.json + messages.jsonl) holding the history.

        The system prompt is injected as an opening user/assistant exchange,
        because kiro-cli keeps its own system prompt.

        Returns:
            Tuple of (session_id, session_dir)
        """
        session_id = f"sess_{uuid.uuid4()}"
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True)

        turns: List[Message] = []
        if system:
            turns.append(Message(role="user", content=system))
            turns.append(Message(role="assistant", content="Understood. I will follow these instructions."))
        turns.extend(m for m in history if m.role in ("user", "assistant"))

        now = datetime.now(timezone.utc)

        def iso(t: datetime) -> str:
            return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")

        workspace = str(self.workspace_dir)
        metadata = {
            "schemaVersion": "1.0.0",
            "dataModelVersion": 1,
            "id": session_id,
            "title": "Expert code analysis session",
            "agentMode": "vibe",
            "status": "idle",
            "workspacePaths": [workspace],
            "rootPaths": [workspace],
            "createdAt": iso(now),
            "lastModifiedAt": iso(now),
            # Keep the user's Kiro memories out of expert prompts, and don't
            # write new memories from these sessions.
            "memoryConfig": {"mode": "off", "reflection": False},
            "memoryConfigSource": "explicit",
        }
        (session_dir / "session.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

        start = now - timedelta(milliseconds=len(turns))
        lines = [
            json.dumps(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": iso(start + timedelta(milliseconds=i)),
                    "payload": {"type": msg.role, "content": msg.content},
                },
                ensure_ascii=False,
            )
            for i, msg in enumerate(turns)
        ]
        (session_dir / "messages.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
        return session_id, session_dir

    def _cleanup_session(self, session_id: str) -> None:
        """Delete the session folder and its lease (kept when debug is enabled)."""
        if DebugLogger.is_enabled():
            return
        shutil.rmtree(self.sessions_dir / session_id, ignore_errors=True)
        try:
            (self.leases_dir / f"{session_id}.lease").unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Command and event stream
    # ------------------------------------------------------------------

    def _build_command(self, session_id: str, model: str) -> List[str]:
        return [
            self.cli_path, "chat",
            "--no-interactive",
            "--agent-engine", "v3",
            "--resume-id", session_id,
            "--agent", AGENT_NAME,
            "--model", model,
            "--trust-tools=",
            "--output-format", "stream-json",
        ]

    @staticmethod
    def _parse_event(line: str) -> Optional[dict]:
        line = line.strip()
        if not line.startswith("{"):
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _text_delta(event: dict) -> Optional[str]:
        """Return reply text from an agent_message_chunk event, else None."""
        if event.get("type") != "sessionUpdate":
            return None
        update = event.get("data", {}).get("update", {})
        if update.get("sessionUpdate") != "agent_message_chunk":
            return None
        content = update.get("content", {})
        if content.get("type") == "text":
            return content.get("text") or None
        return None

    @staticmethod
    def _raise_run_error(event: dict, session_id: str) -> None:
        data = event.get("data", {})
        raise LLMError(
            f"kiro-cli error ({data.get('stage', 'unknown')} stage): "
            f"{data.get('message', 'no message')}. Session: {session_id}"
        )

    @staticmethod
    def _raise_no_result(returncode: Optional[int], stderr: str, session_id: str) -> None:
        raise LLMError(
            f"kiro-cli exited with code {returncode} without a result. "
            f"Session: {session_id}\nstderr: {stderr.strip()[-500:]}"
        )

    # ------------------------------------------------------------------
    # LLMProvider API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> LLMResponse:
        """Generate a complete response. max_tokens and temperature are not supported by kiro-cli."""
        history, prompt = self._split_prompt(messages)
        session_id, _ = self._create_session(history, system)
        cmd = self._build_command(session_id, model)
        request_id = self._log_request(session_id, prompt, history, system, cmd, model, debug_category)

        try:
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt.encode("utf-8"),
                    capture_output=True,
                    cwd=self.workspace_dir,
                    timeout=self.response_timeout,
                )
            except subprocess.TimeoutExpired:
                raise LLMError(
                    f"Timed out waiting for kiro-cli response after {self.response_timeout}s. "
                    f"Session: {session_id}"
                )

            chunks: List[str] = []
            finished: Optional[dict] = None
            for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
                event = self._parse_event(line)
                if event is None:
                    continue
                delta = self._text_delta(event)
                if delta:
                    chunks.append(delta)
                elif event.get("type") == "runError":
                    self._raise_run_error(event, session_id)
                elif event.get("type") == "runFinished":
                    finished = event.get("data", {})

            if finished is None:
                self._raise_no_result(
                    proc.returncode, proc.stderr.decode("utf-8", errors="replace"), session_id
                )

            content = "".join(chunks) or finished.get("finalText") or ""
            if not content.strip():
                raise LLMError(f"kiro-cli returned empty response. Session: {session_id}")

            self._log_response(session_id, content, request_id, debug_category)
            return LLMResponse(
                content=content,
                model=model,
                stop_reason=finished.get("stopReason") or "end_turn",
                usage=_NO_USAGE,
            )
        finally:
            self._cleanup_session(session_id)

    async def stream(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        debug_category: str = "expert",
    ) -> AsyncIterator[StreamChunk]:
        """Stream reply text as kiro-cli emits it. max_tokens and temperature are not supported."""
        history, prompt = self._split_prompt(messages)
        session_id, _ = self._create_session(history, system)
        cmd = self._build_command(session_id, model)
        request_id = self._log_request(session_id, prompt, history, system, cmd, model, debug_category)

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
                limit=16 * 1024 * 1024,  # single events can be large
            )
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.response_timeout
            chunks: List[str] = []
            finished: Optional[dict] = None

            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                if not raw:
                    break
                event = self._parse_event(raw.decode("utf-8", errors="replace"))
                if event is None:
                    continue
                delta = self._text_delta(event)
                if delta:
                    chunks.append(delta)
                    yield StreamChunk(delta=delta)
                elif event.get("type") == "runError":
                    self._raise_run_error(event, session_id)
                elif event.get("type") == "runFinished":
                    finished = event.get("data", {})
                    break

            if finished is None:
                stderr = (await proc.stderr.read()).decode("utf-8", errors="replace")
                await proc.wait()
                self._raise_no_result(proc.returncode, stderr, session_id)

            if not chunks and finished.get("finalText"):
                chunks.append(finished["finalText"])
                yield StreamChunk(delta=finished["finalText"])
            if not "".join(chunks).strip():
                raise LLMError(f"kiro-cli returned empty response. Session: {session_id}")

            self._log_response(session_id, "".join(chunks), request_id, debug_category)
            yield StreamChunk(
                delta="",
                stop_reason=finished.get("stopReason") or "end_turn",
                usage=_NO_USAGE,
            )
        except asyncio.TimeoutError:
            raise LLMError(
                f"Timed out waiting for kiro-cli response after {self.response_timeout}s. "
                f"Session: {session_id}"
            )
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            self._cleanup_session(session_id)

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_request(session_id, prompt, history, system, cmd, model, category) -> Optional[str]:
        if not DebugLogger.is_enabled():
            return None
        return DebugLogger.log_request(
            "kiro_cli",
            {
                "session_id": session_id,
                "current_prompt": prompt,
                "history_count": len(history),
                "system_prompt": system[:200] if system else None,
                "command": cmd,
                "model": model,
            },
            category=category,
        )

    @staticmethod
    def _log_response(session_id, content, request_id, category) -> None:
        if DebugLogger.is_enabled():
            DebugLogger.log_response(
                "kiro_cli",
                {"content": content, "session_id": session_id},
                request_id,
                category=category,
            )
