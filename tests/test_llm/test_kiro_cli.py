"""Tests for KiroCliLLM (kiro-cli v3 headless provider)."""

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from expert_among_us.llm.base import LLMError, LLMInvalidRequestError, Message
from expert_among_us.llm.kiro_cli import AGENT_NAME, KiroCliLLM


def _events(*events: dict) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode("utf-8")


def _chunk(text: str) -> dict:
    return {
        "type": "sessionUpdate",
        "data": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}},
    }


FINISHED = {"type": "runFinished", "data": {"status": "success", "stopReason": "end_turn", "finalText": "x"}}


@pytest.fixture
def llm(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    with patch("expert_among_us.llm.kiro_cli.shutil.which", return_value="/bin/kiro-cli"):
        yield KiroCliLLM(workspace_dir=tmp_path / "ws")


def _completed(stdout: bytes, returncode: int = 0, stderr: bytes = b""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestInit:
    def test_missing_cli_raises(self, tmp_path):
        with patch("expert_among_us.llm.kiro_cli.shutil.which", return_value=None):
            with pytest.raises(LLMError, match="kiro-cli not found"):
                KiroCliLLM(workspace_dir=tmp_path)

    def test_writes_toolless_agent(self, llm):
        config = json.loads((llm.workspace_dir / ".kiro" / "agents" / f"{AGENT_NAME}.json").read_text())
        assert config["tools"] == []
        assert config["includeMcpJson"] is False

    def test_sessions_dir_matches_kiro_workspace_key(self, llm):
        import hashlib
        key = hashlib.sha256(llm.workspace_dir.as_posix().lower().encode()).hexdigest()[:16]
        assert llm.sessions_dir.name == key
        assert llm.sessions_dir.parent.name == "sessions"


class TestGenerate:
    def test_history_and_command(self, llm):
        captured = {}

        def fake_run(cmd, input, capture_output, cwd, timeout):
            sid = cmd[cmd.index("--resume-id") + 1]
            session_dir = llm.sessions_dir / sid
            captured["meta"] = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            captured["lines"] = [
                json.loads(line)
                for line in (session_dir / "messages.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            captured.update(cmd=cmd, input=input, cwd=cwd)
            return _completed(_events(_chunk("Te"), _chunk("al."), FINISHED))

        messages = [
            Message(role="user", content="color is teal"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="what color?"),
        ]
        with patch("expert_among_us.llm.kiro_cli.subprocess.run", side_effect=fake_run):
            response = llm.generate(messages, model="claude-haiku-4.5", system="be terse")

        assert response.content == "Teal."
        assert response.stop_reason == "end_turn"
        assert captured["input"] == "what color?".encode("utf-8")
        assert captured["cwd"] == llm.workspace_dir
        cmd = captured["cmd"]
        for flag in ("--no-interactive", "--output-format", "stream-json", "--agent", AGENT_NAME, "--trust-tools="):
            assert flag in cmd
        assert cmd[cmd.index("--agent-engine") + 1] == "v3"
        assert [(e["payload"]["type"], e["payload"]["content"]) for e in captured["lines"]] == [
            ("user", "be terse"),
            ("assistant", "Understood. I will follow these instructions."),
            ("user", "color is teal"),
            ("assistant", "ok"),
        ]
        assert captured["meta"]["memoryConfig"]["mode"] == "off"
        assert list(llm.sessions_dir.iterdir()) == []  # cleaned up

    def test_falls_back_to_final_text(self, llm):
        with patch("expert_among_us.llm.kiro_cli.subprocess.run", return_value=_completed(_events(FINISHED))):
            assert llm.generate([Message(role="user", content="hi")], model="m").content == "x"

    def test_run_error(self, llm):
        error = {"type": "runError", "data": {"stage": "prompt", "message": "model not available"}}
        with patch("expert_among_us.llm.kiro_cli.subprocess.run", return_value=_completed(_events(error), 1)):
            with pytest.raises(LLMError, match="model not available"):
                llm.generate([Message(role="user", content="hi")], model="bad")
        assert list(llm.sessions_dir.iterdir()) == []

    def test_no_result(self, llm):
        with patch("expert_among_us.llm.kiro_cli.subprocess.run", return_value=_completed(b"", 2, b"boom")):
            with pytest.raises(LLMError, match=r"(?s)code 2.*boom"):
                llm.generate([Message(role="user", content="hi")], model="m")

    def test_timeout(self, llm):
        with patch("expert_among_us.llm.kiro_cli.subprocess.run", side_effect=subprocess.TimeoutExpired("kiro-cli", 1)):
            with pytest.raises(LLMError, match="Timed out"):
                llm.generate([Message(role="user", content="hi")], model="m")
        assert list(llm.sessions_dir.iterdir()) == []

    def test_requires_user_message(self, llm):
        with pytest.raises(LLMInvalidRequestError):
            llm.generate([Message(role="assistant", content="hi")], model="m")


class _FakeStdout:
    def __init__(self, data: bytes):
        self._lines = data.splitlines(keepends=True)

    async def readline(self):
        return self._lines.pop(0) if self._lines else b""


def _fake_process(stdout: bytes, returncode: int = 0, stderr: bytes = b""):
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.drain = MagicMock(return_value=asyncio.sleep(0))
    proc.stdout = _FakeStdout(stdout)
    proc.stderr = MagicMock()

    async def read():
        return stderr

    async def wait():
        proc.returncode = returncode
        return returncode

    proc.stderr.read = read
    proc.wait = wait
    proc.returncode = None
    return proc


async def _collect(llm, messages):
    return [c async for c in llm.stream(messages, model="m")]


class TestStream:
    def test_yields_chunks_then_stop(self, llm):
        proc = _fake_process(_events(_chunk("a"), _chunk("b"), FINISHED))

        async def fake_exec(*cmd, **kwargs):
            return proc

        with patch("expert_among_us.llm.kiro_cli.asyncio.create_subprocess_exec", side_effect=fake_exec):
            chunks = asyncio.run(_collect(llm, [Message(role="user", content="hi")]))

        assert [c.delta for c in chunks] == ["a", "b", ""]
        assert chunks[-1].stop_reason == "end_turn"
        proc.stdin.write.assert_called_once_with(b"hi")
        assert list(llm.sessions_dir.iterdir()) == []

    def test_run_error(self, llm):
        error = {"type": "runError", "data": {"stage": "init", "message": "bad session"}}

        async def fake_exec(*cmd, **kwargs):
            return _fake_process(_events(error), 1)

        with patch("expert_among_us.llm.kiro_cli.asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(LLMError, match="bad session"):
                asyncio.run(_collect(llm, [Message(role="user", content="hi")]))
        assert list(llm.sessions_dir.iterdir()) == []
