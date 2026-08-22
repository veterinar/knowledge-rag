"""Focused vault-rag CLI contract tests (no network, no server, no Hermes).

Proves the terminal pipeline's failure boundary for versioned serving:
when the MCP server answers ``retrieval_blocked`` (or any non-success
status), the CLI reports the server's message, exits nonzero, and NEVER
invokes answer generation (``_generate_answer``) — there is no answer
without verified retrieval.

Also proves the bearer-auth transport contract: with a token the CLI
builds the exact Authorization header over streamable HTTP reusing one
async client; without a token it stays on the plain URL client.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from mcp_server import vault_rag_cli as cli

# Distinctive fake bytes; never a real credential.
_FAKE_TOKEN = "fake-cli-owner-only-token-0123456789abcdef"
_QUERY = "что такое АЧС"
_URL = "http://vault-rag.invalid/mcp"


def _run_main(monkeypatch, payload, query=_QUERY):
    """Run cli.main() with a faked MCP search and a booby-trapped generator."""

    async def _fake_search(url, query, limit, method, token=""):
        return payload

    monkeypatch.setattr(cli, "_search", _fake_search)

    def _boom(prompt):  # pragma: no cover - must never be reached
        raise AssertionError("_generate_answer must not run on blocked retrieval")

    monkeypatch.setattr(cli, "_generate_answer", _boom)
    monkeypatch.setattr(sys, "argv", ["vault-rag", query])

    out, err = io.StringIO(), io.StringIO()
    # cli.main() restores stdout from the PACKAGE's _original_stdout
    # (mcp_server/__init__.py redirects sys.stdout to stderr for MCP stdio
    # safety); point that attribute at the capture buffer so the controlled
    # terminal stream is observable. Production output policy is unchanged.
    monkeypatch.setattr(sys.modules["mcp_server"], "_original_stdout", out)
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main()
    return code, out.getvalue(), err.getvalue()


def test_retrieval_blocked_stops_before_answer_generation(monkeypatch):
    payload = {
        "status": "error",
        "error": "retrieval_blocked",
        "message": "retrieval blocked: the pinned generation is stale (restart required)",
        "restart_required": True,
    }
    code, out, err = _run_main(monkeypatch, payload)
    assert code == 1
    assert "retrieval_blocked" in err
    # The safe server message is reported verbatim after the stable code.
    assert "the pinned generation is stale (restart required)" in err
    # No answer text, no source rendering: generation never produced output.
    assert "Ответ по источникам" not in out
    assert "Источники" not in out


def test_non_success_status_never_generates(monkeypatch):
    payload = {"status": "no_results", "results": []}
    code, out, err = _run_main(monkeypatch, payload)
    assert code == 1
    assert "ничего не найдено" in err
    assert "Ответ по источникам" not in out


def test_success_with_results_is_the_only_generation_path(monkeypatch):
    """Sanity: the guard does not over-block a genuine success payload."""
    payload = {
        "status": "success",
        "results": [
            {
                "content": "АЧС — вирусная болезнь свиней.",
                "source": "asf.md",
                "filename": "asf.md",
                "category": "diseases",
                "score": 0.9,
            }
        ],
    }

    async def _fake_search(url, query, limit, method, token=""):
        return payload

    monkeypatch.setattr(cli, "_search", _fake_search)
    # Provide a deterministic generator stub (selection JSON).
    monkeypatch.setattr(cli, "_generate_answer", lambda prompt: '{"evidence_ids": ["u0"]}')
    monkeypatch.setattr(sys, "argv", ["vault-rag", "что такое АЧС"])
    out, err = io.StringIO(), io.StringIO()
    # Same controlled-terminal capture as _run_main: cli.main() resets
    # sys.stdout from the package's _original_stdout.
    monkeypatch.setattr(sys.modules["mcp_server"], "_original_stdout", out)
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main()
    assert code == 0
    # The answer is rendered on the CONTROLLED terminal stream captured via
    # the package's _original_stdout (out.getvalue(), not redirect_stdout's
    # still-active buffer read after main() restored sys.stdout).
    assert "Ответ по источникам" in out.getvalue()


# ── Bearer-auth transport contract (docs/acceptance/bearer-auth-runtime.v1.md) ──


class _AsyncCM:
    """Async context manager that counts __aexit__ runs."""

    def __init__(self):
        self.exited = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited += 1
        return False


class _FakeSession(_AsyncCM):
    """Session fake whose explicit call_tool returns a text result."""

    def __init__(self, payload):
        super().__init__()
        self.result = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            is_error=False,
        )

    async def call_tool(self, *args, **kwargs):
        return self.result


def test_search_with_token_sends_exact_bearer_and_reuses_async_client(monkeypatch):
    """With a token: exact Authorization bearer header, streamable transport
    over one async http client, Client(transport, cache=None), token never in
    the URL, and the HTTP async context is exited exactly once."""
    recorder = {}
    payload = {"status": "success", "results": []}

    http_cm = _AsyncCM()

    def fake_create_mcp_http_client(headers=None):
        recorder["http"] = {"headers": headers, "cm": http_cm}
        return http_cm

    transport_sentinel = object()

    def fake_streamable_http_client(url, http_client=None):
        recorder["streamable"] = {"url": url, "http_client": http_client}
        return transport_sentinel

    session = _FakeSession(payload)

    def fake_client(*args, **kwargs):
        recorder["client"] = {"args": args, "kwargs": kwargs}
        return session

    monkeypatch.setattr(cli, "create_mcp_http_client", fake_create_mcp_http_client)
    monkeypatch.setattr(cli, "streamable_http_client", fake_streamable_http_client)
    monkeypatch.setattr(cli, "Client", fake_client)

    result = asyncio.run(cli._search(_URL, _QUERY, 5, "auto", token=_FAKE_TOKEN))

    assert result == payload
    # The HTTP client factory receives ONLY the exact Authorization header.
    assert recorder["http"]["headers"] == {"Authorization": "Bearer " + _FAKE_TOKEN}
    # The streamable transport reuses one async http client instance.
    assert recorder["streamable"]["http_client"] is not None
    # Client wraps the streamable transport sentinel and disables caching.
    assert recorder["client"]["args"][0] is transport_sentinel
    assert recorder["client"]["kwargs"].get("cache") is None
    # The token must never leak into the URL itself.
    assert _FAKE_TOKEN not in recorder["streamable"]["url"]
    # The HTTP async context was exited exactly once.
    assert http_cm.exited == 1
    assert session.exited == 1


def test_search_without_token_uses_plain_client_only(monkeypatch):
    """Without a token: Client(url) only; auth transport/header constructors
    are never used."""
    recorder = {}
    payload = {"status": "success", "results": []}

    def _forbidden_transport(*args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("streamable_http_client must not be used without a token")

    def _forbidden_http_client(*args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("create_mcp_http_client must not be used without a token")

    session = _FakeSession(payload)

    def fake_client(*args, **kwargs):
        recorder["client"] = {"args": args, "kwargs": kwargs}
        return session

    monkeypatch.setattr(cli, "streamable_http_client", _forbidden_transport)
    monkeypatch.setattr(cli, "create_mcp_http_client", _forbidden_http_client)
    monkeypatch.setattr(cli, "Client", fake_client)

    result = asyncio.run(cli._search(_URL, _QUERY, 5, "auto"))

    assert result == payload
    # Plain URL client, no transport, no cache/auth kwargs.
    assert recorder["client"]["args"] == (_URL,)
    assert recorder["client"]["kwargs"] == {}
    assert "streamable" not in recorder
    assert "http" not in recorder
    assert session.exited == 1


def test_main_invalid_token_file_refuses_sanitized_exit_2(monkeypatch, tmp_path):
    """An unsafe/invalid token file fails closed: exit 2, one fixed stderr
    line, no traceback, no token-file path, no token bytes, no MCP call."""
    distinctive_dir = tmp_path / "distinctive-token-dir-9f3ab1"
    distinctive_dir.mkdir()
    distinctive_path = distinctive_dir / "distinctive-token-file-77cde2"
    distinctive_path.write_text(_FAKE_TOKEN + "\n", encoding="utf-8")
    # Group/world-readable: violates the owner-only contract.
    os.chmod(distinctive_path, 0o644)
    monkeypatch.setenv("KNOWLEDGE_RAG_BEARER_TOKEN_FILE", str(distinctive_path))

    def _forbidden_search(*args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("_search must not run on an invalid token file")

    monkeypatch.setattr(cli, "_search", _forbidden_search)
    monkeypatch.setattr(sys, "argv", ["vault-rag", _QUERY])
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys.modules["mcp_server"], "_original_stdout", out)
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main()
    assert code == 2
    assert "Traceback" not in err
    assert str(distinctive_path) not in out
    assert str(distinctive_path) not in err
    assert _FAKE_TOKEN not in out
    assert _FAKE_TOKEN not in err
    # One concise fixed refusal line, not a raw exception text.
    assert err.getvalue().strip() == "KNOWLEDGE_RAG_BEARER_TOKEN_FILE: файл токена отсутствует, недоступен или недопустим."


def test_main_empty_env_token_file_fails_closed(monkeypatch):
    """An explicitly empty KNOWLEDGE_RAG_BEARER_TOKEN_FILE must not fall back
    to the unauthenticated legacy path: exit 2, fixed stderr line, and no
    _search call (nothing is served or sent)."""
    monkeypatch.setenv("KNOWLEDGE_RAG_BEARER_TOKEN_FILE", "")

    def _forbidden_search(*args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("_search must not run when the token-file env var is empty")

    monkeypatch.setattr(cli, "_search", _forbidden_search)
    monkeypatch.setattr(sys, "argv", ["vault-rag", _QUERY])
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys.modules["mcp_server"], "_original_stdout", out)
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main()
    assert code == 2
    assert "Traceback" not in err
    assert err.getvalue().strip() == "KNOWLEDGE_RAG_BEARER_TOKEN_FILE: файл токена отсутствует, недоступен или недопустим."


def test_main_search_exception_leaks_nothing(monkeypatch):
    """A raising _search fails closed with one fixed line: exit 2, and neither
    the caller-controlled URL nor raw exception text (which a lower layer may
    load with request/header material) reaches stdout or stderr."""
    secret = "fake-cli-owner-only-token-0123456789abcdef"  # distinctive fake; never real
    detail = "secret-bearer-was-9f3ab1-header-detail"

    async def _exploding_search(url, query, limit, method, token=""):
        raise RuntimeError(f"GET {url} failed; Authorization: Bearer {secret} {detail}")

    monkeypatch.setattr(cli, "_search", _exploding_search)
    monkeypatch.setenv("KNOWLEDGE_RAG_MCP_URL", _URL)
    monkeypatch.setattr(sys, "argv", ["vault-rag", _QUERY])
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys.modules["mcp_server"], "_original_stdout", out)
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main()

    assert code == 2
    assert err.getvalue().strip() == "Ошибка обращения к MCP-серверу."
    assert out.getvalue() == ""
    combined = out.getvalue() + err.getvalue()
    assert secret not in combined
    assert _URL not in combined
    assert detail not in combined
    assert "Traceback" not in combined


def test_main_forwards_resolved_token_from_env_token_file(monkeypatch, tmp_path):
    """main() resolves KNOWLEDGE_RAG_BEARER_TOKEN_FILE (0600 file) and passes
    the trimmed token through to _search."""
    token_path = tmp_path / "token"
    token_path.write_text(_FAKE_TOKEN + "\n", encoding="utf-8")
    os.chmod(token_path, 0o600)
    monkeypatch.setenv("KNOWLEDGE_RAG_BEARER_TOKEN_FILE", str(token_path))

    captured = {}

    async def _fake_search(url, query, limit, method, token=""):
        captured["token"] = token
        return {
            "status": "success",
            "results": [
                {
                    "content": "АЧС — вирусная болезнь свиней.",
                    "source": "asf.md",
                    "filename": "asf.md",
                    "category": "diseases",
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(cli, "_search", _fake_search)
    monkeypatch.setattr(cli, "_generate_answer", lambda prompt: '{"evidence_ids": ["u0"]}')
    monkeypatch.setattr(sys, "argv", ["vault-rag", _QUERY])
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys.modules["mcp_server"], "_original_stdout", out)
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main()
    assert code == 0
    # The exact resolved token (LF trimmed) is forwarded to the transport.
    assert captured["token"] == _FAKE_TOKEN
    assert "Ответ по источникам" in out.getvalue()


def test_main_search_exception_prints_only_fixed_line_exit_2(monkeypatch):
    """A failing _search must surface ONLY the fixed refusal: no URL, no
    exception text, no traceback — lower layers may embed credentials."""
    secret = "leaky-bearer-token-4b7e91c0d2"
    detail = "GET http://vault-rag.invalid/mcp failed: header leak"

    async def _fake_search(url, query, limit, method, token=""):
        raise RuntimeError(f"{detail} Authorization: Bearer {secret}")

    monkeypatch.setattr(cli, "_search", _fake_search)
    monkeypatch.setattr(sys, "argv", ["vault-rag", _QUERY])
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys.modules["mcp_server"], "_original_stdout", out)
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main()
    assert code == 2
    assert err.getvalue().strip() == "Ошибка обращения к MCP-серверу."
    assert out.getvalue() == ""
    assert secret not in out.getvalue() and secret not in err.getvalue()
    assert detail not in out.getvalue() and detail not in err.getvalue()
    assert _URL not in out.getvalue() and _URL not in err.getvalue()
    assert "Traceback" not in err.getvalue()
