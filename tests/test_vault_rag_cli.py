"""Focused vault-rag CLI contract tests (no network, no server, no Hermes).

Proves the terminal pipeline's failure boundary for versioned serving:
when the MCP server answers ``retrieval_blocked`` (or any non-success
status), the CLI reports the server's message, exits nonzero, and NEVER
invokes answer generation (``_generate_answer``) — there is no answer
without verified retrieval.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout

from mcp_server import vault_rag_cli as cli


def _run_main(monkeypatch, payload, query="что такое АЧС"):
    """Run cli.main() with a faked MCP search and a booby-trapped generator."""

    async def _fake_search(url, query, limit, method):
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

    async def _fake_search(url, query, limit, method):
        return payload

    monkeypatch.setattr(cli, "_search", _fake_search)
    # Provide a deterministic generator stub (selection JSON).
    monkeypatch.setattr(
        cli, "_generate_answer", lambda prompt: '{"evidence_ids": ["u0"]}'
    )
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
