"""vault-rag — local terminal Vault-RAG client for the knowledge-rag MCP server.

Retrieves ranked veterinary excerpts from the already-running local MCP
endpoint (``search_knowledge`` over streamable HTTP), then generates a
grounded Russian answer through the local Hermes CLI (inference-only) and
prints the answer followed by a compact numbered Sources section. MCP and
Hermes are internal transports only — the user sees plain terminal output.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

DEFAULT_URL = "http://127.0.0.1:8179/mcp"  # local knowledge-rag endpoint (task order, 2026-08-14)

DEFAULT_HERMES_BIN = "/Users/alis/.local/bin/hermes"  # task order, 2026-08-14
DEFAULT_HERMES_PROVIDER = "vault-rag-local"
DEFAULT_HERMES_MODEL = "qwen2.5-vl-7b-instruct"
# The task order requires a bounded timeout; the exact figure is a default
# sized for a local 7B model answering from a few kilobytes of context.
HERMES_TIMEOUT_SECONDS = 300
# Deterministic context caps (task order: bound the prompt size).
MAX_FRAGMENT_CHARS = 1200
MAX_CONTEXT_CHARS = 9000

# CLI method -> search_knowledge arguments. The tool accepts only
# auto|hybrid|fts5; per its docstring semantic/keyword are hybrid with
# hybrid_alpha 1.0 (semantic-only) / 0.0 (keyword-only).
_METHOD_PARAMS = {
    "auto": {"search_method": "auto"},
    "hybrid": {"search_method": "hybrid"},
    "semantic": {"search_method": "hybrid", "hybrid_alpha": 1.0},
    "keyword": {"search_method": "hybrid", "hybrid_alpha": 0.0},
    "fts5": {"search_method": "fts5"},
}

_PROMPT_HEADER = (
    "Ты — ассистент локальной ветеринарной базы знаний. Ответь на вопрос, "
    "используя ТОЛЬКО приведённые ниже фрагменты.\n"
    "Правила:\n"
    "1. Каждое фактическое утверждение подтверждай ссылкой на фрагмент в виде [1], [2].\n"
    "2. Ничего не выдумывай. Если фрагменты не содержат ответа или его части, "
    "прямо напиши, что данных в базе недостаточно.\n"
    "3. Разделяй, что именно говорят источники (доказательства), и не превращай это "
    "в клинические назначения: это справка по базе знаний, а не рекомендация по лечению "
    "конкретного животного.\n"
    "4. Отвечай по-русски, кратко и по существу.\n"
)


def _limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"не число: {value!r}") from exc
    if not 1 <= limit <= 10:
        raise argparse.ArgumentTypeError("допустимый диапазон --limit: 1..10")
    return limit


async def _search(url: str, query: str, limit: int, method: str) -> dict:
    """Call search_knowledge over streamable HTTP and return its parsed JSON payload."""
    arguments = {"query": query, "max_results": limit, **_METHOD_PARAMS[method]}
    async with streamable_http_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("search_knowledge", arguments)

    texts = [item.text for item in (result.content or []) if getattr(item, "type", "") == "text"]
    if getattr(result, "is_error", False):
        raise RuntimeError("; ".join(texts) or "инструмент search_knowledge вернул ошибку без текста")
    if not texts:
        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            return structured
        raise RuntimeError("пустой ответ инструмента: нет текстового содержимого")
    try:
        payload = json.loads("\n".join(texts))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ответ инструмента не является корректным JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"неожиданная форма ответа инструмента: {type(payload).__name__}")
    return payload


def _fragment_label(item: dict) -> str:
    return str(item.get("filename") or item.get("source") or "<файл не указан>")


def _build_prompt(query: str, results: list) -> tuple:
    """Build the bounded Russian prompt; return (prompt, fragments actually included)."""
    used, blocks, total = [], [], 0
    for item in results:
        content = " ".join(str(item.get("content") or "").split())
        if len(content) > MAX_FRAGMENT_CHARS:
            content = content[:MAX_FRAGMENT_CHARS].rstrip() + "…"
        block = f"[{len(used) + 1}] {_fragment_label(item)}: {content}"
        if used and total + len(block) > MAX_CONTEXT_CHARS:
            break  # deterministic bound: keep the highest-ranked fragments
        used.append(item)
        blocks.append(block)
        total += len(block)
    prompt = f"{_PROMPT_HEADER}\nВопрос: {query}\n\nФрагменты:\n" + "\n\n".join(blocks) + "\n"
    return prompt, used


def _generate_answer(prompt: str) -> str:
    """Run the local Hermes CLI non-interactively and return the answer text."""
    hermes_bin = os.environ.get("VAULT_RAG_HERMES_BIN") or DEFAULT_HERMES_BIN
    provider = os.environ.get("VAULT_RAG_HERMES_PROVIDER") or DEFAULT_HERMES_PROVIDER
    model = os.environ.get("VAULT_RAG_HERMES_MODEL") or DEFAULT_HERMES_MODEL
    # chat -q -Q: non-interactive single query, banner/spinner suppressed,
    # only the final response on stdout (session info goes to stderr).
    # --reasoning/--ignore-rules/--source are honored on the chat path
    # (hermes -z oneshot ignores all three — verified in hermes_cli 0.20.0).
    argv = [
        hermes_bin, "chat",
        "-q", prompt,
        "--quiet",
        "--provider", provider,
        "--model", model,
        "--reasoning", "none",
        "--ignore-rules",
        "--source", "tool",
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=HERMES_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Hermes CLI не найден: {exc.filename or hermes_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Hermes не ответил за {HERMES_TIMEOUT_SECONDS} с") from exc
    if proc.returncode != 0:
        detail = " ".join((proc.stderr or "").split())[-200:] or "нет сообщения об ошибке"
        raise RuntimeError(f"Hermes завершился с кодом {proc.returncode}: {detail}")
    answer = (proc.stdout or "").strip()
    if not answer:
        raise RuntimeError("Hermes вернул пустой ответ")
    return answer


def _render(answer: str, used: list, total_found: int) -> None:
    print(answer)
    print()
    print("Источники:")
    for rank, item in enumerate(used, 1):
        score = item.get("score")
        category = item.get("category") or "—"
        print(f"[{rank}] {_fragment_label(item)} · категория: {category} · релевантность: {score}")
    if len(used) < total_found:
        print(f"\nПримечание: в контекст модели вошли первые {len(used)} из {total_found} найденных фрагментов.")


def main() -> int:
    # mcp_server/__init__.py redirects stdout to stderr for MCP stdio safety;
    # this is a terminal application, so restore the real stdout first.
    package = sys.modules.get("mcp_server")
    sys.stdout = getattr(package, "_original_stdout", None) or sys.__stdout__

    parser = argparse.ArgumentParser(
        prog="vault-rag",
        description=(
            "Готовый ответ по локальной ветеринарной базе знаний: поиск через работающий "
            "knowledge-rag MCP-сервер, генерация — через локальный Hermes CLI."
        ),
        epilog=(
            "Переменные окружения: KNOWLEDGE_RAG_MCP_URL (адрес MCP, иначе " + DEFAULT_URL + "), "
            "VAULT_RAG_HERMES_BIN, VAULT_RAG_HERMES_PROVIDER, VAULT_RAG_HERMES_MODEL."
        ),
    )
    parser.add_argument("query", nargs="+", metavar="ЗАПРОС", help="вопрос (можно несколько слов)")
    parser.add_argument("--limit", type=_limit, default=5, help="число фрагментов, 1..10 (по умолчанию 5)")
    parser.add_argument(
        "--method",
        choices=sorted(_METHOD_PARAMS),
        default="auto",
        help="метод поиска: auto|fts5|hybrid|keyword|semantic (по умолчанию auto)",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        parser.error("запрос не может быть пустым")
    url = os.environ.get("KNOWLEDGE_RAG_MCP_URL") or DEFAULT_URL

    try:
        payload = asyncio.run(_search(url, query, args.limit, args.method))
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 — terminal tool: one concise line, nonzero exit
        print(f"Ошибка обращения к MCP-серверу {url}: {exc}", file=sys.stderr)
        return 2

    status = payload.get("status")
    if status == "no_results":
        print(f"По запросу «{query}» ничего не найдено.", file=sys.stderr)
        return 1
    results = payload.get("results") or []
    if status != "success" or not results:
        message = payload.get("message") or payload.get("error") or f"статус ответа: {status!r}"
        print(f"Сервер вернул ошибку: {message}", file=sys.stderr)
        return 1

    prompt, used = _build_prompt(query, results)
    try:
        answer = _generate_answer(prompt)
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"Ошибка генерации ответа (Hermes): {exc}", file=sys.stderr)
        return 3

    _render(answer, used, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
