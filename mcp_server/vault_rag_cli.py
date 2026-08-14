"""vault-rag — local terminal client for the knowledge-rag MCP server.

Calls the already-running local MCP endpoint over streamable HTTP, invokes
the ``search_knowledge`` tool and prints ranked veterinary excerpts with
their source filenames. MCP is an internal transport only — the user sees
plain Russian terminal output.
"""

import argparse
import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

DEFAULT_URL = "http://127.0.0.1:8179/mcp"  # local knowledge-rag endpoint (task order, 2026-08-14)

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


def _render(payload: dict, method: str) -> None:
    results = payload.get("results") or []
    print(f"Запрос: {payload.get('query', '')}")
    print(f"Метод: {method} · результатов: {len(results)}")
    print()
    for rank, item in enumerate(results, 1):
        filename = item.get("filename") or item.get("source") or "<файл не указан>"
        score = item.get("score")
        category = item.get("category") or "—"
        via = item.get("search_method") or "?"
        print(f"{rank}. {filename}")
        print(f"   релевантность: {score} · категория: {category} · путь поиска: {via}")
        content = str(item.get("content") or "").strip()
        for line in content.splitlines():
            print(f"   {line}")
        print()


def main() -> int:
    # mcp_server/__init__.py redirects stdout to stderr for MCP stdio safety;
    # this is a terminal application, so restore the real stdout first.
    package = sys.modules.get("mcp_server")
    sys.stdout = getattr(package, "_original_stdout", None) or sys.__stdout__

    parser = argparse.ArgumentParser(
        prog="vault-rag",
        description="Поиск по локальной ветеринарной базе знаний через работающий knowledge-rag MCP-сервер.",
        epilog="Адрес сервера берётся из переменной окружения KNOWLEDGE_RAG_MCP_URL, иначе " + DEFAULT_URL,
    )
    parser.add_argument("query", nargs="+", metavar="ЗАПРОС", help="текст запроса (можно несколько слов)")
    parser.add_argument("--limit", type=_limit, default=5, help="число результатов, 1..10 (по умолчанию 5)")
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
    if status == "success" and payload.get("results"):
        _render(payload, args.method)
        return 0
    if status == "no_results":
        print(f"По запросу «{query}» ничего не найдено.", file=sys.stderr)
        return 1
    message = payload.get("message") or payload.get("error") or f"статус ответа: {status!r}"
    print(f"Сервер вернул ошибку: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
