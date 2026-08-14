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
import re
import subprocess
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

DEFAULT_URL = "http://127.0.0.1:8179/mcp"  # local knowledge-rag endpoint (task order, 2026-08-14)

DEFAULT_HERMES_BIN = "/Users/alis/.local/bin/hermes"  # task order, 2026-08-14
DEFAULT_HERMES_PROVIDER = "vault-rag-local"
DEFAULT_HERMES_MODEL = "qwen2.5-vl-7b-instruct"
# Isolated Hermes config root: the global HERMES_HOME carries unrelated MCP
# servers that must not be launched from this CLI (task order, 2026-08-14).
DEFAULT_HERMES_HOME = "/Users/alis/.local/share/vetclub-knowledge-rag/hermes-vault-rag"
# The task order requires a bounded timeout; the exact figure is a default
# sized for a local 7B model answering from a few kilobytes of context.
HERMES_TIMEOUT_SECONDS = 300
# Deterministic context caps (task order: bound the prompt size; raised
# 2026-08-14 after the E2E showed 1200 truncated a decisive azotemia matrix).
MAX_FRAGMENT_CHARS = 4000
MAX_CONTEXT_CHARS = 16000
# Full-document evidence windows (task order, 2026-08-14): ~5000 chars per
# document so up to three windows fit the 16000-char context budget.
DOC_WINDOW_CHARS = 5000
WINDOW_STEP_CHARS = 1000  # deterministic scan stride for window selection

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
    "Жёсткие правила:\n"
    "1. Каждый пункт с фактом завершай ссылкой [n] сразу после утверждения "
    "(например: «… повышен [2]»). Утверждение без ссылки на фрагмент запрещено.\n"
    "2. Направления показателей и отрицания переноси дословно из фрагментов: "
    "«повышен», «снижен», «в норме», «не повышен», «не изменяется» нельзя менять "
    "на противоположные или перефразировать с потерей отрицания.\n"
    "3. Никогда не делай вывод, что показатель в норме, снижен или повышен, если "
    "это прямо не написано во фрагментах. Показатель, о котором фрагменты молчат, "
    "перечисли отдельно как «нет данных в базе».\n"
    "4. Если фрагменты не отвечают на вопрос или отвечают лишь частично, прямо "
    "напиши, что данных в базе недостаточно, и не дополняй ответ знаниями вне фрагментов.\n"
    "5. Разделяй, что именно говорят источники (доказательства), и не превращай это "
    "в клинические назначения: это справка по базе знаний, а не рекомендация по лечению "
    "конкретного животного.\n"
    "6. Отвечай по-русски, кратко, пунктами.\n"
)


def _limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"не число: {value!r}") from exc
    if not 1 <= limit <= 10:
        raise argparse.ArgumentTypeError("допустимый диапазон --limit: 1..10")
    return limit


def _tool_payload(result) -> dict:
    """Extract and parse a tool's JSON envelope from an MCP CallToolResult."""
    texts = [item.text for item in (result.content or []) if getattr(item, "type", "") == "text"]
    if getattr(result, "is_error", False):
        raise RuntimeError("; ".join(texts) or "инструмент вернул ошибку без текста")
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


def _document_candidates(item: dict) -> list:
    """Paths to try with get_document for a search result, most specific first."""
    source = str(item.get("source") or "").strip()
    filename = str(item.get("filename") or "").strip()
    category = str(item.get("category") or "").strip()
    candidates = []
    if source:
        candidates.append(source)
    if category and filename:
        candidates.append(f"{category}/{filename}")
    if filename:
        candidates.append(filename)
    return [c for i, c in enumerate(candidates) if c not in candidates[:i]]


async def _fetch_full_document(session, item: dict, cache: dict) -> str:
    """Fetch the full document behind a search result; '' when unavailable."""
    candidates = _document_candidates(item)
    key = "\x00".join(candidates)
    if not candidates:
        return ""
    if key in cache:
        return cache[key]
    content = ""
    for candidate in candidates:
        try:
            payload = _tool_payload(await session.call_tool("get_document", {"filepath": candidate}))
        except Exception:  # noqa: BLE001 — per-document failure keeps the original chunk
            continue
        document = payload.get("document") if payload.get("status") == "success" else None
        if isinstance(document, dict) and str(document.get("content") or "").strip():
            content = str(document["content"])
            break
    cache[key] = content
    return content


async def _search(url: str, query: str, limit: int, method: str) -> dict:
    """Search, then attach full documents to top results in the same MCP session."""
    arguments = {"query": query, "max_results": limit, **_METHOD_PARAMS[method]}
    async with streamable_http_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            payload = _tool_payload(await session.call_tool("search_knowledge", arguments))
            if payload.get("status") == "success":
                cache = {}
                for item in (payload.get("results") or [])[:limit]:
                    if not isinstance(item, dict):
                        continue
                    full = await _fetch_full_document(session, item, cache)
                    if full:
                        item["_full_document"] = full  # private: never printed in Sources
    return payload


def _query_stems(query: str) -> list:
    """Deterministic crude stems: lowercase word tokens >=3 chars, first 6 chars."""
    stems = []
    for token in re.findall(r"\w+", query.lower()):
        if len(token) < 3:
            continue
        stem = token[:6]
        if stem not in stems:
            stems.append(stem)
    return stems


def _select_window(full_text: str, stems: list, size: int = DOC_WINDOW_CHARS) -> str:
    """Pick the query-densest window of the document; '' when no stem matches."""
    if not full_text:
        return ""
    if len(full_text) <= size:
        return full_text
    lowered = full_text.lower()
    starts = list(range(0, len(full_text) - size + 1, WINDOW_STEP_CHARS))
    if starts[-1] != len(full_text) - size:
        starts.append(len(full_text) - size)
    best_start, best_score = 0, (0, 0)
    for start in starts:
        window = lowered[start:start + size]
        distinct, weighted = 0, 0
        for stem in stems:
            count = window.count(stem)
            if count:
                distinct += 1
                weighted += len(stem) * count
        score = (distinct, weighted)
        if score > best_score:  # strict: earliest window wins ties (deterministic)
            best_score, best_start = score, start
    if best_score == (0, 0):
        return ""
    return full_text[best_start:best_start + size]


def _fragment_label(item: dict) -> str:
    return str(item.get("filename") or item.get("source") or "<файл не указан>")


def _fragment_evidence(item: dict, stems: list) -> str:
    """Evidence text for one result: query-dense full-doc window, else the chunk."""
    window = _select_window(str(item.get("_full_document") or ""), stems)
    if window:
        return " ".join(window.split())
    content = " ".join(str(item.get("content") or "").split())
    if len(content) > MAX_FRAGMENT_CHARS:
        content = content[:MAX_FRAGMENT_CHARS].rstrip() + "…"
    return content


def _build_prompt(query: str, results: list) -> tuple:
    """Build the bounded Russian prompt; return (prompt, fragments actually included)."""
    stems = _query_stems(query)
    used, blocks, total = [], [], 0
    for item in results:
        block = f"[{len(used) + 1}] {_fragment_label(item)}: {_fragment_evidence(item, stems)}"
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
    hermes_home = os.environ.get("VAULT_RAG_HERMES_HOME") or DEFAULT_HERMES_HOME
    # Never inherit the ambient HERMES_HOME: it points at the user's global
    # Hermes config whose MCP servers would be launched on agent startup.
    env = {**os.environ, "HERMES_HOME": hermes_home}
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
        # context_engine is a valid built-in toolset that enables no external
        # tools here; with --max-turns 1 the model answers in a single turn.
        "--toolsets", "context_engine",
        "--max-turns", "1",
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=HERMES_TIMEOUT_SECONDS,
            env=env,
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
            "VAULT_RAG_HERMES_BIN, VAULT_RAG_HERMES_PROVIDER, VAULT_RAG_HERMES_MODEL, VAULT_RAG_HERMES_HOME."
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
