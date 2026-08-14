"""vault-rag — local terminal Vault-RAG client for the knowledge-rag MCP server.

Retrieves ranked veterinary excerpts from the already-running local MCP
endpoint (``search_knowledge`` over streamable HTTP), derives concise
natural-language evidence units from the fetched documents, asks the local
Hermes CLI (inference-only) to select the most relevant units by their
opaque IDs, and prints a deterministic Russian answer followed by a compact
numbered Sources section. MCP and Hermes are internal transports only — the
user sees plain terminal output. Модель только выбирает готовые фрагменты
по их идентификаторам, поэтому в ответ не может попасть текст, которого нет
в найденных документах; при непригодном выборе печатаются детерминированно
отобранные самые релевантные фрагменты.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from collections import namedtuple

# mcp 2.0.0's Client is the supported entry point: given a URL it builds the
# streamable-HTTP transport itself and negotiates the protocol era (mode="auto"
# probes server/discover at 2026-07-28, falling back to the initialize
# handshake). The knowledge-rag server serves the 2026-07-28 core, which
# retires that handshake — a hand-rolled ClientSession pinned to
# session.initialize() dies on it, inside the transport's anyio task group, so
# the real cause reaches the terminal only as "unhandled errors in a TaskGroup".
from mcp import Client

DEFAULT_URL = "http://127.0.0.1:8179/mcp"  # local knowledge-rag endpoint (task order, 2026-08-14)

DEFAULT_HERMES_BIN = "/Users/alis/.local/bin/hermes"  # task order, 2026-08-14
DEFAULT_HERMES_PROVIDER = "vault-rag-local"
DEFAULT_HERMES_MODEL = "qwen2.5-vl-7b-instruct"
# Isolated Hermes config root: the global HERMES_HOME carries unrelated MCP
# servers that must not be launched from this CLI (task order, 2026-08-14).
DEFAULT_HERMES_HOME = "/Users/alis/.local/share/vetclub-knowledge-rag/hermes-vault-rag"
# The task order requires a bounded timeout; the exact figure is a default
# sized for a local 7B model selecting from a few kilobytes of context.
HERMES_TIMEOUT_SECONDS = 300
# Deterministic context cap (task order: bound the prompt size).
MAX_CONTEXT_CHARS = 16000
# Evidence-unit bounds (task order, 2026-08-14): concise natural-language
# units derived from fetched documents/chunks. The model only selects among
# them by ID, so these caps also bound the printed answer.
MIN_UNIT_CHARS = 30
MAX_UNIT_CHARS = 500
MAX_UNITS = 12
# The model must pick 1..MAX_SELECTED unit IDs; when its output is unusable,
# the deterministic fallback prints the units that passed the relevance gate,
# under the same bound — never a fixed number of bullets.
MAX_SELECTED = 5

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

# Selection contract (task order, 2026-08-14): the model never writes answer
# text. It only returns the opaque IDs of the evidence units to print, as one
# strict JSON object, so no unsupported model prose can enter the answer.
_Unit = namedtuple("_Unit", "uid rank position score text")


def _selection_prompt(query: str, units: list) -> str:
    lines = [
        "Ты — селектор доказательств для локальной ветеринарной базы знаний.",
        "Ниже вопрос и пронумерованные фрагменты (E1, E2, …) из найденных документов.",
        f"Выбери от 1 до {MAX_SELECTED} фрагментов, которые лучше всего отвечают на вопрос.",
        "Верни СТРОГО один JSON-объект и ничего больше, по образцу:",
        '{"evidence_ids": ["E3", "E7"]}',
        "Правила: только идентификаторы из списка ниже, без повторов, в том порядке,",
        "в котором фрагменты должны идти в ответе; никакого другого текста, пояснений",
        "или Markdown вне JSON.",
        "",
        f"Вопрос: {query}",
        "",
        "Фрагменты:",
    ]
    for unit in units:
        lines.append(f"{unit.uid}: {unit.text}")
    return "\n".join(lines) + "\n"


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


async def _fetch_full_document(client, item: dict, cache: dict) -> str:
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
            payload = _tool_payload(await client.call_tool("get_document", {"filepath": candidate}))
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
    async with Client(url) as client:  # mode="auto": negotiates the server's protocol era
        payload = _tool_payload(await client.call_tool("search_knowledge", arguments))
        if payload.get("status") == "success":
            cache = {}
            for item in (payload.get("results") or [])[:limit]:
                if not isinstance(item, dict):
                    continue
                full = await _fetch_full_document(client, item, cache)
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


def _fragment_label(item: dict) -> str:
    return str(item.get("filename") or item.get("source") or "<файл не указан>")


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if 0 < end < 2000:
            return text[end + 4:]
    return text


def _score_block(block: str, stems: list) -> tuple:
    """(distinct stems matched, length-weighted occurrences, matched stems).

    Only the first two elements order units; the third names *which* query
    terms a block answered, which the relevance gate reads."""
    lowered = block.casefold()
    distinct, weighted, matched = 0, 0, []
    for stem in stems:
        count = lowered.count(stem)
        if count:
            distinct += 1
            weighted += len(stem) * count
            matched.append(stem)
    return (distinct, weighted, frozenset(matched))


# Fenced ``` / ~~~ blocks (code, YAML, mermaid…) are never evidence.
_FENCED_BLOCK_RE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$", re.MULTILINE | re.DOTALL)


def _natural_lines(block: str) -> list:
    """Keep only natural-language lines of a block: drop headings, table rows,
    table rules and stray fence markers; strip list markers so a unit reads
    as plain prose."""
    kept = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "|", "```", "~~~")):
            continue
        if re.fullmatch(r"[|\s:+=_-]+", line):
            continue
        line = re.sub(r"^(?:[-*•>–—]|\d+[.)])\s+", "", line).strip()
        if line:
            kept.append(line)
    return kept


def _evidence_units(query: str, results: list) -> list:
    """Bounded, query-relevant natural-language evidence units derived from
    the fetched documents (search chunks when a document is unavailable).
    Each unit stays bound to its source rank and carries an opaque ID; the
    printed answer is assembled only from these units. Units that fail the
    relevance gate below are never offered to the model, so neither its
    selection nor the fallback can print them."""
    stems = _query_stems(query)
    scored, seen = [], set()
    for rank, item in enumerate(results, 1):
        text = _strip_frontmatter(str(item.get("_full_document") or ""))
        if not text.strip():
            text = str(item.get("content") or "")
        text = _FENCED_BLOCK_RE.sub("", text)
        for position, block in enumerate(re.split(r"\n\s*\n", text)):
            unit_text = " ".join(" ".join(_natural_lines(block)).split())
            if len(unit_text) < MIN_UNIT_CHARS:
                continue
            if len(unit_text) > MAX_UNIT_CHARS:
                unit_text = unit_text[:MAX_UNIT_CHARS].rstrip() + "…"
            key = unit_text.casefold()
            if key in seen:  # the same document can back several results
                continue
            seen.add(key)
            scored.append((_score_block(unit_text, stems), rank, position, unit_text))
    # Deterministic query-relative relevance gate (task order, 2026-08-14): a
    # unit is eligible only while it covers more than half as many distinct
    # query stems as the strongest unit found, so a block with weak incidental
    # overlap cannot be printed beside substantially stronger evidence. The
    # majority shape is the gate itself, not a measured threshold, and it reads
    # only the query's own stems — no subject term is hard-coded. When nothing
    # matches any stem the best coverage is 0, no unit is eligible, and the
    # caller keeps its safe no-evidence answer instead of filling from
    # unrelated blocks.
    #
    # Counting stems alone treats every query word as evidence, so on a query
    # whose subject is one term ("…сказано об азотемии") a block answering only
    # the question's scaffolding ties the best coverage and prints beside real
    # evidence (acceptance 2026-08-14). A unit must therefore also share a stem
    # with the strongest unit — the query terms the retrieved corpus actually
    # answered — which a scaffolding-only match never does. The anchor is one
    # unit taken in the ordering below, never the union of units tied at the
    # top: a scaffolding-only unit tied there would otherwise readmit itself.
    # No stop-word list, subject term or result score enters this decision.
    anchor = min(
        scored,
        key=lambda entry: (-entry[0][0], -entry[0][1], entry[1], entry[2]),
        default=None,
    )
    best_distinct = anchor[0][0] if anchor else 0
    core = anchor[0][2] if anchor else frozenset()
    relevant = [
        entry for entry in scored if 2 * entry[0][0] > best_distinct and entry[0][2] & core
    ]
    relevant.sort(key=lambda entry: (-entry[0][0], -entry[0][1], entry[1], entry[2]))
    chosen, total = [], 0
    for entry in relevant:
        if len(chosen) >= MAX_UNITS or total + len(entry[3]) > MAX_CONTEXT_CHARS:
            break
        chosen.append(entry)
        total += len(entry[3])
    chosen.sort(key=lambda entry: (entry[1], entry[2]))  # document order for the prompt
    return [
        _Unit(f"E{number}", rank, position, score, text)
        for number, (score, rank, position, text) in enumerate(chosen, 1)
    ]


# Optional fence around the model's JSON — tolerated, everything else strict.
_FENCE_WRAP_RE = re.compile(r"^```[\w-]*\s*\n(.*?)\n?\s*```$", re.DOTALL)


def _parse_selection(raw: str, units: list):
    """IDs from a strict {"evidence_ids": [...]} model output, or None.
    A fenced ```json``` wrapper is tolerated; anything else — extra keys,
    prose, unknown/repeated IDs, wrong count — rejects the whole output and
    the deterministic fallback answers instead."""
    text = (raw or "").strip()
    fenced = _FENCE_WRAP_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"evidence_ids"}:
        return None
    ids = payload["evidence_ids"]
    known = {unit.uid for unit in units}
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_SELECTED:
        return None
    # Element types are validated before the duplicate set is built: an
    # unhashable element (a nested list or object) would otherwise raise
    # instead of rejecting the output.
    if not all(isinstance(uid, str) and uid in known for uid in ids):
        return None
    if len(set(ids)) != len(ids):
        return None
    return ids


def _fallback_ids(units: list) -> list:
    """Deterministic choice when the model output is unusable: the strongest
    units that passed the relevance gate, cited in source order. The count
    follows eligibility — MAX_SELECTED only bounds it, and fewer eligible
    units mean fewer bullets."""
    ranked = sorted(units, key=lambda u: (-u.score[0], -u.score[1], u.rank, u.position))
    picked = sorted(ranked[:MAX_SELECTED], key=lambda u: (u.rank, u.position))
    return [unit.uid for unit in picked]


def _compose_answer(ids: list, units: list) -> str:
    """Concise deterministic answer: neutral heading, then one clean bullet
    per selected unit with its [source-rank] citation."""
    by_id = {unit.uid: unit for unit in units}
    lines = ["Ответ по источникам:"]
    for uid in ids:
        unit = by_id[uid]
        lines.append(f"— {unit.text} [{unit.rank}]")
    return "\n".join(lines)


def _generate_answer(prompt: str) -> str:
    """Run the local Hermes CLI non-interactively and return its raw output
    (expected: one strict JSON object with the selected evidence IDs)."""
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

    units = _evidence_units(query, results)
    if not units:
        answer = ("Ответ по источникам:\n"
                  "— Подходящих текстовых фрагментов выделить не удалось; "
                  "см. список источников ниже.")
        _render(answer, results, len(results))
        return 0

    try:
        selection = _generate_answer(_selection_prompt(query, units))
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"Ошибка генерации ответа (Hermes): {exc}", file=sys.stderr)
        return 3

    # The model only selects unit IDs; an unusable selection falls back to
    # the deterministic top-scored units, never to the raw model output.
    ids = _parse_selection(selection, units) or _fallback_ids(units)
    _render(_compose_answer(ids, units), results, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
