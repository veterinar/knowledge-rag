#!/usr/bin/env python3
"""Build a knowledge-rag corpus from a Notion snapshot directory.

Провенанс формата: снимок — Pet-dog/vetpilot ``scripts/notion_export.py``
(``knowledge/notion/*.json`` + ``manifest.json``). Мост однонаправленный:
читает git-снимок, пишет markdown-корпус, в сеть не ходит и в Notion
не пишет (обратной синхронизации нет by design).

Usage:

    build_notion_corpus.py --snapshot-dir DIR --out DIR --source-commit HEX [--check]

Режим по умолчанию — сборка корпуса (атомарно: во временный каталог,
затем перенос в ``--out``). ``--check`` — пересборка во временный каталог,
побайтовое сравнение с ``--out`` и канарейка свежести базы «pravila».

Exit codes: 0 — успех; 2 — ошибка входа (целостность/JSON/аргументы);
3 — расхождение корпуса при --check; 4 — красная канарейка при --check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import cast

# Базы координации VetPilot — не знания, в корпус не входят (владелец
# разделил площадки); попадают в выходной манифест списком "excluded".
EXCLUDED_BASES = {"vetpilot-zadachi", "vetpilot-voprosy"}

# Порог длинного поля для тела markdown: поле короче порога считается
# метаданным и живёт только во frontmatter. Порог — договор контракта
# docs/criteria-notion-corpus-bridge.md (N2), меняется правкой константы.
BODY_MIN_LEN = 80

# Поля, всегда идущие в тело, независимо от длины (контракт N2).
ALWAYS_BODY_FIELDS = ("Установлено", "Открыто")

MANIFEST_NAME = "manifest.json"
OUT_MANIFEST_NAME = "corpus-manifest.json"

REQUIRED_ARG_NAMES = ("--snapshot-dir", "--out", "--source-commit")

# Компонент пути (ключ базы, id записи): только безопасные имена файлов —
# ревью R1 27.08 нашло traversal через id вида "../../evil" (fail-closed).
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def ensure_safe_component(kind: str, value: str) -> str:
    """Fail-closed: компонент пути обязан быть безопасным именем файла."""
    if not value or "/" in value or not SAFE_COMPONENT.match(value):
        print(f"unsafe: {kind} {value!r} не является безопасным именем файла", file=sys.stderr)
        raise SystemExit(2)
    return value


def sha256_file(path: Path) -> str:
    """sha256 байтов файла."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(snapshot_dir: Path) -> dict[str, dict[str, object]]:
    """Прочитать manifest.json; без обёртки "databases" весь объект — карта баз."""
    raw = (snapshot_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{MANIFEST_NAME}: ожидан объект, получен {type(data).__name__}")
    databases = data.get("databases", data)
    if not isinstance(databases, dict):
        raise ValueError(f"{MANIFEST_NAME}: 'databases' должен быть объектом")
    return databases


def load_pages(snapshot_dir: Path) -> dict[str, dict[str, object]]:
    """Карта страниц из "pages" манифеста; {} без ключа (плоский манифест
    ключа pages не несёт — обёртки, в отличие от баз, нет)."""
    raw = (snapshot_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{MANIFEST_NAME}: ожидан объект, получен {type(data).__name__}")
    pages = data.get("pages", {})
    if not isinstance(pages, dict):
        raise ValueError(f"{MANIFEST_NAME}: 'pages' должен быть объектом")
    return pages


def verify_pages(snapshot_dir: Path, pages: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """N1-страницы: unsafe-гейт slug + сверка sha256 до использования (fail-closed)."""
    verified: dict[str, dict[str, object]] = {}
    for slug, meta in pages.items():
        ensure_safe_component("страница", slug)
        if not isinstance(meta, dict) or "sha256" not in meta:
            raise ValueError(f"{MANIFEST_NAME}: страница {slug!r} без sha256")
        if not isinstance(meta["sha256"], str):
            raise ValueError(f"{MANIFEST_NAME}: страница {slug!r}: sha256 обязан быть строкой")
        page_path = snapshot_dir / "pages" / f"{slug}.md"
        actual = sha256_file(page_path) if page_path.is_file() else "<отсутствует>"
        expected = meta["sha256"]
        if actual != expected:
            print(
                f"integrity: страница {slug}: sha256 ожидаем {expected}, факт {actual}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        verified[slug] = meta
    return verified


def verify_snapshot(snapshot_dir: Path, databases: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """N1: сверка sha256 каждой невыключенной базы ДО разбора (fail-closed)."""
    verified: dict[str, dict[str, object]] = {}
    for key, meta in databases.items():
        if key in EXCLUDED_BASES:
            continue
        ensure_safe_component("база", key)
        if not isinstance(meta, dict) or "sha256" not in meta:
            raise ValueError(f"{MANIFEST_NAME}: база {key!r} без sha256")
        base_path = snapshot_dir / f"{key}.json"
        actual = sha256_file(base_path) if base_path.is_file() else "<отсутствует>"
        expected = meta["sha256"]
        if actual != expected:
            print(
                f"integrity: база {key}: sha256 ожидаем {expected}, факт {actual}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        verified[key] = meta
    return verified


def scalar_value(value: object) -> str | None:
    """Скалярное frontmatter-значение; списки строк — через запятую."""
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value)
        return text if text != "" else None
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return ",".join(value) if value else None
    return None


def render_record(
    base_key: str,
    meta: dict[str, object],
    record: dict[str, object],
    source_commit: str,
    snapshot_sha: str,
) -> str | None:
    """Markdown одной записи; None, если записей без текста для тела — skip."""
    lines = ["---"]
    lines.append(f"notion_id: {record.get('id', '')}")
    lines.append(f"database: {base_key}")
    lines.append(f"database_id: {meta.get('database_id', '')}")
    lines.append(f"source_commit: {source_commit}")
    lines.append(f"snapshot_sha256: {snapshot_sha}")
    title = None
    for name in sorted(k for k in record if k != "id"):
        rendered = scalar_value(record[name])
        if rendered is None:
            continue
        lines.append(f"{name}: {rendered}")
        if name == "Название" and title is None:
            title = rendered
    if title is not None:
        lines.append(f"title: {title}")
    lines.append("---")

    body_parts: list[tuple[str, str]] = []
    for name in sorted(record):
        value = record[name]
        if not isinstance(value, str) or value == "":
            continue
        if name in ALWAYS_BODY_FIELDS or len(value) > BODY_MIN_LEN:
            body_parts.append((name, value))
    for name, value in body_parts:
        lines.append("")
        lines.append(f"## {name}")
        lines.append("")
        lines.append(value)
    if not body_parts:
        return None
    return "\n".join(lines) + "\n"


def build_corpus(snapshot_dir: Path, out_dir: Path, source_commit: str) -> dict[str, object]:
    """Сборка корпуса в out_dir (уже временный); возвращает выходной манифест."""
    databases = verify_snapshot(snapshot_dir, load_manifest(snapshot_dir))
    pages_manifest = verify_pages(snapshot_dir, load_pages(snapshot_dir))
    files: dict[str, str] = {}
    per_base: dict[str, dict[str, object]] = {}
    for key, meta in databases.items():
        rows = json.loads((snapshot_dir / f"{key}.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{key}.json: ожидан список записей")
        base_dir = out_dir / key
        base_dir.mkdir(parents=True, exist_ok=True)
        snapshot_sha = cast(str, meta["sha256"])
        files_out = 0
        skipped = 0
        for record in rows:
            if not isinstance(record, dict):
                raise ValueError(f"{key}.json: запись не является объектом")
            rendered = render_record(key, meta, record, source_commit, snapshot_sha)
            if rendered is None:
                skipped += 1
                continue
            rid = ensure_safe_component("id записи", str(record.get("id", "")))
            rel = f"{key}/{rid}.md"
            (out_dir / rel).write_text(rendered, encoding="utf-8")
            files[rel] = sha256_file(out_dir / rel)
            files_out += 1
        per_base[key] = {
            "rows_in": len(rows),
            "files_out": files_out,
            "skipped": skipped,
            "snapshot_sha256": snapshot_sha,
        }
    pages_out: dict[str, dict[str, str]] = {}
    if pages_manifest:
        pages_dir = out_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
    for slug, meta in pages_manifest.items():
        snapshot_sha = cast(str, meta["sha256"])
        body = (snapshot_dir / "pages" / f"{slug}.md").read_bytes()
        # вторая проверка целостности: файл мог измениться между verify и
        # этим чтением — пересчитываем sha по прочитанным байтам (fail-closed)
        actual_sha = hashlib.sha256(body).hexdigest()
        if actual_sha != snapshot_sha:
            print(
                f"integrity: страница {slug}: sha256 ожидаем {snapshot_sha}, факт {actual_sha}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        page_id = ensure_safe_component("page_id страницы", str(meta.get("page_id", "")))
        title = meta.get("title", "")
        if not isinstance(title, str) or not title or "\n" in title or "\r" in title:
            raise ValueError(f"{MANIFEST_NAME}: страница {slug!r}: title обязан быть непустой однострочной строкой")
        front = (
            "---\n"
            f"notion_page_id: {page_id}\n"
            f"page: {slug}\n"
            f"title: {title}\n"
            f"source_commit: {source_commit}\n"
            f"snapshot_sha256: {snapshot_sha}\n"
            "---\n"
        ).encode("utf-8")
        rel = f"pages/{slug}.md"
        (out_dir / rel).write_bytes(front + body)
        files[rel] = sha256_file(out_dir / rel)
        pages_out[slug] = {"snapshot_sha256": snapshot_sha, "file": rel}
    manifest: dict[str, object] = {
        "source_commit": source_commit,
        "excluded": sorted(EXCLUDED_BASES & set(load_manifest(snapshot_dir))),
        "bases": per_base,
        "files": dict(sorted(files.items())),
    }
    # обратная совместимость: ключ "pages" — только когда страницы есть;
    # прежний мост (без страниц) ключ не писал, снимок без страниц обязан
    # давать байт-идентичный манифест
    if pages_out:
        manifest["pages"] = pages_out
    (out_dir / OUT_MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def check_corpus(snapshot_dir: Path, out_dir: Path, source_commit: str) -> None:
    """N5: побайтовое сравнение пересборки с --out + канарейка «pravila»."""
    with tempfile.TemporaryDirectory(prefix="notion-corpus-check-") as tmp:
        fresh = Path(tmp) / "corpus"
        fresh.mkdir()
        manifest = build_corpus(snapshot_dir, fresh, source_commit)
        expected = {p.relative_to(fresh).as_posix() for p in fresh.rglob("*") if p.is_file()}
        actual = (
            {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()}
            if out_dir.is_dir()
            else set()
        )
        mismatch = False
        for rel in sorted(expected - actual):
            print(f"check: отсутствует файл {rel}", file=sys.stderr)
        for rel in sorted(actual - expected):
            print(f"check: лишний файл {rel}", file=sys.stderr)
        for rel in sorted(expected & actual):
            if sha256_file(fresh / rel) != sha256_file(out_dir / rel):
                print(f"check: расхождение содержимого {rel}", file=sys.stderr)
                mismatch = True
    if (expected != actual) or mismatch:
        raise SystemExit(3)
    probe_canary(snapshot_dir, out_dir, manifest, source_commit)


def probe_canary(snapshot_dir: Path, out_dir: Path, manifest: dict[str, object], source_commit: str) -> None:
    """Зонд: свежейшая запись «pravila» обязана существовать с непустым телом."""
    rows = json.loads((snapshot_dir / "pravila.json").read_text(encoding="utf-8"))
    dated = [r for r in rows if isinstance(r.get("Дата"), str) and r["Дата"]]
    record = max(dated, key=lambda r: r["Дата"]) if dated else rows[0]
    rid = ensure_safe_component("id записи", str(record.get("id", "")))
    rel = f"pravila/{rid}.md"
    path = out_dir / rel
    if not path.is_file():
        print(f"canary: отсутствует {rel}", file=sys.stderr)
        raise SystemExit(4)
    text = path.read_text(encoding="utf-8")
    body = text.split("---", 2)[-1] if text.count("---") >= 2 else ""
    if not body.strip():
        print(f"canary: пустое тело {rel}", file=sys.stderr)
        raise SystemExit(4)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка knowledge-rag корпуса из снимка Notion (см. докстринг).")
    parser.add_argument(
        "--snapshot-dir", required=True, type=Path, help="каталог со снимком: manifest.json + <база>.json"
    )
    parser.add_argument("--out", required=True, type=Path, help="каталог корпуса (целевой или проверяемый при --check)")
    parser.add_argument("--source-commit", required=True, help="hex sha git-коммита снимка (провенанс)")
    parser.add_argument("--check", action="store_true", help="не писать: пересобрать и сравнить с --out")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    snapshot_dir: Path = args.snapshot_dir
    if not snapshot_dir.is_dir():
        print(f"ошибка: --snapshot-dir не каталог: {snapshot_dir}", file=sys.stderr)
        return 2
    try:
        if args.check:
            check_corpus(snapshot_dir, args.out, args.source_commit)
            print("check: OK — корпус совпадает, канарейка зелёная")
            return 0
        with tempfile.TemporaryDirectory(prefix="notion-corpus-build-") as tmp:
            staging = Path(tmp) / "corpus"
            staging.mkdir()
            manifest = build_corpus(snapshot_dir, staging, args.source_commit)
            if args.out.exists():
                shutil.rmtree(args.out)
            shutil.move(str(staging), str(args.out))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — понятная строка вместо трейсбека
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2
    bases = cast(dict[str, dict[str, object]], manifest["bases"])
    total = sum(cast(int, b["files_out"]) for b in bases.values())
    skipped = sum(cast(int, b["skipped"]) for b in bases.values())
    pages_count = len(cast(dict[str, object], manifest.get("pages", {})))
    print(
        f"corpus: {total} файлов, пропущено записей: {skipped}, "
        f"баз: {len(bases)}, исключено: {len(cast(list[str], manifest['excluded']))}, "
        f"страниц: {pages_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
