"""Тесты scripts/build_notion_corpus.py (контракт docs/criteria-notion-corpus-bridge.md).

N1 fail-closed, N2 содержимое, N3 детерминизм, N4 манифест, N5 check
красный/зелёный, исключение баз задач, канарейка. Без сети: только
фикстурный мини-снимок в tmp_path.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_notion_corpus", Path(__file__).resolve().parents[1] / "scripts" / "build_notion_corpus.py"
)
bnc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bnc)

SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_db(snapshot_dir: Path, name: str, rows: list) -> str:
    data = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
    (snapshot_dir / f"{name}.json").write_bytes(data)
    return _sha256_bytes(data)


def make_snapshot(tmp_path: Path) -> Path:
    """Мини-снимок: pravila (2, обе с «Дата»), razbory (2, одна пустая),
    vetpilot-zadachi (1, исключаемая); manifest.json с реальными sha256."""
    snap = tmp_path / "snapshot"
    snap.mkdir()

    long_text = "Установлено" + " правило дня." * 10  # >80 символов
    # свежая запись ПЕРВОЙ: канарейка обязана выбирать по дате, не по
    # позиции (mutation rows[-1] не должна проходить — ревью Devin R1)
    pravila_rows = [
        {
            "id": "pr-fresh",
            "Название": "Свежее правило",
            "Дата": "2026-08-27",
            "Статус": "Действует",
            "Установлено": long_text + " (свежая)",
        },
        {
            "id": "pr-old",
            "Название": "Старое правило",
            "Дата": "2026-01-01",
            "Статус": "Действует",
            "Установлено": long_text + " (старая)",
        },
    ]
    razbory_rows = [
        {
            "id": "rz-1",
            "Название": "Разбор один",
            "Установлено": "Установлено: краткий вывод разбора номер один.",
        },
        {"id": "rz-empty", "Статус": "Черновик"},  # пустая — для skipped
    ]
    zadachi_rows = [{"id": "z-1", "Задача": "скоординировать"}]

    databases = {}
    for name, rows, db_id in (
        ("pravila", pravila_rows, "db-pravila-0001"),
        ("razbory", razbory_rows, "db-razbory-0002"),
        ("vetpilot-zadachi", zadachi_rows, "db-zadachi-0003"),
    ):
        databases[name] = {
            "database_id": db_id,
            "rows": len(rows),
            "sha256": _write_db(snap, name, rows),
        }
    (snap / "manifest.json").write_text(
        json.dumps({"databases": databases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return snap


def _run_build(snap: Path, out: Path):
    return bnc.main(["--snapshot-dir", str(snap), "--out", str(out), "--source-commit", SOURCE_COMMIT])


def _run_check(snap: Path, out: Path):
    return bnc.main(["--snapshot-dir", str(snap), "--out", str(out), "--source-commit", SOURCE_COMMIT, "--check"])


def make_snapshot_with_pages(tmp_path: Path, pages: dict) -> Path:
    """Обёртка над make_snapshot: добавляет "pages" в manifest.json и
    пишет pages/<slug>.md с реальными sha256 (make_snapshot не трогаем)."""
    snap = make_snapshot(tmp_path)
    pages_dir = snap / "pages"
    pages_dir.mkdir()
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    entries = {}
    for slug, meta in pages.items():
        data = meta["content"].encode("utf-8")
        (pages_dir / f"{slug}.md").write_bytes(data)
        entries[slug] = {
            "title": meta.get("title", f"# {slug}"),
            "page_id": meta["page_id"],
            "blocks": meta.get("blocks", 1),
            "sha256": _sha256_bytes(data),
        }
    manifest["pages"] = entries
    (snap / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return snap


PAGE_BODY = "# Руководство клуба\n\nОбщие правила общения.\n"


def _out_manifest(out: Path) -> dict:
    return json.loads((out / "corpus-manifest.json").read_text(encoding="utf-8"))


def test_t1_happy_build_and_content(tmp_path):
    snap = make_snapshot(tmp_path)
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0
    assert (out / "pravila" / "pr-old.md").is_file()
    assert (out / "pravila" / "pr-fresh.md").is_file()
    assert (out / "razbory" / "rz-1.md").is_file()
    text = (out / "pravila" / "pr-fresh.md").read_text(encoding="utf-8")
    assert "notion_id: pr-fresh" in text
    assert "database: pravila" in text
    assert f"source_commit: {SOURCE_COMMIT}" in text
    sha = hashlib.sha256((snap / "pravila.json").read_bytes()).hexdigest()
    assert f"snapshot_sha256: {sha}" in text
    body = text.split("---", 2)[2]
    assert "Установлено" in body
    assert "(свежая)" in body


def test_t2_fail_closed_on_corrupt_input(tmp_path):
    snap = make_snapshot(tmp_path)
    data = bytearray((snap / "pravila.json").read_bytes())
    data[0] = data[0] ^ 0x01  # портим один байт ПОСЛЕ сборки манифеста
    (snap / "pravila.json").write_bytes(bytes(data))
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as ei:
        _run_build(snap, out)
    assert ei.value.code == 2
    assert not out.exists()


def test_t3_excluded_bases(tmp_path):
    snap = make_snapshot(tmp_path)
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0
    assert not (out / "vetpilot-zadachi").exists()
    assert "vetpilot-zadachi" in _out_manifest(out)["excluded"]


def test_t4_determinism(tmp_path):
    snap = make_snapshot(tmp_path)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    assert _run_build(snap, out1) == 0
    assert _run_build(snap, out2) == 0

    def tree(out):
        return {p.relative_to(out).as_posix(): bnc.sha256_file(p) for p in out.rglob("*") if p.is_file()}

    assert set(tree(out1)) == set(tree(out2))
    assert tree(out1) == tree(out2)


def test_t5_output_manifest(tmp_path):
    snap = make_snapshot(tmp_path)
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0
    man = _out_manifest(out)
    assert man["source_commit"] == SOURCE_COMMIT
    assert man["bases"]["pravila"]["rows_in"] == 2
    assert man["bases"]["razbory"]["skipped"] == 1
    for rel, sha in man["files"].items():
        assert rel.endswith(".md")
        assert bnc.sha256_file(out / rel) == sha
    assert "pravila/pr-fresh.md" in man["files"]


def test_t6_check_green(tmp_path):
    snap = make_snapshot(tmp_path)
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0
    assert _run_check(snap, out) == 0


def test_t7_check_red_on_corrupted_output(tmp_path):
    snap = make_snapshot(tmp_path)
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0
    target = out / "razbory" / "rz-1.md"
    data = bytearray(target.read_bytes())
    data[-2] = data[-2] ^ 0x01  # меняем байт тела
    target.write_bytes(bytes(data))
    with pytest.raises(SystemExit) as ei:
        _run_check(snap, out)
    assert ei.value.code == 3


def test_t8_canary_missing_and_empty_body(tmp_path):
    snap = make_snapshot(tmp_path)
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0

    # свежейшая запись pravila отсутствует → расхождение сравнения, код 3
    canary = out / "pravila" / "pr-fresh.md"
    canary.unlink()
    with pytest.raises(SystemExit) as ei:
        _run_check(snap, out)
    assert ei.value.code == 3

    # код 4: файл есть, но тело пустое — прямой вызов зонда
    assert _run_build(snap, out) == 0
    canary.write_text("---\nx: y\n---\n", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        bnc.probe_canary(snap, out, {}, SOURCE_COMMIT)
    assert ei.value.code == 4


@pytest.mark.parametrize("bad_id", ["../../evil-record", "trailing\n"])
def test_t9_traversal_id_fail_closed(tmp_path, bad_id):
    # Devin R1: traversal через id записи. Целостность должна ПРОЙТИ
    # (sha256 в manifest.json пересчитан) — красным обязан стать unsafe-гейт.
    snap = make_snapshot(tmp_path)
    rows = json.loads((snap / "pravila.json").read_text(encoding="utf-8"))
    rows[0]["id"] = bad_id
    new_sha = _write_db(snap, "pravila", rows)
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    manifest["databases"]["pravila"]["sha256"] = new_sha
    (snap / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    out = tmp_path / "out"
    with pytest.raises(SystemExit) as ei:
        _run_build(snap, out)
    assert ei.value.code == 2  # unsafe-гейт, а не integrity
    assert not out.exists()
    assert not list(tmp_path.rglob("evil-record.md"))


def test_t10_snapshot_without_pages_leaves_no_pages_trace(tmp_path):
    # B1: снимок БЕЗ "pages" не оставляет следов pages: ни каталога
    # out/pages, ни ключа "pages" в выходном манифесте (обратная
    # совместимость с корпусами, собранными прежним мостом).
    snap = make_snapshot(tmp_path)
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0
    assert not (out / "pages").exists()
    assert "pages" not in _out_manifest(out)


def test_t11_page_build_content_and_manifest(tmp_path):
    snap = make_snapshot_with_pages(
        tmp_path,
        {"rukovodstvo": {"title": "Руководство клуба", "page_id": "abc123", "content": PAGE_BODY}},
    )
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0
    page = out / "pages" / "rukovodstvo.md"
    assert page.is_file()
    text = page.read_text(encoding="utf-8")
    assert "notion_page_id: abc123" in text
    assert "page: rukovodstvo" in text
    assert "title: Руководство клуба" in text
    assert f"source_commit: {SOURCE_COMMIT}" in text
    snapshot_sha = hashlib.sha256((snap / "pages" / "rukovodstvo.md").read_bytes()).hexdigest()
    assert f"snapshot_sha256: {snapshot_sha}" in text
    # тело после frontmatter — байт-в-байт телу из снимка
    body = text.split("---", 2)[2]
    assert body.lstrip("\n") == PAGE_BODY
    assert body.encode("utf-8") == b"\n" + PAGE_BODY.encode("utf-8")
    man = _out_manifest(out)
    assert "pages/rukovodstvo.md" in man["files"]
    assert man["files"]["pages/rukovodstvo.md"] == hashlib.sha256(page.read_bytes()).hexdigest()
    assert man["pages"]["rukovodstvo"] == {"snapshot_sha256": snapshot_sha, "file": "pages/rukovodstvo.md"}


def test_t12_page_tampered_byte_fail_closed(tmp_path):
    # подмена одного байта pages/<slug>.md при старом манифестном sha → 2
    # ДО записи наружного --out (staging обрезается атомарностью переноса)
    snap = make_snapshot_with_pages(
        tmp_path,
        {"rukovodstvo": {"title": "Руководство клуба", "page_id": "abc123", "content": PAGE_BODY}},
    )
    target = snap / "pages" / "rukovodstvo.md"
    data = bytearray(target.read_bytes())
    data[0] = data[0] ^ 0x01
    target.write_bytes(bytes(data))
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as ei:
        _run_build(snap, out)
    assert ei.value.code == 2
    assert not out.exists()


def test_t13_page_missing_file_fail_closed(tmp_path):
    # манифест обещает страницу, файла нет → SystemExit(2)
    snap = make_snapshot_with_pages(
        tmp_path,
        {"rukovodstvo": {"title": "Руководство клуба", "page_id": "abc123", "content": PAGE_BODY}},
    )
    (snap / "pages" / "rukovodstvo.md").unlink()
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as ei:
        _run_build(snap, out)
    assert ei.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize("bad_slug", ["../../evil-page", "trailing\n"])
def test_t14_page_slug_traversal_fail_closed(tmp_path, bad_slug):
    # unsafe-гейт обязан срабатывать ДО integrity-чтения файла, поэтому файл
    # для этого кейса не пишем вовсе — в манифесте синтаксически честный
    # hex-sha, красным становится именно unsafe-гейт
    snap = make_snapshot(tmp_path)
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    manifest["pages"] = {bad_slug: {"title": "T", "page_id": "hex", "blocks": 1, "sha256": "a" * 64}}
    (snap / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as ei:
        _run_build(snap, out)
    assert ei.value.code == 2  # unsafe-гейт, а не integrity
    assert not out.exists()
    assert not list(tmp_path.rglob("evil-page*"))
    assert not list(tmp_path.rglob("*trailing*"))


def test_t15_check_green_then_red_on_corrupted_page(tmp_path):
    snap = make_snapshot_with_pages(
        tmp_path,
        {"rukovodstvo": {"title": "Руководство клуба", "page_id": "abc123", "content": PAGE_BODY}},
    )
    out = tmp_path / "out"
    assert _run_build(snap, out) == 0
    assert _run_check(snap, out) == 0
    target = out / "pages" / "rukovodstvo.md"
    data = bytearray(target.read_bytes())
    data[-2] = data[-2] ^ 0x01  # портим байт тела корпусного файла
    target.write_bytes(bytes(data))
    with pytest.raises(SystemExit) as ei:
        _run_check(snap, out)
    assert ei.value.code == 3


def test_t16_page_title_newline_fail_closed(tmp_path):
    # title с \n в манифесте — инъекция во frontmatter, rc 2
    snap = make_snapshot_with_pages(
        tmp_path,
        {"rukovodstvo": {"title": "Заголовок\nподделка: true", "page_id": "abc123", "content": PAGE_BODY}},
    )
    out = tmp_path / "out"
    assert _run_build(snap, out) == 2  # ValueError пойман в main → rc 2
    assert not out.exists()
