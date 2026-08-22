"""Focused fast-path tests: same-size rewrite detection, tree add/delete
drift, symlink rejection, walk-error sanitization, plus the REAL full
fingerprint over one complete temporary fixture — every covered filesystem
input class must invalidate it.  The gate-composition proof (real
fingerprint flowing through ``server._versioned_stale_state``) lives in
tests/test_generation_integration.py and reuses this fixture builder."""

import hashlib
import importlib.metadata
import os
import time
import types
from pathlib import Path

import pytest

from mcp_server import config as mcp_config
from mcp_server import freshness
from mcp_server import generations as gens
from mcp_server.freshness import FreshnessProbeError, _probe_file, _walk_tree, capture_freshness_fingerprint


def test_probe_file_same_size_rewrite(tmp_path):
    p = tmp_path / "same.bin"
    p.write_bytes(b"AAAA")
    st = os.stat(p)
    mtime_ns = st.st_mtime_ns
    h1 = hashlib.sha256()
    _probe_file(h1, "file", p, p.name)
    d1 = h1.hexdigest()

    deadline = time.monotonic() + 0.01
    while time.monotonic() < deadline:
        time.sleep(0.001)
    p.write_bytes(b"BBBB")
    os.utime(p, ns=(mtime_ns, mtime_ns))

    h2 = hashlib.sha256()
    _probe_file(h2, "file", p, p.name)
    d2 = h2.hexdigest()

    st1 = os.stat(p)
    assert st1.st_size == st.st_size
    assert st1.st_mtime_ns == mtime_ns
    assert st1.st_ctime_ns != st.st_ctime_ns
    assert d1 != d2


def test_walk_tree_add_delete_and_symlink(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    f = root / "a.txt"
    f.write_bytes(b"a")

    h1 = hashlib.sha256()
    _walk_tree(h1, "tree", root)
    base = h1.hexdigest()

    g = root / "b.txt"
    g.write_bytes(b"b")
    h2 = hashlib.sha256()
    _walk_tree(h2, "tree", root)
    added = h2.hexdigest()
    assert base != added

    g.unlink()
    h3 = hashlib.sha256()
    _walk_tree(h3, "tree", root)
    deleted = h3.hexdigest()
    assert added != deleted

    link = root / "link"
    link.symlink_to(f)
    with pytest.raises(FreshnessProbeError):
        _walk_tree(hashlib.sha256(), "tree", root)


def test_walk_tree_error_sanitized(tmp_path, monkeypatch):
    def fake_walk(top, topdown=True, onerror=None, followlinks=False):
        onerror(PermissionError(13, "Permission denied", "/private/DO_NOT_LEAK"))
        yield (), [], []

    monkeypatch.setattr("mcp_server.freshness.os.walk", fake_walk)
    with pytest.raises(FreshnessProbeError) as exc:
        _walk_tree(hashlib.sha256(), "tree", tmp_path)
    assert "DO_NOT_LEAK" not in str(exc.value)
    assert "/private/DO_NOT_LEAK" not in str(exc.value)


# ============================================================================
# REAL capture_freshness_fingerprint over one complete temporary fixture
# ============================================================================

_GEN_ID = "gen-fingerprint0001"
_DIST_NAME = "fakerag"


class _FakeDistribution:
    """Minimal ``importlib.metadata.Distribution`` stand-in whose RECORD,
    file list, Name metadata and ``locate_file`` all resolve into the
    temporary fixture tree (real files on disk — the probes still stat them)."""

    def __init__(self, name, root, record_rows, metadata_rel):
        self._root = root
        self._record_rows = list(record_rows)
        self._metadata_rel = metadata_rel
        self.metadata = {"Name": name}

    def read_text(self, filename):
        if filename != "RECORD":
            return None
        return "".join(row + ",,\n" for row in self._record_rows)

    def locate_file(self, rel):
        return self._root / str(rel)

    @property
    def files(self):
        return [self._metadata_rel] + list(self._record_rows)


def rewrite_same_size(path) -> None:
    """Rewrite ``path`` with different SAME-SIZE bytes, retrying until the
    filesystem actually reports changed metadata — the invalidation proofs
    below demonstrate metadata drift, never size drift."""
    original = path.read_bytes()
    assert original, "fixture file must not be empty"
    mutated = bytes([original[0] ^ 0x01]) + original[1:]
    before = path.stat()
    deadline = time.monotonic() + 5.0
    while True:
        path.write_bytes(mutated)
        after = path.stat()
        if after.st_size == before.st_size and (
            after.st_mtime_ns != before.st_mtime_ns or after.st_ctime_ns != before.st_ctime_ns
        ):
            return
        if time.monotonic() >= deadline:
            raise AssertionError("same-size rewrite produced no metadata change")
        time.sleep(0.005)


def build_freshness_fixture(monkeypatch, tmp_path):
    """Build ONE complete temporary filesystem fixture accepted by the REAL
    ``capture_freshness_fingerprint`` — no probe function is bypassed.

    Creates a fake cfg plus, all under ``tmp_path``: the current pointer, a
    pinned generation directory with its ``generation.json`` receipt and every
    REQUIRED_ARTIFACTS entry (tree and file kinds), an admitted live corpus,
    a package source tree, ``config.yaml``, a dependency lock, embedding and
    reranker artifact trees, RECORD members and installed METADATA evidence.

    Only module boundaries are monkeypatched: ``freshness.__file__``,
    ``config.BASE_DIR``, the dependency-lock candidates, the
    ``importlib.metadata`` distribution/distributions registry, and the
    distribution-name seam (``gens.RECORD_EVIDENCE_DISTRIBUTIONS``).

    Returns ``(cfg, paths)`` where ``paths`` names every mutation target.
    """
    base = Path(tmp_path)

    # Generation store root: pointer + pinned generation + all artifacts.
    data = base / "data"
    gen_root = data / gens.GENERATIONS_DIRNAME / _GEN_ID
    gen_root.mkdir(parents=True)
    pointer = data / gens.CURRENT_FILENAME
    pointer.write_text(_GEN_ID + "\n")
    receipt = gen_root / gens.RECEIPT_FILENAME
    receipt.write_text('{"schema_version": 3, "generation_id": "%s"}\n' % _GEN_ID)
    sealed_file_artifact = None
    sealed_tree_member = None
    for name, (rel, kind) in sorted(gens.REQUIRED_ARTIFACTS.items()):
        target = gen_root / rel
        payload = ("sealed-" + name + "-payload-A\n").encode()
        if kind == "tree":
            target.mkdir()
            (target / "payload.bin").write_bytes(payload)
            if name == gens.CORPUS_ARTIFACT:
                sealed_tree_member = target / "payload.bin"
        else:
            target.write_bytes(payload)
            if name == gens.FTS_ARTIFACT:
                sealed_file_artifact = target

    # Admitted live corpus.
    live = base / "live"
    live.mkdir()
    (live / "note1.md").write_bytes(b"live corpus note one-A\n")
    (live / "note2.md").write_bytes(b"live corpus note two-A\n")

    # Package source tree (freshness.__file__ seam).
    source_root = base / "pkg" / "mcp_server"
    source_root.mkdir(parents=True)
    (source_root / "freshness.py").write_bytes(b"# fixture freshness module\n")
    source_member = source_root / "server.py"
    source_member.write_bytes(b"# fixture server module\n")

    # Retrieval configuration file (config.BASE_DIR seam).
    config_yaml = base / "config.yaml"
    config_yaml.write_bytes(b"# fixture config payload-A\n")

    # Dependency lock (lock-candidates seam).
    dependency_lock = base / "requirements.lock"
    dependency_lock.write_bytes(b"# fixture lock payload-A\n")

    # Embedding / reranker artifact trees.
    embed_root = base / "embed"
    embed_root.mkdir()
    embed_member = embed_root / "model.onnx"
    embed_member.write_bytes(b"embedding artifact payload-A\n")
    rerank_root = base / "rerank"
    rerank_root.mkdir()
    rerank_member = rerank_root / "model.onnx"
    rerank_member.write_bytes(b"reranker artifact payload-A\n")

    # Installed distribution: RECORD members + METADATA evidence on disk.
    dist_root = base / "dist"
    record_member = dist_root / _DIST_NAME / "files" / "record_member.bin"
    record_member.parent.mkdir(parents=True)
    record_member.write_bytes(b"installed record payload-A\n")
    dist_info = dist_root / (_DIST_NAME + "-1.0.dist-info")
    dist_info.mkdir()
    installed_metadata = dist_info / "METADATA"
    installed_metadata.write_bytes(b"Metadata-Version: 2.1\nName: fakerag\n")
    record_rel = record_member.relative_to(dist_root).as_posix()
    metadata_rel = dist_info.relative_to(dist_root).as_posix() + "/METADATA"
    dist = _FakeDistribution(_DIST_NAME, dist_root, [record_rel], metadata_rel)

    def _distribution(name):
        if str(name).strip().lower() == _DIST_NAME:
            return dist
        raise importlib.metadata.PackageNotFoundError(name)

    fake_importlib = types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            distribution=_distribution,
            distributions=lambda: [dist],
            PackageNotFoundError=importlib.metadata.PackageNotFoundError,
        )
    )

    monkeypatch.setattr(freshness, "__file__", str(source_root / "freshness.py"))
    monkeypatch.setattr(mcp_config, "BASE_DIR", str(base))
    monkeypatch.setattr(gens, "_dependency_lock_candidates", lambda: [str(dependency_lock)])
    monkeypatch.setattr(gens, "RECORD_EVIDENCE_DISTRIBUTIONS", (_DIST_NAME,))
    monkeypatch.setattr(freshness, "importlib", fake_importlib)

    class _FakeCfg:
        """Only the attributes the REAL probes read."""

        def __init__(self):
            self.index_mode = "versioned"
            self.generation_build = False
            self.data_dir = str(data)
            self.active_generation_id = _GEN_ID
            self.source_documents_dir = str(live)
            self.supported_formats = [".md"]
            self.exclude_patterns = []
            self.embedding_artifact_path = str(embed_root)
            self.reranker_local_artifact = str(rerank_root)
            self.embedding_model = "fixture-embedding"
            self.embedding_dim = 384
            self.chunk_size = 512
            self.chunk_overlap = 64
            self.gpu_mode = "auto"
            self.reranker_enabled = False

    paths = {
        "pointer": pointer,
        "receipt": receipt,
        "sealed_file_artifact": sealed_file_artifact,
        "sealed_tree_artifact_member": sealed_tree_member,
        "package_source_member": source_member,
        "config_yaml": config_yaml,
        "dependency_lock": dependency_lock,
        "embedding_artifact_member": embed_member,
        "reranker_artifact_member": rerank_member,
        "record_member": record_member,
        "installed_metadata": installed_metadata,
    }
    assert all(p is not None for p in paths.values())
    return _FakeCfg(), paths


_MUTATION_TARGETS = (
    ("pointer", "pointer"),
    ("receipt", "receipt"),
    ("sealed-file-artifact", "sealed_file_artifact"),
    ("sealed-tree-artifact", "sealed_tree_artifact_member"),
    ("package-source", "package_source_member"),
    ("config-yaml", "config_yaml"),
    ("dependency-lock", "dependency_lock"),
    ("embedding-artifact", "embedding_artifact_member"),
    ("reranker-artifact", "reranker_artifact_member"),
    ("record-member", "record_member"),
    ("installed-metadata", "installed_metadata"),
)


@pytest.mark.parametrize(
    "target",
    [key for _, key in _MUTATION_TARGETS],
    ids=[case_id for case_id, _ in _MUTATION_TARGETS],
)
def test_real_fingerprint_invalidates_on_every_covered_class(monkeypatch, tmp_path, target):
    """The REAL full fingerprint changes after mutating ANY covered input
    class — same-size rewrites, so the proof is metadata invalidation, not
    size drift.  Every case starts from a fresh fixture."""
    cfg, paths = build_freshness_fixture(monkeypatch, tmp_path)
    baseline = capture_freshness_fingerprint(cfg)
    assert len(baseline) == 64
    # Deterministic over an unchanged fixture: pure invalidation bound.
    assert capture_freshness_fingerprint(cfg) == baseline
    rewrite_same_size(paths[target])
    assert capture_freshness_fingerprint(cfg) != baseline
