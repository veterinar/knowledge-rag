"""Deterministic metadata fingerprint over every filesystem input of the
complete content-bound freshness proof (freshness fast path).

Public API:
    FreshnessProbeError              — sanitized, fixed-message failure
    capture_freshness_fingerprint(cfg) -> str

The fingerprint is an invalidation bound, NOT a content digest: it never
hashes file contents and never embeds absolute paths, secrets, or file
data.  Every covered entry contributes its stable logical label, relative
path, entry type, device, inode, mode, size, ``mtime_ns`` and ``ctime_ns``.
Missing roots, symlinks, special files and unreadable evidence fail closed
with a fixed sanitized message.  Stdlib only.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import stat
import struct
from pathlib import Path
from typing import Any, List, Optional, Tuple

from . import generations as gens

_FP_VERSION = b"knowledge-rag-freshness-fingerprint-v1\n"

_ERR_ROOT = "freshness probe: a required evidence root is missing or unreadable"
_ERR_ENTRY = "freshness probe: an evidence entry is missing, unreadable, a symlink, or a special file"
_ERR_WALK = "freshness probe: evidence tree could not be walked"
_ERR_RECORD = "freshness probe: installed RECORD evidence is unreadable"
_ERR_METADATA = "freshness probe: installed distribution metadata is unreadable"
_ERR_UNEXPECTED = "freshness probe: fingerprint capture failed"

# Safe in-memory identity fields, in addition to gens.RETRIEVAL_CONFIG_FIELDS:
# embedding runtime/dimension/pooling/model/prefix, chunking and reranker.
_EXTRA_CONFIG_FIELDS: Tuple[str, ...] = (
    "embedding_runtime_version",
    "embedding_model",
    "embedding_dim",
    "query_prefix",
    "passage_prefix",
    "embedding_artifact_path",
    "models_cache_dir",
    "gpu_mode",
    "chunk_size",
    "chunk_overlap",
    "reranker_enabled",
    "reranker_model",
    "reranker_local_artifact",
)

# Config fields whose string values are filesystem paths: only these ever
# reduce to basename; every other allowlisted value stays exact.
_PATH_CONFIG_FIELDS = frozenset({"embedding_artifact_path", "reranker_local_artifact", "models_cache_dir"})

# Package-source exclusions, mirroring gens.code_identity exactly.
_SOURCE_EXCLUDED_DIRS = frozenset({"__pycache__"})
_SOURCE_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pyd", ".log", ".swp", ".tmp")


class FreshnessProbeError(RuntimeError):
    """A fixed, sanitized freshness-probe failure (no paths, no secrets)."""


def _fail(message: str = _ERR_ENTRY) -> FreshnessProbeError:
    raise FreshnessProbeError(message)


def _walk_onerror(err: BaseException) -> None:
    # Fail closed on any os.walk error; never surface err or its path.
    raise FreshnessProbeError(_ERR_WALK)


def _entry_type_char(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "f"
    if stat.S_ISDIR(mode):
        return "d"
    return "?"  # callers reject before this for covered entries


def _absorb_entry(h: Any, label: str, rel: str, st: os.stat_result) -> None:
    """Absorb one entry's metadata into the running digest."""
    h.update(label.encode("utf-8"))
    h.update(b"\x00")
    h.update(rel.encode("utf-8"))
    h.update(b"\x00")
    h.update(_entry_type_char(st.st_mode).encode("ascii"))
    h.update(b"\x00")
    h.update(struct.pack(">QQQQQQ", st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    h.update(b"\n")


def _lstat_entry(path: Path) -> os.stat_result:
    try:
        st = os.lstat(path)
    except OSError:
        _fail()
    mode = st.st_mode
    if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        _fail()
    return st


def _probe_file(h: Any, label: str, path: Path, rel: str) -> None:
    st = _lstat_entry(path)
    if not stat.S_ISREG(st.st_mode):
        _fail()
    _absorb_entry(h, label, rel, st)


def _walk_tree(h: Any, label: str, root: Path) -> None:
    """Bind a whole tree: root dir metadata, every entry, sorted child sets."""
    st = _lstat_entry(root)
    if not stat.S_ISDIR(st.st_mode):
        _fail()
    _absorb_entry(h, label, ".", st)
    entries: List[Tuple[str, Path]] = []
    it = os.walk(root, topdown=True, followlinks=False, onerror=_walk_onerror)
    for dirpath, dirnames, filenames in it:
        dirnames.sort()
        filenames.sort()
        base = Path(dirpath)
        for name in dirnames + filenames:
            entries.append((base, name))
        # Prune symlinked dirs defensively; lstat below also rejects.
        dirnames[:] = [d for d in dirnames if not Path(base, d).is_symlink()]
    children: List[Tuple[str, str]] = []
    for base, name in entries:
        p = base / name
        rel = p.relative_to(root).as_posix()
        st = _lstat_entry(p)
        _absorb_entry(h, label, rel, st)
        children.append((rel, _entry_type_char(st.st_mode)))
    # Sorted child path/type set (path-set drift binding).
    h.update(("sorted:" + label).encode("utf-8"))
    h.update(b"\x00")
    for rel, kind in sorted(children):
        h.update(rel.encode("utf-8"))
        h.update(b":")
        h.update(kind.encode("ascii"))
        h.update(b"\n")


def _probe_optional_dir(h: Any, label: str, root: Any) -> None:
    """Probe a configured artifact tree; explicit deterministic absence marker."""
    value = str(root) if root not in (None, "") else ""
    if not value:
        h.update((label + ":absent").encode("utf-8"))
        h.update(b"\n")
        return
    p = Path(value)
    if p.is_symlink() or not p.is_dir():
        _fail()
    _walk_tree(h, label, p)


def _json_normalize(value: Any, field: str = "") -> Any:
    """Recursive deterministic normalization: only known path fields reduce
    to basename; dict keys sort; list order stays; other values exact."""
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if field in _PATH_CONFIG_FIELDS:
            return os.path.basename(value.rstrip("/\\")) or "."
        return value
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_normalize(v, field) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(v, field) for v in value]
    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _absorb_config_identity(h: Any, cfg: Any) -> None:
    fields = set(getattr(gens, "RETRIEVAL_CONFIG_FIELDS", ()) or ())
    fields.update(_EXTRA_CONFIG_FIELDS)
    h.update(b"config-identity\x00")
    for name in sorted(fields):
        if not hasattr(cfg, name):
            continue
        h.update(name.encode("utf-8"))
        h.update(b"=")
        h.update(_dumps(_json_normalize(getattr(cfg, name), name)).encode("utf-8"))
        h.update(b"\n")
    # Live embedding runtime contract: registry mutation invalidates.
    contract = getattr(cfg, "require_embedding_runtime_contract", None)
    if callable(contract):
        try:
            result = contract()
        except Exception:
            raise FreshnessProbeError(_ERR_UNEXPECTED)
        h.update(b"embedding-runtime-contract\x00")
        h.update(_dumps(_json_normalize(result)).encode("utf-8"))
        h.update(b"\n")


def _record_distributions() -> List[str]:
    names = list(gens.RECORD_EVIDENCE_DISTRIBUTIONS)
    try:
        importlib.metadata.distribution("tokenizers")
        if "tokenizers" not in names:
            names.append("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception:
        raise FreshnessProbeError(_ERR_RECORD)
    return names


def _parse_record_rows(raw: str) -> List[str]:
    """Exact three-column CSV parse of a RECORD; returns sorted row paths."""
    rows: List[str] = []
    reader = csv.reader(io.StringIO(raw))
    for row in reader:
        if not row:
            continue
        if len(row) != 3:
            raise FreshnessProbeError(_ERR_RECORD)
        rows.append(row[0])
    rows.sort()
    if not rows:
        raise FreshnessProbeError(_ERR_RECORD)
    return rows


def _probe_record_files(h: Any) -> None:
    """Stat every file named by the RECORDs of the evidence distributions."""
    for name in _record_distributions():
        try:
            dist = importlib.metadata.distribution(name)
            raw = dist.read_text("RECORD")
        except Exception:
            raise FreshnessProbeError(_ERR_RECORD)
        if raw is None:
            raise FreshnessProbeError(_ERR_RECORD)
        for row_path in _parse_record_rows(raw):
            try:
                target = Path(str(dist.locate_file(row_path)))
            except Exception:
                raise FreshnessProbeError(_ERR_RECORD)
            st = _lstat_entry(target)
            if not stat.S_ISREG(st.st_mode):
                _fail()
            _absorb_entry(h, "record:" + name, row_path, st)


def _probe_installed_metadata(h: Any) -> None:
    """Stat the metadata file of every installed distribution (version drift)."""
    try:
        dists = list(importlib.metadata.distributions())
    except Exception:
        raise FreshnessProbeError(_ERR_METADATA)
    rows: List[Tuple[str, str, os.stat_result]] = []
    seen: List[str] = []
    for dist in dists:
        try:
            name = (dist.metadata.get("Name") or "").strip().lower() or "unknown"
            meta_rel: Optional[str] = None
            for entry in dist.files or ():
                entry_str = str(entry)
                if entry_str.endswith(".dist-info/METADATA") or entry_str.endswith(".egg-info/PKG-INFO"):
                    meta_rel = entry_str
                    break
        except Exception:
            raise FreshnessProbeError(_ERR_METADATA)
        if meta_rel is None:
            raise FreshnessProbeError(_ERR_METADATA)
        try:
            meta_path = Path(str(dist.locate_file(meta_rel)))
        except Exception:
            raise FreshnessProbeError(_ERR_METADATA)
        st = _lstat_entry(meta_path)
        if not stat.S_ISREG(st.st_mode):
            _fail()
        rows.append((name, meta_rel, st))
        seen.append(name)
    rows.sort(key=lambda item: (item[0], item[1]))
    for name, meta_rel, st in rows:
        _absorb_entry(h, "dist-metadata:" + name, meta_rel, st)
    h.update(b"dist-metadata:set\x00")
    for name in sorted(seen):
        h.update(name.encode("utf-8"))
        h.update(b"\n")


def _package_source_entries() -> List[Tuple[str, Path]]:
    """Package source selection mirroring gens.code_identity exclusions."""
    root = Path(__file__).resolve().parent
    out: List[Tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False, onerror=_walk_onerror):
        base = Path(dirpath)
        for dname in dirnames:
            if Path(base, dname).is_symlink():
                _fail()
        dirnames[:] = sorted(d for d in dirnames if d not in _SOURCE_EXCLUDED_DIRS)
        base = Path(dirpath)
        for fname in sorted(filenames):
            if fname.endswith(_SOURCE_EXCLUDED_SUFFIXES):
                continue
            out.append(((base / fname).relative_to(root).as_posix(), base / fname))
    out.sort(key=lambda item: item[0])
    return out


def _probe_package_source(h: Any) -> None:
    root = Path(__file__).resolve().parent
    if root.is_symlink() or not root.is_dir():
        _fail(_ERR_ROOT)
    st = _lstat_entry(root)
    _absorb_entry(h, "package-source", ".", st)
    for rel, path in _package_source_entries():
        st = _lstat_entry(path)
        if not stat.S_ISREG(st.st_mode):
            _fail()
        _absorb_entry(h, "package-source", rel, st)


def _probe_lock_candidates(h: Any) -> None:
    try:
        candidates = gens._dependency_lock_candidates()
    except Exception:
        _fail(_ERR_ROOT)
    found = False
    for candidate in candidates:
        path = Path(str(candidate))
        if not path.exists():
            continue
        found = True
        _probe_file(h, "dependency-lock", path, path.name)
    if not found:
        _fail(_ERR_ROOT)


def _probe_generation_tree(h: Any, cfg: Any) -> None:
    root = Path(str(cfg.data_dir))
    if root.is_symlink() or not root.is_dir():
        _fail(_ERR_ROOT)
    # Pointer.
    _probe_file(h, "pointer", root / gens.CURRENT_FILENAME, gens.CURRENT_FILENAME)
    # Pinned generation dir, receipt, and every required artifact.
    gen_id = str(cfg.active_generation_id)
    gen_root = root / gens.GENERATIONS_DIRNAME / gen_id
    if gen_root.is_symlink() or not gen_root.is_dir():
        _fail(_ERR_ROOT)
    _walk_tree(h, "generation", gen_root)
    for name, (rel, kind) in sorted(gens.REQUIRED_ARTIFACTS.items()):
        target = gen_root / rel
        if kind == "tree":
            if target.is_symlink() or not target.is_dir():
                _fail(_ERR_ROOT)
            _walk_tree(h, "artifact:" + name, target)
        else:
            _probe_file(h, "artifact:" + name, target, rel)


def _probe_live_corpus(h: Any, cfg: Any) -> None:
    root = Path(str(cfg.source_documents_dir))
    if root.is_symlink() or not root.is_dir():
        _fail(_ERR_ROOT)
    st = _lstat_entry(root)
    _absorb_entry(h, "live-corpus", ".", st)
    try:
        entries = gens.corpus_file_entries(
            root,
            supported_suffixes=set(cfg.supported_formats or []),
            exclude_patterns=list(getattr(cfg, "exclude_patterns", None) or []),
        )
    except gens.UnsafePathError:
        _fail()
    except Exception:
        _fail(_ERR_ROOT)
    admitted: List[str] = []
    for rel, path in entries:
        st = _lstat_entry(path)
        if not stat.S_ISREG(st.st_mode):
            _fail()
        _absorb_entry(h, "live-corpus", rel, st)
        admitted.append(rel)
    # Sorted admitted path set (add/delete path drift binding).
    h.update(b"live-corpus:set\x00")
    h.update(struct.pack(">Q", len(admitted)))
    for rel in admitted:
        h.update(rel.encode("utf-8"))
        h.update(b"\n")


def capture_freshness_fingerprint(cfg: Any) -> str:
    """Deterministic SHA-256 invalidation fingerprint over the full proof inputs."""
    try:
        h = hashlib.sha256(_FP_VERSION)
        _probe_generation_tree(h, cfg)
        _probe_live_corpus(h, cfg)
        _probe_package_source(h)
        # Retrieval configuration file.
        from .config import BASE_DIR

        _probe_file(h, "config", Path(BASE_DIR) / "config.yaml", "config.yaml")
        _probe_lock_candidates(h)
        _probe_optional_dir(h, "embedding-artifact", getattr(cfg, "embedding_artifact_path", None))
        _probe_optional_dir(h, "reranker-artifact", getattr(cfg, "reranker_local_artifact", None))
        _probe_record_files(h)
        _probe_installed_metadata(h)
        _absorb_config_identity(h, cfg)
        return h.hexdigest()
    except FreshnessProbeError:
        raise
    except Exception:
        raise FreshnessProbeError(_ERR_UNEXPECTED)
