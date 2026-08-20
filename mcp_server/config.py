"""Configuration for Knowledge RAG System v4.0.0 — YAML-configurable"""

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ============================================================================
# BASE DIRECTORY RESOLUTION
# ============================================================================
# Priority: 1. KNOWLEDGE_RAG_DIR env var  2. Source checkout  3. Venv parent  4. CWD

_source_dir = Path(__file__).parent.parent


_SUPPORTED_SUFFIXES = frozenset(
    [
        ".md",
        ".txt",
        ".pdf",
        ".py",
        ".c",
        ".h",
        ".cpp",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".json",
        ".xml",
        ".docx",
        ".xlsx",
        ".pptx",
        ".csv",
        ".ipynb",
    ]
)


def _has_documents(path: Path) -> bool:
    """Check if path has a documents/ dir with actual supported files (follows symlinks)."""
    docs_dir = path / "documents"
    if not docs_dir.exists():
        return False
    for root, _, files in os.walk(docs_dir, followlinks=True):
        for f in files:
            if Path(f).suffix.lower() in _SUPPORTED_SUFFIXES:
                return True
    return False


def _venv_project_dir():
    """Detect project root from venv location (pip install from PyPI)."""
    candidates = [Path(sys.prefix), Path(sys.executable), Path(sys.executable).resolve()]
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if parent.name in ("venv", ".venv", "env", ".env"):
                return parent.parent
    return None


def _is_project_root(path):
    """Check if path looks like a knowledge-rag project (has config or documents)."""
    if path is None:
        return False
    return (path / "config.yaml").exists() or (path / "config.example.yaml").exists() or _has_documents(path)


_venv_dir = _venv_project_dir()

if os.environ.get("KNOWLEDGE_RAG_DIR"):
    BASE_DIR: Path = Path(os.environ["KNOWLEDGE_RAG_DIR"])
elif _venv_dir is not None and (_venv_dir / "config.yaml").exists():
    # Prefer venv parent if it has an actual config.yaml (editable installs, PyPI installs)
    BASE_DIR = _venv_dir
elif _is_project_root(_source_dir) and (_source_dir / "config.yaml").exists():
    BASE_DIR = _source_dir
elif _is_project_root(Path.cwd()):
    BASE_DIR = Path.cwd()
elif _is_project_root(_source_dir):
    BASE_DIR = _source_dir
elif _venv_dir is not None and _is_project_root(_venv_dir):
    BASE_DIR = _venv_dir
else:
    BASE_DIR = _venv_dir if _venv_dir is not None else Path.cwd()


# ============================================================================
# YAML CONFIG LOADER
# ============================================================================


def _load_yaml_config() -> dict:
    """Load config.yaml from BASE_DIR if it exists, otherwise return empty dict."""
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            print("[WARN] config.yaml is not a valid mapping, ignoring")
            return {}
        print(f"[INFO] Loaded config from {config_path}")
        return data
    except yaml.YAMLError as e:
        print(f"[WARN] Failed to parse config.yaml: {e} — using defaults")
        return {}


_yaml = _load_yaml_config()


def _get(section: str, key: str, default):
    """Get a value from the YAML config with section.key path, falling back to default."""
    s = _yaml.get(section, {})
    if not isinstance(s, dict):
        return default
    val = s.get(key)
    if val is None:
        return default
    # Skip type check when default is None (caller handles validation)
    if default is None:
        return val
    # YAML parses "yes"/"no" as bool, but explicit string "yes" stays str
    if not isinstance(val, type(default)):
        print(
            f"[WARN] config.yaml: {section}.{key} has wrong type "
            f"(expected {type(default).__name__}, got {type(val).__name__}), using default"
        )
        return default
    return val


def _get_nested(section: str, subsection: str, key: str, default):
    """Get a nested value from ``section.subsection.key`` in the YAML config.

    Reproduces the ``_get`` pattern one level deeper so ``fts5_*`` fields
    (backed by ``search.lexical_fast_path.*``) share the same fallback and
    type-mismatch semantics as the existing ``models.reranker.*`` block.
    """
    outer = _yaml.get(section, {})
    if not isinstance(outer, dict):
        return default
    inner = outer.get(subsection, {})
    if not isinstance(inner, dict):
        return default
    val = inner.get(key)
    if val is None:
        return default
    if default is None:
        return val
    if not isinstance(val, type(default)):
        print(
            f"[WARN] config.yaml: {section}.{subsection}.{key} has wrong type "
            f"(expected {type(default).__name__}, got {type(val).__name__}), using default"
        )
        return default
    return val


def _advanced_float(key: str, default: float) -> float:
    """Read ``advanced.<key>`` as float; YAML ints coerce, everything else falls back."""
    section = _yaml.get("advanced", {})
    if not isinstance(section, dict):
        return default
    val = section.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return default
    return float(val)


def _get_top(key: str, default):
    """Get a top-level value from YAML, falling back to default if missing or None."""
    val = _yaml.get(key)
    if val is None:
        return default
    if not isinstance(val, type(default)):
        print(f"[WARN] config.yaml: {key} has wrong type, using default")
        return default
    return val


def _normalize_query_term(term: str) -> str:
    """Normalize a query-expansion term for consistent matching."""
    return term.strip().lower()


def _append_unique(values: List[str], value: str) -> None:
    """Append a normalized value while preserving order and removing duplicates."""
    if value and value not in values:
        values.append(value)


def _merge_query_expansion_sources(
    expansions: Dict[str, List[str]], expansion_groups: List[List[str]]
) -> Dict[str, List[str]]:
    """
    Merge legacy directional expansions and symmetric expansion groups.

    Legacy `query_expansions` entries are copied first. Then every
    `query_expansion_groups` entry contributes pairwise synonym links for all
    normalized terms in the group. Overlaps are merged by union while keeping
    insertion order stable.
    """
    merged: Dict[str, List[str]] = {}

    for raw_term, raw_synonyms in expansions.items():
        if not isinstance(raw_term, str):
            continue
        term = _normalize_query_term(raw_term)
        if not term:
            continue

        bucket = merged.setdefault(term, [])
        if isinstance(raw_synonyms, list):
            for synonym in raw_synonyms:
                if not isinstance(synonym, str):
                    continue
                _append_unique(bucket, _normalize_query_term(synonym))

    for raw_group in expansion_groups:
        if not isinstance(raw_group, list):
            continue

        group_terms: List[str] = []
        for raw_term in raw_group:
            if not isinstance(raw_term, str):
                continue
            _append_unique(group_terms, _normalize_query_term(raw_term))

        if len(group_terms) < 2:
            continue

        for term in group_terms:
            bucket = merged.setdefault(term, [])
            for related_term in group_terms:
                if related_term != term:
                    _append_unique(bucket, related_term)

    return merged


# ============================================================================
# DEFAULTS (used when no config.yaml or field is omitted)
# ============================================================================

_DEFAULT_CATEGORY_MAPPINGS = {
    "security/redteam": "redteam",
    "security/blueteam": "blueteam",
    "security/ctf": "ctf",
    "security": "security",
    "aar": "aar",
    "logscale": "logscale",
    "development": "development",
    "general": "general",
}

_DEFAULT_KEYWORD_ROUTES = {
    "logscale": [
        "logscale",
        "lql",
        "cql",
        "humio",
        "crowdstrike query",
        "formattime",
        "groupby",
        "base64decode",
        "case{}",
        "regex",
    ],
    "redteam": [
        "pentest",
        "exploit",
        "payload",
        "reverse shell",
        "privilege escalation",
        "lateral movement",
        "c2",
        "beacon",
        "cobalt strike",
        "metasploit",
        "gtfobins",
        "lolbas",
        "lolbin",
        "suid",
        "sudo",
        "byovd",
        "lol driver",
        "lolad",
        "lolapps",
        "hacktricks",
        "privesc",
        "kerberoast",
        "dcsync",
        "golden ticket",
        "pass-the-hash",
        "bloodhound",
        "mimikatz",
        "rubeus",
        "certipy",
        "adcs",
        "sqli",
        "xss",
        "ssti",
        "ssrf",
        "lfi",
        "rfi",
        "xxe",
        "deserialization",
        "ysoserial",
        "upload bypass",
        "web shell",
        "hash cracking",
        "hashcat",
        "waf bypass",
        "amsi bypass",
        "uac bypass",
        "potato",
        "searchsploit",
        "exploit-db",
        "cve",
    ],
    "blueteam": [
        "detection",
        "sigma",
        "yara",
        "ioc",
        "threat hunting",
        "incident response",
        "forensics",
        "malware analysis",
    ],
    "ctf": [
        "ctf",
        "flag",
        "hackthebox",
        "htb",
        "tryhackme",
        "picoctf",
        "writeup",
        "challenge",
    ],
    "development": [
        "python",
        "typescript",
        "javascript",
        "api",
        "fastapi",
        "django",
        "react",
        "nodejs",
    ],
    "security": [
        "anti-bot",
        "antibot",
        "js challenge",
        "javascript challenge",
        "cdp detection",
        "runtime.enable",
        "puppeteer",
        "playwright",
        "selenium",
        "nodriver",
        "stealth",
        "undetected",
        "ja3",
        "ja4",
        "tls fingerprint",
        "fingerprinting",
        "curl_cffi",
        "got-scraping",
        "impersonate",
        "http/2 settings",
        "browser fingerprint",
        "canvas fingerprint",
        "webgl fingerprint",
        "navigator.webdriver",
        "audio context",
        "hardware concurrency",
        "waf bypass",
        "aws waf",
        "cloudflare bypass",
        "akamai bypass",
        "datadome",
        "perimeterx",
        "imperva bypass",
        "8kb bypass",
        "body size limit",
        "json sqli",
        "behavioral",
        "mouse movement",
        "ghost-cursor",
        "humanized",
        "flaresolverr",
        "turnstile",
        "rebrowser",
        "botbrowser",
    ],
}

_DEFAULT_QUERY_EXPANSIONS = {
    "sqli": ["sql injection", "sqli"],
    "sql injection": ["sql injection", "sqli"],
    "xss": ["cross-site scripting", "xss"],
    "cross-site scripting": ["cross-site scripting", "xss"],
    "ssrf": ["server-side request forgery", "ssrf"],
    "lfi": ["local file inclusion", "lfi"],
    "rfi": ["remote file inclusion", "rfi"],
    "rce": ["remote code execution", "rce"],
    "xxe": ["xml external entity", "xxe"],
    "ssti": ["server-side template injection", "ssti"],
    "idor": ["insecure direct object reference", "idor"],
    "csrf": ["cross-site request forgery", "csrf"],
    "privesc": ["privilege escalation", "privesc"],
    "priv esc": ["privilege escalation", "privesc"],
    "privilege escalation": ["privilege escalation", "privesc"],
    "deserialization": ["deserialization", "deserialisation", "insecure deserialization"],
    "pth": ["pass-the-hash", "pth"],
    "pass-the-hash": ["pass-the-hash", "pth"],
    "dcsync": ["dcsync", "dc sync", "domain controller sync"],
    "kerberoast": ["kerberoasting", "kerberoast"],
    "kerberoasting": ["kerberoasting", "kerberoast"],
    "asrep": ["as-rep roasting", "asrep", "asreproast"],
    "bloodhound": ["bloodhound", "sharphound"],
    "mimikatz": ["mimikatz", "sekurlsa", "logonpasswords"],
    "hashcat": ["hashcat", "hash cracking", "hash crack"],
    "john": ["john the ripper", "john", "jtr"],
    "revshell": ["reverse shell", "revshell", "rev shell"],
    "reverse shell": ["reverse shell", "revshell"],
    "webshell": ["web shell", "webshell"],
    "web shell": ["web shell", "webshell"],
    "waf": ["web application firewall", "waf"],
    "amsi": ["antimalware scan interface", "amsi", "amsi bypass"],
    "uac": ["user account control", "uac", "uac bypass"],
    "potato": ["potato", "juicypotato", "sweetpotato", "godpotato", "efspotato", "printspoofer"],
    "ntlm": ["ntlm", "net-ntlmv2", "ntlmv2"],
    "smb": ["smb", "server message block", "samba"],
    "ldap": ["ldap", "lightweight directory access protocol"],
    "ad": ["active directory", "ad"],
    "active directory": ["active directory", "ad"],
    "defender": ["windows defender", "defender", "wdfilter"],
    "responder": ["responder", "llmnr", "nbt-ns", "netbios"],
    "suid": ["suid", "setuid", "set-uid"],
    "cron": ["cron", "crontab", "cronjob", "scheduled task"],
    "lolbin": ["lolbin", "lolbas", "living off the land"],
    "c2": ["c2", "command and control", "command-and-control", "beacon"],
    "sliver": ["sliver", "sliver c2"],
    "cobalt": ["cobalt strike", "cobalt", "cs beacon"],
    "phishing": ["phishing", "spearphishing", "social engineering"],
    "forensics": ["forensics", "forensic", "dfir"],
    "volatility": ["volatility", "memory forensics", "memory analysis"],
    "steganography": ["steganography", "stego", "steghide"],
    "stego": ["steganography", "stego", "steghide"],
    "rbcd": ["resource-based constrained delegation", "rbcd"],
    "dpapi": ["dpapi", "data protection api", "credential manager"],
    "printnightmare": ["printnightmare", "cve-2021-34527", "spoolsv", "printspooler"],
    "cve-2021-34527": ["printnightmare", "cve-2021-34527", "spoolsv"],
    "eternalblue": ["eternalblue", "ms17-010", "smbv1"],
    "ms17-010": ["eternalblue", "ms17-010", "smbv1"],
    "pwnkit": ["pwnkit", "cve-2021-4034", "pkexec"],
    "cve-2021-4034": ["pwnkit", "cve-2021-4034", "pkexec"],
    "log4shell": ["log4shell", "cve-2021-44228", "log4j"],
    "cve-2021-44228": ["log4shell", "cve-2021-44228", "log4j"],
    "zerologon": ["zerologon", "cve-2020-1472", "netlogon"],
    "cve-2020-1472": ["zerologon", "cve-2020-1472", "netlogon"],
    "petitpotam": ["petitpotam", "cve-2021-36942", "efs", "ntlm relay"],
    "certifried": ["certifried", "cve-2022-26923", "adcs"],
    "nopac": ["nopac", "samaccountname", "cve-2021-42278", "cve-2021-42287"],
    "proxylogon": ["proxylogon", "cve-2021-26855", "exchange"],
    "proxyshell": ["proxyshell", "cve-2021-34473", "exchange"],
}

_DEFAULT_QUERY_EXPANSION_GROUPS: List[List[str]] = []


# ============================================================================
# EMBEDDING PROFILES (v4.8.0)
# ============================================================================
# Named shortcuts for common embedding model + dimensions + prefix combinations.
# Setting ``models.embedding.profile: "<name>"`` in config.yaml applies these
# defaults automatically. Use ``profile: "custom"`` (default) to opt out and
# respect whatever ``models.embedding.model`` / ``dimensions`` /
# ``query_prefix`` / ``passage_prefix`` are declared in YAML.
#
# WARNING: Changing profile (or the underlying model / prefix) requires a full
# reindex — existing chunks were embedded with the previous configuration and
# similarity scoring will be degraded otherwise.

_EMBEDDING_PROFILES: Dict[str, Dict[str, object]] = {
    "compact": {
        "model": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "query_prefix": "",
        "passage_prefix": "",
    },
    "quality": {
        "model": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "query_prefix": "",
        "passage_prefix": "",
    },
    "multilingual": {
        "model": "intfloat/multilingual-e5-large",
        "dimensions": 1024,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    # "custom" is a sentinel — resolver leaves models.embedding.* untouched.
    "custom": {},
}


def _yaml_embedding_has(key: str) -> bool:
    """Return True when ``models.embedding.<key>`` was explicitly declared in YAML.

    Used by the profile resolver to decide whether a user-provided override
    should win over a profile default (query_prefix / passage_prefix) or emit
    a ``[WARN] profile takes precedence`` message (embedding.model).
    """
    models = _yaml.get("models", {})
    if not isinstance(models, dict):
        return False
    emb = models.get("embedding", {})
    if not isinstance(emb, dict):
        return False
    return key in emb


class EmbeddingRuntimeContractError(ValueError):
    """Installed FastEmbed runtime does not match the versioned pins.

    Raised only by the live registry/runtime verification
    (``Config.require_embedding_runtime_contract`` /
    ``Config._resolve_embedding_runtime_contract``) for registry,
    distribution, or declared-value mismatch failures. Malformed STATIC
    versioned configuration (bad mode, missing artifact/runtime/pooling
    fields) keeps raising plain ``ValueError`` from ``_validate_index_mode``.
    Subclasses ``ValueError`` so existing value-error handling still works.
    """


# Exact installed-runtime contract for the ONLY embedding models admitted in
# versioned mode (aligned with _EMBEDDING_PROFILES). Keys are lowercase
# canonical registered FastEmbed spellings for case-insensitive lookup;
# values carry the canonical spelling, the exact expected implementation
# class family, the registered dimension, and the pooling contract.
#
# "cls-or-prepooled" is the exact FastEmbed 0.8.0 OnnxTextEmbedding branch
# contract — CLS (``embeddings[:, 0]``) for rank-3 sequence output,
# pass-through for already-pooled rank-2 output, then L2 normalization. It
# is NOT a claim that a given artifact internally uses CLS; the artifact
# tree SHA plus installed RECORD/runtime bind the bytes. Registry metadata
# cannot prove output rank, so no generic per-class pooling claim is made.
_EMBEDDING_RUNTIME_CONTRACTS: Dict[str, Dict[str, object]] = {
    entry["model"].lower(): entry
    for entry in (
        {"model": "BAAI/bge-small-en-v1.5", "impl": "onnx", "dim": 384, "pooling": "cls-or-prepooled"},
        {"model": "BAAI/bge-large-en-v1.5", "impl": "onnx", "dim": 1024, "pooling": "cls-or-prepooled"},
        {"model": "intfloat/multilingual-e5-large", "impl": "pooled", "dim": 1024, "pooling": "mean"},
    )
}


# ============================================================================
# CONFIG DATACLASS
# ============================================================================


def _resolve_path(raw, default: Path) -> Path:
    """Resolve a path from YAML (string) or use default (Path).

    Expands ``~`` to the user home directory on all platforms
    (Linux/macOS: $HOME, Windows: %USERPROFILE%).
    """
    if raw is None:
        return default
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def _resolve_artifact_dir(raw) -> Optional[Path]:
    """Resolve an optional exact-artifact directory against BASE_DIR.

    ``None`` stays ``None`` (unset); any relative value resolves against
    the canonical config directory, never the process cwd, so a builder
    launched from anywhere pins the same bytes.
    """
    if raw is None:
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


@dataclass
class Config:
    """Central configuration for the RAG system — loads from config.yaml when available."""

    # Paths
    data_dir: Path = field(default_factory=lambda: _resolve_path(_get("paths", "data_dir", None), BASE_DIR / "data"))
    chroma_dir: Path = field(
        default_factory=lambda: _resolve_path(_get("paths", "data_dir", None), BASE_DIR / "data") / "chroma_db"
    )
    documents_dir: Path = field(
        default_factory=lambda: _resolve_path(_get("paths", "documents_dir", None), BASE_DIR / "documents")
    )
    models_cache_dir: Path = field(
        default_factory=lambda: _resolve_path(_get("paths", "models_cache_dir", None), BASE_DIR / "models_cache")
    )

    # Chunking
    chunk_size: int = field(
        default_factory=lambda: (
            _get("documents", "chunking", {}).get("chunk_size", 1000)
            if isinstance(_get("documents", "chunking", {}), dict)
            else 1000
        )
    )
    chunk_overlap: int = field(
        default_factory=lambda: (
            _get("documents", "chunking", {}).get("chunk_overlap", 200)
            if isinstance(_get("documents", "chunking", {}), dict)
            else 200
        )
    )

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("model", "BAAI/bge-small-en-v1.5")
            if isinstance(_get("models", "embedding", {}), dict)
            else "BAAI/bge-small-en-v1.5"
        )
    )
    embedding_dim: int = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("dimensions", 384)
            if isinstance(_get("models", "embedding", {}), dict)
            else 384
        )
    )
    # GPU acceleration mode (v4.8.0+): "auto" (default) | "true" | "false".
    # Legacy YAML `gpu: true/false` (bool) is normalized to string in __post_init__.
    #   "auto"  — probe CUDA at startup; use if ready, fall back to CPU otherwise
    #   "true"  — force CUDA attempt; fall back to CPU only if load actually fails
    #   "false" — never probe; runs on CPU with zero startup overhead
    gpu_mode: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("gpu", "auto")
            if isinstance(_get("models", "embedding", {}), dict)
            else "auto"
        )
    )
    # Legacy alias — derived in __post_init__. True when CUDA MAY be attempted
    # (gpu_mode is "true" or "auto"). Kept for backwards compatibility with
    # callers that check `config.gpu_acceleration` as a bool.
    gpu_acceleration: bool = False

    # Embedding profile (v4.8.0) — named shortcut for model+dim+prefix.
    # See _EMBEDDING_PROFILES above. "custom" (default) opts out.
    embedding_profile: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("profile", "custom")
            if isinstance(_get("models", "embedding", {}), dict)
            else "custom"
        )
    )
    # Prefix prepended to text before embedding. Required by some model
    # families (e.g. intfloat/e5) which were trained on "query: " / "passage: "
    # scoped inputs. Leave empty for BGE / GTE / MxBai families.
    query_prefix: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("query_prefix", "")
            if isinstance(_get("models", "embedding", {}), dict)
            else ""
        )
    )
    passage_prefix: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("passage_prefix", "")
            if isinstance(_get("models", "embedding", {}), dict)
            else ""
        )
    )

    # Reranker
    reranker_model: str = field(
        default_factory=lambda: (
            _get("models", "reranker", {}).get("model", "Xenova/ms-marco-MiniLM-L-6-v2")
            if isinstance(_get("models", "reranker", {}), dict)
            else "Xenova/ms-marco-MiniLM-L-6-v2"
        )
    )
    reranker_enabled: bool = field(
        default_factory=lambda: (
            _get("models", "reranker", {}).get("enabled", True)
            if isinstance(_get("models", "reranker", {}), dict)
            else True
        )
    )
    reranker_top_k_multiplier: int = field(
        default_factory=lambda: (
            _get("models", "reranker", {}).get("top_k_multiplier", 3)
            if isinstance(_get("models", "reranker", {}), dict)
            else 3
        )
    )

    # FTS5 Lexical Fast-Path (v4.8.2+, opt-in via search.lexical_fast_path.*).
    # Default OFF preserves v4.8.1 behavior byte-for-byte. Task 03 wires the
    # dispatch inside KnowledgeOrchestrator; Fase 1 only exposes the fields.
    # Defaults chosen per ADR-002 (regex router heuristics) and ADR-003
    # (rerank OFF by default in fast-path).
    fts5_enabled: bool = field(default_factory=lambda: _get_nested("search", "lexical_fast_path", "enabled", False))
    fts5_min_hits: int = field(default_factory=lambda: _get_nested("search", "lexical_fast_path", "min_hits", 3))
    fts5_rerank_enabled: bool = field(
        default_factory=lambda: _get_nested("search", "lexical_fast_path", "rerank_enabled", False)
    )
    fts5_patterns: List[str] = field(
        default_factory=lambda: _get_nested(
            "search",
            "lexical_fast_path",
            "patterns",
            [
                r"[A-Z]{2,}-\d+",
                r"CVE-\d{4}-\d+",
                r"^[a-f0-9]{32,64}$",
            ],
        )
    )

    # ChromaDB
    collection_name: str = field(default_factory=lambda: _get("search", "collection_name", "knowledge_base"))

    # Supported formats
    supported_formats: List[str] = field(
        default_factory=lambda: _get(
            "documents",
            "supported_formats",
            [
                ".md",
                ".txt",
                ".pdf",
                ".py",
                ".c",
                ".h",
                ".cpp",
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".json",
                ".xml",
                ".docx",
                ".xlsx",
                ".pptx",
                ".csv",
                ".ipynb",
            ],
        )
    )

    # Exclude patterns for directory traversal
    exclude_patterns: List[str] = field(default_factory=lambda: _get("documents", "exclude_patterns", []))

    # Category mappings
    category_mappings: Dict[str, str] = field(
        default_factory=lambda: _get_top("category_mappings", _DEFAULT_CATEGORY_MAPPINGS)
    )

    # Keyword routes
    keyword_routes: Dict[str, List[str]] = field(
        default_factory=lambda: _get_top("keyword_routes", _DEFAULT_KEYWORD_ROUTES)
    )

    # Query expansions
    query_expansions: Dict[str, List[str]] = field(
        default_factory=lambda: _get_top("query_expansions", _DEFAULT_QUERY_EXPANSIONS)
    )

    # Symmetric query expansion groups
    query_expansion_groups: List[List[str]] = field(
        default_factory=lambda: _get_top("query_expansion_groups", _DEFAULT_QUERY_EXPANSION_GROUPS)
    )

    # Search settings
    default_results: int = field(default_factory=lambda: _get("search", "default_results", 5))
    # v4.8.0 Fase 3: default raised 20 → 100 to eliminate asymmetric candidate
    # pool in hybrid retrieval. Previously `_do_semantic()` clamped candidates
    # to `min(max_results * 3, 20) = 20` while BM25 pulled up to
    # `max_results * 20 = 400` — semantic silently starved on hybrid mode.
    # Callers passing `max_results` explicitly are unaffected.
    max_results: int = field(default_factory=lambda: _get("search", "max_results", 100))

    # Indexing (v4.8.0 Fase 3)
    # `batch_size` controls how many chunks are added per ChromaDB batch call
    # in `_index_document`. Higher = fewer SQLite round-trips = faster indexing
    # at the cost of RAM (batch_size * embedding_dim * 4 bytes for float32).
    # Default 500 preserves prior hardcoded behavior byte-for-byte.
    batch_size: int = field(default_factory=lambda: _get("documents", "batch_size", 500))
    # `parallel_workers` opts into a ThreadPoolExecutor around ChromaDB batch
    # adds. Gain comes from SQLite writes overlapping with the NEXT batch's
    # ONNX inference (embedding itself is serialized by ONNX session lock).
    # Default 1 = single-threaded = safe on all platforms.
    parallel_workers: int = field(default_factory=lambda: _get("documents", "parallel_workers", 1))

    # Server (new in v4.0.0)
    transport: str = field(default_factory=lambda: _get("server", "transport", "stdio"))
    server_host: str = field(default_factory=lambda: _get("server", "host", "127.0.0.1"))
    server_port: int = field(default_factory=lambda: _get("server", "port", 8179))
    auth_bearer_token: str = field(
        default_factory=lambda: (
            _get("server", "auth", {}).get("bearer_token", "") if isinstance(_get("server", "auth", {}), dict) else ""
        )
    )
    rate_limit_enabled: bool = field(
        default_factory=lambda: (
            _get("server", "rate_limit", {}).get("enabled", False)
            if isinstance(_get("server", "rate_limit", {}), dict)
            else False
        )
    )
    rate_limit_rpm: int = field(
        default_factory=lambda: (
            _get("server", "rate_limit", {}).get("requests_per_minute", 60)
            if isinstance(_get("server", "rate_limit", {}), dict)
            else 60
        )
    )
    rate_limit_burst: int = field(
        default_factory=lambda: (
            _get("server", "rate_limit", {}).get("burst", 10)
            if isinstance(_get("server", "rate_limit", {}), dict)
            else 10
        )
    )
    metrics_enabled: bool = field(
        default_factory=lambda: (
            _get("server", "metrics", {}).get("enabled", False)
            if isinstance(_get("server", "metrics", {}), dict)
            else False
        )
    )
    metrics_port: int = field(
        default_factory=lambda: (
            _get("server", "metrics", {}).get("port", 9179) if isinstance(_get("server", "metrics", {}), dict) else 9179
        )
    )
    log_format: str = field(
        default_factory=lambda: (
            _get("server", "logging", {}).get("format", "text")
            if isinstance(_get("server", "logging", {}), dict)
            else "text"
        )
    )
    log_level: str = field(
        default_factory=lambda: (
            _get("server", "logging", {}).get("level", "INFO")
            if isinstance(_get("server", "logging", {}), dict)
            else "INFO"
        )
    )

    # ─────────────────────────────────────────────────────────────────────
    # Advanced (v4.9.0): watcher control — typed fields replacing the former
    # hardcoded DocumentWatcher debounce. Legacy mode honors both; versioned
    # mode still forcibly disables the watcher (sealed corpus), and the
    # KNOWLEDGE_RAG_WATCHER_DISABLED=1 env var remains the emergency override.
    # ─────────────────────────────────────────────────────────────────────
    watch_for_changes: bool = field(default_factory=lambda: _get("advanced", "watch_for_changes", True))
    watch_debounce_seconds: float = field(default_factory=lambda: _advanced_float("watch_debounce_seconds", 10.0))

    # ─────────────────────────────────────────────────────────────────────
    # Versioned generations (Phase B)
    # ─────────────────────────────────────────────────────────────────────
    # indexing.mode: "legacy" (default — in-place mutable index, exact prior
    # behavior) | "versioned" (immutable sealed generations + CAS `current`).
    index_mode: str = field(default_factory=lambda: _get("indexing", "mode", "legacy"))

    # Exact local model-artifact directory pinned into compatibility
    # (versioned only). Never a model NAME and never the whole cache dir —
    # compatibility must bind the exact bytes that embed the corpus.
    # Relative values resolve against the canonical BASE_DIR (the config
    # directory), NEVER the process cwd.
    embedding_artifact_path: Optional[Path] = field(
        default_factory=lambda: _resolve_artifact_dir(
            _get("models", "embedding", {}).get("artifact_path")
            if isinstance(_get("models", "embedding", {}), dict)
            else None
        )
    )
    embedding_runtime_version: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("runtime_version", "")
            if isinstance(_get("models", "embedding", {}), dict)
            else ""
        )
    )
    embedding_pooling: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("pooling", "")
            if isinstance(_get("models", "embedding", {}), dict)
            else ""
        )
    )

    # Versioned-only: exact LOCAL reranker artifact directory (never a model
    # name, never the whole cache). In versioned serving the reranker either
    # loads from this pinned directory or is disabled — there are NO
    # first-use downloads. Empty/absent + enabled reranker in versioned mode
    # disables reranking for the process (fail-safe, never fail-open).
    # Relative values resolve against the canonical BASE_DIR (the config
    # directory), NEVER the process cwd.
    reranker_local_artifact: Optional[str] = field(
        default_factory=lambda: (
            str(_resolve_artifact_dir(_get("models", "reranker", {}).get("local_artifact")))
            if (
                isinstance(_get("models", "reranker", {}), dict)
                and _get("models", "reranker", {}).get("local_artifact") is not None
            )
            else None
        )
    )

    # Runtime-only (never YAML): frozen at process start, then mutated ONLY
    # by bind_generation(). source_documents_dir / data_dir NEVER change.
    source_documents_dir: Optional[Path] = None  # the operator's live corpus (initial documents_dir)
    index_dir: Optional[Path] = None  # legacy: data_dir; versioned: the pinned generation dir
    active_generation_id: Optional[str] = None
    active_receipt_sha256: Optional[str] = None
    active_generation_receipt: Optional[Dict[str, Any]] = None
    generation_build: bool = False  # True only inside the offline builder process

    # Runtime-only (never YAML): safe reason string when the live installed
    # FastEmbed runtime contract does not match the declared versioned pins.
    # Set by _validate_index_mode so Config() construction — and therefore
    # module import and get_index_stats — SURVIVES runtime drift (the process
    # degrades to stats-only). Every generation_compatibility() call
    # re-verifies freshly and raises EmbeddingRuntimeContractError instead.
    _embedding_runtime_contract_error: Optional[str] = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        """Validate config values and ensure directories exist."""
        self._validate_chunking()
        self._validate_indexing()
        self._resolve_embedding_profile()
        self._validate_embedding_types()
        self._normalize_gpu_mode()
        self._validate_server_transport()
        self._validate_advanced()
        self._validate_supported_formats()
        self._validate_lists_and_maps()
        self._validate_fts5()
        self._validate_index_mode()
        self._init_generation_runtime_fields()
        self._warn_missing_documents_dir()
        self._ensure_directories()

    def _validate_chunking(self) -> None:
        """Bound-check chunk_size / chunk_overlap / default_results / max_results."""
        if not isinstance(self.chunk_size, int) or self.chunk_size < 100:
            print(f"[WARN] chunk_size={self.chunk_size} invalid, using 1000")
            self.chunk_size = 1000
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            print(f"[WARN] chunk_overlap={self.chunk_overlap} invalid, using 200")
            self.chunk_overlap = 200
        if self.chunk_overlap >= self.chunk_size:
            print(
                f"[WARN] chunk_overlap ({self.chunk_overlap}) >= "
                f"chunk_size ({self.chunk_size}), using {self.chunk_size // 5}"
            )
            self.chunk_overlap = self.chunk_size // 5
        if not isinstance(self.default_results, int) or self.default_results < 1:
            self.default_results = 5
        if not isinstance(self.max_results, int) or self.max_results < 1:
            self.max_results = 100

    def _validate_indexing(self) -> None:
        """v4.8.0 Fase 3 — clamp batch_size [1,5000] and parallel_workers [1,16]."""
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            print(f"[WARN] batch_size={self.batch_size!r} invalid, clamping to 1")
            self.batch_size = 1
        elif self.batch_size > 5000:
            print(f"[WARN] batch_size={self.batch_size} exceeds 5000, clamping to 5000")
            self.batch_size = 5000

        if not isinstance(self.parallel_workers, int) or self.parallel_workers < 1:
            print(f"[WARN] parallel_workers={self.parallel_workers!r} invalid, clamping to 1")
            self.parallel_workers = 1
        elif self.parallel_workers > 16:
            print(f"[WARN] parallel_workers={self.parallel_workers} exceeds 16, clamping to 16")
            self.parallel_workers = 16
        elif self.parallel_workers > 4:
            import platform

            if platform.system() == "Windows":
                print(
                    f"[WARN] parallel_workers={self.parallel_workers} on Windows may hit "
                    f"ONNX threading issues or SQLite lock contention; monitor stability"
                )

    def _resolve_embedding_profile(self) -> None:
        """v4.8.0 — resolve profile into model/dim/prefixes (runs BEFORE dim validation).

        A valid profile may set a non-384 dim without triggering the fallback
        in ``_validate_embedding_types``.
        """
        if not isinstance(self.embedding_profile, str):
            print(f"[WARN] embedding_profile={self.embedding_profile!r} invalid, using 'custom'")
            self.embedding_profile = "custom"

        if self.embedding_profile == "custom":
            return

        profile = _EMBEDDING_PROFILES.get(self.embedding_profile)
        if not profile:
            print(f"[WARN] Invalid embedding profile '{self.embedding_profile}'; falling back to 'custom'")
            self.embedding_profile = "custom"
            return

        self._apply_embedding_profile(profile)

    def _apply_embedding_profile(self, profile: dict) -> None:
        """Copy profile model/dim into config; prefixes only if user did not declare them."""
        if _yaml_embedding_has("model"):
            print(
                f"[WARN] Both models.embedding.model and profile="
                f"'{self.embedding_profile}' set; profile takes precedence"
            )
        self.embedding_model = profile["model"]
        self.embedding_dim = profile["dimensions"]
        # Empty string is a valid explicit user override, hence _yaml_*_has.
        if not _yaml_embedding_has("query_prefix"):
            self.query_prefix = profile["query_prefix"]
        if not _yaml_embedding_has("passage_prefix"):
            self.passage_prefix = profile["passage_prefix"]

    def _validate_embedding_types(self) -> None:
        """Type-check prefixes + embedding_dim + reranker settings (runs AFTER profile)."""
        if not isinstance(self.query_prefix, str):
            print(f"[WARN] query_prefix={self.query_prefix!r} invalid, using ''")
            self.query_prefix = ""
        if not isinstance(self.passage_prefix, str):
            print(f"[WARN] passage_prefix={self.passage_prefix!r} invalid, using ''")
            self.passage_prefix = ""

        if not isinstance(self.embedding_dim, int) or self.embedding_dim < 1:
            self.embedding_dim = 384
        if not isinstance(self.reranker_enabled, bool):
            print(f"[WARN] reranker_enabled={self.reranker_enabled!r} invalid, using True")
            self.reranker_enabled = True
        if not isinstance(self.reranker_top_k_multiplier, int) or self.reranker_top_k_multiplier < 1:
            self.reranker_top_k_multiplier = 3

    def _normalize_gpu_mode(self) -> None:
        """v4.8.0+ — bool → str, str → validated one of {'auto','true','false'}.

        Accepts legacy YAML ``gpu: true/false`` (bool) and new
        ``gpu: "auto"|"true"|"false"``. Also derives ``gpu_acceleration``
        legacy alias: True when CUDA may be attempted.
        """
        raw_gpu = self.gpu_mode
        if isinstance(raw_gpu, bool):
            self.gpu_mode = "true" if raw_gpu else "false"
        elif isinstance(raw_gpu, str):
            normalized = raw_gpu.strip().lower()
            if normalized in ("auto", "true", "false"):
                self.gpu_mode = normalized
            else:
                print(f"[WARN] Invalid gpu value {raw_gpu!r}; falling back to 'auto'")
                self.gpu_mode = "auto"
        else:
            print(f"[WARN] Invalid gpu value {raw_gpu!r} (type {type(raw_gpu).__name__}); falling back to 'auto'")
            self.gpu_mode = "auto"
        self.gpu_acceleration = self.gpu_mode in ("true", "auto")

    def _validate_server_transport(self) -> None:
        """Bound-check transport / server_port / metrics_port / rate limits."""
        if self.transport not in ("stdio", "sse", "streamable-http"):
            print(f"[WARN] server.transport={self.transport!r} invalid, using 'stdio'")
            self.transport = "stdio"
        if not isinstance(self.server_port, int) or not (1 <= self.server_port <= 65535):
            self.server_port = 8179
        if not isinstance(self.metrics_port, int) or not (1 <= self.metrics_port <= 65535):
            self.metrics_port = 9179
        if not isinstance(self.rate_limit_rpm, int) or self.rate_limit_rpm < 1:
            self.rate_limit_rpm = 60
        if not isinstance(self.log_format, str) or self.log_format not in ("text", "json"):
            print(f"[WARN] server.logging.format={self.log_format!r} invalid, using 'text'")
            self.log_format = "text"
        _valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if not isinstance(self.log_level, str) or self.log_level.upper() not in _valid_levels:
            print(f"[WARN] server.logging.level={self.log_level!r} invalid, using 'INFO'")
            self.log_level = "INFO"
        else:
            self.log_level = self.log_level.upper()
        if not isinstance(self.rate_limit_burst, int) or self.rate_limit_burst < 0:
            self.rate_limit_burst = 10

    def _validate_advanced(self) -> None:
        """v4.9.0 — type-check watcher fields; debounce must be finite and > 0."""
        if not isinstance(self.watch_for_changes, bool):
            print(f"[WARN] advanced.watch_for_changes={self.watch_for_changes!r} invalid, using True")
            self.watch_for_changes = True
        debounce = self.watch_debounce_seconds
        if (
            isinstance(debounce, bool)
            or not isinstance(debounce, (int, float))
            or not math.isfinite(debounce)
            or debounce <= 0
        ):
            print(
                f"[WARN] advanced.watch_debounce_seconds={debounce!r} invalid "
                "(must be a finite positive number), using 10.0"
            )
            self.watch_debounce_seconds = 10.0
        else:
            self.watch_debounce_seconds = float(debounce)

    def _validate_supported_formats(self) -> None:
        """Ensure supported_formats is a non-empty list; fall back to canonical defaults."""
        if isinstance(self.supported_formats, list) and self.supported_formats:
            return
        print("[WARN] supported_formats is empty or invalid, using defaults")
        self.supported_formats = [
            ".md",
            ".txt",
            ".pdf",
            ".py",
            ".c",
            ".h",
            ".cpp",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".json",
            ".xml",
            ".docx",
            ".xlsx",
            ".pptx",
            ".csv",
            ".ipynb",
        ]

    def _validate_lists_and_maps(self) -> None:
        """Type-check exclude_patterns + keyword_routes + query_expansions in order."""
        self._validate_exclude_and_routes()
        self._validate_query_expansions()

    def _validate_exclude_and_routes(self) -> None:
        """Ensure exclude_patterns is a str-only list and keyword_routes values are lists."""
        if not isinstance(self.exclude_patterns, list):
            print(f"[WARN] exclude_patterns={self.exclude_patterns!r} invalid, using []")
            self.exclude_patterns = []
        else:
            self.exclude_patterns = [p for p in self.exclude_patterns if isinstance(p, str)]

        for cat, keywords in list(self.keyword_routes.items()):
            if not isinstance(keywords, list):
                print(f"[WARN] keyword_routes.{cat} is not a list, removing")
                del self.keyword_routes[cat]

    def _validate_query_expansions(self) -> None:
        """Type-check + merge query_expansions with query_expansion_groups."""
        if not isinstance(self.query_expansions, dict):
            print("[WARN] query_expansions is invalid, using defaults")
            self.query_expansions = dict(_DEFAULT_QUERY_EXPANSIONS)

        for term, synonyms in list(self.query_expansions.items()):
            if not isinstance(term, str) or not isinstance(synonyms, list):
                print(f"[WARN] query_expansions.{term} is invalid, removing")
                del self.query_expansions[term]

        if not isinstance(self.query_expansion_groups, list):
            print("[WARN] query_expansion_groups is invalid, ignoring")
            self.query_expansion_groups = []

        self.query_expansions = _merge_query_expansion_sources(self.query_expansions, self.query_expansion_groups)

    def _validate_fts5(self) -> None:
        """Type-check and sanity-check the FTS5 fast-path config fields.

        Invalid regex patterns raise ``re.error`` here (fail-fast on startup).
        Overly broad or high-count pattern lists log warnings.
        """
        import re as _re

        if not isinstance(self.fts5_enabled, bool):
            print(f"[WARN] fts5_enabled={self.fts5_enabled!r} invalid, using False")
            self.fts5_enabled = False
        if not isinstance(self.fts5_rerank_enabled, bool):
            print(f"[WARN] fts5_rerank_enabled={self.fts5_rerank_enabled!r} invalid, using False")
            self.fts5_rerank_enabled = False
        if not isinstance(self.fts5_min_hits, int) or self.fts5_min_hits < 1:
            print(f"[WARN] fts5_min_hits={self.fts5_min_hits!r} invalid, using 3")
            self.fts5_min_hits = 3
        if not isinstance(self.fts5_patterns, list):
            print(f"[WARN] fts5_patterns={self.fts5_patterns!r} invalid, using []")
            self.fts5_patterns = []
        else:
            self.fts5_patterns = [p for p in self.fts5_patterns if isinstance(p, str)]

        for idx, pattern in enumerate(self.fts5_patterns):
            try:
                _re.compile(pattern)
            except _re.error as exc:
                raise _re.error(
                    f"config.yaml: search.lexical_fast_path.patterns[{idx}] ({pattern!r}) is not a valid regex: {exc}"
                )
            if pattern in (r".+", r".*", r"^.+$", r"^.*$"):
                print(
                    f"[WARN] fts5_patterns[{idx}]={pattern!r} is overly broad and will classify most queries as lexical"
                )
        if not self.fts5_patterns:
            print("[WARN] fts5_patterns is empty — the FTS5 router will never classify a query as lexical")
        elif len(self.fts5_patterns) > 20:
            print(
                f"[WARN] fts5_patterns has {len(self.fts5_patterns)} entries; "
                f"high pattern count may impact router performance (recommended <=5)"
            )

    def _warn_missing_documents_dir(self) -> None:
        """Emit a hint if documents_dir was set explicitly and does not exist."""
        raw_docs = _get("paths", "documents_dir", None)
        if raw_docs is not None and not self.documents_dir.exists():
            print(
                f"[WARN] documents_dir '{raw_docs}' resolved to "
                f"'{self.documents_dir}' which does not exist — creating it. "
                f"Verify the path in config.yaml if reindex returns 0 files."
            )

    def _ensure_directories_legacy(self) -> None:
        """Create data/chroma/documents/models directories if missing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_embedding_runtime_contract(self, model_name: str) -> Dict[str, Any]:
        """Registry-only FastEmbed runtime contract for versioned validation.

        Inspects the installed FastEmbed 0.8.0 ``TextEmbedding`` dispatch
        registry exactly as upstream defines it: iterate
        ``EMBEDDINGS_REGISTRY`` (a ``list[Type[TextEmbeddingBase]]`` in
        dispatch order, never a mapping) calling each implementation's
        ``_list_supported_models()``. Never constructs ``TextEmbedding``,
        never reads model files, never touches the network.

        Admits EXACTLY the three project built-ins in
        ``_EMBEDDING_RUNTIME_CONTRACTS`` (aligned with ``_EMBEDDING_PROFILES``)
        and requires exactly one installed registry match with the exact
        canonical registered spelling, the exact expected implementation
        CLASS IDENTITY (lazily imported from its exact module — never
        ``__name__``, so a same-name lookalike fails closed), and the exact
        expected dimension. Every other model/class — including other exact
        Onnx models, PooledNormalized, Custom, CLIP, Jina, sparse/image/
        late-interaction — fails closed. Registry metadata cannot prove
        output rank or pre-pooling semantics, so no generic per-class
        pooling claim is made; BGE rows carry the honest
        ``cls-or-prepooled`` FastEmbed 0.8.0 branch contract. Returns the
        canonical model spelling, installed runtime string
        (``"fastembed <version>"``), registered dimension, and pooling
        contract. All failures raise ``EmbeddingRuntimeContractError``.
        """
        import importlib.metadata as _importlib_metadata

        try:
            from fastembed import TextEmbedding as _TextEmbedding
        except Exception as exc:  # pragma: no cover - fastembed is a hard runtime dep
            raise EmbeddingRuntimeContractError(
                f"cannot import fastembed registry to verify embedding runtime contract: {exc}"
            ) from exc

        model_want = str(model_name or "").strip().lower()
        if not model_want:
            raise EmbeddingRuntimeContractError(
                "models.embedding.model is required to verify the versioned runtime contract"
            )
        expected = _EMBEDDING_RUNTIME_CONTRACTS.get(model_want)
        if expected is None:
            allowed = ", ".join(sorted(str(e["model"]) for e in _EMBEDDING_RUNTIME_CONTRACTS.values()))
            raise EmbeddingRuntimeContractError(
                f"embedding model {str(model_name).strip()!r} is not admitted in versioned mode "
                f"(supported: {allowed}); no name-table fallback"
            )

        # matched entries: (exact implementation class, registered dim, exact
        # registered model spelling). Classification below uses CLASS IDENTITY
        # against the exact lazily-imported FastEmbed implementation classes —
        # never ``__name__`` — so a specialized or shadowing same-name class
        # fails closed.
        matched: list[tuple[type, Any, str]] = []
        try:
            registry = _TextEmbedding.EMBEDDINGS_REGISTRY
            if isinstance(registry, dict) or not isinstance(registry, (list, tuple)):
                raise EmbeddingRuntimeContractError(
                    "fastembed EMBEDDINGS_REGISTRY must be a dispatch-ordered list of implementation classes"
                )
            for impl in registry:
                impl_label = impl.__name__ if isinstance(impl, type) else type(impl).__name__
                try:
                    descriptions = impl._list_supported_models()
                except Exception as exc:
                    raise EmbeddingRuntimeContractError(
                        f"failed to list supported models for fastembed implementation {impl_label!r}: {exc}"
                    ) from exc
                if isinstance(descriptions, dict):  # pragma: no cover - defensive
                    descriptions = descriptions.get("models", [])
                for description in descriptions:
                    try:
                        registered = str(getattr(description, "model", "") or "")
                        dim = getattr(description, "dim", None)
                    except Exception as exc:
                        raise EmbeddingRuntimeContractError(
                            f"unreadable model description in fastembed {impl_label!r}: {exc}"
                        ) from exc
                    if registered.strip().lower() == model_want:
                        matched.append((impl, dim, registered))
        except EmbeddingRuntimeContractError:
            raise
        except Exception as exc:
            raise EmbeddingRuntimeContractError(f"failed to inspect the fastembed embedding registry: {exc}") from exc

        if not matched:
            raise EmbeddingRuntimeContractError(
                f"embedding model {str(model_name).strip()!r} is not registered in the installed "
                "fastembed runtime registry (versioned mode fails closed; no name-table fallback)"
            )
        if len(matched) > 1:
            impls = ", ".join(sorted({getattr(i, "__name__", str(i)) for i, _, _ in matched}))
            raise EmbeddingRuntimeContractError(
                f"embedding model {str(model_name).strip()!r} is registered by multiple fastembed "
                f"implementations ({impls}); versioned mode cannot prove which one dispatches"
            )

        impl, dim, registered = matched[0]
        # Exact canonical registered spelling (case-insensitive user input is
        # accepted only because the canonical spelling is returned/used).
        if registered != expected["model"]:
            raise EmbeddingRuntimeContractError(
                f"fastembed registers {str(model_name).strip()!r} with non-canonical spelling "
                f"{registered!r}; expected exactly {expected['model']!r}"
            )
        if isinstance(dim, bool) or not isinstance(dim, int) or dim != expected["dim"]:
            impl_label = getattr(impl, "__name__", str(impl))
            raise EmbeddingRuntimeContractError(
                f"fastembed registers dimension {dim!r} for {registered!r} ({impl_label}); "
                f"versioned mode requires exactly {expected['dim']}"
            )

        # Exact-class identity: import the two admitted implementation classes
        # from their exact modules and admit ONLY their exact identities.
        try:
            from fastembed.text.onnx_embedding import OnnxTextEmbedding as _ExactOnnxTextEmbedding
            from fastembed.text.pooled_embedding import PooledEmbedding as _ExactPooledEmbedding
        except Exception as exc:
            raise EmbeddingRuntimeContractError(
                f"cannot import the exact fastembed implementation classes for identity "
                f"verification; versioned mode fails closed: {exc}"
            ) from exc

        if expected["impl"] == "onnx":
            if impl is not _ExactOnnxTextEmbedding:
                impl_label = getattr(impl, "__name__", str(impl))
                raise EmbeddingRuntimeContractError(
                    f"embedding model {registered!r} must dispatch to the exact "
                    f"fastembed.text.onnx_embedding.OnnxTextEmbedding class; got {impl_label!r} "
                    "(exact-class identity check failed; a same-name class is not admitted)"
                )
        elif expected["impl"] == "pooled":
            if impl is not _ExactPooledEmbedding:
                impl_label = getattr(impl, "__name__", str(impl))
                raise EmbeddingRuntimeContractError(
                    f"embedding model {registered!r} must dispatch to the exact "
                    f"fastembed.text.pooled_embedding.PooledEmbedding class; got {impl_label!r} "
                    "(exact-class identity check failed; a same-name class is not admitted)"
                )
        else:  # pragma: no cover - allowlist only carries onnx/pooled
            raise EmbeddingRuntimeContractError(f"internal error: unknown implementation family {expected['impl']!r}")

        # Distribution identity: the exact production lock is CPU fastembed.
        # A shadowing fastembed-gpu distribution must never be reported as
        # CPU ``fastembed <version>``; both installed (or CPU missing) fails.
        try:
            version = _importlib_metadata.version("fastembed")
        except Exception as exc:
            raise EmbeddingRuntimeContractError(
                f"fastembed (CPU) distribution metadata unavailable; versioned mode fails closed: {exc}"
            ) from exc
        try:
            _importlib_metadata.version("fastembed-gpu")
        except _importlib_metadata.PackageNotFoundError:
            pass
        except Exception as exc:
            raise EmbeddingRuntimeContractError(
                f"cannot determine whether the fastembed-gpu distribution shadows the CPU "
                f"fastembed pin; versioned mode fails closed: {exc}"
            ) from exc
        else:
            raise EmbeddingRuntimeContractError(
                "both 'fastembed' and 'fastembed-gpu' distributions are installed; the GPU "
                "distribution must not shadow the exact CPU fastembed production pin "
                "(indexing.mode=versioned fails closed)"
            )

        return {
            "embedding_model": expected["model"],
            "runtime_version": f"fastembed {version}",
            "embedding_dim": int(dim),
            "pooling": str(expected["pooling"]),
        }

    def require_embedding_runtime_contract(self) -> Dict[str, Any]:
        """Freshly verify the declared pins against the installed runtime.

        Re-resolves the live FastEmbed registry/distribution on EVERY call
        (a cached previous success is never authority), compares the declared
        ``embedding_runtime_version`` / ``embedding_dim`` /
        ``embedding_pooling`` (pooling case-insensitively) against the
        canonical contract, and returns the canonical values. Only
        ``EmbeddingRuntimeContractError`` is raised for registry,
        distribution, or mismatch failures. Called by versioned
        ``_validate_index_mode`` (which stores the safe message instead of
        raising, so ``Config()`` and module import survive runtime drift)
        and by ``generation_compatibility()`` (which hard-fails — offline
        generation must never proceed on an unproven runtime).
        """
        contract = self._resolve_embedding_runtime_contract(self.embedding_model)
        declared_runtime = str(self.embedding_runtime_version or "").strip()
        if declared_runtime != contract["runtime_version"]:
            raise EmbeddingRuntimeContractError(
                f"models.embedding.runtime_version {declared_runtime!r} does not match the "
                f"installed embedding runtime {contract['runtime_version']!r} "
                "(indexing.mode=versioned fails closed)"
            )
        declared_dim = self.embedding_dim
        if (
            isinstance(declared_dim, bool)
            or not isinstance(declared_dim, int)
            or declared_dim != contract["embedding_dim"]
        ):
            raise EmbeddingRuntimeContractError(
                f"models.embedding.dimensions {declared_dim!r} does not match the registered "
                f"dimension {contract['embedding_dim']} for embedding model "
                f"{str(self.embedding_model).strip()!r} (indexing.mode=versioned fails closed)"
            )
        declared_pooling = str(self.embedding_pooling or "").strip().lower()
        if declared_pooling != contract["pooling"]:
            raise EmbeddingRuntimeContractError(
                f"models.embedding.pooling {str(self.embedding_pooling).strip()!r} does not match the "
                f"proven pooling contract {contract['pooling']!r} for embedding model "
                f"{str(self.embedding_model).strip()!r} (indexing.mode=versioned fails closed)"
            )
        return contract

    def _validate_index_mode(self) -> None:
        """Validate ``indexing.mode`` plus the fail-closed versioned inputs.

        Legacy is the default and imposes no new requirements. Versioned mode
        requires the exact local embedding artifact directory plus its declared
        runtime version and pooling so generations stay byte-compatible. The
        artifact is validated, never created or modified.
        """
        raw = self.index_mode if isinstance(self.index_mode, str) else "legacy"
        mode = raw.strip().lower() or "legacy"
        if mode not in ("legacy", "versioned"):
            raise ValueError(f"indexing.mode must be exactly 'legacy' or 'versioned'; got {raw!r}")
        self.index_mode = mode
        if mode != "versioned":
            return
        artifact = self.embedding_artifact_path
        if not artifact or not str(artifact).strip():
            raise ValueError(
                "indexing.mode=versioned requires models.embedding.artifact_path "
                "(exact local model artifact directory; fail-closed)"
            )
        artifact_path = Path(str(artifact).strip()).expanduser()
        if artifact_path.is_symlink() or not artifact_path.is_dir():
            raise ValueError(
                "models.embedding.artifact_path must be an existing local "
                f"artifact directory (symlinks rejected): {artifact_path}"
            )
        self.embedding_artifact_path = str(artifact_path)
        if not str(self.embedding_runtime_version or "").strip():
            raise ValueError("indexing.mode=versioned requires models.embedding.runtime_version")
        if not str(self.embedding_pooling or "").strip():
            raise ValueError("indexing.mode=versioned requires models.embedding.pooling")
        # Live registry/runtime verification. Unlike the STATIC checks above,
        # a live mismatch (installed FastEmbed registry/dimension/pooling/
        # distribution drift) must NOT abort Config construction: the module
        # would fail to import and get_index_stats could never report the
        # degraded state. Store the safe reason instead; generation pinning
        # and every generation_compatibility() call re-verify freshly and
        # fail closed via EmbeddingRuntimeContractError.
        try:
            contract = self.require_embedding_runtime_contract()
        except EmbeddingRuntimeContractError as exc:
            self._embedding_runtime_contract_error = str(exc)
            return
        self._embedding_runtime_contract_error = None
        # Canonicalize ONLY on a proven match: the declared values become the
        # exact registry/runtime actuals so generation compatibility binds
        # the real runtime behavior (unproven declarations are never kept).
        self.embedding_model = contract["embedding_model"]
        self.embedding_runtime_version = contract["runtime_version"]
        self.embedding_dim = contract["embedding_dim"]
        self.embedding_pooling = contract["pooling"]

    def _init_generation_runtime_fields(self) -> None:
        """Populate runtime-only generation fields. Performs no I/O."""
        self.source_documents_dir = self.documents_dir
        if self.index_mode == "versioned":
            # Versioned mode creates NOTHING here. index_dir stays unbound
            # until bind_generation() pins the sealed generation directory.
            self.index_dir = None
            return
        self.index_dir = self.data_dir

    def _ensure_directories(self) -> None:
        """Versioned-aware directory setup.

        Versioned mode creates no data_dir/chroma_dir/documents_dir (nor any
        lock, pointer, or generations entry): the generation store owns the
        on-disk layout and every read is fail-closed. Legacy mode keeps the
        exact original behavior via ``_ensure_directories_legacy``.
        """
        mode = str(self.index_mode or "legacy").strip().lower()
        if mode == "versioned":
            return
        self._ensure_directories_legacy()

    def generation_compatibility(self) -> Dict[str, Any]:
        """Deterministic 10-key compatibility object (builder and startup).

        ``model_artifact_sha256`` is the canonical tree digest of the exact
        local artifact directory — never a model name or cache-wide hash.
        """
        if self.index_mode != "versioned":
            raise RuntimeError("generation_compatibility() requires indexing.mode=versioned")
        from mcp_server.generations import digest_tree

        # Fresh live verification on EVERY call — a startup-time success is
        # never authority for the offline builder or for serving drift
        # rechecks (the installed runtime may have changed since Config()).
        # Raises EmbeddingRuntimeContractError on any drift so generation
        # building/pinning fails closed; compatibility below is built from
        # the canonical returned actuals, not the declared YAML values.
        contract = self.require_embedding_runtime_contract()
        artifact_path = Path(self.embedding_artifact_path).expanduser()
        artifact_sha, _entries = digest_tree(artifact_path)
        # Attribute names vary across config revisions — resolve defensively;
        # the receipt validator rejects anything invalid downstream.
        collection = getattr(self, "collection_name", None) or getattr(self, "collection", None)
        return {
            "collection_name": collection,
            "embedding_model": contract["embedding_model"],
            "embedding_dimension": contract["embedding_dim"],
            "query_prefix": getattr(self, "query_prefix", ""),
            "passage_prefix": getattr(self, "passage_prefix", ""),
            "model_artifact_sha256": artifact_sha,
            "runtime_version": contract["runtime_version"],
            "pooling": contract["pooling"],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            # Reranker binding (P0 #9/A): enabled flag + logical model name +
            # exact artifact digest. When the reranker is disabled this is
            # the explicit disabled identity state (None model/digest); when
            # enabled without a materialized artifact it is None-None and
            # serving fails closed.
            "reranker_enabled": bool(getattr(self, "reranker_enabled", False)),
            "reranker_model": (
                str(getattr(self, "reranker_model", "") or "") if getattr(self, "reranker_enabled", False) else None
            ),
            "reranker_artifact_sha256": self._reranker_artifact_sha256(),
        }

    def _reranker_artifact_sha256(self) -> Optional[str]:
        """Exact tree digest of the configured reranker artifact, or None.

        Called only in versioned mode. Digests the materialized local
        artifact directory (symlink forests are rejected by digest_tree).
        """
        if self.index_mode != "versioned":
            return None
        if not getattr(self, "reranker_enabled", False):
            return None
        artifact = getattr(self, "reranker_local_artifact", None)
        if not artifact:
            return None
        from mcp_server.generations import digest_tree

        return digest_tree(Path(str(artifact)).expanduser())[0]

    def bind_generation(self, current, *, building: bool = False) -> None:
        """Bind this process to exactly one sealed (or in-build) generation.

        Rewrites only the runtime binding fields — ``index_dir``,
        ``chroma_dir``, ``documents_dir`` and the active identity snapshot.
        ``data_dir`` (generation root) and ``source_documents_dir`` (original
        live source tree) never change. Must run before any index handle opens.
        """
        if self.index_mode != "versioned":
            raise RuntimeError("bind_generation() requires indexing.mode=versioned")
        gen_dir = getattr(current, "generation_dir", None)
        if gen_dir is None:
            raise TypeError(
                "bind_generation() expects the CurrentGeneration returned by "
                "GenerationStore.require_current()/resolve_current()"
            )
        from mcp_server.generations import CHROMA_ARTIFACT as CHROMA_DIRNAME
        from mcp_server.generations import CORPUS_ARTIFACT as CORPUS_DIRNAME

        gen_dir = Path(gen_dir)
        self.index_dir = gen_dir
        self.chroma_dir = gen_dir / CHROMA_DIRNAME
        self.documents_dir = gen_dir / CORPUS_DIRNAME
        self.generation_build = bool(building)
        self.active_generation_id = getattr(current, "generation_id", None)
        self.active_receipt_sha256 = getattr(current, "receipt_sha256", None)
        receipt = getattr(current, "receipt", None)
        self.active_generation_receipt = receipt if isinstance(receipt, dict) else None


config = Config()
