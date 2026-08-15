#!/usr/bin/env python3
"""Release verification receipt builder (v4.9.0 final review revision).

Usage: release_receipt.py <dist-dir> <output-json>

Produces the RELEASE receipt — deliberately distinct from a generation
receipt (which records installed RECORD + runtime identities and never a
wheel SHA). Contents:

- github_sha (from env GITHUB_SHA; REQUIRED non-empty)
- EXACTLY ONE wheel: >1 wheel in dist/ fails (never first-vs-last)
- wheel filename + wheel sha256
- sorted archive manifest: every wheel member with size + sha256
- actual build tool versions (python, pip, build, hatchling, platform)
- runtime lock sha256 (requirements.lock) + build lock sha256
  (build-requirements.lock), both read from the EXACT git tree
- the PARSED hatch force-include mapping (source -> wheel path) with
  per-entry git-tree source sha256 and wheel-entry sha256, asserting:
    * every mapping target EXISTS in the wheel
    * tree bytes == wheel bytes for every mapping entry
    * the embedded mcp_server/data/requirements.lock sha256 equals the
      tree requirements.lock sha256 EXACTLY
- wheel mcp_server package-file drift check against the git tree (not the
  mutable worktree): every wheel package file is compared byte-for-byte
  against ``git show <sha>:<path>``; missing/extra/drifted files fail.

SOURCE BYTES COME FROM THE EXACT GITHUB_SHA GIT TREE via read-only git
plumbing (``git ls-tree`` / ``git show`` through subprocess) — the
worktree may be dirty or touched by the build, so it is never trusted.
Fails if the commit SHA is missing or the named tree/path is unavailable.
The wheel itself is read from dist/. No secrets are read or exposed.

Exit code is non-zero on any assertion failure.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_WHEEL_SUFFIXES = (".pyc", ".pyo", ".pyd")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tool_version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run read-only git plumbing in ROOT; raise on failure."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed (exit {proc.returncode}): {proc.stderr.strip()}")
    return proc


def git_tree_bytes(commit: str, path: str) -> bytes:
    """EXACT bytes of ``path`` in commit ``commit``'s tree (read-only).

    Fails loudly when the path is absent from the tree.
    """
    proc = _git(["show", f"{commit}:{path}"])
    return proc.stdout.encode("utf-8", errors="surrogateescape")


def git_tree_paths_under(commit: str, prefix: str) -> set[str]:
    """All tracked paths under ``prefix`` in ``commit``'s tree."""
    proc = _git(["ls-tree", "-r", "--name-only", commit, "--", prefix])
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def main() -> int:
    dist_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    commit = os.environ.get("GITHUB_SHA", "").strip()
    if not commit:
        print("GITHUB_SHA env var is empty — cannot bind receipt to a Git tree", file=sys.stderr)
        return 1
    # Fail early if the named tree is unavailable (bad SHA, shallow clone...).
    try:
        _git(["cat-file", "-e", f"{commit}^{{tree}}"])
    except RuntimeError as exc:
        print(f"commit tree {commit!r} unavailable: {exc}", file=sys.stderr)
        return 1

    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        print(
            f"expected EXACTLY ONE wheel in {dist_dir}, found {len(wheels)}: {[w.name for w in wheels]}",
            file=sys.stderr,
        )
        return 1
    whl = wheels[0]

    with zipfile.ZipFile(whl) as zf:
        names = zf.namelist()
        manifest = [
            {
                "name": info.filename,
                "size": info.file_size,
                "sha256": sha256_bytes(zf.read(info.filename)),
            }
            for info in sorted(zf.infolist(), key=lambda i: i.filename)
        ]
        wheel_bytes = {name: zf.read(name) for name in names}

    # ── force-include mapping, parsed from the TREE's pyproject.toml ────
    pyproject_bytes = git_tree_bytes(commit, "pyproject.toml")
    pyproject = tomllib.loads(pyproject_bytes.decode("utf-8"))
    force_include = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("force-include", {})
    )

    mapping = []
    for src_rel, dest_rel in sorted(force_include.items()):
        dest_norm = dest_rel.replace("\\", "/")
        if dest_norm not in wheel_bytes:
            print(f"force-include target missing from wheel: {src_rel} -> {dest_rel}", file=sys.stderr)
            return 1
        try:
            src_sha = sha256_bytes(git_tree_bytes(commit, src_rel))
        except RuntimeError as exc:
            print(f"force-include source {src_rel!r} unavailable in tree {commit}: {exc}", file=sys.stderr)
            return 1
        wheel_sha = sha256_bytes(wheel_bytes[dest_norm])
        if src_sha != wheel_sha:
            print(
                f"force-include byte drift vs git tree: {src_rel} (tree {src_sha[:12]}) != "
                f"wheel {dest_rel} ({wheel_sha[:12]})",
                file=sys.stderr,
            )
            return 1
        mapping.append({"source": src_rel, "wheel": dest_rel, "tree_sha256": src_sha, "wheel_sha256": wheel_sha})

    # ── embedded runtime lock must equal the TREE lock EXACTLY ──────────
    embedded_lock = "mcp_server/data/requirements.lock"
    if embedded_lock not in wheel_bytes:
        print("wheel does not embed mcp_server/data/requirements.lock", file=sys.stderr)
        return 1
    try:
        tree_lock_sha = sha256_bytes(git_tree_bytes(commit, "requirements.lock"))
    except RuntimeError as exc:
        print(f"requirements.lock unavailable in tree {commit}: {exc}", file=sys.stderr)
        return 1
    embedded_lock_sha = sha256_bytes(wheel_bytes[embedded_lock])
    if tree_lock_sha != embedded_lock_sha:
        print(
            f"embedded requirements.lock sha {embedded_lock_sha[:12]} != tree lock sha {tree_lock_sha[:12]}",
            file=sys.stderr,
        )
        return 1

    # ── wheel package-file drift vs the git tree ────────────────────────
    package_prefix = "mcp_server/"
    wheel_pkg_files = {
        name
        for name in names
        if name.startswith(package_prefix)
        and not name.endswith(EXCLUDED_WHEEL_SUFFIXES)
        and ".dist-info" not in name.split("/")
        and "__pycache__" not in name
    }
    # Tree package files EXCLUDING force-include targets (validated above)
    # and data/ (which is force-included into mcp_server/data/ wholesale).
    tree_pkg_files = {
        p
        for p in git_tree_paths_under(commit, "mcp_server")
        if not p.endswith(EXCLUDED_WHEEL_SUFFIXES)
        and "__pycache__" not in p.split("/")
        and ".dist-info" not in p.split("/")
        and not p.startswith("mcp_server/data/")
    }
    force_dests = {dest.replace("\\", "/") for dest in force_include.values()}
    expected = {f"mcp_server/{p[len('mcp_server/') :]}" for p in tree_pkg_files}
    missing = sorted(expected - wheel_pkg_files - {d for d in force_dests if d.startswith(package_prefix)})
    extra = sorted(wheel_pkg_files - expected - force_dests)
    if missing or extra:
        print(f"wheel package-file set mismatch vs git tree: missing={missing} extra={extra}", file=sys.stderr)
        return 1
    drifted = []
    for name in sorted(wheel_pkg_files - force_dests):
        tree_sha = sha256_bytes(git_tree_bytes(commit, name))
        if sha256_bytes(wheel_bytes[name]) != tree_sha:
            drifted.append(name)
    if drifted:
        print("wheel package files drifted from git tree:", drifted, file=sys.stderr)
        return 1

    try:
        build_lock_sha = sha256_bytes(git_tree_bytes(commit, "build-requirements.lock"))
    except RuntimeError:
        build_lock_sha = None  # build lock absent from the tree -> recorded as null

    receipt = {
        "github_sha": commit,
        "wheel": whl.name,
        "wheel_sha256": sha256_file(whl),
        "archive_manifest": manifest,
        "build_tools": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "pip": tool_version("pip"),
            "build": tool_version("build"),
            "hatchling": tool_version("hatchling"),
        },
        "runtime_lock": {"file": "requirements.lock", "tree_sha256": tree_lock_sha},
        "build_lock": {"file": "build-requirements.lock", "tree_sha256": build_lock_sha},
        "force_include_mapping": mapping,
        "embedded_lock_matches_tree": True,
        "source_of_truth": "git-tree",
        "wheel_package_files_checked": len(wheel_pkg_files),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "wheel": receipt["wheel"],
                "wheel_sha256": receipt["wheel_sha256"],
                "github_sha": receipt["github_sha"],
                "source_of_truth": "git-tree",
                "embedded_lock_matches_tree": True,
                "package_files": receipt["wheel_package_files_checked"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
