"""Compare two pytest-benchmark JSON outputs and fail on >10% regression.

Pillar 5 — Scalability: every PR runs ``bench/`` against master AND
against the branch. This script ingests both result files and emits a
regression report. Median is the comparison metric (least sensitive to
outlier samples).

A benchmark regresses when its median wall time grows by more than
``REGRESSION_THRESHOLD`` (10% by default). Improvements (faster) are
celebrated, never blocking.

Run locally:
    pytest bench/ --benchmark-json=branch.json
    git stash; git checkout master
    pytest bench/ --benchmark-json=master.json
    git checkout -; git stash pop
    python scripts/check_perf_regression.py master.json branch.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD = 0.10  # 10%
BYPASS_LABEL = "skip-perf-gate"

# Sub-floor microbenchmarks are noise-dominated: measured runner noise on
# sub-millisecond benches is ±30–36%, while multi-ms benches are stable to
# units of percent. Pairs where BOTH medians are below this floor are
# reported, never gated. pytest-benchmark JSON stats are in seconds.
MICRO_FLOOR_MS = 1.0
_MICRO_FLOOR_S = MICRO_FLOOR_MS / 1000.0

# Both memory benches come from bench/test_bench_memory.py; their median in
# bench-JSON is the wall-time of measure() in SECONDS (perf_counter; the
# returned MB delta never lands in the JSON, extra_info is empty). Wall-time
# of an RSS measurement with two gc.collect runs swings ±14–36% on no-op
# diffs and is NOT the subject of those benches — the subject (RSS) is
# guarded by their own absolute asserts (<50 MB / <80 MB) and Pillar 3.
# Hence these pairs are excluded from the relative time gate. (The
# m_median <= 0 continue above is unreachable for wall-time medians.)
# Names come from the bench file; basis:
# docs/criteria-perf-gate-memory-units.md.
MEMORY_BENCH_NAMES = {
    "test_bench_orchestrator_idle_rss",
    "test_bench_query_cache_5000_entries",
}


def _load(path: Path) -> dict[str, dict[str, Any]]:
    """Return mapping of bench name -> stats dict."""
    if not path.exists():
        print(f"[ERROR] Benchmark file missing: {path}", file=sys.stderr)
        raise SystemExit(2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for bench in payload.get("benchmarks", []):
        out[bench["name"]] = bench["stats"]
    if not out:
        print(f"[ERROR] No benchmarks found in {path}", file=sys.stderr)
        raise SystemExit(2)
    return out


def _format_delta(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master_json", type=Path, help="benchmark JSON from master")
    parser.add_argument("branch_json", type=Path, help="benchmark JSON from PR branch")
    parser.add_argument(
        "--threshold",
        type=float,
        default=REGRESSION_THRESHOLD,
        help=f"Fractional regression that fails the gate (default: {REGRESSION_THRESHOLD})",
    )
    args = parser.parse_args()

    master = _load(args.master_json)
    branch = _load(args.branch_json)

    common = sorted(set(master) & set(branch))
    only_master = sorted(set(master) - set(branch))
    only_branch = sorted(set(branch) - set(master))

    if only_master:
        print(f"[WARN] Benchmarks present in master but missing from branch: {only_master}", file=sys.stderr)
    if only_branch:
        print(f"[INFO] New benchmarks added by branch: {only_branch}")

    regressions: list[tuple[str, float, float, float]] = []
    improvements: list[tuple[str, float]] = []
    sub_floor: list[tuple[str, float, float, float]] = []
    memory_mb: list[tuple[str, float, float, float]] = []

    for name in common:
        m_median = master[name]["median"]
        b_median = branch[name]["median"]
        if m_median <= 0:
            continue
        delta = (b_median - m_median) / m_median
        if name in MEMORY_BENCH_NAMES:
            memory_mb.append((name, m_median, b_median, delta))
        elif m_median < _MICRO_FLOOR_S and b_median < _MICRO_FLOOR_S:
            sub_floor.append((name, m_median, b_median, delta))
        elif delta > args.threshold:
            regressions.append((name, m_median, b_median, delta))
        elif delta < -args.threshold:
            improvements.append((name, delta))

    print(f"\nBenchmarks compared: {len(common)}")
    print(f"Threshold: ±{args.threshold * 100:.0f}%; micro-floor: {MICRO_FLOOR_MS} ms")
    print(f"Memory-bench wall-time pairs excluded from gate: {len(memory_mb)}\n")

    if improvements:
        print("Improvements (faster, no action needed):")
        for name, delta in improvements:
            print(f"  ✓ {name}  {_format_delta(delta)}")
        print()

    if sub_floor:
        print("[INFO] sub-floor microbenchmarks (noise-dominated, not gated):")
        for name, m, b, delta in sub_floor:
            print(f"  ~ {name}  median {m * 1000:.2f} -> {b * 1000:.2f} ms  ({_format_delta(delta)})")
        print()

    if memory_mb:
        print("[INFO] memory-bench wall-times (seconds; RSS is asserted inside the benches, wall-time is not their subject):")
        for name, m, b, delta in memory_mb:
            print(f"  ~ {name}  median {m:.2f} -> {b:.2f} s  ({_format_delta(delta)})")
        print()

    if regressions:
        # Honor the bypass label when set deliberately on a PR
        labels = {label.strip() for label in os.environ.get("PR_LABELS", "").split(",") if label.strip()}
        if BYPASS_LABEL in labels:
            print(f"[WARN] Regressions detected but PR has '{BYPASS_LABEL}' label — bypassing:", file=sys.stderr)
            for name, m, b, delta in regressions:
                print(
                    f"  - {name}  median {m:.2f} -> {b:.2f}  ({_format_delta(delta)})",
                    file=sys.stderr,
                )
            return 0

        print("[FAIL] Performance regressions detected:", file=sys.stderr)
        for name, m, b, delta in regressions:
            print(
                f"  ✗ {name}  median {m:.2f} -> {b:.2f}  ({_format_delta(delta)})",
                file=sys.stderr,
            )
        print(
            "\nIf this regression is intentional and accepted:\n"
            "  - Document the trade-off in the PR description\n"
            f"  - Apply the '{BYPASS_LABEL}' label to bypass\n",
            file=sys.stderr,
        )
        return 1

    print("[OK] No benchmarks regressed beyond threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
