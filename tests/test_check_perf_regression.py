"""Тесты scripts/check_perf_regression.py (контракт docs/criteria-perf-gate-floor.md).

Критерий 1: пары с ОБЕИМИ медианами < MICRO_FLOOR_MS (1.0) не валят гейт
и печатаются отдельным [INFO]-блоком; пары, где хотя бы одна медиана ≥
floor, гейтятся как раньше. Критерий 2: порог и floor печатаются в сводке
рядом. Формат [FAIL]-блока не меняется.

pytest-benchmark stats хранят секунды.
"""

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_perf_regression", Path(__file__).resolve().parents[1] / "scripts" / "check_perf_regression.py"
)
cpr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpr)


def _write_bench(path: Path, medians_s: dict[str, float]) -> Path:
    payload = {"benchmarks": [{"name": name, "stats": {"median": median}} for name, median in medians_s.items()]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(
    monkeypatch, capsys, tmp_path: Path, master: dict[str, float], branch: dict[str, float]
) -> tuple[int, str, str]:
    m = _write_bench(tmp_path / "master.json", master)
    b = _write_bench(tmp_path / "branch.json", branch)
    monkeypatch.setattr(sys, "argv", ["check_perf_regression.py", str(m), str(b)])
    code = cpr.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_c1_subfloor_pair_not_gated_with_info(monkeypatch, capsys, tmp_path):
    # (0.11 -> 0.15, +36.4%) — обе медианы под floor: нет fail, есть info
    code, out, err = _run(monkeypatch, capsys, tmp_path, {"flaky_micro": 0.00011}, {"flaky_micro": 0.00015})
    assert code == 0
    assert "[FAIL]" not in err
    assert "[OK] No benchmarks regressed beyond threshold." in out
    assert "[INFO] sub-floor microbenchmarks (noise-dominated, not gated):" in out
    assert "flaky_micro" in out
    assert "0.11 -> 0.15 ms" in out
    assert "+36.4%" in out


def test_c1_above_floor_pair_still_fails(monkeypatch, capsys, tmp_path):
    # (6.0 -> 7.0, +16.7%) — над floor: fail как раньше
    code, out, err = _run(monkeypatch, capsys, tmp_path, {"stable_bench": 0.006}, {"stable_bench": 0.007})
    assert code == 1
    assert "[FAIL] Performance regressions detected:" in err
    assert "stable_bench" in err
    assert "median 0.01 -> 0.01" in err  # формат [FAIL]-строки не менялся
    assert "sub-floor" not in out


def test_c1_floor_crossing_pair_fails(monkeypatch, capsys, tmp_path):
    # (0.11 -> 1.5) — ветка пересекает floor: обязана валить
    code, _, err = _run(monkeypatch, capsys, tmp_path, {"crosser": 0.00011}, {"crosser": 0.0015})
    assert code == 1
    assert "[FAIL] Performance regressions detected:" in err
    assert "crosser" in err


def test_c1_just_under_floor_both_sides_info(monkeypatch, capsys, tmp_path):
    # (0.9 -> 0.99, +10% ровно, обе под floor) — info, не fail
    code, out, err = _run(monkeypatch, capsys, tmp_path, {"edge_micro": 0.0009}, {"edge_micro": 0.00099})
    assert code == 0
    assert "[FAIL]" not in err
    assert "[INFO] sub-floor microbenchmarks (noise-dominated, not gated):" in out
    assert "0.90 -> 0.99 ms" in out


def test_c1_floor_boundary_is_gated(monkeypatch, capsys, tmp_path):
    # master ровно на floor (1.0 ms) — «хотя бы одна ≥ floor» => гейтится
    code, _, err = _run(monkeypatch, capsys, tmp_path, {"at_floor": 0.001}, {"at_floor": 0.0015})
    assert code == 1
    assert "[FAIL] Performance regressions detected:" in err


def test_c2_summary_prints_threshold_and_floor(monkeypatch, capsys, tmp_path):
    _, out, _ = _run(monkeypatch, capsys, tmp_path, {"b": 0.006}, {"b": 0.006})
    assert "Threshold: ±10%; micro-floor: 1.0 ms" in out
