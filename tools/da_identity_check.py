"""Compare two cycle runs BYTE for BYTE, not within a tolerance.

The concurrent member path is only allowed to exist if it produces the
same numbers as the serial one.  Members are mathematically independent,
so any difference is a defect in shared state, allocator ordering,
handle-to-stream association or a cross-member reduction -- never
"acceptable noise".  This walks two run directories and compares:

* every per-leg member checkpoint the driver staged, array by array,
  with ``np.array_equal`` on the raw bytes (NaN-aware: two NaNs in the
  same slot compare equal here, because a NaN that is bitwise identical
  in both arms is agreement, not disagreement);
* the analysis increments the filter produced;
* the report's numeric leaves, excluding the timings and the execution
  block, which are expected to differ and are the point of the exercise.

Exit status is 0 only when every compared array is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: Report keys whose values are wall-clock or arrangement facts rather
#: than computed science, and which the two arms are SUPPOSED to differ
#: in.  Everything else in the report is compared.
TIMING_KEYS = frozenset({
    "wall_seconds", "phase_seconds", "solve_seconds", "total_wall_seconds",
    "members_wall_seconds", "execution", "args",
})


def _walk_numeric(node, prefix=""):
    """Yield ``(path, value)`` for every leaf that is not a timing."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in TIMING_KEYS:
                continue
            yield from _walk_numeric(value, f"{prefix}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_numeric(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


def compare_arrays(a: np.ndarray, b: np.ndarray) -> bool:
    """Bitwise equality, counting identical NaNs as equal."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True)
                if a.dtype.kind == "f" else np.array_equal(a, b))


def compare_npz(path_a: Path, path_b: Path, failures: list) -> int:
    checked = 0
    with np.load(path_a, allow_pickle=False) as da, \
            np.load(path_b, allow_pickle=False) as db:
        keys_a, keys_b = sorted(da.files), sorted(db.files)
        if keys_a != keys_b:
            failures.append(f"{path_a.name}: key sets differ "
                            f"({set(keys_a) ^ set(keys_b)})")
            return checked
        for key in keys_a:
            arr_a, arr_b = np.asarray(da[key]), np.asarray(db[key])
            checked += 1
            if not compare_arrays(arr_a, arr_b):
                diff = (np.abs(arr_a.astype(np.float64)
                               - arr_b.astype(np.float64))
                        if arr_a.shape == arr_b.shape
                        and arr_a.dtype.kind == "f" else None)
                failures.append(
                    f"{path_a.parent.name}/{path_a.name}:{key} DIFFERS"
                    + (f" (max |delta| {np.nanmax(diff):.6e}, "
                       f"{int(np.sum(diff != 0))} of {diff.size} slots)"
                       if diff is not None else ""))
    return checked


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--serial", type=Path, required=True)
    p.add_argument("--concurrent", type=Path, required=True)
    a = p.parse_args()

    failures: list = []
    checked = 0

    # -- the staged member checkpoints and the saved increments --------
    for path_a in sorted(a.serial.rglob("*.npz")):
        rel = path_a.relative_to(a.serial)
        path_b = a.concurrent / rel
        if not path_b.is_file():
            failures.append(f"{rel}: missing from the concurrent run")
            continue
        checked += compare_npz(path_a, path_b, failures)
    for path_b in sorted(a.concurrent.rglob("*.npz")):
        rel = path_b.relative_to(a.concurrent)
        if not (a.serial / rel).is_file():
            failures.append(f"{rel}: present only in the concurrent run")

    # -- the report's computed leaves ----------------------------------
    rep_a = json.loads((a.serial / "cycle-report.json").read_text("utf-8"))
    rep_b = json.loads(
        (a.concurrent / "cycle-report.json").read_text("utf-8"))
    leaves_a = dict(_walk_numeric(rep_a))
    leaves_b = dict(_walk_numeric(rep_b))
    if set(leaves_a) != set(leaves_b):
        failures.append(
            f"report shape differs: {sorted(set(leaves_a) ^ set(leaves_b))[:8]}")
    for key in sorted(set(leaves_a) & set(leaves_b)):
        checked += 1
        if leaves_a[key] != leaves_b[key]:
            failures.append(f"report {key}: {leaves_a[key]!r} != "
                            f"{leaves_b[key]!r}")

    print(f"compared {checked} arrays and report leaves")
    if failures:
        print(f"\nNOT IDENTICAL -- {len(failures)} difference(s):")
        for line in failures[:40]:
            print(f"  {line}")
        return 1
    print("BYTE-IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
