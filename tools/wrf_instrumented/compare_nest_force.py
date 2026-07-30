"""Compare gpuwm nest-force arrays with an instrumented-WRF N1.5 dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from dump_reader import DumpFormatError, NestForceDump, TableKey, read_dump


def _repo_root() -> Path:
    cwd = Path.cwd().resolve()
    return cwd if (cwd / "gpuwm").is_dir() else Path(__file__).parents[2]


sys.path.insert(0, str(_repo_root()))

from gpuwm.verify.nest_gates import gate  # noqa: E402


VALUE_GATE = gate("N1.5", "bdy_value_tables_fp32_relative")
TENDENCY_GATE = gate("N1.5", "bdy_tendency_tables_fp32_relative")
if VALUE_GATE.threshold != TENDENCY_GATE.threshold:
    raise RuntimeError("N1.5 value and tendency bands unexpectedly differ")
BAND = VALUE_GATE.threshold
assert BAND is not None


def load_candidate(path: str | Path) -> dict[TableKey, np.ndarray]:
    """Load canonical ``family.field.side.kind`` arrays from an NPZ."""
    result: dict[TableKey, np.ndarray] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        for raw_key in archive.files:
            if raw_key.startswith("__"):
                continue
            key = TableKey.parse(raw_key)
            values = np.asarray(archive[raw_key])
            if values.dtype != np.float32:
                raise DumpFormatError(
                    f"candidate {key} is {values.dtype}; gpuwm N1.5 arrays must be float32")
            result[key] = values
    return result


def _successor(value: np.ndarray, tendency: np.ndarray,
               parent_dt: float) -> np.ndarray:
    """FP32 successor E = fl32(VALUE + fl32(cdt*TENDENCY)) (F18).

    Exactly the expression WRF's boundary consumer applies; numpy float32
    arithmetic rounds to nearest per operation, matching the REAL
    statement semantics.
    """
    cdt = np.float32(parent_dt)
    return (value.astype(np.float32)
            + (cdt * tendency.astype(np.float32)).astype(np.float32)
            ).astype(np.float32)


def compare(
    reference: NestForceDump,
    candidate: dict[TableKey, np.ndarray],
) -> dict[str, object]:
    """Apply the exact F10 metric registered in ``nest_gates.py``.

    F18 amendment (gate record ``bdy_tendency_tables_fp32_relative``):
    the four ``state.t.*.tendency`` tables are compared at the FP32
    successor scale (:func:`_successor`, cdt = dump metadata parent_dt)
    under the SAME band -- the registered treatment of the PROVENANCE
    N1.5 harness theta-restoration seam.  Every other table keeps the
    raw metric.
    """
    expected = set(reference.tables)
    actual = set(candidate)
    missing = sorted(str(key) for key in expected - actual)
    extra = sorted(str(key) for key in actual - expected)
    records: list[dict[str, object]] = []
    group_max = {"value": 0.0, "tendency": 0.0}
    passed = not missing and not extra

    for key in sorted(expected & actual):
        wrf = reference.tables[key]
        gpuwm = candidate[key]
        record: dict[str, object] = {
            "key": str(key),
            "shape": list(wrf.shape),
        }
        theta_successor = (key.family == "state" and key.field == "t"
                           and key.kind == "tendency")
        if theta_successor and gpuwm.shape == wrf.shape:
            value_key = TableKey(key.family, key.field, key.side, "value")
            if value_key in reference.tables and value_key in candidate:
                parent_dt = float(reference.metadata.parent_dt)
                wrf = _successor(reference.tables[value_key], wrf, parent_dt)
                gpuwm = _successor(candidate[value_key], gpuwm, parent_dt)
                record["treatment"] = "f18_theta_successor"
                record["parent_dt"] = parent_dt
        if gpuwm.shape != wrf.shape:
            record.update({
                "pass": False,
                "error": f"candidate shape {gpuwm.shape}, expected {wrf.shape}",
            })
            records.append(record)
            passed = False
            continue
        if not np.isfinite(gpuwm).all():
            bad = np.argwhere(~np.isfinite(gpuwm))[0]
            record.update({
                "pass": False,
                "error": f"non-finite candidate at {tuple(int(i) for i in bad)}",
            })
            records.append(record)
            passed = False
            continue

        wrf64 = wrf.astype(np.float64)
        gpuwm64 = gpuwm.astype(np.float64)
        absolute = np.abs(gpuwm64 - wrf64)
        table_scale = float(np.max(np.abs(wrf64), initial=0.0))
        denominator = np.abs(wrf64) + table_scale
        tolerance = BAND * denominator
        element_pass = absolute <= tolerance
        if np.all(element_pass):
            max_metric = 0.0 if table_scale == 0.0 else float(
                np.max(absolute / denominator, initial=0.0))
            worst = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
        else:
            normalized = np.divide(
                absolute,
                denominator,
                out=np.full_like(absolute, np.inf),
                where=denominator != 0.0,
            )
            max_metric = float(np.max(normalized))
            worst = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
            passed = False
        group_max[key.kind] = max(group_max[key.kind], max_metric)
        record.update({
            "pass": bool(np.all(element_pass)),
            "max_metric": max_metric,
            "max_abs_error": float(np.max(absolute, initial=0.0)),
            "table_scale": table_scale,
            "worst_index": [int(index) for index in worst],
        })
        records.append(record)

    return {
        "pass": passed,
        "band": BAND,
        "metric": "abs(gpuwm-wrf) <= band*(abs(wrf)+max_abs(wrf_table))",
        "reference_tables": len(reference.tables),
        "candidate_tables": len(candidate),
        "missing": missing,
        "extra": extra,
        "value_max_metric": group_max["value"],
        "tendency_max_metric": group_max["tendency"],
        "tables": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wrf_dump", type=Path)
    parser.add_argument(
        "gpuwm_npz", type=Path,
        help="NPZ with canonical family.field.side.kind float32 arrays")
    parser.add_argument(
        "--summary", action="store_true", help="omit per-table results")
    args = parser.parse_args()

    report = compare(read_dump(args.wrf_dump), load_candidate(args.gpuwm_npz))
    if args.summary:
        report.pop("tables")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
