#!/usr/bin/env python3
"""Measure gpuwm Morrison against the committed unmodified-WRF fixture.

This is intentionally a measurement command, not the standing assertion.
It prints every field's observed FP32 total-order ULP maximum and a machine-
readable JSON summary.  The pytest gate imports :func:`measure_oracle` and
pins the observed result separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.core.fp32_ulp import (  # noqa: E402
    bitwise_identical,
    fp32_ulp_distance,
)

DEFAULT_FIXTURE_DIR = REPO_ROOT / "gpuwm" / "data" / "morrison" / "oracle"

LEVEL_INPUTS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg",
    "ni", "ns", "nr", "ng", "rho", "pii", "pressure", "dz",
    "qrcuten", "qscuten", "qicuten",
)
LEVEL_OUTPUTS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg",
    "ni", "ns", "nr", "ng",
)
SURFACE_FIELDS = (
    "rainnc", "rainncv", "snownc", "snowncv",
    "graupelnc", "graupelncv", "sr",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _f32(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows], dtype=np.float32)


def _record(
    aggregate: dict[str, dict[str, object]],
    field: str,
    got: np.ndarray,
    want: np.ndarray,
    *,
    case: int,
    mode: int,
    surface: bool = False,
) -> None:
    got = np.ascontiguousarray(got, dtype=np.float32)
    want = np.ascontiguousarray(want, dtype=np.float32)
    distance = fp32_ulp_distance(got, want)
    bits_got = got.view(np.uint32)
    bits_want = want.view(np.uint32)
    mismatch = bits_got != bits_want
    entry = aggregate.setdefault(
        field,
        {
            "max_ulp": 0,
            "mismatch_count": 0,
            "value_count": 0,
            "bitwise": True,
            "worst": None,
        },
    )
    local_max = int(distance.max()) if distance.size else 0
    entry["mismatch_count"] = int(entry["mismatch_count"]) + int(mismatch.sum())
    entry["value_count"] = int(entry["value_count"]) + int(distance.size)
    entry["bitwise"] = bool(entry["bitwise"]) and bitwise_identical(got, want)
    if local_max > int(entry["max_ulp"]) or (
            local_max == int(entry["max_ulp"])
            and entry["worst"] is None and mismatch.any()):
        flat = int(np.argmax(distance))
        entry["max_ulp"] = local_max
        entry["worst"] = {
            "case": case,
            "mode": mode,
            "k": None if surface else flat + 1,
            "got": float(got.ravel()[flat]),
            "want": float(want.ravel()[flat]),
            "got_bits": f"0x{int(bits_got.ravel()[flat]):08x}",
            "want_bits": f"0x{int(bits_want.ravel()[flat]):08x}",
        }


def measure_oracle(
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    *,
    fmad_false_diagnostic: bool = False,
) -> dict[str, object]:
    """Run every fixture column and return field/overall parity metrics."""
    import cupy as cp

    if fmad_false_diagnostic:
        # Diagnostic only: compile Morrison with contraction disabled without
        # changing the production loader or source.  The report must never be
        # used as a shipped workaround; it answers whether contraction is the
        # dominant residue.
        from functools import lru_cache
        from gpuwm.core import kernels

        original_load_module = kernels.load_module

        @lru_cache(maxsize=None)
        def diagnostic_load_module(name: str):
            if name != "morrison":
                return original_load_module(name)
            source = (
                kernels._preamble()
                + (kernels._KDIR / "morrison.cu").read_text()
            )
            module = cp.RawModule(
                code=source,
                options=("-std=c++17", "-fmad=false"),
                name_expressions=None,
            )
            module.compile()
            return module

        kernels.load_module = diagnostic_load_module
        kernels.get_kernel.cache_clear()

    from gpuwm.core.morrison import launch_morrison
    from gpuwm.core.refl import launch_refl10cm_morrison

    fixture_dir = Path(fixture_dir)
    level_rows = _read_rows(fixture_dir / "morrison-levels.csv")
    surface_rows = _read_rows(fixture_dir / "morrison-surface.csv")
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in level_rows:
        grouped[(int(row["case"]), int(row["mode"]))].append(row)
    surfaces = {
        (int(row["case"]), int(row["mode"])): row for row in surface_rows
    }
    if set(grouped) != set(surfaces):
        raise ValueError("level/surface fixture keys differ")

    aggregate: dict[str, dict[str, object]] = {}
    for (case, mode), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["k"]))
        nz = int(rows[0]["nz"])
        if len(rows) != nz or [int(row["k"]) for row in rows] != list(
                range(1, nz + 1)):
            raise ValueError(f"case={case} mode={mode}: non-contiguous levels")
        dt = np.float32(rows[0]["dt"])

        host = {
            name: _f32(rows, f"{name}_in").reshape(nz, 1, 1)
            for name in LEVEL_INPUTS
        }
        device = {
            name: cp.asarray(host[name])
            for name in (
                "theta", "qv", "qc", "qr", "qi", "qs", "qg",
                "ni", "ns", "nr", "ng", "rho", "pii", "pressure", "dz",
            )
        }
        # WRF's production INUM=1 wrapper neither accepts nor returns NC3D:
        # it constructs fixed 250 cm-3 cloud number internally.  gpuwm's
        # wider state API carries nc, but its process stage performs that
        # same overwrite before the first read, so its incoming value is
        # deliberately an inert zero here and is not compared as an output.
        device["nc"] = cp.zeros((nz, 1, 1), dtype=cp.float32)
        cu = {
            name: cp.asarray(host[name])
            for name in ("qrcuten", "qscuten", "qicuten")
        }

        surface = surfaces[(case, mode)]
        precip = {
            name: cp.asarray(
                np.asarray([[surface[f"{name}_in"]]], dtype=np.float32))
            for name in SURFACE_FIELDS
        }
        effective = {
            name: cp.empty((nz, 1, 1), dtype=cp.float32)
            for name in ("effc", "effr", "effi", "effs")
        }

        launch_morrison(
            device["theta"], device["qv"], device["qc"], device["qr"],
            device["qi"], device["qs"], device["qg"], device["nc"],
            device["nr"], device["ni"], device["ns"], device["ng"],
            device["rho"], device["pii"], device["pressure"], device["dz"],
            precip["rainnc"], precip["rainncv"],
            precip["snownc"], precip["snowncv"],
            precip["graupelnc"], precip["graupelncv"], precip["sr"],
            float(dt), **effective, **cu, morr_rimed_ice=mode,
        )

        temperature = device["theta"] * device["pii"]
        refl = cp.empty_like(device["theta"])
        launch_refl10cm_morrison(
            device["qv"], device["qr"], device["nr"],
            device["qs"], device["ns"], device["qg"], device["ng"],
            temperature, device["pressure"], refl, morr_rimed_ice=mode,
        )
        cp.cuda.get_current_stream().synchronize()

        for name in LEVEL_OUTPUTS:
            got = cp.asnumpy(device[name]).reshape(nz)
            want = _f32(rows, f"{name}_out")
            _record(aggregate, name, got, want, case=case, mode=mode)
        _record(
            aggregate,
            "refl_10cm",
            cp.asnumpy(refl).reshape(nz),
            _f32(rows, "refl_10cm_out"),
            case=case,
            mode=mode,
        )
        for name in SURFACE_FIELDS:
            _record(
                aggregate,
                name,
                cp.asnumpy(precip[name]).reshape(1),
                np.asarray([surface[f"{name}_out"]], dtype=np.float32),
                case=case,
                mode=mode,
                surface=True,
            )

    worst_field = max(
        aggregate,
        key=lambda name: (int(aggregate[name]["max_ulp"]), name),
    )
    return {
        "fixture_dir": str(fixture_dir),
        "fmad_false_diagnostic": fmad_false_diagnostic,
        "column_count": len(grouped),
        "level_count": len(level_rows),
        "fields": aggregate,
        "overall": {
            "max_ulp": int(aggregate[worst_field]["max_ulp"]),
            "worst_field": worst_field,
            "mismatch_count": sum(
                int(entry["mismatch_count"]) for entry in aggregate.values()
            ),
            "value_count": sum(
                int(entry["value_count"]) for entry in aggregate.values()
            ),
            "bitwise": all(
                bool(entry["bitwise"]) for entry in aggregate.values()
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture_dir", nargs="?", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fmad-false-diagnostic", action="store_true")
    args = parser.parse_args()
    report = measure_oracle(
        args.fixture_dir,
        fmad_false_diagnostic=args.fmad_false_diagnostic,
    )
    if not args.json:
        print("field,max_ulp,mismatches,values")
        for name, entry in sorted(report["fields"].items()):
            print(
                f"{name},{entry['max_ulp']},{entry['mismatch_count']},"
                f"{entry['value_count']}"
            )
        print(
            "overall,"
            f"{report['overall']['max_ulp']},"
            f"{report['overall']['mismatch_count']},"
            f"{report['overall']['value_count']}"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
