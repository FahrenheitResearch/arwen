"""Deterministic synthetic run pair for the metric-delegation regression.

The N5S scorer is being re-pointed at the generic metric implementations in
``gpuwm.verify.field_metrics``.  The only thing that proves the move changed
nothing is a set of numbers produced by the scorer BEFORE the move and
re-produced by it afterwards, bit for bit.  This module builds the run pair
those numbers are measured on: deterministic frames written by a seeded
PCG64 generator, non-trivial in every scored carrier (state fields differ,
boundary increments differ, both runs grow qualifying reflectivity objects at
different times).

Run as a script it writes the baseline record consumed by
``tests/test_field_metrics.py``::

    python -m tests.n5s_delegation_fixture <baseline.json>

Floats are recorded as ``float.hex()`` so the comparison is bit-exact rather
than merely close.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile

import netCDF4
import numpy as np

FIXTURE_SCHEMA = 1
#: The synthetic pair's start instant.  It is fixture data, not a pin: the
#: registration under test supplies its own.
FIXTURE_START = "2001-02-03T04:00:00"
_NZ, _NY, _NX = 2, 14, 16


def _seeded(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        np.random.default_rng(seed).standard_normal(shape), dtype=np.float32)


def _reflectivity(seed: int, *, active: bool) -> np.ndarray:
    field = 10.0 + 5.0 * _seeded(seed, (1, _NZ, _NY, _NX))
    if active:
        field[0, 0, 3:7, 4:9] = 48.5
        field[0, 1, 9:12, 2:5] = 44.25
    return np.asarray(field, dtype=np.float32)


def write_frame(path: Path, *, seed: int, active_objects: bool) -> None:
    """One WRF-shaped history frame with deterministic contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mass = ("Time", "bottom_top", "south_north", "west_east")
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("bottom_top", _NZ),
                ("bottom_top_stag", _NZ + 1), ("south_north", _NY),
                ("south_north_stag", _NY + 1), ("west_east", _NX),
                ("west_east_stag", _NX + 1)):
            dataset.createDimension(name, size)
        fields = {
            "U": (("Time", "bottom_top", "south_north", "west_east_stag"),
                  _seeded(seed + 1, (1, _NZ, _NY, _NX + 1))),
            "V": (("Time", "bottom_top", "south_north_stag", "west_east"),
                  _seeded(seed + 2, (1, _NZ, _NY + 1, _NX))),
            "W": (("Time", "bottom_top_stag", "south_north", "west_east"),
                  _seeded(seed + 3, (1, _NZ + 1, _NY, _NX))),
            "T": (mass, 300.0 + _seeded(seed + 4, (1, _NZ, _NY, _NX))),
            "PH": (("Time", "bottom_top_stag", "south_north", "west_east"),
                   _seeded(seed + 5, (1, _NZ + 1, _NY, _NX))),
            "MU": (("Time", "south_north", "west_east"),
                   _seeded(seed + 6, (1, _NY, _NX))),
            "QVAPOR": (mass,
                       0.01 + 0.001 * _seeded(seed + 7, (1, _NZ, _NY, _NX))),
            "REFL_10CM": (mass, _reflectivity(seed + 8, active=active_objects)),
        }
        for name, (dimensions, values) in fields.items():
            variable = dataset.createVariable(name, "f4", dimensions)
            variable[...] = values


def build_run(root: Path, name: str, *, seed: int, domains, cadence_seconds: int,
              duration_seconds: int, object_second: int) -> Path:
    """A run directory of frames on the registered domain/time ladder."""
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(FIXTURE_START)
    for domain_index, domain in enumerate(domains):
        for seconds in range(0, duration_seconds + 1, cadence_seconds):
            valid = start + timedelta(seconds=seconds)
            path = run / valid.strftime(f"wrfout_{domain}_%Y-%m-%d_%H_%M_%S")
            write_frame(
                path,
                seed=seed + 1000 * domain_index + seconds,
                active_objects=seconds >= object_second)
    return run


def build_pair(root: Path, *, domains, cadence_seconds: int,
               duration_seconds: int) -> tuple[Path, Path]:
    """The scored pair: the two runs differ in seed and in object timing."""
    left = build_run(
        root, "left", seed=11, domains=domains, cadence_seconds=cadence_seconds,
        duration_seconds=duration_seconds, object_second=cadence_seconds)
    right = build_run(
        root, "right", seed=907, domains=domains,
        cadence_seconds=cadence_seconds, duration_seconds=duration_seconds,
        object_second=2 * cadence_seconds)
    return left, right


def as_hex(scores) -> dict[str, str]:
    return {key: float(value).hex() for key, value in sorted(scores.items())}


def _n5s_scores() -> dict[str, str]:
    """Score the pair through ``n5s_metrics.score_run_pair`` as it stands."""
    from gpuwm.verify import n5s_metrics
    from gpuwm.verify.n5s_common import write_json

    registration = n5s_metrics.make_registration(
        run_minutes=30, history_minutes=10, commit="4" * 40,
        start_time=FIXTURE_START)
    cadence = int(registration["cadence_seconds"])
    duration = int(registration["run_duration_seconds"])
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        left, right = build_pair(
            root, domains=n5s_metrics.N5S_DOMAINS, cadence_seconds=cadence,
            duration_seconds=duration)
        for run in (left, right):
            write_json(run / "n5s-preregistration.json", registration)
            (run / "exit.status").write_text("0\n", encoding="utf-8")
            n5s_metrics._artifact(
                run, root, run_id=run.name, registration=registration)
        scores = n5s_metrics.score_run_pair(
            left, right, registration, registration)
    return as_hex(scores)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("usage: python -m tests.n5s_delegation_fixture <out.json>")
    payload = {
        "schema": FIXTURE_SCHEMA,
        "note": (
            "scores from gpuwm.verify.n5s_metrics.score_run_pair over the "
            "synthetic pair built by tests/n5s_delegation_fixture.py, "
            "recorded as float.hex() before the metric bodies moved to "
            "gpuwm/verify/field_metrics.py"),
        "start_time": FIXTURE_START,
        "run_minutes": 30,
        "history_minutes": 10,
        "scores_hex": _n5s_scores(),
    }
    target = Path(argv[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
