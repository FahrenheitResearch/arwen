"""Measure gpuwm's RUC soil ingest against the WRF v4.6.1 oracle.

Usage::

    python tools/ruc_soil_ingest_wrf461_oracle/validate_ruc_soil_ingest_oracle.py \
        [gpuwm/data/ruc/oracle/soil_ingest.csv]

Prints per-experiment max ULP for TSLB, SMOIS and TSK, and exits non-zero if
any parity experiment is not bit-identical.  The two experiments the ingest
deliberately refuses are reported as such rather than scored; the test suite
in ``tests/test_ruc_soil_ingest.py`` is what binds those refusals.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpuwm.ingest.ruc_soil import (  # noqa: E402
    remap_soil_to_ruc_levels, ruc_soil_depths,
)

NSOIL = 9
#: Rows WRF produces but gpuwm refuses; see gpuwm/ingest/ruc_soil.py.
REFUSED_EXPERIMENTS = ("era5_layers_reversed_adj",)
REFUSED_COLUMNS = {("era5_layers_adj_nosst", 10), ("noah_layers_adj_nosst", 10)}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


def ulp_distance(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    a = got.view(np.int32).astype(np.int64)
    b = want.view(np.int32).astype(np.int64)
    a = np.where(a < 0, np.int64(np.iinfo(np.int32).min) - a, a)
    b = np.where(b < 0, np.int64(np.iinfo(np.int32).min) - b, b)
    return np.abs(a - b)


def main(argv: list[str]) -> int:
    default = (Path(__file__).resolve().parents[2]
               / "gpuwm" / "data" / "ruc" / "oracle" / "soil_ingest.csv")
    path = Path(argv[1]) if len(argv) > 1 else default
    rows = load_rows(path)

    worst: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tslb": 0, "smois": 0, "tsk": 0, "rows": 0})
    refused: dict[str, int] = defaultdict(int)
    failures: list[str] = []

    for row in rows:
        experiment = row["experiment"]
        column = int(row["col"])
        nlev = int(row["nlev"])
        levels = np.array(
            [int(row[f"lev_cm_{k}"]) for k in range(1, nlev + 1)], dtype=np.int64)
        st = np.array(
            [np.float32(row[f"st_src_{k}"]) for k in range(1, nlev + 1)],
            dtype=np.float32)[:, None]
        sm = np.array(
            [np.float32(row[f"sm_src_{k}"]) for k in range(1, nlev + 1)],
            dtype=np.float32)[:, None]
        tsk = np.array([np.float32(row["tsk_in"])], dtype=np.float32)
        tmn = np.array([np.float32(row["tmn"])], dtype=np.float32)
        sst = np.array([np.float32(row["sst"])], dtype=np.float32)
        mask = np.array([np.float32(row["landmask"])], dtype=np.float32)

        geometry = "levels" if row["flag_soil_levels"] == "1" else "layers"
        kwargs = dict(
            source_temperature=st,
            source_moisture=sm,
            source_levels_cm=levels,
            source_geometry=geometry,
            skin_temperature=tsk,
            deep_temperature=tmn,
            landmask=mask,
            sea_surface_temperature=sst if row["flag_sst"] == "1" else None,
            num_soil_layers=NSOIL,
            moisture_adjustment=(
                row["flag_sm_adj"] == "1" and geometry == "layers"),
        )

        expected_refusal = (
            experiment in REFUSED_EXPERIMENTS
            or (experiment, column) in REFUSED_COLUMNS
        )
        try:
            result = remap_soil_to_ruc_levels(**kwargs)
        except ValueError as error:
            if expected_refusal:
                refused[experiment] += 1
                continue
            failures.append(f"{experiment} col {column}: refused -- {error}")
            continue
        if expected_refusal:
            failures.append(
                f"{experiment} col {column}: expected a refusal, got numbers")
            continue

        want_t = np.array(
            [np.float32(row[f"tslb_{k}"]) for k in range(1, NSOIL + 1)],
            dtype=np.float32)
        want_m = np.array(
            [np.float32(row[f"smois_{k}"]) for k in range(1, NSOIL + 1)],
            dtype=np.float32)
        want_tsk = np.float32(row["tsk_out"])

        stats = worst[experiment]
        stats["rows"] += 1
        stats["tslb"] = max(
            stats["tslb"],
            int(ulp_distance(result.soil_temperature[:, 0], want_t).max()))
        stats["smois"] = max(
            stats["smois"],
            int(ulp_distance(result.soil_moisture[:, 0], want_m).max()))
        stats["tsk"] = max(
            stats["tsk"],
            int(ulp_distance(result.skin_temperature[0], want_tsk).max()))

    zs, dzs = ruc_soil_depths(NSOIL)
    print(f"zs  = {[float(v) for v in zs]}")
    print(f"dzs = {[float(v) for v in dzs]}  sum={float(dzs.sum())!r}")
    print()
    print(f"{'experiment':32s} {'rows':>5s} {'tslb':>6s} {'smois':>6s} {'tsk':>5s}")
    status = 0
    for experiment in sorted(worst):
        stats = worst[experiment]
        print(f"{experiment:32s} {stats['rows']:5d} {stats['tslb']:6d} "
              f"{stats['smois']:6d} {stats['tsk']:5d}")
        if max(stats["tslb"], stats["smois"], stats["tsk"]) != 0:
            status = 1
    for experiment in sorted(refused):
        print(f"{experiment:32s} {refused[experiment]:5d}  (refused by design)")
    for failure in failures:
        print(f"FAIL {failure}")
        status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
