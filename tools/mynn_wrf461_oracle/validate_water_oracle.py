"""Assert the gpuwm CPU reference against the over-water ISFTCFLX oracles.

Run under ``PYTHONPATH=<repo root>`` so ``gpuwm.core.mynn_surface`` imports.
Reports max_abs / max_rel / max_ulp per output and fails closed on a parity
regression or a dead branch.

The two gates are different on purpose:

* the leaf sweep is gated at ``max_ulp 0`` with no table at all, because the
  water roughness leaves route their transcendentals through
  ``gpuwm.core.noahmp_libm`` and therefore return the glibc 2.39 words
  gfortran linked against, on every platform;
* the column sweep is gated by the measured three-platform union that
  ``tests/test_mynn_surface_water.py`` carries, imported from there rather
  than copied, so the build and the suite can never drift apart.

Branch liveness is checked against the fixture's own numbers: if a rebuilt
sweep stops binding an arm -- an ISFTCFLX identity that no longer moves ZNT, a
clamp no sample reaches, ``garratt_1992``'s land arm going unreached -- this
fails instead of passing with the arm dead.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.mynn_surface import (
    _charnock_1955,
    _davis_etal_2008,
    _edson_etal_2013,
    _fairall_etal_2003,
    _fairall_etal_2014,
    _garratt_1992,
    _taylor_yelland_2001,
    mynn_surface_layer_default,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from test_mynn_surface_water import (  # noqa: E402
    CASES,
    INPUT_ALIASES,
    INPUT_NAMES,
    ISFTCFLX_SWEEP,
    OUTPUT_NAMES,
    STAGES,
    WATER_ULP,
    Z0_CEILING,
    Z0_FLOOR,
    ZTQ_FLOOR,
    ZTQ_GARRATT_CEILING,
)

F = np.float32
LEAVES = (
    "charnock_1955", "edson_etal_2013", "davis_etal_2008",
    "taylor_yelland_2001", "fairall_etal_2003", "fairall_etal_2014",
    "garratt_1992",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f32(rows, name) -> np.ndarray:
    return np.asarray([F(row[name]) for row in rows], dtype=F)


def _receipts(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    scale = np.maximum(np.abs(expected.astype(np.float64)), 1e-30)
    return (float(diff.max()), float((diff / scale).max()),
            int(fp32_ulp_distance(actual, expected).max()))


def _leaf_value(leaf, row):
    ustar, wsp10 = F(row["ustar"]), F(row["wsp10"])
    visc, zu = F(row["visc"]), F(row["zu"])
    ren, z0_in, landsea = F(row["ren"]), F(row["z0_in"]), F(row["landsea"])
    if leaf == "charnock_1955":
        return {"z0_out": _charnock_1955(ustar, wsp10, visc, zu)}
    if leaf == "edson_etal_2013":
        return {"z0_out": _edson_etal_2013(ustar, wsp10, visc, zu)}
    if leaf == "davis_etal_2008":
        return {"z0_out": _davis_etal_2008(ustar)}
    if leaf == "taylor_yelland_2001":
        return {"z0_out": _taylor_yelland_2001(wsp10)}
    if leaf == "fairall_etal_2003":
        zt, zq = _fairall_etal_2003(ren)
    elif leaf == "fairall_etal_2014":
        zt, zq = _fairall_etal_2014(ren)
    elif leaf == "garratt_1992":
        zt, zq = _garratt_1992(z0_in, ren, landsea)
    else:
        raise SystemExit(f"unknown leaf in fixture: {leaf}")
    return {"zt_out": zt, "zq_out": zq}


def _validate_leaves(rows, failures) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["leaf"], []).append(row)
    if sorted(grouped) != sorted(LEAVES):
        failures.append(f"leaf fixture covers {sorted(grouped)}, want "
                        f"{sorted(LEAVES)}")
        return
    for leaf, leaf_rows in sorted(grouped.items()):
        worst = 0
        for row in leaf_rows:
            for name, value in _leaf_value(leaf, row).items():
                got = np.asarray([value], dtype=F)
                want = np.asarray([F(row[name])], dtype=F)
                worst = max(worst, int(fp32_ulp_distance(got, want)[0]))
                if worst:
                    failures.append(
                        f"{leaf} sample {row['sample']} {name}: "
                        f"{float(value)!r} != {float(F(row[name]))!r}")
        print(f"  leaf {leaf:<22} {len(leaf_rows):>3} rows  max_ulp {worst}")

    # Branch liveness inside the leaves, read off the oracle's own outputs.
    def z0(name):
        return _f32(grouped[name], "z0_out")

    ustar = _f32(grouped["davis_etal_2008"], "ustar")
    checks = {
        "davis ZW saturated (u* >= 1.06)": int((ustar >= 1.06).sum()),
        "davis ZW unsaturated": int((ustar < 1.06).sum()),
        "davis z0 ceiling": int((z0("davis_etal_2008") == Z0_CEILING).sum()),
        "taylor z0 ceiling": int(
            (z0("taylor_yelland_2001") == Z0_CEILING).sum()),
        "taylor z0 floor": int((z0("taylor_yelland_2001") == Z0_FLOOR).sum()),
        "taylor wsp10 below 0.1": int(
            (_f32(grouped["taylor_yelland_2001"], "wsp10") < 0.1).sum()),
        "charnock z0 ceiling": int((z0("charnock_1955") == Z0_CEILING).sum()),
        "edson z0 ceiling": int((z0("edson_etal_2013") == Z0_CEILING).sum()),
        "edson u* below its 0.07 floor": int((ustar < 0.07).sum()),
    }
    water = [r for r in grouped["garratt_1992"] if F(r["landsea"]) > F(1.5)]
    land = [r for r in grouped["garratt_1992"] if F(r["landsea"]) <= F(1.5)]
    zt_water = _f32(water, "zt_out")
    checks["garratt water arm"] = len(water)
    checks["garratt land arm"] = len(land)
    checks["garratt zt ceiling"] = int((zt_water == ZTQ_GARRATT_CEILING).sum())
    checks["garratt zt floor"] = int((zt_water == ZTQ_FLOOR).sum())
    checks["garratt zt != zq (water only)"] = int(
        (zt_water != _f32(water, "zq_out")).sum())
    ren = _f32(grouped["fairall_etal_2003"], "ren")
    checks["fairall Ren <= 2"] = int((ren <= 2.0).sum())
    checks["fairall Ren > 2"] = int((ren > 2.0).sum())
    for label, count in sorted(checks.items()):
        print(f"  arm {label:<32} bound by {count:>3} samples")
        if count < 1:
            failures.append(f"leaf arm went dead: {label}")


def _validate_columns(rows, failures) -> None:
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["isftcflx"]), int(row["itimestep"]), int(row["isfflx"]))
        groups.setdefault(key, []).append(row)
    want_keys = sorted((flx, step, flux)
                       for flx in ISFTCFLX_SWEEP for step, flux in STAGES)
    if sorted(groups) != want_keys:
        failures.append(f"column fixture stages {sorted(groups)} != "
                        f"{want_keys}")
        return
    for key in want_keys:
        stage_rows = groups[key]
        if tuple(row["case"] for row in stage_rows) != CASES:
            failures.append(f"{key}: cases {[r['case'] for r in stage_rows]}")
            continue
        values = {name: _f32(stage_rows, INPUT_ALIASES.get(name, name))
                  for name in INPUT_NAMES}
        actual = mynn_surface_layer_default(
            values, dx=float(F(stage_rows[0]["dx"])), itimestep=key[1],
            isfflx=key[2], isftcflx=key[0],
            mol=_f32(stage_rows, "mol_input"),
            ustm=_f32(stage_rows, "ustm_input"))
        worst = (0.0, 0.0, 0)
        for name in OUTPUT_NAMES:
            expected = _f32(stage_rows, name)
            residue = fp32_ulp_distance(actual[name], expected)
            budget = np.asarray(
                WATER_ULP.get(key, {}).get(name, (0,) * len(stage_rows)),
                dtype=np.int64)
            over = np.nonzero(residue > budget)[0]
            for index in over:
                failures.append(
                    f"isftcflx={key[0]} step={key[1]} isfflx={key[2]} {name} "
                    f"{CASES[index]}: {int(residue[index])} ULP over a budget "
                    f"of {int(budget[index])}")
            receipts = _receipts(actual[name], expected)
            worst = tuple(max(a, b) for a, b in zip(worst, receipts))
        print(f"  columns isftcflx={key[0]} step={key[1]} isfflx={key[2]}  "
              f"max_abs {worst[0]:.3e}  max_rel {worst[1]:.3e}  "
              f"max_ulp {worst[2]}")

    # Branch liveness across the sweep.
    znt = {flx: {row["case"]: F(row["znt"])
                 for row in groups[(flx, 1, 1)]} for flx in ISFTCFLX_SWEEP}
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 3), (2, 3)):
        moved = [c for c in CASES if znt[a][c] != znt[b][c]]
        print(f"  ZNT moves on {len(moved):>2} columns between isftcflx "
              f"{a} and {b}")
        if len(moved) < 5:
            failures.append(f"isftcflx {a} and {b} are indistinguishable")
    if any(znt[1][c] != znt[2][c] for c in CASES):
        failures.append("isftcflx 1 and 2 must share davis_etal_2008 z0")
    base = {row["case"]: row for row in groups[(0, 1, 1)]}
    for flx in (1, 2, 3):
        rows_by_case = {row["case"]: row for row in groups[(flx, 1, 1)]}
        for case in ("control_land", "control_snow_land"):
            moved = [name for name in OUTPUT_NAMES
                     if F(rows_by_case[case][name]) != F(base[case][name])]
            if moved:
                failures.append(f"land control {case} moved with "
                                f"isftcflx={flx}: {moved}")
    for key, stage_rows in groups.items():
        row = next(r for r in stage_rows if r["case"] == "xland_exactly_1p5")
        if F(row["hfx"]) != F(row["hfx_input"]):
            failures.append(f"{key}: XLAND=1.5 HFX was rewritten")
        if F(row["znt"]) == F(row["znt_input"]):
            failures.append(f"{key}: XLAND=1.5 did not take a water ZNT")
    print("  XLAND=1.5 keeps its HFX and takes a water ZNT in all "
          f"{len(groups)} stages")


def validate(column_path: Path, leaf_path: Path) -> None:
    failures: list[str] = []
    print(f"{leaf_path}:")
    _validate_leaves(_read(leaf_path), failures)
    print(f"{column_path}:")
    _validate_columns(_read(column_path), failures)
    if failures:
        for line in failures:
            print(f"FAIL {line}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} water-oracle failures")
    print("water oracle: CPU reference matches the unmodified WRF module")


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        raise SystemExit(
            "usage: validate_water_oracle.py COLUMNS.csv LEAVES.csv")
    validate(Path(argv[1]), Path(argv[2]))


if __name__ == "__main__":
    main(sys.argv)
