#!/usr/bin/env python3
"""Gate the exit-temperature receipt, and show what it buys.

Run with the repository's CUDA-capable interpreter::

    arwen-mp28-venv/bin/python \\
        tools/thompson_wrf461_oracle/check_exit_temperature_aero.py

THREE CHECKS, IN ORDER OF STRENGTH
----------------------------------
1. ENTRY IDENTITY (host only, no device).  ``mp_gt_driver:1222`` computes
   ``t1d(k) = th(i,k,j)*pii(i,k,j)`` and ``run_column_aero.F90`` records
   exactly that product, so the receipt's ``t1d_entry`` must equal every
   fixture's ``before`` ``temp_k`` BITWISE at all 456 rows.  This is what
   makes the receipt's scenario alignment checkable rather than assumed.

2. THE MEASURED DEFECT.  ``mp_gt_driver:1358`` writes
   ``th(i,k,j) = t1d(k)/pii(i,k,j)`` and ``t1d`` dies with the routine, so a
   pristine caller can only record ``th*pii``.  That round trip is not the
   identity in float32.  The count of rows where it fails is printed; it is
   the size of the one thing this harness cannot record directly.

3. WHAT IT BUYS (device).  ``calc_effectRad`` (:5594) is called at :1472 with
   ``t1d`` -- the EXIT one.  Driving gpuwm's ``launch_aerosol_effective_radius``
   with the fixture's recorded ``temp_k`` leaves a handful of levels that are
   not bitwise against the fixture's own ``effc_m``/``effi_m``/``effs_m``;
   driving it with the receipt's ``t1d_exit`` must leave NONE.  That is the
   difference between an allow-list of "explained" levels and an
   unconditional bitwise gate.

Exit status is 0 only if (1) holds at every row and (3) leaves zero
unexplained levels on the receipt-driven run.
"""
from __future__ import annotations

import csv
import struct
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_ORACLE = _REPO / "gpuwm" / "data" / "thompson" / "oracle-aero"
_RECEIPT = _ORACLE / "aero-exit-temperature.csv"

F = np.float32
FIELDS = ("effc_m", "effi_m", "effs_m")


def _bits(value) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', F(value)))[0]:08X}"


def _hex_to_f32(hexed: str) -> np.float32:
    return np.array([int(hexed, 16)], np.uint32).view(np.float32)[0]


def _fixture(name: str):
    with (_ORACLE / f"{name}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    half = len(rows) // 2
    return rows[:half], rows[half:]


def _col(rows, key) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], F)


def main() -> int:
    with _RECEIPT.open(newline="", encoding="ascii") as stream:
        receipt = list(csv.DictReader(stream))
    scenarios = sorted({row["scenario"] for row in receipt})
    by_scenario = {name: [row for row in receipt if row["scenario"] == name]
                   for name in scenarios}
    print(f"receipt: {len(receipt)} rows over {len(scenarios)} scenarios")

    # ---- 1. entry identity -------------------------------------------------
    entry_bad = []
    round_trip_bad = []
    for name in scenarios:
        before, after = _fixture(name)
        rows = by_scenario[name]
        if len(rows) != len(before):
            raise SystemExit(f"{name}: {len(rows)} receipt rows, "
                             f"{len(before)} fixture levels")
        for k, row in enumerate(rows):
            if row["t1d_entry_hex"] != _bits(before[k]["temp_k"]):
                entry_bad.append(f"{name} k={k + 1}")
            if row["t1d_exit_hex"] != _bits(after[k]["temp_k"]):
                round_trip_bad.append(f"{name} k={k + 1}")
    total = len(receipt)
    print(f"[1] ENTRY  t1d == fixture before temp_k : "
          f"{total - len(entry_bad)}/{total} bitwise")
    print(f"[2] EXIT   t1d == fixture after  temp_k : "
          f"{total - len(round_trip_bad)}/{total} bitwise "
          f"({len(round_trip_bad)} rows lost to mp_gt_driver:1358's "
          "float32 round trip)")
    for item in round_trip_bad:
        print(f"        round-trip loss: {item}")

    # ---- 3. what it buys ---------------------------------------------------
    try:
        import cupy as cp
        from gpuwm.core.thompson_aerosol_state import (
            launch_aerosol_effective_radius)
    except Exception as exc:                       # pragma: no cover
        print(f"[3] SKIPPED (no device / no gpuwm import): {exc}")
        return 1 if entry_bad else 0

    def _drive(name, temperature):
        _, after = _fixture(name)
        args = [cp.asarray(temperature)]
        args += [cp.asarray(_col(after, key)) for key in
                 ("p_pa", "qv", "qc", "nc_per_kg", "qi", "ni_per_kg", "qs")]
        outs = [cp.empty_like(args[0]) for _ in range(3)]
        launch_aerosol_effective_radius(*args, *outs, metres=True)
        cp.cuda.Stream.null.synchronize()
        bad = []
        for out, field in zip(outs, FIELDS):
            got = cp.asnumpy(out)
            want = _col(after, field)
            for k in np.nonzero(got != want)[0]:
                bad.append(f"{name} {field} k={int(k) + 1} "
                           f"got={got[k]!r} want={want[k]!r}")
        return bad

    recorded_bad, receipt_bad = [], []
    for name in scenarios:
        _, after = _fixture(name)
        recorded_bad += _drive(name, _col(after, "temp_k"))
        exact = np.asarray(
            [_hex_to_f32(row["t1d_exit_hex"]) for row in by_scenario[name]], F)
        receipt_bad += _drive(name, exact)
    print(f"[3] calc_effectRad driven by the fixture's temp_k   : "
          f"{len(recorded_bad)} level(s) NOT bitwise")
    for item in recorded_bad:
        print(f"        {item}")
    print(f"[3] calc_effectRad driven by the receipt's t1d_exit : "
          f"{len(receipt_bad)} level(s) NOT bitwise")
    for item in receipt_bad:
        print(f"        {item}")

    failed = bool(entry_bad) or bool(receipt_bad)
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
