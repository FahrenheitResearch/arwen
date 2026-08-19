#!/usr/bin/env python3
"""Measure gpuwm's device kernels against the FULL regenerated Fortran probes.

``check_probe_oracles_aero.py`` proves the literals embedded in the two GPU
test files are reproduced by the committed Fortran probes.  This script does
the other half: it runs the gpuwm probe kernels over EVERY row of the
regenerated oracles -- 12348 warm rate rows, 11025 balance rows and 11340
sub-freezing rows -- and prints the maximum relative difference per field.

Those maxima are what the two test docstrings quote as their justification for
each tolerance, and they are exactly the numbers an auditor cannot otherwise
re-derive, because the committed test tables are only a stratified subset.

USAGE
-----
    python3 measure_probe_oracles_gpu_aero.py PROBE_OUTPUT_DIR [TABLE_ROOT]

TABLE_ROOT defaults to this install's packaged classic-table root
(``gpuwm.physics_compat.packaged_thompson_table_root``, which resolves into
the ``gpuwm-data`` companion distribution) and must hold the four
classic assets, including the 255 MB freezeH2O.dat, because
``probe_cold_warm_loop`` takes the real rain/cloud collection efficiency
table.  Requires a CUDA device.

This script asserts nothing.  It reports.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


_REPO = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {name: np.asarray([float(row[name]) for row in rows],
                             dtype=np.float64)
            for name in rows[0]}


def _relative(got: np.ndarray, expected: np.ndarray) -> float:
    """Max |got-expected|/|expected| over rows whose reference is non-trivial."""
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    if scale == 0.0:
        return 0.0 if np.array_equal(got, expected) else float("inf")
    mask = np.abs(expected) > 1.0e-12 * scale
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(got[mask] - expected[mask])
                        / np.abs(expected[mask])))


def _report(title: str, pairs: list[tuple[str, np.ndarray, np.ndarray]]):
    print(title)
    for name, got, expected in pairs:
        exact = np.array_equal(np.asarray(got, dtype=np.float64),
                               np.asarray(expected, dtype=np.float64))
        value = _relative(got, expected)
        flag = "  EXACT" if exact else ""
        print(f"    {name:<14s} {value:.4e}{flag}")
    print()


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 2
    probe_dir = Path(argv[1]).resolve()
    sys.path.insert(0, str(_REPO))
    if len(argv) == 3:
        table_root = Path(argv[2]).resolve()
    else:
        # Resolved, never joined: since 2.5.0 this directory ships in the
        # gpuwm-data companion distribution, and only gpuwm.data_assets
        # knows where that landed.
        from gpuwm.physics_compat import packaged_thompson_table_root
        table_root = packaged_thompson_table_root().resolve()

    import cupy as cp

    from gpuwm.core.thompson_aerosol_cold import probe_cold_warm_loop
    from gpuwm.core.thompson_aerosol_warm import (
        launch_ncten_balance, probe_warm_rates)
    from gpuwm.core.thompson_contract import (
        AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS,
        read_sequential_records)

    # WRF's t_Efrw(nbr,nbc), from the canonical packaged auxiliary record --
    # the same array gpuwm.core.thompson_runtime uploads for mp=8.
    records = read_sequential_records(
        table_root / AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS)
    efficiency = cp.asarray(np.asfortranarray(records["t_Efrw"]))

    # ---- cold: the sub-freezing half of WRF's warm-rain loop -------------
    cold = _load(probe_dir / "aero-cold-warm-loop.csv")
    dt_cold = float(cold["dt"][0])

    def device(table, name):
        return cp.asarray(table[name].astype(np.float32).copy())

    out = probe_cold_warm_loop(
        device(cold, "qc"), device(cold, "nc_per_kg"),
        device(cold, "qr"), device(cold, "nr_per_kg"),
        device(cold, "nwfa_per_kg"), device(cold, "nifa_per_kg"),
        device(cold, "temp_k"), device(cold, "p_pa"), device(cold, "qv"),
        efficiency, dt_cold)
    cp.cuda.Stream.null.synchronize()
    got = {key: cp.asnumpy(value) for key, value in out.items()}
    _report(
        f"cold-warm loop, {cold['qc'].size} rows, dt = {dt_cold} s:",
        [(name, got[name], cold[name]) for name in (
            "nu_c_entry", "nu_c_working", "nc_m3", "mvd_c", "mvd_r",
            "prr_wau", "pnr_wau", "pnc_wau", "pnc_rcw", "pna_rca",
            "pnd_rcd")])

    # ---- warm rates -------------------------------------------------------
    warm = _load(probe_dir / "aero-warm-rates.csv")
    dt_warm = float(warm["dt"][0])
    out = probe_warm_rates(
        device(warm, "pres"), device(warm, "temp"), device(warm, "qv"),
        device(warm, "qc"), device(warm, "nc_per_kg"),
        device(warm, "qr"), device(warm, "nr_per_kg"),
        device(warm, "nwfa_per_kg"), device(warm, "nifa_per_kg"),
        efficiency, dt_warm)
    cp.cuda.Stream.null.synchronize()
    got = {key: cp.asnumpy(value) for key, value in out.items()}
    _report(
        f"warm rates, {warm['qc'].size} rows, dt = {dt_warm} s:",
        [(name, got[name], warm[name]) for name in sorted(got)
         if name in warm])

    # ---- ncten balance limiter -------------------------------------------
    balance = _load(probe_dir / "aero-ncten-balance.csv")
    dt_balance = float(balance["dt"][0])
    ncten = device(balance, "ncten_in")
    launch_ncten_balance(
        device(balance, "qc_entry"), device(balance, "qc_after"),
        device(balance, "nc_per_kg"), device(balance, "rho"),
        ncten, dt_balance)
    cp.cuda.Stream.null.synchronize()
    _report(
        f"ncten balance, {balance['qc_entry'].size} rows, "
        f"dt = {dt_balance} s:",
        [("ncten_out", cp.asnumpy(ncten), balance["ncten_out"])])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
