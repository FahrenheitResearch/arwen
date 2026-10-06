#!/usr/bin/env python3
"""Which cell does the positive-definite limiter's low-order VERTICAL flux take?

A native measurement, CPU only: the compiled HRRR fork routine
(NOAA-EMC/HRRR 40ee6058c WRFV3.9 ``advect_scalar_pd``) and the compiled WRF
4.7.1 routine read the same input words.  The input is one fixture case
with the horizontal mass fluxes set to zero, a uniform Omega of one sign at
a vertical Courant number of 0.2, and a scalar that is 1 on one level and 0
everywhere else, so the limiter engages around the occupied level and the
answer shows the low-order flux itself.

What the fork's text says (``module_advect_em.F`` vert_order 5 arm,
F:8378-8501, the same in every vert_order arm): for a face Courant number
of magnitude at most 1 the low-order flux is

    fqzl(k) = max(vel,0.)*field_old(k-1) + min(vel,0.)*field_old(k)

where WRF 4.7.1 forms ``mu*(dz/dt)*flux_upwind(field_old(k-1), field_old(k), cr)``
with ``cr = vel*dt/dz/mu``.  ``dz`` is negative in eta (rdzw < 0), so ``cr``
has the sign opposite to ``vel``: 4.7.1 takes ``field_old(k)`` for
``vel > 0`` (Omega positive is downward, so the cell ABOVE the face is the
upwind one) and the fork takes ``field_old(k-1)``, the cell below.  The two
therefore agree only where the limiter does not engage (the total is then
the high-order flux, whatever the low-order one was).  This probe records
what the two compiled routines do with that difference.

usage: pd_low_order_probe.py FORK_RUN_ADVECT WRF471_RUN_ADVECT SCRATCH RECEIPT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from gpuwm.verify.advect_oracle import (ADVECT_ORACLE_DIR, load_advect_cases,
                                        read_fortran_output, write_fortran_input)

CASE = "real_interior"
LEVEL = 20              # 0-based mass level holding the scalar
COURANT = 0.2
COLUMN = (3, 3)         # (j, i) of the reported column; every column is the same


def _probe_case(base, sign):
    a = {key: value.copy() for key, value in base.inputs.items()}
    nz = base.shape[0]
    a["ru"][:] = 0.0
    a["rv"][:] = 0.0
    rdnw = a["rdnw"]
    dz = 2.0 / (float(rdnw[LEVEL]) + float(rdnw[LEVEL - 1]))          # < 0
    mu = float(a["muts"].mean())
    vel = np.float32(sign * COURANT * abs(dz) * mu / base.metadata["dt"])
    a["rw"][:] = 0.0
    a["rw"][1:nz] = vel
    q0 = np.zeros_like(a["q0"])
    q0[LEVEL] = 1.0
    a["q0"] = q0
    a["scalar_pd"] = q0.copy()
    a["tend_pd"] = np.zeros_like(a["tend_pd"])
    return replace(base, name=f"pd-low-order-probe-{'down' if sign > 0 else 'up'}",
                   inputs=a, reference={}), float(vel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fork", type=Path)
    parser.add_argument("wrf471", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    legacy = ADVECT_ORACLE_DIR.parent / "wrf_legacy_advect"
    base = next(case for case in load_advect_cases(legacy) if case.name == CASE)
    j, i = COLUMN
    rows = []
    for sign in (1.0, -1.0):
        case, vel = _probe_case(base, sign)
        columns = {}
        digest = None
        for tag, executable in (("fork", args.fork), ("wrf471", args.wrf471)):
            source = args.scratch / f"{case.name}-{tag}.in.bin"
            output = args.scratch / f"{case.name}-{tag}.out.bin"
            write_fortran_input(case, "advect_scalar_pd", source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            subprocess.run([str(executable), str(source), str(output)], check=True)
            arrays = read_fortran_output(case, output, diagnostic=True)
            columns[tag] = arrays["z_tendency"][:, 4 + j, 4 + i].astype(np.float64)
        levels = list(range(LEVEL - 3, LEVEL + 4))
        nz = base.shape[0]
        dnw = 1.0 / base.inputs["rdnw"][:nz].astype(np.float64)
        # The upwind neighbour receives the scalar; the other one is empty
        # and upstream, so any tendency there is taken from nothing.
        upstream = LEVEL + 1 if sign > 0 else LEVEL - 1
        rows.append({
            "omega_sign": "positive (downward)" if sign > 0 else "negative (upward)",
            "omega": vel, "courant": COURANT, "occupied_level": LEVEL,
            "input_sha256": digest, "levels": levels,
            "fork_z_tendency": [float(columns["fork"][k]) for k in levels],
            "wrf471_z_tendency": [float(columns["wrf471"][k]) for k in levels],
            "empty_upstream_level": upstream,
            "fork_tendency_in_the_empty_upstream_cell": float(columns["fork"][upstream]),
            "wrf471_tendency_in_the_empty_upstream_cell": float(columns["wrf471"][upstream]),
            # Flux form on both sides: sum(tendency * dnw) is the lid flux
            # minus the surface flux, zero here.  The fork's negative
            # tendency is a redistribution out of the empty cell, not a leak.
            "fork_mass_weighted_column_sum": float((columns["fork"][:nz] * dnw).sum()),
            "wrf471_mass_weighted_column_sum": float((columns["wrf471"][:nz] * dnw).sum()),
        })
    receipt = {
        "what": ("advect_scalar_pd, vert_order 5, zero horizontal flux, uniform Omega at a vertical "
                 "Courant number of 0.2, scalar 1 on one level and 0 elsewhere; z_tendency of one column"),
        "case": CASE, "fixture": "tests/data/wrf_legacy_advect",
        "fork": "NOAA-EMC/HRRR 40ee6058c WRFV3.9 module_advect_em.F (tools/advect_wrf_legacy_oracle/build.sh)",
        "wrf471": "WRF 4.7.1 module_advect_em.F (tools/advect_wrf471_oracle/build.sh)",
        "rows": rows,
    }
    for row in rows:
        fork = row["fork_tendency_in_the_empty_upstream_cell"]
        modern = row["wrf471_tendency_in_the_empty_upstream_cell"]
        row["fork_drains_the_empty_upstream_cell"] = bool(fork < 0.0)
        row["wrf471_drains_the_empty_upstream_cell"] = bool(modern < 0.0)
    args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="ascii")
    for row in rows:
        print(row["omega_sign"], "fork", row["fork_z_tendency"], "wrf471", row["wrf471_z_tendency"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
