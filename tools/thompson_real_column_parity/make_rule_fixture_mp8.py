#!/usr/bin/env python3
"""Hand-built rule columns for classic Thompson (mp_physics=8) with WRF
v4.6.1's own answers.

``rule_checks.py`` builds columns that hold the cells saved states rarely
hold: a column with no microphysics, vapour at exactly zero, ice above 0 C
with no number or far too many crystals, snow and graupel present only as a
mixing ratio in thin air, melting snow under rain.  For mp=28 each check
compares the port with what the rule says.  For mp=8 the production unit
(gpuwm/core/kernels/thompson_classic.cu) is a transcription of WRF itself,
so the columns go through WRF's own classic driver (``build_wrf.sh``'s
``run_columns_classic``) instead and the unit is held word for word to its
answers (tests/test_thompson_real_column_host_parity.py,
``mp8_wrf_ulp.py``).  This writes that fixture.

Columns (12 levels of 500 m, dt 5 s, ``rule_checks.columns``):
  0  G: no condensate above R1, top-level vapour zero, orphans at or
        below R1 (WRF leaves at :2020);
  1  G: the same plus 1e-4 kg/kg of cloud at level 0 (it has microphysics:
        vapour floored at 1.E-10, :3974);
  2  P: 97 percent humidity, snow 1.1e-12 kg/kg at level 7 and graupel
        1.1e-12 at level 6 (mixing ratio above R1, concentration below);
  3  A-warm: ice 2e-5 kg/kg with no number at level 5, 1e-5 with 1e9 per
        kg at level 4, both above 0 C (:3033-3055);
  4  I: snow 2e-6 kg/kg melting at level 6, no rain;
  5  I: the same with 2e-4 kg/kg of rain at level 9;
  6  J/T: cloud 1e-4 at level 0, cloud and ice of 2e-12 each at level 3
        and graupel 2e-12 at level 4, above 0 C;
  7  R: rain 1e-4 kg/kg at 50 percent humidity under rain 5e-13 at level 6.

usage: make_rule_fixture_mp8.py WRF_BUILD_DIR OUT.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import real_column_parity as R  # noqa: E402
import rule_checks as C  # noqa: E402

f32 = np.float32
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")


def rule_columns():
    cols = C.columns(8)
    top = cols["qv"].shape[1] - 1
    # 0, 1: G
    for c in (0, 1):
        cols["qv"][c, top] = 0.0
        cols["qc"][c, 2] = 5.0e-13
        cols["qi"][c, 9] = 5.0e-13
        cols["ni"][c, 9] = 100.0
    cols["qc"][1, 0] = 1.0e-4
    # 2: P, at 97 percent humidity
    humid = C.columns(1, rh=0.97)
    cols["qv"][2] = humid["qv"][0]
    cols["qs"][2, 7] = 1.1e-12
    cols["qg"][2, 6] = 1.1e-12
    # 3: A-warm
    cols["qi"][3, 5] = 2.0e-5
    cols["ni"][3, 5] = 0.0
    cols["qi"][3, 4] = 1.0e-5
    cols["ni"][3, 4] = 1.0e9
    # 4, 5: I
    for c in (4, 5):
        cols["qs"][c, 6] = 2.0e-6
    cols["qr"][5, 9] = 2.0e-4
    cols["nr"][5, 9] = 2.0e3
    # 6: J/T
    cols["qc"][6, 0] = 1.0e-4
    cols["qc"][6, 3] = 2.0e-12
    cols["qi"][6, 3] = 2.0e-12
    cols["ni"][6, 3] = 1.0
    cols["qg"][6, 4] = 2.0e-12
    # 7: R
    cols["qr"][7, 2] = 1.0e-4
    cols["nr"][7, 2] = 1.0e4
    cols["qr"][7, 6] = 5.0e-13
    cols["nr"][7, 6] = 10.0
    return cols


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    build, out = Path(argv[1]), Path(argv[2])
    cols = rule_columns()
    dt = float(cols["dt"])
    inp = R.prepare(cols, 8)
    ncol, nz = inp["p"].shape
    run_dir = build / "run"
    stem = "rule-mp8"
    R.write_wrf_input(run_dir / f"{stem}-in.bin", inp, dt, 8)
    R.run_wrf(build / "pristine" / R.WRF_BINARY[8], run_dir,
              run_dir / f"{stem}-in.bin", run_dir / f"{stem}-out.bin")
    wrf = R.read_wrf_output(run_dir / f"{stem}-out.bin", ncol, nz, 8)
    for suffix in ("-in.bin", "-out.bin"):
        (run_dir / f"{stem}{suffix}").unlink(missing_ok=True)
    arrays = {f"col_{k}": np.asarray(cols[k], f32)
              for k in ("p", "th", "geop", "w", *SPECIES)}
    arrays.update({f"wrf_{k}": np.asarray(v, f32) for k, v in wrf.items()})
    arrays["dt"] = f32(dt)
    arrays["mp_physics"] = np.int32(8)
    arrays["wrf_build_receipt"] = np.array(
        (build / "BUILD-RECEIPT.txt").read_text(encoding="utf-8"))
    np.savez_compressed(out, **arrays)
    print(out, ncol, "columns,", nz, "levels, mp_physics 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
