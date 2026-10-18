"""P3 on the card against WRF 4.6.1's Fortran, every word of every frame.

The column oracle (tools/p3_wrf461_oracle, README there) runs WRF's
unmodified ``module_mp_p3.F`` (sha256 716950a3...f6a3c6) through
``mp_p3_wrapper_wrf`` with the mp_physics=50 call shape, compiled
``-O0 -ffp-contract=off`` and linked with correctly rounded ``tgammaf`` and
``log10f`` (what WRF computes on a C library that rounds those correctly).
These gates replay committed slices of it through ``run_p3_device`` -- the
call ``gpuwm.core.p3.apply`` makes -- and require the float32 bit patterns
to be EQUAL, in the arithmetic the process was started with (the shipped
default, or GPUWM_WRF_EXACT=1).

Breakage each prevents, measured before the fixes this file landed with:

* rain tables generated with NumPy's float32 libm moved rain mass and
  number by up to 6e-3 relative in one step in 59 of 64 regime columns
  (``R01``), and 829 ULP on the 40-step warm-rain case (``F12``);
* the subnormal flush CuPy's RawModule route imposes left the snow column
  of ``S30`` at step 4, 19 ULP in the ice diagnostics by step 28;
* the twelve p3-fortref fixtures plus F13 (evidence/p3-fortran-oracle-
  20260828, evidence/p3-cuda-verify-20260829), whose committed arm-A dumps
  are that same oracle, carried 1-7 ULP and two bifurcations (F07, F10).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "oracles" / "p3"
LEGACY = ROOT / "evidence" / "p3-fortran-oracle-20260828"
F13 = ROOT / "evidence" / "p3-cuda-verify-20260829"

IN3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
        "th_old", "qv_old", "pii", "p", "dz", "w"]
IN2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]
OUT3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
         "th_old", "qv_old", "zdbz", "effc", "effi", "vmi", "di", "rhopo"]
OUT2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]


def _cases():
    rows = []
    for case in json.loads((DATA / "index.json").read_text())["cases"]:
        rows.append((case, DATA / f"{case['name']}.bin",
                     DATA / f"{case['name']}.ref"))
    for case in json.loads((LEGACY / "fixtures" / "index.json")
                           .read_text())["cases"]:
        rows.append((case, LEGACY / "fixtures" / f"{case['name']}.bin",
                     LEGACY / "dumps" / "A" / f"{case['name']}.ref"))
    for case in json.loads((F13 / "fixtures2" / "index.json")
                           .read_text())["cases"]:
        rows.append((case, F13 / "fixtures2" / f"{case['name']}.bin",
                     F13 / "dumps" / f"{case['name']}.ref"))
    return rows


CASES = _cases()


def _run(case, fixture, arm):
    import cupy as cp

    from gpuwm.core import p3_device as PD

    ncol, nk = int(case["ncol"]), int(case["nk"])
    nsteps, dt = int(case["nsteps"]), float(case["dt"])
    buf = np.fromfile(fixture, dtype="<f4")
    assert buf.size == len(IN3D) * ncol * nk + len(IN2D) * ncol
    host, off = {}, 0
    for name in IN3D:      # (ncol, nk) Fortran order == (nk, ncol) C order
        host[name] = buf[off:off + ncol * nk].reshape((nk, ncol)).copy()
        off += ncol * nk
    fields = {name: cp.asarray(host[name]) for name in
              ("qc", "qr", "nr", "qi", "qir", "ni", "qib", "th", "qv",
               "th_old", "qv_old")}
    fields["pres"] = cp.asarray(host["p"])
    fields["dz"] = cp.asarray(host["dz"])
    fields["nc"] = cp.zeros((nk, ncol), dtype=cp.float32)
    fields["ssat"] = cp.zeros((nk, ncol), dtype=cp.float32)
    diag = {name: cp.full((nk, ncol), -8.888e30, dtype=cp.float32)
            for name in PD.DIAG_SLOTS}
    surf = {"prt_liq": cp.zeros(ncol, dtype=cp.float32),
            "prt_sol": cp.zeros(ncol, dtype=cp.float32)}
    for name in IN2D:
        surf[name] = cp.asarray(buf[off:off + ncol].copy())
        off += ncol
    workspace = PD.make_workspace(ncol, nk)
    frames = []

    def snap():
        parts = [cp.asnumpy(fields[n] if n in fields else diag[n]).ravel()
                 for n in OUT3D]
        parts += [cp.asnumpy(surf[n]).ravel() for n in OUT2D]
        frames.append(np.concatenate(parts).astype("<f4"))

    snap()
    for it in range(1, nsteps + 1):
        PD.run_p3_device(fields, diag, surf, workspace=workspace, dt=dt,
                         it=it, arm=arm)
        snap()
    return np.concatenate(frames)


@pytest.mark.parametrize("arm", ["sedlevels", "unfused"])
@pytest.mark.parametrize("case,fixture,reference", CASES,
                         ids=[row[0]["name"] for row in CASES])
def test_every_word_of_every_frame_is_wrfs(case, fixture, reference, arm):
    got = _run(case, fixture, arm)
    want = np.fromfile(reference, dtype="<f4")
    assert got.size == want.size
    differ = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
    if differ.size:
        ncol, nk = int(case["ncol"]), int(case["nk"])
        per = len(OUT3D) * ncol * nk + len(OUT2D) * ncol
        step, at = divmod(int(differ[0]), per)
        field = (OUT3D[at // (ncol * nk)] if at < len(OUT3D) * ncol * nk
                 else OUT2D[(at - len(OUT3D) * ncol * nk) // ncol])
        pytest.fail(
            f"{case['name']} [{arm}]: {differ.size} of {got.size} words "
            f"differ from WRF 4.6.1; first at step {step}, field {field}: "
            f"WRF {want[differ[0]]!r}, WOOF {got[differ[0]]!r}")
