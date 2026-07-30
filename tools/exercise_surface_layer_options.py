"""Execute the ported MM5 surface-layer knobs on the GPU through the real
physics driver, and cross-check against the independent CPU reference.

This is a model execution, not a unit test: it builds a RunConfig the way a
case config does, initializes the real PhysicsDriver, and calls the driver's
own surface-layer entry point on device.  The CPU reference in
gpuwm/verify/npref.py is a float64 mirror that the GPU path never touches.
"""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cupy as cp
import numpy as np

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core import physics as physics_mod
from gpuwm.core.landuse import initialize_landuse
from gpuwm.verify.npref import np_sfclay

NY, NX, NZ = 1, 4, 16
SHAPE = (NY, NX)
FIELDS = ("hfx", "qfx", "ust", "znt")
INOUT = ("znt", "ust", "mol", "hfx", "qfx", "qsfc", "zol")

# Two land columns and two water columns, so the land-only iz0tlnd branch and
# the water-only isftcflx branch are both exercised in one launch.
LU_INDEX = np.array([[1, 10, 17, 17]], np.int32)          # 17 = water
SOIL = np.array([[6, 6, 14, 14]], np.int32)
LANDMASK = np.array([[1.0, 1.0, 0.0, 0.0]])
TSK = np.array([[298.0, 300.0, 292.0, 292.0]], np.float32)
MAVAIL = np.array([[0.4, 0.3, 1.0, 1.0]], np.float32)
WIND_U, WIND_V = 9.0, 0.5
ATM = dict(t=295.0, qv=0.010, p=1.0e5, dz=60.0, psfc=1.0065e5, pblh=600.0)


def _levels(value):
    return cp.full((1, NY, NX), value, dtype=cp.float32)


def run(isftcflx: int, iz0tlnd: int):
    cfg = validate_run_config(RunConfig(
        nx=NX, ny=NY, nz=NZ, dx=3000.0, dy=3000.0, ztop=8000.0,
        dt=10.0, run_seconds=0.0, time_step_sound=4,
        sf_sfclay_physics=91, isftcflx=isftcflx, iz0tlnd=iz0tlnd))
    landuse = initialize_landuse(
        LU_INDEX, soil_type=SOIL, landmask=LANDMASK, snow=0.0, xice=0.0,
        valid_time=datetime(2020, 6, 1, 18), cen_lat=39.0,
        mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21, isice=15)
    state = SimpleNamespace(mup=cp.ones(SHAPE, dtype=cp.float32),
                            p=cp.ones((NZ, NY, NX), dtype=cp.float32),
                            physics=None)
    driver = physics_mod.initialize_physics(state, cfg, landuse=landuse)
    f = driver.fields
    f["tsk"][...] = cp.asarray(TSK)
    f["pblh"][...] = cp.float32(ATM["pblh"])
    f["mavail"][...] = cp.asarray(MAVAIL)

    incoming = {name: cp.asnumpy(getattr(driver.sfclay_result, name)).copy()
                for name in INOUT}
    xland = cp.asnumpy(f["xland"]).copy()

    atmosphere = {
        "u": _levels(WIND_U), "v": _levels(WIND_V),
        "temperature": _levels(ATM["t"]), "qv": _levels(ATM["qv"]),
        "pressure": _levels(ATM["p"]), "dz": _levels(ATM["dz"]),
        "p_interface": _levels(ATM["psfc"]),
    }
    driver._run_sfclay(atmosphere, cfg)
    out = {name: cp.asnumpy(getattr(driver.sfclay_result, name)).ravel().copy()
           for name in FIELDS}
    return cfg, out, incoming, xland


def reference(isftcflx: int, iz0tlnd: int, incoming, xland):
    full = lambda v: np.full(SHAPE, v, np.float64)  # noqa: E731
    result = np_sfclay(
        full(WIND_U), full(WIND_V), full(ATM["t"]), full(ATM["qv"]),
        full(ATM["p"]), full(ATM["dz"]), full(ATM["psfc"]), TSK,
        incoming["znt"], full(ATM["pblh"]), MAVAIL, xland,
        option=91, qsfc=incoming["qsfc"], zol=incoming["zol"],
        ust=incoming["ust"], mol=incoming["mol"], hfx=incoming["hfx"],
        qfx=incoming["qfx"], dx=3000.0, isfflx=True,
        isftcflx=isftcflx, iz0tlnd=iz0tlnd)
    return {name: np.asarray(result[name]).ravel() for name in FIELDS}


def main() -> int:
    print("device:",
          cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
          f"| cupy {cp.__version__}")
    print("classic MM5 surface layer (sf_sfclay_physics=91), "
          "columns = [land, land, water, water]\n")

    _, base, incoming0, xland = run(0, 0)
    print("baseline isftcflx=0 iz0tlnd=0")
    for name in FIELDS:
        print(f"    {name:4s} {np.array2string(base[name], precision=6)}")

    land, water = [0, 1], [2, 3]
    cases = (
        ("iz0tlnd=1  Chen-Zhang thermal roughness", (0, 1), land),
        ("iz0tlnd=2  fixed CZIL 0.1", (0, 2), land),
        ("isftcflx=1 Garratt water roughness", (1, 0), water),
        ("isftcflx=2 Donelan water roughness", (2, 0), water),
    )
    ok = True
    for label, (a, b), affected in cases:
        _, out, _, _ = run(a, b)
        untouched = [i for i in range(NX) if i not in affected]
        moved = [n for n in FIELDS
                 if not np.allclose(out[n][affected], base[n][affected])]
        clean = all(np.allclose(out[n][untouched], base[n][untouched])
                    for n in FIELDS)
        surface = "land" if affected == land else "water"
        print(f"\n  {label}   (acts on {surface} columns)")
        for name in FIELDS:
            print(f"    {name:4s} base={np.array2string(base[name][affected], precision=6)}"
                  f"  ->  {np.array2string(out[name][affected], precision=6)}")
        print(f"    fields changed: {moved or 'NONE'}; "
              f"other surface untouched: {clean}")
        ok &= bool(moved) and clean

    print("\n  agreement with the float64 CPU reference (gpuwm/verify/npref.py)")
    for a, b in ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)):
        _, gpu, incoming, xl = run(a, b)
        ref = reference(a, b, incoming, xl)
        worst, worst_name = 0.0, ""
        for name in FIELDS:
            scale = max(np.abs(ref[name]).max(), 1e-9)
            rel = np.abs(gpu[name] - ref[name]).max() / scale
            if rel > worst:
                worst, worst_name = rel, name
        good = worst < 2e-4
        ok &= good
        print(f"    isftcflx={a} iz0tlnd={b}: max relative difference "
              f"{worst:.2e} ({worst_name})  agrees={good}")

    print("\nRESULT:", "both ported knobs change the model state on the correct "
          "surface and match the CPU reference" if ok else "MISMATCH, see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
