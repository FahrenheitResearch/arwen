"""WOOF's side of the Noah-MP column oracle: the shipped device path.

Reads the same input file ``run_column`` reads, cold-starts the slab with
:func:`gpuwm.core.noahmp_runtime.noahmp_cold_start` (the CUDA
``NOAHMP_INIT``/``SNOW_INIT`` arm), then calls
:func:`gpuwm.core.noahmp_runtime.noahmp_lsm_step` once per step -- the same
entry ``gpuwm/core/physics.py`` calls in a forecast, which takes the whole
slab orchestration (CUDA leaf batches, the CUDA glacier column, the device
write-back) by default.  Writes the cold-started state and every step's
state in the record format and field order of ``run_column.F90``.

Nothing here substitutes a host routine: the run refuses to start if the
slab path or the device cold start is not the one selected.

Usage:  python run_woof_column.py INPUT.bin OUTPUT.bin
        (GPUWM_WRF_EXACT=1 in the environment selects the strict build)
"""

from __future__ import annotations

import os
import struct
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.noahmp_wrf461_oracle.column.gen_columns import CONFIGS, read_records  # noqa: E402

#: The compared fields, in run_column.F90's dump order.
FIELDS_2D = (
    "tsk", "hfx", "qfx", "lh", "grdflx", "smstav", "smstot", "sfcrunoff",
    "udrunoff", "albedo", "snowc", "snow", "snowh", "canwat", "acsnom",
    "acsnow", "emiss", "qsfc", "z0", "znt", "lai",
    "tvxy", "tgxy", "canicexy", "canliqxy", "eahxy", "tahxy", "cmxy",
    "chxy", "fwetxy", "sneqvoxy", "alboldxy", "qsnowxy", "qrainxy",
    "wslakexy", "waxy", "xsaixy", "taussxy",
    "t2mvxy", "t2mbxy", "q2mvxy", "q2mbxy", "tradxy", "fsaxy", "firaxy",
    "ecanxy", "edirxy", "etranxy", "runsfxy", "runsbxy", "fvegxy",
    "rssunxy", "rsshaxy", "chvxy", "chbxy", "fpicexy", "qsnbotxy",
    "qmeltxy", "pondingxy", "canhsxy", "eflxbxy", "soilenergy",
    "snowenergy", "rs",
)
FIELDS_INT = ("isnowxy", "pgsxy")
FIELDS_3D = ("tslb", "smois", "sh2o", "tsnoxy", "snicexy", "snliqxy",
             "zsnsoxy")
DZS = (0.1, 0.3, 0.6, 1.0)


def _record(name, kind, values) -> bytes:
    arr = np.ascontiguousarray(values, dtype="<f4" if kind == 1 else "<i4")
    return (name.ljust(16).encode("ascii")[:16]
            + struct.pack("<ii", kind, arr.size) + arr.tobytes())


def config_for(inputs) -> dict:
    """Find the generator configuration that wrote this input file."""
    nstep = int(inputs["nstep"][0][0])
    dt = float(inputs["dt"][0][0])
    for name, cfg in CONFIGS.items():
        if cfg["nstep"] == nstep and np.float32(cfg["dt"]) == np.float32(dt):
            return dict(cfg, name=name)
    raise SystemExit(f"no generator configuration has nstep={nstep} dt={dt}")


def run(in_path: str, out_path: str, *, noahmp_snow_age=None) -> dict:
    import cupy as cp

    import gpuwm  # noqa: F401  (installs the strict compiler hook if asked)
    from gpuwm import wrf_exact
    from gpuwm.core import noahmp_runtime as rt
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing

    # The shipped path, asserted rather than assumed.
    for env in (rt.HOST_LEAVES_ENV, rt.STAGED_COLUMNS_ENV,
                rt.HOST_COLD_START_ENV):
        if os.environ.get(env, "") == "1":
            raise SystemExit(f"{env}=1 selects a host path; unset it")
    if not rt._slab_path_selected():
        raise SystemExit("the slab orchestration is not the selected path")
    if rt._resolve_cold_start() is not rt.cold_start_on_device:
        raise SystemExit("the device cold start is not the selected arm")
    if rt._resolve_glacier_evaluator() is not rt._glacier_evaluate_on_device:
        raise SystemExit("the device glacier column is not the selected arm")

    inp = read_records(in_path)
    cfg = config_for(inp)
    n = int(inp["nx"][0][0])
    nstep = int(inp["nstep"][0][0])
    dt = float(inp["dt"][0][0])
    dx = float(inp["dx"][0][0])
    julian = inp["julian"][0]
    years = inp["yr"][0]

    def static(name):
        return inp[name][0].astype(np.float32).reshape(1, n)

    def forcing(name):
        return inp[name][0].astype(np.float32).reshape(nstep, n)

    shape = (1, n)
    f = {}

    def zeros(name, dtype=cp.float32, lead=None):
        f[name] = cp.zeros(shape if lead is None else (lead, *shape),
                           dtype=dtype)

    for name in ("xland", "xice", "tsk", "tmn", "snow", "snowh", "lai",
                 "shdmax"):
        f[name] = cp.asarray(static(name))
    f["ivgtyp"] = cp.asarray(inp["ivgtyp"][0].astype(np.int32).reshape(shape))
    f["isltyp"] = cp.asarray(inp["isltyp"][0].astype(np.int32).reshape(shape))
    f["tslb"] = cp.asarray(inp["tslb"][0].astype(np.float32).reshape(4, *shape))
    f["smois"] = cp.asarray(
        inp["smois"][0].astype(np.float32).reshape(4, *shape))
    f["sh2o"] = cp.zeros((4, *shape), dtype=cp.float32)
    for name in ("swdown", "glw", "rainbl", "sr", "qsfc", "canwat", "hfx",
                 "qfx", "lh", "grdflx", "albedo", "snowc", "emiss", "z0",
                 "znt", "smstav", "smstot", "sfcrunoff", "udrunoff",
                 "acsnow", "acsnom", "psfc", "cqs2", "chs2", "t2", "q2",
                 "th2", "rainc_acc", "rainnc_acc", "rainsh_acc",
                 "snownc_acc", "graupnc_acc", "hailnc_acc",
                 *rt.NOAHMP_STATE_2D, *rt.NOAHMP_DIAGNOSTICS_2D):
        zeros(name)
    for name in rt.NOAHMP_STATE_INT_2D:
        zeros(name, cp.int32)
    for name in rt.NOAHMP_STATE_SNOW_3D:
        zeros(name, lead=rt.NSNOW)
    for name in rt.NOAHMP_STATE_SNOWSOIL_3D:
        zeros(name, lead=rt.NSNOW + 4)

    params = rt.NoahmpRuntimeParameters()
    geometry = rt.NoahmpSolarGeometry(cfg["start"], static("xlat"),
                                      static("xlong"))
    for k in range(nstep):
        want = np.float32(geometry.julian(k * dt))
        if want.tobytes() != np.float32(julian[k]).tobytes():
            raise SystemExit(f"julian mismatch at step {k}: {want} vs "
                             f"{julian[k]}")
        if geometry.year(k * dt) != int(years[k]):
            raise SystemExit(f"year mismatch at step {k}")

    rt.noahmp_cold_start(f, params=params, dzs=DZS)

    out = bytearray()

    def dump(step):
        out.extend(_record("step", 2, [step]))
        for name in FIELDS_2D:
            out.extend(_record(name, 1, cp.asnumpy(f[name]).reshape(n)))
        for name in FIELDS_INT:
            out.extend(_record(name, 2, cp.asnumpy(f[name]).reshape(n)))
        for name in FIELDS_3D:
            arr = cp.asnumpy(f[name])
            out.extend(_record(name, 1, arr.reshape(arr.shape[0], n)))

    dump(0)
    fc = {name: forcing(name) for name in (
        "coszen", "t3d", "qv3d", "u", "v", "dz8w", "p8w1", "p8w2", "swdown",
        "glw", "rainbl", "sr", "rainc", "rainnc", "rainsh", "snownc",
        "graupnc", "hailnc")}
    census = []
    for k in range(nstep):
        def lev(name, levels=1):
            return cp.asarray(np.broadcast_to(
                fc[name][k].reshape(1, *shape), (levels, *shape)).copy())
        atmosphere = {
            "temperature": lev("t3d", 2), "qv": lev("qv3d", 2),
            "u": lev("u", 2), "v": lev("v", 2), "dz": lev("dz8w", 2),
            "p_interface": cp.asarray(np.stack(
                [fc["p8w1"][k].reshape(shape),
                 fc["p8w2"][k].reshape(shape)])),
        }
        f["swdown"][...] = cp.asarray(fc["swdown"][k].reshape(shape))
        f["glw"][...] = cp.asarray(fc["glw"][k].reshape(shape))
        f["rainbl"][...] = cp.asarray(fc["rainbl"][k].reshape(shape))
        f["sr"][...] = cp.asarray(fc["sr"][k].reshape(shape))
        f["psfc"][...] = cp.asarray(fc["p8w1"][k].reshape(shape))
        precipitation = SurfacePrecipitationForcing(
            rain_convective=cp.asarray(fc["rainc"][k].reshape(shape)),
            rain_nonconvective=cp.asarray(fc["rainnc"][k].reshape(shape)),
            rain_shallow_convective=cp.asarray(fc["rainsh"][k].reshape(shape)),
            snow_nonconvective=cp.asarray(fc["snownc"][k].reshape(shape)),
            graupel_nonconvective=cp.asarray(fc["graupnc"][k].reshape(shape)),
            hail_nonconvective=cp.asarray(fc["hailnc"][k].reshape(shape)),
        )
        census.append(rt.noahmp_lsm_step(
            f, atmosphere, params=params, geometry=geometry,
            precipitation=precipitation,
            coszen=cp.asarray(fc["coszen"][k].reshape(shape)),
            dt=dt, dx=dx, dzs=DZS, itimestep=k + 1,
            elapsed_seconds=k * dt, dveg=4, opt_run=3, opt_crop=0,
            opt_irr=0, opt_tdrn=0, opt_soil=1,
            **({} if noahmp_snow_age is None else {"noahmp_snow_age": noahmp_snow_age})))
        dump(k + 1)
    Path(out_path).write_bytes(bytes(out))
    receipt = wrf_exact.compile_receipt()
    return {"config": cfg["name"], "columns": n, "steps": nstep,
            "census_first": census[0], "strict": bool(receipt["enabled"]),
            "strict_options": receipt["strict_options"],
            "compile_records": len(receipt["records"])}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    import json
    print(json.dumps(run(sys.argv[1], sys.argv[2]), default=str))
