#!/usr/bin/env python3
"""WOOF side of the RUC LSM column oracle: the production GPU step.

Builds a one-row physics driver whose columns are :func:`columns.build_case`,
lets WOOF's own cold start (``ruclsminit``'s port) run inside
``initialize_physics``, sets every LSMRUC argument and every seam field to the
case's values, writes ``inputs.bin`` for the Fortran driver from those exact
words, then runs :func:`gpuwm.core.ruc_runtime.ruc_lsm_step` -- the forecast
entry point, which dispatches the fused CUDA kernels -- for every step and
saves every compared field after each call.

Run under ``GPUWM_WRF_EXACT=1`` for the strict build; unset for default
arithmetic.  ``--replay WRF_OUTPUTS`` restarts every step from WRF's state
after the previous step, which localises a difference to one call.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from columns import ATMOS, PROFILES, STATE_2D, STATIC_2D, STEP_2D, build_case  # noqa: E402
import layout  # noqa: E402

#: Fields the case does NOT override after WOOF's cold start: these four are
#: ruclsminit's outputs, so S0 carries WOOF's own cold-start words for them.
COLD_START_OWNED = ("mavail", "znt")


def case_from_args(args) -> dict:
    return build_case(nzs=args.nzs, nsteps=args.steps, dt=args.dt,
                      fractional_seaice=args.fractional_seaice,
                      mosaic=bool(args.mosaic), lakemodel=args.lakemodel,
                      rdlai2d=bool(args.rdlai2d),
                      round_xice=not args.keep_fractional_xice)


def build_driver(case):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    ncol, nzs = case["ncol"], case["nzs"]
    cfg = RunConfig(nx=ncol, ny=1, nz=10, dx=3000.0, dy=3000.0, ztop=16000.0,
                    dt=float(case["dt"]), run_seconds=0.0, time_step_sound=4,
                    moist=True, mp_physics=8, sf_sfclay_physics=1,
                    sf_surface_physics=3, num_soil_layers=nzs,
                    bl_pbl_physics=1, bldt=0.0, ra_physics=0,
                    radt_minutes=12.0,
                    fractional_seaice=int(case["fractional_seaice"]),
                    rdlai2d=bool(case["rdlai2d"]),
                    mosaic_lu=int(case["mosaic"]),
                    mosaic_soil=int(case["mosaic"]))
    coord = make_vertical_coord(cfg.nz, stretch=2.0)
    base = make_base_state(coord, lambda z: 300.0 + 0.0 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                                lambda z: 0.005 + 0.0 * np.asarray(z))
    st, pr = case["state"], case["profiles"]
    row = lambda a: np.asarray(a, np.float64)[None, :]
    kwargs = {}
    if case["mosaic"]:
        kwargs = dict(landusef=case["landusef"][:, None, :].astype(np.float64),
                      soilctop=case["soilctop"][:, None, :].astype(np.float64))
    driver = initialize_physics(
        state, cfg, landmask=row(st["xland"] < 1.5), tsk=row(st["tsk"]),
        soil_temperature=pr["tslb"][:, None, :].astype(np.float64),
        soil_moisture=pr["smois"][:, None, :].astype(np.float64),
        liquid_moisture=pr["sh2o"][:, None, :].astype(np.float64),
        ivgtyp=case["ivgtyp"][None, :], isltyp=case["isltyp"][None, :],
        vegfra=row(st["vegfra"]), tmn=row(st["tmn"]), xice=row(st["xice"]),
        snow=row(st["snow"]), snow_depth=row(st["snowh"]),
        canwat=row(st["canwat"]), swdown=0.0, glw=300.0, pblh=500.0,
        lakemask=row(st["lakemask"]), **kwargs)
    return driver


def snapshot(fields, names, profiles=()):
    import cupy as cp
    out = {}
    for name in names:
        out[name] = cp.asnumpy(fields[name]).reshape(-1).astype(np.float32)
    for name in profiles:
        a = cp.asnumpy(fields[name])
        out[name] = a.reshape(a.shape[0], -1).astype(np.float32)
    return out


def run(case, outdir: Path, replay: Path | None = None):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing

    driver = build_driver(case)
    f = driver.fields
    params = driver.ruc_params
    ncol, nzs = case["ncol"], case["nzs"]

    # What WOOF's cold start saw and produced.
    raw = snapshot(f, ("ivgtyp", "isltyp", "xice"), ("tslb", "smois"))
    raw["ivgtyp"] = raw["ivgtyp"].astype(np.int32)
    raw["isltyp"] = raw["isltyp"].astype(np.int32)
    init = snapshot(f, layout.INIT_2D, layout.INIT_PROFILES)

    # S0: the case's words for every LSMRUC argument and seam field, except
    # the cold-start outputs, which stay WOOF's.
    for name in STATE_2D + STATIC_2D:
        if name in COLD_START_OWNED:
            continue
        f[name][...] = cp.asarray(case["state"][name].reshape(f[name].shape))
    f["keepfr3dflag"][...] = 0
    if case["mosaic"]:
        f["landusef"][...] = cp.asarray(case["landusef"].reshape(f["landusef"].shape))
        f["soilctop"][...] = cp.asarray(case["soilctop"].reshape(f["soilctop"].shape))
    s0 = snapshot(f, STATE_2D + STATIC_2D, PROFILES)
    s0["ivgtyp"] = cp.asnumpy(f["ivgtyp"]).reshape(-1).astype(np.int32)
    s0["isltyp"] = cp.asnumpy(f["isltyp"]).reshape(-1).astype(np.int32)
    s0["landusef"] = case["landusef"]
    s0["soilctop"] = case["soilctop"]
    if case["mosaic"]:
        s0["landusef"] = cp.asnumpy(f["landusef"]).reshape(-1, ncol).astype(np.float32)
        s0["soilctop"] = cp.asnumpy(f["soilctop"]).reshape(-1, ncol).astype(np.float32)
    layout.write_inputs(outdir / "inputs.bin", case, s0, raw,
                        params.xice_threshold, params.seaice_albedo_default)

    wrf = None
    if replay is not None:
        wrf = layout.read_outputs(replay, ncol, nzs, case["nsteps"])

    saved = {f"init__{k}": v for k, v in init.items()}
    saved.update({f"s0__{k}": v for k, v in s0.items()})
    census = []
    for k, step in enumerate(case["steps"], start=1):
        if wrf is not None and k > 1:
            prev = wrf["steps"][k - 2]
            for name in layout.OUT_2D:
                f[name][...] = cp.asarray(prev[name].reshape(f[name].shape))
            for name in PROFILES:
                f[name][...] = cp.asarray(prev[name].reshape(f[name].shape))
        for name in STEP_2D:
            f[name][...] = cp.asarray(step["forcing"][name].reshape(f[name].shape))
        atmosphere = {name: cp.asarray(step["atmos"][name]).reshape(1, 1, ncol)
                      for name in ATMOS}
        result = ruc_lsm_step(
            f, atmosphere, params=params,
            precipitation=SurfacePrecipitationForcing.from_fields(f),
            dt=float(case["dt"]), itimestep=k,
            mosaic_lu=case["mosaic"], mosaic_soil=case["mosaic"],
            flag_sm_adj=0, spp_lsm=0, lakemodel=case["lakemodel"],
            ruc_irrigation="wrf_461", ruc_soilprop="wrf_461",
            ruc_qvg_cold_start="wrf", ruc_2m_diagnostic="flux",
            ruc_snow="wrf_461")
        census.append(result)
        out = snapshot(f, layout.OUT_2D, PROFILES)
        saved.update({f"step{k}__{n}": v for n, v in out.items()})
    np.savez_compressed(outdir / ("woof-replay.npz" if replay else "woof.npz"),
                        **saved)
    meta = {"ncol": ncol, "nzs": nzs, "nsteps": case["nsteps"],
            "dt": case["dt"], "fractional_seaice": case["fractional_seaice"],
            "mosaic": case["mosaic"], "lakemodel": case["lakemodel"],
            "rdlai2d": case["rdlai2d"], "regime": case["regime"],
            "xice_threshold": float(params.xice_threshold),
            "census": census,
            "wrf_exact": os.environ.get("GPUWM_WRF_EXACT", "0"),
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
    (outdir / "case.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("outdir", type=Path)
    p.add_argument("--nzs", type=int, default=9)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--dt", type=float, default=20.0)
    p.add_argument("--fractional-seaice", type=int, default=1)
    p.add_argument("--mosaic", type=int, default=0)
    p.add_argument("--lakemodel", type=int, default=0)
    p.add_argument("--rdlai2d", type=int, default=0)
    p.add_argument("--keep-fractional-xice", action="store_true",
                   help="with --fractional-seaice 0, keep 0<xice<1 instead "
                        "of the 0/1 mask real.exe writes (seam probe)")
    p.add_argument("--replay", type=Path)
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.replay is None:
        layout.Path(args.outdir / "run_columns.F90").write_text(layout.fortran_driver())
    meta = run(case_from_args(args), args.outdir, args.replay)
    print(json.dumps({k: meta[k] for k in ("ncol", "nzs", "census", "wrf_exact", "device")}))


if __name__ == "__main__":
    main()
