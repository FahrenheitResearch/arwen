"""Twin for radar latent heating on the real model: one draw, not a skill
claim.

Truth is the Weisman-Klemp quarter-circle supercell of
``gpuwm.verify.cases.wk82`` (its sounding, hodograph and 3 K thermal, open
boundaries; Kessler microphysics as the case defines it, or ``--mp 8`` for
classic Thompson) on a smaller grid.  The paired arm is
the same start without the thermal: horizontally uniform, so it stays quiet
unless something puts a storm in it.

Observations are the truth's simulated reflectivity
(``gpuwm.da.obsop.simulated_reflectivity``) at 15, 30, 45 and 60 minutes
with full coverage: at or above ``--echo-dbz`` a cell is echo at its value,
below it the cell is observed clear air.  They go through the same document
adapter the cycle driver uses.

Arms, each 90 minutes:

* ``truth``: the storm start, unforced.
* ``quiet``: the quiet start, unforced.
* ``quiet_forced``: the quiet start, forced 60 minutes with four slots built
  from the four observation times against the quiet start's own state at
  t = 0 (NOAA builds its four slots against the one background the hour
  starts from), then 30 minutes free.  While forced, the microphysics runs
  under HRRR's companion clamp, ``mp_tend_lim = 0.07`` K/s
  (``parm/conus/hrrr_wrfpre.nl:108``), as every forcing does by default.
* ``storm_clear``: the storm start forced 60 minutes with four all-clear
  slots (built against the storm start at t = 0), then 30 minutes free.

Reported at 30, 60 and 90 minutes: the largest updraft and the 35 dBZ area
inside the truth's storm footprint (truth composite at or above 20 dBZ,
widened by ``--footprint-pad-km``), and the same over the whole domain.

    python -m tools.radar_tten_proof.twin --out <receipt.json>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

OBS_MINUTES = (15, 30, 45, 60)
REPORT_MINUTES = (30, 60, 90)
FORCED_MINUTES = 60
RUN_MINUTES = 90


def _config(nx, nz, mp):
    from gpuwm.config import validate_run_config
    from gpuwm.verify.cases import wk82

    return validate_run_config(dataclasses.replace(
        wk82.default_config(), nx=nx, ny=nx, nz=nz, mp_physics=mp,
        run_seconds=RUN_MINUTES * 60.0, output_interval_s=1800.0))


def _start(cfg, *, storm: bool):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.verify.cases import wk82

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82.wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    saved = wk82.BUBBLE_DELT
    try:
        if not storm:
            wk82.BUBBLE_DELT = 0.0
        return wk82.build(cfg, coord, base)
    finally:
        wk82.BUBBLE_DELT = saved


def _fields(state, cfg):
    """(composite dBZ, column-max w) as host float32 (ny, nx), and the 3-D
    reflectivity as a device copy."""
    import cupy as cp
    from gpuwm.da import obsop

    refl = obsop.simulated_reflectivity(state, cfg).copy()
    composite = cp.asnumpy(refl.max(axis=0))
    wmax = cp.asnumpy(state.w.max(axis=0))
    return composite, wmax, refl


def _run(cfg, state, *, forcing=None, label=""):
    """Integrate RUN_MINUTES, forced for the first FORCED_MINUTES when a
    forcing is given.  Returns the snapshots and the forcing receipt."""
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.da import radar_tten

    steps_per_minute = int(round(60.0 / cfg.dt))
    total = RUN_MINUTES * steps_per_minute
    keep = set(OBS_MINUTES) | set(REPORT_MINUTES)
    snaps = {}
    receipt = None
    t0 = time.time()
    if forcing is not None:
        radar_tten.attach(state, forcing, cfg)
    try:
        for n in range(1, total + 1):
            step(state, cfg)
            if forcing is not None and n == FORCED_MINUTES * steps_per_minute:
                radar_tten.detach(state)
                receipt = forcing.receipt()
                forcing = None
            if n % steps_per_minute == 0 and n // steps_per_minute in keep:
                minute = n // steps_per_minute
                composite, wmax, refl = _fields(state, cfg)
                snaps[minute] = {"composite": composite, "wmax": wmax,
                                 "refl": (cp.asnumpy(refl)
                                          if minute in OBS_MINUTES else None)}
                print(f"{label} {minute} min: max w {wmax.max():.1f} m/s, "
                      f"max dBZ {composite.max():.1f}", flush=True)
    finally:
        if forcing is not None:
            radar_tten.detach(state)
    cp.cuda.Stream.null.synchronize()
    return snaps, receipt, time.time() - t0


def _document(refl, echo_dbz, *, clear_everywhere=False):
    if clear_everywhere:
        z_mask = np.zeros(refl.shape, np.int8)
    else:
        z_mask = (refl >= echo_dbz).astype(np.int8)
    return {"variables": {"z_obs": refl.astype(np.float32), "z_mask": z_mask,
                          "z0_mask": (1 - z_mask).astype(np.int8)},
            "clear_air_source": "finite_below_floor"}


def _forcing(state, documents):
    from gpuwm.da import radar_tten

    return radar_tten.build_forcing_from_documents(
        state, documents, [float(m) for m in OBS_MINUTES])


def _footprint(composite, pad_cells):
    import cupy as cp
    from cupyx.scipy import ndimage

    echo = cp.asarray(composite >= 20.0)
    size = 2 * pad_cells + 1
    return cp.asnumpy(ndimage.maximum_filter(echo.astype(cp.uint8), size=size,
                                             mode="constant") > 0)


def _metrics(snaps, footprints, cell_km2):
    out = {}
    for minute in REPORT_MINUTES:
        snap, inside = snaps[minute], footprints[minute]
        composite, wmax = snap["composite"], snap["wmax"]
        out[str(minute)] = {
            "w_max_in_footprint_ms": float(wmax[inside].max())
            if inside.any() else None,
            "area_35dbz_in_footprint_km2": float(
                np.count_nonzero((composite >= 35.0) & inside) * cell_km2),
            "w_max_domain_ms": float(wmax.max()),
            "area_35dbz_domain_km2": float(
                np.count_nonzero(composite >= 35.0) * cell_km2),
            "area_20dbz_domain_km2": float(
                np.count_nonzero(composite >= 20.0) * cell_km2),
            "max_dbz_domain": float(composite.max()),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=112)
    parser.add_argument("--nz", type=int, default=60)
    parser.add_argument("--echo-dbz", type=float, default=5.0)
    parser.add_argument("--footprint-pad-km", type=float, default=5.0)
    parser.add_argument("--mp", type=int, default=1,
                        help="microphysics scheme (the case's own is "
                             "Kessler, 1; 8 is classic Thompson)")
    args = parser.parse_args()
    import cupy as cp

    cfg = _config(args.nx, args.nz, args.mp)
    cell_km2 = cfg.dx * cfg.dy / 1.0e6
    pad = int(round(args.footprint_pad_km * 1000.0 / cfg.dx))
    started = time.time()

    truth, _, wall_truth = _run(cfg, _start(cfg, storm=True), label="truth")
    documents = [_document(truth[m]["refl"], args.echo_dbz)
                 for m in OBS_MINUTES]
    obs_census = {str(m): {"echo_cells": int(np.count_nonzero(
        d["variables"]["z_mask"])), "clear_cells": int(np.count_nonzero(
            d["variables"]["z0_mask"]))}
        for m, d in zip(OBS_MINUTES, documents)}
    footprints = {m: _footprint(truth[m]["composite"], pad)
                  for m in REPORT_MINUTES}

    quiet, _, wall_quiet = _run(cfg, _start(cfg, storm=False), label="quiet")

    state = _start(cfg, storm=False)
    forcing = _forcing(state, documents)
    slot_receipts = list(forcing.receipts)
    quiet_forced, forced_receipt, wall_forced = _run(
        cfg, state, forcing=forcing, label="quiet_forced")
    del state, forcing

    state = _start(cfg, storm=True)
    clear_docs = [_document(truth[m]["refl"], args.echo_dbz,
                            clear_everywhere=True) for m in OBS_MINUTES]
    forcing = _forcing(state, clear_docs)
    clear_receipts = list(forcing.receipts)
    storm_clear, clear_receipt, wall_clear = _run(
        cfg, state, forcing=forcing, label="storm_clear")
    del state, forcing

    arms = {"truth": truth, "quiet": quiet, "quiet_forced": quiet_forced,
            "storm_clear": storm_clear}
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    receipt = {
        "schema": "gpuwm-da.radar-tten-twin.v1",
        "device": props["name"].decode(),
        "grid": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "dx_m": cfg.dx,
                 "dt_s": cfg.dt, "mp_physics": cfg.mp_physics},
        "case": "WK82 quarter-circle supercell sounding and hodograph; "
                "storm start has the 3 K thermal, quiet start has none",
        "reflectivity_operator": "gpuwm.da.obsop.simulated_reflectivity "
                                 f"for mp_physics {cfg.mp_physics}",
        "observations": {"minutes": list(OBS_MINUTES),
                         "echo_threshold_dbz": args.echo_dbz,
                         "coverage": "full: echo at or above the threshold, "
                                     "observed clear air below it",
                         "census": obs_census},
        "footprint": {"definition": "truth composite >= 20 dBZ, widened by "
                                    f"{args.footprint_pad_km} km",
                      "cells": {str(m): int(footprints[m].sum())
                                for m in REPORT_MINUTES}},
        "metrics": {name: _metrics(snaps, footprints, cell_km2)
                    for name, snaps in arms.items()},
        "forcing": {"quiet_forced": {"slots": slot_receipts,
                                     "run": forced_receipt},
                    "storm_clear": {"slots": clear_receipts,
                                    "run": clear_receipt}},
        "wall_seconds": {"truth": round(wall_truth, 1),
                         "quiet": round(wall_quiet, 1),
                         "quiet_forced": round(wall_forced, 1),
                         "storm_clear": round(wall_clear, 1),
                         "total": round(time.time() - started, 1)},
    }
    args.out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt["metrics"], indent=1))


if __name__ == "__main__":
    main()
