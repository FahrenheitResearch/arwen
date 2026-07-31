"""Byte-identity trajectory for the already-admitted MYNN/MYNN/Noah suite.

This is a regression baseline, not a physics oracle.  WRF source oracles
establish the component answers; this harness establishes that changing the
RUC and Noah-MP ownership sequencing does not change the existing
``sf_sfclay_physics=5``, ``bl_pbl_physics=5``,
``sf_surface_physics=2`` trajectory.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))


def _hash_array(digest, name, array):
    import cupy as cp

    host = np.ascontiguousarray(cp.asnumpy(array))
    digest.update(name.encode("utf-8"))
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(np.asarray(host.shape, np.int64).tobytes())
    digest.update(host.tobytes())


def run_identity() -> dict[str, object]:
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = RunConfig(
        nx=6, ny=4, nz=24, dx=3000.0, dy=3000.0, ztop=12000.0,
        dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=6, sf_sfclay_physics=5, sf_surface_physics=2,
        bl_pbl_physics=5, bldt=0.0, ra_sw_physics=1, ra_lw_physics=0,
        radt=1.0, top_lid=True, num_soil_layers=4,
    )

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(
            z < 1400.0,
            300.0,
            np.where(z < 1650.0, 300.0 + 0.024 * (z - 1400.0),
                     306.0 + 0.004 * (z - 1650.0)),
        )

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(
            z < 1400.0,
            0.0125,
            np.maximum(0.0125 - 5.5e-6 * (z - 1400.0), 1.0e-5),
        )

    coord = make_vertical_coord(cfg.nz, stretch=2.8)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(6.0)
    state.v[...] = cp.float32(1.25)

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -1] = 0.0
    tsk = np.full((cfg.ny, cfg.nx), 302.0, np.float64)
    tsk[:, -1] = 296.0
    soil_t = np.stack([tsk - offset for offset in (0.5, 1.0, 2.0, 3.0)])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.29, np.float64)
    soil_m[:, :, -1] = 1.0
    latitude = np.full((cfg.ny, cfg.nx), 40.0, np.float64)
    longitude = np.full((cfg.ny, cfg.nx), -100.0, np.float64)
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=np.where(landmask > 0.5, 10, 17),
        isltyp=np.where(landmask > 0.5, 6, 14),
        vegfra=55.0, tmn=287.0, swdown=575.0, glw=325.0, pblh=500.0,
        radiation_start_time=datetime(2026, 7, 1, 18),
        radiation_latitude=latitude, radiation_longitude=longitude,
    )
    for _ in range(3):
        update_diagnostics(state)
        step(state, cfg)

    digest = hashlib.sha256()
    for field in sorted(state.__dict__):
        value = getattr(state, field)
        if isinstance(value, cp.ndarray):
            _hash_array(digest, f"state/{field}", value)
    for field in sorted(driver.fields):
        _hash_array(digest, f"fields/{field}", driver.fields[field])
    for field in sorted(driver.microphysics.__dataclass_fields__):
        value = getattr(driver.microphysics, field)
        if value is not None:
            _hash_array(digest, f"microphysics/{field}", value)
    return {
        "elapsed_seconds": float(state.elapsed_seconds),
        "field_inventory_sha256": hashlib.sha256(
            "\n".join(sorted(driver.fields)).encode("utf-8")).hexdigest(),
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    print(json.dumps(run_identity(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
