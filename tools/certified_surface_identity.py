"""Short byte-identity trajectory for the four certified Noah profiles.

The fixture consumed by ``tests/test_surface_certified_identity.py`` is
written by running this unchanged script once against the certified v1.1.2
commit and recording its JSON.  The lane under test changes only RUC and
Noah-MP surface seams; these Noah profiles must retain every hashed byte.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime

import numpy as np


# The script can live in the changed tree while being executed with a
# detached baseline worktree as CWD.  That is how one unchanged harness
# produces both halves of the before/after comparison.
MODEL_ROOT = Path(os.environ.get(
    "GPUWM_IDENTITY_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(MODEL_ROOT))


PROFILE_SPECS = {
    "wsm6_dudhia": {
        "mp_physics": 6, "ra_lw_physics": 0, "ra_sw_physics": 1,
        "radt": 1.0, "cu_physics": 0, "cudt_minutes": 0.0,
        "top_lid": True, "moist_cq": False,
        "wrf_rrtmg_compatibility": "none",
    },
    "thompson_dudhia": {
        "mp_physics": 8, "ra_lw_physics": 0, "ra_sw_physics": 1,
        "radt": 1.0, "cu_physics": 0, "cudt_minutes": 0.0,
        "top_lid": False, "moist_cq": True,
        "wrf_rrtmg_compatibility": "none",
    },
    "morrison_rte": {
        "mp_physics": 10, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "radt": 12.0, "cu_physics": 1, "cudt_minutes": 5.0,
        "top_lid": False, "moist_cq": True,
        "wrf_rrtmg_compatibility": "wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
    },
    "nssl2_rte": {
        "mp_physics": 18, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "radt": 12.0, "cu_physics": 1, "cudt_minutes": 5.0,
        "top_lid": False, "moist_cq": True,
        "wrf_rrtmg_compatibility": "wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
    },
}


def _hash_array(digest, name, array):
    import cupy as cp

    host = np.ascontiguousarray(cp.asnumpy(array))
    digest.update(name.encode("utf-8"))
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(np.asarray(host.shape, np.int64).tobytes())
    digest.update(host.tobytes())


def _one_profile(name, switches):
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = RunConfig(
        nx=4, ny=2, nz=16, dx=3000.0, dy=3000.0, ztop=9000.0,
        dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
        sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=1,
        # The certified PHYSICS selectors are exact. The tiny synthetic
        # state is flat because init_moist_balanced deliberately has no
        # terrain constructor; terrain transport is outside this seam.
        num_soil_layers=4, terrain_opt=0, km_opt=4,
        diff_6th_opt=2, diff_6th_factor=0.12,
        diff_6th_slopeopt=1, epssm=0.5, morr_rimed_ice=1,
        wsm6_hail_opt=0, **switches)
    coord = make_vertical_coord(cfg.nz, stretch=1.4)
    base = make_base_state(
        coord, lambda z: 299.0 + 0.004 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: np.maximum(
            0.010 * np.exp(-np.asarray(z, np.float64) / 2200.0), 1.0e-5))
    state.u[...] = cp.float32(5.0)
    state.v[...] = cp.float32(0.75)

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -1] = 0.0
    tsk = np.full((cfg.ny, cfg.nx), 301.0, np.float64)
    tsk[:, -1] = 295.0
    soil_t = np.stack([tsk - offset for offset in (0.5, 1.0, 2.0, 3.0)])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.30, np.float64)
    soil_m[:, :, -1] = 1.0
    latitude = np.full((cfg.ny, cfg.nx), 40.0, np.float64)
    longitude = np.full((cfg.ny, cfg.nx), -100.0, np.float64)
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=np.where(landmask > 0.5, 10, 17),
        isltyp=np.where(landmask > 0.5, 6, 14),
        vegfra=55.0, tmn=287.0, swdown=500.0, glw=320.0, pblh=600.0,
        radiation_start_time=datetime(2026, 7, 1, 18),
        radiation_latitude=latitude, radiation_longitude=longitude)
    for _ in range(2):
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
    field_names = sorted(driver.fields)
    return {
        "sha256": digest.hexdigest(),
        "field_inventory_sha256": hashlib.sha256(
            "\n".join(field_names).encode("utf-8")).hexdigest(),
        "elapsed_seconds": float(state.elapsed_seconds),
        "profile": name,
    }


def run_profiles():
    return {
        name: _one_profile(name, switches)
        for name, switches in PROFILE_SPECS.items()
    }


def main():
    print(json.dumps(run_profiles(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
