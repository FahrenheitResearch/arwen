"""Off means the bytes the tip always wrote: a two-tree proof for the seam.

Builds one small radar world (members with rain, snow, graupel and rain
number; velocity, echo and clear-air observations written through the radar
lane's own writer), runs ``assimilate_radar_grid`` over four arms that never
name ``precip_analysis``, and prints one sha256 per arm over every increment
array and the JSON provenance.  Run it with the unmodified tip on
``PYTHONPATH`` and again with the lane's tree: equal digests mean the stage,
when off, changes no byte of the seam's outputs.

Host solve only; no device is touched.

Usage: python off_identity.py <work-dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

NZ, NY, NX = 6, 16, 16
DX_M = 3000.0
TOP_M = 12000.0
MEMBERS = 6


def _smooth(field):
    for _ in range(2):
        for axis in (1, 2):
            field = (np.roll(field, 1, axis) + field
                     + np.roll(field, -1, axis)) / 3.0
    return field


def build(work: Path):
    from gpuwm.da import obsop
    from gpuwm.da.radar_assimilation import (grid_rotation, mass_to_u_faces,
                                             mass_to_v_faces, mass_to_w_faces)
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import GriddedObservations, SuperobParams
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.0, dx=DX_M, dy=DX_M, e_we=NX + 1, e_sn=NY + 1)
    grid = TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, TOP_M, NZ + 1), name="off-identity")
    rng = np.random.default_rng(20261003)
    shape = (NZ, NY, NX)
    checkpoints, dbz = {}, {}
    for index in range(MEMBERS):
        column_p = np.linspace(95000.0, 30000.0, NZ)
        column_t = np.linspace(296.0, 240.0, NZ)
        p = np.broadcast_to(column_p[:, None, None], shape).copy()
        t = np.broadcast_to(column_t[:, None, None], shape).copy()
        qv = np.broadcast_to((np.linspace(8.0e-3, 2.0e-4, NZ)
                              * (1.0 + 0.02 * index))[:, None, None],
                             shape).copy()
        precip = np.maximum(_smooth(rng.normal(5.0e-4, 6.0e-4, shape)), 0.0)
        fields = {
            "u": mass_to_u_faces(8.0 + _smooth(rng.normal(0.0, 1.5, shape))),
            "v": mass_to_v_faces(-2.0 + _smooth(rng.normal(0.0, 1.5, shape))),
            "w": mass_to_w_faces(_smooth(rng.normal(0.0, 0.3, shape))),
            "thp": _smooth(rng.normal(0.0, 0.5, shape)), "qv": qv,
            "qr": precip,
            "nr": np.where(precip > 0.0, 1.0e4 * (1.0 + 0.1 * index), 0.0),
            "qs": 0.5 * precip, "qg": 0.2 * precip, "p": p,
            "alt": 287.0 * t * (1.0 + 1.61 * qv) / p,
        }
        member_dir = work / f"member_{index:03d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        path = member_dir / "gpuwmrst_d01_000600.npz"
        np.savez(path, **{f"state/{name}": np.asarray(value, np.float32)
                          for name, value in fields.items()})
        checkpoints[index] = path
        dbz[index] = 10.0 * np.log10(1.0e-3 + 1.0e6 * precip)

    site = obsop.RadarSite(latitude_deg=float(grid.lat[NY // 2, 1]),
                           longitude_deg=float(grid.lon[NY // 2, 1]),
                           altitude_m=350.0, name="AAAA")
    beam = obsop.beam_geometry(obsop.GridGeometry.from_target_grid(grid),
                               site)
    east, north, up = (np.broadcast_to(np.asarray(c, np.float64),
                                       shape).copy()
                       for c in beam.unit_vector_enu())
    vr_mask = np.zeros((1,) + shape, np.int8)
    vr_mask[0, 1:5, 3:13, 3:13] = 1
    sina, cosa = grid_rotation(grid)
    u_e, v_n = obsop.earth_relative_winds(
        np.full(shape, 9.0), np.full(shape, -1.0), sina, cosa)
    z_obs = np.zeros(shape, np.float32)
    z_mask = np.zeros(shape, np.int8)
    z_obs[1:5, 4:8, 4:8] = 35.0
    z_mask[1:5, 4:8, 4:8] = 1
    z0_mask = np.zeros(shape, np.int8)
    z0_mask[:, 9:13, 4:12] = 1
    observations = GriddedObservations(
        z_obs=z_obs, z_mask=z_mask,
        z_err=np.where(z_mask == 1, 5.0, 0.0).astype(np.float32),
        z_max=z_obs, z_mean=z_obs, z_count=z_mask.astype(np.int32) * 4,
        vr_obs=(u_e * east + v_n * north)[None].astype(np.float32),
        vr_mask=vr_mask,
        vr_err=np.where(vr_mask == 1, 1.0, 0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((1,) + shape, np.int32),
        vr_beam_east=east[None].astype(np.float32),
        vr_beam_north=north[None].astype(np.float32),
        vr_beam_up=up[None].astype(np.float32),
        radars=[{"id": site.name, "lat_deg": site.latitude_deg,
                 "lon_deg": site.longitude_deg, "alt_m": site.altitude_m,
                 "valid_time": "2026-01-01T00:00:00Z"}],
        counts=[], provenance=[],
        z0_mask=z0_mask, z0_count=z0_mask.astype(np.int32) * 8,
        z0_err=np.where(z0_mask == 1, 7.5, 0.0).astype(np.float32))
    obs_path = work / "obs-radar-grid.nc"
    write_radar_grid(obs_path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z",
                     params=SuperobParams(), overwrite=True)
    return grid, checkpoints, obs_path, dbz


def digest(increments, provenance) -> str:
    sha = hashlib.sha256()
    for index in sorted(increments):
        for name in sorted(increments[index]):
            array = np.ascontiguousarray(increments[index][name])
            sha.update(f"{index}/{name}/{array.dtype}/{array.shape}".encode())
            sha.update(array.tobytes())
    # Wall-clock fields differ run to run; everything else is compared.
    timing = ("seconds", "solve_stage_seconds", "solve_unstage_seconds",
              "wall_seconds", "elapsed_seconds")

    def scrub(value):
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()
                    if not any(str(key).endswith(t) for t in timing)}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    sha.update(json.dumps(scrub(provenance), sort_keys=True,
                          default=str).encode())
    return sha.hexdigest()


def main() -> int:
    from gpuwm.da.letkf import Localization
    from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                             assimilate_radar_grid)

    work = Path(sys.argv[1])
    grid, checkpoints, obs_path, dbz = build(work)
    base = dict(solve_device="host", rtps_alpha=0.9,
                localization=Localization(horizontal_m=15000.0,
                                          vertical_m=4000.0))
    hydro = ("u", "v", "thp", "qv", "qr", "nr", "qs", "qg")
    arms = {
        "winds": dict(analysis_fields=("u", "v")),
        "hydro_velocity": dict(analysis_fields=hydro,
                               positivity_policy="clip", mp_physics=8),
        "hydro_reflectivity_clear_air": dict(
            analysis_fields=hydro, positivity_policy="clip", mp_physics=8,
            reflectivity=True, clear_air=True),
        "hydro_none_policy": dict(analysis_fields=hydro,
                                  positivity_policy="none", mp_physics=8),
    }
    out = {}
    for name, extra in arms.items():
        cfg = RadarAssimilationConfig(**base, **extra)
        provider = (lambda index, state: dbz[index]) if extra.get(
            "reflectivity") else None
        increments, provenance = assimilate_radar_grid(
            checkpoints, obs_path, grid, cfg,
            reflectivity_provider=provider)
        out[name] = {"sha256": digest(increments, provenance),
                     "fields": sorted(next(iter(increments.values()))),
                     "provenance_keys": len(provenance)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
