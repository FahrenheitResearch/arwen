"""A georeference-only wrfout for the observation stage, from a domain spec.

``tools/obs_radar_grid_build.py`` -- the production spine that turns
Level-II volumes into ``gpuwm-obs.radar-grid.v1`` -- binds every
observation to a model georeference it reads out of a wrfout
(:meth:`gpuwm.obs.target_grid.TargetGrid.from_wrfout`: XLAT/XLONG,
PH+PHB, HGT and the projection globals).  That is the right binding, and
it has one ordering problem: **the first cycle of a case has no wrfout
yet.**  Observations for the analysis time have to be gridded before the
forecast that would have written one exists.

This writes that georeference and nothing else: a real projected domain,
its own mass-point coordinates, terrain, and layer-interface heights from
a stated stretched profile.  It is not a forecast and does not pretend to
be one -- every prognostic field is absent, ``TITLE`` says what the file
is, and ``GPUWM_GEOREFERENCE_ONLY`` is set so a reader that opens it
looking for weather finds the answer in the file rather than in an empty
plot.

    python -m tools.obs_target_domain_wrfout --out grid.nc \\
        --center-lat 39.79 --center-lon -104.55 --dx-km 3 --nx 300 --ny 300

The coordinates are computed by the SAME projection class the observation
stage rebuilds with (:func:`gpuwm.static.projection.projection_class`), so
the file's own XLAT/XLONG and the projection its globals describe are the
same numbers to within float32 storage -- which is what that stage's
agreement check exists to verify.

VERTICAL: layer interfaces are terrain-following, geometrically stretched
from the surface to ``--model-top-km``, and the profile is stated in the
file (``GPUWM_VERTICAL_PROFILE``).  Superobbing bins a gate into the layer
whose interfaces bracket it in its own column, so the profile is a real
input to where an observation lands and is never left implicit.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Global attribute declaring what this file is.
GEOREFERENCE_ATTR = "GPUWM_GEOREFERENCE_ONLY"

#: Its value.
GEOREFERENCE_ONLY = "observation-target-domain"

#: Standard gravity, the constant the observation stage divides PH+PHB by
#: to get height.  Spelled here so the two halves cannot disagree.
STANDARD_GRAVITY = 9.81


def stretched_interfaces(nz: int, top_m: float, *, ratio: float
                         ) -> np.ndarray:
    """``(nz + 1,)`` heights above ground, 0 at the surface, ``top_m`` at
    the model top, with each layer ``ratio`` times the one below it.

    Geometric rather than uniform because a radar's lowest tilt is where
    the observations are: a uniform 40-layer column to 15 km puts one
    interface every 375 m and bins the whole boundary layer into four
    layers.
    """

    if nz < 1:
        raise ValueError(f"nz must be at least 1, got {nz}")
    weights = np.power(float(ratio), np.arange(nz, dtype=np.float64))
    edges = np.concatenate(([0.0], np.cumsum(weights)))
    return (edges / edges[-1]) * float(top_m)


def build(*, out: Path, center_lat: float, center_lon: float, dx_m: float,
          nx: int, ny: int, nz: int, top_m: float, ratio: float,
          truelat1: float, truelat2: float, stand_lon: float | None,
          terrain_m: float, valid_time: datetime.datetime,
          title: str) -> Path:
    from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs
    from gpuwm.static.projection import projection_class

    stand_lon = center_lon if stand_lon is None else stand_lon
    projection = projection_class("lambert")(
        ref_lat=center_lat, ref_lon=center_lon, truelat1=truelat1,
        truelat2=truelat2, stand_lon=stand_lon, dx=dx_m, dy=dx_m,
        e_we=nx + 1, e_sn=ny + 1)
    lat, lon = projection.latlon_mass()
    lat = np.asarray(lat, np.float32)
    lon = np.asarray(lon, np.float32)

    terrain = np.full((ny, nx), float(terrain_m), np.float32)
    above_ground = stretched_interfaces(nz, top_m, ratio=ratio)
    height = (terrain[None, :, :].astype(np.float64)
              + above_ground[:, None, None])
    geopotential = (height * STANDARD_GRAVITY).astype(np.float32)

    attrs = wrf_global_attrs(
        # The projection object carries exactly the descriptor
        # wrf_global_attrs reads, so the globals and the coordinates come
        # from one source.
        _GridDescriptor(center_lat, center_lon, truelat1, truelat2,
                        stand_lon),
        valid_time, grid_id=1, parent_id=1, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1)
    attrs[GEOREFERENCE_ATTR] = GEOREFERENCE_ONLY
    attrs["GPUWM_VERTICAL_PROFILE"] = (
        f"geometric, ratio {ratio}, {nz} layers, surface to "
        f"{top_m:.0f} m above ground, terrain-following")

    frame = {
        "XLAT": lat, "XLONG": lon, "HGT": terrain,
        "PH": np.zeros((nz + 1, ny, nx), np.float32),
        "PHB": geopotential,
        "SINALPHA": np.zeros((ny, nx), np.float32),
        "COSALPHA": np.ones((ny, nx), np.float32),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with WrfoutWriter(out, nx=nx, ny=ny, nz=nz, dx=dx_m, dy=dx_m,
                      title=title, global_attrs=attrs) as writer:
        writer.write_frame(valid_time.strftime("%Y-%m-%d_%H:%M:%S"), frame)
    return out


class _GridDescriptor:
    """The five projection numbers ``wrf_global_attrs`` reads.

    A tiny object rather than a SimpleNamespace so the attribute names are
    declared in one visible place; ``wrf_global_attrs`` reads them by name
    and defaults the rest.
    """

    def __init__(self, ref_lat, ref_lon, truelat1, truelat2, stand_lon):
        self.ref_lat = float(ref_lat)
        self.ref_lon = float(ref_lon)
        self.cen_lat = float(ref_lat)
        self.cen_lon = float(ref_lon)
        self.moad_cen_lat = float(ref_lat)
        self.truelat1 = float(truelat1)
        self.truelat2 = float(truelat2)
        self.stand_lon = float(stand_lon)
        self.wrf_map_proj = 1
        self.map_proj_char = "Lambert Conformal"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--center-lat", type=float, required=True)
    ap.add_argument("--center-lon", type=float, required=True)
    ap.add_argument("--dx-km", type=float, required=True)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, default=40)
    ap.add_argument("--model-top-km", type=float, default=15.0)
    ap.add_argument("--stretch-ratio", type=float, default=1.06)
    ap.add_argument("--truelat1", type=float, default=30.0)
    ap.add_argument("--truelat2", type=float, default=60.0)
    ap.add_argument("--stand-lon", type=float, default=None)
    ap.add_argument("--terrain-m", type=float, default=0.0,
                    help="uniform terrain height; a flat surface is a "
                         "STATEMENT, and it is written into the file")
    ap.add_argument("--valid-time", required=True,
                    help="ISO-8601 UTC, e.g. 2021-12-30T17:00:00Z")
    ap.add_argument("--title", default="gpuwm observation target domain")
    args = ap.parse_args(argv)

    when = datetime.datetime.fromisoformat(
        args.valid_time.replace("Z", "+00:00")).replace(tzinfo=None)
    path = build(
        out=args.out, center_lat=args.center_lat,
        center_lon=args.center_lon, dx_m=args.dx_km * 1000.0, nx=args.nx,
        ny=args.ny, nz=args.nz, top_m=args.model_top_km * 1000.0,
        ratio=args.stretch_ratio, truelat1=args.truelat1,
        truelat2=args.truelat2, stand_lon=args.stand_lon,
        terrain_m=args.terrain_m, valid_time=when, title=args.title)
    print(f"target domain: {path} "
          f"({args.ny}x{args.nx} at dx {args.dx_km} km, {args.nz} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
