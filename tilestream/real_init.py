"""Initialise a DECOMPOSED run from a real analysis.

The single-GPU real-data path is production code and is not rebuilt here.
``gpuwm.ingest`` decodes the GRIB, interpolates HRRR's native hybrid levels
onto the Lambert target, reconciles the soil and writes a prepared cache;
``gpuwm.prepared_single_domain_forecast`` restores that cache into a
``DomainState`` and attaches physics.  Every one of those steps is called, not
reimplemented.

What did not exist is the step between "one analysis-derived domain" and
"eight cards": handing each rank ITS WINDOW of that domain.

The obvious wrong way to do it is to run the ingest again per rank against a
smaller target.  That re-centres the Lambert projection on the rank's own
extents -- the same defect ``tilestream.driver`` documents for a rebuilt
geography, where a 3x3 split at 12 km displaces a tile by up to 827 km -- so
the eight ranks would initialise eight different places and the seams would be
discontinuities in the earth, not just in the weather.

So the inputs are SLICED, exactly once each, from the single global analysis:

``state/*``   the prognostic arrays, read straight out of the prepared cache
              as host NumPy (``PreparedCacheReader`` needs no CUDA) and
              windowed before they ever reach a device.  A rank never
              materialises the global field.
``static``    terrain, land use, soil category, greenness, albedo, map
              factors, Coriolis, rotation -- windowed.
``met``/``surface``
              the near-surface and canonical Noah inventories -- windowed.
              ``U10`` is x-staggered and ``V10`` is y-staggered, and the
              variant is derived from the SHAPE rather than from a list, so a
              field cannot be windowed on the wrong stagger.
``base``      base state; the horizontally varying members are windowed, the
              purely vertical ones are shared verbatim.
``coord``     the vertical coordinate.  Shared verbatim -- it has no
              horizontal extent, and the eta grid must be identical across
              ranks or the sliced 3-D fields sit on different levels.
``grid``      a ``LambertGrid`` for the window with the SAME projection
              parameters and ``known_x``/``known_y`` shifted by the window
              origin.  Not a new projection: the same one, read at an offset.

Everything else follows, because ``initialize_prepared_physics`` validates its
inputs against ``cfg.ny``/``cfg.nx`` rather than against the domain -- it is
already window-agnostic, and it is what builds the 2-D surface and soil
carriers that a slice of the 3-D state cannot supply.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from tilestream import decomp as _decomp
from tilestream import spec as _spec

__all__ = [
    "RealAnalysis",
    "open_analysis",
    "open_bundle",
    "subwindow_grid",
    "slice_mapping",
    "build_rank_state",
    "domain_setup_flags",
]


class RealInitError(RuntimeError):
    """A real-data decomposition that cannot be built as asked."""


# --------------------------------------------------------------------------
# geometry: the same projection, read at an offset
# --------------------------------------------------------------------------

def subwindow_grid(grid, i0: int, j0: int, nx: int, ny: int):
    """The window's own :class:`LambertGrid`: same projection, moved origin.

    ``known_x``/``known_y`` are the grid coordinates of the reference
    latitude/longitude.  Shifting them by the window origin and leaving
    ``ref_lat``, ``ref_lon``, ``truelat1``, ``truelat2``, ``stand_lon`` and
    ``dx``/``dy`` untouched describes THE SAME map, cropped -- so a column at
    global ``(j, i)`` has the same latitude, longitude, map factor and
    Coriolis parameter whichever rank holds it.

    This is the recipe the boundary-strip initialiser already uses to give a
    W/E/S/N rectangle its own projected grid
    (``tools/hrrr_single_domain_benchmark.py``,
    ``_boundary_mapping_targets``); a decomposition is the same operation with
    a different rectangle.

    ``e_we``/``e_sn`` are STAGGERED counts, hence ``nx + 1`` -- the same
    convention ``HrrrTargetDomain.grid()`` and ``harness.make_geography`` use.
    """
    from gpuwm.static.lambert import LambertGrid

    return LambertGrid(
        e_we=int(nx) + 1, e_sn=int(ny) + 1,
        dx=float(grid.dx), dy=float(grid.dy),
        ref_lat=float(grid.ref_lat), ref_lon=float(grid.ref_lon),
        truelat1=float(grid.truelat1), truelat2=float(grid.truelat2),
        stand_lon=float(grid.stand_lon),
        known_x=float(grid.known_x) - float(i0),
        known_y=float(grid.known_y) - float(j0))


def assert_grid_window_faithful(grid, sub, i0: int, j0: int, *,
                                tol_deg: float = 1e-9) -> dict[str, float]:
    """Prove the sub-grid is the domain grid cropped, not a new projection.

    Compares the window's per-column latitude and longitude against the
    corresponding block of the DOMAIN's, which is the property the whole
    design rests on and the one a re-centred rebuild violates by hundreds of
    kilometres.  Cheap: two ``latlon_mass`` calls and a max.
    """
    lat, lon = grid.latlon_mass()
    slat, slon = sub.latlon_mass()
    ny, nx = slat.shape
    ref_lat = np.asarray(lat)[j0:j0 + ny, i0:i0 + nx]
    ref_lon = np.asarray(lon)[j0:j0 + ny, i0:i0 + nx]
    if ref_lat.shape != slat.shape:
        raise RealInitError(
            f"window at ({j0},{i0}) of shape {slat.shape} does not fit the "
            f"domain grid {np.asarray(lat).shape}")
    dlat = float(np.max(np.abs(np.asarray(slat) - ref_lat)))
    dlon = float(np.max(np.abs(np.asarray(slon) - ref_lon)))
    great_circle_km = 111.32 * max(dlat, dlon)
    if max(dlat, dlon) > tol_deg:
        raise RealInitError(
            f"sub-window grid is NOT the domain grid cropped: max |dlat|="
            f"{dlat:.3e} deg, max |dlon|={dlon:.3e} deg "
            f"(~{great_circle_km:.3f} km).  A re-centred rebuild looks exactly "
            f"like this, and it is the failure this path exists to avoid")
    return dict(max_dlat_deg=dlat, max_dlon_deg=dlon,
                displacement_km=great_circle_km)


# --------------------------------------------------------------------------
# slicing the input mappings
# --------------------------------------------------------------------------

def slice_mapping(source: Mapping[str, Any], spec: _spec.TileSpec, *,
                  nz: int, ny: int, nx: int,
                  label: str = "input") -> dict[str, np.ndarray]:
    """Window every horizontally-extended entry; pass the rest through.

    The stagger is derived from each array's SHAPE
    (:func:`tilestream.gather.classify`), never from a name list, so a field
    that is x-staggered -- ``MAPFAC_U``, ``U10`` -- cannot be windowed as if
    it were mass-centred.  Leading axes that are not vertical (12 months of
    greenness, 4 soil layers, 3 snow layers) ride through as the copy's depth.

    Entries with no horizontal extent (scalars, vertical profiles) are shared
    verbatim: they are domain constants and windowing them would be wrong.
    """
    out: dict[str, Any] = {}
    for name, value in source.items():
        array = np.asarray(value)
        if array.ndim < 2 or array.shape[-1] not in (nx, nx + 1) \
                or array.shape[-2] not in (ny, ny + 1):
            out[name] = value
            continue
        variant = _decomp.variant_of(f"{label}/{name}", array.shape,
                                     nz, ny, nx)
        out[name] = _decomp.slice_array(array, spec, variant)
    return out


# --------------------------------------------------------------------------
# the analysis handle
# --------------------------------------------------------------------------

@dataclass
class RealAnalysis:
    """One global analysis, opened once, sliceable many times.

    Deliberately holds HOST arrays and a cache READER rather than a
    ``DomainState``: a rank must be able to take its window without the global
    domain ever having been resident on a device, which is what lets the
    decomposed domain be larger than one card.
    """

    reader: Any
    cfg: Any
    grid: Any
    static: Mapping[str, np.ndarray]
    met_fields: Mapping[str, np.ndarray]
    surface_fields: Mapping[str, np.ndarray]
    coord_values: Mapping[str, Any]
    base_values: Mapping[str, Any]
    surface_pressure: np.ndarray
    surface_qv: np.ndarray
    state_names: Sequence[str]
    valid_time: Any
    landuse_identity: Mapping[str, Any]
    attrs: Mapping[str, Any] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @property
    def native(self) -> bool:
        """True when physics must come from the HRRR-native initialiser.

        A cache written by the native HRRR route carries the RAW soil in
        ``met`` (``SOILT``/``SOILW``) and no canonical Noah ``surface``
        inventory; the source-neutral route carries the reverse.  Which one
        applies is a property of the cache, so it is read off the cache rather
        than passed in and got wrong.
        """
        return not self.surface_fields

    @property
    def nx(self) -> int:
        return int(self.cfg.nx)

    @property
    def ny(self) -> int:
        return int(self.cfg.ny)

    @property
    def nz(self) -> int:
        return int(self.cfg.nz)

    def summary(self) -> str:
        return (f"RealAnalysis {self.nz}x{self.ny}x{self.nx} valid "
                f"{self.valid_time}, {len(self.state_names)} state arrays, "
                f"{len(self.static)} static, {len(self.met_fields)} met, "
                f"{len(self.surface_fields)} surface")


def open_analysis(prepared_cache, *, cfg, grid, static, valid_time,
                  cache_identity=None, landuse_identity=None,
                  attrs=None) -> RealAnalysis:
    """Open a prepared cache for slicing, on the HOST, with no CUDA.

    ``PreparedCacheReader`` is pure NumPy and validates the header, the
    per-array digests and the inventory before returning a byte.  Reading the
    payload here rather than through ``restore_prepared_cache`` is not a way
    around those checks -- it is the same reader -- it is a way around
    ``restore_prepared_cache``'s unconditional ``import cupy`` and its
    allocation of the WHOLE domain on one device, which is precisely what a
    decomposed run must not do.
    """
    from gpuwm.ingest.prepared_cache import PreparedCacheReader
    from gpuwm.prepared_single_domain_forecast import _LANDUSE_IDENTITY

    reader = PreparedCacheReader(Path(prepared_cache),
                                 expected_identity=cache_identity)
    metadata = reader.header["metadata"]

    coord_values = dict(metadata["coord_scalars"])
    for name in metadata["coord_arrays"]:
        coord_values[name] = reader.read_array(f"coord/{name}")
    base_values = dict(metadata["base_scalars"])
    for name in metadata["base_arrays"]:
        base_values[name] = reader.read_array(f"base/{name}")

    met_fields = {}
    surface_fields = {}
    for key in reader.arrays:
        if key.startswith("met/"):
            met_fields[key[4:]] = reader.read_array(key)
        elif key.startswith("surface/"):
            surface_fields[key[8:]] = reader.read_array(key)

    return RealAnalysis(
        reader=reader, cfg=cfg, grid=grid,
        static=static, met_fields=met_fields, surface_fields=surface_fields,
        coord_values=coord_values, base_values=base_values,
        surface_pressure=reader.read_array("result/surface_pressure"),
        surface_qv=reader.read_array("result/surface_qv"),
        state_names=list(metadata["state_names"]),
        valid_time=valid_time,
        landuse_identity=(landuse_identity if landuse_identity is not None
                          else _LANDUSE_IDENTITY),
        attrs=dict(attrs or {}),
        provenance=dict(prepared_cache=str(prepared_cache),
                        content_sha256=reader.content_sha256,
                        payload_bytes=int(reader.payload_bytes)))


def open_bundle(root) -> RealAnalysis:
    """Open a prepared HRRR bundle as written by ``tools/prepare_hrrr_wrf.py``.

    Everything needed is already inside the bundle and is taken from there
    rather than retyped:

    * the ``RunConfig`` from the prepared cache's own ``identity``
      ``domain_config.run`` -- the exact config the cache was written under, so
      the restore cannot be attempted under a different one;
    * the projection from ``native-geometry-receipt.json``;
    * the static geography from ``native-static.npz``;
    * the initial valid time from the cache metadata.

    A bundle laid out differently can still be opened field by field with
    :func:`open_analysis`; this is the convenience path, not the contract.
    """
    from datetime import datetime, timezone

    from gpuwm.config import RunConfig
    from gpuwm.static.lambert import LambertGrid

    root = Path(root)
    cache = root / "native" / "prepared-cache"
    if not cache.is_dir():
        cache = root if (root / "header.json").is_file() else cache
    header = json.loads((cache / "header.json").read_text())
    identity = header["identity"]
    geometry = json.loads(
        (root / "native-geometry-receipt.json").read_text())["geometry"]

    cfg = RunConfig(**dict(identity["domain_config"]["run"]))
    grid = LambertGrid(
        e_we=int(cfg.nx) + 1, e_sn=int(cfg.ny) + 1,
        dx=float(cfg.dx), dy=float(cfg.dy),
        ref_lat=float(geometry["ref_lat"]), ref_lon=float(geometry["ref_lon"]),
        truelat1=float(geometry["truelat1"]),
        truelat2=float(geometry["truelat2"]),
        stand_lon=float(geometry["stand_lon"]))
    static = {k: v for k, v in np.load(root / "native-static.npz").items()}
    valid_time = datetime.fromisoformat(
        header["metadata"]["user"]["initial_valid_time"])
    if valid_time.tzinfo is None:
        valid_time = valid_time.replace(tzinfo=timezone.utc)
    return open_analysis(cache, cfg=cfg, grid=grid, static=static,
                         valid_time=valid_time, cache_identity=identity,
                         attrs={"CEN_LAT": float(geometry["center_lat"])})


# --------------------------------------------------------------------------
# one rank
# --------------------------------------------------------------------------

def _window_cfg(cfg, spec: _spec.TileSpec):
    from dataclasses import replace

    return replace(cfg, nx=int(spec.cnx), ny=int(spec.cny))


def build_rank_state(analysis: RealAnalysis, spec: _spec.TileSpec, *,
                     verify_grid: bool = True):
    """``(cfg, state, driver, report)`` for ONE rank, from the global analysis.

    Mirrors ``restore_prepared_cache`` + ``initialize_prepared_physics``
    step for step, with every horizontally-extended input replaced by its
    window.  The order is the production order and it is load-bearing:
    ``load_base`` decides whether ``thb/pb/alb/phb`` are 1-D or 3-D and
    whether ``mub`` is retired for ``mub2d``; ``set_map_coriolis`` derives
    ``has_msf``/``rotational``; only then can the prognostic arrays be
    written, and only then does the physics driver have shapes to allocate
    against.

    ``has_msf``/``rotational`` are re-imposed from the DOMAIN afterwards.
    ``set_map_coriolis`` ends by deriving them from ``.any()`` over the arrays
    it was handed, so a rank whose window happens to be uniform -- an ocean
    rank, a rank far from the standard parallels -- would silently take a
    different branch in the Coriolis kernel than its neighbours.
    """
    import cupy as cp

    from gpuwm.core.grid import BaseState, VerticalCoord
    from gpuwm.core.state import DomainState
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics

    nz, ny, nx = analysis.nz, analysis.ny, analysis.nx
    cfg_r = _window_cfg(analysis.cfg, spec)

    grid_r = subwindow_grid(analysis.grid, spec.ci0, spec.cj0,
                            spec.cnx, spec.cny)
    grid_report = (assert_grid_window_faithful(
        analysis.grid, grid_r, spec.ci0, spec.cj0)
        if verify_grid and 0 <= spec.ci0 and 0 <= spec.cj0
        and spec.ci0 + spec.cnx <= nx and spec.cj0 + spec.cny <= ny else None)

    coord = VerticalCoord(**dict(analysis.coord_values))
    base_r = slice_mapping(analysis.base_values, spec, nz=nz, ny=ny, nx=nx,
                           label="base")
    base = BaseState(**base_r)

    static_r = slice_mapping(analysis.static, spec, nz=nz, ny=ny, nx=nx,
                             label="static")
    met_r = slice_mapping(analysis.met_fields, spec, nz=nz, ny=ny, nx=nx,
                          label="met")
    surface_r = slice_mapping(analysis.surface_fields, spec, nz=nz, ny=ny,
                              nx=nx, label="surface")

    state = DomainState(cfg_r)
    state.load_base(coord, base)
    state.set_map_coriolis(
        static_r["MAPFAC_M"], static_r["MAPFAC_U"], static_r["MAPFAC_V"],
        static_r["F"], static_r["E"], sina=static_r["SINALPHA"],
        cosa=static_r["COSALPHA"])

    for name in analysis.state_names:
        target = getattr(state, name, None)
        if target is None:
            raise RealInitError(
                f"prepared cache carries state/{name} but the rank config has "
                f"no such attribute")
        host = analysis.reader.read_array(f"state/{name}")
        variant = _decomp.variant_of(f"state/{name}", host.shape, nz, ny, nx)
        window = _decomp.slice_array(host, spec, variant)
        if tuple(window.shape) != tuple(target.shape):
            raise RealInitError(
                f"state/{name} window {window.shape} != rank array "
                f"{target.shape}")
        target[...] = cp.asarray(window.astype(target.dtype, copy=False))

    result = SimpleNamespace(
        state=state, coord=coord, base=base,
        surface_pressure=_decomp.slice_array(
            analysis.surface_pressure, spec, "mass"),
        surface_qv=_decomp.slice_array(analysis.surface_qv, spec, "mass"))

    met_ns = SimpleNamespace(fields=MappingProxyType(dict(met_r)))
    if analysis.native:
        # The native route derives the canonical Noah surface from the raw
        # HRRR soil in `met` (preprocess_land_surface_soil).  That is a
        # COLUMN-LOCAL transform, so running it on the window gives the same
        # answer it would give on the window of a domain-wide run -- which is
        # why the soil can be sliced as an input rather than having to be
        # sliced as a result.
        from gpuwm.ingest.hrrr_physics import initialize_hrrr_physics

        attrs = dict(analysis.landuse_identity)
        attrs.update(analysis.attrs)
        driver = initialize_hrrr_physics(
            result, cfg_r, met_ns, static_r, attrs, grid_r,
            analysis.valid_time)
    else:
        driver = initialize_prepared_physics(
            result, cfg_r, met_ns,
            SimpleNamespace(fields=MappingProxyType(dict(surface_r))),
            static_r, analysis.landuse_identity, grid_r, analysis.valid_time)

    report = dict(rank=spec.index, cfg=(cfg_r.ny, cfg_r.nx),
                  origin=(spec.cj0, spec.ci0), grid=grid_report,
                  interior=(spec.interior_ny, spec.interior_nx))
    return cfg_r, state, driver, report


def domain_setup_flags(analysis: RealAnalysis) -> dict[str, bool]:
    """``has_msf``/``rotational`` over the WHOLE domain, from the static file.

    Computed from the domain's own map factors and Coriolis rather than from
    any rank's window, and re-imposed on every rank -- see
    ``tilestream.driver.geography_scalars`` for why a per-window derivation is
    a correctness bug rather than an optimisation.
    """
    s = analysis.static

    def any_ne(key: str, value: float) -> bool:
        arr = s.get(key)
        return False if arr is None else bool(np.any(np.asarray(arr) != value))

    has_msf = (any_ne("MAPFAC_M", 1.0) or any_ne("MAPFAC_U", 1.0)
               or any_ne("MAPFAC_V", 1.0))
    rotational = has_msf or any_ne("F", 0.0) or any_ne("E", 0.0)
    return {"has_msf": has_msf, "rotational": rotational}
