"""WRF v4.6.1 CLM lake column driver on CUDA.

The complete WRF lake module is translated to one CUDA thread per lake
column. WRF uses double precision inside the lake model and single precision
for its persistent grid arrays; this driver preserves that boundary. The
ordinary land-surface scheme still runs first, as in WRF's surface driver.
Only cells in the explicit lake mask are gathered and updated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gpuwm.core.lake_schema import (
    LAKE_DEFAULT_DEPTH,
    LAKE_DEFAULT_USE_DEPTH,
    LAKE_FORCING_WORDS,
    LAKE_OUTPUT_LAYOUT,
    LAKE_OUTPUT_WORDS,
    LAKE_STATE_WORDS,
    LAKE_STATIC_WORDS,
)

LAKE_RESTART_FIELDS = ("lake_columns", "lake_static", "lake_latitude")

#: 8-byte arena slots one lake column thread holds at most; must equal
#: ``LAKE_ARENA_SLOTS`` in kernels/lake_support.cuh (tests/test_lake_contract.py).
LAKE_ARENA_SLOTS = 1283
#: Columns one lake launch steps at once, and so the arena's width.  The
#: arena replaced a per-thread local frame that CUDA backed for every
#: resident thread of the card (3.3 GiB on an RTX 5090); at this width it
#: holds 0.31 GiB, and a domain with fewer lake columns holds less.
LAKE_LAUNCH_COLUMNS = 32768


def lake_arena_bytes(columns: int) -> int:
    """Device bytes of the arena a lake model with ``columns`` columns holds."""
    return min(max(int(columns), 0), LAKE_LAUNCH_COLUMNS) * LAKE_ARENA_SLOTS * 8


def lake_arena(columns: int, held=None):
    """The arena for ``columns`` lake columns, reusing ``held`` when it fits."""
    import cupy as cp

    size = lake_arena_bytes(columns)
    if held is not None and int(held.nbytes) == size:
        return held
    return cp.empty(max(size, 8), cp.uint8)


def launch_lake_columns(kernel, n: int, args: tuple, arena) -> None:
    """Launch a lake column kernel over ``n`` columns, one arena width at a time.

    ``args`` are the kernel's own arguments through ``errors``; the launch
    window and the arena are appended.  Columns are independent, so the
    windows change no column's arithmetic.
    """
    import numpy as np

    n = int(n)
    for col0 in range(0, n, LAKE_LAUNCH_COLUMNS):
        count = min(LAKE_LAUNCH_COLUMNS, n - col0)
        kernel(((count + 31) // 32,), (32,),
               (*args, np.int32(col0), np.int32(count), arena))


@dataclass
class LakeModel:
    """Sparse computation with horizontally addressable restart arrays.

    The persisted arrays have trailing (y, x) dimensions so the ordinary
    resident, tile and device-rank carrier preserves them without special
    packing. Only lake columns are copied into the working arrays below.
    """

    indices: Any
    columns: Any
    static: Any
    latitude: Any
    forcing: Any
    output: Any
    errors: Any
    _step_kernel: Any = None
    _needs_refresh: bool = False
    _arena: Any = None
    #: WRF's xice_threshold as module_surface_driver.F:1365-1368 selects it
    #: (0.5, or 0.02 under fractional_seaice = 1); lakeini and lake read
    #: the same value the land-surface seam runs.
    xice_threshold: float = 0.5

    @property
    def column_count(self) -> int:
        return int(self.indices.size)

    def refresh_columns(self, fields) -> None:
        """Rebuild sparse work after a horizontal tile or rank gather."""
        import cupy as cp

        indices = cp.flatnonzero(fields["lakemask"].reshape(-1) == 1).astype(cp.int32)
        n = int(indices.size)
        if self.columns is None or self.columns.shape[1] != n:
            # Drop the old tile's work before allocating a changed occupancy.
            # Otherwise two complete lake work sets overlap at a tile switch.
            self.columns = self.static = self.latitude = None
            self.forcing = self.output = self.errors = None
            self.columns = cp.empty((LAKE_STATE_WORDS, n), cp.float32)
            self.static = cp.empty((LAKE_STATIC_WORDS, n), cp.float32)
            self.latitude = cp.empty(n, cp.float32)
            self.forcing = cp.empty((LAKE_FORCING_WORDS, n), cp.float32)
            self.output = cp.empty((LAKE_OUTPUT_WORDS, n), cp.float32)
            self.errors = cp.empty(n, cp.int32)
            self._arena = None
            self._arena = lake_arena(n)
        self.indices = indices
        cp.take(fields["lake_columns"].reshape(LAKE_STATE_WORDS, -1),
                indices, axis=1, out=self.columns)
        cp.take(fields["lake_static"].reshape(LAKE_STATIC_WORDS, -1),
                indices, axis=1, out=self.static)
        cp.take(fields["lake_latitude"].reshape(-1), indices, out=self.latitude)
        self.errors.fill(0)
        self._needs_refresh = False

    def invalidate_columns(self) -> None:
        """Defer sparse gathering until prior carrier copies have completed."""
        self._needs_refresh = True

    def step(self, fields, atmosphere, *, dt: float, precipitation) -> None:
        """Advance the complete lake model and replace its surface outputs.

        ``precipitation`` is the accumulated water amount in mm before the
        ordinary LSM clears RAINBL. The WRF driver divides it by dt itself.
        Temperature, pressure, layer thickness and water-vapor mixing ratio
        are the ordinary ARW atmospheric arrays in (level, y, x) order.
        """
        import cupy as cp
        from gpuwm.core.kernels import get_kernel

        if self._needs_refresh:
            self.refresh_columns(fields)
        n = self.column_count
        if n == 0:
            return
        if dt <= 0:
            raise ValueError("Lake time step must be positive to convert accumulated precipitation to a rate")
        cp.take(fields["lake_columns"].reshape(LAKE_STATE_WORDS, -1),
                self.indices, axis=1, out=self.columns)
        p_interface = atmosphere["p_interface"]
        values = (
            atmosphere["temperature"][0], p_interface[0], p_interface[1],
            atmosphere["dz"][0], atmosphere["qv"][0], atmosphere["u"][0],
            atmosphere["v"][0], fields["glw"], fields["emiss"], precipitation,
            fields["swdown"], fields["albedo"],
        )
        for row, value in enumerate(values):
            self.forcing[row] = cp.asarray(value).reshape(-1)[self.indices]
        self.forcing[12] = self.latitude
        if self._step_kernel is None:
            self._step_kernel = get_kernel("lake", "lake_step_columns")
        self._arena = lake_arena(n, self._arena)
        launch_lake_columns(
            self._step_kernel, n,
            (n, self.forcing, self.columns, self.static, self.output,
             cp.float32(dt), cp.float32(self.xice_threshold), self.errors),
            self._arena)
        self.check_errors()
        fields["lake_columns"].reshape(LAKE_STATE_WORDS, -1)[:, self.indices] = self.columns
        for row, (name, _) in enumerate(LAKE_OUTPUT_LAYOUT):
            fields[name].reshape(-1)[self.indices] = self.output[row]

    def check_errors(self) -> None:
        """Raise the WRF fatal condition before committing surface fluxes."""
        if self.column_count:
            code = int(self.errors.max().item())
            if code:
                raise RuntimeError(
                    f"WRF CLM lake rejected a column at module_sf_lake.F:{code}; "
                    "its state and fluxes were not committed")


def initialize_lake(
        fields, *, latitude, lake_depth=None, lake_depth_flag=None,
        use_lakedepth=LAKE_DEFAULT_USE_DEPTH,
        lakedepth_default=LAKE_DEFAULT_DEPTH, iswater=17,
        xice_threshold: float = 0.5) -> LakeModel:
    """Run WRF lakeini for the explicit mask and retain restart storage.

    WRF defaults to using the bathymetry field. A missing bathymetry dataset
    fails by name; ``use_lakedepth=0`` explicitly selects the WRF 50 m default.
    Nonpositive values in an available field also use lakedepth_default,
    exactly as WRF does. A missing lake mask means no lake columns.
    ``xice_threshold`` is the run's sea-ice threshold (0.5, or 0.02 under
    ``fractional_seaice = 1``), read by lakeini and by every lake step.
    """
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    threshold = float(xice_threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"lake xice_threshold must be in (0, 1], got {xice_threshold!r}")
    shape = fields["tsk"].shape
    mask = cp.asarray(fields["lakemask"]) if "lakemask" in fields else cp.zeros(shape, cp.float32)
    if mask.shape != shape:
        raise ValueError("Lake mask must cover the surface grid to avoid shifting lake fluxes onto land")
    fields.setdefault("lakemask", mask)
    indices = cp.flatnonzero(mask.reshape(-1) == 1).astype(cp.int32)
    n = int(indices.size)
    if use_lakedepth not in (0, 1):
        raise ValueError(
            f"use_lakedepth={use_lakedepth!r} must be 0 (lakedepth_default "
            "everywhere) or 1 (the LAKE_DEPTH bathymetry): WRF's lakeini "
            "defines no other depth rule, so another value would start the "
            "lake columns at depths the run did not name")
    if lake_depth is None:
        lake_depth = fields.get("lake_depth")
    has_depth = lake_depth is not None
    depth_flag = int(has_depth) if lake_depth_flag is None else int(lake_depth_flag)
    if n and use_lakedepth == 1 and (not has_depth or depth_flag == 0):
        raise ValueError(
            "sf_lake_physics=1 with use_lakedepth=1 requires LAKE_DEPTH; "
            "missing bathymetry would silently replace the requested lake depths. "
            "Prepare lake-depth geography or set use_lakedepth=0 for the WRF default depth")
    if latitude is None:
        if n:
            raise ValueError("Lake columns require latitude for the latitude-dependent eddy diffusivity")
        lat = cp.zeros(shape, cp.float32)
    else:
        lat = cp.asarray(latitude, dtype=cp.float32)
        if lat.ndim == 0:
            lat = cp.full(shape, lat, cp.float32)
        elif lat.shape == shape:
            lat = cp.ascontiguousarray(lat)
        else:
            raise ValueError("Lake latitude must cover the surface grid for the latitude-dependent eddy diffusivity")
        if n and not bool(cp.all(cp.isfinite(lat.reshape(-1)[indices])).item()):
            raise ValueError("Lake latitude must be finite for the latitude-dependent eddy diffusivity")
    present = [name in fields for name in LAKE_RESTART_FIELDS]
    if any(present) and not all(present):
        raise ValueError("Incomplete lake restart would discard water, ice or sediment heat storage")
    if all(present):
        if (fields["lake_columns"].shape != (LAKE_STATE_WORDS, *shape)
                or fields["lake_static"].shape != (LAKE_STATIC_WORDS, *shape)
                or fields["lake_latitude"].shape != shape):
            raise ValueError(
                "Lake restart dimensions disagree with the surface grid: "
                f"lake_columns {tuple(fields['lake_columns'].shape)}, "
                f"lake_static {tuple(fields['lake_static'].shape)} and "
                f"lake_latitude {tuple(fields['lake_latitude'].shape)} must "
                f"be ({LAKE_STATE_WORDS}, ...), ({LAKE_STATIC_WORDS}, ...) "
                f"and (...) over {tuple(shape)}. The lake kernels index "
                "this storage by surface cell, so a restart from another "
                "grid would hand each lake column another cell's water, "
                "ice and sediment state")
        if (not all(fields[name].dtype == cp.float32 for name in LAKE_RESTART_FIELDS)
                or not all(fields[name].flags.c_contiguous for name in LAKE_RESTART_FIELDS)):
            raise ValueError("Lake restart requires contiguous float32 arrays to preserve the kernel memory layout")
    else:
        fields["lake_columns"] = cp.zeros((LAKE_STATE_WORDS, *shape), cp.float32)
        fields["lake_static"] = cp.zeros((LAKE_STATIC_WORDS, *shape), cp.float32)
        fields["lake_latitude"] = lat
    owner = LakeModel(None, None, None, None, None, None, None,
                      xice_threshold=threshold)
    owner.refresh_columns(fields)
    if n and not all(present):
        seed = cp.empty((5, n), cp.float32)
        for row, name in ((0, "isltyp"), (2, "tsk"), (3, "snow"), (4, "xice")):
            if name in fields:
                seed[row] = cp.asarray(fields[name]).reshape(-1)[indices]
            elif name in ("snow", "xice"):
                seed[row].fill(0)
            else:
                raise ValueError(f"Lake initialization requires {name} for its sediment or thermal state")
        if bool(cp.any((seed[0] < 1) | (seed[0] > 19)).item()):
            raise ValueError("Lake soil categories must be in WRF's 1..19 table to avoid out-of-bounds sediment properties")
        if has_depth:
            depth = cp.asarray(lake_depth, dtype=cp.float32)
            if depth.shape != shape:
                raise ValueError("LAKE_DEPTH must cover the surface grid to avoid assigning bathymetry to the wrong lake")
            seed[1] = depth.reshape(-1)[indices]
        else:
            seed[1].fill(0)
        if not bool(cp.all(cp.isfinite(seed)).item()):
            raise ValueError(
                "Lake initial soil, depth, temperature, snow and ice must "
                "be finite: lakeini builds every layer temperature and "
                "thickness of a lake column from them, so one value that "
                "is not finite makes that column's whole initial profile "
                "not finite")
        owner._arena = lake_arena(n, owner._arena)
        launch_lake_columns(
            get_kernel("lake", "lake_init_columns"), n,
            (n, seed, owner.columns, owner.static, int(use_lakedepth), depth_flag,
             cp.float32(lakedepth_default), cp.float32(threshold), owner.errors),
            owner._arena)
        owner.check_errors()
        fields["lake_columns"].reshape(LAKE_STATE_WORDS, -1)[:, indices] = owner.columns
        fields["lake_static"].reshape(LAKE_STATIC_WORDS, -1)[:, indices] = owner.static
        # WRF lakeini:5040-5046 converts frozen masked lake points from the
        # generic sea-ice branch to lake water before the first surface call,
        # at the run's threshold (the HRRR v4.1.21 fork's surface driver
        # makes the same handover at :1452-1461, ``xice > xice_threshold``),
        # so under fractional_seaice = 1 a lake cell above 0.02 leaves the
        # sea-ice seam exactly as the lake kernel above saw it leave.
        frozen = seed[4] > cp.float32(threshold)
        for name, value in (("ivgtyp", int(iswater)), ("xland", 2.0), ("xice", 0.0)):
            if name in fields:
                target = fields[name].reshape(-1)
                target[indices] = cp.where(frozen, value, target[indices])
    return owner
