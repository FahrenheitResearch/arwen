"""CAM ozone required by nested consumers, independent of radiation selection.

WRF module_radiation_driver.F:1802-1823 evaluates this field on the root
under o3input=2 before spectrum dispatch. Existing legacy root arithmetic
remains the reference; this owner supplies the same chain when a root's
selected radiation does not itself produce CAM ozone.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Integral

import numpy as np


from gpuwm.core.ozone_contract import cam_ozone_domain_ids


# ---------------------------------------------------------------------------
# How a domain obtained its ozone, named once.
# ---------------------------------------------------------------------------
#
# WRF v4.7.1, EM_CORE, under ``o3input = 2``: ``oznini`` reads the packaged
# climatology and interpolates it to a domain's OWN XLAT, and the radiation
# driver runs ``ozn_time_int``/``ozn_p_int`` on that domain's own columns,
# both guarded by ``o3input .EQ. 2 .AND. id .EQ. 1``
# (phys/module_physics_init.F:2203-2212, phys/module_radiation_driver.F:
# 1801-1823).  A NEST does neither; it receives the root's field through the
# parent-to-child forcing stream the Registry declares on the variable
# (``rdf=(p2c)``, Registry/Registry.EM_COMMON:1264).
#
# So the question WRF answers is "is this domain a nest?", not "is this grid
# a refinement of another forecast?".  These four names are the four answers
# gpuwm can give, and each one is written into the radiation identity that
# reaches a restart manifest and into the run's own report.

#: ``o3input = 0``: the legacy wrapper builds O3DATA from pressure and
#: latitude inside lwrad/swrad prep and reads no climatology at all.
ROUTING_WRAPPER_O3DATA = "wrapper-o3data"

#: A resident root (``id == 1``): the climatology chain on its own grid.
ROUTING_ROOT_CLIMATOLOGY = "root-climatology"

#: A resident nest: the parent's retained field, SINT-interpolated onto the
#: child grid, updating exactly when the parent's does.
ROUTING_PARENT_INTERPOLATED = "parent-interpolated"

#: A child integrated with no resident parent (``gpuwm.offline_child_run``,
#: the ndown-equivalent route).  That domain is configured as a WRF ROOT and
#: reads its lateral boundaries from a file rather than from a live parent:
#: ``specified = true``, ``nested = false``, and the route stamps
#: ``parent_id = 0`` on its own ticks.  WRF's own answer for such a domain is
#: the climatology on its own grid, which is what this route evaluates.
#: Named apart from :data:`ROUTING_ROOT_CLIMATOLOGY` because the grid IS a
#: refinement of an archived forecast, and a reader of the report is owed
#: that distinction rather than being left to infer it from ``parent_id``.
ROUTING_CHILD_GRID_CLIMATOLOGY = "child-grid-climatology"

OZONE_ROUTINGS = frozenset((
    ROUTING_WRAPPER_O3DATA, ROUTING_ROOT_CLIMATOLOGY,
    ROUTING_PARENT_INTERPOLATED, ROUTING_CHILD_GRID_CLIMATOLOGY))


def resolve_ozone_routing(declared, *, o3input, has_parent):
    """The routing name an adapter records, checked against how it was built.

    ``declared`` is the constructing route's own word for what it did, or
    ``None`` to take the name from the construction alone.  A declared name
    that contradicts the construction is refused rather than recorded: this
    string is what a report and a restart manifest are read from, and a
    wrong one sends a reader looking for a parent field that was never
    there.  The way out of the refusal is to pass the name that matches the
    construction, or to pass ``None`` and take the derived one.
    """
    derived = (ROUTING_WRAPPER_O3DATA if o3input == 0 else
               ROUTING_PARENT_INTERPOLATED if has_parent else
               ROUTING_ROOT_CLIMATOLOGY)
    if declared is None:
        return derived
    if declared not in OZONE_ROUTINGS:
        raise ValueError(
            f"unknown ozone routing {declared!r}; the recorded names are "
            f"{sorted(OZONE_ROUTINGS)}")
    permitted = {derived}
    if derived == ROUTING_ROOT_CLIMATOLOGY:
        # The climatology-on-its-own-grid arithmetic is one chain; only the
        # route differs, so a route may say which of the two it is.
        permitted.add(ROUTING_CHILD_GRID_CLIMATOLOGY)
    if declared not in permitted:
        raise ValueError(
            f"ozone routing {declared!r} contradicts how this adapter was "
            f"built (o3input={o3input}, "
            f"parent provider {'given' if has_parent else 'absent'}), which "
            f"obtains ozone by {derived!r}; a report carrying the declared "
            "name would send a reader after an ozone source this run never "
            f"had; declare {sorted(permitted)} or pass ozone_routing=None to "
            "take the derived name")
    return declared



def _host(value):
    if type(value).__module__.split(".")[0] == "cupy":
        import cupy as cp
        return cp.asnumpy(value)
    return np.asarray(value)


@dataclass
class CamOzoneState:
    """Rebuilt producer setup; its retained field belongs to PhysicsDriver.

    Root-climatology evaluates CAM. Legacy-root observes the existing
    legacy engine's exact field. Parent-interpolated domains receive the
    common FORCE transfer, so they never evaluate local climatology.
    """
    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    mode: str
    column_chunk: int = 4096


    def __post_init__(self):
        if not isinstance(self.start_time, datetime):
            raise TypeError("CAM ozone start_time must be a datetime")
        if self.mode not in (ROUTING_ROOT_CLIMATOLOGY, "legacy-root",
                             ROUTING_PARENT_INTERPOLATED):
            raise ValueError("unknown CAM ozone producer mode")
        self.latitude_deg = np.ascontiguousarray(_host(self.latitude_deg), dtype=np.float32)
        self.longitude_deg = np.ascontiguousarray(_host(self.longitude_deg), dtype=np.float32)
        if (self.latitude_deg.ndim != 2
                or self.latitude_deg.shape != self.longitude_deg.shape):
            raise ValueError("CAM ozone geography must share one 2-D grid")
        if (isinstance(self.column_chunk, bool) or not isinstance(self.column_chunk, Integral)
                or self.column_chunk < 1):
            raise ValueError("CAM ozone column chunk must be a positive integer")
        self._ozone_climo = None

    def evaluate(self, pressure, elapsed_seconds):
        """The existing WRF CAM chain, in Pa and FP32, on model layers."""
        if self.mode != ROUTING_ROOT_CLIMATOLOGY:
            raise ValueError("only the root CAM producer may evaluate climatology")
        from gpuwm.ingest import wrf_ozone
        p = np.asarray(_host(pressure), dtype=np.float32)
        if p.ndim != 3 or p.shape[1:] != self.latitude_deg.shape:
            raise ValueError("CAM ozone pressure must be (nz, ny, nx) on its geography")
        if self._ozone_climo is None:
            self._ozone_climo = wrf_ozone.load_ozone_climatology()
        valid_time = self.start_time + timedelta(seconds=float(elapsed_seconds))
        julday = valid_time.timetuple().tm_yday
        hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0 + valid_time.microsecond / 3.6e9)
        julian = np.float32((julday - 1) + hour / 24.0)
        nz, ny, nx = p.shape
        columns = np.ascontiguousarray(p.transpose(1, 2, 0).reshape(-1, nz))
        latitude = self.latitude_deg.reshape(-1)
        ozone = np.empty_like(columns)
        # Every CAM column is independent. Bound the monthly interpolation
        # workspace by the existing experiment column cap; retaining all
        # 59*12 coefficients per domain column would add gigabytes to a
        # wide modern/off ancestor solely because one child needs ozone.
        for left in range(0, len(columns), self.column_chunk):
            right = min(len(columns), left + self.column_chunk)
            latitudes = wrf_ozone.interp_ozone_to_latitudes(
                latitude[left:right], self._ozone_climo)
            timed = wrf_ozone.ozn_time_int(julday, julian, latitudes)
            ozone[left:right] = wrf_ozone.ozn_p_int(
                columns[left:right], self._ozone_climo.plev, timed)
        return np.ascontiguousarray(ozone.reshape(ny, nx, nz).transpose(2, 0, 1))

    @property
    def restart_identity(self):
        from gpuwm.ingest.wrf_ozone import OZONE_SHA256, OZONE_LAT_SHA256, OZONE_PLEV_SHA256
        return {
            "algorithm": "wrf-v4.6.1-root-cam-ozone",
            "mode": self.mode,
            "start_time": self.start_time.isoformat(),
            "assets": {"ozone": OZONE_SHA256, "latitude": OZONE_LAT_SHA256,
                       "pressure": OZONE_PLEV_SHA256},
        }


CARRIER_KEY = "radiation/o33d_grid"


class DriverOzoneProvider:
    """Routing marker for a child whose field is carried by its driver."""
    def __call__(self):
        raise RuntimeError("child CAM ozone requires the driver's retained field; "
                           "the parent-to-child FORCE transfer has not been bound")


def attach_cam_ozone(state, cfg, owner):
    """Allocate the one retained device field before restart/store inventory."""
    import cupy as cp
    from gpuwm.core.physics import initialize_physics
    driver = getattr(state, "physics", None)
    if driver is None:
        driver = initialize_physics(state, cfg, cam_ozone=owner)
    else:
        driver.cam_ozone = owner
    if getattr(driver, "o3rad", None) is None:
        driver.o3rad = cp.zeros(state.p.shape, dtype=cp.float32)
    driver.call_counts.setdefault("cam_ozone", 0)
    if owner.mode == ROUTING_PARENT_INTERPOLATED:
        from gpuwm.core.radiation_composition import legacy_radiation_adapter
        legacy = legacy_radiation_adapter(driver.radiation_callable)
        if legacy is not None and legacy.o3input == 2:
            legacy._ozone_provider = DriverOzoneProvider()
    return driver


def cam_ozone_setup(*, exp, dc, grid):
    """Resolve the tree dependency before any driver or store is allocated."""
    if dc.grid_id not in cam_ozone_domain_ids(exp):
        return None
    cfg = dc.run
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    legacy_root = (4 in radiation_scheme_ids(cfg)
                   and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY
                   and cfg.o3input == 2)
    mode = (ROUTING_PARENT_INTERPOLATED if dc.parent_id else
            "legacy-root" if legacy_root else ROUTING_ROOT_CLIMATOLOGY)
    latitude, longitude = grid.latlon_mass()
    return CamOzoneState(exp.start_time, latitude, longitude, mode, exp.column_chunk)


def ozone_parent_for(owner):
    """Use the common retained carrier for a derived nested consumer."""
    return (DriverOzoneProvider() if owner is not None
            and owner.mode == ROUTING_PARENT_INTERPOLATED else None)


def configure_cam_ozone(state, cfg, *, exp, dc, grid):
    """Bind the derived tree dependency without changing a selected scheme."""
    owner = cam_ozone_setup(exp=exp, dc=dc, grid=grid)
    return (getattr(state, "physics", None) if owner is None else
            attach_cam_ozone(state, cfg, owner))


def transfer_parent_ozone(node, registration):
    """SINT the parent's held CAM field through the common FORCE/store seam."""
    from gpuwm.core.nest import parent_footprint_window
    from gpuwm.core.nest_interp import sint
    from gpuwm.core.streaming import (refresh_from_store, commit_to_store, domain_store,
                                      domain_call_counts)
    driver = getattr(node.state, "physics", None)
    owner = getattr(driver, "cam_ozone", None)
    if owner is None or owner.mode != ROUTING_PARENT_INTERPOLATED:
        return 0
    parent = node.parent.state
    parent_driver = getattr(parent, "physics", None)
    source = getattr(parent_driver, "o3rad", None)
    streamed_child = getattr(node.state, "_streamed_domain", None)
    streamed_parent = getattr(parent, "_streamed_domain", None)
    if streamed_child is not None or streamed_parent is not None:
        from gpuwm.core.nest_operands import NestWindowSource, streamed_chunk_shape
        from gpuwm.core.nest_interp import window_registration

        coarse = NestWindowSource(parent)
        fine = NestWindowSource(node.state)
        if coarse.store is not None:
            source = coarse.store.get(CARRIER_KEY)
        if source is None or (fine.store is not None and CARRIER_KEY not in fine.store):
            raise RuntimeError("streamed CAM ozone carrier is missing from the canonical store")
        target = getattr(driver, "o3rad", None)
        if fine.store is None and target is None:
            raise RuntimeError("child CAM ozone FORCE has no retained ozone field to receive its parent")
        parent_counts = domain_call_counts(getattr(parent, "_streamed_domain", None), parent)
        if parent_counts.get("cam_ozone", 0) == 0:
            raise RuntimeError("child CAM ozone FORCE preceded its parent's first ozone producer")
        sy, sx = streamed_chunk_shape(node.state if streamed_child is not None else parent)
        for j in range(0, registration.nyc, sy):
            for i in range(0, registration.nxc, sx):
                window = (slice(j, min(j+sy, registration.nyc)),
                          slice(i, min(i+sx, registration.nxc)))
                cropped, donor = window_registration(registration, window)
                result = sint(coarse.device_array(source, donor), cropped)
                if fine.store is not None:
                    fine.write(CARRIER_KEY, window, result)
                else:
                    target[(...,) + window] = result
        count = domain_call_counts(streamed_child, node.state).get("cam_ozone", 0) + 1
        driver.call_counts["cam_ozone"] = count
        if streamed_child is not None:
            streamed_child.scalars["call_counts"]["cam_ozone"] = count
        return coarse.host_to_device_bytes + fine.device_to_host_bytes
    for state in (parent, node.state):
        store = domain_store(state)
        if store is not None and CARRIER_KEY not in store:
            raise RuntimeError("streamed CAM ozone carrier is missing from the canonical store")
    moved = refresh_from_store(parent, (CARRIER_KEY,),
                               window=parent_footprint_window(node.cfg))
    parent_counts = domain_call_counts(getattr(parent, "_streamed_domain", None), parent)
    if source is None or parent_counts.get("cam_ozone", 0) == 0:
        raise RuntimeError("child CAM ozone FORCE preceded its parent's first "
                           "ozone producer; local climatology cannot replace it")
    sint(source, registration, out=driver.o3rad)
    streamed = getattr(node.state, "_streamed_domain", None)
    count = domain_call_counts(streamed, node.state).get("cam_ozone", 0) + 1
    driver.call_counts["cam_ozone"] = count
    if streamed is not None:
        streamed.scalars["call_counts"]["cam_ozone"] = count
    moved += commit_to_store(node.state, (CARRIER_KEY,))
    return moved


def memory_increment_per_cell(cfg):
    """Increment for the measured planner, including an auxiliary-only driver.

    Active drivers gain exactly one FP32 cell. The otherwise absent driver
    uses the shared allocation inventory at 1x1, an upper bound for staggered
    edge faces at every actual window size. A pressure-only CAM call also
    needs the existing atmosphere work arrays if local physics never called
    that producer before. These are derived bytes, not another measured rung.
    """
    from dataclasses import replace
    from math import prod
    from gpuwm.core.preflight import physics_array_shapes, atmosphere_transient_shapes
    unit = replace(cfg, nx=1, ny=1)
    old = physics_array_shapes(unit)
    new = physics_array_shapes(unit, cam_ozone=True)
    persistent = sum(prod(shape) * 4 for name, shape in new.items() if name not in old)
    old_atmosphere = atmosphere_transient_shapes(unit)
    new_atmosphere = atmosphere_transient_shapes(unit, cam_ozone=True)
    transient = sum(prod(shape) * 4 for name, shape in new_atmosphere.items()
                    if name not in old_atmosphere)
    return ((persistent + transient) / unit.nz, persistent / unit.nz)
