"""Measure the health-descriptor inventory of a configured experiment.

``gpuwm.core.health.collect_state_fields`` auto-walks ``driver.fields``, so
every admitted physics scheme silently adds descriptors, and the inventory is
capped by ``MAX_HEALTH_FIELDS``.  That cap is enforced *inside*
``collect_state_fields``, which the forecast calls on the first synchronized
boundary -- i.e. after the run has allocated, ingested and started.  A
configuration that exceeds it is a configuration a user can select and then
lose a forecast to.  Nothing measured the count, so this does.

WHAT IS MEASURED, AND WHY THIS IS NOT AN ESTIMATE
-------------------------------------------------
The census runs the production constructors -- ``DomainState`` and
``initialize_physics`` -- at the experiment's real dimensions, materializes
every persistent scratch slot the authoritative preflight manifests declare
for that configuration, installs a child's rolling nest boundary through the
same ``attach_nest_boundaries`` its first FORCE calls, and then calls the
production ``collect_state_fields``.  The number reported is what that
function returns.

Calibration, so this is checkable rather than merely asserted: the one
descriptor count previously recorded anywhere in this repository is
``health.py``'s own "a four-domain NSSL-2 step currently reaches 527
descriptors", and this census reproduces 527 exactly, on the four-domain
reference case, at ``mp18-lsm2-pbl0-sfclay1-cu0``.

The cap is applied PER ``DomainState``: ``gpuwm/core/model.py`` builds one
``StateHealthValidator`` per domain, each with its own descriptor tables, and
``collect_state_fields`` walks exactly one state (production passes no
``extra_tables``).  Counts therefore do NOT sum across a nest; a four-domain
experiment's exposure is its single largest domain.  Nor are they a per-domain
delta times the domain count: the ROOT and a CHILD are structurally different
inventories -- measured 229 vs 601 on the worst selectable combination -- so
multiplying either one by four measures nothing that exists.  The census
reports the per-domain counts and their peak.

NO DEVICE IS OPENED
-------------------
A NumPy-backed module is installed as ``cupy`` in ``sys.modules`` before any
gpuwm import, so ``import cupy as cp`` in the production modules binds NumPy.
Every constructor then runs unchanged on host arrays.  Installation refuses to
proceed if the real CuPy has already been imported, so the census can never
fall back to the device, and every measurement additionally asserts that the
arrays it just built are ``numpy.ndarray``.

Host cost: about 0.23 GiB resident and ~0.05 s per domain at full dimensions
-- but ONLY because every measurement collects its own garbage.  The states
are not free-on-scope-exit (``state.physics``/``driver.state`` is a reference
cycle) and they are not demand-zero either (the constructors write into them),
so without an explicit ``gc.collect()`` the sweep grows by ~0.7 GiB per
four-domain selection and dies of ``MemoryError`` partway through the matrix.

The two land-surface host cold starts (``NOAHMP_INIT``/``SNOW_INIT`` and
``ruclsminit``) are replaced by no-ops for the census.  Both are per-column
Python loops over the whole slab -- minutes at 600x600 -- and neither adds or
removes a ``fields`` key: every write is into an array
``initialize_physics`` has already allocated.  The descriptor inventory is
therefore identical with and without them.

WHAT THE CEILING COSTS
----------------------
``MAX_HEALTH_FIELDS`` sizes seven fixed ``integration_health_*`` device slots
per ``DomainState`` -- 48 KiB per domain at 1024, reported as
``ceiling_metadata_bytes_per_domain``.  That is the entire resource cost:
raising the ceiling to 2048 would add 48 KiB per domain, 192 KiB across a
four-domain nest.  The ceiling is a tripwire against unnoticed inventory
growth, not a VRAM constraint, and anyone who needs to raise it should say so
rather than reach for the descriptor count.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np

#: The census never touches a device, so make the ban explicit for anything it
#: imports that consults it.
os.environ.setdefault("GPUWM_NO_LOCAL_GPU", "1")

# ``gpuwm`` is installed editable, and its finder points at the checkout that
# ran ``pip install -e``, NOT at this file's tree.  Run as a script from a
# worktree, ``sys.path[0]`` is ``tools/``, which contains no ``gpuwm``, so the
# editable finder wins and the census silently measures a DIFFERENT checkout.
# With three worktrees open on this repository that is not a hypothetical: it
# is how the first run of this tool reported an inventory for code nobody was
# editing.  Claim the repository that owns this file, before any gpuwm import.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class DeviceAlreadyImportedError(RuntimeError):
    """The real CuPy is loaded, so the host backend cannot be guaranteed."""


def install_host_array_backend() -> None:
    """Bind ``cupy`` to NumPy for this interpreter, or refuse.

    Fails closed.  If the real CuPy is already in ``sys.modules`` then some
    production module may have captured it as its ``cp``, and every
    allocation this census makes would land on a device.  On the workstation
    that owns this repository that is not a performance question.
    """
    existing = sys.modules.get("cupy")
    if existing is not None:
        if getattr(existing, "__gpuwm_host_backend__", False):
            return
        raise DeviceAlreadyImportedError(
            "the real cupy is already imported; run the census in a fresh "
            "interpreter so no production module can hold a device handle")

    stub = types.ModuleType("cupy")
    stub.__gpuwm_host_backend__ = True

    def _delegate(name):
        try:
            return getattr(np, name)
        except AttributeError as exc:
            raise AttributeError(
                f"gpuwm host array backend has no {name!r}; the census does "
                "not execute kernels, so a CuPy-only symbol reaching it means "
                "a constructor started computing") from exc

    stub.__getattr__ = _delegate
    stub.asnumpy = lambda value, **kwargs: np.asarray(value)

    def _no_device(*args, **kwargs):
        raise DeviceAlreadyImportedError(
            "gpuwm host array backend: this census opens no CUDA device")

    runtime = types.ModuleType("cupy.cuda.runtime")
    runtime.getDeviceCount = _no_device
    runtime.memGetInfo = _no_device
    runtime.deviceSynchronize = _no_device
    cuda = types.ModuleType("cupy.cuda")
    cuda.runtime = runtime
    stub.cuda = cuda
    sys.modules["cupy"] = stub
    sys.modules["cupy.cuda"] = cuda
    sys.modules["cupy.cuda.runtime"] = runtime


# ---------------------------------------------------------------------------
# What a user can select.  Derived from the production tables, never listed
# here, so an admitted scheme joins the census the moment it is routed.
# ---------------------------------------------------------------------------

def selectable_slot_values(selector: str) -> tuple[int, ...]:
    """Routed values for one surface/LSM/PBL selector.

    ``PHYSICS_SLOT_DISPATCH`` is the authority the driver itself dispatches
    on: a value absent from it raises ``UnroutedPhysicsSelectorError`` at
    driver construction, so it is exactly the set a user can run.
    """
    from gpuwm.core.physics import PHYSICS_SLOT_DISPATCH

    return tuple(sorted(PHYSICS_SLOT_DISPATCH[selector]))


def selectable_scalar_values(cfg, selector: str,
                             candidates=range(0, 128)) -> tuple[int, ...]:
    """Values of ``selector`` the production config validator accepts.

    Probing beats transcribing: the validator's accepted set lives in a
    ``raise`` inside :func:`gpuwm.config.validate_run_config`, and a copy of
    it here would go stale the first time a scheme is admitted.
    """
    from gpuwm.config import validate_run_config

    accepted = []
    for value in candidates:
        try:
            validate_run_config(dataclasses.replace(cfg, **{selector: value}))
        except Exception:
            continue
        accepted.append(int(value))
    return tuple(accepted)


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Selection:
    """One combination a user can choose, for one domain.

    ``km_opt`` is here for a reason that is not obvious: it is not a physics
    scheme, but the reference configuration's ``km_opt=4`` makes
    ``bl_pbl_physics=0`` unselectable ("WRF diff_opt=2 also runs
    vertical_diffusion_2 when the PBL is off"), and ``km_opt=1`` unlocks it.
    A matrix that fixed ``km_opt`` at the reference value would silently be
    reporting the maximum over a SUBSET of what a user can select.  The
    no-PBL corner turns out to be smaller, but that is a measurement, not an
    assumption.
    """

    mp_physics: int
    sf_surface_physics: int
    bl_pbl_physics: int
    sf_sfclay_physics: int
    cu_physics: int
    km_opt: int

    def key(self) -> str:
        return (f"mp{self.mp_physics}"
                f"-lsm{self.sf_surface_physics}"
                f"-pbl{self.bl_pbl_physics}"
                f"-sfclay{self.sf_sfclay_physics}"
                f"-cu{self.cu_physics}"
                f"-km{self.km_opt}")


def resolved_config(dc, selection: Selection):
    """The domain's RunConfig with one selection applied and validated."""
    from gpuwm.config import validate_run_config, validated_soil_layer_count

    cfg = dataclasses.replace(
        dc.run,
        mp_physics=selection.mp_physics,
        sf_surface_physics=selection.sf_surface_physics,
        bl_pbl_physics=selection.bl_pbl_physics,
        sf_sfclay_physics=selection.sf_sfclay_physics,
        cu_physics=selection.cu_physics,
        km_opt=selection.km_opt,
        num_soil_layers=validated_soil_layer_count(
            selection.sf_surface_physics))
    return validate_run_config(cfg)


def _attach_driver(state, cfg, start_time: datetime, latitude_deg: float,
                   longitude_deg: float):
    """``initialize_physics`` plus the persistent arrays step one adds.

    The extras mirror ``gpuwm.core.preflight._materialize_physics``: the
    composed cumulus/PBL optional tendency components are allocated on the
    first due call, not at construction, and they are descriptors.  Without
    them the census would understate a real run's steady state.
    """
    import gpuwm.core.physics as physics_mod
    from gpuwm.config import radiation_enabled

    grid = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
    cold_starts = {name: getattr(physics_mod, name)
                   for name in dir(physics_mod)
                   if name.endswith("_cold_start")}
    for name in cold_starts:
        setattr(physics_mod, name, lambda *a, **k: None)
    try:
        driver = physics_mod.initialize_physics(
            state, cfg, landmask=1.0, tsk=290.0, soil_temperature=285.0,
            soil_moisture=0.30, ivgtyp=10, isltyp=6,
            radiation=(lambda **kw: None) if radiation_enabled(cfg) else None,
            cumulus=(lambda **kw: None) if cfg.cu_physics else None,
            radiation_start_time=start_time,
            radiation_latitude=grid + latitude_deg,
            radiation_longitude=grid + longitude_deg,
            noahmp_start_time=start_time,
            noahmp_latitude=grid + latitude_deg,
            noahmp_longitude=grid + longitude_deg)
    finally:
        for name, function in cold_starts.items():
            setattr(physics_mod, name, function)

    def zero_mass():
        return np.zeros(state.p.shape, dtype=np.float32)

    if cfg.cu_physics:
        target = (driver.pbl_tendencies
                  if physics_mod.physics_reuses_pbl_composition(cfg)
                  else driver.tendencies)
        for component in ("rqr", "rqi", "rqs"):
            if getattr(driver.cumulus_tendencies, component) is None:
                setattr(driver.cumulus_tendencies, component, zero_mass())
            if getattr(target, component) is None:
                setattr(target, component, zero_mass())
    if cfg.bl_pbl_physics and cfg.mp_physics in (6, 8, 10, 18):
        if driver.pbl_tendencies.rqi is None:
            driver.pbl_tendencies.rqi = zero_mass()
        if ((radiation_enabled(cfg) or cfg.cu_physics)
                and not physics_mod.physics_reuses_pbl_composition(cfg)):
            if driver.tendencies.rqi is None:
                driver.tendencies.rqi = zero_mass()
    return driver


def _materialize_persistent_scratch(state, cfg, dc, parent_dc,
                                    spec_bdy_width: int,
                                    n_lbc_intervals: int) -> None:
    """Allocate every persistent scratch slot a running domain holds.

    Slot names and shapes come from the preflight manifests
    (``scratch_slot_registry`` for the per-domain families, the F4/F16 nest
    manifest for a child's rolling boundary and SINT geometry tables), which
    are the same declarations the VRAM preflight and the arena allocator
    consume.  Nothing is listed here, so a slot family added later is counted
    here without this file changing.

    This matters: a child's ``nest_*`` slots are bound lazily on its first
    FORCE, so a census that only constructed state and physics would miss the
    single largest descriptor family in the whole inventory.
    """
    from gpuwm.core.preflight import (nest_slot_dtypes, nest_slot_shapes,
                                      scratch_slot_registry)

    for slot, shape in sorted(
            scratch_slot_registry(
                cfg, n_lbc_intervals=n_lbc_intervals).items()):
        state.scratch(shape, slot)
    if parent_dc is None:
        return
    shapes = nest_slot_shapes(dc, spec_bdy_width, parent_dc)
    dtypes = nest_slot_dtypes(dc, spec_bdy_width, parent_dc)
    for slot, shape in sorted(shapes.items()):
        state.scratch(shape, slot, dtype=np.dtype(dtypes[slot]))


def _attach_rolling_nest_boundary(state, cfg, clock, spec_bdy_width: int
                                  ) -> None:
    """Install the child's rolling boundary the way its first FORCE does.

    THIS WAS THE CENSUS'S BIGGEST BLIND SPOT.  A child's descriptor inventory
    is not complete when its ``nest_*`` scratch exists: ``NestCoupler.force``
    then calls ``attach_nest_boundaries``, which sets
    ``state._lateral_boundary_device``, and ``collect_state_fields`` walks that
    object for every child (a child has no ``lbc_forcing_tables``, so it takes
    the unpacked ``_walk_arrays`` branch, not the one-descriptor packed
    branch).  A census that stopped at scratch reported a child inventory that
    no running child ever has -- and it under-reported it, which is the
    dangerous direction.

    The tables handed to ``attach_nest_boundaries`` are the SAME arrays as the
    ``nest_{kind}_b*`` scratch slots (``NestCoupler._rolling_out`` hands out
    exactly those slots), so these descriptors are aliases: the same device
    memory is described twice.  That does not make them free -- every alias
    occupies a descriptor slot and the ceiling counts slots.

    Side and rename tables are IMPORTED from the coupler rather than
    transcribed, so a future kind or side joins the census automatically.
    """
    from gpuwm.core.nest import _APPLICATION_NAME, _SIDES
    from gpuwm.core.preflight import nest_field_kinds
    from gpuwm.ingest.lateral_bc import (_active_device_interval,
                                         _resident_weights,
                                         attach_nest_boundaries)

    scratch = state._scratch
    fields = {}
    for kind in nest_field_kinds(cfg):
        fields[_APPLICATION_NAME.get(kind, kind)] = {
            side: (scratch[f"nest_{kind}_b{suffix}"],
                   scratch[f"nest_{kind}_bt{suffix}"])
            for side, suffix in _SIDES}
    attach_nest_boundaries(
        state, fields, clock=clock, spec_bdy_width=spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    # The Davies weight cache is lazily filled on the first relaxation, and
    # for a child it hangs off ``resident.weights`` -- which _walk_arrays
    # descends, so it is descriptors too.  Both production call sites pass
    # identical (width, zones, dt, spec_exp, wrf_real=nested) arguments, so a
    # domain accrues exactly ONE key; call it once, with the production
    # argument expressions, rather than asserting that it is only two
    # descriptors.
    device_interval, _dtbc, dt, spec_exp = _active_device_interval(state, cfg)
    _resident_weights(
        state, device_interval.fields["u"].west.value.shape[-1],
        cfg.spec_zone, cfg.relax_zone, dt, spec_exp,
        wrf_real=bool(cfg.nested))


def domain_descriptor_names(dc, parent_dc, selection: Selection, *,
                            spec_bdy_width: int, n_lbc_intervals: int,
                            start_time: datetime, clock=None,
                            latitude_deg: float = 40.0,
                            longitude_deg: float = -84.0) -> tuple[str, ...]:
    """Descriptor names ``collect_state_fields`` returns for one domain.

    Returns names, not the ``HealthField`` objects, on purpose: a field holds
    the array, and one d04 state at 600x600x49 is around 0.7 GiB.  Returning
    the arrays to a caller that sweeps a matrix is how the census ran the host
    out of memory.
    """
    from gpuwm.core.health import collect_state_fields
    from gpuwm.core.state import DomainState

    cfg = resolved_config(dc, selection)
    state = DomainState(cfg)
    try:
        _attach_driver(state, cfg, start_time, latitude_deg, longitude_deg)
        _materialize_persistent_scratch(
            state, cfg, dataclasses.replace(dc, run=cfg), parent_dc,
            spec_bdy_width, n_lbc_intervals)
        if parent_dc is not None:
            _attach_rolling_nest_boundary(state, cfg, clock, spec_bdy_width)
        _assert_host_backed(state)
        # backend="gpu" is the production path: it is what
        # StateHealthValidator collects, and it differs from "cpu" in how
        # resident LBC tables are counted.
        return tuple(field.name
                     for field in collect_state_fields(state, backend="gpu"))
    finally:
        # DomainState <-> PhysicsDriver is a reference cycle (``state.physics``
        # and ``driver.state``), so every measured state is unreachable-but-
        # uncollected until the generational GC happens to run.  CPython's GC
        # triggers on object COUNT, and a domain is a few hundred objects
        # holding hundreds of megabytes each, so it does not run nearly often
        # enough: measured growth was +0.70 GiB per four-domain selection,
        # which reached MemoryError partway through the 140-combination
        # matrix.  Those MemoryErrors were then recorded as "the production
        # validators refuse", i.e. a host-memory failure was reported as
        # evidence that a configuration is not user-selectable -- which is
        # exactly backwards, because an unmeasured combination is the one that
        # could be over the ceiling.  Collect explicitly.
        del state
        gc.collect()


def _assert_host_backed(state) -> None:
    """Fail if any measured array is not a host array.

    The census's whole safety argument is that no device is opened.  Asserting
    the negative (cupy is not imported) is weaker than asserting the positive
    (the arrays we just built are NumPy), so do both.
    """
    for name in ("u", "thp", "p"):
        value = getattr(state, name, None)
        if value is None:
            continue
        if type(value) is not np.ndarray:
            raise DeviceAlreadyImportedError(
                f"state.{name} is {type(value)!r}, not numpy.ndarray; the "
                "host array backend was not in force and this census may "
                "have allocated on a device")


#: Descriptor-name prefix -> family, for the per-family breakdown.  Ordered:
#: the first matching prefix wins.
_FAMILY_PREFIXES = (
    ("nest.scratch.", "nest.scratch"),
    ("nest.", "nest.tables"),
    ("lbc.", "lbc"),
    ("held.", "held"),
    ("surface.microphysics", "surface.microphysics"),
    ("surface.", "surface"),
)


def family_breakdown(names) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        family = "state"
        for prefix, label in _FAMILY_PREFIXES:
            if name.startswith(prefix):
                family = label
                break
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# The whole selectable matrix
# ---------------------------------------------------------------------------

def load_any_experiment(path: Path):
    """Load an experiment TOML with or without a ``[case_data]`` table.

    ``[case_data]`` is popped and DISCARDED rather than validated.  The
    descriptor inventory is a property of the configuration's shape, not of
    whether this machine happens to hold the case's forcing files, and
    ``load_experiment_case`` refuses to return an experiment whose declared
    forcing is absent.  Insisting on the data made the census unable to
    measure the very configuration whose 527-descriptor count is quoted in
    ``health.py``'s own ceiling justification -- a measurement tool that
    cannot measure the cited case is not much of a tool.  Everything the
    census reads (dimensions, nest topology, per-domain physics, spec_bdy_
    width, run_seconds) comes from the validated ``ExperimentConfig``.
    """
    import tomllib

    from gpuwm.config import load_config
    from gpuwm.experiment import (build_experiment, experiment_from_run_config,
                                  is_experiment_toml)

    if is_experiment_toml(path):
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        raw.pop("case_data", None)
        return build_experiment(raw, source=str(path))
    return experiment_from_run_config(load_config(path), datetime(1970, 1, 1))


def selectable_matrix(reference_cfg) -> tuple[Selection, ...]:
    """Every (mp, LSM, PBL, surface-layer, cumulus, km_opt) a user can select.

    Cumulus is part of the matrix because it is per-domain in the
    configuration and it moves the count: an active cumulus scheme carries
    its own held tendency stack.  A reference case that enables it only on
    the outer domain is not the bound -- a user may enable it on the nest
    that also carries the boundary tables.

    Every axis is derived from a production table or probed against the
    production validator; nothing is transcribed, so a scheme admitted after
    this file was written joins the matrix without editing it.
    """
    from gpuwm.config import validated_soil_layer_count

    microphysics = selectable_scalar_values(reference_cfg, "mp_physics")
    cumulus = selectable_scalar_values(reference_cfg, "cu_physics")
    turbulence = selectable_scalar_values(reference_cfg, "km_opt")
    selections = []
    for mp in microphysics:
        for lsm in selectable_slot_values("sf_surface_physics"):
            for pbl in selectable_slot_values("bl_pbl_physics"):
                for sfclay in selectable_slot_values("sf_sfclay_physics"):
                    for cu in cumulus:
                        for km in turbulence:
                            try:
                                validated_soil_layer_count(lsm)
                            except Exception:
                                continue
                            selections.append(
                                Selection(mp, lsm, pbl, sfclay, cu, km))
    return tuple(selections)


#: Exception types the production configuration/driver validators use to say
#: "no user can run this".  Anything else escaping a measurement is a failure
#: of the census and must not be filed as a refusal.
_REFUSAL_TYPES = (ValueError, NotImplementedError, KeyError, TypeError)


def _is_ceiling_breach(exc: BaseException) -> bool:
    """Whether ``exc`` is ``collect_state_fields`` refusing at the ceiling.

    Detected structurally -- innermost frame is that function, in health.py --
    rather than by message text.  It matters that this is not lumped in with
    refusals: the breach is raised as a bare ``ValueError`` from inside
    ``collect_state_fields``, so a census that treats every ``ValueError`` as
    "the validators refuse this combination" reports the one thing it exists
    to find as proof that the thing cannot happen.
    """
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    if not frames:
        return False
    innermost = frames[-1]
    return (innermost.f_code.co_name == "collect_state_fields"
            and Path(innermost.f_code.co_filename).name == "health.py")


def _measured_count_from_breach(exc: BaseException) -> int:
    """The count ``collect_state_fields`` reported, or 0 if unparsable."""
    match = re.search(r"has (\d+) descriptors", str(exc))
    return int(match.group(1)) if match else 0


def _refusal_reason(exc: BaseException) -> str:
    """``exc`` as a production refusal, or re-raise it.

    A ``MemoryError`` is not a refusal.  Neither is an ``AttributeError`` from
    a stubbed backend.  Filing those as "not user-selectable" is how 58 of the
    140 configuration-valid combinations went unmeasured while the report said
    the validators had refused them.
    """
    if not isinstance(exc, _REFUSAL_TYPES):
        raise exc
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        raise exc
    # A refusal is something PRODUCTION said, so it has to have come out of
    # production code.  The same exception type raised by numpy, by the host
    # backend stub, or by this file is a census bug in a refusal's clothes.
    if "gpuwm" not in Path(tb.tb_frame.f_code.co_filename).parts:
        raise exc
    return f"{type(exc).__name__}: {exc}"


def _domain_clocks(experiment) -> dict:
    """The experiment's real per-domain clocks, keyed by grid id.

    ``attach_nest_boundaries`` stores the child clock on the resident boundary
    object.  ``DomainClock`` uses ``__slots__``, so ``_walk_arrays`` finds no
    ``__dict__`` and it contributes no descriptors -- but pass the production
    object anyway rather than a stand-in, so that stops being true LOUDLY if a
    clock ever grows an array.
    """
    from gpuwm.core.clock import resolve_clock

    return resolve_clock(experiment).clocks()


def experiment_census(path: Path, selections=None) -> dict:
    """Measure every selectable combination on every domain of ``path``.

    Combinations the production validators reject (an LSM without a surface
    layer, a PBL scheme without one, a moist scheme on a dry state) are
    recorded as ``rejected`` rather than counted: they are not configurations
    a user can run, and silently dropping them would hide the difference
    between "measured zero" and "cannot exist".

    Combinations that go OVER the ceiling are recorded in ``over_ceiling``,
    never in ``rejected``.
    """
    from gpuwm.core.health import MAX_HEALTH_FIELDS
    from gpuwm.core.preflight import (DEFAULT_FORCING_INTERVAL_SECONDS,
                                      lbc_intervals)

    experiment = load_any_experiment(path)
    by_id = {dc.grid_id: dc for dc in experiment.domains}
    n_lbc_intervals = lbc_intervals(experiment.run_seconds,
                                    DEFAULT_FORCING_INTERVAL_SECONDS)
    start_time = experiment.start_time
    clocks = _domain_clocks(experiment)
    if selections is None:
        selections = selectable_matrix(experiment.domains[0].run)

    rows = []
    rejected = []
    over_ceiling = []
    for selection in selections:
        per_domain = {}
        breakdown = {}
        try:
            for dc in experiment.domains:
                names = domain_descriptor_names(
                    dc, by_id.get(dc.parent_id), selection,
                    spec_bdy_width=experiment.spec_bdy_width,
                    n_lbc_intervals=n_lbc_intervals, start_time=start_time,
                    clock=clocks.get(dc.grid_id))
                per_domain[dc.grid_id] = len(names)
                breakdown[dc.grid_id] = family_breakdown(names)
        except Exception as exc:
            if _is_ceiling_breach(exc):
                over_ceiling.append({
                    "selection": selection.key(),
                    "grid_id": dc.grid_id,
                    "count": _measured_count_from_breach(exc),
                    "reason": str(exc)})
                continue
            rejected.append({"selection": selection.key(),
                             "reason": _refusal_reason(exc)})
            continue
        rows.append({"selection": selection.key(),
                     "mp_physics": selection.mp_physics,
                     "sf_surface_physics": selection.sf_surface_physics,
                     "bl_pbl_physics": selection.bl_pbl_physics,
                     "sf_sfclay_physics": selection.sf_sfclay_physics,
                     "cu_physics": selection.cu_physics,
                     "km_opt": selection.km_opt,
                     "per_domain": per_domain,
                     "peak": max(per_domain.values()),
                     "breakdown": breakdown})
    worst = max(rows, key=lambda row: row["peak"]) if rows else None
    return {
        "config": str(path),
        "max_health_fields": MAX_HEALTH_FIELDS,
        "ceiling_metadata_bytes_per_domain": descriptor_metadata_bytes(
            experiment.domains[0].run, n_lbc_intervals=n_lbc_intervals),
        "selectable_axes": {
            "mp_physics": list(
                selectable_scalar_values(experiment.domains[0].run,
                                         "mp_physics")),
            "sf_surface_physics": list(
                selectable_slot_values("sf_surface_physics")),
            "bl_pbl_physics": list(selectable_slot_values("bl_pbl_physics")),
            "sf_sfclay_physics": list(
                selectable_slot_values("sf_sfclay_physics")),
            "cu_physics": list(
                selectable_scalar_values(experiment.domains[0].run,
                                         "cu_physics")),
            "km_opt": list(
                selectable_scalar_values(experiment.domains[0].run,
                                         "km_opt")),
        },
        "domains": [{"grid_id": dc.grid_id, "nx": dc.run.nx,
                     "ny": dc.run.ny, "nz": dc.run.nz,
                     "parent_id": dc.parent_id}
                    for dc in experiment.domains],
        "rows": rows,
        "rejected": rejected,
        "over_ceiling": over_ceiling,
        "worst_selection": None if worst is None else worst["selection"],
        "worst_count": 0 if worst is None else worst["peak"],
        "headroom": (MAX_HEALTH_FIELDS
                     - (0 if worst is None else worst["peak"])),
    }


def descriptor_metadata_bytes(cfg, *, n_lbc_intervals: int = 0) -> int:
    """Device bytes one domain spends on the fixed descriptor tables.

    This is what ``MAX_HEALTH_FIELDS`` actually costs: the seven
    ``integration_health_*`` metadata slots are sized by the ceiling, not by
    the measured count, and they are per ``DomainState``.  Raising the
    ceiling raises this, and on this hardware VRAM is a correctness bar.
    """
    from gpuwm.core.preflight import scratch_slot_registry

    registry = scratch_slot_registry(cfg, n_lbc_intervals=n_lbc_intervals)
    fixed = {name: shape for name, shape in registry.items()
             if name.startswith("integration_health_")
             and name not in ("integration_health_partial",
                              "integration_health_result",
                              "integration_health_validation")}
    return sum(4 * math.prod(shape) for shape in fixed.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path,
                        help="experiment TOML to measure")
    parser.add_argument("--json", action="store_true",
                        help="emit the full census as JSON")
    parser.add_argument("--breakdown", action="store_true",
                        help="print the per-family descriptor breakdown")
    args = parser.parse_args(argv)

    install_host_array_backend()
    census = experiment_census(args.config)
    if args.json:
        json.dump(census, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(f"config              {census['config']}")
    print(f"MAX_HEALTH_FIELDS   {census['max_health_fields']} "
          f"({census['ceiling_metadata_bytes_per_domain']} B of fixed device "
          "metadata per domain)")
    for domain in census["domains"]:
        print("  d{grid_id:02d} nx={nx} ny={ny} nz={nz} "
              "parent={parent_id}".format(**domain))
    grid_ids = [domain["grid_id"] for domain in census["domains"]]
    header = "selection".ljust(38) + "".join(
        f"  d{grid_id:02d}" for grid_id in grid_ids) + "   peak"
    print(header)
    print("-" * len(header))
    for row in sorted(census["rows"], key=lambda item: item["peak"]):
        line = row["selection"].ljust(38) + "".join(
            f"  {row['per_domain'][grid_id]:>4d}" for grid_id in grid_ids)
        print(line + f"  {row['peak']:>5d}")
        if args.breakdown:
            for grid_id in grid_ids:
                print(f"      d{grid_id:02d} {row['breakdown'][grid_id]}")
    print(f"\nworst selectable    {census['worst_selection']} -> "
          f"{census['worst_count']} descriptors")
    print(f"headroom            {census['headroom']} of "
          f"{census['max_health_fields']}")
    print(f"measured            {len(census['rows'])} selectable "
          f"combination(s); {len(census['rejected'])} refused by the "
          "production validators")
    if args.breakdown and census["rejected"]:
        for entry in census["rejected"]:
            print(f"  refused {entry['selection']}: {entry['reason'][:110]}")
    if census["over_ceiling"]:
        print(f"\nOVER CEILING: {len(census['over_ceiling'])} selectable "
              "combination(s) exceed MAX_HEALTH_FIELDS and die mid-forecast:")
        for entry in census["over_ceiling"]:
            print(f"  {entry['selection']} d{entry['grid_id']:02d} -> "
                  f"{entry['count']} descriptors")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
