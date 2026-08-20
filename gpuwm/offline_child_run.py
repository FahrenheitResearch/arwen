"""Run a standalone CUDA child from archived gpuwm/WRF parent history.

This is gpuwm's native offline-nest driver.  It consumes ordinary parent
history plus authoritative source-physics evidence, performs SINT cold-start
and lateral-boundary preparation, destroys any need for a live parent, and
advances only the requested child on the GPU.  It never invokes WPS,
``real.exe``, ``wrf.exe``, or ``ndown.exe``.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import netCDF4
import numpy as np

from gpuwm.config import (
    load_config,
    load_history_selection,
    load_streaming_options,
    radiation_scheme_ids,
    soil_layer_count,
)
from gpuwm.io.history_selection import HISTORY_VOCABULARY
from gpuwm.io.wrfout import INITIAL_CONDITION_GLOBAL_ATTRS
from gpuwm.offline_child import (
    OfflineChildContractError,
    OfflineChildPlacement,
    bind_parent_physics_from_gpuwm_restart,
    bind_parent_physics_from_wrf_namelist,
    build_offline_child_domain_state,
    build_offline_lateral_boundaries,
    child_surface_requirement,
    interpolate_parent_initial_state,
    read_child_surface_state,
    validate_parent_history,
)


_PROJECTION_ATTRS = (
    "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "MOAD_CEN_LAT",
    "CEN_LAT", "CEN_LON", "POLE_LAT", "POLE_LON",
)

_CAPABILITIES = {
    "schema": "gpuwm-offline-child-capabilities-v1",
    "runner": "gpuwm.offline_child_run",
    "status": "IMPLEMENTED_UNVERIFIED",
    "explicit_expert_consent_required": False,
    "parent_producers": ["gpuwm", "stock-wrf"],
    "minimum_parent_frames": 2,
    "physics_evidence": ["gpuwm-restart", "wrf-namelist"],
    # 28 (Thompson aerosol-aware) is same-scheme only: an mp=28 parent
    # forcing an mp=28 child.  Every CROSS-scheme edge touching 28 is
    # refused by name (gpuwm/offline_child.py::
    # _CROSS_SCHEME_REFUSED_MP_PHYSICS), matching the online nest lane's
    # refusal in gpuwm/core/microphysics_transition.py.
    # 16 (WDM6) is absent from BOTH lists and from OFFLINE_CHILD_MP_PHYSICS:
    # this runner cannot read a WDM6 parent at all, because nn and NSSL's
    # qnn share the QNCCN wrfout name and the field map has no
    # scheme-qualified row.  It is cross-refused as well, so the mirror with
    # the online lane stays exact.
    "same_scheme_mp_physics": [6, 8, 10, 18, 28],
    "cross_scheme_transitions": [],
    "vertical_remapping": False,
    "terrain_policy": "sint-parent-inherited",
    "forecast_backend": "cuda",
    "preprocess_backends": ["cuda", "cpu"],
    "output_ownership": "create-only",
    # Davies clock bind era: the standalone child binds a DomainClock to
    # its external LBC mirror, so boundary consumers take WRF's
    # post-increment dtbc recurrence exactly like the production tree
    # root (gpuwm/ingest/lateral_bc.py bind_lateral_boundary_clock).
    "boundary_clock_semantics": "wrf-dtbc-bound",
    # Full-physics children (LSM/surface-layer/PBL) require a child-grid
    # surface source (ndown-equivalent contract); mp-only children run
    # without one.
    "full_physics_surface_source": "child-grid-file-required",
    # ``[tiles]`` in the child config: this route wires a streamed-domain
    # builder, so it HONORS the block rather than refusing it.  A child
    # refined out of an archived parent is the domain most likely to outgrow
    # the card it is being run on, and until this it was the one route that
    # could not ask to stream -- the RunConfig schema refused the table
    # outright as unknown.
    "tiles": "honored",
}


def _log(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise OfflineChildContractError(
            f"offline-child input is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _verify_file_receipts(
        receipts: list[dict[str, object]], *, label: str) -> None:
    for receipt in receipts:
        path = Path(str(receipt["path"]))
        observed = _file_receipt(path)
        if observed["bytes"] != receipt["bytes"] or (
                observed["sha256"] != receipt["sha256"]):
            raise OfflineChildContractError(
                f"{label} changed while the offline child was running: {path}")


def _exact_steps(seconds: float, dt: float, label: str) -> int:
    raw = float(seconds) / float(dt)
    rounded = int(round(raw))
    if rounded < 1 or not np.isclose(raw, rounded, rtol=0.0, atol=1e-8):
        raise OfflineChildContractError(
            f"{label}/dt must be a positive integer, got {raw}")
    return rounded


def _memory_snapshot(cp) -> dict[str, int]:
    pool = cp.get_default_memory_pool()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_reserved_bytes": int(pool.total_bytes()),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


def _child_boundary_clock(cfg, *, lbc_interval_seconds: float, steps: int,
                          output_steps: int):
    """One bound DomainClock for the standalone child (Davies bind era).

    The production tree binds the root's DomainClock to the external LBC
    mirror so every Davies consumer takes WRF's post-increment ``dtbc``
    recurrence (dyn_em/solve_em.F:371-372) with interval selection from
    the solve-entry time.  The offline child is its own root, so it
    constructs the same integer-tick clock from the child config and the
    proven parent cadence; the runner drives the executor's exact
    per-step recurrence (seam reset -> prepare_step -> solve -> advance).
    """
    from fractions import Fraction
    import math

    from gpuwm.core.clock import DomainClock, DomainTicks

    dt = Fraction(cfg.dt).limit_denominator(1_000_000)
    if float(dt) != float(cfg.dt):
        raise OfflineChildContractError(
            f"child dt={cfg.dt!r} is not exactly rational within 1e-6; "
            "the bound boundary clock requires an exact tick lattice")
    interval = Fraction(lbc_interval_seconds).limit_denominator(1_000_000)
    if float(interval) != float(lbc_interval_seconds):
        raise OfflineChildContractError(
            f"parent cadence {lbc_interval_seconds!r} s is not exactly "
            "rational within 1e-6")
    tick_den = math.lcm(dt.denominator, interval.denominator)
    step_ticks = int(dt * tick_den)
    interval_ticks = int(interval * tick_den)
    if interval_ticks % step_ticks != 0:
        raise OfflineChildContractError(
            f"parent cadence {lbc_interval_seconds:g} s is not a whole "
            f"number of child steps (dt={cfg.dt:g} s); the boundary seam "
            "must fall on a child step boundary")
    spec = DomainTicks(
        grid_id=int(cfg.grid_id), parent_id=0, parent_time_step_ratio=1,
        step_ticks=step_ticks, dt_fp32=np.float32(cfg.dt),
        history_ticks=int(output_steps) * step_ticks,
        restart_ticks=None, radt_ticks=None, stepra=None,
        cudt_ticks=None, stepcu=None, bldt_ticks=None, stepbl=None,
        lbc_interval_ticks=interval_ticks)
    return DomainClock(spec, tick_den, int(steps) * step_ticks)


def _initialize_child_physics(child, cfg, initial, surface, start_time):
    """Attach the child physics driver with an honest warm start.

    mp-only children keep the established default initialization.  A
    radiation scheme needs the child latitude/longitude and UTC start
    time (SINT of the parent's XLAT/XLONG unless the surface source
    carries the child's own).  Land-surface/surface-layer/PBL schemes
    require a child-grid surface source: soil state and land identity
    are never fabricated from scalar defaults on a real-data child.
    """
    from gpuwm.core.physics import initialize_physics

    needs_surface = bool(cfg.sf_surface_physics or cfg.sf_sfclay_physics
                         or cfg.bl_pbl_physics)
    ra_lw, ra_sw = radiation_scheme_ids(cfg)
    radiation_active = bool(ra_lw or ra_sw)
    if needs_surface and surface is None:
        # The predicate and the sentence live in one place
        # (offline_child.child_surface_requirement) so the front door's
        # early refusal and this late guard cannot drift apart.
        raise OfflineChildContractError(child_surface_requirement(cfg))
    if surface is None and not radiation_active:
        return initialize_physics(child, cfg)

    if surface is not None and "XLAT" in surface.fields:
        lat = np.asarray(surface.fields["XLAT"], dtype=np.float64)
        lon = np.asarray(surface.fields["XLONG"], dtype=np.float64)
    else:
        lat = np.asarray(initial.fields["XLAT"], dtype=np.float64)
        lon = np.asarray(initial.fields["XLONG"], dtype=np.float64)

    radiation = None
    if radiation_scheme_ids(cfg) == (4, 4):
        from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
            if cfg.o3input == 2:
                # FAIL CLOSED, the same refusal runtime._child_radiation_
                # adapter raises for the in-memory child routes.
                #
                # o3input = 2 means the child takes its ozone INTERPOLATED
                # FROM THE PARENT: WRF evaluates the CAM climatology on
                # id == 1 only and passes o3rad down.  This route has no
                # parent to interpolate from -- it is the ndown-equivalent
                # offline path, and it stamps parent_id = 0 on its own
                # DomainTicks precisely because no parent domain is
                # resident.  So the constructor below cannot be given an
                # ozone_parent even in principle.
                #
                # It used to be called WITHOUT one, which is not a
                # degradation but a silent wrong answer: with
                # ozone_parent=None the constructor takes its ROOT branch
                # and evaluates a fresh CAM climatology on the CHILD's own
                # latitudes, then identity() reports
                # "ozone_routing": "root-climatology" for a nested domain
                # without complaint.  o3input = 2 is also the RunConfig
                # DEFAULT, and `gpuwm downscale --point` copies every
                # RunConfig field from the parent (o3input and
                # ra_rrtmg_variant are not in its geometry-override set),
                # so the default path walked straight into it.
                raise ValueError(
                    "ra_rrtmg_variant='rrtmg_legacy' with o3input=2 needs "
                    "ozone interpolated from the parent domain, and the "
                    "offline child route has no resident parent to take it "
                    "from (this route stamps parent_id=0). Set o3input=0 "
                    "to use the legacy-RRTMG wrapper's own O3DATA profile, "
                    "or run the child on the nested route "
                    "(gpuwm.runtime.prepare_child_case), which wires the "
                    "parent's o3rad through ParentOzoneProvider.")
            radiation = RRTMGLegacyRadiation(
                start_time, lat, lon, p_top=float(initial.receipt["p_top"]),
                o3input=cfg.o3input)

    if surface is None:
        return initialize_physics(
            child, cfg, radiation=radiation,
            radiation_start_time=start_time,
            radiation_latitude=lat, radiation_longitude=lon)

    from gpuwm.core.landuse import initialize_landuse
    fields = surface.fields
    identity = surface.identity
    xice = fields.get("SEAICE", fields.get("XICE"))
    if xice is None:
        xice = np.zeros_like(fields["LANDMASK"])
    landuse = initialize_landuse(
        fields["LU_INDEX"], soil_type=fields["ISLTYP"],
        landmask=fields["LANDMASK"], snow=fields["SNOW"], xice=xice,
        valid_time=start_time, cen_lat=float(np.mean(lat)),
        mminlu=str(identity["MMINLU"]), iswater=int(identity["ISWATER"]),
        islake=int(identity["ISLAKE"]), isice=int(identity["ISICE"]),
        isoilwater=int(identity["ISOILWATER"]),
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=fields["TSLB"], sst=fields.get("SST"))
    driver = initialize_physics(
        child, cfg, landuse=landuse, tsk=fields["TSK"],
        soil_temperature=fields["TSLB"], soil_moisture=fields["SMOIS"],
        liquid_moisture=fields.get("SH2O"),
        ivgtyp=fields["LU_INDEX"], isltyp=fields["ISLTYP"],
        vegfra=fields["VEGFRA"], tmn=fields["TMN"], xice=xice,
        snow=fields["SNOW"],
        snow_depth=fields.get("SNOWH", np.zeros_like(fields["SNOW"])),
        pblh=fields.get("PBLH", 0.0),
        radiation=radiation, radiation_start_time=start_time,
        radiation_latitude=lat, radiation_longitude=lon)
    # Seed time-zero surface diagnostics from the child-grid source; the
    # first model step replaces them through SFCLAY/LSM/PBL in WRF
    # ordering (same convention as the experiment path's warm seed).
    import cupy as cp
    for source_name, field_name in (
            ("PSFC", "psfc"), ("T2", "t2"), ("Q2", "q2"), ("TH2", "th2"),
            ("U10", "u10"), ("V10", "v10"), ("UST", "ust")):
        value = fields.get(source_name)
        if value is not None and field_name in driver.fields:
            driver.fields[field_name][...] = cp.asarray(
                value, dtype=cp.float32)
    return driver


def _create_output_root(path: Path) -> Path:
    """Reserve one output tree without ever adopting prior contents."""
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _parent_grid_metadata(path: Path) -> tuple[float, float, dict[str, object]]:
    """Parent geometry, plus the lineage attributes the child inherits.

    A downscaled child's initial state is the parent's history, so the
    child's initial condition descends from whatever the parent's did.
    The parent's initial-condition provenance is therefore copied
    forward verbatim -- it describes the ROOT of the lineage, which is
    the fact a published child chart must not lose.  The child's own
    time zero stays in ``START_DATE``, where WRF puts it.  A parent that
    carries no provenance (a stock-WRF archive, or a pre-1.4.1 file)
    hands the child nothing, and the child says nothing rather than
    inventing an analysis.
    """
    with netCDF4.Dataset(path) as dataset:
        try:
            dx = float(dataset.getncattr("DX"))
            dy = float(dataset.getncattr("DY"))
        except AttributeError as exc:
            raise OfflineChildContractError(
                f"{path} lacks authoritative DX/DY attributes") from exc
        present = set(dataset.ncattrs())
        attrs = {
            name: dataset.getncattr(name)
            for name in (*_PROJECTION_ATTRS, *INITIAL_CONDITION_GLOBAL_ATTRS)
            if name in present
        }
    return dx, dy, attrs


def _output_fields(state, initial, refl_field=None,
                   surface=None) -> dict[str, np.ndarray]:
    import cupy as cp
    from gpuwm.io.wrfout import state_frame

    result = state_frame(state, include_diagnostic_pressure=True)
    if refl_field is not None:
        result["REFL_10CM"] = cp.asnumpy(refl_field)
    result.update({
        "MAPFAC_M": cp.asnumpy(state.msft),
        "MAPFAC_U": cp.asnumpy(state.msfu),
        "MAPFAC_V": cp.asnumpy(state.msfv),
        "F": cp.asnumpy(state.f),
        "E": cp.asnumpy(state.e),
        "SINALPHA": cp.asnumpy(state.sina),
        "COSALPHA": cp.asnumpy(state.cosa),
        "XLAT": np.asarray(initial.fields["XLAT"], dtype=np.float32),
        "XLONG": np.asarray(initial.fields["XLONG"], dtype=np.float32),
    })
    if surface is not None:
        # The two static land fields the experiment routes take from the
        # geography (gpuwm.runtime._metadata_frame) and this route has no
        # geography for.  It has something better: the child-grid surface
        # source it was warm-started from, which is where its land identity
        # legitimately comes from.  Written verbatim, so a child's own
        # history is itself a valid --child-surface-from file -- the same
        # completeness the parent's history now has, one generation down.
        result.update({
            "LANDMASK": np.asarray(surface.fields["LANDMASK"],
                                   dtype=np.float32),
            "LU_INDEX": np.asarray(surface.fields["LU_INDEX"],
                                   dtype=np.float32),
        })
    return result


def _write_frame(path: Path, state, cfg, initial, valid_time,
                 projection_attrs: dict[str, object], placement,
                 refl_field=None, surface=None,
                 history_selection=None) -> None:
    from gpuwm.io.wrfout import WrfoutWriter

    attrs = dict(projection_attrs)
    if surface is not None:
        # The land-use table identity, forwarded from the child's own
        # surface source.  read_child_surface_state requires these four as
        # EVIDENCE rather than assuming a table, so a child history file
        # that omitted them could not seed a grandchild however complete its
        # fields were -- and inventing them here would be exactly the
        # assumption that refusal exists to prevent.  ISOILWATER rides along
        # because stock WRF writes it too (share/output_wrf.F:973).
        identity = surface.identity
        attrs.update({
            "MMINLU": str(identity["MMINLU"]),
            "ISWATER": np.int32(identity["ISWATER"]),
            "ISLAKE": np.int32(identity["ISLAKE"]),
            "ISICE": np.int32(identity["ISICE"]),
            "ISOILWATER": np.int32(identity["ISOILWATER"]),
        })
    attrs.update({
        "GRID_ID": np.int32(cfg.grid_id),
        "PARENT_ID": np.int32(0),
        "I_PARENT_START": np.int32(placement.i_parent_start),
        "J_PARENT_START": np.int32(placement.j_parent_start),
        "PARENT_GRID_RATIO": np.int32(placement.parent_grid_ratio),
        "DT": np.float32(cfg.dt),
        "HYBRID_OPT": np.int32(cfg.hybrid_opt),
        "ETAC": np.float32(cfg.etac),
        "START_DATE": initial.valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "SIMULATION_START_DATE": initial.valid_time.strftime(
            "%Y-%m-%d_%H:%M:%S"),
        "GPUWM_OFFLINE_CHILD": np.int32(1),
    })
    frame = _output_fields(state, initial, refl_field=refl_field,
                           surface=surface)
    # The child's own [output] selection, from its own child config.  A
    # refined child at a fraction of the parent's spacing is exactly the
    # domain whose history fills a disk, and `gpuwm downscale` is the one
    # RunConfig-TOML route that writes a wrfout, so the table is real
    # here rather than validated-and-ignored.  ``None`` -- and the FULL
    # default -- leaves the frame and the header byte-identical.
    if history_selection is not None:
        frame, history_attrs = history_selection.apply(frame)
        attrs.update(history_attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with WrfoutWriter(
            path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
            dx=cfg.dx, dy=cfg.dy,
            title="gpuwm native standalone offline child",
            global_attrs=attrs,
            # The soil axis is the selected LSM's geometry, not a constant.
            soil_layers=soil_layer_count(cfg)) as writer:
        writer.write_frame(valid_time.strftime("%Y-%m-%d_%H:%M:%S"), frame)


def run(args: argparse.Namespace) -> dict[str, object]:
    import cupy as cp
    from gpuwm.core import streaming
    from gpuwm.core.refl import consume_refl_10cm, refl_10cm_is_stashed
    from gpuwm.ingest.lateral_bc import (
        attach_streaming_lateral_boundaries,
        bind_lateral_boundary_clock,
        lateral_boundary_reload_count,
        lateral_boundary_resident_bytes,
    )
    from gpuwm.io.restart import write_restart
    from gpuwm.io.wrfout import wrfout_filename

    started = time.perf_counter()
    outdir = _create_output_root(args.outdir)
    cfg = load_config(args.child_config)
    # Read at ADMISSION, beside the config it belongs to, and before any
    # parent frame is opened: mode = 'on' streams unconditionally and needs
    # no card to be knowable, so a malformed block fails here rather than
    # after the whole archive has been interpolated.
    tiles = load_streaming_options(args.child_config)
    # Read at ADMISSION as well, and for the same reason: an unknown
    # variable name or a history_vars/history_drop clash refuses HERE,
    # before a parent archive is opened, rather than at the first frame.
    history_selection = load_history_selection(args.child_config)
    history_selection.warn_lost_products(
        HISTORY_VOCABULARY, where=f"child d{cfg.grid_id:02d}")
    if not cfg.specified or cfg.nested:
        raise OfflineChildContractError(
            "child config must set specified=true and nested=false")
    if args.parent_restart is not None:
        binding = bind_parent_physics_from_gpuwm_restart(args.parent_restart)
    else:
        binding = bind_parent_physics_from_wrf_namelist(
            args.parent_namelist, domain_id=args.parent_domain_id)
    contract = validate_parent_history(
        args.parent_history,
        max_boundary_interval_seconds=args.max_boundary_interval_seconds,
        physics_binding=binding)
    parent_file_receipts = [
        _file_receipt(frame.path) for frame in contract.frames]
    dims = contract.frames[0].dimensions
    if int(cfg.nz) != int(dims["bottom_top"]):
        raise OfflineChildContractError(
            f"child nz={cfg.nz} differs from parent nz={dims['bottom_top']}; "
            "native offline-child vertical remapping is not implemented")
    if float(cfg.run_seconds) > (
            contract.end_time - contract.start_time).total_seconds():
        raise OfflineChildContractError(
            "child run_seconds exceeds the archived parent forcing window")
    placement = OfflineChildPlacement(
        parent_nx=int(dims["west_east"]),
        parent_ny=int(dims["south_north"]),
        child_nx=int(cfg.nx), child_ny=int(cfg.ny),
        parent_grid_ratio=int(args.parent_grid_ratio),
        i_parent_start=int(args.i_parent_start),
        j_parent_start=int(args.j_parent_start))
    parent_dx, parent_dy, projection_attrs = _parent_grid_metadata(
        contract.frames[0].path)
    expected_dx = parent_dx / placement.parent_grid_ratio
    expected_dy = parent_dy / placement.parent_grid_ratio
    if not np.isclose(cfg.dx, expected_dx, rtol=2e-7, atol=1e-6):
        raise OfflineChildContractError(
            f"child dx={cfg.dx} != parent DX/ratio={expected_dx}")
    if not np.isclose(cfg.dy, expected_dy, rtol=2e-7, atol=1e-6):
        raise OfflineChildContractError(
            f"child dy={cfg.dy} != parent DY/ratio={expected_dy}")
    steps = _exact_steps(cfg.run_seconds, cfg.dt, "run_seconds")
    output_steps = _exact_steps(
        cfg.output_interval_s, cfg.dt, "output_interval_s")
    health_steps = _exact_steps(
        args.health_interval_seconds, cfg.dt, "health_interval_seconds")
    surface = None
    surface_from = getattr(args, "child_surface_from", None)
    if surface_from is not None:
        surface = read_child_surface_state(
            surface_from, child_ny=int(cfg.ny), child_nx=int(cfg.nx),
            num_soil_layers=soil_layer_count(cfg))
    surface_file_receipts = (
        [] if surface is None else [_file_receipt(surface.path)])
    _log("contract_pass", frames=len(contract.frames),
         cadence_seconds=contract.interval_seconds,
         geometry_sha256=contract.geometry_sha256,
         source_physics=dict(binding.receipt()),
         target_mp_physics=int(cfg.mp_physics),
         child_shape=[cfg.nz, cfg.ny, cfg.nx],
         child_spacing_m=[cfg.dy, cfg.dx])

    initial = interpolate_parent_initial_state(
        contract.frames[0].path, placement,
        physics_binding=binding, target_mp_physics=cfg.mp_physics,
        backend=args.preprocess_backend)
    prepared = build_offline_lateral_boundaries(
        contract, placement,
        target_mp_physics=cfg.mp_physics,
        backend=args.preprocess_backend,
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    child = build_offline_child_domain_state(initial, cfg)
    attach_streaming_lateral_boundaries(child, prepared.boundaries)
    # Davies clock bind (production semantics): boundary consumers take
    # WRF's post-increment dtbc recurrence from a bound integer-tick
    # clock, exactly like the experiment tree's root.
    clock = _child_boundary_clock(
        cfg, lbc_interval_seconds=contract.interval_seconds,
        steps=steps, output_steps=output_steps)
    bind_lateral_boundary_clock(child, clock)
    _initialize_child_physics(child, cfg, initial, surface,
                              initial.valid_time)
    cp.cuda.runtime.deviceSynchronize()
    # ``[tiles]``, wired exactly the way the prepared front doors wire it
    # (gpuwm.prepared_single_domain_forecast: decide ONCE, hand the decision
    # to make_stepper, record it).  With no block this is
    # ``gpuwm.core.dycore.step`` ITSELF -- the same function object the loop
    # below has always called -- so a child that configures nothing is
    # byte-for-byte the run it was before this seam existed.
    #
    # AFTER the physics driver is attached, never before: the builder fills
    # the store from the PREPARED state and builds every tile buffer with the
    # domain's own physics selectors, so a stepper made against a bare
    # DomainState would carry a different inventory than the domain it is
    # meant to be.
    streaming_decision = streaming.decide(cfg, tiles)
    stepper = streaming.make_stepper(
        child, cfg, tiles, decision=streaming_decision,
        build=streaming.standalone_domain_builder(grid_id=int(cfg.grid_id)))
    streaming_report = streaming.streaming_receipt(
        tiles, {int(cfg.grid_id): streaming_decision})
    if streaming_report:
        _log("child_tiles", **streaming_report)
    # ``stability_report`` ITSELF when resident; the per-tile fold when
    # streamed.  The state is not where a streamed domain lives, so the
    # whole-field reduction would inspect the snapshot that filled the store
    # and pass forever.
    child_stability = streaming.stability_observer(stepper)
    boundary_bytes = lateral_boundary_resident_bytes(child)
    child_memory_initial = _memory_snapshot(cp)
    child_pool_reserved_peak = child_memory_initial["pool_reserved_bytes"]
    _log("child_launch", pid=os.getpid(), steps=steps,
         boundary_intervals=len(prepared.boundaries.intervals),
         boundary_device_resident_bytes=boundary_bytes,
         boundary_device_reload_count=lateral_boundary_reload_count(child),
         memory=child_memory_initial)

    output_paths: list[Path] = []

    def emit_output() -> None:
        valid = initial.valid_time + timedelta(
            seconds=float(clock.elapsed_seconds))
        # THE HISTORY CADENCE, which is where StreamedDomain.refresh_state
        # says this belongs.  A streamed domain's forecast is in the pinned
        # host store and this DomainState is the snapshot that filled it, so
        # without the copy every frame after the cold-start one would be the
        # initial condition under a later timestamp -- correct inventory,
        # correct Times, no forecast.  Zero and a getattr when resident.
        streaming.refresh_streamed_state(stepper, child)
        refl = (consume_refl_10cm(child)
                if refl_10cm_is_stashed(child) else None)
        path = outdir / wrfout_filename(valid, domain_id=cfg.grid_id)
        _write_frame(path, child, cfg, initial, valid,
                     projection_attrs, placement, refl_field=refl,
                     surface=surface, history_selection=history_selection)
        output_paths.append(path)
        # History-interval reset of the UP_HELI_MAX window (no-op unless
        # the child config enables nwp_diagnostics; the synchronous
        # writer above snapshotted the accumulator already).
        from gpuwm.core.uh_diag import reset_up_heli_max
        reset_up_heli_max(child)
        _log("child_output", elapsed_seconds=float(clock.elapsed_seconds),
             path=str(path), bytes=path.stat().st_size)

    emit_output()
    step_seconds = []
    child_health = child_stability(child, cfg)
    for step_index in range(1, steps + 1):
        # The executor's exact per-step recurrence (core/clock.py
        # execute_schedule): dtbc zeroes at every external interval seam
        # including t=0, prepare_step applies WRF's post-increment
        # ``grid%dtbc = grid%dtbc + grid%dt`` before the solve, and the
        # calendar advances after it.
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        output_due = step_index % output_steps == 0 or step_index == steps
        step_started = time.perf_counter()
        stepper(child, cfg, refl_10cm_due=output_due)
        cp.cuda.runtime.deviceSynchronize()
        step_seconds.append(time.perf_counter() - step_started)
        clock.advance()
        if step_index % health_steps == 0 or step_index == steps:
            child_health = child_stability(child, cfg)
            memory = _memory_snapshot(cp)
            child_pool_reserved_peak = max(
                child_pool_reserved_peak, memory["pool_reserved_bytes"])
            _log("child_step", step=step_index, total_steps=steps,
                 elapsed_seconds=float(clock.elapsed_seconds),
                 nan=bool(child_health["nan"]),
                 cfl=float(child_health["cfl"]),
                 w_max=float(child_health["w_max"]),
                 boundary_device_reload_count=lateral_boundary_reload_count(child),
                 memory=memory,
                 wall_seconds=time.perf_counter() - started)
            if child_health["nan"]:
                raise RuntimeError(
                    f"offline child became non-finite at step {step_index}")
        if output_due:
            emit_output()
    # "and once more before its final read" -- the checkpoint and every
    # summary field below read the DomainState, and the last emit_output
    # refreshed it only because the final step is always output-due.  Stated
    # here rather than inferred from that coincidence: a run whose cadence
    # changed would otherwise checkpoint the analysis.
    carriers_refreshed = streaming.refresh_streamed_state(stepper, child)
    restart = write_restart(
        outdir / f"gpuwmrst_d{cfg.grid_id:02d}_final.npz", child, cfg)
    sample = np.asarray(step_seconds, dtype=np.float64)
    warm = sample[1:] if sample.size > 1 else sample
    _verify_file_receipts(
        parent_file_receipts, label="parent history input")
    _verify_file_receipts(
        surface_file_receipts, label="child surface source")
    report = {
        "result": "PASS" if not child_health["nan"] else "FAIL",
        "pipeline": "archived-parent-to-native-standalone-cuda-child",
        "online_parent_present_during_child": False,
        "parent_frames": [str(frame.path) for frame in contract.frames],
        "parent_frame_receipts": parent_file_receipts,
        "parent_geometry_sha256": contract.geometry_sha256,
        "parent_physics_binding": dict(binding.receipt()),
        "child_config": str(args.child_config.resolve()),
        "child_config_sha256": _sha256(args.child_config.resolve()),
        "target_mp_physics": int(cfg.mp_physics),
        "placement": {
            "parent_grid_ratio": placement.parent_grid_ratio,
            "i_parent_start": placement.i_parent_start,
            "j_parent_start": placement.j_parent_start,
            "child_nx": placement.child_nx,
            "child_ny": placement.child_ny,
        },
        "preprocess_backend": args.preprocess_backend,
        # Which cadence flag the invoker gave (audit finding 5): True
        # means the ceiling was the archive's own cadence, accepted via
        # --accept-parent-cadence; False means an explicit
        # --max-boundary-interval-seconds.  The effective interval the
        # child was forced at is boundary_clock.lbc_interval_seconds.
        "boundary_cadence_provenance": {
            "accepted_parent_cadence": bool(
                getattr(args, "accepted_parent_cadence", False)),
            "max_boundary_interval_seconds": float(
                args.max_boundary_interval_seconds),
            "effective_interval_seconds": float(contract.interval_seconds),
        },
        "boundary_clock": {
            "semantics": "wrf-dtbc-bound",
            "tick_den": int(clock.tick_den),
            "step_ticks": int(clock.spec.step_ticks),
            "lbc_interval_seconds": float(contract.interval_seconds),
            "final_ticks": int(clock.ticks),
        },
        "child_surface_source": (
            None if surface is None else dict(surface.receipt)),
        "child_surface_file_receipts": surface_file_receipts,
        "preparation_seconds": prepared.preparation_seconds,
        # ``{}`` for a child that configures no [tiles], which is what keeps
        # every report written before this seam existed byte-identical.  A
        # configured child gets the per-grid verdict, and ``streamed_any``
        # is the field to assert on: a mode='auto' child that declined and
        # an unconfigured one are otherwise indistinguishable.
        "tiles": streaming_report,
        "streamed_carriers_refreshed": carriers_refreshed,
        "boundary_intervals": len(prepared.boundaries.intervals),
        "boundary_device_resident_bytes": boundary_bytes,
        "boundary_device_reload_count": lateral_boundary_reload_count(child),
        "child_steps": steps,
        "child_simulated_seconds": float(child.elapsed_seconds),
        "child_step_warm_mean_seconds": float(warm.mean()),
        "child_memory_initial": child_memory_initial,
        "child_pool_reserved_peak_bytes": child_pool_reserved_peak,
        "child_health": child_health,
        "outputs": [str(path) for path in output_paths],
        "output_receipts": [_file_receipt(path) for path in output_paths],
        "final_restart": str(restart),
        "final_restart_sha256": _sha256(restart),
        "final_restart_receipt": _file_receipt(restart),
        "wall_seconds": time.perf_counter() - started,
    }
    temporary = outdir / "report.json.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, outdir / "report.json")
    _log("complete", **report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-history", type=Path, nargs="+", required=True)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--parent-restart", type=Path)
    evidence.add_argument("--parent-namelist", type=Path)
    parser.add_argument("--parent-domain-id", type=int, default=1)
    parser.add_argument("--child-config", type=Path, required=True)
    parser.add_argument("--parent-grid-ratio", type=int, required=True)
    parser.add_argument("--i-parent-start", type=int, required=True)
    parser.add_argument("--j-parent-start", type=int, required=True)
    parser.add_argument("--max-boundary-interval-seconds", type=float,
                        required=True)
    parser.add_argument("--accepted-parent-cadence", action="store_true",
                        help="provenance marker: the ceiling above was "
                             "taken from the parent archive's own cadence "
                             "(gpuwm downscale --accept-parent-cadence) "
                             "rather than chosen explicitly; recorded in "
                             "report.json")
    parser.add_argument("--child-surface-from", type=Path, default=None,
                        help="child-grid wrfinput/history file supplying "
                             "land identity and soil warm-start state "
                             "(required for surface-physics children)")
    parser.add_argument("--preprocess-backend", choices=("cuda", "cpu"),
                        default="cuda")
    parser.add_argument("--health-interval-seconds", type=float, default=60.0)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--show-capabilities"]:
        print(json.dumps(_CAPABILITIES, sort_keys=True))
        return 0
    report = run(_parser().parse_args(arguments))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
