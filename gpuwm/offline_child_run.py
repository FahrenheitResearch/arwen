"""Run a standalone CUDA child from archived gpuwm/WRF parent history.

This is gpuwm's native offline-nest driver.  It consumes ordinary parent
history plus authoritative source-physics evidence, performs SINT cold-start
and lateral-boundary preparation, destroys any need for a live parent, and
advances only the requested child on the GPU.  It never invokes WPS,
``real.exe``, ``wrf.exe``, or ``ndown.exe``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import netCDF4
import numpy as np

from gpuwm import downscale_pricing
from gpuwm.aerosol_source_receipt import aerosol_source_report_entry
from gpuwm.config import (
    load_config,
    load_history_selection,
    load_streaming_options,
    radiation_scheme_ids,
    soil_layer_count,
)
from gpuwm.io.history_selection import HISTORY_VOCABULARY
from gpuwm.physics_registry import consumer_rows_by_selector
from gpuwm.io.wrfout import INITIAL_CONDITION_GLOBAL_ATTRS
from gpuwm.explain import warn
from gpuwm.offline_child import (
    DERIVED_CHILD_SURFACE_CAVEAT,
    OfflineChildContractError,
    OfflineChildPlacement,
    bind_parent_physics_from_gpuwm_restart,
    bind_parent_physics_from_wrf_namelist,
    build_offline_child_domain_state,
    build_offline_lateral_boundaries,
    child_surface_requirement,
    derive_child_surface_from_parent,
    interpolate_parent_initial_state,
    read_child_surface_state,
    reserve_output_root,
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
    # 50 (P3) is same-scheme only on the same terms: the lane reads its
    # whole transported set (qv,qc,qr,qi + ni/nr + the rime pair qir/qib,
    # Registry.EM_COMMON:3038), and every cross-scheme edge touching 50 is
    # refused by the same derived set.
    # 16 (WDM6) is absent from BOTH lists and from OFFLINE_CHILD_MP_PHYSICS:
    # this runner cannot read a WDM6 parent at all, because nn and NSSL's
    # qnn share the QNCCN wrfout name and the field map has no
    # scheme-qualified row.  It is cross-refused as well, so the mirror with
    # the online lane stays exact.
    # 9 (Milbrandt-Yau) joined the same-scheme list when the lane learned
    # its own QHAIL/QNHAIL rows (_MY2_WRF_TO_STATE), the third
    # scheme-qualified map beside NSSL's -- the shape 16 is still in.  0
    # (passiveqv) and 1 (Kessler) joined with it: their transported sets
    # are qv, and qv/qc/qr, which the lane already built.
    # DERIVED from the registry's consumers.offline_child rows, the same
    # source gpuwm.offline_child.OFFLINE_CHILD_MP_PHYSICS is built from, so
    # the capability receipt users read cannot disagree with the gate.
    "same_scheme_mp_physics": sorted(
        int(mp) for mp, row in
        consumer_rows_by_selector("microphysics", "offline_child").items()
        if row.get("same_scheme") is True),
    "cross_scheme_transitions": [],
    # A child may carry its OWN eta ladder, deeper than the archived
    # parent's, when it declares one (``eta_levels`` in the child config,
    # written by ``gpuwm downscale --child-levels``).  The remap is
    # conservative in dry mass and every water substance
    # (gpuwm/vertical_remap.py) and runs once at preparation on the host; the
    # integration loop is unchanged.  p_top/hybrid_opt/etac stay shared with
    # the parent -- that is what gives the two ladders coincident endpoints.
    "vertical_remapping": "conservative-offline-prepare",
    "terrain_policy": "sint-parent-inherited",
    "forecast_backend": "cuda",
    "preprocess_backends": ["cuda", "cpu"],
    "output_ownership": "create-only",
    # Davies clock bind era: the standalone child binds a DomainClock to
    # its external LBC mirror, so boundary consumers take WRF's
    # post-increment dtbc recurrence exactly like the production tree
    # root (gpuwm/ingest/lateral_bc.py bind_lateral_boundary_clock).
    "boundary_clock_semantics": "wrf-dtbc-bound",
    # Full-physics children (LSM/surface-layer/PBL) take a child-grid
    # surface source when one is given (--child-surface-from, the
    # ndown-equivalent contract, higher fidelity) and otherwise derive
    # one from the parent's own history through WRF's nest-birth
    # operators; mp-only children run without either.
    "full_physics_surface_source":
        "child-grid-file-or-parent-history-interpolated",
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


from gpuwm.first_products import DEFAULT_WAIT_SECONDS as _EARLY_RENDER_WAIT

#: One sentence for the two doors that ask it (this runner at admission and
#: gpuwm downscale before it opens a frame), so they cannot drift apart.
RENDERER_MISSING_REMEDY = (
    "The Rust renderer is unavailable, so this child's products cannot be "
    "drawn. Next: gpuwm setup, then repeat this command. Use "
    "--render-products none to run the child without pictures.")


class _ChildProgress:
    """The offline child's own run manifest and native event stream.

    A child used to publish ``report.json`` at the end and a stream of
    ``{"event": ...}`` lines on stdout, and nothing bound those lines to
    the process that wrote them.  No run browser could show a downscale
    in flight, and a finished one could not be listed beside the
    forecasts it came from.

    So this publishes THE SAME TWO RECEIPTS every other ArWen run
    publishes -- ``gpuwm.run-manifest.v1`` beside the frames and
    ``gpuwm.run-plan.event.v1`` in ``events.jsonl`` -- rather than
    inventing a third progress shape for one route.  Every existing
    reader (the controller's local progress reader, the desktop's live
    view) then works on a child unchanged, and the stream is mirrored to
    stdout so the job log keeps the lines it always had.

    Every method is a no-op until :meth:`start` has run, so a refusal
    raised before the contracts pass still costs nothing.
    """

    def __init__(self) -> None:
        self.events = None
        self.manifest_path = None
        self.run_id = None
        self.outdir = None
        #: The plan the child's pictures are drawn from -- the very dict
        #: :func:`gpuwm.go_cli._render_stage` is handed, so the early
        #: render and the finalize one cannot drift apart in what they
        #: draw or where they put it.  ``None`` when no products were
        #: asked for.
        self.render_plan = None
        self._stage = None
        self._stage_phases: list[str] = []
        self._stage_started_wall = None
        self._started_wall = time.perf_counter()
        self._first_products = None
        self._first_products_seconds = None
        #: The domain a coarse progress sample is attributed to.  A
        #: child is one domain and :meth:`start` learns which.
        self._root_domain = 1
        #: Set by :class:`gpuwm.runplan._GoObserver`, which is what the
        #: shared render stage reports through on every route.
        self._render_summary = None

    def start(self, *, outdir: Path, child_config: Path, ratio: int,
              start_time, parent: dict, name: str) -> None:
        from datetime import datetime, timezone
        import uuid

        from gpuwm import runplan
        from gpuwm.supervisor import atomic_write_json

        self.outdir = Path(outdir)
        events_path = self.outdir / "events.jsonl"
        # The stream first: the manifest names this file, and a reader
        # that finds the manifest must find the file it points at.
        self.events = runplan.EventStream(events_path)
        self.run_id = f"downscale-{uuid.uuid4().hex}"
        config = Path(child_config).resolve()
        document = {
            "schema": runplan.MANIFEST_SCHEMA,
            "name": name,
            "route": "downscale",
            "run_id": self.run_id,
            "pid": os.getpid(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            # A child's "plan" IS its configuration: this route has no
            # run-plan document, and the reader's binding is the same
            # one either way -- name the source and hash it.
            "plan_source": f"gpuwm downscale {config}",
            "plan_sha256": _sha256(config),
            "run_dir": str(self.outdir),
            "outputs_dir": str(self.outdir),
            "events_path": str(events_path),
            "events_schema": runplan.EVENT_SCHEMA,
            # No supervisor heartbeat on this route.  Null rather than a
            # path to a file nothing writes: a reader that opened it
            # would wait forever for a first sample.
            "progress_path": None,
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parent": dict(parent),
        }
        self.manifest_path = self.outdir / runplan.MANIFEST_FILENAME
        atomic_write_json(self.manifest_path, document)
        self.emit("resolved_plan", config_source=str(config),
                  config_sha256=document["plan_sha256"])

    def emit(self, event: str, **fields) -> None:
        if self.events is not None:
            self.events.emit(event, **fields)

    def failed(self, error: BaseException | None = None, *,
               stage: str = "forecast") -> None:
        """This run failed.  Both callers of that sentence reach here.

        The runner's progress protocol calls ``failed()`` with no
        argument -- :meth:`gpuwm.runplan.RunObserver.failed` takes none,
        and :meth:`gpuwm.runplan._GoObserver.failed` forwards to it --
        while this module's own exit paths call it with the exception
        they caught.  A signature that took only the second spelling
        made this object incompatible with the very surface it is handed
        to, and the incompatibility was reachable: it is a
        ``TypeError`` raised on a failure path, which is where a
        readable message matters most.
        """

        if error is None:
            self.emit("failed", stage=stage)
            return
        message = " ".join(f"{type(error).__name__}: {error}".split())
        self.emit("failed", stage=stage, message=message[:1600])

    # -- the run-plan observer surface the shared render stage drives --
    #
    # `gpuwm.runplan._finish_render` is the finalize render EVERY other
    # route uses, and it reports through `_GoObserver`, which asks its
    # observer for exactly these five things.  Supplying them here is
    # what lets a child's pictures be drawn by that function rather than
    # by a second render stage written for this route -- two renderers
    # for one product set is how a downscaled run ends up drawing a
    # different catalog, into a different folder, than the forecast it
    # was cut from.

    def warn(self, code: str, message: str, **fields) -> None:
        self.emit("warning", code=code, message=message, **fields)

    def stage_progress(self, *, phase: str, elapsed_seconds: float,
                       model_seconds: float, status=None) -> None:
        """One coarse sample from a stage that runs out of process.

        :meth:`gpuwm.runplan._GoObserver.stage_heartbeat` calls this
        whenever the running stage's progress file carries a model
        clock, so an observer that lacks it raises ``AttributeError``
        out of a heartbeat -- inside the render stage of a finished
        forecast.

        Spelled as :meth:`gpuwm.runplan.RunObserver.stage_progress`
        spells it, down to ``source``: one event tag, one field set, so
        a reader of a downscaled run's stream and a reader of a
        forecast's are the same reader.
        """

        speed = (round(model_seconds / elapsed_seconds, 4)
                 if elapsed_seconds > 0.0 and model_seconds > 0.0 else None)
        self.emit("model_progress", domain=self._root_domain,
                  model_seconds=float(model_seconds),
                  wall_seconds=round(float(elapsed_seconds), 6),
                  speed_x=speed, step_ms=None, phase=phase,
                  status=status, source="stage_progress_file")

    def enter_stage(self, stage: str, *, phase: str | None = None) -> None:
        """Close whatever stage is open and open ``stage``.

        Re-entering the open stage is a no-op, exactly as
        :meth:`gpuwm.runplan.RunObserver.enter_stage` treats it: the
        finalize render is announced by its caller AND by the stage
        hook, and one render is one ``stage_started``.
        """

        if stage == self._stage:
            return
        self.finish_stage()
        self._stage = stage
        self._stage_phases = [] if phase is None else [phase]
        self._stage_started_wall = time.perf_counter()
        fields = {"stage": stage}
        if phase is not None:
            fields["phase"] = phase
        self.emit("stage_started", **fields)

    def finish_stage(self, **fields) -> None:
        """Close the open stage, carrying the render summary on finalize."""

        if self._stage is None:
            return
        stage, started = self._stage, self._stage_started_wall
        self._stage = None
        if stage == "finalize" and self._render_summary is not None:
            fields.setdefault("render_summary", self._render_summary)
        self.emit("stage_finished", stage=stage,
                  wall_seconds=round(time.perf_counter() - started, 6),
                  phases=list(self._stage_phases), **fields)

    def arm_render(self, *, outdir, render_products) -> dict | None:
        """Arm this child's pictures: the plan, and the early render.

        The plan is built ONCE, here, and both renders read it: the
        early one this arms and the finalize one
        :func:`_finish_child_render` runs.  ``render_products`` absent
        or ``none`` arms nothing and leaves :attr:`render_plan` ``None``,
        which is the single answer to "does this run draw?".
        """

        from gpuwm.first_products import FirstProducts, early_render_requested

        if not early_render_requested(render_products):
            self.render_plan = None
            return None
        root = Path(outdir)
        self.render_plan = {"run": root, "wrfout_dir": root,
                            "render": root / "png",
                            "render_products": str(render_products)}
        self._first_products = FirstProducts(
            self.render_plan, report=self._first_products_ready,
            warn=self.warn)
        return self.render_plan

    @property
    def first_products(self):
        """The armed early render, read by the finalize stage."""

        return self._first_products

    @property
    def first_products_seconds(self) -> float | None:
        """Time to first plot, or ``None`` if no early render published."""

        return self._first_products_seconds

    def _first_products_ready(self, receipt) -> None:
        """The early render published.  This is the TTFP number."""

        elapsed = round(time.perf_counter() - self._started_wall, 6)
        self._first_products_seconds = elapsed
        self.emit(
            "first_products_ready",
            domain=receipt["domain"], valid_time=receipt["valid_time"],
            frame=receipt["frame"], paths=list(receipt["paths"]),
            render_products=receipt["render_products"],
            render_seconds=receipt["render_seconds"],
            seconds_from_plan_accepted=elapsed)

    def output_committed(self, **fields) -> None:
        """One child history frame is durable.  Draws the first one.

        The event is the same one this route always emitted; the
        dispatch beside it is what makes the analysis frame a picture
        while the rest of the forecast is still integrating, as it is on
        every other route.  Only the first frame wins, and
        :class:`gpuwm.first_products.FirstProducts` decides that, not a
        counter kept here.
        """

        # The child's own domain, learned from the frames it commits:
        # this route has one, and a coarse progress sample has to name
        # it rather than the root of a hierarchy it does not have.
        self._root_domain = int(fields["domain"])
        self.emit("output_committed", **fields)
        if self._first_products is not None:
            self._first_products.frame_committed(
                domain=fields["domain"], valid_time=fields["valid_time"],
                path=fields["path"])

    def wait_early_render(self, *, timeout: float | None = _EARLY_RENDER_WAIT) -> None:
        """Join the early render, wherever this run is exiting from.

        A render dispatched on a worker thread outlives nothing: the
        subprocess it owns is still drawing when the process that armed
        it returns.  The finalize stage waits for exactly this reason,
        and every OTHER exit -- a refused forecast, a raised contract, a
        render stage that failed -- has the same obligation, so the wait
        lives on the one method all of them pass through.
        """

        if self._first_products is not None:
            self._first_products.wait(timeout=timeout)

    def early_pictures(self) -> int:
        """How many pictures the early render has published, after joining it."""

        if self._first_products is None:
            return 0
        # The bounded wait: this is a count for a sentence, and a wedged
        # renderer must not hold a run that already failed.  The receipt
        # lists what the early render published under "written".
        receipt = self._first_products.wait()
        written = receipt.get("written") if isinstance(receipt, dict) else None
        return len(written) if isinstance(written, list) else 0

    def withdraw_early_render(self, reason: str) -> int:
        """Drop what the early render published; return how many pictures.

        THE DECISION, recorded where it is enforced: a child that does
        not pass publishes NO picture.  The early render draws the
        analysis frame while the run still looks healthy, and a run that
        then refuses itself would otherwise leave pictures as its only
        artifact that does not carry the verdict -- and no frame of a
        run that went non-finite can be shown to be the frame that was
        still finite.  The frames, the checkpoints and the report stay:
        they are the evidence.  The pictures are a derivative of it and
        can be redrawn by hand from the frames at any time.

        Nothing else writes into this directory on a failed run -- the
        finalize render never runs -- so the whole picture tree is
        exactly what the early render put there.
        """

        if self._first_products is None or self.render_plan is None:
            return 0
        from gpuwm.first_products import withdraw

        # No timeout here: a withdrawal that ran ahead of the render
        # thread would remove a tree that thread is still writing into.
        self.wait_early_render(timeout=None)
        render_dir = Path(self.render_plan["render"])
        pictures = withdraw(render_dir)
        # A `warning` carrying its own code, because the event
        # vocabulary is a closed schema shared with every reader of
        # every route (:data:`gpuwm.runplan.EVENT_TAGS`) -- a tag
        # invented for one route is a record nothing can read.
        self.warn("early_render_withdrawn", reason,
                  render=str(render_dir), pictures=pictures)
        return pictures

    def close(self) -> None:
        # The floor under wait_early_render: every exit path closes the
        # stream, so every exit path waits.
        self.wait_early_render()
        if self.events is not None:
            self.events.close()
            self.events = None


#: Folder names every prepared route uses for its layout rather than for
#: the run's identity: ``<run>/wrfout/`` holds the frames and ``<run>``
#: is itself called ``run`` under a stamped folder.  A child named after
#: one of these would be "Downscale of wrfout".
_LAYOUT_FOLDERS = frozenset({"wrfout", "run"})


def _parent_label(frame: Path) -> str:
    """The parent folder a child is named after: the nearest ancestor of
    its first history frame that is not a layout folder, so the stamped
    run folder names the parent on the prepared routes and the history
    directory itself does everywhere else."""

    for ancestor in frame.parents:
        if ancestor.name and ancestor.name not in _LAYOUT_FOLDERS:
            return ancestor.name
    return frame.parent.name


def child_run_name(frame: Path, *, grid_id: int, ratio: int, dx: float) -> str:
    """``Downscale of <parent> · d03 ×3 · 1.33 km``: the run browser's name.

    Three significant figures on the spacing, so a grandchild at a third
    of 4 km reads as 1.33 km rather than a long fraction, and 4 km stays
    ``4 km``.
    """
    return (f"Downscale of {_parent_label(frame)} "
            f"· d{int(grid_id):02d} ×{int(ratio)} "
            f"· {float(dx) / 1000.0:.3g} km")


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
            f"{label}/dt must be a positive integer, got {raw}: the "
            f"child integrates in whole steps of dt = {float(dt):g} s and "
            f"would never land on that instant; set {label} to a value that is "
            f"a whole multiple of dt")
    return rounded


def checkpoint_schedule(steps: int, restart_steps: int | None) -> frozenset:
    """The step indices a child writes a checkpoint at.

    Every whole multiple of ``restart_steps`` inside the run, and the last
    step always: a run with ``restart_interval_s`` unset or zero writes
    exactly one checkpoint, at its end, which is what it always wrote,
    now under a name the next downscale can discover.
    """
    due = {int(steps)}
    if restart_steps is not None and int(restart_steps) > 0:
        due.update(range(int(restart_steps), int(steps) + 1,
                         int(restart_steps)))
    return frozenset(due)


@dataclass(frozen=True)
class ChildCadence:
    """Every step count a child integrates on, as whole steps of ``dt``."""

    steps: int
    output_steps: int
    restart_steps: int | None
    health_steps: int | None
    checkpoint_due: frozenset


def child_cadence(cfg, *, health_interval_seconds: float | None = None
                  ) -> ChildCadence:
    """Check the child's clock once, for the plan review and the run alike.

    ``run_seconds``, ``output_interval_s`` and ``restart_interval_s`` must
    each be a whole number of ``dt`` steps, and so must the health interval
    when the caller has one.  A derived child inherits these from its
    parent with ``dt`` divided by the ratio, so exactness is preserved; a
    hand-written ``--child-config`` can break it, and that used to be
    found only at run start.  ``gpuwm downscale`` reviews with this
    function and the runner integrates on its answer, so the two doors
    cannot disagree about the same clock.  ``restart_interval_s`` unset or
    zero means one checkpoint, at the end.
    """
    steps = _exact_steps(cfg.run_seconds, cfg.dt, "run_seconds")
    output_steps = _exact_steps(
        cfg.output_interval_s, cfg.dt, "output_interval_s")
    restart_steps = (
        _exact_steps(cfg.restart_interval_s, cfg.dt, "restart_interval_s")
        if float(cfg.restart_interval_s) > 0.0 else None)
    health_steps = (
        None if health_interval_seconds is None else
        _exact_steps(health_interval_seconds, cfg.dt,
                     "health_interval_seconds"))
    return ChildCadence(
        steps=steps, output_steps=output_steps, restart_steps=restart_steps,
        health_steps=health_steps,
        checkpoint_due=checkpoint_schedule(steps, restart_steps))


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
    if 4 in radiation_scheme_ids(cfg):
        from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
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
            from gpuwm.core.radiation_composition import make_radiation
            radiation = make_radiation(
                cfg, start_time, lat, lon, p_top=float(initial.receipt["p_top"]))

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
    """Reserve one output tree without ever adopting a prior run's."""
    return reserve_output_root(path, flag="--outdir")


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
    """One offline child, with its progress receipts published.

    The receipts wrap the whole run so that the LAST event a reader sees
    is always terminal: ``completed`` for a run that produced a report,
    ``failed`` carrying the sentence for one that did not -- including a
    refusal raised before the model started, where the stream exists but
    is empty and the emit is a no-op.
    """

    progress = _ChildProgress()
    try:
        report = _run(args, progress)
    except BaseException as error:
        # The run did not finish, so it publishes no picture: the early
        # render's analysis frame is withdrawn before the stream is
        # closed, and the wait inside that withdrawal is what keeps the
        # render subprocess from outliving this process.
        progress.withdraw_early_render(
            "the child did not finish, and a run that did not finish "
            "publishes no picture of itself")
        progress.failed(error)
        progress.close()
        raise
    if str(report["result"]) != "PASS":
        pictures = progress.withdraw_early_render(
            "the child's own health check refused this forecast, and a "
            "run that refused itself publishes no picture of itself")
        if progress.render_plan is not None:
            _record_products(
                progress, report, status="WITHDRAWN",
                reason=("this child did not pass, so the "
                        f"{pictures} picture(s) the early render had "
                        "published were removed; the frames and the "
                        "checkpoints are on disk and can be drawn by "
                        "hand"),
                render_command=_render_command_text(progress.render_plan))
    else:
        try:
            _finish_child_render(progress, report=report)
        except BaseException as error:
            # Before the failure event, so the stream a reader tails
            # ends on the failure rather than on a picture published by
            # a thread that was still running when it was raised.
            progress.wait_early_render()
            progress.failed(error, stage="finalize")
            progress.close()
            raise
    progress.emit(
        "completed", stage="forecast", result=report["result"],
        outputs=len(report.get("outputs", []) or []),
        # Time to first plot and the render's own summary ride the
        # terminal event, exactly as they do on the chain's, so one
        # reader shape serves a forecast and a downscaled forecast.
        first_products_seconds=progress.first_products_seconds,
        **({"render_summary": progress._render_summary}
           if progress._render_summary is not None else {}))
    progress.close()
    return report


def _render_command_text(render_plan: dict) -> str:
    """The render command for this plan, as a reader would type it."""

    from gpuwm.go_cli import printable, render_command

    return printable(render_command(render_plan))


def _record_products(progress: "_ChildProgress", report: dict,
                     **block) -> None:
    """Record what became of this child's pictures, in its own report.

    ``report.json`` is the document that says what this run produced,
    and it used to say ``PASS`` beside an empty picture tree whenever
    the render stage failed -- while the process exited 1 with a
    traceback.  One run cannot have two verdicts: ``result`` stays the
    forecast's, which passed, and this block is the pictures' own.

    The directory is the one the RUN resolved, read off the progress
    object rather than off the arguments: a run that never got as far
    as opening its stream has no report to amend either.
    """

    report["products"] = dict(block)
    if progress.outdir is None:
        return
    _publish_report(report, Path(progress.outdir))


def _publish_report(report: dict, outdir: Path) -> None:
    """Write ``report.json`` through a rename, from its one writer."""

    temporary = outdir / "report.json.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, outdir / "report.json")


def _finish_child_render(progress: "_ChildProgress", *,
                         report: dict) -> None:
    """Draw a finished child's products through the shared render stage.

    :func:`gpuwm.runplan._finish_render` is the finalize render every
    other route runs: it opens the finalize stage, hands the plan to
    :func:`gpuwm.go_cli._render_stage`, and refuses -- naming the render
    command to run by hand -- when a requested product set produced
    nothing.  A child gets that function, not a copy of it.

    A stage that EXITS NONZERO is a different outcome from one that drew
    nothing, and it is the outcome a product the renderer cannot draw
    for these frames produces.  That is a
    :class:`gpuwm.go_cli.GoStageFailed`, which subclasses ``Exception``
    and nothing the CLI boundary recognises -- so it left this door
    printing a traceback at exit 1 over a ``report.json`` that said
    ``PASS``.  It becomes this door's own refusal here: one sentence,
    exit 2, with the command that draws the finished frames by hand, and
    a ``products`` block in the report so the two documents agree.
    """

    if progress.render_plan is None:
        return
    from gpuwm.go_cli import GoStageFailed
    from gpuwm.runplan import _finish_render

    try:
        _finish_render(progress.render_plan, observer=progress,
                       door="downscale")
    except GoStageFailed as failure:
        command = _render_command_text(progress.render_plan)
        early = progress.early_pictures()
        outcome = (f"this run's pictures are incomplete: {early} drawn early "
                   "from the first frame, the rest not drawn" if early
                   else "this run has no pictures")
        _record_products(
            progress, report, status="FAILED",
            reason=(f"the render stage exited {failure.code}; the "
                    "forecast itself passed and its frames are on disk"),
            render_command=command, drawn_early=early)
        raise OfflineChildContractError(
            "The child integrated and its frames are on disk, but the "
            f"render stage exited {failure.code}, so {outcome}. Next: draw "
            "the saved frames by hand, which names the product that could "
            "not be drawn:\n  "
            + command) from failure
    progress.finish_stage()
    _record_products(
        progress, report, status="DRAWN",
        render_products=str(progress.render_plan.get("render_products")),
        render_summary=progress._render_summary,
        render_command=_render_command_text(progress.render_plan))


def _run(args: argparse.Namespace,
         progress: "_ChildProgress") -> dict[str, object]:
    import cupy as cp
    from gpuwm.core import streaming
    from gpuwm.core.refl import consume_refl_10cm, refl_10cm_is_stashed
    from gpuwm.ingest.lateral_bc import (
        attach_streaming_lateral_boundaries,
        bind_lateral_boundary_clock,
        lateral_boundary_reload_count,
        lateral_boundary_resident_bytes,
    )
    from gpuwm.io.restart import restart_filename, write_restart
    from gpuwm.io.wrfout import wrfout_filename

    started = time.perf_counter()
    # Pictures are this route's default, so "this computer cannot draw"
    # is admitted HERE -- before the archived parent is read and the
    # child is integrated -- and not discovered after the forecast,
    # where the only outcome left is a completed child reported as a
    # failure.  `gpuwm go` refuses its own chain at the same point, for
    # the same reason, and names the same two ways out.
    from gpuwm.first_products import early_render_requested

    render_products = getattr(args, "render_products", None)
    if early_render_requested(render_products):
        from gpuwm.go_cli import render_extra_missing

        missing = render_extra_missing()
        if missing is not None:
            from gpuwm.explain import layered

            raise OfflineChildContractError(layered(
                RENDERER_MISSING_REMEDY, missing))
    # ``outdir_reserved`` means the caller already applied the same
    # never-adopt reservation in this process (``gpuwm downscale`` does,
    # so the config it derives can live inside the run it describes).
    # Absent, this is the reservation.
    outdir = (Path(args.outdir).resolve()
              if getattr(args, "outdir_reserved", False)
              else _create_output_root(args.outdir))
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
    # A child at a DIFFERENT level count than its parent is now built by the
    # conservative vertical remap (gpuwm/vertical_remap.py), which requires
    # the child to name the ladder it wants.  A level count on its own is
    # refused rather than filled in: make_vertical_coord's default is a
    # UNIFORM ladder, so a child that asked only for "more levels" off a
    # stretched parent would silently get a different atmosphere, not a finer
    # sampling of the same one.
    if int(cfg.nz) != int(dims["bottom_top"]) and cfg.eta_levels is None:
        raise OfflineChildContractError(
            f"child nz={cfg.nz} differs from parent nz={dims['bottom_top']} "
            "but the child config names no eta_levels: a deeper child has to "
            "declare the ladder it wants, because a bare level count would "
            "be filled in with a uniform ladder and the child would start "
            "from a different atmosphere than its parent, not a finer "
            "sampling of it.  Add eta_levels to the child config (gpuwm "
            "downscale --child-levels writes one for you).")
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
    # The child's clock, checked by the same function the plan review
    # checked it with (child_cadence), so a cadence that is not a whole
    # number of steps was refused when the child was reviewed and cannot
    # surface here for the first time.  The child's OWN restart cadence is
    # honoured: ``restart_interval_s`` rides into every derived child
    # config verbatim from the parent, and the door's own refusal tells a
    # user "the parent needs restart_interval_s inside its window to be
    # downscalable" -- yet the child wrote one final checkpoint under a
    # name no discovery recognised, so no downscaled run was ever
    # downscalable.  A setting accepted and silently dropped is a defect;
    # checkpoint_due is the cadence.
    cadence = child_cadence(
        cfg, health_interval_seconds=float(args.health_interval_seconds))
    steps = cadence.steps
    output_steps = cadence.output_steps
    health_steps = cadence.health_steps
    checkpoint_due = cadence.checkpoint_due
    surface = None
    surface_from = getattr(args, "child_surface_from", None)
    if surface_from is not None:
        surface = read_child_surface_state(
            surface_from, child_ny=int(cfg.ny), child_nx=int(cfg.nx),
            num_soil_layers=soil_layer_count(cfg))
    elif child_surface_requirement(cfg) is not None:
        # RESOLVED HERE, before interpolate_parent_initial_state and
        # build_offline_lateral_boundaries spend minutes on the parent
        # archive.  The guard in _initialize_child_physics used to be the
        # only one on this route, and it fires AFTER that work -- so a
        # child that could never start still paid for the preprocessing
        # first.  Defect #275: derive the child surface from the
        # parent's own history (WRF's input_from_file = .false. route)
        # rather than refusing for a file no command in the product
        # could produce for a config-driven parent.
        try:
            surface = derive_child_surface_from_parent(
                contract.frames[0].path, placement=placement,
                num_soil_layers=soil_layer_count(cfg))
        except OfflineChildContractError as error:
            raise OfflineChildContractError(
                f"{child_surface_requirement(cfg)}\n"
                f"  and this parent archive cannot supply one either: "
                f"{error}") from error
        warn("child surface state interpolated from the parent's own "
             "history rather than built on the child grid -- "
             + DERIVED_CHILD_SURFACE_CAVEAT,
             why="This is WRF's input_from_file = .false. route for a "
                 "nest with no wrfinput of its own (med_nest_initial's "
                 "med_interp_domain), run through the Registry's masked "
                 "land interpolator.")
    surface_file_receipts = (
        [] if surface is None else [_file_receipt(surface.path)])
    _log("contract_pass", frames=len(contract.frames),
         cadence_seconds=contract.interval_seconds,
         geometry_sha256=contract.geometry_sha256,
         source_physics=dict(binding.receipt()),
         target_mp_physics=int(cfg.mp_physics),
         child_shape=[cfg.nz, cfg.ny, cfg.nx],
         child_spacing_m=[cfg.dy, cfg.dx])

    # THE [tiles] DECISION, taken HERE on a COLD card and never again.  The
    # same function ``gpuwm downscale`` reviewed this child with
    # (gpuwm.downscale_pricing.price_child): the configured envelope from
    # the itemized estimator, judged against a machine measured before this
    # process has allocated a byte on the device.  It used to be taken after
    # the initial state, the boundary tables and the physics driver had
    # filled the card, with no estimate and no machine, so the tile planner
    # measured what was LEFT and charged the whole rung's fixed cost against
    # it: a 138x138x49 child the review had admitted at 2.90 GiB against
    # 7.32 GiB free was refused at "no tile fits in 3.49 GiB" after the
    # whole archive had been interpolated.  A card that is genuinely too
    # small refuses here, before anything is interpolated or allocated on
    # the device, with the measured figure and the way out.
    from tilestream.autoplan import CannotPlan

    try:
        # The estimator's default forcing model, as the fitted sizing and
        # the review price it: the child's boundary intervals are streamed
        # from the host one at a time, so counting every archived interval
        # as retained on the device would price this child a third above
        # what it holds and stream a child that fits.
        pricing = downscale_pricing.price_child(
            cfg, tiles, machine=downscale_pricing.cold_machine(tiles),
            basis=downscale_pricing.MEASURED_BASIS)
    except CannotPlan as error:
        raise OfflineChildContractError(str(error)) from error
    tiles = pricing.options
    streaming_decision = pricing.decision
    _log("child_streaming_decision", **pricing.plan_entry())

    # PUBLISHED HERE: after every contract that can refuse this child has
    # passed and before the first minute of preprocessing is spent, so a
    # reader watching the directory sees a run it can trust, and sees it
    # from the beginning of the work rather than the end.
    progress.start(
        outdir=outdir, child_config=Path(args.child_config),
        ratio=int(placement.parent_grid_ratio),
        start_time=contract.start_time,
        parent={
            "run_dir": str(Path(contract.frames[0].path).parent),
            "restart": (None if args.parent_restart is None
                        else str(args.parent_restart)),
            "frames": len(contract.frames),
            "cadence_seconds": float(contract.interval_seconds),
        },
        name=child_run_name(
            Path(contract.frames[0].path), grid_id=int(cfg.grid_id),
            ratio=int(placement.parent_grid_ratio), dx=float(cfg.dx)))
    # Armed beside the manifest, so the analysis frame -- written before
    # a single step is integrated -- becomes a picture while the child
    # is still running, as it does on every other route.
    progress.arm_render(outdir=outdir, render_products=render_products)
    progress.emit("stage_started", stage="initialize", phase="preprocess")

    initial = interpolate_parent_initial_state(
        contract.frames[0].path, placement,
        physics_binding=binding, target_mp_physics=cfg.mp_physics,
        backend=args.preprocess_backend,
        child_eta_levels=cfg.eta_levels)
    prepared = build_offline_lateral_boundaries(
        contract, placement,
        target_mp_physics=cfg.mp_physics,
        backend=args.preprocess_backend,
        child_eta_levels=cfg.eta_levels,
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    # A float32 SINT of a number moment can round across zero.  When it
    # does, say so with the numbers: the cells touched, the tolerance and
    # the most negative value are what tell a reader whether they watched
    # rounding get cleaned up or an interpolation start to go wrong.  An
    # empty account emits nothing, so silence here means it landed clean.
    initial_clamp = initial.receipt.get("positive_definite_clamp") or {}
    if initial_clamp:
        _log("initial_positive_definite_clamp",
             fields={name: dict(account)
                     for name, account in initial_clamp.items()})
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
    # The DECISION was taken above, before preprocessing, on the cold card.
    # The STEPPER is made here, AFTER the physics driver is attached, never
    # before: the builder fills the store from the PREPARED state and builds
    # every tile buffer with the domain's own physics selectors, so a
    # stepper made against a bare DomainState would carry a different
    # inventory than the domain it is meant to be.
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
        progress.output_committed(domain=int(cfg.grid_id),
                                  valid_time=valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  path=str(path), bytes=path.stat().st_size)

    checkpoint_paths: list[Path] = []
    carriers_refreshed = 0

    def emit_checkpoint() -> None:
        # A DISCOVERABLE set (gpuwm.io.restart.restart_filename's instant
        # naming, the one gpuwm.resume.discover_checkpoint_sets recognises),
        # so this run can be the parent of the next downscale.  The
        # streamed state is refreshed first, exactly as the history writer
        # does: a streamed domain's forecast lives in the pinned host store
        # and this DomainState is the snapshot that filled it, so without
        # the copy every checkpoint would be the initial condition under a
        # later clock.  Zero and a getattr when resident.
        nonlocal carriers_refreshed
        carriers_refreshed = streaming.refresh_streamed_state(stepper, child)
        valid = initial.valid_time + timedelta(
            seconds=float(clock.elapsed_seconds))
        path = write_restart(
            outdir / restart_filename(valid, domain=f"d{cfg.grid_id:02d}"),
            child, cfg)
        checkpoint_paths.append(Path(path))
        _log("child_checkpoint", elapsed_seconds=float(clock.elapsed_seconds),
             path=str(path), bytes=Path(path).stat().st_size)

    progress.emit("stage_started", stage="forecast", phase="integrate")
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
            progress.emit("model_progress", domain=int(cfg.grid_id),
                          model_seconds=float(clock.elapsed_seconds),
                          run_seconds=float(cfg.run_seconds),
                          outer_step=int(step_index), total_steps=int(steps),
                          wall_seconds=time.perf_counter() - started,
                          # The reader that offers "downscale from this
                          # run" takes its checkpoint directory from here,
                          # as it does for every other route.
                          **({"last_checkpoint": str(checkpoint_paths[-1])}
                             if checkpoint_paths else {}))
            if child_health["nan"]:
                raise RuntimeError(
                    f"offline child became non-finite at step {step_index}")
        if output_due:
            emit_output()
        if step_index in checkpoint_due:
            # The final step is always due, so the run's last checkpoint
            # is the instant-named set the next downscale discovers.  The
            # refresh inside emit_checkpoint is what keeps a streamed
            # child's checkpoint from being the analysis under a later
            # clock, stated there rather than inferred from the output
            # cadence happening to coincide.
            emit_checkpoint()
    restart = checkpoint_paths[-1]
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
        # The decision the run was integrated with, and what it was judged
        # on: the same block the plan review wrote, so a reader can hold
        # the two documents side by side and find one answer.
        "streaming": pricing.plan_entry(),
        "streamed_carriers_refreshed": carriers_refreshed,
        # Every checkpoint this run wrote, on the child's own
        # restart_interval_s, under the instant naming the next downscale
        # discovers.  The last one is the run's end state.
        "restart_interval_s": float(cfg.restart_interval_s),
        "checkpoints": [str(path) for path in checkpoint_paths],
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
    # THE AEROSOL SOURCE, on the one route that cannot name a dataset.
    # This child is not a real-data initialization: every transported
    # aerosol scalar (QNWFA/QNIFA) and both surface emission fields
    # (QNWFA2D/QNIFA2D) are interpolated out of the archived PARENT
    # history frame, so this run never resolves a climatology and has no
    # ``RealInitResult.aerosol_initialization`` to publish.  Saying
    # nothing would leave an mp=28 child's report looking exactly like a
    # wsm6 child's; saying "no dataset" would be a claim this process
    # cannot make.  So it reports the third state -- applicable, not
    # recorded here -- and names the report that does hold the answer.
    report.update(aerosol_source_report_entry(
        {}, mp_physics=cfg.mp_physics,
        when_unrecorded=(
            "this child's aerosol state is INHERITED, not initialized: "
            "QNWFA/QNIFA and the QNWFA2D/QNIFA2D surface emission fields "
            "are interpolated from the archived parent history frames "
            "listed under parent_frames, and this process runs no aerosol "
            "dataset resolution of its own. Which source filled the "
            "parent's fields -- WRF's monthly WIF climatology or "
            "thompson_init's synthetic profile -- is recorded in the "
            "PARENT run's report, and is the answer for this child too")))
    _publish_report(report, outdir)
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
    parser.add_argument("--render-products", default=None, metavar="LIST",
                        dest="render_products",
                        help="which products this child's frames are drawn "
                             "into <outdir>/png: a comma-separated list of "
                             "catalog slugs, 'all', or 'none'.  Absent draws "
                             "nothing, because this runner is the engine "
                             "door; `gpuwm downscale` is the door that "
                             "defaults to drawing")
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
