"""Complete-roster forecast barrier shared by the public DA controller.

The control is unanalysed. On the serial route it runs first, in process.
On the packed route it is a worker of its own (:class:`ControlLaunch`):
its leg depends on nothing but its own restart, so the controller starts
it the moment the previous barrier passes, on a card the analysis leaves
free, and it is done before the members need that card. Member numerical
work belongs to the complete-tree worker. Inputs stay intact until the
whole roster passes.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
import time

from gpuwm.da.member_wave import atomic_json


def card_uuid(card):
    return card if isinstance(card, str) else card["uuid"]


class ControlLaunch:
    """The control's leg as an owned worker, sealed and started now.

    ``workdir`` is that leg's packed directory (the members' requests and
    the common seal join it later); ``gpu_uuid`` the card it runs on.
    """

    def __init__(self, context, *, workdir, gpu_uuid, server=None):
        from gpuwm.da import member_transport, member_wave

        self.started = time.monotonic()
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=False)
        self.context = context
        self.gpu_uuid = str(gpu_uuid)
        self.common = member_transport.write_common(
            self.workdir/"common.json", context)
        (job,) = member_transport.write_requests(
            self.workdir, self.common, {"control": context}, ["control"])
        self.job = member_wave.OwnedJob(
            job, self.gpu_uuid, self.workdir/"control.log", server=server)

    def stop(self):
        self.job.stop()


def forecast_roster(trajectories, *, context_for, packed, devices,
                    members_per_card, member_peak_bytes, workdir, stage,
                    run_leg, release, timeout_seconds, host_reserve_bytes=48*1024**3,
                    control=None, pending_files=None, before_consume=None,
                    servers=None):
    """Run one leg of every trajectory to a complete barrier.

    Packed: ``control`` is a :class:`ControlLaunch` already running for
    this leg, or ``None`` to start it here on the last card beside the
    members; ``pending_files`` lets each member request link the pending
    file the recovery generation already wrote
    (:func:`gpuwm.da.member_transport.write_requests`).  Requests are
    sealed on threads while the first members already run.
    ``before_consume`` runs after the barrier and before any input restart
    is consumed (the controller waits there for a generation still being
    written from those restarts).
    """
    from gpuwm.da import member_transport, member_wave
    from gpuwm.io.restart import tree_restart_members

    trajectories = list(trajectories)
    if trajectories != ["control", *range(len(trajectories)-1)]:
        raise ValueError("forecast roster must contain control then every member in order")
    started = time.monotonic()
    results, contexts = {}, {}
    wave_receipt = None
    control_receipt = "unanalysed, completed before member wave"
    if packed:
        workdir = Path(workdir)
        in_process = control is None and not devices
        if in_process:
            # No card to give the control a worker of its own (measurement
            # tools drive this route without cards): in process, first.
            workdir.mkdir(parents=True, exist_ok=False)
            contexts["control"] = context_for("control")
            try:
                results["control"] = run_leg(contexts["control"], "control")
            finally:
                release()
            common = member_transport.write_common(
                workdir/"common.json", contexts["control"])
        elif control is None:
            control = ControlLaunch(context_for("control"), workdir=workdir,
                                    gpu_uuid=card_uuid(devices[-1]),
                                    server=(servers or {}).get(
                                        card_uuid(devices[-1])))
        elif Path(control.workdir) != workdir:
            raise ValueError("the running control belongs to another leg's directory")
        if not in_process:
            contexts["control"] = control.context
            common = control.common
        for name in trajectories[1:]:
            contexts[name] = context_for(name)
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=member_wave.HASH_THREADS) as seal:
                jobs = member_transport.write_requests(
                    workdir, common, contexts, trajectories[1:],
                    pending_files=pending_files, pool=seal)
                envelopes = member_wave.run_wave(
                    jobs, devices, members_per_card, workdir/"wave",
                    timeout_seconds, member_peak_bytes=member_peak_bytes,
                    host_reserve_bytes=host_reserve_bytes,
                    occupied=({} if in_process
                              else {control.gpu_uuid: control.job}),
                    servers=servers)
            envelopes = list(envelopes)
            if not in_process:
                envelopes.insert(0, control.job.wait(
                    max(1.0, timeout_seconds - (time.monotonic() - started))))
            # This call returns only after hash, identity, complete-tree clocks
            # and every array artifact passed for every member. No partial solve.
            with ThreadPoolExecutor(max_workers=max(1, min(
                    member_wave.HASH_THREADS, len(envelopes)))) as pool:
                loaded = list(pool.map(member_transport.load_result,
                                       envelopes))
            names = trajectories[1:] if in_process else trajectories
            for name, result in zip(names, loaded, strict=True):
                results[name] = result
            wave_receipt = str(workdir/"wave"/"forecast-leg-receipt.json")
            if not in_process:
                control_receipt = {
                    "route": "own worker, unanalysed",
                    "gpu_uuid": control.gpu_uuid,
                    "seconds_before_this_barrier_began": round(
                        started - control.started, 3),
                    **control.job.record()}
            # Adopt outputs into the existing stage owner before consuming inputs.
            # All files are on this run's filesystem, so adoption is a rename.
            for name in names:
                result = results[name]
                source_members = tree_restart_members(result.restart)
                target = stage.directory(contexts[name].absolute_leg_number, name)
                if target.exists():
                    raise ValueError(f"member publication would replace an existing set: {target}")
                target.mkdir(parents=True)
                root_name = Path(result.restart).name
                for source in source_members.values():
                    shutil.move(str(source), target/source.name)
                result.restart = target/root_name
        except BaseException:
            if control is not None:
                control.stop()
            raise
        finally:
            removed = []
            for directory in [*workdir.glob("m[0-9][0-9][0-9]"),
                              workdir/"control"]:
                if not directory.is_dir():
                    continue
                for path in directory.glob("*.npz"):
                    removed.append({"path": str(path), "bytes": path.stat().st_size})
                    path.unlink()
                scratch = directory/"stage"
                if scratch.is_dir():
                    removed.extend({"path": str(p), "bytes": p.stat().st_size}
                                   for p in scratch.rglob("*") if p.is_file())
                    shutil.rmtree(scratch)
            atomic_json(workdir/"transport-cleanup.json", {"removed": removed})
    else:
        control_context = context_for("control")
        contexts["control"] = control_context
        try:
            results["control"] = run_leg(control_context, "control")
        finally:
            release()
        reference = results["control"]
        for name in trajectories[1:]:
            context = context_for(name)
            contexts[name] = dataclasses.replace(
                context,
                setup_arrays=(reference.setup_arrays if context.setup_arrays is None else context.setup_arrays),
                thb_host=(reference.thb_host if context.thb_host is None else context.thb_host))
        for name in trajectories[1:]:
            try:
                results[name] = run_leg(contexts[name], name)
            finally:
                release()
    if list(results) != trajectories:
        raise RuntimeError("member forecast barrier is missing a trajectory")
    for name in trajectories:
        result, context = results[name], contexts[name]
        if result.record.get("elapsed_seconds") != context.t_end:
            raise ValueError("member forecast ended at a different analysis clock")
        # The complete restart owner was used by both routes. No old array
        # snapshot or jumped clock can be used to join forecast legs.
        from gpuwm.io.restart import read_restart_header
        for member in tree_restart_members(result.restart).values():
            if read_restart_header(member)["elapsed_seconds"] != context.t_end:
                raise ValueError("member restart tree is not at the analysis clock")
    if before_consume is not None:
        before_consume()
    for name in trajectories:
        source = contexts[name].restart
        consumed = source is not None and stage.consume(source)
        if "restart" in results[name].record:
            results[name].record["restart"]["consumed"] = bool(consumed)
    receipt = {
        "complete": True, "analysis_permitted": True,
        "route": "packed-uuid" if packed else "serial-shared-worker",
        "members": len(trajectories)-1, "members_per_card": members_per_card,
        "control": control_receipt,
        "wave_receipt": wave_receipt,
        "forecast_plus_io_wall_seconds": time.monotonic()-started}
    if packed:
        atomic_json(workdir/"barrier.json", receipt)
    return results, receipt
