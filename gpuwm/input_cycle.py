"""Verify a cycle assertion against existing input times without retiming them."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path


def refusal(cycle: str) -> str:
    """The refusal for a ``--cycle`` that differs from the inputs' own start.

    Breakage the refusal prevents: a cycle on a run whose inputs are named
    files, an existing prepared bundle or a checkpoint would be accepted
    and read by nothing, and the run would start at those inputs' own time
    while its command named another.  An assertion equal to that time is
    no such conflict and is recorded as a no-op instead (2.8.1's blanket
    refusal stopped the production ERA5 worker, which passes the bundle's
    own cycle).
    """
    return (f"--cycle {cycle} names the cycle a download route fetches, and "
            "this run takes its inputs from [case_data], an existing "
            "prepared bundle or a checkpoint, whose times a cycle cannot "
            "move. Next: omit --cycle.")


def _utc(value) -> datetime:
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def _bundle_start(root: Path) -> datetime:
    from gpuwm.stage_cli import resolve_bundle

    document = resolve_bundle(root)["payload"]
    values = [document[key] for key in ("valid_time", "model_start_time")
              if document.get(key) is not None]
    times = document.get("forcing_times")
    if isinstance(times, list) and times:
        values.append(times[0])
    starts = {_utc(value) for value in values}
    if len(starts) != 1:
        raise ValueError("prepared document has missing or inconsistent initial times")
    return starts.pop()


def _forcing_start(data) -> datetime:
    from gpuwm.ingest.grib import inspect_era5_forcing_times

    times = inspect_era5_forcing_times(data.forcing, data.vtable)
    if not times:
        raise ValueError("declared forcing has no valid times")
    return _utc(times[0])


def verify(cycle: str, *, prepared_root=None, restart=None, data=None,
           launch_start=None, allow_pending: bool = False) -> dict | None:
    """Return a no-op receipt only after reading the actual input clock.

    A fetch may supply missing declared forcing. In that one case a caller
    can defer the check until its fetch finishes, before preparation starts.
    The experiment and fetch timestamps are never evidence for equality.
    """
    try:
        requested = datetime.strptime(cycle, "%Y-%m-%dT%H")
        basis = "prepared_bundle" if prepared_root is not None else "case_data"
        header = None
        if restart is not None:
            from gpuwm.io.restart import read_restart_header

            header = read_restart_header(Path(restart))
            basis = "checkpoint"
        # Tree headers record the domain clock. Single-domain headers with
        # radiation record its origin; otherwise use the actual input origin.
        origin = None
        if header is not None:
            origin = header.get("domain_start_time")
            if origin is None:
                origin = (header.get("physics_setup") or {}).get("radiation", {}).get("start_time")
        if origin is not None:
            start = _utc(origin)
        elif prepared_root is not None:
            start = _bundle_start(Path(prepared_root))
        elif data is not None:
            missing = [path for path in data.forcing if not path.is_file()]
            if missing and allow_pending and restart is None:
                return None
            start = _forcing_start(data)
            # A wider forcing series can be sliced by the forecast loader.
            # Its first time alone cannot assert that a later configured
            # launch uses it as time zero, including radiationless restarts.
            if launch_start is not None and _utc(launch_start) != start:
                raise ValueError("the configured start differs from the forcing's first valid time")
        else:
            raise ValueError("no input clock is available")
        if header is not None:
            from gpuwm.io.restart import _admissible_elapsed_seconds

            seconds = _admissible_elapsed_seconds(header.get("elapsed_seconds"), "checkpoint")
            if header.get("domain_start_time") is not None:
                offset = float(header.get("domain_start_ticks", 0)) / float(header["tick_den"])
                if not math.isfinite(offset) or offset < 0 or offset > seconds:
                    raise ValueError("checkpoint domain clock offset is invalid")
                seconds -= offset
            start += timedelta(seconds=seconds)
        if requested != start:
            raise ValueError(f"actual input start is {start.isoformat()} UTC")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            AttributeError, OverflowError, ZeroDivisionError) as error:
        raise ValueError(refusal(cycle) + f" Input-time check: {error}") from error
    return {"cycle": cycle, "input_start_time": start.isoformat(),
            "basis": basis, "action": "no-op", "retimed": False}
