"""The prepared front door arming its OWN early render.

THE GAP THIS CLOSES.  Arming lived on :class:`gpuwm.runplan.RunObserver`
and on ``gpuwm go``, so render-on-first-committed-frame was reachable
only through a HOST.  ``python -m gpuwm.prepared_single_domain_forecast``
-- the documented way to run a prepared bundle -- imported
:mod:`gpuwm.first_products` nowhere and declared no render flag, so it
committed its analysis frame (fsynced, self-validated, renamed) minutes
before the forecast ended with nothing watching.  MEASURED consequence:
reaching an 89 s launch-to-first-plot on that route took an EXTERNAL
process polling the frames for ``GPUWM_WRITE_COMPLETE`` -- a watcher
reimplementing a hook the writer already raises.

Separate from tests/test_first_products.py on purpose.  That module ends
with a module-level ``pytest.importorskip("wrf")`` for its
real-renderer group, which skips the WHOLE file on a box without the
render extra -- including its CPU tests.  These are CPU-only and must
run everywhere the runner does, so they live where nothing can skip
them.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm.first_products import (FIRST_PRODUCTS_SCHEMA, FirstProducts,
                                  read_receipt)
from gpuwm.runplan import EventStream, RunObserver

_VALID = datetime(1974, 4, 3, 18, 0, 0)


def _stand_in_renderer(*, names=("refl_d01-1km_x.png",),
                       payload=b"\x89PNG\r\n\x1a\nstand-in"):
    """A renderer that writes the files a real one would, and no more.

    It reads its output directory out of the composed command rather
    than being told it, so a change to how ``render_command`` spells
    ``--out`` breaks these tests instead of silently bypassing them.
    """

    def run(command):
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        for name in names:
            (out / name).write_bytes(payload)
        return subprocess.CompletedProcess(list(command), 0, "", "")

    return run


def _plan(tmp_path, *, products="refl,t2"):
    return {"run": tmp_path / "run", "render": tmp_path / "png",
            "render_products": products}


def _frame(tmp_path, *, under="run"):
    path = (tmp_path / under / "wrfout"
            / "wrfout_d01_1974-04-03_18_00_00")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"one durable history frame")
    return path


class _Recorder:
    def __init__(self):
        self.reports = []
        self.warnings = []

    def report(self, receipt):
        self.reports.append(receipt)

    def warn(self, code, message, **fields):
        self.warnings.append((code, message, fields))


def _runner_argv(tmp_path, *extra):
    return [
        "--source", "hrrr", "--prepared-root", str(tmp_path / "bundle"),
        "--proof-sha256", "0" * 64, "--source-manifest-sha256", "0" * 64,
        "--prepared-content-sha256", "0" * 64,
        "--experiment-config", str(tmp_path / "e.toml"),
        "--wps-namelist", str(tmp_path / "n.wps"),
        "--run-seconds", "60", "--history-interval-seconds", "60",
        "--io-mode", "history", "--outdir", str(tmp_path / "out"),
        *extra,
    ]


def _armed(tmp_path, *extra, observer=None):
    """What the runner's command line arms, without running a forecast."""

    from gpuwm import prepared_single_domain_forecast as runner

    args = runner._parse_args(_runner_argv(tmp_path, *extra))
    return runner._route_owned_first_products(
        args, outdir=tmp_path / "out", observer=observer, started=0.0)


# ---------------------------------------------------------------------------
# Arming: the same rule, from the same function, at a second door
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("products,armed", [
    ((), False), (("--render-products", "none"), False),
    (("--render-products", "NONE"), False),
    (("--render-products", "refl,t2"), True),
    (("--render-products", "all"), True),
])
def test_the_runners_own_arming_is_off_by_absence_too(tmp_path, products,
                                                      armed):
    """Default OFF by ABSENCE rather than by a second flag.

    A run that names no products gets exactly the behaviour it had
    before this existed, and one that spells ``none`` has already said it
    wants no pictures.  Both answers come out of the field that already
    holds "which products", so there is no second switch to disagree
    with it.
    """

    assert (_armed(tmp_path, *products) is not None) is armed


def test_the_runner_arms_with_the_dict_a_finalize_stage_would_use(tmp_path):
    """``run`` is the --outdir, because that is where the wrfouts land.

    ``go_cli.wrfout_frames`` globs ``plan["run"] / "wrfout"`` and this
    runner writes ``outdir / "wrfout"``, so a finalize render pointed at
    the same --outdir composes the same command over the same frames --
    which is what makes an early frame's bytes and a late one's
    comparable at all.
    """

    trigger = _armed(tmp_path, "--render-products", "refl")

    assert trigger.render_dir == tmp_path / "out" / "png"
    assert trigger.render_products == "refl"
    assert trigger._plan["run"] == tmp_path / "out"

    elsewhere = _armed(tmp_path, "--render-products", "refl",
                       "--render-dir", str(tmp_path / "pictures"))
    assert elsewhere.render_dir == tmp_path / "pictures"


def test_a_host_that_already_armed_one_keeps_it(tmp_path, capsys):
    """Two triggers on one directory would draw the first frame twice.

    run-plan's HRRR chain arms the render on its OBSERVER before it calls
    this runner, with the dict its own finalize stage will use.  A second
    trigger armed inside the runner would render the same frame again,
    publish two receipts over each other, and make the finalize skip
    depend on which worker finished last.
    """

    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    host = RunObserver(events, root_domain=1)
    host.arm_first_products(_plan(tmp_path, products="refl"))
    assert host.first_products is not None

    assert _armed(tmp_path, "--render-products", "refl",
                  observer=host) is None
    events.close()
    # And it says so.  A flag that was passed and quietly did nothing is
    # the failure this whole increment is about.
    assert "already armed" in capsys.readouterr().err


def test_a_host_without_one_does_not_stop_the_runner_arming(tmp_path):
    """An observer is not by itself a reason to decline."""

    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    host = RunObserver(events, root_domain=1)
    host.arm_first_products(_plan(tmp_path, products=None))
    trigger = _armed(tmp_path, "--render-products", "refl", observer=host)
    events.close()

    assert trigger is not None


# ---------------------------------------------------------------------------
# The seam: one attach, two consumers
# ---------------------------------------------------------------------------


def test_the_landing_hook_reaches_both_the_host_and_the_early_render(
        tmp_path):
    """ONE attach, with both consumers behind it.

    ``PerDomainWrfoutWriters.attach_progress_callback`` reads ONE
    ``output_committed`` attribute and assigns it to every writer's
    ``landing_observer`` UNCONDITIONALLY -- so a second attach does not
    add a consumer, it silently replaces the first, and the run keeps
    publishing frames with the earlier consumer no longer watching.  The
    REAL method is driven here, over a stand-in writer, so the fan-out is
    proven against the binding contract it has to satisfy rather than
    against a copy of it.
    """

    from gpuwm.io.wrfout import PerDomainWrfoutWriters
    from gpuwm.prepared_single_domain_forecast import _LandingObservers

    class _Writer:
        paths = ()
        pending = 0
        landing_observer = None

    recorder = _Recorder()
    trigger = FirstProducts(_plan(tmp_path), report=recorder.report,
                            warn=recorder.warn,
                            runner=_stand_in_renderer())
    hosted = []
    fanout = _LandingObservers(
        lambda **event: hosted.append(event), trigger.frame_committed)

    writers = object.__new__(PerDomainWrfoutWriters)
    writers._writers = {1: _Writer()}
    writers.attach_progress_callback(fanout)
    assert writers._writers[1].landing_observer == fanout.output_committed

    frame = _frame(tmp_path)
    writers._writers[1].landing_observer(
        domain=1, valid_time=_VALID, path=frame)
    trigger.wait(timeout=30.0)

    assert hosted == [{"domain": 1, "valid_time": _VALID, "path": frame}]
    assert recorder.reports and recorder.reports[0]["frame"] == str(frame)


def test_a_consumer_that_raises_cannot_silence_the_other(tmp_path):
    """The writer keeps that contract for one consumer; so does the pair."""

    from gpuwm.prepared_single_domain_forecast import _LandingObservers

    def angry(**event):
        raise RuntimeError("telemetry blew up")

    seen = []
    _LandingObservers(angry, lambda **event: seen.append(event)
                      ).output_committed(domain=1, valid_time=_VALID,
                                         path="frame")
    assert seen == [{"domain": 1, "valid_time": _VALID, "path": "frame"}]


def test_nothing_is_attached_when_nobody_is_listening(tmp_path):
    """An unhosted, unrendered run behaves exactly as it always did."""

    from gpuwm.prepared_single_domain_forecast import _LandingObservers

    assert not _LandingObservers(None, None)
    assert _LandingObservers(lambda **event: None, None)


def test_the_runner_attaches_exactly_once_and_waits_before_it_returns():
    """Two properties that cannot be read off a return value.

    A second ``attach_progress_callback`` would unhook the host's
    observer, and a runner that returned without joining would publish a
    report for pictures a killed DAEMON thread never wrote.
    """

    import inspect

    from gpuwm import prepared_single_domain_forecast as runner

    source = inspect.getsource(runner.run_prepared_forecast)
    assert source.count("attach_progress_callback(") == 1
    assert "_LandingObservers(" in source
    wait = source.index("first_products.wait()")
    assert wait < source.index('_atomic_json(outdir / "report.json"')
    assert 'report["first_products"] = receipt' in source


def test_the_runner_arms_before_its_preflight_not_after(tmp_path):
    """The frame this renders is durable before a single step is taken.

    The history alarm is true at t = 0, so anything armed after the
    forecast opens is already racing the thing it exists to catch.
    """

    import inspect

    from gpuwm import prepared_single_domain_forecast as runner

    source = inspect.getsource(runner.main)
    assert source.index("_route_owned_first_products(") < source.index(
        "preflight_prepared_forecast(")


# ---------------------------------------------------------------------------
# The contract: same receipt, same event, one number
# ---------------------------------------------------------------------------


def test_the_receipt_is_the_schema_the_hosted_route_publishes(tmp_path):
    """Same module, same receipt, so no reader can tell the doors apart.

    Driven through the RUNNER's own arming rather than through a
    hand-built ``FirstProducts``: what has to be pinned is that this
    door publishes ``gpuwm.first-products.v1`` beside its pictures, so
    ``go_cli._render_stage``'s skip -- which reads that receipt by
    schema id -- works over output this door produced.
    """

    trigger = _armed(tmp_path, "--render-products", "refl,t2")
    trigger._runner = _stand_in_renderer()
    trigger.frame_committed(domain=1, valid_time=_VALID,
                            path=_frame(tmp_path, under="out"))
    receipt = trigger.wait(timeout=30.0)

    assert receipt["schema"] == FIRST_PRODUCTS_SCHEMA
    assert receipt["render_products"] == "refl,t2"
    assert read_receipt(tmp_path / "out" / "png") == receipt


def test_the_runner_reports_the_time_to_first_plot_on_stdout(tmp_path,
                                                             capsys):
    """A process with no event stream still publishes the number.

    ``first_products_ready`` is what a hosted run emits, and its whole
    value is marking the instant the pictures became readable.  A
    standalone run has nowhere to emit it, so it says the same thing in
    the one place it has -- and says WHICH clock it measured from, rather
    than implying the hosted one.
    """

    trigger = _armed(tmp_path, "--render-products", "refl")
    trigger._runner = _stand_in_renderer()
    frame = _frame(tmp_path, under="out")

    trigger.frame_committed(domain=1, valid_time=_VALID, path=frame)
    assert trigger.wait(timeout=30.0) is not None

    printed = capsys.readouterr().out
    assert "first products ready" in printed
    assert "after this runner started" in printed
    assert "--products refl" in printed
    assert "refl_d01-1km_x.png" in printed


def test_a_render_that_declines_never_fails_the_forecast(tmp_path, capsys):
    """Telemetry never fails a run, at this door as at the other one."""

    def angry(command):
        raise RuntimeError("the renderer exploded")

    trigger = _armed(tmp_path, "--render-products", "refl")
    trigger._runner = angry
    trigger.frame_committed(domain=1, valid_time=_VALID,
                            path=_frame(tmp_path, under="out"))

    assert trigger.wait(timeout=30.0) is None
    assert "first_products_failed" in capsys.readouterr().err
