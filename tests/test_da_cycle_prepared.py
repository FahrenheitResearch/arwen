"""The cycling driver's leg-boundary clock placement.

CPU only.  The driver itself needs a GPU and a prepared case; this pins
the one piece of arithmetic it invents -- putting a freshly built clock
at a leg boundary -- against the clock's own stepping, which is the
thing it has to be indistinguishable from.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.clock import DomainClock, DomainTicks
from tools.da_cycle_prepared import jump_clock


TICK_DEN = 1
DT = 60.0
LBC_SECONDS = 10800.0          # the GFS 3-hourly boundary interval


def _spec(*, lbc: bool = True) -> DomainTicks:
    step_ticks = int(DT * TICK_DEN)
    return DomainTicks(
        grid_id=1, parent_id=0, parent_time_step_ratio=1,
        step_ticks=step_ticks, dt_fp32=np.float32(DT),
        history_ticks=int(3600 * TICK_DEN), restart_ticks=None,
        radt_ticks=None, stepra=None, cudt_ticks=None, stepcu=None,
        bldt_ticks=None, stepbl=None,
        lbc_interval_ticks=int(LBC_SECONDS * TICK_DEN) if lbc else None,
        start_ticks=0)


def _fresh(spec: DomainTicks) -> DomainClock:
    return DomainClock(spec, TICK_DEN, int(86400 * TICK_DEN))


def _stepped_to(spec: DomainTicks, seconds: float) -> DomainClock:
    """A clock that got there the way the integrator gets there.

    ``prepare_step`` before every step (WRF's ``dtbc = dtbc + dt``
    recurrence) and the external-LBC reset at every seam, which is what
    the driver's replay has to reproduce.
    """
    clock = _fresh(spec)
    while clock.elapsed_seconds < seconds:
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        clock.advance()
    return clock


@pytest.mark.parametrize("hours", [1, 2, 3, 4, 7])
def test_jumped_clock_is_indistinguishable_from_a_stepped_one(hours):
    spec = _spec()
    seconds = hours * 3600.0
    stepped = _stepped_to(spec, seconds)
    jumped = _fresh(spec)
    jump_clock(jumped, seconds, DT)

    assert jumped.ticks == stepped.ticks
    assert jumped.step_count == stepped.step_count
    assert jumped.elapsed_seconds == seconds
    # The FP32 boundary accumulator, bit for bit -- this is the field a
    # closed-form steps*dt would get subtly wrong.
    assert (jumped.dtbc_fp32.tobytes() == stepped.dtbc_fp32.tobytes()), (
        f"dtbc {jumped.dtbc_fp32!r} != stepped {stepped.dtbc_fp32!r}")


def test_a_leg_landing_on_a_seam_carries_the_interval_not_a_zero():
    """The reset is the integrator's top-of-step work, not the driver's.

    A clock placed exactly on a boundary seam holds a full interval of
    accumulation, which is what a clock that stepped there holds; the
    integrator zeroes it on its own first step.  Pre-zeroing here would
    hand the integrator a state it never produces and would only look
    right because the reset happens to land next.
    """
    spec = _spec()
    at_seam = _fresh(spec)
    jump_clock(at_seam, LBC_SECONDS, DT)
    assert float(at_seam.dtbc_fp32) == pytest.approx(LBC_SECONDS)
    assert at_seam.dtbc_fp32.tobytes() == _stepped_to(
        spec, LBC_SECONDS).dtbc_fp32.tobytes()
    assert at_seam.lbc_reset_due()
    # ...and once the integrator does its top-of-step work, one dt.
    at_seam.mark_force()
    at_seam.prepare_step()
    assert float(at_seam.dtbc_fp32) == pytest.approx(DT)

    past_seam = _fresh(spec)
    jump_clock(past_seam, LBC_SECONDS + DT, DT)
    assert float(past_seam.dtbc_fp32) == pytest.approx(DT)

    # Two legs into the interval the accumulator counts from the seam,
    # not from the start of the run.
    inside = _fresh(spec)
    jump_clock(inside, LBC_SECONDS + 3600.0, DT)
    assert float(inside.dtbc_fp32) == pytest.approx(3600.0)


def test_without_an_external_boundary_stream_nothing_resets():
    spec = _spec(lbc=False)
    clock = _fresh(spec)
    jump_clock(clock, 7200.0, DT)
    assert float(clock.dtbc_fp32) == pytest.approx(7200.0)
    assert clock.dtbc_fp32.tobytes() == _stepped_to(
        spec, 7200.0).dtbc_fp32.tobytes()


def test_a_leg_boundary_off_the_step_lattice_still_lands_on_it():
    """Rounding is to whole steps: a clock is never left between ticks."""
    spec = _spec()
    clock = _fresh(spec)
    jump_clock(clock, 3600.0 + 0.4 * DT, DT)
    assert clock.ticks % spec.step_ticks == 0
    assert clock.elapsed_seconds == 3600.0


# ---------------------------------------------------------------------------
# the fine nest's command-line surface
#
# These refusals fire on the parsed arguments alone, before the driver
# touches CuPy or the prepared authority, so they are reachable from a
# CPU test and a mis-stated nest costs a second rather than a leg.
# ---------------------------------------------------------------------------

_REQUIRED_ARGS = [
    "--prepared-root", "prepared",
    "--proof-sha256", "0" * 64,
    "--source-manifest-sha256", "1" * 64,
    "--prepared-content-sha256", "2" * 64,
    "--physics-profile", "wsm6-ysu-mm5-noah-no-radiation-v1",
    "--run-seconds", "21600",
    "--history-interval-seconds", "900",
    "--out", "out",
]


def _driver_error(monkeypatch, capsys, extra):
    """Run the driver's argument parsing only and return its refusal."""
    import sys

    from tools.da_cycle_prepared import main

    monkeypatch.setattr(sys, "argv", ["da_cycle_prepared"]
                        + _REQUIRED_ARGS + extra)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    return capsys.readouterr().err


def test_a_nest_without_free_legs_is_refused(monkeypatch, capsys):
    """The nest exists to cover the forecast that runs past the obs."""
    message = _driver_error(
        monkeypatch, capsys, ["--nest-half-width-km", "60"])
    assert "--nest-* needs --free-legs" in message


def test_nest_members_cannot_exceed_the_parent_ensemble(monkeypatch, capsys):
    message = _driver_error(monkeypatch, capsys, [
        "--nest-half-width-km", "60", "--free-legs", "6",
        "--members", "10", "--nest-members", "11"])
    assert "exceeds --members" in message


def test_nest_members_without_a_nest_is_refused(monkeypatch, capsys):
    message = _driver_error(monkeypatch, capsys, [
        "--free-legs", "6", "--nest-members", "2"])
    assert "without a nest extent" in message


# ---------------------------------------------------------------------
# the background-source surface, at the driver's own front door
# ---------------------------------------------------------------------

def test_the_driver_offers_the_background_roster_and_defaults_to_gfs(
        monkeypatch, capsys):
    """Selecting nothing is selecting GFS, and the roster is the registry.

    The prepared root already IS one source's case; this flag is the
    caller's statement of which, and the front door refuses a
    disagreement.  What matters here is that the default is unchanged,
    so an existing invocation keeps its existing meaning.
    """

    import sys

    from gpuwm.da import background
    from tools import da_cycle_prepared

    monkeypatch.setattr(sys, "argv", ["da_cycle_prepared", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        da_cycle_prepared.main()
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for name in background.BACKGROUND_SOURCES:
        assert name in text
    assert background.DEFAULT_BACKGROUND_SOURCE == "gfs"


def test_the_driver_refuses_a_source_it_has_no_background_registry_for(
        monkeypatch, capsys):
    import sys

    from tools import da_cycle_prepared

    monkeypatch.setattr(
        sys, "argv", ["da_cycle_prepared", "--source", "20crv3"])
    with pytest.raises(SystemExit) as exit_info:
        da_cycle_prepared.main()
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
