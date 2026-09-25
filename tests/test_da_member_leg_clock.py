"""The member worker's placed clock against one that stepped there.

``tools/da_member_leg.py`` still joins a member's legs the way the
cycling driver used to: a host copy of the serialised atmosphere into a
freshly wired model, and a clock placed at the leg boundary by
:func:`tools.da_member_leg.jump_clock`.  The driver itself now joins its
legs through the restart owner and the worker takes that join when it
is ported to the driver's flags (``docs/da-ensemble-parallel.md``);
until then the worker's placement has to be indistinguishable from a
clock that stepped to the boundary, bit for bit in the FP32 boundary
accumulator, and this is where that is compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.clock import DomainClock, DomainTicks
from tools.da_member_leg import jump_clock

TICK_DEN = 1
DT = 60.0
LBC_SECONDS = 10800.0          # a 3-hourly boundary interval


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
    the worker's replay has to reproduce.
    """
    clock = _fresh(spec)
    while clock.elapsed_seconds < seconds:
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        clock.advance()
    return clock


@pytest.mark.parametrize("hours", [1, 2, 3, 4, 7])
def test_the_workers_placed_clock_is_indistinguishable_from_a_stepped_one(
        hours):
    spec = _spec()
    seconds = hours * 3600.0
    stepped = _stepped_to(spec, seconds)
    placed = _fresh(spec)
    jump_clock(placed, seconds, DT)

    assert placed.ticks == stepped.ticks
    assert placed.step_count == stepped.step_count
    assert placed.elapsed_seconds == seconds
    # The FP32 boundary accumulator, bit for bit: the field a closed-form
    # steps*dt gets subtly wrong.
    assert placed.dtbc_fp32.tobytes() == stepped.dtbc_fp32.tobytes(), (
        f"dtbc {placed.dtbc_fp32!r} != stepped {stepped.dtbc_fp32!r}")


def test_a_leg_landing_on_a_seam_carries_the_interval_not_a_zero():
    """The reset is the integrator's top-of-step work, not the worker's."""
    spec = _spec()
    at_seam = _fresh(spec)
    jump_clock(at_seam, LBC_SECONDS, DT)
    assert float(at_seam.dtbc_fp32) == pytest.approx(LBC_SECONDS)
    assert at_seam.dtbc_fp32.tobytes() == _stepped_to(
        spec, LBC_SECONDS).dtbc_fp32.tobytes()
    assert at_seam.lbc_reset_due()
    at_seam.mark_force()
    at_seam.prepare_step()
    assert float(at_seam.dtbc_fp32) == pytest.approx(DT)

    past_seam = _fresh(spec)
    jump_clock(past_seam, LBC_SECONDS + DT, DT)
    assert float(past_seam.dtbc_fp32) == pytest.approx(DT)

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
