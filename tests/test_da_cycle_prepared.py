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


# ---------------------------------------------------------------------
# plan review for the DA door: the refusal has to land before the legs
# ---------------------------------------------------------------------
#
# Audit R-051 moved the radar operator's scheme refusals into
# ``RadarAssimilationConfig.__post_init__``, which means they fire
# wherever that configuration is first BUILT.  In this driver that used
# to be inside ``for leg in range(legs)``, at the first analysis seam --
# so a cycle whose scheme has no H(x) burned leg 0's whole ensemble
# integration before being told.  These cells pin the placement, not the
# wording: the driver builds the configuration once above the leg loop,
# and the leg's own configuration comes from that same function.


def _plan_args(**updates):
    """The knobs ``plan_radar_assimilation`` reads, at driver defaults."""
    from types import SimpleNamespace

    values = dict(
        horizontal_loc_m=12000.0, vertical_loc_m=3000.0, rtps_alpha=0.9,
        relaxation="rtps", thin_cells=1, err_inflation=1.0, z_thin_cells=1,
        z_err_inflation=1.0, z0_thin_cells=4, z0_err_inflation=1.0,
        cwp_thin_cells=1, cwp_err_inflation=1.0,
        cwp_horizontal_loc_m=None, cwp_vertical_loc_m=None,
        positivity_policy="clip", solve_device="host",
        memory_budget_mib=512.0, hydrometeors=True,
        reflectivity_analysis=False, clear_air_analysis=False,
        goes_cwp=[])
    values.update(updates)
    return SimpleNamespace(**values)


def _routed_and_fused_schemes():
    from gpuwm.physics_registry import consumer_rows_by_selector

    rows = consumer_rows_by_selector("microphysics", "radar_da")
    routed = sorted(mp for mp, row in rows.items()
                    if row.get("reflectivity_route")
                    in ("operator", "scheme-diagnostic"))
    fused = sorted(mp for mp, row in rows.items()
                   if row.get("reflectivity_route") == "native-not-separable")
    return routed, fused


def test_the_plan_probe_refuses_a_scheme_the_radar_operator_cannot_simulate():
    """The refusal a cycle used to get after leg 0, before leg 0.

    Same function the leg builds its configuration with, so what is
    checked here is what will run.
    """
    from gpuwm.da.radar_assimilation import RadarAssimilationError
    from tools.da_cycle_prepared import (plan_radar_assimilation,
                                         planned_analysis_fields)

    routed, fused = _routed_and_fused_schemes()
    assert routed and fused, (routed, fused)

    args = _plan_args(reflectivity_analysis=True)
    with pytest.raises(RadarAssimilationError) as refusal:
        plan_radar_assimilation(
            args, fused[0], cwp=False,
            analysis_fields=planned_analysis_fields(args, fused[0]))
    message = str(refusal.value)
    # A refusal names its way out, and the way out has to work.
    assert "reflectivity=False" in message, message
    kept = plan_radar_assimilation(
        _plan_args(reflectivity_analysis=False), fused[0], cwp=False,
        analysis_fields=planned_analysis_fields(_plan_args(), fused[0]))
    assert kept.reflectivity is False and kept.velocity

    ok = plan_radar_assimilation(
        args, routed[0], cwp=False,
        analysis_fields=planned_analysis_fields(args, routed[0]))
    assert ok.mp_physics == routed[0] and ok.reflectivity


def test_the_plan_probe_asks_for_the_fields_the_legs_will_analyse():
    """Plan time is the leg's field set before spread narrows it.

    The leg drops whole species the ensemble is constant in, so its set
    is a subset of this one.  That direction is what makes the probe
    safe: it cannot refuse a cycle the legs would have run.
    """
    from gpuwm.da import moments
    from tools.da_cycle_prepared import planned_analysis_fields

    routed, _ = _routed_and_fused_schemes()
    mp = routed[0]
    assert planned_analysis_fields(_plan_args(hydrometeors=False), mp) == (
        "u", "v")
    assert planned_analysis_fields(_plan_args(), mp) == tuple(
        moments.analysis_fields(int(mp)))


def _main_body():
    import ast
    import inspect

    from tools import da_cycle_prepared

    tree = ast.parse(inspect.getsource(da_cycle_prepared))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("tools/da_cycle_prepared.py has no main()")


def test_the_driver_plans_the_analysis_before_it_integrates_a_leg():
    """Deleting the probe, or sinking it into the loop, fails here.

    Structural on purpose: the defect is a PLACEMENT, and a cell that
    only called the function would still pass with the call sitting
    where it was -- inside ``for leg in range(legs)``, one ensemble
    integration too late.
    """
    import ast

    main = _main_body()
    probes = [stmt.lineno for stmt in main.body
              if isinstance(stmt, ast.Expr)
              and isinstance(stmt.value, ast.Call)
              and getattr(stmt.value.func, "id", None)
              == "plan_radar_assimilation"]
    assert len(probes) == 1, (
        "the DA plan review must be called exactly once at the top level "
        f"of main(), found {len(probes)}")
    loops = [stmt.lineno for stmt in main.body
             if isinstance(stmt, ast.For)
             and getattr(stmt.target, "id", None) == "leg"]
    assert len(loops) == 1, loops
    assert probes[0] < loops[0], (
        "the analysis configuration is built inside the leg loop, so its "
        "refusals arrive with an ensemble integration already spent")


def test_every_analysis_configuration_in_the_driver_comes_from_one_function():
    """No second construction, so the reviewed plan is the one that runs."""
    import ast

    main = _main_body()
    direct = [node.lineno for node in ast.walk(main)
              if isinstance(node, ast.Call)
              and getattr(node.func, "id", None) == "RadarAssimilationConfig"]
    assert not direct, (
        "RadarAssimilationConfig is constructed directly in main() at "
        f"{direct}; every construction goes through plan_radar_assimilation")
    through = [node.lineno for node in ast.walk(main)
               if isinstance(node, ast.Call)
               and getattr(node.func, "id", None) == "plan_radar_assimilation"]
    assert len(through) == 2, (
        "main() should build the plan-time configuration and the leg's own "
        f"through the same function, found {len(through)} call(s)")
