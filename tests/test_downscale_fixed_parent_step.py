"""A child of a fixed-step parent takes a step every clock it keeps lands on.

WHAT BREAKAGE THIS PINS (gate law).  Downscaling from a 9 km or 12 km
parent, the review refused with "run_seconds/dt must be a positive integer,
got 4692.737430167597: the child integrates in whole steps of dt = 21.48 s
...": the derived child took the parent's 64.44 s step over the ratio, which
a 28 h run is not a whole number of, and the --point route (the desktop's
downscale review) has no way to set a step.  The adaptive parent already took
its step down to the largest one every clock lands on; a fixed parent now
does the same through the same helper (gpuwm.downscale.child_clock_step),
never longer than the parent's step over the ratio, and exactly that step
whenever it already landed on every clock.
"""

from fractions import Fraction

import pytest

from conftest import requires_netcdf_bridge
from gpuwm.config import RunConfig
from gpuwm.downscale import (
    _derive_child_run_config, child_clock_step, derived_child_config_path)
from gpuwm.offline_child import (
    OfflineChildContractError, resolve_child_run_config)
from gpuwm.offline_child_run import _child_boundary_clock, child_cadence

_HOUR = 3600.0
_HEALTH = 60.0


def _parent(dx_m: float, dt: float, **extra) -> dict:
    return {
        "nx": 386, "ny": 308, "nz": 4, "dx": dx_m, "dy": dx_m,
        "ztop": 20000.0, "dt": dt, "run_seconds": 28 * _HOUR,
        "output_interval_s": _HOUR, "hybrid_opt": 2, "etac": 0.2,
        "hypsometric_opt": 2, "moist": True, "mp_physics": 8,
        "specified": True, "nested": False, "terrain_opt": 1, "map_proj": 1,
        "grid_id": 1, "time_step_sound": 4, "spec_bdy_width": 5,
        "spec_zone": 1, "relax_zone": 4, **extra}


def _child(parent_config, *, ratio=3, run_seconds=28 * _HOUR,
           output_interval_s=_HOUR, frame_seconds=_HOUR, health=_HEALTH):
    return _derive_child_run_config(
        parent_config,
        parent={"nx": parent_config["nx"], "ny": parent_config["ny"],
                "dx": parent_config["dx"], "dy": parent_config["dy"]},
        ratio=ratio, child_nx=120, child_ny=120, run_seconds=run_seconds,
        output_interval_s=output_interval_s, centre_lat=40.0,
        clock_seconds=(health, frame_seconds))


def _lands_on_every_clock(merged, *, frame_seconds=_HOUR, health=_HEALTH):
    """The review's clock check and the runner's boundary seam, both."""
    cfg = RunConfig(**merged)
    clock = child_cadence(cfg, health_interval_seconds=health)
    _child_boundary_clock(cfg, lbc_interval_seconds=frame_seconds,
                          steps=clock.steps, output_steps=clock.output_steps)
    step = Fraction(cfg.dt).limit_denominator(10 ** 6)
    for seconds in (cfg.run_seconds, cfg.output_interval_s, health,
                    frame_seconds):
        assert (Fraction(seconds) / step).denominator == 1
    return clock


def test_the_reported_twelve_kilometre_parent_now_reviews():
    """64.44 s over 3 is 21.48 s; 28 h is 4692.74 of those.  The child
    takes 20 s: under 21.48 s, and 28 h, the hour of history, the hour
    between parent frames and the minute of health lines all land on it."""
    merged = _child(_parent(12000.0, 64.44))
    assert merged["dt"] <= 64.44 / 3
    assert merged["dt"] == 20.0
    clock = _lands_on_every_clock(merged)
    assert clock.steps == 5040 and clock.output_steps == 180


def test_the_twelve_kilometre_step_before_the_fix_is_still_refused():
    """A hand-written config that names 21.48 s is the user's own step,
    and the review keeps refusing it in the same sentence."""
    merged = dict(_child(_parent(12000.0, 64.44)), dt=64.44 / 3)
    with pytest.raises(OfflineChildContractError,
                       match="run_seconds/dt must be a positive integer"
                       ) as refused:
        child_cadence(RunConfig(**merged), health_interval_seconds=_HEALTH)
    assert "whole multiple of dt" in str(refused.value)


def test_a_nine_kilometre_parent_on_a_non_integer_step_reviews():
    merged = _child(_parent(9000.0, 48.33))
    assert merged["dt"] <= 48.33 / 3
    assert merged["dt"] == 15.0
    _lands_on_every_clock(merged)


@pytest.mark.parametrize("hours", [1.0, 6.0, 28.0, 30.5])
@pytest.mark.parametrize("parent_dt,dx_m", [(64.44, 12000.0), (48.33, 9000.0),
                                            (54.0, 9000.0), (72.0, 12000.0)])
def test_every_run_length_lands_and_the_step_never_lengthens(
        hours, parent_dt, dx_m):
    merged = _child(_parent(dx_m, parent_dt), run_seconds=hours * _HOUR)
    assert merged["dt"] <= parent_dt / 3
    _lands_on_every_clock(merged)


@pytest.mark.parametrize("parent_dt,ratio", [(60.0, 3), (36.0, 3), (10.0, 3),
                                             (15.0, 1), (30.0, 6)])
def test_a_step_that_already_landed_is_kept_exactly(parent_dt, ratio):
    """No configuration that worked before changes: when the parent's step
    over the ratio lands on every clock, the child's step is that float,
    bit for bit (10 / 3 included)."""
    merged = _child(_parent(9000.0, parent_dt), ratio=ratio,
                    run_seconds=6 * _HOUR)
    assert merged["dt"] == parent_dt / ratio
    assert repr(merged["dt"]) == repr(parent_dt / ratio)
    _lands_on_every_clock(merged)


def test_the_parent_frame_interval_is_a_clock_the_step_lands_on():
    """The boundary seam falls on every parent frame: a 40 s step lands on
    the run, the history and a 120 s health line but not on a 900 s frame
    interval (22.5 steps), which the runner refuses at its first boundary.
    Without the frame interval the step is unchanged."""
    parent = _parent(9000.0, 40.0)
    without_frames = _derive_child_run_config(
        parent, parent={"nx": 386, "ny": 308, "dx": 9000.0, "dy": 9000.0},
        ratio=1, child_nx=120, child_ny=120, run_seconds=2 * _HOUR,
        output_interval_s=_HOUR, clock_seconds=(120.0,))
    assert without_frames["dt"] == 40.0
    with pytest.raises(OfflineChildContractError, match="boundary seam"):
        _child_boundary_clock(RunConfig(**without_frames),
                              lbc_interval_seconds=900.0, steps=180,
                              output_steps=90)
    merged = _child(parent, ratio=1, run_seconds=2 * _HOUR,
                    frame_seconds=900.0, health=120.0)
    assert merged["dt"] == 30.0
    _lands_on_every_clock(merged, frame_seconds=900.0, health=120.0)


def test_the_adaptive_parent_also_lands_on_the_frame_interval():
    """Same helper, same clocks: an adaptive parent's 4 km child is 20 s
    at 5 s per km, which a 1050 s frame interval is 52.5 of; it takes
    15 s, and 20 s again when the frames fall every 900 s."""
    parent = _parent(12000.0, 37.21, use_adaptive_time_step=True)
    merged = _child(parent, run_seconds=_HOUR, output_interval_s=600.0,
                    frame_seconds=1050.0)
    assert merged["dt"] == 15.0
    assert merged["use_adaptive_time_step"] is False
    _lands_on_every_clock(merged, frame_seconds=1050.0)
    assert _child(parent, run_seconds=_HOUR, output_interval_s=600.0,
                  frame_seconds=900.0)["dt"] == 20.0


def test_the_helper_never_exceeds_its_ceiling():
    for ceiling in (21.48, 16.11, 7.3, 0.9, 13.0 / 7):
        for clocks in ((100800.0, 3600.0, 60.0, 3600.0),
                       (3600.0, 900.0, 0.0, 60.0, 900.0),
                       (7.0, None, 21.0)):
            step = child_clock_step(ceiling, clocks)
            assert step <= ceiling
            exact = Fraction(step).limit_denominator(10 ** 6)
            assert float(exact) == step
            for seconds in clocks:
                if seconds:
                    assert (Fraction(seconds) / exact).denominator == 1
    assert child_clock_step(21.48, ()) == 21.48


# A ceiling whose float the runner does not read back as the same step:
# the boundary clock rebuilds dt as the nearest fraction with denominator at
# most a million and refuses unless that fraction's float is dt again.
_NOT_READ_BACK = [
    # 9.6 / 6 = 1.5999999999999999; its fraction 8/5 reads back as 1.6,
    # a hair over the ceiling, so the step is the next division, 600/376.
    (9.6 / 6, Fraction(600, 376)),
    # 33.333333333333336 / 3 = 11.111111111111112; 100/9 reads back as
    # 11.11111111111111, under the ceiling, so that is the step.
    (33.333333333333336 / 3, Fraction(100, 9)),
    # 20 / 3 / 5 = 1.3333333333333335; 4/3 reads back a hair under it.
    (20.0 / 3 / 5, Fraction(4, 3)),
]


@pytest.mark.parametrize("ceiling,expected", _NOT_READ_BACK)
def test_a_ceiling_the_runner_cannot_read_back_is_not_handed_back(
        ceiling, expected):
    """These ceilings divide every clock as fractions, and the helper used
    to hand back the given float, which the runner's boundary clock refused
    at run start ("is not exactly rational within 1e-6").  The step it
    takes now is exact, lands on every clock, and is never over."""
    clocks = (48 * _HOUR, 3 * _HOUR, 0.0, 600.0, 3 * _HOUR)
    step = child_clock_step(ceiling, clocks)
    assert float(Fraction(ceiling).limit_denominator(10 ** 6)) != ceiling
    assert step == float(expected)
    assert step <= ceiling
    exact = Fraction(step).limit_denominator(10 ** 6)
    assert float(exact) == step
    for seconds in clocks:
        if seconds:
            assert (Fraction(seconds) / exact).denominator == 1


def test_a_parent_step_whose_ratio_float_is_not_read_back_runs():
    """The derived child of a 9.6 s parent at ratio 6 reaches the review
    and the runner's boundary clock with a step both accept."""
    parent = _parent(3000.0, 9.6)
    merged = _child(parent, ratio=6, run_seconds=48 * _HOUR,
                    output_interval_s=3 * _HOUR, frame_seconds=3 * _HOUR,
                    health=600.0)
    assert merged["dt"] == float(Fraction(600, 376))
    assert merged["dt"] <= 9.6 / 6
    _lands_on_every_clock(merged, frame_seconds=3 * _HOUR, health=600.0)


# rw_netcdf reads the parent frames the --point route places the child in.
@requires_netcdf_bridge
def test_the_point_route_reviews_a_parent_step_that_does_not_land(
        tmp_path, capsys):
    """The door the desktop's review drives: --point --dry-run over a
    parent whose step does not land on the child's clocks.  It used to
    exit 2 with "run_seconds/dt must be a positive integer"; it now plans
    a child on a step that lands, and says which step it took."""
    from test_downscale_cli import (
        _SURFACE_PARENT_CONFIG, _point_args, _restart_evidence)
    from gpuwm.cli import main as cli_main

    args = _point_args(tmp_path)
    # 3.22 s does not divide the 900 s run: 279.5 steps.
    _restart_evidence(
        tmp_path / "gpuwmrst_d01_1974-04-03_12_00_00.npz",
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1,
             dt=3.22, run_seconds=7200.0, nested=False, specified=False))
    args += ["--dry-run", "--render-products", "none"]
    rc = cli_main(args)
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "must be a positive integer" not in captured.err
    assert "not the parent's 3.22 s over the ratio" in captured.err
    cfg = resolve_child_run_config(derived_child_config_path(
        tmp_path / "child-run", dry_run=True))
    assert cfg.dt <= 3.22
    assert cfg.dt == 60.0 / 19
    clock = child_cadence(cfg, health_interval_seconds=_HEALTH)
    _child_boundary_clock(cfg, lbc_interval_seconds=_HOUR,
                          steps=clock.steps, output_steps=clock.output_steps)


# rw_netcdf reads the parent frames the --point route places the child in.
@requires_netcdf_bridge
def test_the_point_route_keeps_a_step_that_already_landed(tmp_path, capsys):
    from test_downscale_cli import _point_args
    from gpuwm.cli import main as cli_main

    args = _point_args(tmp_path) + ["--dry-run", "--render-products", "none"]
    assert cli_main(args) == 0
    assert "over the ratio" not in capsys.readouterr().err
    cfg = resolve_child_run_config(derived_child_config_path(
        tmp_path / "child-run", dry_run=True))
    assert cfg.dt == 3.0


def test_a_restart_interval_is_one_of_the_clocks():
    merged = _child(_parent(12000.0, 64.44, restart_interval_s=5400.0))
    assert merged["dt"] == 20.0
    cfg = RunConfig(**merged)
    assert child_cadence(cfg).restart_steps == 270
    odd = _child(_parent(12000.0, 64.44, restart_interval_s=1450.0))
    assert odd["dt"] == 10.0
    assert child_cadence(RunConfig(**odd),
                         health_interval_seconds=_HEALTH).restart_steps == 145
