"""The cycle driver's radar heating sequence on a real leg, on the card.

``tools/da_cycle_prepared.py --radar-tten`` builds a member's slot from the
leg's observation document and the member's state, attaches it, runs the
leg through ``execute_experiment`` and detaches it before the leg's restart
set is written.  These cells run that sequence on the hand-built tree the
join tests use (``tests/test_da_cycle_join_gpu.py``: a 33 x 33 x 49 parent
with WSM6, land and radiation, four 15 s steps): the forcing is read on
every step, it heats the observed column, and the restart set writes only
once it is detached -- with it attached the restart owner refuses the
unclassified attribute, which is why the driver detaches in a ``finally``.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_gpu
from test_da_cycle_join_gpu import (LEG_SECONDS, PARENT_DT, _assemble,
                                    _checkpoint, _experiment_to, _run,
                                    _wire_parent)

pytestmark = [pytest.mark.gpu, requires_gpu]


def _column_document(shape):
    """45 dBZ through the middle third of the levels in a 5 x 5 block of
    columns at the centre, observed clear air everywhere else."""
    nz, ny, nx = shape
    z = np.full(shape, -10.0, np.float32)
    echo = np.zeros(shape, np.int8)
    j0, i0 = ny // 2 - 2, nx // 2 - 2
    echo[nz // 5: 2 * nz // 3, j0:j0 + 5, i0:i0 + 5] = 1
    z[echo == 1] = 45.0
    return {"variables": {"z_obs": z, "z_mask": echo,
                          "z0_mask": (1 - echo).astype(np.int8)},
            "clear_air_source": "finite_below_floor"}, echo.astype(bool)


def _forced_leg(tmp_path):
    from gpuwm.da import radar_tten

    exp = _experiment_to(LEG_SECONDS)
    wired = _wire_parent(exp)
    state = wired["root"].state
    model = _assemble(wired)
    document, echo = _column_document(tuple(state.thp.shape))
    forcing = radar_tten.build_forcing_from_documents(
        state, [document], [LEG_SECONDS / 60.0])
    radar_tten.attach(state, forcing, exp.root.run)
    return exp, model, state, forcing, echo


def test_a_forced_leg_reads_the_forcing_every_step_and_heats_the_echo(
        tmp_path):
    import cupy as cp
    from gpuwm.da import radar_tten

    exp = _experiment_to(LEG_SECONDS)
    wired = _wire_parent(exp)
    plain_model = _assemble(wired)
    _run(plain_model)
    thp_plain = cp.asnumpy(wired["root"].state.thp).astype(np.float64)
    del plain_model, wired

    exp, model, state, forcing, echo = _forced_leg(tmp_path)
    try:
        _run(model)
    finally:
        radar_tten.detach(state)
    record = forcing.receipt()
    assert record["calls_by_slot"] == [int(LEG_SECONDS / PARENT_DT)]
    assert record["elapsed_seconds"] == LEG_SECONDS
    slot = record["slots"][0]
    assert slot["points_heated"] > 0
    assert slot["points_zero_tendency"] > 0
    assert slot["max_tendency_k_per_s"] > 0.0
    warmer = cp.asnumpy(state.thp).astype(np.float64) - thp_plain
    assert warmer[echo].max() > 0.1          # K, of at most 0.6 K applied
    root = _checkpoint(model, tmp_path / "forced", LEG_SECONDS)
    assert root.is_file()


def test_a_set_written_with_the_forcing_attached_is_refused(tmp_path):
    from gpuwm.da import radar_tten
    from gpuwm.io.restart import RestartManifestError

    exp, model, state, forcing, _echo = _forced_leg(tmp_path)
    try:
        _run(model)
        with pytest.raises(RestartManifestError,
                           match=radar_tten.STATE_ATTRIBUTE):
            _checkpoint(model, tmp_path / "attached", LEG_SECONDS)
    finally:
        radar_tten.detach(state)


def test_the_cycle_driver_forces_a_real_member_through_its_door(
        monkeypatch, tmp_path):
    """``tools/da_cycle_prepared.py --radar-tten`` end to end on the card:
    the driver reads a real leg document into NOAA's convention, builds
    each member's slot from that member's real state, attaches it, and
    the integration's microphysics calls read it, under HRRR's clamp; the
    control and the scoring leg run unforced.  The integration is four
    real ``microphysics.apply`` calls (Kessler) on a 40 x 40 x 30 balanced
    cloudy state; the analysis is stood in (zero increments), as in
    ``tests/test_radar_tten_cycle_driver.py``.
    """
    import cupy as cp
    from gpuwm.core import microphysics
    from gpuwm.da import radar_tten
    from test_da_nested_forecast import _nowcast_experiment, _nowcast_run
    from test_radar_tten_cycle_driver import observed_drive
    from test_radar_tten_forcing import _moist_state

    nx = ny = 40
    nz = 30
    exp = _nowcast_experiment(_nowcast_run(nx=nx, ny=ny, nz=nz))
    _probe, kessler = _moist_state(nx=nx, ny=ny, nz=nz, mp=1, dt=15.0)
    del _probe
    seen = []

    def execute(model):
        state = model.root.state
        forcing = getattr(state, radar_tten.STATE_ATTRIBUTE, None)
        before = cp.asnumpy(state.thp).copy()
        for _ in range(4):
            microphysics.apply(state, kessler, kessler.dt)
        seen.append((forcing is not None,
                     float(np.abs(cp.asnumpy(state.thp) - before).max())))
        model.root.clock.ticks += 60

    report = observed_drive(
        monkeypatch, tmp_path, obs_legs=2, free_legs=0, shape=(nz, ny, nx),
        experiment=exp, fake_device=False,
        make_state=lambda: _moist_state(nx=nx, ny=ny, nz=nz, mp=1,
                                        dt=15.0)[0],
        execute=execute,
        simulated=lambda state, cfg: cp.zeros((nz, ny, nx), cp.float32))

    first, scoring = report["legs"]
    assert first["radar_tten_observations"]["clear_air"] == "read"
    assert first["radar_tten_observations"]["echo_points"] == 25 * (
        2 * nz // 3 - nz // 5)
    records = {name: entry.get("radar_tten")
               for name, entry in first["trajectories"].items()}
    assert records["control"] is None
    for member in ("0", "1"):
        record = records[member]
        assert record["slot_minutes"] == [1.0]
        assert record["calls_by_slot"] == [4]
        assert record["elapsed_seconds"] == 60.0
        assert record["mp_tend_lim_k_per_s"] == 0.07
        assert record["case_mp_tend_lim_k_per_s"] == kessler.mp_tend_lim
        slot = record["slots"][0]
        assert slot["points_heated"] > 0
        assert slot["points_zero_tendency"] > 0
        assert slot["max_tendency_k_per_s"] > 0.0
    assert all(entry.get("radar_tten") is None
               for entry in scoring["trajectories"].values())
    assert [forced for forced, _change in seen] == [
        False, True, True, False, False, False]
    assert all(change > 0.0 for _forced, change in seen)
    assert report["radar_tten"]["mp_tend_lim_k_per_s"] == 0.07
