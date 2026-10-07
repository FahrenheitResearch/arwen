"""The cycle driver's own ``--radar-tten`` lines, run through its door.

``tools/da_cycle_prepared.py`` forces each member on an observed leg whose
analysis is applied: the leg's reflectivity is read once
(``reflectivity_from_document``), each member builds one slot from its own
state at the start of the leg (``background_from_state`` then
``build_tendency``), the slot lasts the leg (``leg_length(leg) / 60``
minutes), the control is never forced, a verification-only leg and a free
leg are never forced, the forcing is detached before the restart set, and a
forcing nobody read stops the run.  These cells run the driver's ``cycle``
over observed legs with the harness of ``tests/test_da_cycle_memory.py``
and read the report the driver writes.

On the CPU legs the radar heating module is stood in by recording fakes, so
what is checked is the driver's sequence; the same door with the real
module, a real state and the real microphysics read is in
``tests/test_radar_tten_cycle_gpu.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEG_SECONDS = 60.0


def _column_document(shape):
    """45 dBZ through the middle third of the levels in a 5 x 5 block of
    columns at the centre, observed clear air everywhere else."""
    nz, ny, nx = shape
    z = np.full(shape, -10.0, np.float32)
    echo = np.zeros(shape, np.int8)
    j0, i0 = ny // 2 - 2, nx // 2 - 2
    echo[nz // 5: 2 * nz // 3, j0:j0 + 5, i0:i0 + 5] = 1
    z[echo == 1] = 45.0
    # "dims" as gpuwm.obs.radar_grid writes it: the driver's control Vr
    # innovation reads the grid shape from it before the radar loop
    # (control_vr_innovations, since 30c53c5f7), radars or none.
    return {"dims": {"level": nz, "south_north": ny, "west_east": nx},
            "variables": {"z_obs": z, "z_mask": echo,
                          "z0_mask": (1 - echo).astype(np.int8)},
            "clear_air_source": "finite_below_floor", "radars": []}


def observed_drive(monkeypatch, tmp_path, *, obs_legs, free_legs, shape,
                   experiment=None, **drive_kwargs):
    """Run ``cycle`` over ``obs_legs`` observed legs then ``free_legs``
    free ones, with ``--radar-tten``.

    The analysis itself is stood in: the solve returns zero increments, so
    the legs after it run from the same background.  What the stand-ins
    leave untouched is every line of the driver between reading the leg's
    document and writing the leg's record.  Returns the report.
    """
    sys.path.insert(0, str(ROOT / "tests"))
    from test_da_cycle_memory import _drive

    import gpuwm.da.obs_radar as obs_radar_module
    import gpuwm.da.radar_assimilation as assimilation_module
    import gpuwm.ensemble.increments as increments_module
    import gpuwm.obs.target_grid as target_grid_module
    from gpuwm.da import treatment as treatment_module
    from tools import da_cycle_prepared as driver

    documents = {}
    argv = []
    for leg in range(obs_legs):
        path = tmp_path / f"obs_leg{leg}.npz"
        path.write_bytes(b"document")
        documents[path.name] = _column_document(shape)
        grid = tmp_path / f"grid_leg{leg}.nc"
        grid.write_bytes(b"grid")
        argv += ["--obs", str(path), "--grid-wrfout", str(grid)]

    monkeypatch.setattr(
        target_grid_module.TargetGrid, "from_wrfout",
        classmethod(lambda cls, path: SimpleNamespace(path=Path(path))))
    monkeypatch.setattr(obs_radar_module, "read_document",
                        lambda path, expected_grid=None:
                        documents[Path(path).name])
    monkeypatch.setattr(assimilation_module, "analysis_device_price",
                        lambda *a, **k: None)

    def solve(checkpoints, obs_path, grid, cfg_da, **_):
        zeros = {index: {"u": np.zeros(4, np.float32),
                         "v": np.zeros(4, np.float32)}
                 for index in checkpoints}
        return zeros, {
            "innovations": {}, "filter": {"active_points": 0},
            "velocity_thinning": None, "reflectivity_thinning": None,
            "moment_policy": None, "positivity": None,
            "observations": {}, "extra_observations": None,
            "cwp_observations": None, "cwp_thinning": None,
            "cwp_error_inflation": None,
            "cwp_localization_horizontal_m": None,
            "cwp_localization_vertical_m": None}

    monkeypatch.setattr(assimilation_module, "assimilate_radar_grid", solve)
    monkeypatch.setattr(assimilation_module, "grid_rotation",
                        lambda grid: None)
    monkeypatch.setattr(assimilation_module, "member_earth_winds",
                        lambda state, rotation, where=None:
                        (None, None, None))
    monkeypatch.setattr(treatment_module, "cycle_record",
                        lambda *a, **k: {})
    monkeypatch.setattr(driver, "keep_moment_pairs",
                        lambda state, increment, mp_physics:
                        (increment, {"conditioned": False}))
    monkeypatch.setattr(increments_module, "apply_increments",
                        lambda state, increment, mp_physics=None:
                        {"field_count": len(increment)})

    _events, report = _drive(
        monkeypatch, tmp_path, legs=free_legs,
        # The stub states carry no prognostic arrays for additive
        # inflation to draw on; these tests are about the radar forcing.
        extra_argv=(*argv, "--no-hotstart", "--radar-tten",
                    "--additive-inflation", "0", "--echo-noise", "0"),
        experiment=experiment, **drive_kwargs)
    return json.loads(report.read_text(encoding="utf-8"))


class _Recorder:
    """Stands in for gpuwm.da.radar_tten on the CPU legs, keeping the
    real constants and recording every call the driver makes."""

    def __init__(self, monkeypatch, *, read_on_execute=True):
        from gpuwm.da import radar_tten

        self.calls: list = []
        self.read_on_execute = read_on_execute
        recorder = self

        def reflectivity_from_document(document):
            recorder.calls.append(("reflectivity", id(document)))
            return ("ref", id(document)), {"echo_points": int(np.count_nonzero(
                document["variables"]["z_mask"]))}

        def background_from_state(state):
            recorder.calls.append(("background", id(state)))
            return {"theta": state}

        def build_tendency(ref, **background):
            recorder.calls.append(("build", id(background["theta"])))
            return ("slot", id(background["theta"])), {"points_heated": 7}

        class Forcing:
            def __init__(self, slots, slot_minutes, *, receipts=(),
                         provenance=None, mp_tend_lim=None):
                self.slots, self.slot_minutes = list(slots), list(slot_minutes)
                self.receipts, self.provenance = list(receipts), provenance
                self.mp_tend_lim = mp_tend_lim
                self.calls = 0

            def receipt(self):
                return {"slot_minutes": self.slot_minutes,
                        "slots": self.receipts,
                        "provenance": self.provenance,
                        "calls_by_slot": [self.calls],
                        "calls_skipped_no_mp_heating": 0,
                        "mp_tend_lim_k_per_s": self.mp_tend_lim}

        def attach(state, forcing, cfg=None):
            recorder.calls.append(("attach", id(state)))
            setattr(state, radar_tten.STATE_ATTRIBUTE, forcing)

        def detach(state):
            recorder.calls.append(("detach", id(state)))
            return vars(state).pop(radar_tten.STATE_ATTRIBUTE, None)

        for name, value in (
                ("reflectivity_from_document", reflectivity_from_document),
                ("background_from_state", background_from_state),
                ("build_tendency", build_tendency),
                ("RadarTtenForcing", Forcing), ("attach", attach),
                ("detach", detach)):
            monkeypatch.setattr(radar_tten, name, value)

    def execute(self, model):
        """The integration: four microphysics calls that read the forcing,
        as microphysics.apply does, then the clock."""
        from gpuwm.da import radar_tten

        state = model.root.state
        forcing = getattr(state, radar_tten.STATE_ATTRIBUTE, None)
        self.calls.append(("execute", id(state), forcing is not None))
        if forcing is not None and self.read_on_execute:
            forcing.calls += 4
        model.root.clock.ticks += int(LEG_SECONDS)


def _trajectories(record):
    return {name: entry.get("radar_tten")
            for name, entry in record["trajectories"].items()}


def test_members_are_forced_on_applied_legs_and_never_on_a_scoring_leg(
        monkeypatch, tmp_path):
    """Two observed legs, no free legs: leg 0's analysis is applied and
    its members are forced; leg 1's file is the verification and nothing
    is forced with it."""
    recorder = _Recorder(monkeypatch)
    report = observed_drive(monkeypatch, tmp_path, obs_legs=2, free_legs=0,
                            shape=(49, 132, 132), execute=recorder.execute)

    legs = report["legs"]
    assert [leg["leg_in_run"] for leg in legs] == [0, 1]
    first, scoring = legs
    assert first["radar_tten_observations"] == {
        "echo_points": 25 * (2 * 49 // 3 - 49 // 5)}
    forced = _trajectories(first)
    assert forced["control"] is None
    for member in ("0", "1"):
        record = forced[member]
        assert record["slot_minutes"] == [LEG_SECONDS / 60.0]
        assert record["calls_by_slot"] == [4]
        assert record["mp_tend_lim_k_per_s"] == 0.07
        assert record["slots"] == [{"points_heated": 7}]
        assert record["provenance"]["observations"].endswith(
            "obs_leg0.npz")
    assert scoring["radar_tten_observations"].startswith(
        "none: this leg's observations verify the run")
    assert all(value is None for value in _trajectories(scoring).values())

    # The sequence: one read of the leg's file, then per member its own
    # background, its own build, attach, the integration, detach; the
    # control integrates unforced.
    kinds = [call[0] for call in recorder.calls]
    assert kinds.count("reflectivity") == 1
    assert kinds.count("build") == kinds.count("background") == 2
    assert kinds.count("attach") == kinds.count("detach") == 2
    executes = [call for call in recorder.calls if call[0] == "execute"]
    assert [forced for _k, _s, forced in executes] == [
        False, True, True, False, False, False]
    built_from = [call[1] for call in recorder.calls if call[0] == "build"]
    forced_states = [call[1] for call in executes if call[2]]
    assert built_from == forced_states       # each member's own state
    for state_id in forced_states:
        order = [call[0] for call in recorder.calls
                 if len(call) > 1 and call[1] == state_id]
        assert order == ["background", "build", "attach", "execute",
                         "detach"]

    run = report["radar_tten"]
    assert run["mp_tend_lim_k_per_s"] == 0.07
    assert "hrrrdas_wrf.nl:101" in run["declared_divergence"]
    assert "hrrr_wrfpre.nl:109" in run["declared_divergence"]


def test_a_free_leg_is_never_forced(monkeypatch, tmp_path):
    """One observed leg then one free leg: with free legs every observed
    leg's analysis is applied, so leg 0 is forced; the free leg has no
    observations and runs unforced."""
    recorder = _Recorder(monkeypatch)
    report = observed_drive(monkeypatch, tmp_path, obs_legs=1, free_legs=1,
                            shape=(49, 132, 132), execute=recorder.execute)
    first, free = report["legs"]
    assert _trajectories(first)["0"]["calls_by_slot"] == [4]
    assert free["radar_tten_observations"] == "none: a free leg"
    assert all(value is None for value in _trajectories(free).values())
    executes = [call[2] for call in recorder.calls if call[0] == "execute"]
    assert executes == [False, True, True, False, False, False]


def test_a_forcing_no_microphysics_call_read_stops_the_run(monkeypatch,
                                                           tmp_path):
    """The receipt guard: a member whose integration never read its
    forcing ran unforced, and its record would say forced."""
    recorder = _Recorder(monkeypatch, read_on_execute=False)
    with pytest.raises(RuntimeError,
                       match="no microphysics call read it"):
        observed_drive(monkeypatch, tmp_path, obs_legs=2, free_legs=0,
                       shape=(49, 132, 132), execute=recorder.execute)
    # detached on the way out, even though the leg failed after it
    kinds = [call[0] for call in recorder.calls]
    assert kinds.count("attach") == kinds.count("detach") == 1


def test_the_help_states_the_divergence_and_the_clamp(monkeypatch, capsys):
    from tools import da_cycle_prepared as driver

    monkeypatch.setattr(sys, "argv", ["da_cycle_prepared", "--help"])
    with pytest.raises(SystemExit):
        driver.cycle([])
    text = " ".join(capsys.readouterr().out.split())
    assert "the reverse of HRRR's arrangement" in text
    assert "parm/hrrrdas/hrrrdas_wrf.nl:101" in text
    assert "used twice" in text
    assert "mp_tend_lim = 0.07" in text
