"""Each domain's acoustic substeps follow its own terrain.

A generated 500 m forecast over the central Andes (steepest slope 0.89)
ran four acoustic substeps per step and its surface w passed 200 m/s at
model second 40; six substeps ran the same forecast for the hour.  These
tests pin the derivation that picks the count from the ground and the
routes that apply it; tests/test_steep_terrain_step.py pins the stability
it buys through the production step().
"""
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pytest

from gpuwm.acoustic_adaptation import (
    STABLE_SLOPE_BY_OFFCENTERING, STEEP_TERRAIN_SOUND_STEPS,
    acoustic_receipt, adapt_experiment_acoustics, derive_acoustics,
    readings_from_static, stable_slopes, steepest_slope, SlopeReading)


@dataclass(frozen=True)
class _Run:
    dx: float = 500.0
    dy: float = 500.0
    epssm: float = 0.5
    time_step_sound: int = 4


@dataclass(frozen=True)
class _Domain:
    grid_id: int
    run: _Run


@dataclass(frozen=True)
class _Experiment:
    domains: tuple


@dataclass(frozen=True)
class _AdaptiveRun(_Run):
    use_adaptive_time_step: bool = True
    min_time_step_sound: int = 0


def _reading(slope, label="d01"):
    return SlopeReading(label, float(slope), ("x", 0, 1))


def test_steepest_slope_reads_height_step_over_ground_distance():
    terrain = np.zeros((5, 6))
    terrain[:, 4:] = 450.0            # a 450 m step across the x faces at i = 4
    reading = steepest_slope(terrain, 500.0, 500.0, label="d01")
    assert reading.slope == pytest.approx(0.9)
    assert reading.face[0] == "x" and reading.face[2] == 4
    assert reading.degrees == pytest.approx(41.99, abs=0.01)
    # On the ground the faces are dx/msf apart, so a factor above one
    # steepens the same height step.
    mapped = steepest_slope(terrain, 500.0, 500.0,
                            msfu=np.full((5, 7), 1.1),
                            msfv=np.ones((6, 6)))
    assert mapped.slope == pytest.approx(0.99)


def test_a_slope_across_the_grid_is_read_whole():
    """A plane falling at 45 degrees to the grid shows each face 0.71 of
    its slope; four substeps failed on such a ridge at its full slope."""
    y, x = np.mgrid[0:20, 0:20] * 500.0
    terrain = 0.9 / np.sqrt(2.0) * (x + y)
    reading = steepest_slope(terrain, 500.0, 500.0, label="d01")
    assert reading.slope == pytest.approx(0.9)
    assert np.abs(np.diff(terrain, axis=1)).max() / 500.0 == pytest.approx(
        0.9 / np.sqrt(2.0))
    assert derive_acoustics(1, _Run(), reading).time_step_sound == 6


def test_flat_and_moderate_ground_keeps_the_configured_count():
    # Measured steepest slopes of the real forecasts: the Iowa plains at
    # 500 m, Hawaii at 1 km, the Colorado Front Range at 500 m and the
    # Himalaya foothills at 1 km.
    for slope in (0.023, 0.328, 0.357, 0.651):
        adaptation = derive_acoustics(1, _Run(), _reading(slope))
        assert adaptation.time_step_sound == 4
        assert adaptation.status == "AS_CONFIGURED"


def test_ground_inside_the_margin_runs_six():
    # The Himalaya at 500 m and the Andes at 1 km completed on four, but
    # sit inside the 0.05 margin under the lowest slope four failed on
    # (0.75, a ridge across the grid), so they are given six.
    for slope in (0.714, 0.734):
        adaptation = derive_acoustics(1, _Run(), _reading(slope))
        assert adaptation.time_step_sound == 6
        assert adaptation.status == "ADAPTED"


def test_andes_ground_at_500_m_runs_six_substeps():
    adaptation = derive_acoustics(1, _Run(), _reading(0.891))
    assert adaptation.time_step_sound == STEEP_TERRAIN_SOUND_STEPS == 6
    # Six held this forecast for three hours, but a ridge this steep
    # across the grid broke six after an hour and a half at 250 m, so the
    # run says it is past the measured map rather than claiming margin.
    assert adaptation.status == "BEYOND_MEASURED"
    assert ("runs 6 substeps per step instead of 4"
            in adaptation.beyond_sentence())
    band = derive_acoustics(1, _Run(), _reading(0.8))
    assert band.status == "ADAPTED"
    assert "runs 6 substeps per step instead of 4" in band.sentence()


def test_a_larger_configured_count_is_never_lowered():
    for slope in (0.1, 0.888):
        adaptation = derive_acoustics(1, _Run(time_step_sound=8),
                                      _reading(slope))
        assert adaptation.time_step_sound == 8
        assert not adaptation.adapted


def test_ground_past_every_stable_count_takes_six_and_says_so():
    adaptation = derive_acoustics(1, _Run(), _reading(1.07))
    assert adaptation.time_step_sound == 6
    assert adaptation.status == "BEYOND_MEASURED"
    assert "may still stop" in adaptation.beyond_sentence()


def test_one_line_per_domain_past_the_measured_bound():
    exp = _Experiment((_Domain(1, _Run()),))
    announced, cautioned = [], []
    adapted, _ = adapt_experiment_acoustics(
        exp, {1: _reading(1.07)}, announce=announced.append,
        caution=cautioned.append)
    assert announced == [] and len(cautioned) == 1
    assert "runs 6 substeps per step instead of 4" in cautioned[0]
    assert adapted.domains[0].run.time_step_sound == 6


def test_offcentering_between_rows_takes_the_less_stable_row():
    epssms = [row[0] for row in STABLE_SLOPE_BY_OFFCENTERING]
    assert epssms == sorted(epssms)
    for row in STABLE_SLOPE_BY_OFFCENTERING:
        assert 0.0 < row[1] < row[2]
    assert stable_slopes(0.35) == STABLE_SLOPE_BY_OFFCENTERING[2]
    assert stable_slopes(0.9) == STABLE_SLOPE_BY_OFFCENTERING[-1]
    assert stable_slopes(0.05) == STABLE_SLOPE_BY_OFFCENTERING[0]
    # A nearly centred off-centering needs six on far gentler ground.
    low = derive_acoustics(1, _Run(epssm=0.1), _reading(0.5))
    assert low.time_step_sound == 6


def test_experiment_changes_only_the_domain_on_steep_ground():
    exp = _Experiment((_Domain(1, _Run(dx=1500.0, dy=1500.0)),
                       _Domain(2, _Run())))
    lines = []
    adapted, adaptations = adapt_experiment_acoustics(
        exp, {1: _reading(0.3, "d01"), 2: _reading(0.8, "d02")},
        announce=lines.append)
    assert adapted.domains[0] is exp.domains[0]
    assert adapted.domains[1].run.time_step_sound == 6
    assert [a.status for a in adaptations] == ["AS_CONFIGURED", "ADAPTED"]
    assert len(lines) == 1 and lines[0].startswith("acoustic substeps: d02")
    receipt = acoustic_receipt(adaptations)
    assert [row["time_step_sound"] for row in receipt["domains"]] == [4, 6]
    # Nothing steep: the very same experiment object comes back.
    same, _ = adapt_experiment_acoustics(exp, {1: _reading(0.3)})
    assert same is exp


def test_on_the_adaptive_clock_steep_ground_sets_the_floor_the_clock_keeps():
    """The adaptive clock derives its count from the live step, 4 at every
    short one, so the six this rule chooses are written as the floor that
    clock keeps; the domain on gentler ground keeps WRF's derived count."""
    from fractions import Fraction

    from gpuwm.core.adaptive_clock import adaptive_sound_steps

    exp = _Experiment((_Domain(1, _AdaptiveRun(dx=1500.0, dy=1500.0)),
                       _Domain(2, _AdaptiveRun())))
    lines = []
    adapted, adaptations = adapt_experiment_acoustics(
        exp, {1: _reading(0.3, "d01"), 2: _reading(0.8, "d02")},
        announce=lines.append)
    assert adapted.domains[0] is exp.domains[0]
    run = adapted.domains[1].run
    assert run.min_time_step_sound == 6 and run.time_step_sound == 6
    assert len(lines) == 1
    assert ("d02 runs at least 6 substeps per step on its adaptive clock "
            "instead of the 4 that clock takes at short steps") in lines[0]
    rows = acoustic_receipt(adaptations)["domains"]
    assert rows[1]["min_time_step_sound"] == 6 and rows[1]["adaptive"]
    assert "min_time_step_sound" not in rows[0]
    # The generated 500 m step: WRF's count is 4 there, the floor makes 6.
    assert adaptive_sound_steps(Fraction(5, 2), exp.domains[1].run) == 4
    assert adaptive_sound_steps(Fraction(5, 2), run) == 6


def test_a_count_the_adaptive_clock_would_ignore_still_gets_its_floor():
    """A configured six is not what runs on the adaptive clock, so steep
    ground under it is still given the floor, and a fixed clock with the
    same six is left as it was."""
    run = _AdaptiveRun(time_step_sound=6)
    adaptation = derive_acoustics(1, run, _reading(0.8))
    assert adaptation.configured == 4 and adaptation.adapted
    exp = _Experiment((_Domain(1, run),))
    adapted, _ = adapt_experiment_acoustics(exp, {1: _reading(0.8)})
    assert adapted.domains[0].run.min_time_step_sound == 6
    assert adapted.domains[0].run.time_step_sound == 6
    same, _ = adapt_experiment_acoustics(exp, {1: _reading(0.3)})
    assert same is exp
    fixed = _Experiment((_Domain(1, _Run(time_step_sound=6)),))
    same, _ = adapt_experiment_acoustics(fixed, {1: _reading(0.8)})
    assert same is fixed


def test_the_generated_adaptive_configuration_takes_the_floor_and_validates():
    from gpuwm.config import validate_run_config

    exp = _wizard_experiment([(40, 40)], (), 500.0)
    root = exp.domains[0]
    exp = replace(exp, domains=(replace(root, run=replace(
        root.run, use_adaptive_time_step=True)),))
    adapted, _ = adapt_experiment_acoustics(exp, {1: _reading(0.8)})
    run = adapted.domains[0].run
    assert run.min_time_step_sound == 6
    validate_run_config(run)
    fixed = _wizard_experiment([(40, 40)], (), 500.0)
    adapted, _ = adapt_experiment_acoustics(fixed, {1: _reading(0.8)})
    assert adapted.domains[0].run.time_step_sound == 6
    assert adapted.domains[0].run.min_time_step_sound == 0


def test_readings_come_from_each_domains_static_terrain():
    exp = _Experiment((_Domain(1, _Run()), _Domain(2, _Run())))
    steep = np.zeros((4, 4))
    steep[:, 2:] = 440.0
    readings = readings_from_static(exp, {
        1: {"HGT_M": np.full((4, 4), 250.0)},
        2: {"HGT_M": steep, "MAPFAC_U": np.ones((4, 5)),
            "MAPFAC_V": np.ones((5, 4))}})
    assert readings[1].slope == 0.0
    assert readings[2].slope == pytest.approx(0.88)
    assert readings[2].label == "d02"


def test_wrf_input_tree_door_carries_the_derivation():
    """The met_em and wrfinput doors hand the tree runner inputs they
    assembled themselves; the runner derives the count for them too."""
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics
    from gpuwm.wrfinput_forecast import WrfDomainBundle, WrfTreeInputs

    steep = np.zeros((6, 6))
    steep[:, 3:] = 400.0
    bundles = tuple(
        WrfDomainBundle(grid_id=gid, restored=None,
                        static_fields=MappingProxyType({"HGT_M": terrain}),
                        authority_sha256={}, landuse=None,
                        geog_selection=None)
        for gid, terrain in ((1, np.full((6, 6), 300.0)), (2, steep)))
    exp = _Experiment((_Domain(1, _Run(dx=1500.0, dy=1500.0)),
                       _Domain(2, _Run())))
    inputs = WrfTreeInputs(
        prepared_root=None, experiment_config=None, experiment=exp,
        grids=(None, None), domains=bundles, forcing_hours=(0.0, 1.0),
        boundary_interval_seconds=3600, source_identity={},
        execution_plan={}, authority_sha256={}, artifact_paths={},
        boundaries=None)
    derived = _with_terrain_acoustics(inputs)
    assert [dc.run.time_step_sound for dc in derived.experiment.domains] == [4, 6]
    assert derived.acoustic_substeps["domains"][1]["status"] == "ADAPTED"
    assert _with_terrain_acoustics(derived) is derived


def _wizard_experiment(dims, ratios, root_dx_m):
    from datetime import datetime

    from gpuwm import domain_wizard as wizard

    text = wizard.render_config(
        name="acoustic-route", start_time=datetime(2024, 6, 13, 12),
        hours=1, projection={
            "map_proj": "lambert", "ref_lat": -33.0, "ref_lon": -70.1,
            "truelat1": -23.0, "truelat2": -43.0, "stand_lon": -70.1},
        dims=dims, ratios=ratios, fetch_hints={"source": "gfs"},
        case_data=None, root_dx_m=root_dx_m, history_interval_s=300.0)
    return wizard.experiment_from_text(text, source="acoustic-route.toml")


def test_run_route_reads_each_domains_own_terrain(monkeypatch):
    """`gpuwm run` derives the count before it prices any tile halo: the
    root off the memoized static build, each nest off its own terrain."""
    from pathlib import Path
    from types import SimpleNamespace

    from gpuwm import runtime
    import gpuwm.static.build as build

    exp = _wizard_experiment([(40, 40), (30, 30)], (3,), 1500.0)
    assert [dc.run.time_step_sound for dc in exp.domains] == [4, 4]
    steep = np.zeros((30, 30))
    steep[:, 15:] = 450.0                     # 0.9 at the 500 m nest
    monkeypatch.setattr(
        runtime.GeogSelection, "from_case_data",
        classmethod(lambda cls, data, domain_id=1: SimpleNamespace(
            root=data.geog_root)))
    monkeypatch.setattr(
        runtime, "case_static_fields",
        lambda grid, root, **kwargs: {"HGT_M": np.full((40, 40), 300.0)})
    monkeypatch.setattr(build, "build_terrain",
                        lambda grid, root, selection=None: steep)
    data = SimpleNamespace(geog_root=Path("geog"), static_highres=None)
    derived = runtime._terrain_acoustics_for_case(exp, data)
    assert [dc.run.time_step_sound for dc in derived.domains] == [4, 6]
    # The same experiment, flat everywhere, is handed back untouched.
    monkeypatch.setattr(build, "build_terrain",
                        lambda grid, root, selection=None: np.zeros((30, 30)))
    assert runtime._terrain_acoustics_for_case(exp, data) is exp


@pytest.mark.parametrize("highres", [False, True])
def test_run_route_reads_a_following_nests_corridor(monkeypatch, highres):
    """A following nest that starts on gentle ground but can be moved onto
    steep ground takes the count the steep ground needs before the run
    starts, on `gpuwm run` as on the prepared tree door."""
    from pathlib import Path
    from types import SimpleNamespace

    from gpuwm import runtime
    import gpuwm.static.build as build
    import gpuwm.static.corridor as corridor

    exp = _wizard_experiment([(40, 40), (30, 30)], (3,), 1500.0)
    steep = np.zeros((60, 60))
    steep[:, 30:] = 450.0                     # 0.9 at the 500 m nest
    gentle = np.full((30, 30), 300.0)
    planned = []
    monkeypatch.setattr(corridor, "moving_grid_ids",
                        lambda experiment: frozenset({2}))
    monkeypatch.setattr(
        corridor, "planned_corridor",
        lambda experiment, dc: planned.append(int(dc.grid_id))
        or SimpleNamespace(geometry={"corridor_of": int(dc.grid_id)}))
    monkeypatch.setattr(corridor, "corridor_grid",
                        lambda reference, geometry: ("corridor", geometry))
    monkeypatch.setattr(
        runtime.GeogSelection, "from_case_data",
        classmethod(lambda cls, data, domain_id=1: SimpleNamespace(
            root=data.geog_root)))

    def ground(grid):
        if isinstance(grid, tuple):
            return steep
        return gentle if grid.e_we == 31 else np.full((40, 40), 300.0)

    monkeypatch.setattr(
        runtime, "case_static_fields",
        lambda grid, root, **kwargs: {"HGT_M": ground(grid)})
    monkeypatch.setattr(build, "build_terrain",
                        lambda grid, root, selection=None: ground(grid))
    data = SimpleNamespace(
        geog_root=Path("geog"),
        static_highres=SimpleNamespace(enabled=True) if highres else None)
    derived = runtime._terrain_acoustics_for_case(exp, data)
    assert planned == [2]
    assert [dc.run.time_step_sound for dc in derived.domains] == [4, 6]
