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
    offcentering_floor, readings_from_static, stable_slopes, steepest_slope,
    SlopeReading)


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
    #: grid_ids whose epssm is the model's choice (ExperimentConfig's).
    auto_epssm: tuple = ()


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
    assert ("refusing the forecast before GPU time"
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
    assert "refusing the forecast before GPU time" in adaptation.beyond_sentence()


def test_one_line_per_domain_past_the_measured_bound():
    exp = _Experiment((_Domain(1, _Run()),))
    announced, cautioned = [], []
    with pytest.raises(ValueError, match="before GPU time.*non-finite.*Prepare again"):
        adapt_experiment_acoustics(
            exp, {1: _reading(1.07)}, announce=announced.append,
            caution=cautioned.append)
    assert announced == cautioned == []


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
    with pytest.raises(ValueError, match="before GPU time"):
        runtime._terrain_acoustics_for_case(exp, data)
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
    with pytest.raises(ValueError, match="before GPU time"):
        runtime._terrain_acoustics_for_case(exp, data)
    assert planned == [2]


# ---------------------------------------------------------------------------
# The off-centering floor (A171): a default epssm over ground it was not
# measured stable on.  A user's 1 km mountain nest (steepest slope 0.87)
# ran WRF's default 0.1 because its namelist listed epssm once; WRF 4.7.1
# faulted at step 19 and gpuwm went non-finite at step 46 on the same
# real.exe files, and both ran at 0.5.
# ---------------------------------------------------------------------------

def test_the_floor_reads_the_measured_map():
    # WRF's default holds every slope its six-substep bound covers.
    for slope in (0.0, 0.33, 0.49):
        assert offcentering_floor(slope) is None
    # Past it, the least row whose six-substep bound is steeper.
    assert offcentering_floor(0.50) == 0.2
    assert offcentering_floor(0.64) == 0.2
    assert offcentering_floor(0.65) == 0.3
    assert offcentering_floor(0.70) == 0.4
    assert offcentering_floor(0.80) == 0.5
    # Past every row, the most stable off-centering measured.
    assert offcentering_floor(0.87) == 0.5
    assert offcentering_floor(1.07) == 0.5
    # Every row's own six-substep bound is held by that row or a lower one.
    for epssm, _four, six in STABLE_SLOPE_BY_OFFCENTERING:
        floor = offcentering_floor(six - 1e-6)
        assert floor is None or floor <= epssm


def test_a_default_epssm_over_steep_ground_takes_the_floor():
    exp = _Experiment((_Domain(1, _Run(dx=3000.0, dy=3000.0)),
                       _Domain(2, _Run(dx=1000.0, dy=1000.0, epssm=0.1))),
                      auto_epssm=(2,))
    lines = []
    adapted, adaptations = adapt_experiment_acoustics(
        exp, {1: _reading(0.31, "d01"), 2: _reading(0.84, "d02")},
        announce=lines.append, caution=lines.append)
    assert adapted.domains[0] is exp.domains[0]
    child = adapted.domains[1].run
    assert child.epssm == 0.5 and child.time_step_sound == 6
    assert adaptations[1].offcentering_raised
    assert adaptations[1].configured_epssm == 0.1
    row = acoustic_receipt(adaptations)["domains"][1]
    assert row["configured_epssm"] == 0.1 and row["epssm"] == 0.5
    assert row["epssm_basis"] == "measured off-centering floor"
    # The substep row is read at the epssm that runs.
    assert row["six_substeps_stable_below"] == 0.85
    assert "configured_epssm" not in acoustic_receipt(adaptations)[
        "domains"][0]
    assert lines[0].startswith("acoustic off-centering: d02")
    assert "epssm 0.1 is the default" in lines[0]
    assert "runs epssm 0.5" in lines[0]
    assert "past 0.50" in lines[0]
    # The substep count follows the floor on in-limit steep ground.
    assert lines[1].startswith("acoustic substeps: d02")


def test_a_default_epssm_inside_the_map_takes_the_least_row_that_holds():
    exp = _Experiment((_Domain(2, _Run(dx=1000.0, dy=1000.0, epssm=0.1)),),
                      auto_epssm=(2,))
    adapted, adaptations = adapt_experiment_acoustics(
        exp, {2: _reading(0.6, "d02")})
    assert adapted.domains[0].run.epssm == 0.2
    assert adapted.domains[0].run.time_step_sound == 6
    assert adaptations[0].status == "ADAPTED"


def test_a_default_epssm_on_gentle_ground_is_untouched():
    exp = _Experiment((_Domain(1, _Run(epssm=0.1)),), auto_epssm=(1,))
    same, adaptations = adapt_experiment_acoustics(exp, {1: _reading(0.3)})
    assert same is exp
    assert not adaptations[0].offcentering_raised
    assert "configured_epssm" not in acoustic_receipt(adaptations)[
        "domains"][0]
    # A default already at or above the floor is left alone too.
    exp = _Experiment((_Domain(1, _Run(epssm=0.5)),), auto_epssm=(1,))
    _, adaptations = adapt_experiment_acoustics(exp, {1: _reading(0.84)})
    assert not adaptations[0].offcentering_raised


def test_a_chosen_epssm_below_the_floor_is_refused_with_the_floor_named():
    exp = _Experiment((_Domain(1, _Run(dx=3000.0, dy=3000.0)),
                       _Domain(2, _Run(dx=1000.0, dy=1000.0, epssm=0.1))))
    with pytest.raises(ValueError) as refused:
        adapt_experiment_acoustics(
            exp, {1: _reading(0.31, "d01"), 2: _reading(0.87, "d02")})
    message = str(refused.value)
    assert message.startswith("d02's epssm 0.1 is set explicitly")
    assert ("measured stable with any acoustic substep count only below "
            "0.50") in message
    assert "at least 0.5" in message and '"auto"' in message
    # `gpuwm run` showed a worker's error cut to about 250 characters,
    # and the cut fell before the remedy: the remedy now leads.
    assert message.index("set d02's epssm to at least 0.5") < 200
    assert "d01" not in message


def test_a_chosen_epssm_at_the_floor_runs_as_chosen():
    exp = _Experiment((_Domain(2, _Run(dx=1000.0, dy=1000.0, epssm=0.2)),))
    adapted, adaptations = adapt_experiment_acoustics(
        exp, {2: _reading(0.6, "d02")})
    assert adapted.domains[0].run.epssm == 0.2
    assert not adaptations[0].offcentering_raised


_OFFCENTERING_TREE = """\
[experiment]
name = "offcentering"
start_time = 2026-09-22T12:00:00
run_seconds = 600.0
restart_interval_s = 0.0

[shared]
nz = 8
ztop = 20000.0
p_top = 50000.0
eta_levels = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
ra_lw_physics = 0
ra_sw_physics = 0
bl_pbl_physics = 0
{shared}
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 40
ny = 40
time_step = 15
dx = 3000.0
history_interval_s = 600.0

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 12
j_parent_start = 12
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 30
ny = 30
history_interval_s = 600.0
{child}"""


def _tree(tmp_path, shared="", child=""):
    from gpuwm.experiment import load_experiment

    path = tmp_path / "offcentering.toml"
    path.write_text(_OFFCENTERING_TREE.format(
        shared=shared + "\n" if shared else "",
        child=child + "\n" if child else ""))
    return load_experiment(path)


def test_the_loader_labels_each_domain_whose_epssm_the_model_chooses(
        tmp_path):
    unset = _tree(tmp_path)
    assert unset.auto_epssm == (1, 2)
    assert [dc.run.epssm for dc in unset.domains] == [0.1, 0.1]
    written = _tree(tmp_path, shared="epssm = 0.5")
    assert written.auto_epssm == ()
    tail = _tree(tmp_path, shared="epssm = 0.5", child='epssm = "auto"')
    assert tail.auto_epssm == (2,)
    assert [dc.run.epssm for dc in tail.domains] == [0.5, 0.1]
    shared_auto = _tree(tmp_path, shared='epssm = "auto"',
                        child="epssm = 0.3")
    assert shared_auto.auto_epssm == (1,)
    assert [dc.run.epssm for dc in shared_auto.domains] == [0.1, 0.3]
    with pytest.raises(ValueError, match='or the string "auto"'):
        _tree(tmp_path, child='epssm = "default"')


def test_the_epssm_label_never_reaches_the_restart_identity(tmp_path):
    from gpuwm.core.model import restart_identity_payload

    auto = _tree(tmp_path, shared="epssm = 0.5", child='epssm = "auto"')
    written = _tree(tmp_path, shared="epssm = 0.5", child="epssm = 0.1")
    assert "auto_epssm" not in restart_identity_payload(auto)
    assert restart_identity_payload(auto) == restart_identity_payload(
        written)
    assert replace(auto, auto_epssm=()) == written


def test_gpuwm_run_publishes_the_raised_epssm_in_its_folder(tmp_path):
    """`gpuwm run` writes the acoustic record beside its history when the
    floor raised a domain's epssm, and nothing when no epssm moved."""
    import json

    from gpuwm import runtime

    exp = _Experiment((_Domain(2, _Run(dx=1000.0, dy=1000.0, epssm=0.1)),),
                      auto_epssm=(2,))
    _, calm = adapt_experiment_acoustics(exp, {2: _reading(0.3, "d02")})
    assert runtime._write_acoustic_receipt(tmp_path, calm) is None
    assert not (tmp_path / runtime.ACOUSTIC_RECEIPT_NAME).exists()
    _, raised = adapt_experiment_acoustics(exp, {2: _reading(0.84, "d02")})
    path = runtime._write_acoustic_receipt(tmp_path, raised)
    assert path == tmp_path / runtime.ACOUSTIC_RECEIPT_NAME
    row = json.loads(path.read_text())["domains"][0]
    assert (row["configured_epssm"], row["epssm"]) == (0.1, 0.5)
    assert row["epssm_basis"] == "measured off-centering floor"


def test_auto_under_a_named_profile_is_the_profiles_to_supply():
    """The importer writes epssm = "auto" (A171), which means the same as
    leaving epssm unset; under a named physics profile that is the profile's
    value to supply, while a stated 0.1 still differs from the profile."""

    import re
    from datetime import datetime

    from gpuwm import domain_wizard as wizard
    from gpuwm.prepared_single_domain_forecast import (
        PHYSICS_PROFILES, named_profile_config_conflicts)

    text = wizard.render_config(
        name="acoustic-route", start_time=datetime(2024, 6, 13, 12),
        hours=1, projection={
            "map_proj": "lambert", "ref_lat": -33.0, "ref_lon": -70.1,
            "truelat1": -23.0, "truelat2": -43.0, "stand_lon": -70.1},
        dims=[(40, 40)], ratios=(), fetch_hints={"source": "gfs"},
        case_data=None, root_dx_m=500.0, history_interval_s=300.0)
    profile = next(name for name in PHYSICS_PROFILES
                   if name.startswith("thompson-mp8-mynn-mynn-ruc-rte"))

    def epssm_rows(value):
        stated = re.sub(r"^epssm = .*$", f"epssm = {value}", text,
                        count=1, flags=re.M)
        assert stated != text or value == "0.5"
        return [row for row in named_profile_config_conflicts(
            stated, source="gfs", profile=profile) if row["key"] == "epssm"]

    assert epssm_rows('"auto"') == []
    assert [row["config_value"] for row in epssm_rows("0.1")] == [0.1]


# ---------------------------------------------------------------------------
# The pinned clock (RunConfig.terrain_clock = "pinned"): the configured
# count runs and the measured raise is advice in the receipt.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PinnedRun(_Run):
    terrain_clock: str = "pinned"


def test_a_pinned_domain_keeps_its_configured_count_over_steep_ground():
    measured = derive_acoustics(1, _Run(), _reading(0.734))
    assert measured.time_step_sound == 6
    pinned = derive_acoustics(1, _PinnedRun(), _reading(0.734))
    assert pinned.pinned and pinned.status == "PINNED"
    assert pinned.time_step_sound == 4 and not pinned.adapted
    assert pinned.advice == 6 and pinned.advice_differs
    line = pinned.pinned_sentence()
    assert line.startswith("acoustic substeps: d01's clock is pinned")
    assert "would have run 6" in line and "not applied" in line
    row = pinned.receipt()
    assert row["status"] == "PINNED"
    assert row["time_step_sound"] == 4
    assert row["clock"] == "pinned"
    assert row["advice"] == {"applied": False, "time_step_sound": 6,
                             "differs": True}
    # A measured receipt is byte-for-byte what it was.
    assert "clock" not in measured.receipt()


def test_a_pinned_domain_on_gentle_ground_carries_no_differing_advice():
    pinned = derive_acoustics(1, _PinnedRun(), _reading(0.357))
    assert pinned.time_step_sound == 4 and not pinned.advice_differs
    assert pinned.receipt()["advice"] == {
        "applied": False, "time_step_sound": 4, "differs": False}


def test_the_pinned_experiment_door_keeps_the_run_and_cautions():
    exp = _Experiment((_Domain(1, _PinnedRun()),))
    said = []
    adapted, adaptations = adapt_experiment_acoustics(
        exp, {1: _reading(0.734)},
        announce=lambda line: said.append(("announce", line)),
        caution=lambda line: said.append(("caution", line)))
    assert adapted is exp
    assert adaptations[0].time_step_sound == 4
    assert said == [("caution", adaptations[0].pinned_sentence())]


def test_the_pinned_clock_does_not_lift_the_offcentering_refusal():
    """The floor is not a clock rule: a chosen epssm the map holds no count
    at is refused pinned or measured, because WRF stops on it too."""
    exp = _Experiment((_Domain(1, _PinnedRun(epssm=0.1)),))
    with pytest.raises(ValueError, match="epssm 0.1 is set explicitly"):
        adapt_experiment_acoustics(exp, {1: _reading(0.87)})
