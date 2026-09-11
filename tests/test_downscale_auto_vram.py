"""Measured memory must constrain an archived child on a busy large GPU."""
from dataclasses import replace
import json

import pytest

from gpuwm.cli import main
from gpuwm.core import preflight as pf
from gpuwm.core.kernel_frame_recordings import NOAHMP_COMPOSED_FRAME_RECORDINGS
from gpuwm.downscale import _fit_child_size
from gpuwm.domain_wizard import SizingBudget
from test_downscale_cli import _PARENT_CONFIG, _point_args, _add_parent_surface


def test_free_memory_changes_the_actual_fit_without_changing_card_capacity():
    parent = {"nx": 501, "ny": 501, "dx": 1000.0, "dy": 1000.0}
    config = dict(_PARENT_CONFIG, nx=501, ny=501, nz=49)
    args = dict(j0=250, i0=250, ratio=2, run_seconds=3600.0,
                output_interval_s=3600.0, vram_gib=32.0)
    # The fitter returns (size, estimate): the extent and the price it was
    # decided on, so the plan document reports the same number.
    nominal, _ = _fit_child_size(parent, config, **args)
    occupied, _ = _fit_child_size(parent, config, **args, measured_free_bytes=8 * 1024**3)
    assert 8 <= occupied < nominal
    assert occupied % 4 == 0


# --child-size beside --auto-vram is no longer here: a drawn extent priced on
# the measured card is what that pair means (test_downscale_cli), and only a
# declared capacity contradicts a measurement.
@pytest.mark.parametrize("extra", [["--vram-gib", "32"], ["--card", "32gb"]])
def test_ambiguous_automatic_sizing_refuses_before_reading_an_archive(tmp_path, capsys, extra):
    assert main(["downscale", str(tmp_path / "absent"), "--point", "35,-97",
                 "--out", str(tmp_path / "output"), "--auto-vram", *extra]) == 2
    assert "--auto-vram" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_auto_point_plan_uses_one_budget_and_reports_its_basis(tmp_path, capsys, monkeypatch):
    import gpuwm.downscale as downscale
    import gpuwm.domain_wizard as wizard
    args = _point_args(tmp_path)
    index = args.index("--child-size")
    del args[index:index + 2]
    for index in range(3):
        _add_parent_surface(tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00", ny=18, nx=20)
    observed = []
    device_profile = object()
    budget = SizingBudget(32.0, 8 * 1024**3, device_profile, "Measured test device", True)
    def probe(card, capacity):
        observed.append((card, capacity))
        return budget
    fitted = []
    def fit(parent, config, **kwargs):
        fitted.append(kwargs)
        return 12, None
    monkeypatch.setattr(wizard, "resolve_sizing_budget", probe)
    monkeypatch.setattr(downscale, "_fit_child_size", fit)
    assert main([*args, "--auto-vram", "--dry-run"]) == 0
    output = capsys.readouterr().out
    plan = json.loads(output[output.index("{\n"):])
    assert observed == [(None, None)]
    assert fitted[0]["vram_gib"] == 32.0
    assert fitted[0]["measured_free_bytes"] == 8 * 1024**3
    assert fitted[0]["profile"] is device_profile, (
        "the measured card's own profile prices the fit, not a declared card")
    assert plan["gpu_sizing"]["basis"] == "measured-local"
    assert plan["gpu_sizing"]["free_bytes"] == budget.free_bytes


def test_a_noahmp_parent_is_priced_on_the_measured_card_and_from_the_ceiling_on_a_declared_one(monkeypatch):
    """The auto-sizer hands the estimator the card it measured.

    The breakage this prevents: ``_fit_child_size`` passed
    ``estimate_experiment`` no profile, so a Noah-MP parent under
    ``--auto-vram`` -- basis "measured-local", the card just read -- was
    priced as a card that is not in this machine.  A profile carrying a
    recorded platform prices from its own row; a declared card, or a
    measured free figure whose probe returned no profile, is priced on
    the reference profile from the Noah-MP ceiling, never refused and
    never more optimistic than the measured card.
    """
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    parent = {"nx": 501, "ny": 501, "dx": 1000.0, "dy": 1000.0}
    config = dict(_PARENT_CONFIG, nx=501, ny=501, nz=49, sf_surface_physics=4,
                  sf_sfclay_physics=1, bl_pbl_physics=1, num_soil_layers=4)
    args = dict(j0=250, i0=250, ratio=2, run_seconds=3600.0,
                output_interval_s=3600.0, vram_gib=16.0)
    recorded = replace(pf.MEASURED_LOCAL_MEMORY_PROFILE,
                       name="measured test card",
                       compile_platform=NOAHMP_COMPOSED_FRAME_RECORDINGS[0].platform_key)
    size, priced = _fit_child_size(parent, config, **args,
                                   measured_free_bytes=8 * 1024**3, profile=recorded)
    assert size >= 8 and size % 4 == 0
    # The price the fit was decided on is the estimator's own, and it is
    # the one the plan document will report.
    assert priced is not None and priced.peak_envelope_bytes > 0
    noah, _ = _fit_child_size(parent, dict(config, sf_surface_physics=2), **args,
                              measured_free_bytes=8 * 1024**3, profile=recorded)
    assert noah >= size, ("Noah-MP carries more resident state than Noah and "
                          "may size smaller, never larger, on the same card")
    declared, _ = _fit_child_size(parent, config, **args)
    assert declared >= 8 and declared % 4 == 0
    probed, _ = _fit_child_size(parent, config, **args,
                                measured_free_bytes=8 * 1024**3)
    assert probed >= 8 and probed % 4 == 0
    assert probed <= size, ("the reference geometry prices the ceiling frames on "
                            "170 SMs; it can never size larger than the measured card")
