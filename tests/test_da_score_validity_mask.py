"""The DA scorers score only where a radar looked.

Audit item S17: the scorer of record filled every column no radar measured
with ``MISSING_OBS_FILL_DBZ`` and scored it as confidently observed no-echo
(34.4 percent of the KDMX verification domain), and its headline was the
ensemble-mean field at 30 dBZ.  These tests pin the fix on the real scorer
(``tools.da_sweep_score``) reading a real ``gpuwm-obs.radar-grid``-shaped
NetCDF file, through the Rust masked FSS kernel:

* forecast echo placed only where no radar looked cannot move the score
  (it used to count as false alarm against an invented no-echo truth);
* the coverage fraction is in every frame and in the headline;
* a file that carries no clear-air census is scored unmasked and SAYS so
  rather than inventing a mask from echo alone;
* the headline names the per-member mean as primary, keeps the
  ensemble-mean field under its own name, and adds 40 dBZ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from tools import da_sweep_score as score  # noqa: E402

NZ, NY, NX = 3, 48, 60
DX_KM = 3.0
#: Columns 0..39 are inside radar coverage, 40..59 are not.
COVERED_NX = 40


def _blob(cj, ci, *, radius=4.0, peak=50.0, floor=-35.0):
    j, i = np.indices((NY, NX)).astype(np.float64)
    distance = np.hypot(j - cj, i - ci)
    return np.maximum(floor, floor + (peak - floor)
                      * np.exp(-0.5 * (distance / radius) ** 2))


def _write_obs(path: Path, composite: np.ndarray, *, census=True,
               valid="2026-08-05T05:45:00Z"):
    """A radar-grid file whose radars saw only columns < COVERED_NX."""
    covered = np.zeros((NY, NX), dtype=bool)
    covered[:, :COVERED_NX] = True
    echo = covered & (composite > 5.0)
    z_obs = np.zeros((NZ, NY, NX))
    z_mask = np.zeros((NZ, NY, NX), dtype=np.int8)
    z_obs[0][echo] = composite[echo]
    z_mask[0][echo] = 1
    z0_count = np.zeros((NZ, NY, NX), dtype=np.int32)
    z0_count[0][covered & ~echo] = 7
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("z", NZ)
        ds.createDimension("y", NY)
        ds.createDimension("x", NX)
        ds.createVariable("z_obs", "f8", ("z", "y", "x"))[:] = z_obs
        ds.createVariable("z_mask", "i1", ("z", "y", "x"))[:] = z_mask
        if census:
            ds.createVariable("z0_count", "i4", ("z", "y", "x"))[:] = z0_count
        ds.setncattr("valid_time", valid)


def _write_leg(directory: Path, leg: int, members: dict, control):
    directory.mkdir(parents=True, exist_ok=True)
    for name, field in {**members, "control": control}.items():
        np.savez(directory / f"leg{leg:02d}_{name}.npz",
                 refl_colmax=np.asarray(field, np.float64))


def _case(tmp_path: Path, *, outside_echo: bool, census=True):
    """Observed storm inside coverage; members near it; optionally every
    member also carries a storm where no radar looked."""
    observed = _blob(20, 15)
    members = {}
    for m in range(4):
        field = _blob(20 + m, 17 + m)
        if outside_echo:
            field = np.maximum(field, _blob(30, 52))
        members[str(m)] = field
    control = _blob(35, 30)
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir(parents=True)
    _write_obs(obs_dir / "case-verify-0545.nc", observed, census=census)
    composites = tmp_path / "composites"
    _write_leg(composites, 6, members, control)
    return composites, obs_dir / "case-verify-0545.nc", obs_dir


def _const():
    const, _source = score.metric_constants()
    return const


def test_echo_where_no_radar_looked_cannot_move_the_score(tmp_path):
    quiet = _case(tmp_path / "quiet", outside_echo=False)
    noisy = _case(tmp_path / "noisy", outside_echo=True)
    kwargs = dict(leg=6, dx_km=DX_KM, const=_const(), structure=False)
    a = score.score_leg(composites=quiet[0], obs_path=quiet[1], **kwargs)
    b = score.score_leg(composites=noisy[0], obs_path=noisy[1], **kwargs)

    # Masked: identical, for the mean field and for every member.
    assert a["fss30_fcst"] == b["fss30_fcst"]
    assert a["per_member"]["fss30"] == b["per_member"]["fss30"]
    # The old whole-grid score treated the unseen columns as observed
    # no-echo and charged the members for weather nobody measured.
    assert b["fss30_fcst_unmasked"] < a["fss30_fcst_unmasked"]
    assert b["fss30_fcst_unmasked"] < b["fss30_fcst"]


def test_coverage_is_recorded_per_frame(tmp_path):
    composites, obs_path, _ = _case(tmp_path, outside_echo=False)
    row = score.score_leg(composites=composites, obs_path=obs_path, leg=6,
                          dx_km=DX_KM, const=_const(), structure=False)
    cover = row["coverage"]
    assert cover["masked"] is True
    assert cover["source"] == "z_mask | z0_count>0"
    assert cover["observed_cells"] == NY * COVERED_NX
    assert cover["fraction"] == pytest.approx(COVERED_NX / NX, abs=1e-4)


def test_a_file_without_a_clear_air_census_is_scored_unmasked_and_says_so(
        tmp_path):
    composites, obs_path, _ = _case(tmp_path, outside_echo=True,
                                    census=False)
    row = score.score_leg(composites=composites, obs_path=obs_path, leg=6,
                          dx_km=DX_KM, const=_const(), structure=False)
    assert row["coverage"]["masked"] is False
    assert row["coverage"]["source"].startswith("unavailable")
    assert row["fss30_fcst"] == row["fss30_fcst_unmasked"]


def test_unobserved_frame_is_refused(tmp_path):
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    path = obs_dir / "empty-verify.nc"
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("z", NZ)
        ds.createDimension("y", NY)
        ds.createDimension("x", NX)
        ds.createVariable("z_obs", "f8", ("z", "y", "x"))[:] = 0.0
        ds.createVariable("z_mask", "i1", ("z", "y", "x"))[:] = 0
        ds.createVariable("z0_count", "i4", ("z", "y", "x"))[:] = 0
        ds.setncattr("valid_time", "2026-08-05T05:45:00Z")
    with pytest.raises(SystemExit, match="no column of this frame"):
        score.load_observation(path, _const())


def test_headline_is_the_member_score_beside_a_labelled_mean_field(tmp_path):
    composites, _obs_path, obs_dir = _case(tmp_path, outside_echo=True)
    out = tmp_path / "score.json"
    assert score.main(["--composites", str(composites),
                       "--obs-dir", str(obs_dir),
                       "--first-free-leg", "6", "--dx-km", str(DX_KM),
                       "--label", "mask-test", "--out", str(out),
                       "--no-structure"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    head = payload["headline"]
    assert payload["schema"] == "gpuwm-da.sweep-score.v3"
    assert head["primary"] == "fss30_per_member_mean"
    assert head["fss30_per_member_mean"] == payload["fss30_per_member_mean"]
    assert head["fss30_ensemble_mean_field"] == payload["fss30_fcst_mean"]
    assert head["validity_mask"] == "on"
    assert head["coverage_fraction_mean"] == pytest.approx(COVERED_NX / NX,
                                                           abs=1e-4)
    assert "40" in head["by_threshold"]
    assert set(head["by_threshold"]["40"]) >= {
        "per_member_mean", "ensemble_mean_field", "control"}
    assert payload["validity_mask"]["rule"] == score.COVERAGE_RULE
    assert "fss30_fcst_unmasked_mean" in payload


def test_ab_contingency_does_not_count_unobserved_columns(tmp_path):
    from tools import da_ab_score

    truth = np.full((4, 4), -35.0)
    forecast = np.full((4, 4), -35.0)
    forecast[:, 3] = 50.0              # a core only where no radar looked
    coverage = np.ones((4, 4), dtype=bool)
    coverage[:, 3] = False
    whole = da_ab_score.contingency(forecast, truth, 35.0)
    masked = da_ab_score.contingency(forecast, truth, 35.0, coverage)
    assert whole["false_alarms"] == 4
    assert masked["false_alarms"] == 0
    assert masked["correct_negatives"] == 12
