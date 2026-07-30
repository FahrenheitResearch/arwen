"""CPU-only WRF WSM6 integration and independent diagnostic tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig, load_config, validate_run_config
from gpuwm.core import preflight
from gpuwm.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
from gpuwm.core.physics import microphysics_scratch_slots
from gpuwm.core.wsm6_constants import coefficients, rimed_ice_constants
from gpuwm.io.restart import MICROPHYSICS_ALGORITHM_IDENTITIES
from gpuwm.verify.npref import np_calc_cq, np_refl10cm_wsm6_column
from tools.wsm6_gpu_smoke import _smoke_failures


_TINY = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
             dt=1.0, run_seconds=10.0)


def _cfg(**changes):
    return RunConfig(**_TINY, moist=True, mp_physics=6, **changes)


def test_wsm6_vertical_kernel_tiers_cover_80_without_enlarging_z49():
    from gpuwm.core.wsm6 import _kernel_capacity

    assert _kernel_capacity(49) == 64
    assert _kernel_capacity(64) == 64
    assert _kernel_capacity(65) == 80
    assert _kernel_capacity(80) == 80
    with pytest.raises(ValueError, match="2 <= nz <= 80"):
        _kernel_capacity(81)


def test_config_accepts_wsm6_and_validates_hail_mode(tmp_path):
    path = tmp_path / "wsm6.toml"
    path.write_text(
        "[grid]\n"
        "nx=8\nny=6\nnz=4\ndx=1000.0\ndy=1000.0\nztop=10000.0\n"
        "[dynamics]\ndt=1.0\nmoist=true\nmp_physics=6\n"
        "wsm6_hail_opt=1\n"
        "[run]\nrun_seconds=10.0\n",
        encoding="utf-8")
    cfg = load_config(path)
    assert cfg.mp_physics == 6
    assert cfg.wsm6_hail_opt == 1
    with pytest.raises(ValueError, match="wsm6_hail_opt"):
        validate_run_config(replace(cfg, wsm6_hail_opt=2))


def test_wrf_wsm6_coefficients_and_restart_identity():
    graupel = coefficients(0)
    hail = coefficients(1)
    assert graupel.rimed == rimed_ice_constants(0)
    assert hail.rimed == rimed_ice_constants(1)
    assert graupel.rimed.n0g == 4.0e6
    assert graupel.rimed.deng == 500.0
    assert graupel.pvtg == pytest.approx(979.9530898208037, rel=2.0e-14)
    assert hail.rimed.n0g == 4.0e4
    assert hail.rimed.deng == 700.0
    assert hail.pvtg == pytest.approx(846.3231230270577, rel=2.0e-14)
    assert graupel.roqimax == pytest.approx(8.125e-5, rel=2.0e-15)
    assert MICROPHYSICS_ALGORITHM_IDENTITIES[6] == (
        "wsm6-single-moment-six-class-wrf-v4.6.1-v1")


def test_wsm6_state_preflight_nesting_and_diagnostics_inventory():
    cfg = _cfg(km_opt=4, bl_pbl_physics=1)
    state = preflight.state_array_shapes(cfg)
    mass = {"qv", "qc", "qr", "qi", "qs", "qg"}
    assert mass <= state.keys()
    assert {name + "0" for name in mass} <= state.keys()
    assert {"effc", "effi", "effs"} <= state.keys()
    assert not ({"nc", "nr", "ni", "ns", "ng", "effr"} & state.keys())
    assert preflight.nest_field_kinds(cfg) == (
        "u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
        "qi", "qs", "qg")

    scratch = preflight.scratch_slot_registry(cfg)
    for name in ("wsm6_theta", "wsm6_rho", "wsm6_pii", "wsm6_dz",
                 "wsm6_z8w", "mp_rainnc", "mp_snownc", "mp_graupelnc",
                 "mp_sr", "refl_t", "refl_10cm", "smag_rqi", "smag_rqs",
                 "smag_rqg"):
        assert name in scratch
    assert "smag_rnr" not in scratch
    assert dict(microphysics_scratch_slots(6)) == {
        "rainnc": "mp_rainnc", "rainncv": "mp_rainncv", "sr": "mp_sr",
        "snownc": "mp_snownc", "snowncv": "mp_snowncv",
        "graupelnc": "mp_graupelnc", "graupelncv": "mp_graupelncv",
    }
    assert kf_phase_mode_for_microphysics(6) == \
        KFPhaseMode.SEPARATE_ICE_SNOW


def test_wsm6_scratch_lifetime_audit_is_closed_world():
    """Every WSM6 registry slot has an explicit lifetime decision.

    Preparation arrays are rebuilt before every launch and may enter the
    shared arena.  Persistent precipitation accumulators remain carrying
    state and must never be admitted through an over-broad WSM6 prefix.
    """
    slots = preflight.scratch_slot_registry(
        _cfg(km_opt=4, diff_6th_opt=2, specified=True),
        n_lbc_intervals=2)
    assert slots
    assert {slot for slot in slots
            if preflight.scratch_slot_lifetime(slot) is None} == set()

    preparation = {
        "wsm6_theta", "wsm6_rho", "wsm6_pii", "wsm6_dz", "wsm6_z8w",
    }
    for slot in preparation:
        row = preflight.scratch_slot_lifetime(slot)
        assert row is not None
        assert row.kind == "write_before_read"
        assert preflight.scratch_slot_uses_arena(slot)

    persistent = {
        "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
        "mp_graupelnc", "mp_graupelncv", "mp_sr",
    }
    for slot in persistent:
        row = preflight.scratch_slot_lifetime(slot)
        assert row is not None
        assert row.kind == "carrying"
        assert not preflight.scratch_slot_uses_arena(slot)


def test_wsm6_domain_state_has_only_six_mass_species(monkeypatch):
    import gpuwm.core.state as state_module

    monkeypatch.setattr(state_module, "cp", np)
    state = state_module.DomainState(_cfg())
    for name in ("qi", "qs", "qg", "qi0", "qs0", "qg0"):
        assert getattr(state, name).shape == (4, 6, 8)
    for name in ("nc", "nr", "ni", "ns", "ng", "effr"):
        assert getattr(state, name, None) is None
    np.testing.assert_array_equal(state.effc, np.float32(2.49))
    np.testing.assert_array_equal(state.effi, np.float32(4.99))
    np.testing.assert_array_equal(state.effs, np.float32(9.99))


def test_wsm6_acoustic_cq_uses_six_mass_species_not_number_moments():
    shape = (3, 2, 4)
    moisture = {
        name: np.full(shape, index * 1.0e-4, dtype=np.float64)
        for index, name in enumerate(("qv", "qc", "qr", "qi", "qs", "qg"),
                                     start=1)
    }
    got = np_calc_cq(moisture, mp_physics=6)
    expected = np_calc_cq(moisture, mp_physics=10)
    for actual, reference in zip(got, expected):
        np.testing.assert_array_equal(actual, reference)


def test_wsm6_microphysics_dispatch_is_lazy_and_forwards_diagflag(monkeypatch):
    import gpuwm.core.microphysics as dispatcher
    import gpuwm.core.wsm6 as wsm6

    sentinel = object()

    def fake_apply(state, cfg, dt, *, refl_10cm_due=False):
        assert cfg.mp_physics == 6
        assert dt == 7.5
        assert refl_10cm_due is True
        return sentinel

    monkeypatch.setattr(wsm6, "apply", fake_apply)
    state = SimpleNamespace(qv=object())
    assert dispatcher.apply(state, _cfg(), 7.5, refl_10cm_due=True) is sentinel


@pytest.mark.parametrize(
    ("hail_opt", "wrf_dbz"),
    [
        (0, [33.3645859, 33.2047424, 33.0257988, 32.7733612,
             31.9111481, 31.3785324, 30.7436886, 29.9784565]),
        (1, [36.7684822, 36.1788673, 36.1384277, 36.0106049,
             35.0801582, 34.7000847, 34.2050133, 33.5677261]),
    ],
)
def test_wsm6_reflectivity_mirror_matches_direct_wrf_oracle(hail_opt, wrf_dbz):
    k = np.arange(8, dtype=np.float64)
    got = np_refl10cm_wsm6_column(
        0.0045 - 0.0002 * k,
        2.0e-4 + 1.0e-5 * k,
        1.2e-4 + 8.0e-6 * k,
        5.0e-5 + 5.0e-6 * k,
        280.0 - 2.0 * k,
        96000.0 - 7000.0 * k,
        hail_opt=hail_opt)
    np.testing.assert_allclose(got, wrf_dbz, rtol=0.0, atol=6.0e-6)


def test_gpu_smoke_accepts_wrf_physical_cloud_water_depletion():
    """A process category may be consumed; aggregate condensate must live.

    Direct WRF v4.6.1 consumes all qc in mixed-phase scenario 2 after the
    smoke's exact 10 x 1.5 s call cadence.  Requiring qc > 0 afterward is
    therefore a false failure, while requiring initial qc, process activity,
    retained condensate, and valid coupled state remains discriminating.
    """
    oracle = json.loads((Path(__file__).parent / "data" /
                         "wsm6_wrf461_oracle.json").read_text())
    evidence = oracle["repeated_call_evidence"]
    initial = evidence["initial_hydrometeor_maxima_kg_kg-1"]
    final_hydro = evidence["final_hydrometeor_maxima_kg_kg-1"]
    assert final_hydro["qc"] == 0.0
    changes = {name: 0.0 for name in ("qv", *final_hydro)}
    changes["qc"] = initial["qc"]
    failures = _smoke_failures(
        report={"nan": False}, finite=True, minimum=0.0,
        initial_maxima={"qv": 4.5e-3, **initial},
        changes=changes,
        total_condensate_max=evidence[
            "final_total_condensate_max_kg_kg-1"],
        h_diabatic_max=1.0e-4, elapsed_error=0.0)
    assert failures == []

    all_lost = _smoke_failures(
        report={"nan": False}, finite=True, minimum=0.0,
        initial_maxima={"qv": 4.5e-3, **initial},
        changes=changes, total_condensate_max=0.0,
        h_diabatic_max=1.0e-4, elapsed_error=0.0)
    assert all_lost == [
        "all condensate vanished from the mixed-phase smoke"]
