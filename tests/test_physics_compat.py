"""Fail-closed gates for the Thompson + MYNN + RUC port lane."""

from __future__ import annotations

from dataclasses import replace

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.physics_compat import (
    PhysicsCapabilityError,
    UnsupportedPhysicsSuiteError,
    WSM6_PROFILE_ID,
    WRF_RRTMG_TO_RTE_RRTMGP,
    pending_wrf_physics_components,
    single_domain_runtime_switches,
    validate_physics_capabilities,
)


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=4, ny=3, nz=4, dx=1000.0, dy=1000.0,
                  ztop=10000.0, dt=5.0, run_seconds=10.0)
    values.update(overrides)
    return RunConfig(**values)


def test_full_target_reports_every_remaining_coupled_component():
    """The target suite's receipt, after the mp8 promotion.

    MYNN no longer appears because 5/5 runs, "RUC land-surface model" no
    longer appears because RUC runs, and "Thompson microphysics" no longer
    appears because mp8 is first-class now that the canonical classic
    tables ship as package data (product decision, product/v1 packaging
    lane 2026-07-28 -- the byte validation moved to table load, it did not
    disappear).  What survives is the MYNN/RUC PAIR.
    """
    blockers = pending_wrf_physics_components(
        mp_physics=8, sf_sfclay_physics=5, bl_pbl_physics=5,
        sf_surface_physics=3, num_soil_layers=9)
    assert [item.component for item in blockers] == [
        "MYNN surface layer with RUC",
    ]
    assert blockers[0].selectors == (
        ("sf_sfclay_physics", 5), ("sf_surface_physics", 3))

    # RUC's own refusals, each reported with the selector pair that caused it.
    six = pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=3, num_soil_layers=6)
    assert [item.component for item in six] == ["RUC soil geometry"]
    assert six[0].selectors == (
        ("sf_surface_physics", 3), ("num_soil_layers", 6))

    # And the admitted combination reports nothing at all, which is what makes
    # every refusal above a statement rather than a blanket.
    assert pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=3, num_soil_layers=9) == ()


def test_front_door_capability_refusal_cites_unimplemented_registry_option():
    selected = single_domain_runtime_switches(WSM6_PROFILE_ID)
    selected.update(ra_lw_physics=1, ra_sw_physics=1)

    with pytest.raises(PhysicsCapabilityError) as caught:
        validate_physics_capabilities(selected)

    message = str(caught.value)
    assert (
        "gpuwm/physics_registry_v2.json#/components/radiation/options/"
        "wrf-rrtm-dudhia"
    ) in message
    assert "Not implemented: the 1/1 WRF RRTM+Dudhia pair" in message
    assert "selectors {'ra_lw_physics': 1, 'ra_sw_physics': 1}" in message


def test_a_mynn_half_suite_is_the_only_mynn_refusal_left():
    """5/5 is admitted; either half alone is not.

    The receipt has to name the pair, because "MYNN is unsupported" would
    now be false and would send a user looking for missing physics that is
    in fact present.
    """
    assert pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=5, bl_pbl_physics=5,
        sf_surface_physics=2, num_soil_layers=4) == ()
    for surface, pbl in ((5, 1), (91, 5), (1, 5), (5, 0)):
        blockers = pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=surface, bl_pbl_physics=pbl,
            sf_surface_physics=2, num_soil_layers=4)
        assert [item.component for item in blockers] == ["MYNN half-suite"]
        assert blockers[0].selectors == (
            ("sf_sfclay_physics", surface), ("bl_pbl_physics", pbl))


def test_target_runconfig_fails_once_without_substitution():
    """The target suite still fails once, for the one surviving reason.

    RUC at nine layers is admitted, mp8 is first-class (packaged tables;
    the promotion is the same product decision documented in
    ``test_full_target_reports_every_remaining_coupled_component``), so
    neither contributes a token any more.  What still refuses this suite
    is the MYNN/RUC PAIR: RUC runs SFCDIAGS_RUCLSM after the LSM and MYNN
    diagnoses T2/Q2/TH2 itself, so the pair has two 2-m diagnostics and
    the second silently wins.  One receipt, and no substitution.
    """
    cfg = _cfg(moist=True, mp_physics=8, sf_sfclay_physics=5,
               bl_pbl_physics=5, sf_surface_physics=3,
               num_soil_layers=9)
    with pytest.raises(UnsupportedPhysicsSuiteError) as caught:
        validate_run_config(cfg)
    text = str(caught.value)
    assert "no substitutions were applied" in text
    for token in ("MYNN", "RUC", "two 2-m diagnostics"):
        assert token in text
    assert [item.component for item in caught.value.blockers] == [
        "MYNN surface layer with RUC"]

    # The nine-layer geometry is admitted; SIX is what is still refused, and
    # that receipt must name the count so a user is not left guessing.
    six = _cfg(moist=True, mp_physics=6, sf_sfclay_physics=91,
               bl_pbl_physics=1, sf_surface_physics=3, num_soil_layers=6)
    with pytest.raises(UnsupportedPhysicsSuiteError) as caught_six:
        validate_run_config(six)
    assert "num_soil_layers=9 only" in str(caught_six.value)

    # And the admitted combination validates, which is what makes the two
    # refusals above meaningful rather than a blanket "RUC is unsupported".
    validate_run_config(_cfg(moist=True, mp_physics=6, sf_sfclay_physics=91,
                             bl_pbl_physics=1, sf_surface_physics=3,
                             num_soil_layers=9))


def test_thompson_runconfig_is_admitted_without_environment(monkeypatch):
    """mp8 validates with NO Thompson environment set (first-class contract).

    Table bytes are not this gate's question: the packaged root (or an
    override) is byte-validated at load, where a wrong root fails closed.
    """
    monkeypatch.delenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)
    validate_run_config(_cfg(moist=True, mp_physics=8))


def test_rrtmg_compatibility_token_is_explicit_and_pair_bound():
    native = validate_run_config(_cfg(ra_physics=4))
    assert native.wrf_rrtmg_compatibility == "none"

    imported = validate_run_config(replace(
        native, wrf_rrtmg_compatibility=WRF_RRTMG_TO_RTE_RRTMGP))
    assert imported.wrf_rrtmg_compatibility == WRF_RRTMG_TO_RTE_RRTMGP

    with pytest.raises(ValueError, match="requires the resolved 4/4"):
        validate_run_config(_cfg(
            wrf_rrtmg_compatibility=WRF_RRTMG_TO_RTE_RRTMGP))
    with pytest.raises(ValueError, match="wrf_rrtmg_compatibility must"):
        validate_run_config(_cfg(wrf_rrtmg_compatibility="approximate"))


def test_rrtmg_adapter_rejects_cloud_coupling_off():
    with pytest.raises(ValueError, match="always on"):
        validate_run_config(_cfg(ra_physics=4, icloud=0))


def test_current_noah_contract_rejects_non_four_layer_state():
    with pytest.raises(ValueError, match="must be 4"):
        validate_run_config(_cfg(sf_surface_physics=2, num_soil_layers=9))
