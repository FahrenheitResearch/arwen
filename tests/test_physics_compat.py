"""Fail-closed gates for the Thompson + MYNN + RUC port lane."""

from __future__ import annotations

from dataclasses import replace

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.physics_compat import (
    IMPLICIT_RUNTIME_SWITCHES,
    PhysicsCapabilityError,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    UnsupportedPhysicsSuiteError,
    WSM6_PROFILE_ID,
    WRF_RRTMG_TO_RTE_RRTMGP,
    implicit_runtime_switches,
    pending_wrf_physics_components,
    single_domain_runtime_switches,
    validate_physics_capabilities,
)


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=4, ny=3, nz=4, dx=1000.0, dy=1000.0,
                  ztop=10000.0, dt=5.0, run_seconds=10.0)
    values.update(overrides)
    return RunConfig(**values)


def test_full_target_mynn_ruc_pair_has_no_remaining_blocker():
    """The target suite's receipt after the ownership port.

    MYNN no longer appears because 5/5 runs, "RUC land-surface model" no
    longer appears because RUC runs, and "Thompson microphysics" no longer
    appears because mp8 is first-class now that the canonical classic
    tables ship as package data (product decision, product/v1 packaging
    lane 2026-07-28 -- the byte validation moved to table load, it did not
    disappear).  The MYNN/RUC pair is now admitted because WRF's ordered
    RUC write-back and SFCDIAGS_RUCLSM ownership are explicit.
    """
    blockers = pending_wrf_physics_components(
        mp_physics=8, sf_sfclay_physics=5, bl_pbl_physics=5,
        sf_surface_physics=3, num_soil_layers=9)
    assert blockers == ()

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


def test_every_shipped_profile_answers_its_own_implicit_switches():
    """The drift guard for the one authority on moist_cq/top_lid.

    Three places used to decide these: the shipped profiles (read by the
    HRRR root preparer AND the domain wizard) and the WRF namelist
    importer, which invented ``moist_cq = mp_physics > 0`` and WRF's
    open-top Registry default.  For every WSM6-family suite the two
    answers were opposite, which is why a public root prepared from a
    profile could not bind a public hierarchy imported from the same
    namelist.  If a new profile ever disagrees with what this lookup
    returns for its own selectors, that separation is back, and this
    fails.
    """

    for profile in SINGLE_DOMAIN_PHYSICS_PROFILES:
        switches = single_domain_runtime_switches(profile)
        resolved = implicit_runtime_switches(**switches)
        assert profile in resolved["profiles"], profile
        for name in IMPLICIT_RUNTIME_SWITCHES:
            assert resolved[name] == switches[name], (profile, name)
        assert profile in resolved["source"]

    # A suite that is no shipped profile falls back to gpuwm's own
    # RunConfig defaults, and says so rather than guessing a profile.
    defaults = _cfg()
    unknown = implicit_runtime_switches(
        mp_physics=6, sf_sfclay_physics=1, sf_surface_physics=0,
        bl_pbl_physics=0, cu_physics=0, num_soil_layers=4,
        ra_lw_physics=1, ra_sw_physics=1)
    assert unknown["profiles"] == ()
    assert unknown["moist_cq"] == defaults.moist_cq
    assert unknown["top_lid"] == defaults.top_lid
    assert "not one of the shipped" in unknown["source"]

    # An empty selection is not a match for everything.
    assert implicit_runtime_switches()["profiles"] == ()


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


def test_mixed_mynn_pairings_follow_wrf_v461_instead_of_a_half_suite_gate():
    """WRF admits three mixed pairings and fatals the other three."""

    for surface, pbl in (
            (5, 5), (1, 5), (91, 5), (5, 0), (1, 0), (91, 0)):
        assert pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=surface,
            bl_pbl_physics=pbl, sf_surface_physics=2,
            num_soil_layers=4) == ()

    for surface, pbl in ((5, 1), (0, 1), (0, 5)):
        blockers = pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=surface,
            bl_pbl_physics=pbl, sf_surface_physics=2,
            num_soil_layers=4)
        assert [item.component for item in blockers] == [
            "WRF v4.6.1 PBL/surface-layer compatibility"]
        assert blockers[0].selectors == (
            ("sf_sfclay_physics", surface), ("bl_pbl_physics", pbl))
        assert "phys/module_physics_init.F:" in blockers[0].missing[1]


def test_target_runconfig_is_admitted_without_substitution():
    """The target MYNN/RUC suite validates directly.

    RUC at nine layers is admitted, mp8 is first-class (packaged tables;
    the promotion is the same product decision documented in
    ``test_full_target_mynn_ruc_pair_has_no_remaining_blocker``), and the
    surface-driver ownership port makes the MYNN/RUC pair first-class.
    """
    cfg = _cfg(nz=5, moist=True, mp_physics=8, sf_sfclay_physics=5,
               bl_pbl_physics=5, sf_surface_physics=3,
               num_soil_layers=9)
    assert validate_run_config(cfg) is cfg

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
