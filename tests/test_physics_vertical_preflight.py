"""Preparation-time vertical bounds for resolved physics components.

The expected values are the independently audited first-call kernel/adapter
bounds.  Tests deliberately do not import the constants under test.
"""

from __future__ import annotations

import pytest

from gpuwm.physics_compat import (
    PhysicsVerticalPreflightError,
    validate_resolved_physics_vertical_levels,
)


def _selection(nz: int, **overrides):
    selected = {
        "nz": nz,
        "cu_physics": 0,
        "sf_surface_physics": 0,
        "mp_physics": 0,
        "bl_pbl_physics": 0,
        "ra_lw_physics": 0,
        "ra_sw_physics": 0,
        "sf_sfclay_physics": 0,
        "ra_rrtmg_variant": "rte-rrtmgp",
    }
    selected.update(overrides)
    return selected


@pytest.mark.parametrize(
    ("label", "selector", "inside", "outside"),
    (
        ("MYNN PBL", {"bl_pbl_physics": 5}, 5, 4),
        # The PBL column kernels hold one whole column in per-thread local
        # memory at a compiled KMAX, and their implicit vertical solves need
        # a minimum number of interior rows.  Both ends are first-call
        # rejections inside the launcher, so both belong in the preflight.
        ("YSU PBL", {"bl_pbl_physics": 1}, 128, 129),
        ("YSU PBL", {"bl_pbl_physics": 1}, 4, 3),
        ("MYJ PBL", {"bl_pbl_physics": 2}, 128, 129),
        ("MYJ PBL", {"bl_pbl_physics": 2}, 4, 3),
        ("Shin-Hong PBL", {"bl_pbl_physics": 11}, 128, 129),
        ("Shin-Hong PBL", {"bl_pbl_physics": 11}, 4, 3),
        ("SASE PBL", {"bl_pbl_physics": 900}, 128, 129),
        ("SASE PBL", {"bl_pbl_physics": 900}, 2, 1),
        ("Kain-Fritsch cumulus", {"cu_physics": 1}, 8, 7),
        ("Kain-Fritsch cumulus", {"cu_physics": 1}, 128, 129),
        # GF's floor is the inversion-layer search window (kend clamps to
        # ktf-8, the stencil then reaches ktf-1); no upper bound -- the
        # kernel recompiles its level bound through the int-define tier.
        ("Grell-Freitas cumulus", {"cu_physics": 3}, 12, 11),
        ("Kessler microphysics", {"mp_physics": 1}, 256, 257),
        ("WSM6 microphysics", {"mp_physics": 6}, 2, 1),
        ("WSM6 microphysics", {"mp_physics": 6}, 80, 81),
        ("Thompson microphysics", {"mp_physics": 8}, 2, 1),
        ("Thompson microphysics", {"mp_physics": 8}, 256, 257),
        ("Morrison microphysics", {"mp_physics": 10}, 2, 1),
        ("Morrison microphysics", {"mp_physics": 10}, 256, 257),
        ("NSSL-2 microphysics", {"mp_physics": 18}, 3, 2),
        ("NSSL-2 microphysics", {"mp_physics": 18}, 256, 257),
    ),
)
def test_component_owned_model_level_bound_inside_and_outside(
        label, selector, inside, outside):
    receipt = validate_resolved_physics_vertical_levels(
        _selection(inside, **selector))
    assert any(check["component"] == label for check in receipt["checks"])

    with pytest.raises(PhysicsVerticalPreflightError, match=label):
        validate_resolved_physics_vertical_levels(
            _selection(outside, **selector))


def test_rte_rrtmgp_model_plus_cap_bound_inside_and_outside():
    selector = {
        "mp_physics": 8,
        "ra_lw_physics": 4,
        "ra_sw_physics": 4,
    }
    receipt = validate_resolved_physics_vertical_levels(
        _selection(103, **selector), p_top=10000.0)
    longwave = next(
        check for check in receipt["checks"]
        if check["component"] == "RTE+RRTMGP longwave")
    assert longwave["above_model_layers"] == 25
    assert longwave["total_layers"] == 128

    with pytest.raises(
            PhysicsVerticalPreflightError,
            match=r"RTE\+RRTMGP longwave.*104\+25=129"):
        validate_resolved_physics_vertical_levels(
            _selection(104, **selector), p_top=10000.0)


def test_legacy_rrtmg_engine_bounds_inside_and_outside():
    selector = {
        "mp_physics": 8,
        "ra_lw_physics": 4,
        "ra_sw_physics": 4,
        "ra_rrtmg_variant": "rrtmg_legacy",
    }
    receipt = validate_resolved_physics_vertical_levels(
        _selection(63, **selector), p_top=26000.0)
    totals = {
        check["component"]: check["total_layers"]
        for check in receipt["checks"]
        if check["component"].startswith("legacy RRTMG")
    }
    assert totals == {
        "legacy RRTMG longwave": 128,
        "legacy RRTMG shortwave": 64,
    }

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        validate_resolved_physics_vertical_levels(
            _selection(64, **selector), p_top=26000.0)
    message = str(caught.value)
    assert "legacy RRTMG longwave" in message
    assert "64+65=129" in message
    assert "legacy RRTMG shortwave" in message
    assert "64+1=65" in message


def test_preflight_aggregates_more_than_one_resolved_component_failure():
    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        validate_resolved_physics_vertical_levels(_selection(
            4,
            bl_pbl_physics=5,
            cu_physics=1,
            mp_physics=18,
        ))
    message = str(caught.value)
    assert "MYNN PBL requires nz >= 5" in message
    assert "Kain-Fritsch cumulus requires 8 <= nz <= 128" in message
    assert "NSSL-2 microphysics" not in message


@pytest.mark.parametrize("p_top", (0.0, 10000.0, 26000.0))
def test_lightweight_layer_count_contract_matches_both_runtime_adapters(
        p_top):
    from gpuwm.core.rrtmg_legacy import (
        legacy_radiation_layer_counts as runtime_legacy_counts,
    )
    from gpuwm.core.rrtmgp import (
        rrtmgp_above_model_layer_counts as runtime_rrtmgp_counts,
    )
    from gpuwm.physics_vertical_contract import (
        legacy_radiation_layer_counts,
        rrtmgp_above_model_layer_counts,
    )

    assert rrtmgp_above_model_layer_counts(p_top) == (
        runtime_rrtmgp_counts(p_top))
    assert legacy_radiation_layer_counts(49, p_top) == (
        runtime_legacy_counts(49, p_top))
