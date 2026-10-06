"""Concurrency routing preserves selections and names the shared owner."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpuwm.config import RunConfig
from gpuwm.ensemble.ordinary_concurrency import plan_ordinary_concurrency


def _config(**options):
    return RunConfig(nx=8, ny=8, nz=8, dx=3000., dy=3000., ztop=12000.,
        dt=12., run_seconds=120., moist=True, mp_physics=8,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1,
        **options)


def _experiment(*configs):
    return SimpleNamespace(domains=tuple(SimpleNamespace(grid_id=number + 1, run=cfg)
                                        for number, cfg in enumerate(configs)))


def test_legacy_shortwave_routes_to_sequential_members_without_changing_the_suite():
    cfg = _config(ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rrtmg_legacy")
    before = vars(cfg).copy()
    plan = plan_ordinary_concurrency(_experiment(cfg))
    assert not plan.eligible
    assert len(plan.reasons) == 1 and "upload-ready event" in plan.reasons[0]
    assert plan.bindings[0]["binding"] == "legacy_rrtmg_shortwave_table_upload"
    assert plan.receipt()["fallback"] == "ordinary_members_in_sequence_per_card"
    assert vars(cfg) == before


@pytest.mark.parametrize("lw,sw,variant", [(4, 0, "rrtmg_legacy"),
    (4, 4, "rte-rrtmgp"), (0, 0, "rte-rrtmgp"), (1, 1, "rte-rrtmgp")])
def test_other_selected_bindings_do_not_inherit_the_shortwave_cache_reason(lw, sw, variant):
    cfg = _config(ra_lw_physics=lw, ra_sw_physics=sw, ra_rrtmg_variant=variant)
    plan = plan_ordinary_concurrency(_experiment(cfg))
    assert plan.eligible and not plan.reasons and not plan.bindings
    assert plan.receipt()["cfl_owner_policy"] == "member_scope_required_in_every_ordinary_wave"


def test_child_legacy_shortwave_is_diagnosed_under_its_own_domain():
    parent = _config(ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rte-rrtmgp")
    child = replace(parent, ra_rrtmg_variant="rrtmg_legacy")
    plan = plan_ordinary_concurrency(_experiment(parent, child))
    assert not plan.eligible
    assert len(plan.reasons) == 1 and plan.reasons[0].startswith("domain 2:")
    assert plan.bindings[0]["grid_id"] == 2


def test_radiation_alias_resolves_the_actual_legacy_shortwave_selection():
    cfg = _config(ra_physics=4, ra_rrtmg_variant="rrtmg_legacy")
    assert not plan_ordinary_concurrency(_experiment(cfg)).eligible
