"""Unqualified native selections explain their missing complete graph."""
from types import SimpleNamespace

from gpuwm.ensemble.suite_capabilities import plan_suite


def test_nested_product_physics_names_missing_native_graphs():
    cfg = SimpleNamespace(mp_physics=8, sf_surface_physics=3, num_soil_layers=6,
        bl_pbl_physics=5, sf_sfclay_physics=5, cu_physics=0, sf_urban_physics=0,
        sf_surface_mosaic=0, topo_wind=0, gwd_opt=0, slope_rad=0, topo_shading=0,
        ra_physics=4, ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rte-rrtmgp",
        moist=True, use_adaptive_time_step=True, bldt=0)
    reasons = plan_suite(cfg, members=4).native_fallback_reasons
    assert any("RUC state and fused land driver graph" in row for row in reasons)
    assert any("MYNN turbulence, diffusion and TKE transport" in row for row in reasons)
    assert any("surface flux and carrier" in row for row in reasons)
    assert any("column, optics, flux and carrier" in row for row in reasons)
    assert any("independent CFL reductions and integer clocks" in row for row in reasons)
