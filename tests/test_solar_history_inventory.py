"""Default solar histories and memory budgets describe native radiation output."""
from dataclasses import replace
from gpuwm.config import RunConfig
from gpuwm.core.preflight import physics_array_shapes
from gpuwm.io.history_layout import produced_history_shapes


def test_default_and_composed_native_split_is_written_and_priced():
    base = RunConfig(nx=8, ny=6, nz=12, dx=3000., dy=3000., ztop=12000.,
                     dt=15., run_seconds=3600., ra_lw_physics=4, ra_sw_physics=4)
    for cfg in (base, replace(base, ra_lw_physics=0),
                replace(base, ra_rrtmg_variant="rrtmg_legacy"),
                replace(base, ra_lw_physics=1, ra_sw_physics=5)):
        shapes = produced_history_shapes(cfg)
        allocated = physics_array_shapes(cfg)
        for name in ("SWDDNI", "SWDDIR", "SWDDIF"):
            assert shapes[name] == (6, 8)
            assert allocated["fields/" + name.lower()] == (6, 8)
    off = produced_history_shapes(replace(base, ra_sw_physics=0))
    assert not {"SWDDNI", "SWDDIR", "SWDDIF"}.intersection(off)
