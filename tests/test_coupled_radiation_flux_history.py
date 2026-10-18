"""Default radiation history keeps actual boundary fluxes and their energy."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core.physics_inventory import radiation_boundary_flux_names
from gpuwm.core.preflight import physics_array_shapes
from gpuwm.io.history_layout import produced_history_shapes,physics_history_fields
from gpuwm.io.wrf_output_schema import RADIATION_BOUNDARY_OUTPUT_FIELDS


def config():
    return RunConfig(nx=8,ny=6,nz=12,dx=3000.,dy=3000.,ztop=12000.,dt=15.,
                     run_seconds=60.,ra_lw_physics=4,ra_sw_physics=4,
                     ra_rrtmg_variant="rrtmg_legacy",radt=1.,surface_energy_diag=False)


def test_boundary_history_is_default_and_all_carriers_are_priced():
    cfg=config();shapes=produced_history_shapes(cfg);allocated=physics_array_shapes(cfg)
    assert len(radiation_boundary_flux_names(cfg))==16
    for key,field in RADIATION_BOUNDARY_OUTPUT_FIELDS.items():
        assert shapes[field.netcdf_name]==(6,8)
        assert allocated["fields/"+key]==(6,8)
        assert field.wrf_history and field.stagger==""
    off=produced_history_shapes(replace(cfg,ra_lw_physics=0,ra_sw_physics=0))
    assert not set(f.netcdf_name for f in RADIATION_BOUNDARY_OUTPUT_FIELDS.values()) & off.keys()


def test_partial_spectra_publish_only_produced_fluxes():
    cfg=replace(config(),ra_sw_physics=0)
    names=radiation_boundary_flux_names(cfg)
    assert len(names)==8 and all(n.startswith("lw") for n in names)
    shapes=produced_history_shapes(cfg)
    assert "LWUPT" in shapes and "ACLWUPT" in shapes
    assert "SWUPT" not in shapes and "ACSWUPT" not in shapes


def test_combined_radiation_selector_keeps_the_same_boundary_inventory():
    cfg=replace(config(),ra_physics=4,ra_lw_physics=-1,ra_sw_physics=-1)
    assert radiation_boundary_flux_names(cfg)==radiation_boundary_flux_names(config())
    shapes=produced_history_shapes(cfg)
    assert all(f.netcdf_name in shapes for f in RADIATION_BOUNDARY_OUTPUT_FIELDS.values())


def test_default_rte_rrtmgp_publishes_the_same_rows_and_coszen():
    """2.9.2: the default RTE+RRTMGP radiation writes COSZEN, the sixteen
    top/bottom fluxes and their AC* accumulations, as legacy RRTMG does
    (they were missing from every default wrfout)."""
    cfg = replace(config(), ra_rrtmg_variant="rte-rrtmgp")
    assert RunConfig.ra_rrtmg_variant == "rte-rrtmgp"  # the default engine
    assert radiation_boundary_flux_names(cfg) == radiation_boundary_flux_names(config())
    shapes = produced_history_shapes(cfg)
    allocated = physics_array_shapes(cfg)
    for key, field in RADIATION_BOUNDARY_OUTPUT_FIELDS.items():
        assert shapes[field.netcdf_name] == (6, 8)
        assert allocated["fields/" + key] == (6, 8)
    assert shapes["COSZEN"] == (6, 8) and allocated["fields/coszen"] == (6, 8)
    lw_only = produced_history_shapes(replace(cfg, ra_sw_physics=0))
    assert "LWUPTC" in lw_only and "ACLWDNB" in lw_only
    assert "SWUPT" not in lw_only and "ACSWUPT" not in lw_only
