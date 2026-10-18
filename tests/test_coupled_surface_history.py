"""The normal history inventory carries evolving surface and solar fields."""
from dataclasses import replace
import pytest

from gpuwm.config import RunConfig
from gpuwm.io.history_layout import produced_history_shapes
from gpuwm.io.wrf_output_schema import (CARRIED_SURFACE_OUTPUT_FIELDS,
                                       NOAH_OUTPUT_FIELDS, SCHEME_OUTPUT_FIELDS)


@pytest.mark.parametrize("surface", [2, 4])
def test_default_history_has_surface_carriers_without_energy_opt_in(surface):
    cfg = RunConfig(nx=8, ny=6, nz=12, dx=3000., dy=3000., ztop=12000.,
                    dt=6., run_seconds=60., moist=True, mp_physics=6,
                    bl_pbl_physics=1, sf_sfclay_physics=91, sf_surface_physics=surface,
                    ra_lw_physics=4, ra_sw_physics=4,
                    ra_rrtmg_variant="rrtmg_legacy", surface_energy_diag=False)
    shapes = produced_history_shapes(cfg)
    energy_shapes = produced_history_shapes(replace(cfg, surface_energy_diag=True))
    for field in CARRIED_SURFACE_OUTPUT_FIELDS.values():
        # Both selected legacy wrappers now retain the actual solar angle.
        assert shapes[field.netcdf_name] == (6, 8)
        assert shapes[field.netcdf_name] == energy_shapes[field.netcdf_name]


def test_surface_carriers_keep_stock_registry_schema():
    for key, field in CARRIED_SURFACE_OUTPUT_FIELDS.items():
        assert SCHEME_OUTPUT_FIELDS[key] == field
        assert field.dtype == "f4" and field.field_type == 104
        assert field.stagger == "" and field.wrf_history
        assert field.registry.startswith(("Registry.EM_COMMON:", "Registry.EM:"))


def test_noah_default_history_keeps_its_live_soil_runoff_and_energy_fields():
    cfg = RunConfig(nx=8, ny=6, nz=12, dx=3000., dy=3000., ztop=12000.,
                    dt=6., run_seconds=60., moist=True, mp_physics=6,
                    bl_pbl_physics=1, sf_sfclay_physics=91, sf_surface_physics=2)
    shapes = produced_history_shapes(cfg)
    for key, field in NOAH_OUTPUT_FIELDS.items():
        assert shapes[field.netcdf_name] == ((4,6,8) if key == "smcrel" else (6,8))
        assert SCHEME_OUTPUT_FIELDS[key] == field
    other = produced_history_shapes(replace(cfg,sf_surface_physics=4))
    assert all(f.netcdf_name not in other for f in NOAH_OUTPUT_FIELDS.values())


def test_relative_soil_moisture_uses_the_soil_axis():
    from types import SimpleNamespace
    from gpuwm.io.wrfout import WrfoutWriter
    writer = SimpleNamespace(nz=12,ny=6,nx=8,soil_layers=4)
    assert WrfoutWriter._dims_for(writer,"SMCREL",(4,6,8)) == (
        "Time","soil_layers_stag","south_north","west_east")


def test_frozen_precipitation_fraction_is_normal_history():
    cfg=RunConfig(nx=8,ny=6,nz=12,dx=3000.,dy=3000.,ztop=12000.,dt=15.,
                  run_seconds=60.,moist=True,mp_physics=6,
                  ra_physics=4,ra_rrtmg_variant="rrtmg_legacy")
    assert produced_history_shapes(cfg)["SR"]==(6,8)
    assert produced_history_shapes(cfg)["COSZEN"]==(6,8)
    assert produced_history_shapes(cfg)["THM"]==(12,6,8)
    for name in ("ACHFX","ACLHF","ACGRDFLX"):
        assert produced_history_shapes(cfg)[name]==(6,8)
    for name in ("SST","XICEM"):
        assert produced_history_shapes(cfg)[name]==(6,8)
    from gpuwm.wrf_exact import ENABLED
    for name in ("HFX_FORCE","LH_FORCE","TSK_FORCE","HFX_FORCE_TEND",
                 "LH_FORCE_TEND","TSK_FORCE_TEND"):
        # Strict-only (physics_inventory.scm_forcing_fields).
        if ENABLED:
            assert produced_history_shapes(cfg)[name]==()
        else:
            assert name not in produced_history_shapes(cfg)
    for name in ("SSTSK","REFD_MAX","PC","O3_GFS_DU"):
        assert produced_history_shapes(cfg)[name]==(6,8)


def test_hydrostatic_pressure_history_keeps_the_live_producer_array():
    from types import SimpleNamespace
    import numpy as np
    from gpuwm.io.history_layout import physics_history_fields
    cfg=RunConfig(nx=8,ny=6,nz=12,dx=3000.,dy=3000.,ztop=12000.,dt=15.,
                  run_seconds=60.,moist=True,mp_physics=6,
                  bl_pbl_physics=1,sf_sfclay_physics=91,sf_surface_physics=2)
    assert produced_history_shapes(cfg)["P_HYD"]==(12,6,8)
    from gpuwm.wrf_exact import ENABLED
    for name in ("RUBLTEN","RVBLTEN","RTHBLTEN","RQVBLTEN","RQCBLTEN"):
        # Strict-only (physics_inventory.history_tendency_rows).
        if ENABLED:
            assert produced_history_shapes(cfg)[name]==(12,6,8)
        else:
            assert name not in produced_history_shapes(cfg)
    from gpuwm.io.history_layout import _OUTPUT_FIELDS
    fields={name:np.zeros((6,8),np.float32) for name in _OUTPUT_FIELDS.values()}
    fields["p_hyd"]=np.full((12,6,8),80000.,np.float32)
    driver=SimpleNamespace(fields=fields,surface_energy_diag=False,sase_active=False,
                           hmix_k_diag=None,radiation_active=False,rainc=None,
                           mp_physics=0,_zero_accumulator=lambda:np.zeros((6,8),np.float32))
    frame=physics_history_fields(driver)
    assert frame["P_HYD"] is fields["p_hyd"]
    fields["p_hyd"][...]=79000.
    np.testing.assert_array_equal(frame["P_HYD"],79000.)


def test_native_grid_history_keeps_the_input_scalar_words():
    from types import SimpleNamespace
    import numpy as np
    from gpuwm.io.history_layout import wrf_file_grid_history_fields
    raw={"RESM":np.array(np.float32(.125),dtype=np.float64)}
    frame=wrf_file_grid_history_fields(SimpleNamespace(restored=SimpleNamespace(raw=raw)))
    assert frame["RESM"].shape==()
    assert frame["RESM"].dtype==np.float32
    assert frame["RESM"].item()==.125
