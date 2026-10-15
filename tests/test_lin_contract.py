"""Device-free admission and allocation checks for mp_physics=2."""
from pathlib import Path
import json
import numpy as np
from gpuwm.config import RunConfig, validate_run_config, MP_PHYSICS_ACCEPTED
from gpuwm.namelist_import import _MP_MAP
from gpuwm.physics_registry import implemented_selector_values


def config():
    return RunConfig(nx=8,ny=6,nz=40,dx=12000.,dy=12000.,ztop=18000.,
                     dt=30.,run_seconds=600.,moist=True,mp_physics=2,
                     ra_lw_physics=0,ra_sw_physics=0)


def test_lin_selector_is_native_and_default_reachable():
    cfg=config()
    validate_run_config(cfg)
    assert 2 in MP_PHYSICS_ACCEPTED
    assert _MP_MAP[2] == (2,"Purdue Lin","Purdue Lin")
    assert implemented_selector_values("microphysics")[2] == "lin-mp2"


def test_lin_masses_surfaces_and_restart_identity_are_declared():
    cfg=config()
    from gpuwm.core import preflight
    from gpuwm.core.physics_inventory import microphysics_scratch_slots
    from gpuwm.checkpoint_identity import MICROPHYSICS_ALGORITHM_IDENTITIES
    shapes=preflight.state_array_shapes(cfg)
    assert {"qv","qc","qr","qi","qs","qg"} <= shapes.keys()
    assert not {"nc","nr","ni","ns","ng","qndrop"} & shapes.keys()
    scratch=preflight.scratch_slot_registry(cfg)
    assert {"lin_theta","lin_rho","lin_pii","lin_dz","lin_z8w","lin_z"} <= scratch.keys()
    assert len(microphysics_scratch_slots(2)) == 7
    assert MICROPHYSICS_ALGORITHM_IDENTITIES[2] == "lin-wrf461-six-class-v1"


def test_lin_oracle_covers_72_columns_and_gradual_underflow():
    fx=np.load(Path(__file__).parent/'data/oracles/lin/lin-wrf461-columns.npz')
    assert fx['in_th'].shape == (40,72)
    assert len(set(fx['regimes'])) == 8
    prov=json.loads(str(fx['provenance']))
    assert prov['wrf_commit'] == 'd66e442fccc04111067e29274c9f9eaccc3cef28'
    q=fx['in_qc']; tiny=np.abs(q[q!=0])
    assert (tiny<np.finfo(np.float32).tiny).any()
    assert (q.view(np.uint32)==0x80000000).any()
    assert 'out_refl_10cm' in fx


def test_lin_analyzed_mass_is_retained_without_borrowed_number_fields():
    from gpuwm.ingest.real import (HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE,
        HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS,HRRR_ANALYZED_HYDROMETEORS)
    from gpuwm.ingest.microphysics_cold_start import source_absent_microphysics
    assert 2 in HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS
    assert HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE[2]["retained"] == HRRR_ANALYZED_HYDROMETEORS
    assert source_absent_microphysics(config()) == ((),{})


def test_lin_native_radiation_and_vertical_bound_contract():
    from gpuwm.physics_registry import consumer_row_for_selector
    from gpuwm.core.kf import kf_phase_mode_for_microphysics, KFPhaseMode
    from gpuwm.physics_vertical_contract import LIN_VERTICAL_LEVEL_BOUNDS
    assert LIN_VERTICAL_LEVEL_BOUNDS == (2,512)
    assert kf_phase_mode_for_microphysics(2) == KFPhaseMode.SEPARATE_ICE_SNOW
    row=consumer_row_for_selector("microphysics","cloud_optics",2)
    assert row["legacy_declares_radii"] is False
    assert row["rte_rrtmgp_coupling"] is None
    assert "Lin supplies no effective radii" in row["rte_rrtmgp_refusal"]


def test_lin_boundaries_and_direct_wrf_input_carry_all_six_masses():
    import pytest
    from gpuwm.core.nest_fields import nest_field_kinds
    from gpuwm.ingest.wrfinput import active_moisture_inventory
    assert {"qv","qc","qr","qi","qs","qg"} <= set(nest_field_kinds(config()))
    required,allowed=active_moisture_inventory(config())
    assert {"QVAPOR","QCLOUD","QRAIN","QICE","QSNOW","QGRAUP"} <= required
    assert required == allowed
    from gpuwm.core.mynn_pbl_runtime import MYNN_SNOW_MICROPHYSICS
    assert 2 in MYNN_SNOW_MICROPHYSICS
    from dataclasses import replace
    cfg=replace(config(),moist=False)
    with pytest.raises(ValueError,match="requires moist=true"):
        validate_run_config(cfg)


def test_lin_work_buffers_are_rebuilt_and_surface_accumulators_are_serialized():
    import pytest
    from gpuwm.io.restart import classify_scratch_slot, RestartManifestError
    for slot in ("lin_theta","lin_rho","lin_pii","lin_z","lin_dz","lin_z8w"):
        assert classify_scratch_slot(slot) == "rebuild"
    for slot in ("mp_rainnc","mp_snownc","mp_graupelnc"):
        assert classify_scratch_slot(slot) == "serialize"
    with pytest.raises(RestartManifestError):
        classify_scratch_slot("lin_undeclared_accumulator")
