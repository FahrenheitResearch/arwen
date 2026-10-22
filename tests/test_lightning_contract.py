"""Admission, clock and checkpoint checks for the WRF lightning port."""
import dataclasses
from pathlib import Path
import pytest

def test_namelist_defaults_are_executable_config():
    from gpuwm.config import RunConfig
    from gpuwm.lightning_contract import LIGHTNING_DEFAULTS, LIGHTNING_OPTIONS
    fields={f.name:f.default for f in dataclasses.fields(RunConfig)}
    assert all(fields[k] == v for k,v in LIGHTNING_DEFAULTS.items())
    assert set(LIGHTNING_OPTIONS) == {0,1,2,3,11}

def test_no_ftz_no_fma_shared_compile_route():
    from gpuwm.core.kernels import module_options, module_source
    assert module_options("lightning") == ("-std=c++17","--ftz=false","--fmad=false")
    assert "lightning_patch" in module_source("lightning")


def test_missing_cumulus_top_is_refused():
    from gpuwm.lightning_contract import validate_lightning
    from types import SimpleNamespace
    from gpuwm.lightning_contract import LIGHTNING_DEFAULTS
    cfg=SimpleNamespace(**{**LIGHTNING_DEFAULTS,"lightning_option":11},cu_physics=1,mp_physics=10,moist=True,dt=10.)
    with pytest.raises(ValueError,match="cloud top"):
        validate_lightning(cfg)


def test_carried_lightning_is_restart_state():
    from gpuwm.io.restart import classify_scratch_slot
    for name in ("ic_flashcount","cg_flashcount","ic_flashrate","cg_flashrate","ltngacttime","iccg_in_num","iccg_in_den"):
        assert classify_scratch_slot("ltng_"+name) == "serialize"

def test_invalid_ratios_and_lpi_graupel():
    from gpuwm.lightning_contract import validate_lightning,LIGHTNING_DEFAULTS
    from types import SimpleNamespace
    for changes,message in [({"iccg_prescribed_den":0.},"0/0"),({"iccg_prescribed_num":-1.},"-1"),({"lightning_option":3,"mp_physics":1},"graupel")]:
        values={**LIGHTNING_DEFAULTS,"lightning_option":1,"cu_physics":0,"mp_physics":10,"moist":True,"dt":10.,"do_radar_ref":1,**changes}
        with pytest.raises(ValueError,match=message):
            validate_lightning(SimpleNamespace(**values))

def test_clock_inclusive_edges_and_fp32_accumulation():
    from gpuwm.core.lightning import next_activation
    assert next_activation(20.,20.,10.,15.) == 35.
    assert next_activation(20.,0.,10.,15.) is None
    assert next_activation(20.,10.,10.,15.) == 35.
    assert next_activation(20.,10.000001,9.999998,15.) is None

def test_registry_rows_are_generated_and_state_flags_match_wrf():
    import json
    from gpuwm.lightning_contract import LIGHTNING_OPTIONS
    root=Path(__file__).resolve().parents[1]
    registry=json.loads((root/"gpuwm/physics_registry_v2.json").read_text())
    rows=registry["diagnostic_packages"]["lightning_option"]
    assert set(rows)=={str(n) for n in LIGHTNING_OPTIONS}
    assert rows["11"]["requires_cloud_top"]
    state=registry["diagnostic_state"]["lightning"]
    assert all(v["restart"] for v in state.values())
    assert state["lpi"]["units"]=="m^2 s-2"
    assert not state["ic_flashrate"]["history"]

def test_history_shapes_publish_requested_flash_rates_and_lpi():
    from gpuwm.io.history_layout import produced_history_shapes
    from gpuwm.config import RunConfig
    cfg=RunConfig(nx=6,ny=4,nz=5,dx=2000.,dy=2000.,ztop=8000.,dt=10.,run_seconds=0.,
                  lightning_option=3,mp_physics=10,moist=True)
    shapes=produced_history_shapes(cfg)
    for name in ("IC_FLASHRATE","CG_FLASHRATE","IC_FLASHCOUNT","CG_FLASHCOUNT","LPI"):
        assert shapes[name]==(4,6)

def test_bare_namelist_import_reaches_lightning(tmp_path):
    from test_namelist_import import _generic_hierarchy_pair
    from gpuwm.namelist_import import import_namelists
    from gpuwm.experiment import load_experiment
    wps,inp=_generic_hierarchy_pair(tmp_path,1)
    text=inp.read_text().replace("&physics","&physics\n lightning_option=1, lightning_dt=15., do_radar_ref=1, iccg_method=4,")
    inp.write_text(text)
    toml,_=import_namelists(wps,inp,name="lightning")
    target=tmp_path/"lightning.toml";target.write_text(toml)
    run=load_experiment(target).root.run
    assert (run.lightning_option,run.lightning_dt,run.iccg_method,run.do_radar_ref)==(1,15.,4,1)

def test_lightning_checkpoint_roundtrip_and_identity(tmp_path,monkeypatch):
    from test_restart import _cfg,_shim_state,_fill_setup
    from gpuwm.core.lightning import initialize_lightning
    from gpuwm.io.restart import write_restart,restore_restart,_configuration_digest_values
    from gpuwm.lightning_contract import LIGHTNING_DEFAULTS
    import numpy as np
    cfg=_cfg(lightning_option=3,moist=True,mp_physics=10)
    state=_shim_state(cfg,monkeypatch);_fill_setup(state)
    initialize_lightning(state,cfg)
    for name,array in state._scratch.items():
        if name.startswith("ltng_") and array.dtype==np.float32:
            array[...] = np.arange(array.size,dtype=np.float32).reshape(array.shape)*np.float32(.125)
    path=write_restart(tmp_path/"lightning.npz",state,cfg)
    restored=_shim_state(cfg,monkeypatch);_fill_setup(restored)
    initialize_lightning(restored,cfg)
    restore_restart(path,restored,cfg)
    for name,array in state._scratch.items():
        if name.startswith("ltng_"):
            np.testing.assert_array_equal(array.view(np.uint32),restored._scratch[name].view(np.uint32))
    off=dataclasses.asdict(_cfg())
    old={k:v for k,v in off.items() if k not in LIGHTNING_DEFAULTS}
    assert _configuration_digest_values(off)==_configuration_digest_values(old)
    assert _configuration_digest_values(dataclasses.asdict(cfg))!=_configuration_digest_values(off)

def test_oracle_comparator_catches_one_bit():
    import numpy as np
    from tools.lightning_wrf461_oracle.validate_lightning import check_words
    reference=np.array([0.,.125,1.,100.],np.float32).view(np.uint32)
    planted=reference.copy();planted[2]^=np.uint32(1)
    assert check_words(planted,reference)==(1,1)

def test_off_diagnostic_controls_do_not_move_checkpoint_identity():
    from gpuwm.config import RunConfig
    from gpuwm.io.restart import _configuration_digest_values
    cfg=RunConfig(nx=6,ny=4,nz=5,dx=2000.,dy=2000.,ztop=8000.,dt=10.,run_seconds=0.)
    moved=dataclasses.replace(cfg,do_radar_ref=1,lightning_dt=15.,iccg_method=2)
    assert _configuration_digest_values(dataclasses.asdict(cfg)) == _configuration_digest_values(dataclasses.asdict(moved))


# Producer tables (2.9.0). LPI reads state.qg and PR92z reads the cumulus
# scheme's ltng_ktop_deep. A scheme listed as a graupel producer without a
# qg field would feed LPI a zero ice mixture, and a scheme that carries qg
# but is left out is refused with no breakage to name; so the table is
# checked against the state each scheme allocates, not against a list.
def _lightning_cfg(**changes):
    from types import SimpleNamespace
    from gpuwm.lightning_contract import LIGHTNING_DEFAULTS
    values = {**LIGHTNING_DEFAULTS, "cu_physics": 0, "mp_physics": 10,
              "moist": True, "dt": 10., "do_radar_ref": 1, **changes}
    return SimpleNamespace(**values)


def test_graupel_providers_are_exactly_the_schemes_whose_state_carries_qg(monkeypatch):
    from test_restart import _cfg, _shim_state
    from gpuwm.config import MP_PHYSICS_ACCEPTED
    from gpuwm.core.moist import moist_species
    from gpuwm.lightning_contract import GRAUPEL_PROVIDERS
    carrying = set()
    for mp in MP_PHYSICS_ACCEPTED:
        if mp == 0:
            continue
        state = _shim_state(_cfg(moist=True, mp_physics=mp), monkeypatch)
        if getattr(state, "qg", None) is None:
            continue
        # Transported with the scheme's other species, so the graupel the
        # scheme writes is the graupel LPI reads.
        assert "qg" in moist_species(state), mp
        carrying.add(mp)
    assert set(GRAUPEL_PROVIDERS) == carrying


def test_graupel_providers_carry_qg_in_their_wrf_registry_package():
    from gpuwm.lightning_contract import GRAUPEL_PROVIDERS
    from gpuwm.wrf_physics_inventory import (
        stock_wrf_physics_inventory, supported_stock_wrf_mp_physics)
    for mp in supported_stock_wrf_mp_physics():
        names = {field.registry_name
                 for field in stock_wrf_physics_inventory(mp).wrfinput_fields}
        assert ("qg" in names) == (mp in GRAUPEL_PROVIDERS), mp


@pytest.mark.parametrize("mp", (18,))
def test_lpi_admits_the_graupel_schemes_added_in_2_9_0(mp):
    from gpuwm.lightning_contract import validate_lightning
    validate_lightning(_lightning_cfg(lightning_option=3, mp_physics=mp))


@pytest.mark.parametrize("mp", (50,))
def test_lpi_refuses_a_scheme_without_graupel_by_its_number(mp):
    from gpuwm.lightning_contract import validate_lightning
    with pytest.raises(ValueError, match=rf"mp_physics={mp} does not publish a separate qg"):
        validate_lightning(_lightning_cfg(lightning_option=3, mp_physics=mp))


@pytest.mark.parametrize("cu", (3, 5))
def test_pr92_cloud_top_admits_grell_freitas_and_grell_3d_as_wrf_does(cu):
    # WRF 4.6.1 lightning_init admits GD, G3 and GF for lightning_option=11;
    # GD is not ported, G3 (cu_physics=5) publishes ktop_deep like GF.
    from gpuwm.lightning_contract import validate_lightning
    validate_lightning(_lightning_cfg(lightning_option=11, cu_physics=cu))


@pytest.mark.parametrize("cu", (1,))
def test_pr92_cloud_top_refuses_a_scheme_without_ktop_deep_by_its_number(cu):
    from gpuwm.lightning_contract import validate_lightning
    with pytest.raises(ValueError, match=rf"cu_physics={cu} does not publish ltng_ktop_deep"):
        validate_lightning(_lightning_cfg(lightning_option=11, cu_physics=cu))
