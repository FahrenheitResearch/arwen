"""Compiled WRF coordinate flux words and importer/operator selections."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig, validate_km_opt
from gpuwm.namelist_contract import _base_pair
from gpuwm.namelist_import import import_parsed_namelists

DATA=Path(__file__).parent/"data/wrf471_diff_opt1"


@pytest.mark.parametrize("km",[2,4])
@pytest.mark.parametrize("mix",[False,True])
def test_coordinate_diffusion_preserves_imported_selection(km,mix):
    wps,inp=deepcopy(_base_pair())
    inp["dynamics"]["diff_opt"]=[1,1]
    inp["dynamics"]["km_opt"]=[km,km]
    inp["dynamics"]["mix_full_fields"]=[mix,mix]
    if km==2: inp["physics"]["bl_pbl_physics"]=[0,0]
    text,report=import_parsed_namelists(wps,inp,wps_path="namelist.wps",input_path="namelist.input")
    config=tomllib.loads(text)
    assert config["shared"]["diff_opt"]==1
    assert config["shared"]["mix_full_fields"] is mix
    assert not [v for v in report.substitutions if v.key=="mix_full_fields"]


@pytest.mark.parametrize("km",[2,4])
@pytest.mark.parametrize("mix",[False,True])
def test_coordinate_configuration_admits_wrf_logicals(km,mix):
    validate_km_opt(RunConfig(nx=12,ny=9,nz=6,dx=800.,dy=1400.,ztop=3000.,
                             dt=1.,run_seconds=30.,km_opt=km,diff_opt=1,
                             mix_full_fields=mix,bl_pbl_physics=0))


@pytest.mark.parametrize("km,pbl",[(1,0),(1,1),(2,0),(3,0),(4,0),(4,1)])
def test_metric_diffusion_admits_perturbation_mixing(km,pbl):
    # The refusal that stood here retired: WRF's perturbation branch
    # subtracts base-state profiles real.exe leaves at zero, so the metric
    # operator is WRF's operator at either value for a real-data run
    # (gpuwm/config.py, validate_km_opt).  Operational HRRR runs false.
    cfg=RunConfig(nx=12,ny=9,nz=6,dx=800.,dy=1400.,ztop=3000.,dt=1.,
                  run_seconds=30.,km_opt=km,bl_pbl_physics=pbl,
                  mix_full_fields=False)
    validate_km_opt(cfg)


def test_mixed_operator_columns_import_as_declared():
    wps,inp=deepcopy(_base_pair())
    inp["dynamics"]["diff_opt"]=[2,1]
    inp["dynamics"]["mix_full_fields"]=[False,False]
    text,report=import_parsed_namelists(wps,inp,wps_path="namelist.wps",input_path="namelist.input")
    config=tomllib.loads(text)
    assert config["shared"]["diff_opt"]==2
    assert config["shared"]["mix_full_fields"] is True
    assert config["domain"][1]["diff_opt"]==1
    assert config["domain"][1]["mix_full_fields"] is False
    assert [s for s in report.substitutions if s.key=="mix_full_fields"]


@pytest.mark.parametrize("explicit_sentinel", [False, True])
def test_unset_diffusion_children_inherit_root_after_a_mixed_column(explicit_sentinel):
    wps, inp = deepcopy(_base_pair())
    for document in (wps, inp):
        for section in document.values():
            for key, values in section.items():
                if len(values) == 2:
                    values.append(values[-1])
    wps["share"]["max_dom"] = [3]
    inp["domains"]["max_dom"] = [3]
    inp["domains"]["grid_id"] = [1, 2, 3]
    inp["dynamics"]["diff_opt"] = [1, 2] + ([-1] if explicit_sentinel else [])
    inp["dynamics"]["km_opt"] = [2, 4] + ([-1] if explicit_sentinel else [])
    inp["physics"]["bl_pbl_physics"] = [0, 0, 0]
    text, report = import_parsed_namelists(
        wps, inp, wps_path="namelist.wps", input_path="namelist.input")
    config = tomllib.loads(text)
    assert config["shared"]["diff_opt"] == 1
    assert config["shared"]["km_opt"] == 2
    assert config["domain"][1]["diff_opt"] == 2
    assert config["domain"][1]["km_opt"] == 4
    assert config["domain"][2].get("diff_opt", config["shared"]["diff_opt"]) == 1
    assert config["domain"][2].get("km_opt", config["shared"]["km_opt"]) == 2
    assert next(entry for entry in report.fixed if entry.key == "diff_opt").fixed_value == (1, 2, 1)


@pytest.mark.parametrize("root", [None, [-1]])
def test_unset_root_diffusion_names_the_native_breakage_and_remedy(root):
    wps, inp = deepcopy(_base_pair())
    if root is None:
        inp["dynamics"].pop("diff_opt")
    else:
        inp["dynamics"]["diff_opt"] = root
    with pytest.raises(ValueError, match="selects no diffusion operator") as failure:
        import_parsed_namelists(
            wps, inp, wps_path="namelist.wps", input_path="namelist.input")
    assert "Set the root diff_opt to 1" in str(failure.value)


@pytest.mark.parametrize("mix",[False,True])
def test_coordinate_tke_allows_wrf_pbl_pairing(mix):
    from dataclasses import replace
    from tools.wrf_diffopt1_oracle.model_case import configuration
    cfg=replace(configuration(km=2,mix=mix),bl_pbl_physics=1)
    validate_km_opt(cfg)
    with pytest.raises(ValueError,match="bl_pbl_physics=0"):
        validate_km_opt(replace(cfg,diff_opt=2))
    wps,inp=deepcopy(_base_pair())
    inp["dynamics"]["diff_opt"]=[1,1]
    inp["dynamics"]["km_opt"]=[2,2]
    inp["dynamics"]["mix_full_fields"]=[mix,mix]
    import_parsed_namelists(wps,inp,wps_path="namelist.wps",input_path="namelist.input")


def test_compiled_source_and_fixture_bytes_are_pinned():
    metadata=json.loads((DATA/"wrf471.json").read_text())
    assert hashlib.sha256((DATA/"wrf471.npz").read_bytes()).hexdigest()==metadata["archive_sha256"]
    assert metadata["build"]["wrf_release"]=="4.7.1"
    assert metadata["build"]["wrf_commit"]=="f52c197ed39d12e087d02c50f412d90d418f6186"
    assert metadata["build"]["source_sha256"]=="a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54"
    assert set(metadata["build"]["routines"])=={"cal_deform_and_div","calculate_N2","smag2d_km","tke_km","calc_l_scale","pthl","pu","compute_diff_metrics"}


def test_coordinate_reference_has_priced_persistent_storage():
    from tools.wrf_diffopt1_oracle.model_case import configuration
    from gpuwm.core.preflight import scratch_slot_registry,scratch_slot_lifetime
    from gpuwm.io.restart import classify_scratch_slot
    cfg=configuration()
    shapes=scratch_slot_registry(cfg)
    for slot in ("diff1_theta_initial","diff1_theta_work"):
        assert shapes[slot]==(cfg.nz,cfg.ny,cfg.nx)
    assert classify_scratch_slot("diff1_theta_initial")=="serialize"
    assert classify_scratch_slot("diff1_theta_work")=="rebuild"
    assert scratch_slot_lifetime("diff1_theta_initial").kind=="carrying"
    assert scratch_slot_lifetime("diff1_theta_work").arena_eligible
    assert "diff1_theta_initial" not in scratch_slot_registry(configuration(diff=2))


@pytest.mark.parametrize("value",[True,False,1.,2.,0,3])
def test_coordinate_selector_refuses_invalid_types_or_operators(value):
    from dataclasses import replace
    from tools.wrf_diffopt1_oracle.model_case import configuration
    with pytest.raises(ValueError,match="diff_opt must"):
        validate_km_opt(replace(configuration(),diff_opt=value))


def test_metric_defaults_preserve_checkpoint_identity_payloads():
    from dataclasses import asdict
    from gpuwm.checkpoint_identity import drop_default_diffusion_selectors
    from gpuwm.io.restart import _configuration_digest_values,_mosaic_checkpoint_config
    from tools.wrf_diffopt1_oracle.model_case import configuration
    current=asdict(configuration(diff=2,mix=True))
    old={key:value for key,value in current.items() if key not in ("diff_opt","mix_full_fields")}
    assert _configuration_digest_values(current)==_configuration_digest_values(old)
    assert _mosaic_checkpoint_config(current)==_mosaic_checkpoint_config(old)
    for diff,mix in ((1,True),(1,False),(2,False)):
        altered=dict(current,diff_opt=diff,mix_full_fields=mix)
        drop_default_diffusion_selectors(altered)
        assert altered["diff_opt"]==diff
        assert altered["mix_full_fields"] is mix
