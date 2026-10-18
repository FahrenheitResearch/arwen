"""The new selector is table-admitted and carries a distinct scalar."""
import json
from pathlib import Path
import numpy as np
import pytest
from gpuwm.config import RunConfig,CCPP_TKE_EDMF_PBL_SCHEME,validate_run_config
from gpuwm.core.state import DomainState
from gpuwm.core.device_inventory import state_array_shapes
from gpuwm.core.physics_inventory import PHYSICS_SLOT_DISPATCH
from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS
from gpuwm.io.restart import STATE_REBUILT_ATTRS

def cfg(**overrides):
    values=dict(nx=16,ny=16,nz=24,dx=3000.,dy=3000.,dt=1.,ztop=15000.,moist=True,
        mp_physics=6,bl_pbl_physics=901,sf_sfclay_physics=91,
        sf_surface_physics=2,bldt=0.,km_opt=4,run_seconds=4.)
    values.update(overrides)
    return RunConfig(**values)

def test_distinct_non_wrf_selector_and_dispatch():
    assert CCPP_TKE_EDMF_PBL_SCHEME==901
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][901]=="_run_satmedmf"
    document=json.loads((Path(__file__).resolve().parents[1]/"gpuwm/physics_registry_v2.json").read_text())
    row=document["components"]["pbl"]["options"]["ccpp-tke-edmf"]
    assert row["selectors"]=={"bl_pbl_physics":901}
    assert row["wrf_counterpart"]["exists"] is False
    assert row["constraints"]["required_settings"]["bldt"]==0

def test_carrier_allocated_and_inventoried():
    c=cfg();state=DomainState(c,array_module=np)
    assert state.tke_edmf.shape==(24,16,16)
    assert np.all(state.tke_edmf==np.float32(0.2))
    assert state_array_shapes(c)["tke_edmf"]==state.tke_edmf.shape
    assert "tke_edmf" in STATE_SERIALIZED_ATTRS
    assert "tke_edmf0" in STATE_REBUILT_ATTRS

def test_other_scheme_has_no_extra_carrier():
    state=DomainState(cfg(bl_pbl_physics=1),array_module=np)
    assert not hasattr(state,"tke_edmf")
    assert "tke_edmf" not in state_array_shapes(cfg(bl_pbl_physics=1))

@pytest.mark.parametrize("change,reason",[
    ({"moist":False,"mp_physics":0},"moist=true"),
    ({"bldt":1.},"positive cadence"),
    ({"sf_sfclay_physics":5},"fm/fh"),
    ({"nz":4},"mass-flux searches")])
def test_refusal_names_breakage(change,reason):
    with pytest.raises(ValueError,match=reason):
        validate_run_config(cfg(**change))

def test_bare_namelist_keeps_the_non_wrf_selector(tmp_path):
    from gpuwm.namelist_import import import_namelists
    root=Path(__file__).resolve().parents[1]
    src=(root/"configs/gfs_wrf_direct_proof.namelist.input").read_text()
    src=src.replace("bl_pbl_physics                      = 1","bl_pbl_physics                      = 901")
    nml=tmp_path/"namelist.input";nml.write_text(src)
    text,report=import_namelists(root/"configs/gfs_wrf_direct_proof.namelist.wps",nml)
    assert "bl_pbl_physics = 901" in text

def test_upstream_port_is_not_labelled_as_original():
    from gpuwm.physics_compat import _experimental_reason
    row={"wrf_counterpart":{"exists":False,"implementation_origin":"upstream-port"}}
    assert _experimental_reason(row)=="experimental, upstream port with no WRF counterpart"

def test_implementation_origin_does_not_change_physics_identity():
    from copy import deepcopy
    from gpuwm.physics_registry import physics_registry,strip_registry_documentation,canonical_sha256
    row=deepcopy(physics_registry()["components"]["pbl"]["options"]["ccpp-tke-edmf"])
    before=deepcopy(row)
    before["wrf_counterpart"].pop("implementation_origin",None)
    assert canonical_sha256(strip_registry_documentation(before))==canonical_sha256(strip_registry_documentation(row))


@pytest.mark.parametrize("cumulus", ["off", "grell-freitas"])
def test_registry_and_runtime_admit_default_floating_pbl_cadence(cumulus):
    from test_authority_agreement import (
        _permissive_registry, _single_domain_plan, _PERMISSIVE_RUNNER,
        _base_template_id, _config_refusal)
    from gpuwm.physics_registry import validate_physics_plan
    registry = _permissive_registry()
    components = dict(cumulus=cumulus, land_surface="noah",
        microphysics="kessler-mp1", pbl="ccpp-tke-edmf", radiation="off",
        surface_layer="classic-mm5", turbulence="constant-k", urban="none")
    plan = _single_domain_plan(registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry), components=components)
    report = validate_physics_plan(plan, registry=registry)
    settings = report["resolved_domains"][0]["settings"]
    assert type(settings["bldt"]) is float
    assert settings["bldt"] == 0.0
    assert report["launchable"], report["errors"]
    assert _config_refusal(settings, nested=False) is None


def test_measured_sm89_frame_is_priced_by_default():
    from gpuwm.core import preflight
    # GPU parity receipt: SM89, NVRTC 13.4.92, actual local_size_bytes=112.
    assert preflight.KERNEL_MAX_LOCAL_SIZE_BYTES["satmedmf"] >= 112


def test_measured_sm89_frame_keeps_its_platform_provenance():
    from gpuwm.core.kernel_frame_recordings import recording_for
    row = recording_for({"device_compute_capability": "89",
                         "nvrtc_build": "13.4.92"})
    assert row is not None
    assert row.frames.get("satmedmf") == 112
    assert not row.complete
