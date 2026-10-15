"""Water-balance generation doors and bitwise release comparisons."""
import dataclasses
import os
from pathlib import Path
import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_urban_config
from gpuwm.core.urban_ucm import water_balance_params
from gpuwm.io.restart import (_configuration_digest_values, _configuration_fingerprint,
                              _require_config_match)
from gpuwm.verify.urban_ucm_oracle import load_variant, port_params


def cfg(**kw):
    base = dict(nx=8,ny=8,nz=10,dx=3000.,dy=3000.,ztop=20000.,dt=10.,
                run_seconds=60.,sf_urban_physics=1,sf_surface_physics=2,
                sf_sfclay_physics=1,bl_pbl_physics=1)
    base.update(kw)
    return RunConfig(**base)


def test_generic_default_and_explicit_generation_parameter_words():
    _,table,switches = load_variant("ucm-nlcd-imp2")
    p = port_params(table,switches)
    assert cfg().urban_water_balance == "wrf_480"
    a = water_balance_params(p)
    b = water_balance_params(p,"wrf_471")
    assert a.switches[-1] == 1 and b.switches[-1] == 0
    assert a.switches[:-1].tobytes() == b.switches[:-1].tobytes()
    assert a.table.tobytes() == b.table.tobytes()
    assert a.globals_.tobytes() == b.globals_.tobytes()
    assert water_balance_params(p,"wrf_471") is b


def test_unknown_generation_refuses_named_breakage():
    with pytest.raises(ValueError,match="weak rainfall suppresses evaporation"):
        validate_urban_config(cfg(urban_water_balance="missing"))


def test_full_canopy_preserves_tree_switches_when_binding_water():
    from gpuwm.core.urban_tables import load_urban_params
    from gpuwm.core.urban_ucm import pack_params
    p = pack_params(load_urban_params(1,0,generation="wrf_480"))
    old = water_balance_params(p,"wrf_471")
    new = water_balance_params(p)
    assert p.generation == old.generation == new.generation == "wrf_480"
    assert len(old.switches) == len(new.switches) == 14
    assert old.switches[:-1].tobytes() == new.switches[:-1].tobytes()
    assert old.switches[-1] == 0 and new.switches[-1] == 1


def test_generation_flip_and_old_omitted_key_refuse_restart(tmp_path):
    old = cfg(urban_water_balance="wrf_471")
    new = cfg()
    assert _configuration_fingerprint(old) != _configuration_fingerprint(new)
    header = dataclasses.asdict(old)
    header.pop("urban_water_balance")
    assert _configuration_digest_values(header) == _configuration_digest_values(dataclasses.asdict(old))
    _require_config_match(header,old,tmp_path / "restart")
    with pytest.raises(ValueError,match="urban_water_balance"):
        _require_config_match(header,new,tmp_path / "restart")
    with pytest.raises(ValueError,match="urban_water_balance"):
        _require_config_match(dataclasses.asdict(new),old,tmp_path / "restart")


@pytest.mark.parametrize("option",[0,2,3])
def test_inactive_generation_does_not_change_checkpoint_digest(option):
    old = cfg(sf_urban_physics=option,urban_water_balance="wrf_471")
    new = cfg(sf_urban_physics=option)
    assert _configuration_fingerprint(old) == _configuration_fingerprint(new)


@pytest.mark.parametrize("generation",["wrf_480","wrf_471"])
def test_real_namelist_import_selects_generation(tmp_path,generation):
    from test_urban_registry import _generic_hierarchy_pair
    from gpuwm.namelist_import import import_namelists
    wps,inp = _generic_hierarchy_pair(tmp_path,2)
    text = inp.read_text().replace("&physics", "&physics\n sf_urban_physics=1,\n")
    if generation == "wrf_471":
        text = text.replace("&physics", "&physics\n urban_water_balance='wrf_471',")
    inp.write_text(text)
    toml,_ = import_namelists(wps,inp,name="urban-generation")
    assert f'urban_water_balance = "{generation}"' in toml


@pytest.mark.parametrize("dx_count,dy_count",[(1,1),(1,3),(2,1),(2,2),(3,3)])
def test_public_style_root_spacing_derives_unsupplied_children(tmp_path,dx_count,dy_count):
    import re
    from test_urban_registry import _generic_hierarchy_pair
    from gpuwm.namelist_import import import_namelists
    from gpuwm.experiment import load_experiment
    wps,inp = _generic_hierarchy_pair(tmp_path,3)
    text = inp.read_text().replace("&physics", "&physics\n sf_urban_physics=1,\n")
    for key,count in (("dx",dx_count),("dy",dy_count)):
        values = [12000.,4000.,4000.][:count]
        text = re.sub(rf"(?m)^ {key} = .*",f" {key} = {', '.join(map(str,values))},",text)
    inp.write_text(text)
    toml,_ = import_namelists(wps,inp,name="public-spacing")
    out = tmp_path/"experiment.toml"
    out.write_text(toml)
    experiment = load_experiment(out)
    assert [d.run.dx for d in experiment.domains] == [12000.,4000.,4000.]
    assert all(d.run.urban_water_balance == "wrf_480" for d in experiment.domains)


@pytest.mark.parametrize("key",["dx","dy"])
def test_explicit_wrong_child_spacing_still_refuses(tmp_path,key):
    import re
    from test_urban_registry import _generic_hierarchy_pair
    from gpuwm.namelist_import import import_namelists
    wps,inp = _generic_hierarchy_pair(tmp_path,2)
    text = re.sub(rf"(?m)^ {key} = .*",f" {key} = 12000., 12000.,",inp.read_text())
    inp.write_text(text)
    with pytest.raises(ValueError,match="contradicts the parent chain"):
        import_namelists(wps,inp)


def test_planted_bit_is_rejected_by_full_word_grader():
    from tools.urban_water_balance_oracle.check import compare
    want = {"after:drelr":np.asarray([0.0001],np.float32)}
    got = {k:v.copy() for k,v in want.items()}
    got["after:drelr"].view(np.uint32)[0] ^= np.uint32(1)
    assert compare(got,want)["after:drelr"]["count"] == 1


def test_compiled_release_oracle_when_staged():
    root = os.environ.get("GPUWM_URBAN_WATER_ORACLE")
    if not root:
        pytest.skip("stage unmodified WRF sources with tools/urban_water_balance_oracle")
    import subprocess
    import sys
    subprocess.run([sys.executable,"tools/urban_water_balance_oracle/terms.py",root],check=True)
    for tag,generation in (("v4.7.1","wrf_471"),):
        subprocess.run([sys.executable,"tools/urban_water_balance_oracle/check.py",
                        str(Path(root)/f"oracle-{tag}"/"fixture"),
                        str(Path(root)/f"test-{tag}.json"),"--host",
                        str(Path(root)/"ucm.so"),"--generation",generation],check=True)
