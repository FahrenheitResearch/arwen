"""Fork moisture output and the explicitly selected cloud conservation defects."""
from pathlib import Path
import csv
import dataclasses
import numpy as np
import pytest
from test_mynn_gsd41 import requires_gpu


def _fields():
    path = Path(__file__).resolve().parents[1] / 'gpuwm/data/mynn/oracle/transport-cloud-gsd41.csv'
    with path.open(newline='', encoding='ascii') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 * 12
    return {key: np.array([np.float32(row[key]) for row in rows]).reshape(12, 12)
            for key in rows[0] if key not in ('case','k')}


def _values(fields):
    from gpuwm.core.mynn_pbl import MYNN_TENDENCIES_LAYER_INPUTS, MYNN_TENDENCIES_INTERFACE_INPUTS
    shape = (12,12)
    values = {name:fields.get(name,np.zeros(shape,np.float32)) for name in MYNN_TENDENCIES_LAYER_INPUTS}
    values['thl'] = fields['thl0']
    values.update(qc=fields['qc'], qi=fields['qi'])
    for name in MYNN_TENDENCIES_INTERFACE_INPUTS:
        values[name] = (np.concatenate((fields[name][:,:1],fields[name+'_next']),axis=1)
                        if name in fields else np.zeros((12,13),np.float32))
    for name in ('delt','ust','flt','flqv','wspd'):
        values[name]=fields[name][:,0]
    values.update(psfc=np.full(12,95500.,np.float32),uoce=np.zeros(12,np.float32),
                  voce=np.zeros(12,np.float32),flqc=np.zeros(12,np.float32))
    return values


@requires_gpu
@pytest.mark.gpu
def test_as_written_fork_cloud_and_water_tendencies_match_unmodified_source():
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_tendencies_default_cuda
    f = _fields()
    out = mynn_tendencies_default_cuda(_values(f), bl_mynn_version='gsd_41',
                                      bl_mynn_cloud_tendency_form='gsd_41')
    for name in ('du','dv','dth','dqv','dqc','dqi'):
        np.testing.assert_array_equal(cp.asnumpy(getattr(out,name)).view(np.uint32),
                                      f[name].view(np.uint32),err_msg=name)


@requires_gpu
@pytest.mark.gpu
def test_default_fork_water_output_matches_source_where_no_clip_binds():
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_tendencies_default_cuda
    f = _fields()
    out = mynn_tendencies_default_cuda(_values(f), bl_mynn_version='gsd_41')
    # The first ten columns have no negative-condensate repair.
    for name in ('dqv','dqc','dqi'):
        np.testing.assert_array_equal(cp.asnumpy(getattr(out,name))[:10].view(np.uint32),
                                      f[name][:10].view(np.uint32),err_msg=name)


@requires_gpu
@pytest.mark.gpu
def test_source_defect_form_changes_heat_at_cloud_edges_and_water_when_clipped():
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_tendencies_default_cuda
    f = _fields(); values = _values(f)
    out = mynn_tendencies_default_cuda(values, bl_mynn_version='gsd_41')
    conserving = {name:cp.asnumpy(getattr(out,name)).copy() for name in ('dth','dqv','dqc','dqi')}
    out = mynn_tendencies_default_cuda(values, bl_mynn_version='gsd_41',
                                      bl_mynn_cloud_tendency_form='gsd_41')
    source = {name:cp.asnumpy(getattr(out,name)).copy() for name in conserving}
    assert np.max(np.abs(conserving['dth'][8:10]-source['dth'][8:10])) > 1.e-6
    def column_water(rates):
        qv=values['qv'].astype(np.float64)+values['delt'][:,None]*rates['dqv']
        qc=values['qc'].astype(np.float64)+values['delt'][:,None]*rates['dqc']
        qi=values['qi'].astype(np.float64)+values['delt'][:,None]*rates['dqi']
        return (((qv+qc+qi)/(1.+qv))*values['dz']).sum(axis=1)
    assert np.max(column_water(source)-column_water(conserving)) > 1.e-6


@requires_gpu
@pytest.mark.gpu
def test_zero_updated_vapour_denominator_has_a_defined_finite_fallback():
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_tendencies_default_cuda
    from test_mynn_gsd41 import _tendency_column
    values,col=_tendency_column(1,12,0.)
    values['dz'][:]=20.; values['dfh'][:]=0.; values['dfm'][:]=0.
    values['sqv'][:]=0.5; values['qv'][:]=1.; values['flqv'][:]=0.5
    out=mynn_tendencies_default_cuda(values,bl_mynn_version='gsd_41')
    for name in ('dqv','dqc','dqi'):
        got=cp.asnumpy(getattr(out,name))
        assert np.isfinite(got).all()
        assert got[0,0] == 0.


def test_defect_form_is_off_in_global_recipe_and_named_import_defaults(tmp_path):
    from gpuwm.config import RunConfig,validate_run_config
    from gpuwm.experiment import load_experiment,build_experiment
    from gpuwm.namelist_import import import_namelists
    import tomllib
    from test_mynn_gsd41 import _cfg
    assert _cfg().bl_mynn_cloud_tendency_form == 'wrf_461'
    with pytest.raises(ValueError,match='requires bl_mynn_version'):
        validate_run_config(_cfg(bl_mynn_cloud_tendency_form='gsd_41'))
    validate_run_config(_cfg(bl_mynn_version='gsd_41',bl_mynn_cloud_tendency_form='gsd_41'))
    root=Path(__file__).parents[1]
    assert load_experiment(root/'configs/recipes/hrrr_v4_gsd41.toml').root.run.bl_mynn_cloud_tendency_form == 'wrf_461'
    fixtures=root/'tests/fixtures/source_requests'; path=tmp_path/'hrrr_wrf.nl'
    path.write_bytes((fixtures/'hrrr_wrf.nl.c18c').read_bytes())
    text,_=import_namelists(fixtures/'hrrr_namelist.wps.c18',path)
    assert build_experiment(tomllib.loads(text),source="cloud source test").root.run.bl_mynn_cloud_tendency_form == 'wrf_461'


def test_explicit_defect_form_survives_selector_comment_import(tmp_path):
    from gpuwm.physics_source_defaults import with_physics_selector_comment
    from gpuwm.namelist_import import import_namelists
    from gpuwm.experiment import build_experiment
    import tomllib
    root=Path(__file__).parents[1]; fixtures=root/'tests/fixtures/source_requests'
    path=tmp_path/'hrrr_wrf.nl'
    path.write_text(with_physics_selector_comment((fixtures/'hrrr_wrf.nl.c18c').read_text(),
        {'bl_mynn_version':'gsd_41','bl_mynn_cloud_tendency_form':'gsd_41'}))
    text,_=import_namelists(fixtures/'hrrr_namelist.wps.c18',path)
    assert build_experiment(tomllib.loads(text),source="cloud source test").root.run.bl_mynn_cloud_tendency_form == 'gsd_41'
