"""Wesely gas dry deposition against unmodified WRF-Chem v4.7.1 (column oracle).

The fixtures are the Fortran's own output (tools/chem_wrf471_oracle/wesely,
PROVENANCE beside them): dep_init's tables and derived numbers and
wesely_driver's ddvel on 11 cases, 4,752 columns, covering every USGS class
and the MODIS map, both seasons and their boundaries, day and night, dry,
wet and rain flags, stable, neutral and unstable layers including the rmol
snap, low and high ust and znt, every SO2 temperature arm, the ammonia arms
with both highnh3 states, and an A/B/A sequence proving the scheme keeps no
memory.  No CPU mirror is a reference.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import numpy as np
import pytest

from gpuwm import chem_table
from gpuwm.core.chem_drydep_gas import (
    LAUNCH_FIELDS, _require_soil, load_wesely_tables, pack_rows, rows,
    season, wesely_ddvel,
)
from gpuwm.core.fp32_ulp import max_ulp

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/data/oracles/chem/cams/wesely'
PARAMETERS = ROOT / 'gpuwm/data/chem/wesely/gas_parameters.v1.json'
CASES = tuple(sorted(FIXTURES.glob('case_*')))
#: The fixture's gas slots, in the driver's order (run_wesely.F90).
ORACLE_ORDER = ('o3', 'no2', 'co', 'so2', 'nh3', 'hno3')


def fixture(case):
    result = {}
    for line in (case / 'manifest.txt').read_text().splitlines():
        name, dtype, *shape = line.split()
        shape = tuple(map(int, shape))
        array = np.fromfile(case / f'{name}.bin',
                            '<f4' if dtype == 'float32' else '<i4')
        result[name] = array.reshape(shape, order='F')
    return result


def rows_for(nrow):
    """The fixture's gases as rows; the v1 four are the real table rows."""
    table = {row.name: row for row in chem_table.load_sets(['cams_aq']).rows}
    extras = json.loads(PARAMETERS.read_text())
    out = []
    for name in ORACLE_ORDER[:nrow]:
        if name in table:
            out.append(table[name])
        else:
            spec = {k: v for k, v in extras[name].items()
                    if k in ('hstar', 'dhr', 'f0', 'dratio', 'scpr23', 'arm')}
            out.append(SimpleNamespace(name=name, drydep={'scheme': 'wesely',
                                                          'wesely': spec}))
    return out


def test_import_is_cupy_free():
    result = subprocess.run(
        [sys.executable, '-c', "import sys; import gpuwm.core.chem_drydep_gas; "
         "assert 'cupy' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_fixture_inventory_and_memory():
    assert len(CASES) == 11
    assert sum(fixture(case)['ivgtyp'].size for case in CASES) >= 300
    assert sum(p.stat().st_size for p in FIXTURES.rglob('*') if p.is_file()) < 3_000_000
    a, b, repeat = (fixture(FIXTURES / f'case_{i:02}') for i in (9, 10, 11))
    np.testing.assert_array_equal(a['ddvel'], repeat['ddvel'])
    np.testing.assert_array_equal(b['ddvel'], repeat['changed_ddvel'])
    assert not np.array_equal(a['ddvel'], b['ddvel'])
    for case in CASES:
        data = fixture(case)
        assert np.isfinite(data['ddvel']).all()
        assert (data['ddvel'] > 0).all()


@pytest.mark.parametrize('case', CASES, ids=lambda p: p.name)
def test_tables_and_row_numbers_equal_the_fortran(case):
    """The JSON tables and every row's six numbers are dep_init's, bit for bit."""
    data = fixture(case)
    tables = load_wesely_tables()
    for name, array in tables.arrays.items():
        np.testing.assert_array_equal(array, data[name], err_msg=name)
    for name, value in [('iswater_temp', tables.water), ('isice_temp', tables.ice)]:
        assert value == data[name]
    mminlu = 'MODIFIED_IGBP_MODIS_NOAH' if data['map_kind'] else 'USGS'
    np.testing.assert_array_equal(tables.landuse_maps[mminlu], data['luse2usgs'])
    nrow = int(data['nrow'])
    params, arms, nh3, so2, scpr = pack_rows(rows_for(nrow))
    for col, name in enumerate(('hstar', 'dhr', 'f0', 'dratio')):
        np.testing.assert_array_equal(params[:, col], data[name][:nrow], err_msg=name)
    np.testing.assert_array_equal(scpr, data['scpr23'][:nrow])


def test_the_cams_rows_are_the_wesely_rows():
    """The four cams_aq gases are exactly the rows the process acts on."""
    table = chem_table.load_sets(['cams_aq'])
    assert [row.name for row in rows(table)] == ['o3', 'no2', 'co', 'so2']


def test_packing_is_row_driven():
    source = rows_for(6)
    renamed = [SimpleNamespace(name=f'arbitrary_{i}', drydep=row.drydep)
               for i, row in enumerate(source)]
    params, arms, nh3, so2, scpr = pack_rows(renamed)
    np.testing.assert_array_equal(pack_rows(source)[0], params)
    np.testing.assert_array_equal(arms, [1, 0, 0, 2, 3, 0])
    assert (nh3, so2) == (4, 3)
    assert pack_rows([])[0].shape == (0, 4)
    assert pack_rows(source[:4])[2:4] == (-1, 3)


def _spec(row):
    block = row.drydep['wesely']
    return dict(block.items()) if not isinstance(block, dict) else dict(block)


@pytest.mark.parametrize('key,value,message', [
    ('arm', 'unknown', 'unknown Wesely arm'),
    ('dratio', 0, 'nonpositive dratio'),
    ('scpr23', 0, 'nonpositive dratio/scpr23'),
    ('hstar', float('nan'), 'nonfinite Wesely parameter'),
])
def test_bad_row_refusals(key, value, message):
    row = rows_for(1)[0]
    spec = _spec(row)
    spec[key] = value
    bad = SimpleNamespace(name='o3', drydep={'scheme': 'wesely', 'wesely': spec})
    with pytest.raises(ValueError, match=message):
        pack_rows([bad])


def test_a_second_row_on_a_highnh3_arm_is_refused():
    so2 = rows_for(4)[3]
    twin = SimpleNamespace(name='twin', drydep={'scheme': 'wesely',
                                                'wesely': _spec(so2)})
    with pytest.raises(ValueError, match='a second row on the sulfur_dioxide arm'):
        pack_rows([so2, twin])


def test_a_row_naming_the_process_without_the_block_is_refused():
    bad = SimpleNamespace(name='x', drydep={'scheme': 'none'})
    with pytest.raises(ValueError, match='not the wesely scheme'):
        pack_rows([bad])


@pytest.mark.parametrize('name', LAUNCH_FIELDS)
def test_missing_field(name):
    fields = {key: None for key in LAUNCH_FIELDS if key != name}
    with pytest.raises(ValueError, match=f'missing Wesely field: {name}'):
        wesely_ddvel(fields, rows_for(4), julday=90, mminlu='USGS')


def test_landuse_and_soil_refusals():
    with pytest.raises(ValueError, match="dep_init leaves luse2usgs unset for NLCD"):
        wesely_ddvel(dict.fromkeys(LAUNCH_FIELDS), [], julday=90, mminlu='NLCD')
    with pytest.raises(ValueError, match='iland would be zero in deppart'):
        _require_soil(0)
    _require_soil(2)


@pytest.mark.parametrize('day,want', [(1, 2), (89, 2), (90, 1), (270, 1),
                                      (271, 2), (365, 2)])
def test_season(day, want):
    assert season(day) == want


@pytest.mark.gpu
@pytest.mark.parametrize('case', CASES, ids=lambda p: p.name)
def test_gpu_parity(case):
    cp = pytest.importorskip('cupy')
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'no CuPy device: {exc}')
    if count == 0:
        pytest.skip('no CuPy device')
    data = fixture(case)
    nrow = int(data['nrow'])
    fields = {name: cp.asarray(np.ascontiguousarray(data[name].T))
              for name in LAUNCH_FIELDS}
    # Fortran (nx, ny, nrow) bytes map to C (nrow, ny, nx).
    fields['chem_k1'] = cp.asarray(np.ascontiguousarray(data['chem_k1'].transpose(2, 1, 0)))
    for name in ('aer_res_def', 'aer_res_zcen'):
        fields[name] = cp.empty(fields['tsk'].shape, dtype=cp.float32)
    mminlu = 'MODIFIED_IGBP_MODIS_NOAH' if data['map_kind'] else 'USGS'
    acting = rows_for(nrow)
    got = wesely_ddvel(fields, acting, julday=int(data['julday']), mminlu=mminlu)
    measured = {}
    for index, row in enumerate(acting):
        measured[row.name] = max_ulp(cp.asnumpy(got[index]),
                                     data['ddvel'][:, :, index].T)
    for name in ('aer_res_def', 'aer_res_zcen'):
        measured[name] = max_ulp(cp.asnumpy(fields[name]), data[name].T)
    assert all(value == 0 for value in measured.values()), \
        f'{case.name}: measured ULP {measured}'
