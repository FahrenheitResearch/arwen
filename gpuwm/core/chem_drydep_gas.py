"""WRF-Chem v4.7.1 Wesely gas dry deposition, row-driven (process ``drydep.wesely``).

Every active row naming ``drydep.wesely`` in its ``processes`` column gets a
dry deposition velocity (m s-1) per column each chem step, from WRF-Chem's
non-MOZART path: ``wesely_driver`` -> ``rc`` -> ``landusevg``/``depvel`` ->
``cellvg`` (chem/module_dep_simple.F:178-257, :888-1288, :1347-1616), in one
CUDA thread per column (``kernels/chem_drydep_wesely.cu``).  The velocity is
written into the row's plane of ``ctx.ddvel``, which vertical mixing reads as
its lower boundary; this process moves no mass itself.

ROW DATA, NOT NAMES.  The row's ``drydep.wesely`` block carries WRF's four
numbers and the two dep_init derives (``hstar``, ``dhr``, ``f0``, ``dratio``,
``scpr23``, the float32 values the Fortran computed, pinned by the column
oracle) and the resistance ``arm`` of ``rc`` it takes (general, ozone,
sulfur_dioxide, ammonia).  ``rc``'s NH3-over-SO2 test (``highnh3``) reads the
rows whose arms are ammonia and sulfur_dioxide.  No code here tests a
species name or index.  The land-use tables (``wesely_tables.v1.json``) are
dep_init's DATA statements; the land-use dataset is the run's own.

Cadence: WRF-Chem runs dry deposition and vertical mixing only after the
second step (``ktau > 2``, chem/chem_driver.F:1047); so does this process.

Declared difference: depvel clamps ``|rmol| < 1e-6`` to zero through
landusevg's INTENT(INOUT) argument, which in WRF-Chem writes back into
``grid%rmol``.  The kernel applies the clamp to its own copy and never writes
the physics' ``rmol``, so chem cannot move the surface layer; the velocities
are identical.

Imports no CuPy until a launch.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
import json

from gpuwm.core.chem_context import ChemAllocation

_DATA = Path(__file__).resolve().parents[1] / 'data' / 'chem' / 'wesely'
_TABLE_NAMES = ('ri', 'rlu', 'rac', 'rgss', 'rgso', 'rcls', 'rclo')
_ARMS = {'general': 0, 'ozone': 1, 'sulfur_dioxide': 2, 'ammonia': 3}

#: The process key (gpuwm.chem_table.CHEM_PROCESS_MODULES).
KEY = 'drydep.wesely'
PROCESS_KEY = KEY
#: This process computes velocities and moves no mass; vertical mixing
#: books the deposited mass.
LEDGER = 'deposited'
#: The kernel translation units step() launches (gpuwm.core.chem_context's
#: process contract: the preflight prices their local frames and refuses a
#: process that names none).
KERNEL_MODULES = ('chem_drydep_wesely',)
#: Context fields (ChemContext.met names): the lowest-level state and the
#: surface layer's fields.  ``qc``, ``qr`` and ``raincv`` are read when the
#: run carries them and are zero otherwise (no cloud water, no rain water,
#: no cumulus rain: WRF's own values then).
REQUIRES = ('t_phy', 'p_phy', 'p8w', 'qv', 'z_at_w', 'tsk', 'gsw',
            'vegfra', 'ust', 'rmol', 'znt', 'ivgtyp')
#: The launcher's (ny, nx) inputs, level-1 slices of the fields above.
LAUNCH_FIELDS = ('t_phy_k1', 'p_phy_k1', 'p8w_k1', 'qv_k1', 'qc_k1', 'qr_k1',
                 'dz1', 'tsk', 'gsw', 'vegfra', 'ust', 'rmol', 'znt', 'raincv',
                 'ivgtyp')
ALLOCATES = (
    ChemAllocation('wesely_aer_res_def', '2d', restart='rebuild',
                   units='s m-1',
                   description=('aerodynamic resistance from 2 m (aer_res_def, '
                                'chem/module_dep_simple.F:1415-1430), read by '
                                "GOCART's aerosol deposition")),
    ChemAllocation('wesely_aer_res_zcen', '2d', restart='rebuild',
                   units='s m-1',
                   description=('aerodynamic resistance from the lowest '
                                'layer centre (aer_res_zcen)')),
    ChemAllocation('wesely_ddvel', 'rows_2d', restart='rebuild',
                   units='m s-1',
                   description=('the launch output, one plane per Wesely row, '
                                'copied into ctx.ddvel each step')),
    ChemAllocation('wesely_zero', '2d', restart='rebuild',
                   description=('a zero plane standing in for qc, qr and '
                                'raincv when this run carries none; never '
                                'written')),
)


@dataclass(frozen=True)
class WeselyTables:
    arrays: Mapping[str, np.ndarray]
    landuse_maps: Mapping[str, np.ndarray]
    identity: str
    resistance: np.ndarray
    water: int
    ice: int


def load_wesely_tables(path=None) -> WeselyTables:
    raw = Path(path or _DATA / 'wesely_tables.v1.json').read_bytes()
    data = json.loads(raw)
    arrays = {}
    for name in (*_TABLE_NAMES, 'kpart', 'ixxxlu'):
        entry = data[name]
        dtype = np.int32 if name == 'ixxxlu' else np.float32
        array = np.asarray(entry['values'], dtype=dtype)
        expected = (25, 5) if name in _TABLE_NAMES else (25,)
        if tuple(entry['shape']) != expected or array.size != np.prod(expected):
            raise ValueError(f'invalid Wesely table shape: {name}')
        array = array.reshape(expected, order='F')
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite Wesely table: {name}')
        array.setflags(write=False)
        arrays[name] = array
    maps = {}
    for name, entry in data['landuse_maps'].items():
        array = np.asarray(entry['values'], dtype=np.int32)
        if array.ndim != 1 or not ((array >= 1) & (array <= 25)).all():
            raise ValueError(f'invalid Wesely landuse map: {name}')
        array.setflags(write=False)
        maps[name] = array
    resistance = np.concatenate([arrays[name].ravel(order='F')
                                 for name in (*_TABLE_NAMES, 'kpart')])
    resistance.setflags(write=False)
    return WeselyTables(MappingProxyType(arrays), MappingProxyType(maps),
                        sha256(raw).hexdigest(), resistance,
                        int(data['iswater_temp']['values']),
                        int(data['isice_temp']['values']))


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _wesely_block(row):
    drydep = _get(row, 'drydep') or {}
    if _get(drydep, 'scheme') != 'wesely':
        raise ValueError(
            f"{_get(row, 'name', '<unnamed>')}: names drydep.wesely but its "
            "drydep block is not the wesely scheme, so the process has no "
            "numbers to act with")
    return _get(drydep, 'wesely')


def rows(table):
    """The active rows naming ``drydep.wesely``, in table order."""
    rows_for = getattr(table, 'rows_for', None)
    if callable(rows_for):
        return rows_for(KEY)
    return tuple(row for row in _get(table, 'rows', table)
                 if _get(_get(row, 'drydep') or {}, 'scheme') == 'wesely')


ROWS = rows


def pack_rows(rows):
    """``(params (n,4) float32, arms (n,) int32, nh3_row, so2_row, scpr23)``.

    ``params`` columns are hstar, dhr, f0, dratio.  The ammonia and
    sulfur_dioxide arms name the rows rc's highnh3 test reads (-1 when a
    run carries none, WRF's absent species); two rows on one of those arms
    is refused, because the test would compare an arbitrary pair.
    """
    selected = tuple(rows)
    params = np.empty((len(selected), 4), dtype=np.float32)
    scpr = np.empty(len(selected), dtype=np.float32)
    arms = np.empty(len(selected), dtype=np.int32)
    roles = {'ammonia': -1, 'sulfur_dioxide': -1}
    for index, row in enumerate(selected):
        name = _get(row, 'name', '<unnamed>')
        spec = _wesely_block(row)
        arm = _get(spec, 'arm')
        if arm not in _ARMS:
            raise ValueError(f'{name}: unknown Wesely arm {arm!r}')
        arms[index] = _ARMS[arm]
        for col, key in enumerate(('hstar', 'dhr', 'f0', 'dratio')):
            if _get(spec, key) is None:
                raise ValueError(f'{name}: missing Wesely parameter {key}')
            params[index, col] = _get(spec, key)
        if _get(spec, 'scpr23') is None:
            raise ValueError(f'{name}: missing Wesely parameter scpr23')
        scpr[index] = _get(spec, 'scpr23')
        if (not np.isfinite(params[index]).all() or params[index, 3] <= 0
                or not np.isfinite(scpr[index]) or scpr[index] <= 0):
            raise ValueError(f'{name}: nonfinite Wesely parameter or '
                             'nonpositive dratio/scpr23')
        if arm in roles:
            if roles[arm] >= 0:
                raise ValueError(
                    f"{name}: a second row on the {arm} arm; rc's highnh3 "
                    "test compares one ammonia row with one SO2 row")
            roles[arm] = index
    return params, arms, roles['ammonia'], roles['sulfur_dioxide'], scpr


def season(julday):
    # chem/module_dep_simple.F:178-185.
    return 2 if julday < 90 or julday > 270 else 1


def _landmap(tables, mminlu):
    if mminlu not in tables.landuse_maps:
        raise ValueError("WRF-Chem's dep_init leaves luse2usgs unset for "
                         f'{mminlu}, so every land class would index garbage')
    return tables.landuse_maps[mminlu]


def _require_soil(value):
    # chem/module_dep_simple.F:1689-1691.
    if value == 0:
        raise ValueError('WRF-Chem requires a soil model: iland would be zero '
                         'in deppart and index outside every table')


def wesely_ddvel(fields: Mapping[str, object], rows, *, julday, mminlu,
                 tables=None, out=None, sf_surface_physics=None):
    """Device float32 ``(nrow, ny, nx)`` deposition velocity, m s-1.

    ``fields`` holds the (ny, nx) :data:`LAUNCH_FIELDS` (device arrays,
    ``ivgtyp`` int32) and optionally ``chem_k1`` (nrow, ny, nx), the rows'
    lowest-level mixing ratios, which rc's highnh3 test reads when both an
    ammonia and a sulfur_dioxide row are active; ``aer_res_def`` and
    ``aer_res_zcen`` are optional (ny, nx) output buffers.
    """
    for name in LAUNCH_FIELDS:
        if name not in fields:
            raise ValueError(f'missing Wesely field: {name}')
    if sf_surface_physics is not None:
        _require_soil(sf_surface_physics)
    tables = tables or load_wesely_tables()
    landmap = _landmap(tables, mminlu)
    rows = tuple(rows)
    params, arms, nh3_row, so2_row, scpr = pack_rows(rows)
    if nh3_row >= 0 and so2_row >= 0 and fields.get('chem_k1') is None:
        raise ValueError('missing Wesely field: chem_k1 (an ammonia and an '
                         'SO2 row are active, and rc compares them)')
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    shape = fields['tsk'].shape
    if len(shape) != 2:
        raise ValueError('Wesely fields must have shape (ny, nx)')

    def device(value, name, expected_shape, dtype):
        if not isinstance(value, cp.ndarray):
            raise TypeError(f'{name}: expected device array')
        if value.shape != expected_shape or value.dtype != np.dtype(dtype):
            raise ValueError(f'{name}: expected {dtype} shape {expected_shape}')
        if not value.flags.c_contiguous:
            raise ValueError(f'{name}: expected C-order contiguous array')
        return value
    inputs = [device(fields[name], name, shape,
                     'int32' if name == 'ivgtyp' else 'float32')
              for name in LAUNCH_FIELDS]
    land = fields['ivgtyp']
    if bool(cp.any((land < 1) | (land > len(landmap)))):
        raise ValueError('ivgtyp is outside luse2usgs: land class would '
                         'index garbage')
    nrow = len(rows)
    output_shape = (nrow, *shape)
    if out is None:
        out = cp.empty(output_shape, dtype=cp.float32)
    device(out, 'out', output_shape, 'float32')
    diagnostics = []
    for name in ('aer_res_def', 'aer_res_zcen'):
        value = fields.get(name)
        diagnostics.append(device(value, name, shape, 'float32')
                           if value is not None else np.uint64(0))
    chem = fields.get('chem_k1')
    if chem is not None:
        device(chem, 'chem_k1', output_shape, 'float32')
    else:
        chem = np.uint64(0)
    ncol = int(np.prod(shape))
    if ncol == 0 or nrow == 0:
        return out
    kernel = get_kernel('chem_drydep_wesely', 'chem_wesely_ddvel')
    kernel(((ncol + 63) // 64,), (64,), tuple(inputs) + (
        cp.asarray(params), cp.asarray(arms), cp.asarray(scpr),
        cp.asarray(tables.resistance), cp.asarray(tables.arrays['ixxxlu']),
        cp.asarray(landmap), chem, np.int32(julday), np.int32(nrow),
        np.int32(nh3_row), np.int32(so2_row), np.int32(ncol), out,
        *diagnostics, np.int32(tables.water), np.int32(tables.ice)))
    return out


def _landuse_dataset(ctx) -> str:
    """The run's land-use dataset (WRF MMINLU), from the surface scheme's tables."""
    physics = getattr(ctx.state, 'physics', None)
    for owner, attribute in (('noah_params', 'lutype'),
                             ('noahmp_params', 'dataset_identifier'),
                             ('ruc_params', 'dataset_identifier')):
        value = getattr(getattr(physics, owner, None), attribute, None)
        if value:
            return str(value)
    raise ValueError(
        'drydep.wesely: the run\'s land-use dataset (MMINLU) is not known to '
        'the surface scheme\'s parameter tables, so the Wesely land classes '
        'cannot be mapped and every resistance would be read for the wrong '
        'class')


def _level1(value):
    import cupy as cp
    return cp.ascontiguousarray(value[0], dtype=cp.float32)


def _launch_fields(ctx, rows):
    import cupy as cp
    zero = ctx.diag['wesely_zero']
    fields = {
        't_phy_k1': _level1(ctx.met('t_phy')),
        'p_phy_k1': _level1(ctx.met('p_phy')),
        'p8w_k1': _level1(ctx.met('p8w')),
        'qv_k1': _level1(ctx.met('qv')),
        # No cloud or rain water in this run: WRF's own zero.
        'qc_k1': _level1(ctx.met('qc')) if ctx.has('qc') else zero,
        'qr_k1': _level1(ctx.met('qr')) if ctx.has('qr') else zero,
    }
    z_at_w = ctx.met('z_at_w')
    fields['dz1'] = cp.ascontiguousarray(z_at_w[1] - z_at_w[0],
                                         dtype=cp.float32)
    for name in ('tsk', 'gsw', 'vegfra', 'ust', 'rmol', 'znt'):
        fields[name] = cp.ascontiguousarray(ctx.met(name), dtype=cp.float32)
    # No cumulus scheme wrote raincv: WRF's value with cu_physics = 0.
    fields['raincv'] = (cp.ascontiguousarray(ctx.met('raincv'),
                                             dtype=cp.float32)
                        if ctx.has('raincv') else zero)
    fields['ivgtyp'] = cp.ascontiguousarray(ctx.met('ivgtyp'), dtype=cp.int32)
    _, _, nh3, so2, _ = pack_rows(rows)
    if nh3 >= 0 and so2 >= 0:
        fields['chem_k1'] = cp.ascontiguousarray(
            cp.stack([ctx.field(row)[0] for row in rows]), dtype=cp.float32)
    fields['aer_res_def'] = ctx.diag.get('wesely_aer_res_def')
    fields['aer_res_zcen'] = ctx.diag.get('wesely_aer_res_zcen')
    return fields


def init(ctx):
    """Refuse what the scheme cannot run on, before the first step."""
    _require_soil(int(getattr(ctx.cfg, 'sf_surface_physics', 0)))
    _landmap(load_wesely_tables(), _landuse_dataset(ctx))
    pack_rows(rows(ctx.table))


def step(ctx, dt, ktau):
    """Write each Wesely row's velocity into its ``ctx.ddvel`` plane."""
    if int(ktau) <= 2:
        # chem/chem_driver.F:1047: dry deposition and vertical mixing start
        # at the third step; the driver zeroed ctx.ddvel already.
        return
    acting = rows(ctx.table)
    if not acting:
        return
    velocity = wesely_ddvel(_launch_fields(ctx, acting), acting,
                            julday=int(ctx.clock.julday),
                            mminlu=_landuse_dataset(ctx),
                            out=ctx.diag['wesely_ddvel'])
    for index, row in enumerate(acting):
        ctx.ddvel[ctx.row_index(row)] = velocity[index]
