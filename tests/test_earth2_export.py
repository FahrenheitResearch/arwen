"""The two Python contracts: table resolution and real analysis reflectivity."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.earth2_export import AUTHORITY, channel_rows
from gpuwm.ml_export import _load


def test_scheme_and_consumer_rows_resolve_against_019_lexicon():
    channels = _load('profiles')['default_channels']
    consumers = json.loads(AUTHORITY.read_text())['consumers']
    channels += [c for row in consumers.values() for c in row['input_channels']]
    resolved = channel_rows(channels)
    assert {r['names']['earth2'] for r in resolved} == set(channels)
    assert len(consumers['stormcast-conus']['input_channels']) == 26
    source = os.environ.get('GPUWM_EARTH2_LEXICON')
    if source:
        module = ast.parse(Path(source).read_text())
        vocabulary = next(ast.literal_eval(n.value) for n in module.body
                          if isinstance(n, (ast.Assign, ast.AnnAssign))
                          and isinstance(n.value, ast.Dict)
                          and ('u100m' in ast.literal_eval(n.value)))
        assert set(channels) <= vocabulary.keys()
        for row in resolved:
            units = row['units'].replace('**', '')
            assert '(' + units + ')' in vocabulary[row['names']['earth2']]
        assert 'm2 s-2' in vocabulary['z500']
        assert 'Pa s-1' in vocabulary['w500']
        assert '(m)' in vocabulary['tp06']


def test_tick_zero_reflectivity_uses_initial_hydrometeors():
    cp = pytest.importorskip('cupy')
    from gpuwm.core.refl import analysis_refl_10cm, compute_refl_10cm
    shape = (4, 3, 2)
    buffers = {}
    state = SimpleNamespace(p=cp.full(shape, 90000, dtype=cp.float32),
                            thb=cp.full(shape, 300, dtype=cp.float32),
                            thp=cp.zeros(shape, dtype=cp.float32),
                            qv=cp.full(shape, .01, dtype=cp.float32),
                            qr=cp.full(shape, .001, dtype=cp.float32),
                            qs=cp.zeros(shape, dtype=cp.float32),
                            qg=cp.zeros(shape, dtype=cp.float32))
    def scratch(size, name):
        if name not in buffers:
            buffers[name] = cp.empty(size, dtype=cp.float32)
        return buffers[name]
    state.scratch = scratch
    cfg = SimpleNamespace(mp_physics=6, wsm6_hail_opt=0)
    before = state.qr.copy()
    analysis = analysis_refl_10cm(state, cfg).copy()
    later = compute_refl_10cm(state, cfg).copy()
    cp.testing.assert_array_equal(analysis, later)
    cp.testing.assert_array_equal(before, state.qr)
    assert float(analysis.max()) > 0
    assert float(cp.max(analysis, axis=0).min()) > 0
    # Schemes with their radar inside the production column must also work
    # before the first step, without using a dummy microphysics advance.
    state.alt = cp.ones(shape, dtype=cp.float32)
    state.qc = cp.zeros(shape, dtype=cp.float32)
    state.qi = cp.zeros(shape, dtype=cp.float32)
    state.f_rimef = cp.ones(shape, dtype=cp.float32)
    state.nr = cp.full(shape, 1e6, dtype=cp.float32)
    state.ng = cp.zeros(shape, dtype=cp.float32)
    state.volg = cp.zeros(shape, dtype=cp.float32)
    for mp in (5, 900):
        cfg = SimpleNamespace(mp_physics=mp, dt=60)
        prognostic = {n: getattr(state, n).copy() for n in
                      ('p', 'thp', 'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'nr', 'ng', 'volg', 'f_rimef')}
        analysis = analysis_refl_10cm(state, cfg).copy()
        assert float(analysis.max()) > 0
        for name, original in prognostic.items():
            cp.testing.assert_array_equal(original, getattr(state, name))
