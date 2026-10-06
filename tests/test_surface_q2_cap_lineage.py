"""The RUC fork diagnostic set owns the final water-column Q2 bound."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import surface_humidity


def _fields():
    q2 = np.array([.05, .05, .008], np.float32)
    qv1 = np.full(3, .01, np.float32)
    xland = np.array([np.nextafter(np.float32(1.5), np.float32(0)),
                      1.5, 2.0], np.float32)
    return q2, qv1, xland


def test_default_surface_bound_preserves_the_land_wrapper_words():
    q2, qv1, xland = _fields()
    land = q2.copy()
    surface_humidity.cap_land_q2(land, qv1, xland, xp=np)
    surface_humidity.cap_surface_q2(q2, qv1, xland, xp=np)
    np.testing.assert_array_equal(q2.view(np.uint32), land.view(np.uint32))
    np.testing.assert_array_equal(q2.view(np.uint32),
        np.array([0x3c2c0830, 0x3d4ccccd, 0x3c03126f], np.uint32))


def test_fork_surface_bound_caps_water_and_retains_values_below_the_bound():
    q2, qv1, xland = _fields()
    surface_humidity.cap_surface_q2(q2, qv1, xland, xp=np, include_water=True)
    np.testing.assert_array_equal(q2.view(np.uint32),
        np.array([0x3c2c0830, 0x3c2c0830, 0x3c03126f], np.uint32))


def _final_surface_writer():
    # Execute the actual post-LSM/lake writer on CPU without advancing any
    # scheme. This checks selector wiring as well as the standalone bound.
    source = Path(__file__).resolve().parents[1] / 'gpuwm/core/physics.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    candidates = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.If)
                and any(isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == 'cap_surface_q2'
                        for statement in node.body
                        for call in ast.walk(statement))):
            candidates.append(node)
    # The smallest matching if is the RUC-specific dispatch inside the
    # common post-surface writer, after the final lake diagnostic.
    selected = min(candidates, key=lambda node: node.end_lineno - node.lineno)
    return compile(ast.fix_missing_locations(ast.Module(body=[selected],
                                                       type_ignores=[])),
                   str(source), 'exec')


@pytest.mark.parametrize('surface,diagnostic,water_is_capped', [
    (3, 'log_profile', True),
    (3, 'flux', False),
    (2, 'log_profile', False),
    (2, 'flux', False),
    (4, 'log_profile', False),
])
def test_actual_final_writer_selects_the_water_bound_only_for_ruc_fork(
        surface, diagnostic, water_is_capped):
    q2, qv1, xland = _fields()
    namespace = {
        'self': SimpleNamespace(fields={'q2': q2, 'xland': xland}),
        'cfg': SimpleNamespace(sf_surface_physics=surface,
                               ruc_2m_diagnostic=diagnostic),
        'atmosphere': {'qv': qv1[None, :]},
        'surface_humidity': surface_humidity, 'cp': np,
    }
    exec(_final_surface_writer(), namespace)
    expected = np.array([0x3c2c0830,
                         0x3c2c0830 if water_is_capped else 0x3d4ccccd,
                         0x3c03126f], np.uint32)
    np.testing.assert_array_equal(q2.view(np.uint32), expected)
