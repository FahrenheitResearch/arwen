"""WSM7 registration is metadata and does not require a CUDA installation."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_registry_builder_runs_without_cupy(tmp_path):
    code = """
import builtins, runpy, sys
original = builtins.__import__
def no_cupy(name, *args, **kwargs):
    if name == 'cupy' or name.startswith('cupy.'):
        raise ImportError('deliberately unavailable CUDA package')
    return original(name, *args, **kwargs)
builtins.__import__ = no_cupy
sys.argv = ['tools/build_registry.py', '--out', sys.argv[1]]
runpy.run_path('tools/build_registry.py', run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'registry.json')],
        cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_catalog_exposes_wsm7_and_its_door_is_the_selector():
    # The lane's named suite was retired (it sat only on the native HRRR
    # route, whose runner refused it); WSM7 is reached as mp_physics=24.
    from gpuwm.physics_catalog import table
    from gpuwm.physics_registry import physics_registry
    registry = physics_registry()
    option = registry['components']['microphysics']['options']['wsm7-mp24']
    assert option['implemented'] is True
    assert option['selectors'] == {'mp_physics': 24}
    assert option['reachability']['state'] == 'component-override'
    assert 'hail' in table()['descriptions']['microphysics']['wsm7-mp24']
    assert 'wsm7-ysu-revised-mm5-noah-rrtmg-legacy-v1' not in registry['templates']
    for route in registry['runner_routes'].values():
        for ids in (route.get('source_template_ids') or {}).values():
            assert 'wsm7-ysu-revised-mm5-noah-rrtmg-legacy-v1' not in ids


def test_absent_hail_uses_the_declared_native_zero_word():
    from gpuwm.ingest.microphysics_cold_start import cold_start_contract
    fields, words = cold_start_contract({'mp_physics': 24})
    assert fields == ('QHAIL',)
    assert words == {'qh': (0.0, 0)}


def test_existing_source_absent_contracts_keep_their_species():
    from gpuwm.ingest.microphysics_cold_start import cold_start_contract
    assert cold_start_contract({'mp_physics': 6}) == ((), {})
    assert cold_start_contract({'mp_physics': 8}) == (
        ('QNICE', 'QNRAIN'), {'ni': (0.0, 0), 'nr': (0.0, 0)})
