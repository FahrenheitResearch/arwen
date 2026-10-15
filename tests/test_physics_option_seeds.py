"""Real option seed CLI and existing-engine authority checks."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_seed_cli_admits_the_port_rows_without_replacing_existing_engine_options(tmp_path):
    import pytest
    try:
        base = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'cd336e62d6e6b6312c9636e0fdfff84801909b6a:gpuwm/physics_registry_v2.json'],
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # 2.8.9's engine commit is private history; a public snapshot (CI
        # rehearsal routes B and C, 2.9.0 round 2) cannot read its registry.
        # The private clone's Stage 1 runs this check.
        pytest.skip("2.8.9's engine registry is not in this tree's history (public snapshot)")
    source = tmp_path / 'E.json'
    source.write_bytes(base)
    out = tmp_path / 'seeded.json'
    subprocess.run([sys.executable, str(ROOT / 'tools/seed_physics_options.py'), '--registry', str(source), '--out', str(out)], check=True)
    original = json.loads(base)
    seeded = json.loads(out.read_text())
    declarations = json.loads((ROOT / 'gpuwm/physics_option_seeds_v1.json').read_text())
    for component, table in original['components'].items():
        for name, row in table.get('options', {}).items():
            if row.get('implemented') is True:
                assert seeded['components'][component]['options'][name] == row, (component, name)
    for component, options in declarations['components'].items():
        for name, row in options.items():
            assert seeded['components'][component]['options'][name]['implemented'] is True
            assert seeded['components'][component]['options'][name]['selectors'] == row['selectors']
    assert source.read_bytes() == base

def test_runtime_consumers_and_registry_agree_after_regeneration():
    # Import the actual consumer modules. Their import guards validate both
    # selector coverage and the values that dispatch, history and optics use.
    import gpuwm.config
    import gpuwm.core.preflight
    import gpuwm.core.refl
    import gpuwm.core.physics_inventory
    from gpuwm.physics_registry import physics_registry
    registry = physics_registry()
    declarations = json.loads((ROOT / 'gpuwm/physics_option_seeds_v1.json').read_text())
    for component, options in declarations['components'].items():
        for name, declaration in options.items():
            row = registry['components'][component]['options'][name]
            assert row['implemented'] is True
            assert row['selectors'] == declaration['selectors']
