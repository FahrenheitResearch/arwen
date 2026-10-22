"""Full native physical regressions; requires a built bridge and shared GPU slot.

These complement the real atmospheric case, never substitute for that proof.
"""
import json
import os
from pathlib import Path
import subprocess

import pytest

from gpuwm.ocean.surge.driver import run, finish

pytestmark = pytest.mark.skipif(
    not os.environ.get('GPUWM_SURGE_LIBRARY') or not os.environ.get('GPUWM_RW_BATHY'),
    reason='native CUDA physical regressions need built surge library and Rust bridge')


def execute(kind, tmp_path):
    inputs = tmp_path / 'inputs'
    subprocess.run([os.environ['GPUWM_RW_BATHY'], 'fixture', kind, str(inputs)], check=True)
    receipt = run(inputs / 'case.json', tmp_path / 'out', render=False)
    assert receipt['duration_seconds'] == 43200
    assert receipt['cuda_graphs'] is False
    assert receipt['steps'] > 1000
    assert receipt['min_depth_m'] >= 0
    assert receipt['worst_negative_depth_m'] < 1e-4
    assert receipt['max_relative_mass_error'] < 2e-5
    assert receipt['allocated_device_bytes'] > 8 * 384 * 256
    assert (tmp_path / 'out' / 'surge.nc').stat().st_size > 4 * 384 * 256
    return receipt


def test_irregular_lake_and_stationary_shore(tmp_path):
    receipt = execute('lake', tmp_path)
    assert receipt['max_water_level_anomaly_m'] < 2e-4
    assert receipt['max_speed_ms'] < 2e-4
    assert receipt['peak_wet_land_cells'] == 0


def test_wind_setup_moves_a_dry_shore_and_conserves_mass(tmp_path):
    receipt = execute('wind', tmp_path)
    assert receipt['max_water_level_anomaly_m'] > 0.15
    assert receipt['peak_wet_land_cells'] > 100
    assert receipt['max_speed_ms'] > 0.05
    config = tmp_path / 'inputs' / 'case.json'
    recovered = finish(config, tmp_path / 'out', render=False)
    assert recovered['duration_seconds'] == receipt['duration_seconds']
    assert recovered['steps'] == receipt['steps']
    # Changing a variable selection within the same hashed source file must
    # not reuse an integration driven by another wind field.
    changed = json.loads(config.read_text())
    changed['forcing']['u10'] = 'v'
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='different forcing'):
        finish(config, tmp_path / 'out', render=False)


def test_inverse_barometer_enters_open_boundary(tmp_path):
    receipt = execute('pressure', tmp_path)
    assert 0.07 < receipt['max_water_level_anomaly_m'] < 0.2


def test_optional_tide_drives_open_boundary(tmp_path):
    receipt = execute('tide', tmp_path)
    assert 0.15 < receipt['max_water_level_anomaly_m'] < 0.5


def test_deep_ocean_retains_submillimetre_pressure_response(tmp_path):
    receipt = execute('deep-pressure', tmp_path)
    assert 0.07 < receipt['max_water_level_anomaly_m'] < 0.2
    assert receipt['max_relative_mass_error'] < 1e-9
