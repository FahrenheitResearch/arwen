"""CUDA snow columns against the operational branch's unchanged oracle.

The existing WRF v4.6.1 device decks run against the fork fixtures with
the snow and soil-property lineages explicitly bound to ``wrf_45``.
Every float output remains a word-for-word comparison.
"""

from functools import partial
from pathlib import Path
import shutil

import pytest

from conftest import requires_gpu
import test_ruc_gpu as T


pytestmark = [pytest.mark.gpu, requires_gpu]
FORK = Path(__file__).resolve().parents[1] / 'gpuwm' / 'data' / 'ruc' / 'oracle_fork'


@pytest.fixture
def fork_gpu(monkeypatch, tmp_path):
    oracle = tmp_path / 'gpuwm' / 'data' / 'ruc' / 'oracle'
    oracle.mkdir(parents=True)
    for name in ('snowtemp', 'snowtemp_contract', 'snowsoil', 'snowsoil_contract'):
        shutil.copyfile(FORK / f'{name}.csv', oracle / f'{name}.csv')
    (tmp_path / 'tests').mkdir()
    monkeypatch.setattr(T, '__file__', str(tmp_path / 'tests' / 'test_ruc_gpu.py'))
    monkeypatch.setattr(T, 'ORACLE', oracle / 'soilvegin.csv')
    from gpuwm.core import ruc, ruc_gpu
    for module, names in (
        (ruc, ('ruc_snow_temperature_step', 'ruc_snow_soil_step')),
        (ruc_gpu, ('ruc_snow_temperature_step_cuda', 'ruc_snow_soil_step_cuda')),
    ):
        for name in names:
            lineage = {'snow': 'wrf_45'}
            if 'soil' in name:
                lineage['soilprop'] = 'wrf_45'
            monkeypatch.setattr(module, name, partial(getattr(module, name), **lineage))
    return T


@pytest.mark.parametrize('fixture', sorted(T._SNOWTEMP_FIXTURES))
def test_fork_snow_temperature_cuda_matches_every_oracle_word(fork_gpu, fixture):
    fork_gpu.test_ruc_snow_temperature_cuda_matches_unmodified_wrf_bit_for_bit(fixture)


@pytest.mark.parametrize('fixture', sorted(T._SNOWTEMP_FIXTURES))
def test_fork_snow_temperature_cuda_batches_every_regime(fork_gpu, fixture):
    fork_gpu.test_ruc_snow_temperature_cuda_solves_independent_columns_in_one_launch(fixture)


@pytest.mark.parametrize('fixture', sorted(T._SNOWTEMP_FIXTURES))
def test_fork_snow_temperature_cuda_matches_the_host_twin(fork_gpu, fixture):
    fork_gpu.test_ruc_snow_temperature_cuda_agrees_with_the_cpu_transcription(fixture)


def test_fork_snow_soil_cuda_matches_every_oracle_word(fork_gpu):
    fork_gpu.test_ruc_snow_soil_step_cuda_matches_unmodified_wrf_bit_for_bit()


def test_fork_snow_soil_cuda_batches_every_regime(fork_gpu):
    fork_gpu.test_ruc_snow_soil_step_cuda_solves_independent_columns_in_one_launch()


def test_fork_snow_soil_cuda_matches_the_argument_contract(fork_gpu):
    fork_gpu.test_ruc_snow_soil_step_cuda_contract_fixture_matches_wrf_bit_for_bit()
