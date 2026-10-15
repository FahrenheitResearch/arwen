"""Grade source-produced words, never a Python reimplementation of WRF."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.urban_tables import load_urban_params


@pytest.mark.parametrize('tag,generation', [
    ('v4.6.1', 'wrf_461'), ('v4.7.0', 'wrf_470'),
    ('v4.7.1', 'wrf_471'), ('v4.8.0', 'wrf_480'),
])
@pytest.mark.parametrize('option', (1, 2, 3))
@pytest.mark.parametrize('lcz', (0, 1))
def test_release_table_words(tag, generation, option, lcz):
    root_text = os.environ.get('GPUWM_URBAN_RELEASE_ORACLE')
    if not root_text:
        pytest.skip('build_tables.sh release oracle not supplied')
    root = Path(root_text)
    params = load_urban_params(option, lcz, generation=generation,
                               tbl_dir=root / tag)
    with (root / tag / f'opt{option}_lcz{lcz}.csv').open() as handle:
        for name, index, word in csv.reader(handle):
            value = np.asarray(params.values[name])
            got = int(value.reshape(-1, order='F').view(np.uint32)[int(index)-1])
            assert got == int(word, 16), (tag, option, lcz, name, index,
                                         f'{got:08x}', word)
