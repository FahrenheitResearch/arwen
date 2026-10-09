"""Revised MM5 land scalar roughness must keep the four stress witnesses finite."""
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from conftest import requires_gpu
from _sfclayrev_oracle import INPUT_FIELDS, _words
from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
from gpuwm.core.sfclay import sfclay, launch_sfclay


@requires_gpu
@pytest.mark.parametrize('case', [92, 221, 309, 444])
def test_land_scalar_roughness_witness_stays_finite(case):
    path = Path(__file__).parent / 'data/oracles/sfclayrev/nonfinite-inputs.hex'
    row = next(line.split()[1:] for line in path.read_text().splitlines()
               if line and not line.startswith('#') and int(line.split()[0]) == case)
    g = {k: cp.asarray(np.array([[v]], np.float32))
         for k, v in zip(INPUT_FIELDS, _words(row))}
    args = [g[k] for k in ('u', 'v', 't', 'qv', 'p', 'dz8w', 'psfc', 'tsk')]
    for flux in (True, False):
        result = sfclay(*args, *[g[k] for k in ('znt', 'pblh', 'mavail', 'xland')],
            option=1, dx=float(g['dx'].item()), isfflx=flux, iz0tlnd=2,
            **{k: g[k] for k in ('qsfc', 'zol', 'ust', 'ustm', 'mol', 'hfx', 'qfx', 'lakemask')})
        for step in range(9):
            if step:
                launch_sfclay(*args, *[g[k] for k in ('pblh', 'mavail', 'xland', 'lakemask')],
                    result, option=1, dx=float(g['dx'].item()), isfflx=flux, iz0tlnd=2)
            for name in SFCLAY_OUTPUTS:
                assert np.isfinite(cp.asnumpy(getattr(result, name))).all(), (case, flux, step, name)


@requires_gpu
def test_scalar_exponential_guard_starts_at_first_overflow_word():
    from gpuwm.core import kernels
    from gpuwm import wrf_exact
    # Compile the production exponential's own header without compiling a
    # second copy of the surface-column kernel just for four scalar words.
    source = kernels._preamble() + kernels._extra_header_text('sfclay') + '''
    extern "C" __global__ void exp_boundary(const float *x, float *out) {
        int i = threadIdx.x;
        out[i] = gfk_exp(x[i]);
    }
    '''
    options = kernels.module_options('sfclay')
    if wrf_exact.ENABLED:
        options = wrf_exact.effective_options(options)
    module = cp.RawModule(code=source, options=options)
    words = np.array([0x42b17216, 0x42b17217, 0x42b17218, 0x42b17219], np.uint32)
    values = cp.asarray(words.view(np.float32))
    out = cp.empty(4, cp.float32)
    module.get_function('exp_boundary')((1,), (4,), (values, out))
    assert np.array_equal(np.isfinite(cp.asnumpy(out)), [True, True, False, False])
    assert values[1].item() == 88.72283172607421875
