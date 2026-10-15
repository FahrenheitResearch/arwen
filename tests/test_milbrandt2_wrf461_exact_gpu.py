"""Milbrandt-Yau (mp_physics = 9) against unmodified WRF v4.6.1, word for word.

WRF v4.6.1's own ``mp_milbrandt2mom_driver`` (the object a GNU WRF build
links, gfortran 13.3, glibc 2.39; tools/milbrandt2_wrf461_oracle) was run
on four column sets -- 91 named-regime and edge-case columns at dt 20 s and
60 s, and 512 randomised columns at 50 and 80 levels -- for several
successive calls each, and its every output recorded
(tests/data/oracles/milbrandt2).  The shipped launcher
``gpuwm.core.milbrandt2.launch_milbrandt2`` must reproduce every word of
every call: qv, the six masses, the six numbers, theta, Zet, the eight
precipitation accumulators and SR.  The device integrates from its own
output after the first call, so a single rounding difference anywhere
would compound into later calls.

The process's arithmetic is whatever the environment selects: the default
build and the strict GPUWM_WRF_EXACT=1 build must both pass, because the
unit compiles without FMA contraction in both
(gpuwm/core/kernels/__init__.py ``_NO_FMAD_MODULES``) and calls WOOF's own
gfk_exp/gfk_log/gfk_log10/gfk_pow, never CUDA's.

The control proves the comparison can see a difference at all: the same
run with CUDA's ``powf`` compiled in place of ``gfk_pow`` must NOT match.
"""

from __future__ import annotations

import hashlib
import importlib.util
import lzma
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPOSITORY_ROOT / "tests" / "data" / "oracles" / "milbrandt2"
HARNESS = REPOSITORY_ROOT / "tools" / "milbrandt2_wrf461_oracle"

#: set -> (sha256 of the uncompressed inputs, of WRF's outputs)
SETS = {
    "dt20": ("2834b5ddcc1238851d1801d2e783f29efd1cebdc1e4331d3e5a6bfcda7433c59",
             "daea7fbe42477e846508af616180b383eefaf536606b9a7946c890191464c6dc"),
    "dt60": ("474ab1c81f84eefbf797b177bb9cbd83cb324df2a226188656fb5e46bd997ad8",
             "f0445f82103d1707de41bacbcdfb6f3ea5316f9af3bd397bd529f23920dea524"),
    "stress50": ("7233bf199a6ead73572b32d891d22a5612e7e742d6c9792101ff88af7d3d0b44",
                 "1a2fa644f05d25d1f4d652ed6f4204e036a7bb24cdaaa411e71c1132ad52883b"),
    "stress80": ("2cfc5504c3dab9e5411416ee30f3caa71a35d0875d5c111730830082c99f8f41",
                 "84a0b44374f27b21e29a84b5cbe4eda019b80be2fdc76d43ec7fd0cda074ff0c"),
}


def _harness(name):
    path = HARNESS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"my2_oracle_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(set_name, tmp_path):
    raw_in = lzma.decompress((FIXTURES / f"in_{set_name}.bin.xz").read_bytes())
    raw_out = lzma.decompress((FIXTURES / f"wrf_{set_name}.bin.xz").read_bytes())
    want_in, want_out = SETS[set_name]
    assert hashlib.sha256(raw_in).hexdigest() == want_in, set_name
    assert hashlib.sha256(raw_out).hexdigest() == want_out, set_name
    path_in = tmp_path / f"in_{set_name}.bin"
    path_out = tmp_path / f"wrf_{set_name}.bin"
    path_in.write_bytes(raw_in)
    path_out.write_bytes(raw_out)
    return path_in, path_out


@pytest.mark.parametrize("set_name", sorted(SETS))
def test_fixtures_are_the_recorded_bytes(set_name, tmp_path):
    """CPU: the committed fixtures decompress to the digests in the README."""
    path_in, path_out = _fixture(set_name, tmp_path)
    gen = _harness("gen_columns")
    compare = _harness("compare")
    ncol, nz, nsteps, dt, _, _ = gen.read_stream(path_in)
    out = compare.read_output(path_out, ncol, nz, nsteps)
    assert all(np.all(np.isfinite(v)) for v in out.values())
    readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
    for digest in SETS[set_name]:
        assert digest in readme


def _run_and_compare(set_name, tmp_path, layout=None):
    path_in, path_out = _fixture(set_name, tmp_path)
    gen = _harness("gen_columns")
    compare = _harness("compare")
    woof = _harness("woof_run")
    ncol, nz, nsteps, dt, _, _ = gen.read_stream(path_in)
    ref = compare.read_output(path_out, ncol, nz, nsteps)
    got = woof.run(path_in, tmp_path / f"woof_{set_name}.bin", layout)
    rows = compare.compare(ref, got)
    return {name: (differ, max_ulp) for name, differ, _size, max_ulp, _abs, _b
            in rows if differ}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("set_name", sorted(SETS))
def test_device_scheme_is_bitwise_wrf_on_every_output_word(set_name,
                                                           tmp_path):
    misses = _run_and_compare(set_name, tmp_path)
    assert misses == {}, (
        f"{set_name}: fields missing WRF v4.6.1 (differing words, max ULP): "
        f"{misses}")


@pytest.mark.gpu
@requires_gpu
def test_column_layout_does_not_move_a_word(tmp_path):
    """The same columns on a (4, 64) grid instead of (1, 256)."""
    misses = _run_and_compare("stress50", tmp_path, layout=(4, 64))
    assert misses == {}, misses


@pytest.mark.gpu
@requires_gpu
def test_the_comparison_sees_cudas_powf(tmp_path, monkeypatch):
    """Control: with CUDA's powf in place of gfk_pow the oracle must miss."""
    import cupy as cp

    import gpuwm.core.milbrandt2 as my2
    from gpuwm.core.kernels import module_options, module_source

    source = module_source("milbrandt2")
    assert source.count("gfk_pow(") > 60
    # Leave the definition in the prepended header alone; swap the calls.
    head, sep, body = source.partition("// gpuwm/core/kernels/milbrandt2.cu")
    assert sep
    mutated = cp.RawModule(code=head + sep + body.replace("gfk_pow(", "powf("),
                           options=module_options("milbrandt2"))
    real_get = my2.get_kernel

    def get_kernel(module, name):
        if module == "milbrandt2":
            return mutated.get_function(name)
        return real_get(module, name)

    monkeypatch.setattr(my2, "get_kernel", get_kernel)
    misses = _run_and_compare("dt20", tmp_path)
    assert misses, "CUDA's powf matched WRF bitwise; the comparison is blind"
