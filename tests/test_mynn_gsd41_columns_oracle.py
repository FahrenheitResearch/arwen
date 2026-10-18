"""The 84-column gsd_41 MYNN driver oracle covers what it claims (CPU).

The comparison itself is tests/test_mynn_gsd41_columns_exact_gpu.py.
"""

from __future__ import annotations

import numpy as np

import _mynn_columns_gsd41 as C


def test_the_oracle_covers_at_least_64_columns_and_its_regimes():
    oracle = C.load(asis=False)
    names = C.column_names()
    assert oracle["ncol"] == len(names) >= 64
    assert len(oracle["steps"]) == 12
    assert [s[0] for s in oracle["steps"]] == [1] + [0] * 11
    scal = oracle["scal"]
    prof = oracle["prof"]
    assert set(scal[:, C.SCALARS.index("xland")]) == {1.0, 2.0}
    assert scal[:, C.SCALARS.index("ps")].min() <= 65000.0
    assert (scal[:, C.SCALARS.index("hfx")] == 0.0).any()
    assert (scal[:, C.SCALARS.index("rmol")] > 0.0).any()
    assert (scal[:, C.SCALARS.index("rmol")] < 0.0).any()
    assert (prof[:, C.PROFILES.index("qs")] > 0.0).any()
    assert (prof[:, C.PROFILES.index("qi")] > 0.0).any()
    assert (prof[:, C.PROFILES.index("qc")] > 0.0).any()
    # float32 subnormals go in and come out of the fork (the CUDA compile
    # flushes them unless its arithmetic is written around the flag)
    tiny = np.float32(np.finfo(np.float32).tiny)

    def subnormal(a):
        return bool(((a != 0.0) & (np.abs(a) < tiny)).any())
    assert subnormal(prof[:, C.PROFILES.index("qc")])
    assert subnormal(prof[:, C.PROFILES.index("qs")])
    assert any(subnormal(s[2]["rqcblten"]) for s in oracle["steps"])
    assert any(subnormal(s[2]["qc_bl"]) for s in oracle["steps"])
    # the fork's plumes fire somewhere and stay quiet somewhere
    maxmf = np.stack([s[2]["maxmf"] for s in oracle["steps"]])
    assert (maxmf > 0.0).any() and (maxmf.max(axis=0) == 0.0).any()
