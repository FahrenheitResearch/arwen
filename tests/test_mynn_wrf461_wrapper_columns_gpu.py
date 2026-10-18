"""WOOF's MYNN entry against WRF v4.6.1's ``mynnedmf_wrapper_run``, word for word.

The fixtures ``tests/data/oracles/mynn/wrapper-columns-<variant>-ml<n>.npz``
come from ``tools/mynn_pbl_wrf461_oracle/build_columns.sh``: the
unmodified WRF v4.6.1 ``ccpp_kind_types.F``, ``module_model_constants.F``,
``module_bl_mynn_common.F``, ``module_bl_mynn.F`` and
``module_bl_mynn_wrapper.F`` (no stub), gfortran 13.3.0 -O0
-ffp-contract=off, glibc 2.39, driven by ``run_wrapper_columns.F90``.

80 realistic columns (convective and stable land, ocean, high terrain,
snow and ice, moist convection, high wind, and 14 edge cases) in seven
(dx, delt, FLAG_QS) groups from 100 m to 27 km, 40 levels, a cold start and
eleven warm steps, for both mixing lengths and three option sets: the
default (``base``), ``bl_mynn_mixscalars = 1`` with the mp = 28 number
flags (``mixscalars``), and ``spp_pbl = 1`` (``spp``).

The device entry :func:`gpuwm.core.mynn_pbl_runtime.mynn_pbl_step` -- the
one ``physics.py`` calls, wrapper conversions included -- must land on
every output word of every step, both from the oracle's recorded inputs
(``replay``) and free-running from step 1 (``free``).  It does under
default arithmetic and under ``GPUWM_WRF_EXACT=1``; no output has an
allowance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = ROOT / "tests" / "data" / "oracles" / "mynn"
FIXTURES = tuple(f"wrapper-columns-{variant}-ml{ml}.npz"
                 for variant in ("base", "mixscalars", "spp")
                 for ml in (1, 2))


def _tool():
    name = "_mynn_wrapper_columns"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "tools" / "mynn_pbl_wrf461_oracle" / "wrapper_columns.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # The tool's dataclass resolves its string annotations through
    # sys.modules, so the module must be registered before it executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", FIXTURES)
def test_the_wrapper_column_fixture_is_whole(name):
    """What the GPU test grades is what the report claims it grades."""
    tool = _tool()
    fixture = tool.load_fixture(ORACLE_DIR / name)
    meta = fixture["meta"]
    assert meta["nstep"] == 12 and meta["nz"] == 40
    assert meta["variant"] in name and f"ml{meta['mixlength']}" in name
    assert len(fixture["names"]) == 80
    assert set(fixture["regimes"]) == {
        "convective", "stable", "ocean", "high_terrain", "snow_ice",
        "moist", "high_wind", "edge"}
    assert sorted(g[3] for g in fixture["groups"]) == sorted(
        dx for dx, _, _ in tool.GROUPS.values())
    for key, value in fixture.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            assert np.isfinite(value).all(), key
    # Each option set reaches the code it names.
    rqn = fixture["out_rqnwfablten"]
    assert np.count_nonzero(rqn) > 0 if meta["variant"] == "mixscalars" \
        else not rqn.any()
    if meta["variant"] == "spp":
        assert np.abs(fixture["prof_spp"]).max() > 0.5


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("replay", (True, False), ids=("replay", "free"))
@pytest.mark.parametrize("name", FIXTURES)
def test_the_device_entry_is_bitwise_wrf_on_every_column_and_step(name,
                                                                  replay):
    tool = _tool()
    fixture = tool.load_fixture(ORACLE_DIR / name)
    report = tool.grade(fixture, tool.run_woof(fixture, replay=replay))
    differ = {key: (row["words_differ"], row["max_ulp"], row["per_regime"])
              for key, row in report.items() if row["words_differ"]}
    assert differ == {}
