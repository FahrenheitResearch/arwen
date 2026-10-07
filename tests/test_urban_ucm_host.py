"""The compiled shared-source SLUCM host twin against the WRF column oracle."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from gpuwm.verify.urban_ucm_host import HostTwin
from gpuwm.verify.urban_ucm_oracle import (
    UCM_VARIANTS, _flatten, _inputs, _reference, _state_from,
    load_variant, port_params, ulp_table,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("compiled SLUCM host twin needs a C++ compiler")
    output = tmp_path_factory.mktemp("slucm-host") / "liburban_ucm_host.so"
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
        "-fPIC", "-shared", str(ROOT / "gpuwm/core/kernels/urban_ucm_host.cpp"),
        "-o", str(output),
    ], check=True)
    return HostTwin(output)


@pytest.mark.parametrize("name", UCM_VARIANTS)
def test_native_host_matches_each_fortran_row(twin, name):
    rows, table, switches = load_variant(name)
    selected = np.ones(rows["ta"].size, dtype=bool)
    outputs, after, codes = twin.run_columns(
        port_params(table, switches), _inputs(rows, selected),
        _state_from(rows, selected, "_in"),
    )
    assert not np.any(codes), np.unique(codes)
    differences = ulp_table(_flatten(outputs, after), _reference(rows, selected))
    assert not any(differences.values()), differences


@pytest.mark.parametrize("name", UCM_VARIANTS)
def test_native_host_carries_all_six_fortran_steps(twin, name):
    rows, table, switches = load_variant(name)
    params = port_params(table, switches)
    steps = sorted(set(rows["step"].tolist()))
    first = rows["step"] == steps[0]
    state = _state_from(rows, first, "_in")
    for step in steps:
        selected = rows["step"] == step
        outputs, state, codes = twin.run_columns(params, _inputs(rows, selected), state)
        assert not np.any(codes), np.unique(codes)
        differences = ulp_table(_flatten(outputs, state), _reference(rows, selected))
        assert not any(differences.values()), differences
