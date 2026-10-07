"""Fresh original WRF v4.6.1 oracle, with the retention correction declared."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify.urban_ucm_oracle import (
    UCM_ORACLE_DIR, UCM_VARIANTS, _inputs, _reference, _state_from,
    _flatten, carried_replay, load_variant, port_params, replay,
)

ORACLE = UCM_ORACLE_DIR.parent / "ucm461"
ROOT = Path(__file__).resolve().parents[1]
RETENTION_CHANGED_ROWS = {
    "ucm-nlcd-imp2": 45, "ucm-lcz-imp2ahalh": 165, "ucm-nlcd-griri": 45,
}
V461_IDENTICAL = tuple(n for n in UCM_VARIANTS if n not in RETENTION_CHANGED_ROWS)


@pytest.fixture(scope="module")
def host_twin(tmp_path_factory):
    from gpuwm.verify.urban_ucm_host import HostTwin

    path = os.environ.get("GPUWM_URBAN_HOST_LIBRARY")
    if path:
        return HostTwin(Path(path))
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("compiled SLUCM host twin needs a C++ compiler")
    output = tmp_path_factory.mktemp("slucm-host461") / "liburban_ucm_host.so"
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
        "-fPIC", "-shared", str(ROOT / "gpuwm/core/kernels/urban_ucm_host.cpp"),
        "-o", str(output),
    ], check=True)
    return HostTwin(output)


def _csv_words(root, name):
    return gzip.decompress((root / f"{name}.csv.gz").read_bytes())


def _assert_words(port, ref):
    for name, want in ref.items():
        got = np.asarray(port[name], np.float32)
        want = np.asarray(want, np.float32)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), name


def test_original_v461_fixture_hashes():
    sums = (ORACLE / "oracle-sha256sums.txt").read_text().splitlines()
    for line in sums:
        expected, name = line.split()
        assert hashlib.sha256((ORACLE / name).read_bytes()).hexdigest() == expected
    assert len(sums) == 36


@pytest.mark.parametrize("name", V461_IDENTICAL)
def test_original_v461_and_v471_match_without_retention(name):
    assert _csv_words(ORACLE, name) == _csv_words(UCM_ORACLE_DIR, name)


@pytest.mark.parametrize("name,changed", RETENTION_CHANGED_ROWS.items())
def test_corrected_retention_is_a_measured_upstream_difference(name, changed):
    old = list(csv.DictReader(io.StringIO(_csv_words(ORACLE, name).decode("ascii"))))
    new = list(csv.DictReader(io.StringIO(_csv_words(UCM_ORACLE_DIR, name).decode("ascii"))))
    assert len(old) == len(new)
    assert sum(a != b for a, b in zip(old, new)) == changed


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", V461_IDENTICAL)
def test_gpu_replays_original_v461_words(name):
    port, ref, codes = replay(name, root=ORACLE)
    assert not np.any(codes)
    _assert_words(port, ref)
    carried, carried_ref = carried_replay(name, root=ORACLE)
    _assert_words(carried, carried_ref)


@pytest.mark.parametrize("name", V461_IDENTICAL)
def test_host_replays_original_v461_words(host_twin, name):
    rows, table, switches = load_variant(name, root=ORACLE)
    params = port_params(table, switches)
    selected = np.ones(len(rows["ta"]), dtype=bool)
    output, state, codes = host_twin.run_columns(
        params, _inputs(rows, selected), _state_from(rows, selected, "_in"))
    assert not np.any(codes)
    _assert_words(_flatten(output, state), _reference(rows, selected))
    steps = sorted(set(rows["step"].tolist()))
    first = rows["step"] == steps[0]
    state = _state_from(rows, first, "_in")
    for step in steps:
        selected = rows["step"] == step
        output, state, codes = host_twin.run_columns(
            params, _inputs(rows, selected), state)
        assert not np.any(codes)
        _assert_words(_flatten(output, state), _reference(rows, selected))
