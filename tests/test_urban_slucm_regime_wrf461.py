"""SLUCM (sf_urban_physics = 1) against WRF v4.6.1 across weather regimes.

Fixture: tests/data/oracles/urban/slucm_regime461, written by
tools/urban_slucm_regime_oracle (byte-unmodified WRF v4.6.1
module_sf_urban.F, gfortran 15.2 -O0, glibc 2.43; PROVENANCE.md there).
Eighteen regimes x every NLCD and LCZ urban category x six carried steps,
under fourteen switch combinations: 8,856 column steps, 63 float32 words
each (19 outputs, 44 state words after the call).

The claim is bitwise: every word of every row, both from WRF's own state at
entry (replay) and with the kernel carrying its own state (carried), under
whatever arithmetic the process runs (GPUWM_WRF_EXACT=1 strict, or default).

Two WRF behaviours are not copied, and each is pinned here by a control:

* IMP_SCHEME = 2 water retention.  v4.6.1 subtracts the kinematic vapour
  flux from a water depth without the RHOO/1000 unit conversion; v4.7.1
  added it (module_sf_urban.F, the DrelR/DrelB/DrelG updates).  The engine
  keeps the corrected equation.  Control: the same regimes on v4.7.1's
  module (controls/v471) must match bitwise, and every row the kernel misses
  against v4.6.1 must be a row where v4.6.1 and v4.7.1 differ.
* The undefined local hour.  WRF writes the integer `tloc` only when
  AHOPTION = 1 but reads it in the IRI_SCHEME = 1 irrigation arm; with
  AHOPTION = 0 the irrigation decision is whatever the stack held (0 in the
  -finit-real=zero fixture build, so never).  The engine computes tloc by
  WRF's own formula whenever either arm needs it.  Control: the same build
  with that one guard widened (controls/tloc/CONTROL-tloc.diff) must match
  bitwise, and every row missed against the stock fixture must be a row the
  widened guard changes.
"""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify.urban_ucm_oracle import (
    UCM_ORACLE_DIR, _flatten, _inputs, _reference, _state_from,
    carried_replay, load_variant, port_params, replay,
)

ORACLE = UCM_ORACLE_DIR.parent / "slucm_regime461"
TLOC = ORACLE / "controls" / "tloc"
V471 = ORACLE / "controls" / "v471"
ROOT = Path(__file__).resolve().parents[1]

VARIANTS = (
    "slucm-nlcd-default", "slucm-lcz-default", "slucm-nlcd-ch1",
    "slucm-nlcd-ts2", "slucm-nlcd-ch1ts2", "slucm-nlcd-ahalh",
    "slucm-nlcd-imp2", "slucm-nlcd-bound2", "slucm-lcz-imp2ahalh",
    "slucm-lcz-ch1ahalh", "slucm-nlcd-gr", "slucm-nlcd-griri",
    "slucm-lcz-gr", "slucm-lcz-grch1",
)
RETENTION = ("slucm-nlcd-imp2", "slucm-lcz-imp2ahalh", "slucm-nlcd-griri")
TLOC_READ = ("slucm-lcz-grch1",)
STOCK_EXACT = tuple(n for n in VARIANTS if n not in RETENTION + TLOC_READ)


def _check_sums(directory, count):
    lines = (directory / "oracle-sha256sums.txt").read_text().splitlines()
    for line in lines:
        expected, name = line.split()
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    assert len(lines) == count


def _rows(directory, name):
    """The fixture's data rows, the variant label column dropped."""
    text = gzip.decompress((directory / f"{name}.csv.gz").read_bytes()).decode("ascii")
    lines = text.splitlines()
    return [line.split(",", 1)[1] for line in lines[1:]]


def _changed_rows(a_dir, b_dir, name):
    a, b = _rows(a_dir, name), _rows(b_dir, name)
    assert len(a) == len(b)
    return np.asarray([x != y for x, y in zip(a, b)])


def _missed_rows(port, ref):
    n = len(next(iter(ref.values())))
    missed = np.zeros(n, bool)
    for key, want in ref.items():
        got = np.asarray(port[key], np.float32)
        want = np.asarray(want, np.float32)
        missed |= got.view(np.uint32) != want.view(np.uint32)
    return missed


def _assert_words(port, ref):
    for name, want in ref.items():
        got = np.asarray(port[name], np.float32)
        want = np.asarray(want, np.float32)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), name


def test_fixture_hashes():
    _check_sums(ORACLE, 3 * len(VARIANTS))
    _check_sums(TLOC, 4)
    _check_sums(V471, 9)


def test_regimes_and_categories_are_covered():
    rows, table, _ = load_variant("slucm-lcz-default", root=ORACLE)
    assert sorted(set(rows["scenario"].tolist())) == list(range(1, 19))
    assert sorted(set(rows["utype"].tolist())) == list(range(1, 12))
    assert sorted(set(rows["step"].tolist())) == list(range(1, 7))
    rows, _, _ = load_variant("slucm-nlcd-default", root=ORACLE)
    assert sorted(set(rows["utype"].tolist())) == [1, 2, 3]
    # air density of about 1,900 m, arctic cold and a 28 m/s storm are in it
    assert float(rows["rhoo"].min()) < 1.0
    assert float(rows["ta"].min()) < 236.0
    assert float(rows["ua"].max()) >= 28.0


def test_controls_change_only_what_they_name():
    diff = (TLOC / "CONTROL-tloc.diff").read_text().splitlines()
    assert diff == ["623c623", "<    if(ahoption==1) then", "---",
                    ">    if(ahoption==1 .or. iri_scheme==1) then"]
    # The widened guard moves only the dew and irrigation nights (local hour
    # 21-22, July) -- the two regimes whose hour reaches the irrigation arm.
    changed = _changed_rows(ORACLE, TLOC, "slucm-lcz-grch1")
    rows, _, _ = load_variant("slucm-lcz-grch1", root=ORACLE)
    assert changed.any()
    assert set(rows["scenario"][changed].tolist()) == {14, 16}
    # v4.6.1 and v4.7.1 differ only in IMP_SCHEME = 2 retention, so only the
    # retention variants and only rows with water on an impervious surface.
    for name in RETENTION:
        changed = _changed_rows(ORACLE, V471, name)
        rows, _, _ = load_variant(name, root=ORACLE)
        assert changed.any(), name
        assert set(rows["scenario"][changed].tolist()) <= {3, 7, 16}, name


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", STOCK_EXACT)
def test_gpu_matches_wrf461_bitwise(name):
    port, ref, codes = replay(name, root=ORACLE)
    assert not np.any(codes)
    _assert_words(port, ref)
    carried, carried_ref = carried_replay(name, root=ORACLE)
    _assert_words(carried, carried_ref)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", RETENTION)
def test_gpu_retention_rows_are_the_v471_correction(name):
    port, ref, codes = replay(name, root=V471)
    assert not np.any(codes)
    _assert_words(port, ref)
    carried, carried_ref = carried_replay(name, root=V471)
    _assert_words(carried, carried_ref)
    port, ref, _ = replay(name, root=ORACLE)
    missed = _missed_rows(port, ref)
    assert missed.any()
    assert not np.any(missed & ~_changed_rows(ORACLE, V471, name))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", TLOC_READ)
def test_gpu_tloc_rows_are_wrfs_undefined_read(name):
    port, ref, codes = replay(name, root=TLOC)
    assert not np.any(codes)
    _assert_words(port, ref)
    carried, carried_ref = carried_replay(name, root=TLOC)
    _assert_words(carried, carried_ref)
    port, ref, _ = replay(name, root=ORACLE)
    missed = _missed_rows(port, ref)
    assert np.array_equal(missed, _changed_rows(ORACLE, TLOC, name))


@pytest.fixture(scope="module")
def host_twin(tmp_path_factory):
    from gpuwm.verify.urban_ucm_host import HostTwin

    path = os.environ.get("GPUWM_URBAN_HOST_LIBRARY")
    if path:
        return HostTwin(Path(path))
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("compiled SLUCM host twin needs a C++ compiler")
    output = tmp_path_factory.mktemp("slucm-regime") / "liburban_ucm_host.so"
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
        "-fPIC", "-shared", str(ROOT / "gpuwm/core/kernels/urban_ucm_host.cpp"),
        "-o", str(output),
    ], check=True)
    return HostTwin(output)


@pytest.mark.parametrize("name", STOCK_EXACT)
def test_host_twin_matches_wrf461_bitwise(host_twin, name):
    rows, table, switches = load_variant(name, root=ORACLE)
    params = port_params(table, switches)
    selected = np.ones(len(rows["ta"]), dtype=bool)
    output, state, codes = host_twin.run_columns(
        params, _inputs(rows, selected), _state_from(rows, selected, "_in"))
    assert not np.any(codes)
    _assert_words(_flatten(output, state), _reference(rows, selected))
