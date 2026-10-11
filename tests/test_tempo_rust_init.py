"""Rust TEMPO init (tools/rustwx/crates/rw-tempo) against TEMPO's own tempo_init.

TEMPO_BLOB names the dump tools/tempo_ref/tempo_ref_driver.F90 wrote from
TEMPO 9adf9ef built with gfortran -fdefault-real-8; TEMPO_TABLE_DIR holds
the four TEMPO table files that dump read.  Skips without them or without a
built rw-tempo library.

Breakage this prevents: the GPU TEMPO runtime built from the Rust init
instead of the Fortran dump running on different parameters or tables than
TEMPO itself (a wrong constant, bin, table order or byte order).
"""
import os

import numpy as np
import pytest

BLOB = os.environ.get("TEMPO_BLOB")
TABLES = os.environ.get("TEMPO_TABLE_DIR")

pytestmark = pytest.mark.skipif(not (BLOB and TABLES),
                                reason="TEMPO_BLOB / TEMPO_TABLE_DIR not set")

# Largest relative difference accepted where a field is not bitwise.  The
# only expected source is a C library call that gfortran folds at compile
# time with MPFR (correctly rounded) while the Rust port calls the C library.
REL_TOL = 1e-13


def _rust():
    from gpuwm.core import tempo_rust
    try:
        tempo_rust.load()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"rw-tempo library not built: {error}")
    return tempo_rust


def test_rust_layout_matches_the_kernel_header():
    from gpuwm.core.tempo_runtime import KERNEL_DIR, blob_layout
    header = blob_layout((KERNEL_DIR / "tempo_common.cuh").read_text())
    assert _rust().layout() == header


def test_rust_init_matches_tempo_fortran_field_by_field():
    from gpuwm.core.tempo_runtime import KERNEL_DIR, blob_layout, load_blob
    rust = _rust()
    layout = blob_layout((KERNEL_DIR / "tempo_common.cuh").read_text())
    want = load_blob(BLOB, layout)
    got = rust.build_blob(TABLES)
    assert got.shape == want.shape
    o = 0
    bad = []
    for name, n in layout:
        a, b = got[o:o + n], want[o:o + n]
        o += n
        if np.array_equal(a.view(np.int64), b.view(np.int64)):
            continue
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        if not rel < REL_TOL:
            bad.append((name, rel))
    assert not bad, bad


def test_rust_records_file_reads_back_through_load_blob(tmp_path):
    from gpuwm.core.tempo_runtime import KERNEL_DIR, blob_layout, load_blob
    rust = _rust()
    layout = blob_layout((KERNEL_DIR / "tempo_common.cuh").read_text())
    path = tmp_path / "tempo_blob_rust.bin"
    rust.write_records(TABLES, path)
    assert np.array_equal(load_blob(path, layout), rust.build_blob(TABLES))


def test_missing_table_is_refused_by_name(tmp_path):
    rust = _rust()
    with pytest.raises(RuntimeError, match="ccn_activate.bin"):
        rust.build_blob(tmp_path)
