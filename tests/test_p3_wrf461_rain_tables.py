"""P3's start-up rain tables against WRF 4.6.1's own, word for word.

``gpuwm.core.p3_tables.generate_rain_tables`` transcribes
module_mp_p3.F:599-670, a 10,000-bin integral per table row whose
transcendentals gfortran sends to the C library: ``exp`` to ``expf``,
``**`` with a REAL exponent to ``powf`` and ``ALOG10`` to ``log10f``.
Through 2.8.7 the transcription called NumPy's float32 ``exp``, ``log10``
and ``**`` instead, which are different functions (AVX-512 loops on Linux,
the MSVC runtime on Windows): ``vn_table`` differed from WRF in 1,150 of
3,000 words, ``vm_table`` in 940 and ``revap_table`` in 110, and the column
oracle (tools/p3_wrf461_oracle) measured the result as up to 6e-3 relative
in rain mass and number after ONE step.  The breakage these gates prevent
is that table drifting from WRF's again.

The reference, ``tests/data/oracles/p3/wrf461_rain_tables.bin``, is the
table loop cut by line number out of the pinned, unmodified
``module_mp_p3.F`` (tools/p3_wrf461_oracle/rain_table_dump.sh), compiled
``-O0 -ffp-contract=off`` and linked with correctly rounded ``tgammaf`` and
``log10f``.  It is byte-identical to the instrumented ``p3_init`` dump taken
on a host whose C library rounds those two correctly (glibc 2.43,
evidence/p3-fortran-oracle-20260828/dumps/init_state.bin).
"""
from __future__ import annotations

import decimal
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "oracles" / "p3"


def _wrf_tables():
    raw = np.fromfile(DATA / "wrf461_rain_tables.bin", dtype="<f4")
    assert raw.size == 3 * 300 * 10
    return [raw[i * 3000:(i + 1) * 3000].reshape((300, 10), order="F")
            for i in range(3)]


def test_the_rain_tables_are_wrfs_word_for_word():
    from gpuwm.core.p3_tables import generate_rain_tables

    ours = generate_rain_tables()
    for name, mine, theirs in zip(("vn_table", "vm_table", "revap_table"),
                                  ours, _wrf_tables()):
        differing = int(np.count_nonzero(
            np.ascontiguousarray(mine).view(np.uint32)
            != np.ascontiguousarray(theirs).view(np.uint32)))
        assert differing == 0, (
            f"{name} differs from WRF 4.6.1 in {differing} of 3000 words")


def test_the_init_dump_and_the_committed_tables_agree():
    """The committed reference is the one a real p3_init produced."""
    init = (Path(__file__).resolve().parents[1] / "evidence"
            / "p3-fortran-oracle-20260828" / "dumps" / "init_state.bin")
    buf = np.fromfile(init, dtype="<f4")
    committed = np.fromfile(DATA / "wrf461_rain_tables.bin", dtype="<f4")
    # check_init.py's layout: 16 + 150 header words, then vn, vm, revap.
    assert buf[166:166 + 9000].tobytes() == committed.tobytes()


def test_expf_array_is_expf_word_for_word():
    from gpuwm.core.noahmp_libm import expf, expf_array

    rng = np.random.default_rng(20261007)
    x = np.concatenate([
        rng.uniform(-110.0, 90.0, 4000),
        -np.logspace(-8.0, 3.7, 4000),
        [0.0, -0.0, np.inf, -np.inf, np.nan, 88.72283, 88.7229,
         -103.972, -103.98, -87.4, 1e-30],
    ]).astype(np.float32)
    got = expf_array(x)
    want = np.array([expf(v) for v in x], dtype=np.float32)
    assert got.view(np.uint32).tolist() == want.view(np.uint32).tolist()


def test_log10f_cr_is_the_nearest_float():
    """Each answer is within half a unit in the last place of the exact
    logarithm, checked at 90 digits, and its neighbours are not closer."""
    from gpuwm.core.correctly_rounded_libm import log10f_cr

    rng = np.random.default_rng(7)
    x = np.concatenate([
        ((np.arange(1, 2001, dtype=np.float32) * np.float32(2.0)
          - np.float32(1.0)) * np.float32(1.0e-6)).astype(np.float32),
        rng.uniform(1e-30, 1e30, 500).astype(np.float32),
        np.float32([1.0, 10.0, 100.0, 0.1, 3.0]),
    ])
    got = log10f_cr(x)
    ctx = decimal.Context(prec=90)
    for value, answer in zip(x, got):
        exact = ctx.log10(decimal.Decimal(float(value)))
        gap = abs(exact - decimal.Decimal(float(answer)))
        for side in (-np.inf, np.inf):
            other = np.nextafter(answer, np.float32(side))
            assert gap <= abs(exact - decimal.Decimal(float(other))), value
