"""Classic MM5 surface layer (sf_sfclay_physics=91) against WRF 4.6.1, word for word.

``tools/sfclay_classic_wrf461_oracle/`` drives the pinned
``phys/module_sf_sfclay.F`` (SFCLAYINIT + SFCLAY, every optional argument an
EM build passes) on 151 columns in 11 option groups for three chained calls,
and dumps every one of the 34 fields SFCLAY writes, plus both 1001-entry
stability tables.  The fixture in ``tests/data/oracles/sfclay_classic/`` is
that dump; this module runs ``kernels/sfclay.cu`` (option 91 dispatches to
``sfclay_classic.cuh``) on the same input words and compares.

Before this file the scheme had no WRF number at all: the kernel was checked
against ``gpuwm.verify.npref.np_sfclay``, a float64 mirror of the same
transcription, with a relative tolerance.  The first oracle run measured
the shipped kernel off WRF by up to 1.9e5 ULP (see tools/sfclay_classic_wrf461_oracle/README.md).

The pristine scalar reference is the -O0 object (``wrf-O0.txt``). ``wrf-stock.txt`` is the
same source at WRF's own configure.wrf flags, and ``wrf-built.txt`` the object
inside a real wrf.exe; the two are byte-identical to each other and differ
from -O0 because gfortran routes SFCLAYINIT's table loop and SFCLAY1D's
simple loops through libmvec there.  That difference is measured in the
report. The numerical gate uses ``wrf-ck-corrected.txt`` as described below.

Declared divergence: WRF module_sf_sfclay.F:767 uses PSIH in PSIQ10.
WOOF fixes this WRF bug by using PSIH10. The scalar referee used here has
only that line corrected by wrf-ck-correction.patch. All 15,402 words,
including CK, are graded without an exclusion. The pristine scalar and
stock references remain for the compiler comparison.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "sfclay_classic"
HARNESS = ROOT / "tools" / "sfclay_classic_wrf461_oracle"


def _harness(name):
    spec = importlib.util.spec_from_file_location(
        f"_sfclay_classic_{name}", HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = _harness("validate_sfclay_classic_oracle")
COLUMNS = _harness("make_columns")


# ---------------------------------------------------------------- CPU checks

def test_fixture_files_are_the_ones_the_oracle_wrote():
    sums = {}
    for line in (FIXTURE / "oracle-sha256sums.txt").read_text().splitlines():
        digest, name = line.split()
        sums[name] = digest
    assert set(sums) == {"columns.txt", "wrf-O0.txt", "wrf-stock.txt",
                         "wrf-built.txt", "wrf-ck-corrected.txt"}
    for name, digest in sums.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name
    # The pinned WRF source the oracle compiled (LF checkout).
    assert (FIXTURE / "source-sha256.txt").read_text().split()[0] == (
        "dfd7aae5ef1f0bd979246480af5276a1ee956bc216ec8dc6e82eb963ded761dc")


def test_fixture_reaches_every_regime_surface_and_option_group():
    """The coverage claim, read from WRF's own output words."""
    groups, nstep = VALIDATE.load_columns(FIXTURE / "columns.txt")
    _table, rows = VALIDATE.load_wrf(FIXTURE / "wrf-O0.txt")
    assert nstep == 3
    assert sum(len(g[5]["u"]) for g in groups) >= 64
    regime_col = VALIDATE.OUTPUT_FIELDS.index("regime")
    seen = set()
    for gi, (_n, _a, _b, _c, _dx, fields) in enumerate(groups, 1):
        water = fields["xland"] > 1.5
        for step in range(1, nstep + 1):
            reg = rows[(gi, step)][:, regime_col].view(np.float32).astype(int)
            seen |= {(int(r), bool(w)) for r, w in zip(reg, water)}
    assert {1, 2, 3, 4} <= {r for r, _w in seen}
    assert {(1, False), (2, False), (3, False), (4, False),
            (4, True)} <= seen
    assert any(w for r, w in seen if r in (1, 2))   # stable over water
    options = {(g[1], g[2], g[3]) for g in groups}
    assert {(1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 0, 1), (1, 0, 2),
            (0, 0, 0), (1, 1, 1)} <= options
    assert {g[4] for g in groups} >= {3000.0, 5000.0, 12000.0}


def test_stock_wrf_object_differs_from_the_scalar_reference_only_by_libmvec():
    """The stock build is measured, so pin what was measured.

    Same columns, same source, WRF's -O2 -ftree-vectorize: SFCLAYINIT's loop
    and SFCLAY1D's THGB/THCON loops take libmvec's SSE logf/atanf/powf.  The
    object inside a real wrf.exe gives the same words as the stock-flag
    recompile.
    """
    t0, r0 = VALIDATE.load_wrf(FIXTURE / "wrf-O0.txt")
    ts, rs = VALIDATE.load_wrf(FIXTURE / "wrf-stock.txt")
    tb, rb = VALIDATE.load_wrf(FIXTURE / "wrf-built.txt")
    assert all((tb[k] == ts[k]).all() for k in ts)
    assert all((rb[k] == rs[k]).all() for k in rs)
    assert int((t0["psim"] != ts["psim"]).sum()) == 340
    assert int((t0["psih"] != ts["psih"]).sum()) == 254
    assert int(VALIDATE.ulp_distance(t0["psim"], ts["psim"]).max()) == 32
    assert int(VALIDATE.ulp_distance(t0["psih"], ts["psih"]).max()) == 3
    toolchain = (FIXTURE / "toolchain.txt").read_text()
    assert "libmvec symbols, -O0 object:  0" in toolchain
    for symbol in ("_ZGVbN4v_atanf", "_ZGVbN4v_logf", "_ZGVbN4vv_powf"):
        assert symbol in toolchain


def test_classic_atanf_words_are_the_mynn_units_words():
    """sfc_classic_atanf and mynn_glibc_atanf are one routine; hold the words equal."""
    kernels = ROOT / "gpuwm" / "core" / "kernels"

    def words(path, name):
        text = path.read_text(encoding="utf-8")
        body = re.search(name + r"\[19\]\s*=\s*\{([^}]*)\}", text).group(1)
        return [int(w, 16) for w in re.findall(r"0x([0-9A-Fa-f]+)u", body)]

    ours = words(kernels / "sfclay_classic.cuh", "SFC_CLASSIC_ATANF_TAB")
    theirs = words(kernels / "mynn_dmp_sibling.cu", "MYNN_ATANF_TAB")
    assert len(ours) == 19 and ours == theirs


def test_sfclay_compiles_without_contraction_and_with_its_headers():
    from gpuwm.core import kernels
    assert kernels.module_options("sfclay") == ("-std=c++17", "--fmad=false", "--ftz=false")
    assert kernels.EXTRA_HEADERS["sfclay"] == ("glibc_flt32.cuh",
                                               "sfclay_classic.cuh")
    src = kernels.module_source("sfclay")
    assert src.index("gfk_log(") < src.index("sfclay_classic_point(")
    assert src.index("sfclay_classic_point(") < src.index("void sfclay_column(")


def test_ensemble_shared_compile_takes_the_units_own_options():
    """The member-shared sfclay/ysu compile must not contract FMAs the
    native compile does not: the ensemble result would differ from the
    single-member kernel's words (test_mm5_one_launch_matches_each_original_member)."""
    import inspect
    from gpuwm.ensemble import batch_physics
    text = inspect.getsource(batch_physics._native_column_kernel)
    assert "kernels.module_options(module)" in text


# ---------------------------------------------------------------- GPU checks

@pytest.mark.gpu
@requires_gpu
def test_stability_tables_are_wrfs_words():
    table, _rows = VALIDATE.load_wrf(FIXTURE / "wrf-O0.txt")
    got = VALIDATE.woof_table()
    for name in ("psim", "psih"):
        bad = np.nonzero(got[name] != table[name])[0]
        assert bad.size == 0, (name, bad[:10].tolist())


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", ["free", "replay"])
def test_every_output_word_is_wrfs_word(mode):
    """All 15,402 words match the corrected referee in both modes."""
    res = VALIDATE.grade(FIXTURE / "columns.txt", FIXTURE / "wrf-ck-corrected.txt")
    r = res[mode]
    bad = {f: n for f, n in r["differing_words"].items() if n}
    assert not bad, bad
    assert r["words_compared"] == 15_402



@pytest.mark.gpu
@requires_gpu
def test_classic_atanf_is_the_mynn_units_atanf_everywhere():
    """Sweep the two copies of the one atanf against each other."""
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    words = np.arange(0, 1 << 32, 4099, dtype=np.uint64).astype(np.uint32)
    x = cp.asarray(words.view(np.float32))
    n = x.size
    ours = cp.empty_like(x)
    get_kernel("sfclay", "sfclay_classic_atanf_probe")(
        ((n + 255) // 256,), (256,), (x, ours, np.int32(n)))
    out3 = cp.empty(3 * n, cp.float32)
    get_kernel("mynn_dmp_sibling", "mynn_glibc_libm_probe")(
        ((n + 255) // 256,), (256,), (x, x, out3, np.int32(n)))
    a = ours.get().view(np.uint32)
    b = out3.get()[1::3].view(np.uint32)
    nan = np.isnan(a.view(np.float32)) & np.isnan(b.view(np.float32))
    mismatch = np.nonzero((a != b) & ~nan)[0]
    assert mismatch.size == 0, words[mismatch[:10]].tolist()
