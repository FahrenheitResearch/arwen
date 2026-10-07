"""The GSL WRF 3.9 fork's MYNN surface layer (mynn_sfclay_variant = "gsl_wrf39").

Oracle: the unmodified fork module (NOAA-EMC/HRRR v4.1.21,
sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_sf_mynn.F, SHA-256 47bc9943...)
compiled with gfortran by tools/mynn_gsl_wrf39_oracle/build.sh over the
284-column deck of make_columns.py, six carried steps.  The deck binds the
fork's give-up branch (z/L = 8 Ri after five secant passes, forest and urban
columns in moderately stable air), the Richardson clamp at 50, the z/L cap
at 50 and the neutral branch's z/L from MOL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
ORACLE_DIR = ROOT / "tests" / "data" / "oracles" / "mynn"
ORACLE = ORACLE_DIR / "surface-layer-gsl-wrf39.csv.gz"
DECK = ORACLE_DIR / "surface-layer-gsl-wrf39-columns.txt"
TOOLS = ROOT / "tools" / "mynn_gsl_wrf39_oracle"
sys.path.insert(0, str(TOOLS))

import compare_fork_oracle as cmp  # noqa: E402

#: Maximum FP32 ULP distance of the CPU reference from the fork oracle, per
#: output, over all 1704 (step, column) words, measured on node-1 (Ubuntu,
#: glibc 2.39) 2026-10-03.  The column solver's own np.log/np.exp/** sites
#: (shared with the wrf_461 reference) are the residue; the search, the
#: psi tables and the fork's new log terms are glibc-routed.
CPU_MAX_ULP = {
    "regime": 0, "zol": 19, "rmol": 19, "ust": 2, "ustm": 2, "mol": 4,
    "psim": 52, "psih": 7, "chs": 6, "chs2": 5, "cqs2": 5, "ch": 6,
    "flhc": 7, "flqc": 6, "qgh": 0, "qsfc": 1, "hfx": 8, "qfx": 10,
    "lh": 10, "u10": 2, "v10": 2, "th2": 1, "t2": 1, "q2": 3, "gz1oz0": 1,
    "wspd": 2, "br": 4, "ck": 12, "cka": 13, "cd": 19, "cda": 10,
    "wstar": 3, "qstar": 6, "cpm": 0, "znt": 3,
}


@pytest.fixture(scope="module")
def oracle():
    return cmp.load_oracle(ORACLE)


@pytest.fixture(scope="module")
def cpu_fork(oracle):
    return cmp.run_reference(DECK, max(oracle), "gsl_wrf39")


def test_deck_is_the_generator_output(tmp_path):
    from make_columns import write_deck
    out = tmp_path / "columns.txt"
    write_deck(out, 6)
    assert out.read_bytes() == DECK.read_bytes()


def test_oracle_binds_every_fork_branch(oracle):
    """The fork's give-up, clamp, cap and neutral branches are all live."""
    giveup = cap = clamp = neutral = 0
    for step, out in oracle.items():
        stable = out["regime"] <= 2
        giveup += int(((out["zol"] == np.float32(8.0) * out["br"])
                       & stable & (out["zol"] < 50)).sum())
        cap += int((out["zol"] >= np.float32(50.0)).sum())
        clamp += int((out["br"] > 4.0).sum())
        neutral += int(((out["regime"] == 3) & (out["zol"] != 0)).sum())
    assert giveup >= 20 and cap >= 40 and clamp >= 5 and neutral >= 1


def test_cpu_reference_tracks_the_fork_oracle(oracle, cpu_fork):
    table = cmp.ulp_table(cpu_fork, oracle)
    over = {k: (v["max"], CPU_MAX_ULP[k]) for k, v in table.items()
            if v["max"] > CPU_MAX_ULP[k]}
    assert not over, over
    exact = sum(v["exact"] for v in table.values())
    assert exact >= 54418, exact


def test_wrf_461_form_is_not_the_fork(oracle):
    """The default stays WRF v4.6.1's: far from the fork on these columns."""
    table = cmp.ulp_table(cmp.run_reference(DECK, max(oracle), "wrf_461"),
                          oracle)
    assert table["zol"]["exact"] < 50 and table["zol"]["max"] > 10**6


def _cfg(**kwargs):
    from gpuwm.config import RunConfig
    base = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                dt=6.0, run_seconds=60.0)
    base.update(kwargs)
    return RunConfig(**base)


def test_unknown_variant_is_refused():
    from gpuwm.config import validate_run_config
    from gpuwm.core.mynn_surface import (
        mynn_sfclay_variant_form, mynn_surface_layer_default)
    with pytest.raises(ValueError, match="mynn_sfclay_variant"):
        mynn_sfclay_variant_form("hrrr")
    _, cols = __import__("make_columns").load_columns(DECK)
    with pytest.raises(ValueError, match="mynn_sfclay_variant"):
        mynn_surface_layer_default(cols, variant="wrf_39")
    with pytest.raises(ValueError, match="mynn_sfclay_variant"):
        validate_run_config(_cfg(mynn_sfclay_variant="fork"))
    validate_run_config(_cfg(mynn_sfclay_variant="gsl_wrf39"))


def test_default_is_wrf_461():
    assert _cfg().mynn_sfclay_variant == "wrf_461"


def test_restart_flip_refuses_and_the_old_header_reads_as_the_default():
    from dataclasses import asdict, replace
    from gpuwm.io.restart import (_configuration_digest_values,
                                  _require_config_match, configuration_echo)
    cfg = _cfg()
    stored = asdict(cfg)
    stored.pop("mynn_sfclay_variant")
    _require_config_match(stored, cfg, "checkpoint")
    assert _configuration_digest_values(stored) ==         _configuration_digest_values(asdict(cfg))
    assert "mynn_sfclay_variant" not in configuration_echo(cfg)
    fork = replace(cfg, mynn_sfclay_variant="gsl_wrf39")
    assert configuration_echo(fork)["mynn_sfclay_variant"] == "gsl_wrf39"
    with pytest.raises(ValueError, match="mynn_sfclay_variant"):
        _require_config_match(stored, fork, "checkpoint")
    with pytest.raises(ValueError, match="mynn_sfclay_variant"):
        _require_config_match(asdict(fork), cfg, "checkpoint")


def test_the_variant_is_preparation_inert():
    from gpuwm.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS
    assert "run.mynn_sfclay_variant" in PREPARATION_INERT_RUN_FIELDS


def test_spp_source_keeps_its_default_and_carries_the_define():
    from gpuwm.core.spp_kernel_sources import specialized_source
    plain = specialized_source("mynn_surface")
    fork = specialized_source(
        "mynn_surface", defines=(("MYNN_SFCLAY_GSL_WRF39", 1),))
    assert "#define MYNN_SFCLAY_GSL_WRF39" not in plain
    assert fork == "#define MYNN_SFCLAY_GSL_WRF39 1\n" + plain
    with pytest.raises(ValueError):
        specialized_source("mynn_surface", defines=(("OTHER", 1),))


def test_default_kernel_source_compiles_the_wrf_461_lines():
    """Without the define the preprocessor keeps exactly the v4.6.1 lines."""
    text = (ROOT / "gpuwm" / "core" / "kernels" / "mynn_surface.cu").read_text()
    assert text.count("#ifdef MYNN_SFCLAY_GSL_WRF39") == 9
    assert "mynn_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol)" in text


def _cuda():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("no CUDA device")
    return cupy


@pytest.mark.gpu
def test_kernel_fork_tracks_cpu_and_oracle(oracle, cpu_fork):
    _cuda()
    kernel = cmp.run_kernel(DECK, max(oracle), "gsl_wrf39")
    vs_cpu = cmp.ulp_table(kernel, cpu_fork)
    vs_oracle = cmp.ulp_table(kernel, oracle)
    # REGIME and the give-up decisions must agree word for word.
    assert vs_cpu["regime"]["max"] == 0 and vs_oracle["regime"]["max"] == 0
    giveup_k = sum(int(((k["zol"] == np.float32(8.0) * k["br"])
                        & (k["regime"] <= 2) & (k["zol"] < 50)).sum())
                   for k in kernel.values())
    giveup_o = sum(int(((o["zol"] == np.float32(8.0) * o["br"])
                        & (o["regime"] <= 2) & (o["zol"] < 50)).sum())
                   for o in oracle.values())
    assert giveup_k == giveup_o
    # A matching count can hide decisions swapped between columns.
    for step, k in kernel.items():
        o = oracle[step]
        np.testing.assert_array_equal(
            (k["zol"] == np.float32(8.0) * k["br"])
            & (k["regime"] <= 2) & (k["zol"] < 50),
            (o["zol"] == np.float32(8.0) * o["br"])
            & (o["regime"] <= 2) & (o["zol"] < 50))
    worst = {k: v["max"] for k, v in vs_oracle.items()}
    assert max(worst.values()) <= KERNEL_MAX_ULP, worst


#: Kernel against the fork oracle, worst output over all 1704 words,
#: measured on the RTX 4090 (sm_89) 2026-10-03: qfx/lh/qstar 107, z/L 19.
#: CUDA's logf/powf are not glibc's; the give-up decisions and REGIME agree
#: word for word.
KERNEL_MAX_ULP = 107


@pytest.mark.gpu
def test_kernel_default_unchanged_by_the_variant_plumbing(oracle):
    """wrf_461 through the new launcher argument is the pre-variant kernel."""
    cupy = _cuda()
    from gpuwm.core.kernels import get_kernel
    from gpuwm.core.mynn_sfclay import mynn_surface_kernel
    assert mynn_surface_kernel("wrf_461") is get_kernel(
        "mynn_surface", "mynn_surface_column")
    a = cmp.run_kernel(DECK, 3, "wrf_461")
    cpu = cmp.run_reference(DECK, 3, "wrf_461")
    table = cmp.ulp_table(a, cpu)
    assert table["regime"]["max"] == 0
    del cupy


def _leaf_oracle():
    return np.genfromtxt(ORACLE_DIR / "zolri-gsl-wrf39.csv", delimiter=",",
                         names=True, dtype=np.float32)


def test_leaf_search_binds_unstable_giveup_and_sign_reset():
    """These branches were absent from the full-column oracle deck."""
    from gpuwm.core.mynn_surface import _zolri, _zolri2
    rows = _leaf_oracle()
    unstable = rows["ri"] < 0
    assert np.count_nonzero(rows["zol"][unstable]
                           == np.float32(5) * rows["ri"][unstable]) >= 4
    for row in rows:
        ri, za, z0, zt, guess = (row[k] for k in
                                 ("ri", "za", "z0", "zt", "guess"))
        assert _zolri(ri, za, z0, zt, guess) == row["zol"]
        residual, reset_arg = _zolri2(-guess, ri, za, z0, zt)
        assert reset_arg == row["reset_arg"] == np.float32(0)
        assert residual == row["residual"]


def test_nonfinite_iterate_takes_defined_giveup_before_table_lookup():
    from gpuwm.core.mynn_surface import _zolri
    for ri, guess, factor in ((0.1, np.inf, 8), (-0.1, -np.inf, 5)):
        assert _zolri(ri, 8, 0.8, 0.001, guess) == np.float32(ri) * factor


@pytest.mark.gpu
def test_kernel_leaf_search_binds_unstable_giveup_and_sign_reset():
    cp = _cuda()
    from gpuwm.core.kernels import module_source
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    rows = _leaf_oracle()
    inputs = np.column_stack([rows[k] for k in
                               ("ri", "za", "z0", "zt", "guess")])
    inputs = np.vstack([inputs, np.array([
        [0.1, 8, 0.8, 0.001, np.inf],
        [-0.1, 8, 0.8, 0.001, -np.inf]], dtype=np.float32)])
    source = "#define MYNN_SFCLAY_GSL_WRF39 1\n" + module_source("mynn_surface")
    source += r'''
extern "C" __global__ void probe_zolri(const real *x, real *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const real *p = x + 5 * i;
    y[3 * i] = mynn_zolri(p[0], p[1], p[2], p[3], p[4]);
    real z = -p[4];
    y[3 * i + 2] = mynn_zolri2(z, p[0], p[1], p[2], p[3]);
    y[3 * i + 1] = z;
}
'''
    kernel = cp.RawModule(code=source, options=("-std=c++17",)).get_function(
        "probe_zolri")
    out = cp.empty((len(inputs), 3), dtype=cp.float32)
    kernel((1,), (32,), (cp.asarray(inputs), out, np.int32(len(inputs))))
    all_got = cp.asnumpy(out)
    np.testing.assert_array_equal(all_got[-2:, 0],
                                  inputs[-2:, 0] * np.array([8, 5], np.float32))
    got = all_got[:len(rows)]
    want = np.column_stack([rows[k] for k in ("zol", "reset_arg", "residual")])
    # Give-up values and the sign reset are arithmetic identities.
    giveup = rows["zol"] == np.where(rows["ri"] < 0, np.float32(5),
                                      np.float32(8)) * rows["ri"]
    np.testing.assert_array_equal(got[giveup, 0], want[giveup, 0])
    np.testing.assert_array_equal(got[:, 1:], want[:, 1:])
    assert fp32_ulp_distance(got[:, 0], want[:, 0]).max() <= 16
