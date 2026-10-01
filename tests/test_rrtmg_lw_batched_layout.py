"""Exact layout and unchanged-band-body gates for the batched LW path."""
from pathlib import Path
import os
import runpy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SLABS = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")


def test_band_helpers_preserve_standalone_statements():
    from gpuwm.core import rrtmg_lw as lw
    source = lw._gpu_source()
    def body(name):
        start = source.index(name + "(")
        start = source.index("{", start)
        end, depth = start + 1, 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end]
    for band in range(1, 17):
        assert body("rlw_band%d" % band) == body("rlw_taugb%d" % band)


def compare_layout(kwargs):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import rrtmg_mcica as mc
    old = mc.gpu_generate_lw_subcolumns(**kwargs)
    for run in range(2):
        new = mc.gpu_generate_lw_subcolumns(**kwargs, layout="column")
        for name in old:
            want = old[name].transpose(1, 2, 0) if name in SLABS else old[name]
            assert new[name].shape == want.shape
            assert bool(cp.array_equal(new[name].view(cp.uint32), want.view(cp.uint32))), (name, run)


def test_mcica_column_fixture_values():
    t = runpy.run_path(str(ROOT / "tests/test_rrtmg_mcica.py"))
    count = 0
    for path in t["FIXTURES"]:
        data = t["_load"](path)
        side, kwargs = t["_fixture_kwargs"](data)
        if side == "lw" and int(data["icld"]) == 2:
            compare_layout(kwargs)
            count += 1
    assert count > 0


def test_mcica_column_real_values():
    cp = pytest.importorskip("cupy")
    from gpuwm.core import rrtmg_legacy_device as dev, rrtmg_mcica as mc
    folder = os.environ.get("GPUWM_RRTMG_LEGACY_DECKS")
    if not folder:
        pytest.skip("real prep decks not configured")
    paths = sorted(Path(folder).glob("lw_*.npz"))
    assert paths
    for path in paths:
        with np.load(path) as deck:
            kw = {k: v.item() if v.ndim == 0 else v for k, v in deck.items()}
        def generator(*args, layout):
            assert layout == "column"
            keys = ("iplon", "ncol", "nlay", "icld", "permuteseed", "irng", "play", "cldfrac", "ciwp", "clwp", "cswp", "rei", "rel", "res", "tauc", "hgt", "idcor", "juldat", "lat")
            compare_layout(dict(zip(keys, args)))
            return mc.gpu_generate_lw_subcolumns(*args, layout=layout)
        dkw = {k: cp.asarray(v) if isinstance(v, np.ndarray) and v.ndim else v
               for k, v in kw.items()}
        dev.lwrad_prep_batch_device(**dkw, subcolumn_generator=generator)


@pytest.mark.parametrize("chunk", [1, 256, 1536, 4096])
def test_column_engine_saved_oracle_dual(chunk):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import rrtmg_lw as lw
    from gpuwm.core.rrtmg_legacy import _lw_coeffs
    inputs, oracle = os.environ.get("LW_SPEED_INPUTS"), os.environ.get("LW_SPEED_ORACLE")
    if not inputs or not oracle:
        pytest.skip("saved engine inputs and oracle not configured")
    paths = sorted(Path(inputs).glob("lw_*.npz"))
    assert paths
    for path in paths:
        with np.load(path) as deck:
            ins = {k: v.item() if v.ndim == 0 else cp.asarray(v) for k, v in deck.items()}
        for name in SLABS:
            ins[name] = cp.ascontiguousarray(ins[name].transpose(1, 2, 0))
        with np.load(Path(oracle) / path.name) as want:
            for run in range(2):
                out = lw.gpu_rrtmg_lw_batched_device(**ins, C=_lw_coeffs(),
                    column_chunk=chunk, mcica_layout="column")
                for name in out:
                    assert np.array_equal(cp.asnumpy(out[name]).view(np.uint32),
                                          want[name].view(np.uint32)), (path.name, chunk, run, name)


def test_standalone_saved_columns_dual():
    pytest.importorskip("cupy")
    from gpuwm.core import rrtmg_lw as lw
    from gpuwm.core.rrtmg_legacy import _lw_coeffs
    inputs, oracle = os.environ.get("LW_SPEED_INPUTS"), os.environ.get("LW_SPEED_ORACLE")
    if not inputs or not oracle:
        pytest.skip("saved engine inputs and oracle not configured")
    paths = sorted(Path(inputs).glob("*.npz"))
    assert paths
    for path in paths:
        with np.load(path) as deck:
            ins = {k: v.item() if v.ndim == 0 else v[:, :1, :] if k in SLABS
                   else v[:1] for k, v in deck.items()}
        ins["ncol"] = 1
        with np.load(Path(oracle) / path.name) as want:
            for run in range(2):
                out = lw.gpu_rrtmg_lw(**ins, C=_lw_coeffs())
                for name in out:
                    assert np.array_equal(out[name].view(np.uint32),
                                          want[name][:1].view(np.uint32)), (path.name, run, name)


def test_gpoint_band_helpers_preserve_standalone_statements():
    from gpuwm.core import rrtmg_lw as lw
    source = lw._gpu_source()
    def body(name):
        start = source.index(name + "(")
        start = source.index("{", start)
        end, depth = start + 1, 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end]
    for band in range(1, 17):
        expected = body("rlw_taugb%d" % band).replace(
            "TAUGB_PROLOGUE", "TAUGB_GPOINT_PROLOGUE").replace(
            "for (int ig = 1; ig <= ng%d; ++ig)" % band,
            "for (int ig = gpoint; ig <= ng%d; ig += 16)" % band)
        assert body("rlw_gband%d" % band) == expected


def test_prol_gpoint_arithmetic_and_accum_order_unchanged():
    kernels = ROOT / "gpuwm/core/kernels"
    chain = (kernels / "rrtmg_lw_chain_coalesced.cu").read_text()
    staged = (kernels / "rrtmg_lw_zbatched.cu").read_text()
    def branch(source, name):
        source = source[source.index(name + "("):]
        start = source.index("        if (MC2(cldfmc)")
        end = source.index("\n    }", start)
        return source[start:end]
    assert branch(chain, "rlw_rtrn_prol_coalesced") == branch(staged, "rlw_rtrn_prol_gpoints")
    import re

    def sums(source, name):
        # The per-band sums and the band totals, from the zeroed band
        # accumulators through the last total.
        source = source[source.index(name + "("):]
        start = source.index("float urad = 0.0f, drad = 0.0f")
        end = source.index(";", source.index("totdc = RLW_AD(totdc")) + 1
        return re.sub(r"\s+", " ", source[start:end])

    # The original's slab reads, each replaced by the staged copy the row
    # kernel reads in its place; nothing else may differ.
    staged_reads = (
        ("LP0(radld_p, lane)", "down[r][lane - g_lo]"),
        ("LP0(radclrd_p, lane)", "downc[r][lane - g_lo]"),
        ("iclddn_p[((long long)col * nl + lev) * NGPTLW + lane]", "flag[r][lane - g_lo]"),
        ("iclddn_p[((long long)col * nl + 0) * NGPTLW + lane]", "flag0[r][lane - g_lo]"),
        ("radlu_sfc[(long long)col * NGPTLW + lane]", "up[r][lane - g_lo]"),
        ("LP(radlu_p, lane)", "up[r][lane - g_lo]"),
        ("radclru_sfc[(long long)col * NGPTLW + lane]", "upc[r][lane - g_lo]"),
        ("LP(radclru_p, lane)", "upc[r][lane - g_lo]"),
    )
    original = sums(chain, "rlw_rtrn_accum_coalesced")
    for read, copy in staged_reads:
        assert read in original, read
        original = original.replace(read, copy)
    assert sums(staged, "rlw_rtrn_accum_rows") == original
