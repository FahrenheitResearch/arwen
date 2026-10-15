"""The MYJ PBL kernel against WRF v4.6.1's own MYJPBL, word for word.

``tools/myjpbl_wrf461_oracle`` compiles the byte-unmodified
``phys/module_bl_myjpbl.F`` (WRF commit d66e442f) at gfortran -O0 and drives
WRF's ``MYJPBL`` over 176 columns in five cases -- 72 columns of convective,
stable, neutral, ocean, snow, sea-ice, cloudy and edge regimes on 50 levels
with terrain from -25 m to 4500 m, called once with QCI present and once
without; eight columns at nz = 4 (the kernel's floor), eight at nz = 128
(its ceiling) and sixteen on a coarse 35-level grid at DT = 90 s -- for
three consecutive calls each, writing every input and output
(``tests/data/oracles/myjpbl/myjpbl-wrf461.npz``).  The harness also proves
that WRF's own -O2 -ftree-vectorize -funroll-loops build writes the same
bytes and that the fixture executes every line of MYJPBL's call tree except
PRODQ2's ``ELOQN <= EPS1`` arm (module_bl_myjpbl.F:1145-1146, reachable only
when the clamped ELOQ13 equals EPS1 = 1e-12).

MEASURED on W1 (RTX PRO 6000, NVRTC 12.9, 2026-10-07):

* strict build (GPUWM_WRF_EXACT=1): every word of every output field --
  TKE, EL_MYJ, EXCH_H, the six tendencies, PBLH, MIXHT, KPBL, THZ0, QZ0,
  QSFC, CT -- bit-identical on every column and call, replayed and
  free-running;
* default arithmetic: bit-identical except where CuPy's default -ftz=true
  flushes a subnormal: RQCBLTEN/RQIBLTEN at cloud edges, where diffusion
  leaves subnormal condensate, and on the column seeded with subnormal
  condensate -- every such word differs from WRF's by less than FLT_MIN
  (largest 1.2e-38).

What made it exact (each a default-on change, each needed):

* ZINT(KTE+1)=HT (:312): the kernel had seeded ground-relative heights,
  which round differently in float32 (51 ULP on PBLH, 231 on EXCH_H over
  terrain); ``test_zeroing_ht_is_seen`` keeps that sensitivity visible;
* EXP and REAL powers are WOOF's own float32 ``gfk_exp_fma``/``gfk_pow``,
  not CUDA's ``expf``/``powf``; ``tools/myjpbl_wrf461_oracle/libm_sweep.py``
  matched both against the oracle host's C library over all 2**32 inputs;
* no FMA contraction (``myjpbl`` in the loader's ``_NO_FMAD_MODULES``);
* the constants are the words gfortran folded, dumped by the driver and
  compared here by ``test_the_kernel_constants_are_wrfs_folded_words``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
STRICT = os.environ.get("GPUWM_WRF_EXACT") == "1"


def _fixture():
    from gpuwm.verify import myjpbl_oracle
    return myjpbl_oracle.load()


def test_the_fixture_reaches_the_regimes_and_arms():
    fx = _fixture()
    assert sorted(fx) == ["coarse35", "main_ice", "main_noice", "nz128", "nz4"]
    total = 0
    for case, rec in fx.items():
        ncol, nz, nstep, flqi = (int(x) for x in rec["meta"])
        total += ncol
        assert nstep == 3
        for step in (1, 2, 3):
            for name in ("tke", "el_myj", "exch_h", "rublten", "rvblten",
                         "rthblten", "rqvblten", "rqcblten"):
                assert np.isfinite(rec[f"s{step}_out_{name}"][:nz]).all(), (
                    case, step, name)
    assert total == 176
    main = fx["main_ice"]
    assert int(main["meta"][0]) >= 64
    assert int(fx["main_ice"]["meta"][3]) == 1
    assert int(fx["main_noice"]["meta"][3]) == 0
    assert int(fx["nz4"]["meta"][1]) == 4 and int(fx["nz128"]["meta"][1]) == 128
    s1 = {k[len("s1_in_"):]: v for k, v in main.items()
          if k.startswith("s1_in_")}
    assert s1["ht"].max() >= 4500.0 and s1["ht"].min() < 0.0
    assert (s1["xland"] == 2).any() and (s1["snow"] > 0).any()
    assert (s1["sice"] > 0.5).any() and (s1["chklowq"] == 0).any()
    assert (s1["ct"] != 0).any() and (s1["elflx"] < 0).any()
    assert (s1["ustar"] == 0).any() and (s1["qcw"] > 0).any()
    assert (s1["qci"] > 0).any()
    tiny = np.finfo(np.float32).tiny
    assert ((s1["qcw"] > 0) & (s1["qcw"] < tiny)).any()
    nz = int(main["meta"][1])
    kpbl = main["s1_out_kpbl"]
    assert kpbl.min() <= 2 and kpbl.max() == nz       # LPBL = 1 is reached
    # MIXLEN's CT fix moves DTH only where CT /= 0; MYJPBL zeroes CT (:852).
    assert (main["s1_out_ct"] == 0).all()


def test_the_kernel_constants_are_wrfs_folded_words():
    """Every #define literal in kernels/myjpbl.cu equals the float32 word
    WRF's -O0 objects carry (run_myjpbl.F90 put_constants)."""
    fx = _fixture()["main_ice"]
    names_a = ("g r_d cp xlv xls p608 pq0 a2 a3 a4 epsq2 vkarman capa rlivwv "
               "elocp eps1 eps2 epsl epsru epsrs epstrb fh alph beta el0max "
               "el0min elfc a1 a2x b1 b2 c1 elz0 esq pq0sea btg rb1 adnh adnm "
               "anmh anmm bdnh bdnm beqh beqm bnmh bnmm b1pow23").split()
    names_b = ("bshh bshm bsmh cesh cesm aeqh aeqm requ epsgh epsgm ubryl "
               "ubry ubry3 aubh aubm bubh bubm cubr rcubr esqhf").split()
    want = dict(zip(names_a, fx["const_a"]))
    want.update(zip(names_b, fx["const_b"]))
    alias = {"MYJP_G": "g", "MYJP_RD": "r_d", "MYJP_CP": "cp",
             "MYJP_XLV": "xlv", "MYJP_XLS": "xls", "MYJP_P608": "p608",
             "MYJP_PQ0": "pq0", "MYJP_A2": "a2", "MYJP_A3": "a3",
             "MYJP_A4": "a4", "MYJP_EPSQ2": "epsq2", "PBL_B1POW23": "b1pow23"}
    src = (ROOT / "gpuwm" / "core" / "kernels" / "myjpbl.cu").read_text(
        encoding="utf-8")
    found = re.findall(r"#define (\w+)\s+(-?[0-9.]+(?:e[+-]?\d+)?)f\b", src)
    assert len(found) >= 55
    for name, literal in found:
        key = alias.get(name, name.replace("PBL_", "").lower())
        assert key in want, name
        got = np.float32(float(literal)).view(np.uint32)
        assert got == np.float32(want[key]).view(np.uint32), (
            name, literal, want[key])


def _assert_exact(summary, *, strict):
    for mode, fields in summary.items():
        for name, st in fields.items():
            if strict:
                assert st["bit_mismatch"] == 0, (mode, name, st)
            else:
                # CuPy's default -ftz=true: a word may differ only by less
                # than FLT_MIN (what flushing a subnormal operand can move
                # it), and only in the condensate rows.
                assert st["beyond_ftz"] == 0, (mode, name, st)
                assert st["max_abs"] < float(np.finfo(np.float32).tiny)
                if st["bit_mismatch"]:
                    assert name in ("rqcblten", "rqiblten"), (mode, name, st)


@pytest.mark.gpu
@requires_gpu
def test_myjpbl_matches_wrf_word_for_word():
    from gpuwm.verify import myjpbl_oracle
    report = myjpbl_oracle.measure(_fixture())
    summary = myjpbl_oracle.summarize(report)
    print("strict" if STRICT else "default", summary)
    _assert_exact(summary, strict=STRICT)


@pytest.mark.gpu
@requires_gpu
def test_zeroing_ht_is_seen():
    """Negative control: with HT = 0 (the ground-relative column the kernel
    used before), the terrain columns stop matching -- so the gate above is
    sensitive to the ZINT seed, not passing by construction."""
    from gpuwm.verify import myjpbl_oracle
    fx = _fixture()
    report = myjpbl_oracle.measure({"main_ice": fx["main_ice"]},
                                   zero_ht=True)
    summary = myjpbl_oracle.summarize(report)["replay"]
    assert summary["pblh"]["bit_mismatch"] > 0
    assert summary["exch_h"]["bit_mismatch"] > 0
    assert summary["tke"]["bit_mismatch"] > 0


def test_the_engine_hands_myj_the_terrain_height():
    """The shipped call site passes HT (module_pbl_driver.F hands MYJPBL
    ``HT=ht``), and the launcher's surface roster requires it."""
    from gpuwm.core import myjpbl
    assert "ht" in myjpbl._SURFACE_INPUTS
    src = (ROOT / "gpuwm" / "core" / "physics.py").read_text(encoding="utf-8")
    body = src[src.index("def _run_myj_pbl"):src.index("def _run_noah_mosaic")]
    assert '"ht": cp.ascontiguousarray(self.state.ht)' in body
