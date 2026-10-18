"""Grell-Freitas against WRF v4.6.1 on 280 realistic columns, word for word.

The fixture is ``tests/data/oracles/gf/regime/``, written by
``tools/gf_wrf461_oracle/build_regime.sh``: soundings across the regimes
cumulus parameterisation meets (tropical ocean, tropical cyclone,
continental severe, capped, nocturnal, winter snow, lake effect, arctic,
plateau and mountain terrain, trade cumulus, stratocumulus, desert), each in
a climatological and a convectively active state, at three grid spacings
and both ``ishallow`` arms; a ``clos_choice`` 1..16 sweep; and named edge
cases.  Every column went through WRF's own byte-pinned ``GFDRV``.

Two WRF builds, identical but for one symbol:

* ``crgamma`` -- ``tgammaf`` resolves to the correctly rounded gamma WOOF
  ships (gf_crgamma_tgammaf.c).  The kernel's shipped path computes the
  same gamma, so this is the bitwise target, with NOTHING pinned;
* ``stock`` -- WRF as it links, C-library ``tgammaf``.  Not correctly
  rounded, the documented deliberate divergence
  (docs/gf_gamma_known_delta.md); graded here as a measured delta.

This file checks the fixture and its receipts and needs no GPU;
``tests/test_gf_regime_column_oracle_gpu.py`` runs ``gf_gfdrv_stage`` -- the
kernel the engine launches -- and compares every output word, under
``GPUWM_WRF_EXACT=1`` (strict) and without it (default).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools" / "gf_wrf461_oracle"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gf_regime_columns as grc                                    # noqa: E402

FIX = _ROOT / "tests" / "data" / "oracles" / "gf" / "regime"


@pytest.fixture(scope="module")
def oracle():
    import gf_regime_gpu as rg

    return rg.load_oracle(FIX)


def test_corrected_referee_receipts_reconstruct_independent_output_hashes():
    import gf_regime_gpu as rg
    for arm in ("safe", "corrected"):
        reference = rg.load_reference(FIX, arm)
        assert (reference["ncol"], reference["nz"], reference["kte_mode"]) == (280, 50, 0)


# ==========================================================================
# the fixture and its receipts (no GPU)
# ==========================================================================
def test_the_committed_files_are_the_ones_the_receipt_names():
    sums = {}
    for line in (FIX / "regime-sha256sums.txt").read_text(
            encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        sums[name.strip()] = digest
    for name in ("gf-regime-inputs.bin", "gf-regime-manifest.json",
                 "gf-regime-wrf-crgamma-k0.bin", "gf-regime-wrf-stock-k0.bin"):
        got = hashlib.sha256((FIX / name).read_bytes()).hexdigest()
        assert got == sums[name], name


def test_the_receipt_pins_the_wrf_file_set_and_the_oracle_sources():
    text = (FIX / "regime-sha256sums.txt").read_text(encoding="ascii")
    for digest in (
            "0cead99c913b6a74f3a492aea1e1c7ef6d34f95402574336022d1a6c716a4382",
            "798ad7271963757b88be36ce01e24222f7af045dc0692908d6e7c270d6b4ea11",
            "0dd1ed7c620b39e13f7ef77dc4ff1a269fa1f8d746492d360e203df0215b529a",
            "2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52",
            "300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25"):
        assert digest in text
    for src in ("gf_regime_columns.py", "run_gf_columns.F90",
                "gf_crgamma_tgammaf.c", "gf_crgamma_check.c"):
        want = hashlib.sha256((_TOOLS / src).read_bytes()).hexdigest()
        assert f"{want}  <oracle-workspace>/tools/gf_wrf461_oracle/{src}" \
            in text, f"{src} changed since the fixture was built; rebuild it"


def test_the_correctly_rounded_gamma_was_the_one_linked_and_checked():
    binding = (FIX / "binding.txt").read_text(encoding="ascii")
    assert " T tgammaf" in binding          # crgamma defines it itself
    assert " U tgammaf" in binding          # stock takes the shared libm's
    check = (FIX / "crgamma-check.txt").read_text(encoding="ascii").strip()
    assert check == "checked 65638, mismatched 0"


def test_wrf_read_nothing_undefined_and_nothing_out_of_bounds():
    bounds = (FIX / "bounds.txt").read_text(encoding="ascii")
    assert bounds.count("no out-of-bounds access on any of 280 columns") == 2
    cons = (FIX / "regime-consistency.txt").read_text(encoding="ascii")
    assert cons.count("(snan init vs default), kte_mode 0: same") == 1
    assert cons.count("(snan init vs default), kte_mode 1: same") == 1
    # kte = kde = nz + 1 with NaN inputs on the extra level changes only
    # HBOT's REAL(KTE) initial value: no level above ktf is read
    assert "stock: kte_mode 1 differs from 0 only in HBOT's initial value" \
        in cons and cons.count(": True") == 2
    assert "stock vs crgamma (the gamma divergence must show): DIFFERENT" \
        in cons


def test_the_columns_span_the_regimes_and_the_edges(oracle):
    import gf_regime_gpu as rg

    inp, cr, _st, man = oracle
    cov = rg.coverage(cr, man)
    assert cov["columns"] == 280 and inp["nz"] == 50
    for name in ("tropical-ocean-deep", "tropical-cyclone-core",
                 "continental-severe", "nocturnal-stable-llj",
                 "stable-night-dry", "winter-continental-snow",
                 "arctic-polar-night", "lake-effect-cold-air-outbreak",
                 "plateau-monsoon", "mountain-winter-orographic",
                 "trade-cumulus", "marine-stratocumulus", "desert-afternoon",
                 "moisture-floor", "tn-reset-below-200k", "zero-forcing",
                 "kpbl-1", "supersaturated", "below-sea-level", "dx-500m",
                 "dx-100km", "exactly-258k"):
        assert name in cov["regimes"], name
    assert cov["clos_choice"] == list(range(17))
    assert cov["ishallow"] == [0, 1] and cov["xland"] == [1.0, 2.0]
    assert cov["ht_range"][0] < 0.0 and cov["ht_range"][1] >= 4500.0
    # the paths that matter are actually exercised, not just present
    # (measured on this fixture; a rebuilt fixture that loses coverage fails)
    assert cov["precipitating"] == 146
    assert cov["shallow_fired"] == 17
    assert cov["ice_tendency_columns"] == 96
    assert cov["liquid_tendency_columns"] == 138
    assert cov["deep_top_reached"] == 138
    assert not cov["any_nan"]


def test_the_gamma_divergence_is_displacement_not_a_branch_flip(oracle):
    """WRF-with-C-library-gamma against WRF-with-correct-gamma, both WRF:
    the integer indices and the rain/no-rain decision never move, so the
    documented divergence displaces amounts and flips nothing."""
    _inp, cr, st, _man = oracle
    assert np.array_equal(cr["isc"], st["isc"])
    pr_c = cr["sca"][:, grc.OUT_SCA.index("pratec")]
    pr_s = st["sca"][:, grc.OUT_SCA.index("pratec")]
    assert np.array_equal(pr_c > 0, pr_s > 0)
    moved = (cr["lev"].view(np.uint32) != st["lev"].view(np.uint32)).reshape(
        cr["ncol"], -1).any(axis=1)
    assert 0 < int(moved.sum()) <= int((pr_c > 0).sum() +
                                       (cr["sca"][:, 4] > 0).sum())


@pytest.mark.parametrize("arm", ("safe", "corrected"))
def test_strict_scalar_reference_has_independent_flags_and_full_output_hash(arm):
    import json
    import gf_regime_gpu as rg
    from record_scalar_reference import FLAGS, PINS
    receipt = json.loads((FIX / f"gf-regime-wrf-scalar-{arm}-delta.json").read_text())
    proof = receipt["provenance"]
    assert tuple(proof["fortran_flags"]) == FLAGS
    assert proof["wrf_source_sha256"] == PINS
    assert proof["stock_dynamic_binding"] is True
    assert proof["driver_sha256"] == hashlib.sha256((_TOOLS / "run_gf_columns.F90").read_bytes()).hexdigest()
    result = rg.load_reference(FIX, arm, arithmetic="strict")
    assert (result["ncol"], result["nz"], result["kte_mode"]) == (280, 50, 0)
