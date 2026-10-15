"""The CUDA WSM7 (mp_physics = 24) against WRF v4.6.1, word for word.

THE BREAKAGE THIS PREVENTS
--------------------------
WSM7 is WSM6 plus a seventh class, hail, with eleven hail process terms,
a fourth sedimenting species and a fourth surface accumulator.  A port that
re-associates one of WRF's sums, takes CUDA's expf/powf instead of the words
the gfortran reference gets, recomputes wsm7init's float32 Gamma products in
float64, or drops one of the hail terms would run, look plausible and be a
different scheme.  This file holds the kernel to WRF's own words, so any
edit that moves one word away from WRF goes red here.

THE ORACLE (tests/data/oracles/wsm7, built by
tools/wsm7_wrf461_column_oracle/build.sh: gfortran 13.3.0 -O0, glibc 2.39,
WRF v4.6.1 at d66e442f, module_mp_wsm7.F byte-unmodified in the stock arm)
drives WRF's own entry point ``module_mp_wsm7::wsm7`` with WRF's constants
module, the radiation effective radii on and REFL_10CM requested, over 72
columns x 45 levels and 36 columns x 72 levels: deep hail convection, hail
melting through a dry sub-cloud layer, hail aloft sublimating, deep
convection, stratiform melting, freezing rain, stable nights, fog, marine
stratocumulus, winter snow, high terrain, cirrus, supercooled cloud below
-40 C and 16 edge columns (every branch threshold, negative, negative-zero
and subnormal inputs, condensate only at the top or bottom, thin layers,
both sides of hail's wet-growth test).  Five runs: dt 30 s x 12 steps,
60 s x 8, 150 s x 3 (dtcld > dtcldcr), 300 s x 4 (three minor loops), and
the 72-level set at 30 s x 4.

THE TWO DECLARED DIVERGENCES (docs/wsm7-mp24.md): WOOF fixes WRF's PLM remap
search (D1, WRF drops a layer's arriving mass) and puts hail into REFL_10CM
(D2, WRF leaves the seventh class out).  The kernel is graded against the
DEFINED arm: the same Fortran with exactly those two substitutions, written
in WRF's own Fortran.  Against the stock arm the differences are confined to
the columns the probe arm identifies (D1) and to hail-bearing reflectivity
(D2), recorded in arm-attribution.json.

THE CLAIMS
----------
* strict build (GPUWM_WRF_EXACT=1): every word of every step of every run is
  the defined arm's, at both WSM7_KMAX tiers;
* default build: the same -- the WSM-family unit compiles with --fmad=false
  --ftz=false in both builds (gpuwm/core/wsm_family.py);
* a planted one-bit change in a hail term is caught.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "wsm7"
KERNEL = ROOT / "gpuwm" / "core" / "kernels" / "wsm7.cu"
TABLE = ROOT / "gpuwm" / "data" / "wsm_family" / "schemes.json"
WORDS = ROOT / "gpuwm" / "data" / "wsm_family" / "wsm7-wrf461-words.json"


def test_frozen_words_are_the_words_wrf_computed():
    """The kernel's constants are wsm7init's and radar_init's own words.

    Breakage prevented: a constant recomputed on the host (a float64 Gamma
    function, an exact 720 for WGAMMA(7.)) instead of copied from WRF.
    """
    from tools.wsm7_wrf461_column_oracle.freeze_words import parse

    dumped = parse((FIXTURE / "wsm7-init-words.txt").read_text())
    frozen = json.loads(WORDS.read_text())
    for part in ("init_f32", "radar_f32", "radar_f64"):
        assert frozen[part] == dumped[part], part
    # WGAMMA(7.) is 720.00006 in WRF's float32, not 720.
    assert frozen["radar_f32"]["xcrg4"] == "44340001"


def _enum(name: str) -> list[str]:
    text = KERNEL.read_text(encoding="utf-8")
    body = re.search(r"enum %s \{(.*?)\};" % name, text, re.S).group(1)
    return [m.split("=")[0].strip() for m in body.replace("\n", " ").split(",")
            if m.strip()]


def test_the_kernel_enums_follow_the_table_order():
    """The kernel indexes ``coef``/``rf``/``rd`` by enum; the runner fills
    them in the table's order.  Breakage prevented: a word bound to the
    wrong slot (pacrh read where pacrg was meant) -- a silent wrong number."""
    row = json.loads(TABLE.read_text())["schemes"]["24"]
    coef = _enum("Wsm7Coef")
    assert coef[-1] == "WSM7_NCOEF"
    assert [c[2:].lower() for c in coef[:-1]] == row["init_words"]
    rf = _enum("Wsm7RadarF32")
    assert [c[3:].lower() for c in rf[:-1]] == row["radar_f32_words"]
    rd = [c for c in _enum("Wsm7RadarF64")]
    assert [c[3:].lower() for c in rd[:7]] == row["radar_f64_words"][:7]
    assert [c[3:].lower() for c in rd[7:12]] == row["radar_f64_words"][7:]


def test_the_column_sets_are_the_oracle_input_and_span_the_regimes():
    """Breakage prevented: a fixture regenerated from a column set other than
    the one WRF ran, or one that lost a regime."""
    digests = json.loads((FIXTURE / "wrf461-step-digests.json").read_text())
    for stem in ("columns", "columns72"):
        assert hashlib.sha256((FIXTURE / f"{stem}.bin").read_bytes()).hexdigest() \
            == digests["sets"][stem], stem
    meta = json.loads((FIXTURE / "columns.json").read_text())
    assert meta["ncol"] >= 64 and meta["nz"] == 45
    regimes = {c["regime"] for c in meta["columns"]}
    for needed in ("deep hail convection", "hail melting sub-cloud",
                   "hail aloft sublimating", "deep convection",
                   "stratiform melting layer", "freezing rain",
                   "stable clear night", "radiation fog", "ocean stratocumulus",
                   "winter snow", "high terrain", "cirrus",
                   "supercooled below -40C"):
        assert needed in regimes, needed
    assert sum(r.startswith("edge:") for r in regimes) >= 10
    assert json.loads((FIXTURE / "columns72.json").read_text())["nz"] == 72


def test_the_reference_was_built_without_vector_libm():
    """The -O0 reference pulled no glibc vector-math symbol (build.sh fails
    otherwise), and the positive control proves the grep can see one."""
    report = (FIXTURE / "libmvec-report.txt").read_text()
    o0 = report.split("# -O0 (THE REFERENCE)")[1].split("# -O2")[0]
    assert "_ZGV" not in o0
    assert "_ZGVbN4v_expf" in report.split("# positive control")[1]


def test_d1_moves_only_the_probe_identified_columns_and_d2_only_reflectivity():
    """The stock comparison, attributed.  D1 (the remap search) changes WRF's
    words only in columns where WRF's own find_kt fell through with arrival
    mass above kb; D2 (hail in REFL_10CM) changes REFL_10CM and nothing else.
    Breakage prevented: a "declared divergence" that silently moves columns
    its cause cannot reach."""
    attribution = json.loads((FIXTURE / "arm-attribution.json").read_text())
    assert len(attribution) == 5
    for tag, run in attribution.items():
        assert set(run["d1_vs_stock"]) <= set(run["probe"]), tag
        assert set(run["defined_vs_d1_fields"]) <= {"refl_10cm"}, tag
    # the fixture does exercise both
    assert any(run["d1_vs_stock"] for run in attribution.values())
    assert any(run["defined_vs_d1"] for run in attribution.values())


def _check(extra_env: dict, *args, script: str | None = None) -> dict:
    env = dict(os.environ)
    env.pop("GPUWM_WRF_EXACT", None)
    env.update(extra_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")])
    cmd = ([sys.executable, "-c", script] if script else
           [sys.executable, "-m", "tools.wsm7_wrf461_column_oracle.run_gpu",
            "check-fixture", str(FIXTURE), *args])
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                          text=True, timeout=1800)
    lines = proc.stdout.strip().splitlines()
    assert lines, proc.stderr[-4000:]
    return json.loads(lines[-1])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("tier", (64, 80))
def test_strict_build_is_the_defined_arm_on_every_word_of_every_step(tier):
    """Strict arithmetic: zero differing words, every column, every step."""
    result = _check({"GPUWM_WRF_EXACT": "1"}, "--tier", str(tier))
    assert result["strict_build"] is True
    assert result["total_mismatch"] == 0, result
    attribution = json.loads((FIXTURE / "arm-attribution.json").read_text())
    for tag, row in result["summary"].items():
        moved = set(attribution[tag]["d1_vs_stock_final"]) | set(attribution[tag]["defined_vs_d1_final"])
        assert set(row["columns_differing_from_stock_wrf"]) == moved, tag


@pytest.mark.gpu
@requires_gpu
def test_default_build_is_the_defined_arm_on_every_word_of_every_step():
    """The arithmetic users run is the same: no FMA, no flush to zero."""
    result = _check({})
    assert result["strict_build"] is False
    assert result["total_mismatch"] == 0, result


_PLANT = r"""
import sys
import numpy as np
import gpuwm.core.wsm_family as fam
from tools.wsm7_wrf461_column_oracle import run_gpu as rg
original = fam.init_words
def planted(mp):
    words = original(mp).copy()
    i = fam.row(mp).init_words.index("pacrh")
    words.view(np.uint32)[i] ^= np.uint32(1)       # one bit, hail collecting cloud
    return words
fam.init_words = planted
report = rg.check_fixture(rg.Path(sys.argv[1]) if len(sys.argv) > 1 else rg.Path("%s"),
                          runs=rg.RUNS[:1])
import json
print(json.dumps({"total_mismatch": report["total_mismatch"]}))
"""


@pytest.mark.gpu
@requires_gpu
def test_a_planted_one_bit_change_in_a_hail_term_is_caught():
    """Flip the last bit of pacrh (hail collecting cloud water, phacw): the
    gate must go red.  Breakage prevented: a gate too coarse to see the
    hail terms it claims to hold."""
    result = _check({"GPUWM_WRF_EXACT": "1"}, script=_PLANT % FIXTURE)
    assert result["total_mismatch"] > 0, result
