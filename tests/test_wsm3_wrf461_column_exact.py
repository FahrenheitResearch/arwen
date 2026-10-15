"""The CUDA WSM3 (mp_physics = 3) against unmodified WRF v4.6.1, word for word.

THE BREAKAGE THIS PREVENTS
--------------------------
WSM3 is a new port.  Without this gate any edit that moves one word of
kernels/wsm3.cu away from WRF's own module_mp_wsm3.F -- a re-associated
sum, CUDA's expf for WOOF's own float32 routine, a float64 constant for
wsm3init's float32 one, an FMA contraction -- would ship silently.

THE ORACLE (tests/data/oracles/wsm3, built by
tools/wsm3_wrf461_column_oracle/build.sh: gfortran 13.3.0 -O0, glibc 2.39,
WRF v4.6.1 at d66e442f, no source edited) drives WRF's own entry point
``module_mp_wsm3::wsm3`` with WRF's own constants module and the radiation
effective radii on, over 77 columns x 45 levels: warm rain, deep convection
across the melting level in updrafts and downdrafts, stratiform melting
layers, freezing rain, cold winter columns, cirrus, dry and supersaturated
columns, a 47 km model top, fog, high terrain, and edge columns on qcrmin,
qc0, qmin, roqimax, t0c and the triple point, with negative, negative-zero
and subnormal inputs.  Five runs: dt 30 s (12 free-running steps), 72 s
(8), 150 s (3, one minor loop longer than dtcldcr), 180 s (3, two minor
loops) and 300 s (4, three minor loops).

TWO REFERENCES.  ``O0`` is unmodified WRF.  ``O0fix`` is WRF with the one
moved line of gpuwm's declared divergence D1 (cloud ice falling out of a
warm lowest level is counted as rain; WRF loses it).

THE CLAIMS
----------
* strict build (GPUWM_WRF_EXACT=1: no FMA, no FTZ): every word of every
  step of every run equals unmodified WRF, including all surface words,
  at both WSM3_KMAX tiers the launcher compiles;
* a single planted 1-ULP change to the kernel's theta store is caught;
* default build (CuPy's -ftz=true): every difference is counted; no columns
  are excluded from the report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "wsm3"
#: Columns whose WRF trajectory holds subnormal words, which a flushing
#: build cannot reproduce by construction.  The strict build excludes none.
FTZ_COLUMNS = ("edge-negative", "edge-dry-tiny")


def _dump_words() -> dict:
    out = {}
    for line in (FIXTURE / "wsm3-init-f32.txt").read_text().splitlines():
        name, word, _ = line.split()
        out[name] = int(word, 16)
    return out


def test_frozen_init_words_are_the_words_wrf_computed():
    """Breakage prevented: a wsm3init constant recomputed on the host
    (float64 Gamma product) instead of copied from WRF's float32 init."""
    from gpuwm.core.wsm3_constants import WRF_INIT_F32

    dump = _dump_words()
    for name, bits in WRF_INIT_F32:
        assert dump[name] == bits, name


def test_the_kernel_reads_the_words_in_the_order_they_are_frozen():
    """Breakage prevented: an enum reordered against WRF_INIT_F32, which
    would hand the kernel pvts for pvtr with every test of the table green."""
    import re

    from gpuwm.core.wsm3_constants import WRF_INIT_NAMES

    source = (ROOT / "gpuwm" / "core" / "kernels" / "wsm3.cu").read_text()
    body = re.search(r"enum Wsm3Coef \{(.*?)\};", source, re.S).group(1)
    names = [n.strip()[2:].lower() for n in body.replace("\n", " ").split(",")
             if n.strip().startswith("C_")]
    assert tuple(names) == WRF_INIT_NAMES


def test_the_column_set_is_the_oracle_input_and_spans_the_regimes():
    """Breakage prevented: a fixture regenerated from a different column set
    than the one WRF ran, or one that lost a regime the brief requires."""
    import hashlib

    meta = json.loads((FIXTURE / "columns.json").read_text())
    digests = json.loads((FIXTURE / "wrf461-step-digests.json").read_text())
    assert hashlib.sha256((FIXTURE / "columns.bin").read_bytes()).hexdigest() \
        == digests["columns_bin_sha256"]
    assert meta["ncol"] >= 64 and meta["nz"] == 45
    regimes = {c["regime"] for c in meta["columns"]}
    for needed in ("warm rain", "deep convection", "stratiform melting layer",
                   "freezing rain", "cold ice", "cirrus", "dry",
                   "supersaturated", "high model top", "radiation fog",
                   "high terrain"):
        assert needed in regimes, needed
    assert sum(r.startswith("edge:") for r in regimes) >= 10
    names = {c["name"] for c in meta["columns"]}
    for name in FTZ_COLUMNS:
        assert name in names, name


def test_every_fixture_field_matches_the_unmodified_wrf_digests():
    """A stored oracle word cannot drift away from the independent WRF receipt."""
    import numpy as np
    from tools.wsm3_wrf461_column_oracle.run_gpu import digest, LEVEL_OUT, SURFACE_OUT
    recorded = json.loads((FIXTURE / "wrf461-step-digests.json").read_text())
    with np.load(FIXTURE / "wrf461-final-step.npz") as words:
        for flavour in ("O0", "O0fix"):
            for tag, steps in recorded[flavour].items():
                for istep, step in enumerate(steps, 1):
                    for name in LEVEL_OUT + SURFACE_OUT:
                        assert digest(words[f"{flavour}/{tag}/{istep}/{name}"]) == step[name]


def test_the_declared_divergence_is_one_moved_line_and_moves_only_rain():
    """D1 is exactly the one-line edit build.sh refuses to exceed, and in
    WRF's own two builds it moves only RAINNC, RAINNCV and SR.

    Breakage prevented: a divergence that grows past what it declares."""
    diff = (FIXTURE / "d1-fix.diff").read_text().splitlines()
    changed = [line for line in diff if line[:1] in "<>"]
    assert changed == ["<         if((t0c-t(i,1)).gt.0) then",
                       ">         if((t0c-t(i,1)).gt.0) then"]
    digests = json.loads((FIXTURE / "wrf461-step-digests.json").read_text())
    fields = {name for run in digests["d1_census"].values() for name in run}
    assert fields and fields <= {"rain", "rainncv", "sr"}
    assert digests["d1_columns"]


def test_the_reference_was_built_without_vector_libm():
    """The -O0 reference pulled no glibc vector-math symbol (build.sh fails
    otherwise), and the positive control proves the grep can see one."""
    report = (FIXTURE / "libmvec-report.txt").read_text()
    o0 = report.split("# -O0 (THE REFERENCE)")[1].split("# -O2")[0]
    assert "_ZGV" not in o0
    assert "_ZGVbN4v_expf" in report.split("# positive control:")[1]


def _check(extra_env: dict, *args) -> dict:
    env = dict(os.environ)
    env.pop("GPUWM_WRF_EXACT", None)
    env.update(extra_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-m", "tools.wsm3_wrf461_column_oracle.run_gpu",
         "check-fixture", str(FIXTURE), *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
    lines = proc.stdout.strip().splitlines()
    assert lines, proc.stderr[-4000:]
    result = json.loads(lines[-1])
    result["returncode"] = proc.returncode
    return result


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("tier", (64, 80))
def test_strict_build_is_wrf461_on_every_word_of_every_step(tier):
    """Strict arithmetic: zero differing words, every column, every step."""
    result = _check({"GPUWM_WRF_EXACT": "1"}, "--tier", str(tier))
    assert result["strict_build"] is True
    assert result["total_mismatch"] == 0, result
    assert result["surface_rain_difference_words_vs_O0"] == 0, result
    assert result["reference"] == "O0", result


@pytest.mark.gpu
@requires_gpu
def test_a_planted_one_ulp_change_is_caught():
    """The gate's sensitivity: flipping the last bit of theta's store fails."""
    result = _check({"GPUWM_WRF_EXACT": "1"}, "--plant")
    assert result["returncode"] == 1
    assert result["total_mismatch"] > 0, result
    assert result["first_step_th_max_ulp"] == [1] * 5, result


@pytest.mark.gpu
@requires_gpu
def test_default_build_counts_every_column_and_field():
    """Default mode must report all differences, including subnormal columns."""
    result = _check({})
    assert result["strict_build"] is False
    assert result["reference"] == "O0"
    assert result["returncode"] in (0, 1)
    counted = sum(row[0] for run in result["summary"].values() for row in run.values())
    assert result["total_mismatch"] == counted
