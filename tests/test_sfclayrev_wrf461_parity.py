"""Revised MM5 surface layer (sf_sfclay_physics=1) against WRF v4.6.1, word for word.

``tools/sfclayrev_wrf461_oracle/`` builds WRF's own ``SFCLAYREV`` wrapper over
the byte-unmodified ``phys/physics_mmm/sf_sfclayrev.F90`` at gfortran ``-O0``
and runs it on 110 columns (``tests/data/oracles/sfclayrev/``): convective and
stable land, calm very stable air, the previously-unstable ``br = 0`` clamp,
open ocean, lakes, hurricane winds, snow and sea ice, high terrain, hot
desert, tall roughness on thin layers, and edge cases (``ust < 0.001``,
``br < -250``, zero wind, dry air, zero ``ust``, ``qsfc <= 0`` on land,
``-0.0`` and one-ULP contrasts).  Six switch arms (WRF default;
``isftcflx`` 1 and 2; ``iz0tlnd`` 1 and 2; ``isfflx = 0``), three chained
steps each: 1,980 column calls, 34 output words per call.

Before lane/verify-revised-mm5-sfclay the kernel had never met a WRF number
for this scheme.  It measured 3,721 ULP from WRF under the strict build and
786,432 ULP under default arithmetic.  It is now WRF's word on every compared
lane under both builds.

One word per arm is not compared, and the reason is WRF's, not the port's:
with ``isfflx = 0`` ``sf_sfclayrev_run`` never assigns ``LH`` and the wrapper
copies an uninitialised local into it, so that word is whatever the stack
held (two builds of the same oracle disagree on it).  The port writes 0.

The reference is the -O0 object, the scheme's own arithmetic.  The objects a
stock WRF 4.6.1 build compiles (-O2 -ftree-vectorize -funroll-loops,
``tools/sfclayrev_wrf461_oracle/build_asbuilt.sh``, receipt
``sfclayrev-outputs-asbuilt.hex``) give the same words except one: PSIM,
arm ``iz0tlnd = 1``, step 2, case 69, one ULP.  The cause is the build, not
the scheme: at -O2 gfortran turns the four ``x**2.`` powers of the
unstable similarity functions (sf_sfclayrev.F90:1012, 1014, 1027, 1029)
into ``x*x`` (the -O2 tree dump has none of the -O0 dump's four
``powf(x, 2.0)`` calls).  Measured on W1: an -O0 build of a scratch copy
with only those four powers written ``x**2`` reproduces the as-built words
exactly, and -O2 with the vectoriser off gives the same single difference.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from _sfclayrev_oracle import ARMS, ORACLE_DIR, STEPS, WRF_UNDEFINED, load_fixture

REPO = Path(__file__).resolve().parents[1]
KERNELS = REPO / "gpuwm" / "core" / "kernels"
TOOL = REPO / "tools" / "sfclayrev_wrf461_oracle"

#: Worst float32 ULP distance from kernels/sfclay.cu to WRF's word over every
#: compared lane, per build.  Asserted for EQUALITY: a silent change in
#: either direction is drift.  Measured on an RTX PRO 6000 (sm_120), NVRTC
#: 12.9.86, against the gfortran 13.3.0 / glibc 2.39 oracle.
MAX_ULP_STRICT = 0
MAX_ULP_DEFAULT = 0


def test_fixture_is_complete_and_matches_its_receipt():
    fixture = load_fixture()
    assert len(fixture.cases) >= 64
    assert fixture.outputs.shape == (len(ARMS), STEPS, len(fixture.cases), 34)
    assert not np.isnan(fixture.outputs).any(), "a WRF word decoded as NaN"
    sums = {}
    for line in (ORACLE_DIR / "oracle-sha256sums.txt").read_text().splitlines():
        digest, path = line.split()
        sums[Path(path).name] = digest
    for name in ("sfclayrev-inputs.hex", "sfclayrev-outputs.hex"):
        assert hashlib.sha256((ORACLE_DIR / name).read_bytes()).hexdigest() == sums[name]
    assert hashlib.sha256((TOOL / "run_sfclayrev.F90").read_bytes()).hexdigest() \
        == sums["run_sfclayrev.F90"], "the driver changed after the fixture was written"


def test_fixture_reaches_every_regime_and_switch_branch():
    fixture = load_fixture()
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
    regime = fixture.outputs[..., SFCLAY_OUTPUTS.index("regime")]
    br = fixture.outputs[..., SFCLAY_OUTPUTS.index("br")]
    assert set(np.unique(regime)) == {1.0, 3.0, 4.0}
    assert (br == 0.0).any() and (br > 0).any() and (br < 0).any()
    assert (fixture.inputs["xland"] == 2.0).any() and (fixture.inputs["lakemask"] == 1.0).any()
    assert (fixture.inputs["ust"] < 0.001).any()
    labels = set(fixture.labels)
    for need in ("convective-land", "stable-night-land", "ocean-unstable", "snow-ice-land",
                 "high-terrain", "hurricane-ocean", "rough-thin-layer"):
        assert need in labels


def test_make_inputs_reproduces_the_committed_columns(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sfclayrev_make_inputs", TOOL / "make_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write(tmp_path)
    for name in ("sfclayrev-inputs.hex", "sfclayrev-cases.csv"):
        assert (tmp_path / name).read_bytes() == (ORACLE_DIR / name).read_bytes()


def test_reference_object_carries_no_vector_libm():
    report = (ORACLE_DIR / "libmvec-report.txt").read_text()
    reference = report.split("# -O2")[0]
    assert "_ZGV" not in reference
    for word in ("logf", "expf", "powf", "atanf"):
        assert f"U {word}" in reference


def test_sfclay_atanf_table_is_mynns_word_for_word():
    """sf_atan and mynn_glibc_atanf are one routine; their tables cannot drift."""
    def table(path, name):
        text = path.read_text(encoding="utf-8")
        body = re.search(name + r"\[19\]\s*=\s*\{([^}]*)\}", text).group(1)
        return [int(w, 16) for w in re.findall(r"0x([0-9A-Fa-f]+)u", body)]
    sfc = table(KERNELS / "sfclay.cu", "SFC_ATANF_TAB")
    assert len(sfc) == 19
    assert sfc == table(KERNELS / "mynn_pbl.cu", "MYNN_ATANF_TAB")
    assert sfc == table(KERNELS / "mynn_dmp_sibling.cu", "MYNN_ATANF_TAB")


def test_sfclay_unit_uses_no_cuda_libm_and_no_contraction():
    """The breakage this prevents: one CUDA logf/powf or one contracted
    multiply-add was worth hundreds to thousands of ULP on this oracle."""
    from gpuwm.core import kernels
    text = (KERNELS / "sfclay.cu").read_text(encoding="utf-8")
    code = "\n".join(line.split("//")[0] for line in text.splitlines())
    for builtin in ("logf", "expf", "powf", "atanf", "hypotf", "log10f", "cbrtf"):
        assert not re.search(r"(?<![A-Za-z0-9_])" + builtin + r"\s*\(", code), builtin
    assert "--fmad=false" in kernels.module_options("sfclay")
    assert "glibc_flt32.cuh" in kernels.EXTRA_HEADERS["sfclay"]


def test_only_the_wrf_undefined_lh_word_is_excluded():
    assert WRF_UNDEFINED == frozenset({(5, "lh")})
    assert ARMS[5] == (0, 0, 0)


def test_surface_slice_freeze_pin_matches_the_fixed_production_source():
    """A stale surface-layer pin must not keep the two fixed defects as a release refusal."""
    import ast
    from gpuwm.core import kernels
    tree = ast.parse((REPO / 'tests/test_mp8_frozen.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'FROZEN_MODULE_DIGESTS'
                        for t in n.targets))
    pins = ast.literal_eval(node.value)
    actual = (hashlib.sha256((KERNELS / 'sfclay.cu').read_bytes()).hexdigest(),
              hashlib.sha256(kernels.module_source('sfclay').encode()).hexdigest())
    assert pins['sfclay'] == actual


def _port_and_report():
    from _sfclayrev_oracle import measure, port_outputs
    fixture = load_fixture()
    return fixture, measure(fixture, port_outputs(fixture))


@requires_gpu
def test_revised_mm5_is_wrfs_word_on_every_field():
    from gpuwm import wrf_exact
    fixture, report = _port_and_report()
    want = MAX_ULP_STRICT if wrf_exact.ENABLED else MAX_ULP_DEFAULT
    worst = {k: v["max_ulp"] for k, v in report["fields"].items() if v["max_ulp"]}
    assert report["max_ulp"] == want, (worst, report["lanes"][:10])


@requires_gpu
def test_ensemble_shared_mask_launch_is_the_scalar_launch_word_for_word():
    """prepare_sfclay_column_batch with shared xland/lakemask compiles its own
    copy of the unit; the breakage this prevents is that copy compiling with
    the plain C++17 tuple (multiply-adds contracted) while the scalar loader
    compiles without contraction, so two members of one ensemble would round
    differently from a deterministic run."""
    import cupy as cp
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
    from gpuwm.ensemble.batch_physics import SFCLAY_INPUTS, prepare_sfclay_column_batch
    from _sfclayrev_oracle import port_outputs

    fixture = load_fixture()
    scalar = port_outputs(fixture)
    inp = fixture.inputs
    dx = 3000.0
    sel = np.flatnonzero(inp["dx"] == dx)
    members, n = 2, sel.size
    inputs, outputs = {}, {}
    for name in SFCLAY_INPUTS:
        col = inp[name][sel][None, :]
        inputs[name] = cp.asarray(col if name in ("xland", "lakemask")
                                  else np.vstack([col] * members))
    seed = {"znt": "znt", "ust": "ust", "ustm": "ustm", "mol": "mol", "hfx": "hfx",
            "qfx": "qfx", "qsfc": "qsfc", "zol": "zol"}
    for name in SFCLAY_OUTPUTS:
        base = inp[seed[name]][sel][None, :] if name in seed else np.zeros((1, n), np.float32)
        outputs[name] = cp.asarray(np.vstack([base] * members))
    launch = prepare_sfclay_column_batch(inputs, outputs, members=members, ny=1, nx=n,
                                         option=1, dx=dx, shared_fields=("xland", "lakemask"))
    launch()
    cp.cuda.Device().synchronize()
    for f, name in enumerate(SFCLAY_OUTPUTS):
        got = cp.asnumpy(outputs[name])
        for m in range(members):
            assert np.array_equal(got[m].view(np.uint32), scalar[0, 0, sel, f].view(np.uint32)), name
