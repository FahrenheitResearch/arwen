"""Morrison against the byte-unmodified WRF v4.6.1 module.

The older Morrison GPU tests compare with ``np_morrison_column``, a float64
transcription mirror.  This file is the first test in this repository whose
Morrison expected values were written by WRF itself.  The fixture comes from
the public ``MP_MORR_TWO_MOMENT`` wrapper and includes its radar diagnostic.

The measured residual is deliberately pinned exactly.  Improvement and
regression both require an explicit baseline change, while the strict xfail
keeps max_ulp 0 visible as the acceptance contract.
"""

from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.morrison_constants import rimed_ice_constants
from tools.morrison_wrf461_oracle.validate_morrison_oracle import (
    DEFAULT_FIXTURE_DIR,
    LEVEL_INPUTS,
    measure_oracle,
)


PINNED_WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
WRF_SOURCE_SHA256 = {
    "ccpp_kind_types.F":
        "a76e1e5b52fc7cd40be9ccb506fde28b1ef2486b812d518f7f1c882766d484db",
    "module_model_constants.F":
        "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
    "module_mp_radar.F":
        "aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688",
    "module_mp_morr_two_moment.F":
        "2dcd1c70127ea57c8a2437baf2b8a2388b1722e4d4ead9411834946439b8c273",
}

# Per-field max_ulp measured per platform.  The Windows driver and Linux/CUDA
# 12.9 NVRTC produce slightly different transcendental results on the same
# sm_120 card; sm_89 and Linux sm_86 add distinct measured vectors.  Pin each complete
# signature rather than making platform-specific mismatch counts part of the
# physics contract.  fp32_ulp_distance in gpuwm.core.fp32_ulp is the sole ULP
# owner. Across the measured platforms graupelnc stays exactly 0, theta
# stays 154 and sr is the worst field. These signatures describe an
# unresolved scientific residual; they do not establish WRF agreement.
# RTX 5090 (sm_120), Windows driver.
MEASURED_WINDOWS_MAX_ULP = {
    "graupelnc": 0,
    "graupelncv": 1643044151,
    "ng": 1108899129,
    "ni": 1219365430,
    "nr": 1013995714,
    "ns": 1112428311,
    "qc": 163796,
    "qg": 908160525,
    "qi": 851838551,
    "qr": 718204424,
    "qs": 895249780,
    "qv": 8388606,
    "rainnc": 4645,
    "rainncv": 817215992,
    "refl_10cm": 11072910,
    "snownc": 5,
    "snowncv": 1580788530,
    "sr": 1709094255,
    "theta": 154,
}
# RTX 5090 (sm_120), Linux / CUDA 12.9 NVRTC.
MEASURED_LINUX_MAX_ULP = {
    "graupelnc": 0,
    "graupelncv": 1642221535,
    "ng": 1108899127,
    "ni": 1219365430,
    "nr": 1013995714,
    "ns": 1112428308,
    "qc": 163796,
    "qg": 908160527,
    "qi": 851838551,
    "qr": 718204424,
    "qs": 895249780,
    "qv": 8388606,
    "rainnc": 4645,
    "rainncv": 817215992,
    "refl_10cm": 11072910,
    "snownc": 5,
    "snowncv": 1563689139,
    "sr": 1706351510,
    "theta": 154,
}
# First sm_89 measurement: RTX 4090 (Ada), Linux, driver 590.48.01,
# CUDA driver/NVRTC 13.1, CUDA runtime 12.9 (cupy-cuda12x 14.1.1),
# Python 3.12.3, 2026-08-03 -- the first user-zero cross-architecture
# stress run of the published wheel (arwen-stress-4090 part2-parity,
# MORRISON-ULP-sm89-vs-sm120.tsv).  Differs from BOTH sm_120 tuples on
# qi, ns, refl_10cm, snowncv and graupelncv; matches the Windows sm_120
# tuple (and not the Linux one) on qg, ng and sr; identical to both on
# the remaining 11 fields.
MEASURED_SM89_LINUX_MAX_ULP = {
    "graupelnc": 0,
    "graupelncv": 1632923731,
    "ng": 1108899129,
    "ni": 1219365430,
    "nr": 1013995714,
    "ns": 1112428312,
    "qc": 163796,
    "qg": 908160525,
    "qi": 851838674,
    "qr": 718204424,
    "qs": 895249780,
    "qv": 8388606,
    "rainnc": 4645,
    "rainncv": 817215992,
    "refl_10cm": 11477310,
    "snownc": 5,
    "snowncv": 815499543,
    "sr": 1709094255,
    "theta": 154,
}
# RTX 3080 (sm_86), WSL Ubuntu 24.04, Python 3.12.3, CuPy 14.2.0,
# CUDA runtime 12.9, NVRTC 12.8, CUDA driver API 13.3; measured 2026-09-04.
# The same card under Windows/CuPy 14.0.1/NVRTC 13.0 exactly reproduces
# MEASURED_SM89_LINUX_MAX_ULP. A fresh build of the byte-pinned WRF source
# reproduces both committed CSVs exactly; disabling FMAD changes but does
# not close the residual. See tools/morrison_wrf461_oracle/README.md.
MEASURED_SM86_LINUX_MAX_ULP = {
    "graupelnc": 0,
    "graupelncv": 1607604645,
    "ng": 1108899128,
    "ni": 1219365430,
    "nr": 1013995714,
    "ns": 1112428308,
    "qc": 163796,
    "qg": 908160530,
    "qi": 851838674,
    "qr": 718204424,
    "qs": 895249780,
    "qv": 8388606,
    "rainnc": 4645,
    "rainncv": 817215992,
    "refl_10cm": 11477310,
    "snownc": 5,
    "snowncv": 1563689139,
    "sr": 1706351512,
    "theta": 154,
}
MEASURED_PLATFORM_MAX_ULP = (
    MEASURED_WINDOWS_MAX_ULP,
    MEASURED_LINUX_MAX_ULP,
    MEASURED_SM89_LINUX_MAX_ULP,
    MEASURED_SM86_LINUX_MAX_ULP,
)

#: Why the acceptance contract is unmet, carried on the xfail itself.
#: An xfail whose reason states only the symptom ("max_ulp is not 0")
#: records no cause, so the next reader has to rediscover it; the causes
#: below are measured on the shipped sources and are listed in the
#: physics registry's morrison-mp10 "NOT ESTABLISHED" warning too.
#:
#: 1. LIBM DISPATCH.  morrison.cu calls CUDA's builtins 164 times -- 84
#:    powf, 45 tgammaf, 14 sqrtf, 10 cbrtf, 7 expf, 4 log10f -- against a
#:    gfortran/glibc oracle.  gpuwm/core/kernels/glibc_flt32.cuh states
#:    the rule ("CUDA's expf/powf/tgammaf are DIFFERENT functions ... any
#:    kernel graded bitwise against a gfortran/glibc oracle must call
#:    these and not the builtins") and morrison is deliberately NOT in the
#:    loader's _EXTRA_HEADERS allow-list, because listing it would close
#:    at most the 84 powf and 7 expf sites: WRF's Morrison does not call
#:    glibc's gamma at all, it calls its own Cody rational REAL FUNCTION
#:    GAMMA (module_mp_morr_two_moment.F:4153-4384), which is what the
#:    oracle links, so gfk_tgamma is the wrong substitute for all 45
#:    tgammaf sites; and the header transcribes no cbrtf and no log10f.
#: 2. FP CONTRACTION.  load_module compiles with options=("-std=c++17",),
#:    i.e. NVRTC's default --fmad=true, so every a*b + c outside
#:    morr_polysvp fuses while the -O0 oracle does not.  morr_polysvp's
#:    own __f{add,sub,mul,div}_rn pinning stops at morrison.cu:99.
#: 3. WHICH ORACLE IS THE TARGET is itself undecided.  The fixture oracle
#:    is built at -O0 (tools/morrison_wrf461_oracle/build.sh) so it does
#:    not contract, but gfortran's default is -ffp-contract=fast and the
#:    reference WRF build is NVHPC, so bitwise agreement with the -O0
#:    oracle is not the same target as bitwise agreement with a production
#:    WRF binary.  Nothing may change here before that is chosen.
BITWISE_XFAIL_REASON = (
    "WRF acceptance contract is max_ulp 0; the measured production "
    "kernel is pinned by test_measured_morrison_wrf_residual_cannot_"
    "drift_silently.  Known causes, none of them a formula: morrison.cu "
    "calls CUDA's libm builtins (84 powf, 45 tgammaf, 10 cbrtf, 7 expf, "
    "4 log10f) against a gfortran/glibc -O0 oracle, and NVRTC's default "
    "--fmad=true contracts every a*b+c outside morr_polysvp.  Listing "
    "morrison in the kernel loader's _EXTRA_HEADERS would not close it: "
    "WRF calls its own REAL FUNCTION GAMMA "
    "(module_mp_morr_two_moment.F:4153-4384), not glibc's, and "
    "glibc_flt32.cuh transcribes no cbrtf and no log10f.  See this "
    "module's BITWISE_XFAIL_REASON comment."
)


def _csv_rows(name: str) -> list[dict[str, str]]:
    path = DEFAULT_FIXTURE_DIR / name
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _receipt_hashes() -> dict[str, str]:
    lines = (
        DEFAULT_FIXTURE_DIR / "oracle-sha256sums.txt"
    ).read_text(encoding="ascii").splitlines()
    return {
        Path(path).name: digest
        for digest, path in (line.split(maxsplit=1) for line in lines)
    }


@lru_cache(maxsize=1)
def _measurement() -> dict[str, object]:
    return measure_oracle()


# ---------------------------------------------------------------------------
# CPU-only provenance and coverage checks.
# ---------------------------------------------------------------------------


def test_fixture_is_from_the_pinned_unmodified_wrf_sources():
    assert (
        DEFAULT_FIXTURE_DIR / "wrf-commit.txt"
    ).read_text(encoding="ascii").strip() == PINNED_WRF_COMMIT
    receipts = _receipt_hashes()
    assert {name: receipts[name] for name in WRF_SOURCE_SHA256} == (
        WRF_SOURCE_SHA256
    )
    for name in ("morrison-levels.csv", "morrison-surface.csv"):
        digest = hashlib.sha256((DEFAULT_FIXTURE_DIR / name).read_bytes())
        assert digest.hexdigest() == receipts[name]


def test_oracle_build_excludes_libmvec_and_proves_guard_can_fire():
    report = (
        DEFAULT_FIXTURE_DIR / "libmvec-report.txt"
    ).read_text(encoding="ascii")
    reference, separator, controls = report.partition(
        "# -O2 -ftree-vectorize -funroll-loops")
    assert separator
    assert "_ZGV" not in reference
    assert "U expf" in reference and "U powf" in reference
    assert "_ZGV" in controls
    assert "_ZGVbN4v_expf" in controls


def test_fixture_spans_phases_thresholds_signed_zero_and_subnormals():
    rows = _csv_rows("morrison-levels.csv")
    keys = {(int(row["case"]), int(row["mode"])) for row in rows}
    assert keys == {(case, mode) for mode in (0, 1) for case in range(1, 15)}
    assert len(rows) == 28 * 32

    def field(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float32)

    temperature = field("theta_in") * field("pii_in")
    assert float(temperature.min()) < 195.0
    assert float(temperature.max()) > 298.0
    assert float(field("pressure_in").min()) < 21000.0
    assert float(field("pressure_in").max()) > 97000.0

    input_bits: set[int] = set()
    for name in LEVEL_INPUTS:
        input_bits.update(field(f"{name}_in").view(np.uint32).tolist())
    assert {
        0x00000000,  # +0
        0x80000000,  # -0
        0x00000001,  # smallest positive subnormal
        0x80000001,  # smallest negative subnormal
        0x007FFFFF,  # largest positive subnormal
        0x00800000,  # smallest positive normal
        0x00800001,  # its next representable neighbour
    } <= input_bits

    for species in ("qc", "qr", "qi", "qs", "qg"):
        values = field(f"{species}_in")
        assert (values == 0.0).any()
        assert ((values > 0.0) & (values < 1.0e-13)).any()
        assert values.max() > 1.0e-4


def test_graupel_and_hail_are_distinct_and_plumbed_through_both_diagnostics():
    """This tree is not the hardcoded-graupel tree from the separate history."""
    from gpuwm.core.morrison import launch_morrison
    from gpuwm.core.refl import launch_refl10cm_morrison, radar_init

    assert RunConfig.__dataclass_fields__["morr_rimed_ice"].default == 1
    graupel = rimed_ice_constants(0)
    hail = rimed_ice_constants(1)
    assert (graupel.ag, graupel.bg, graupel.rhog) == (19.3, 0.37, 400.0)
    assert (hail.ag, hail.bg, hail.rhog) == (114.5, 0.5, 900.0)
    assert radar_init(0).xam_g != radar_init(1).xam_g
    assert inspect.signature(launch_morrison).parameters[
        "morr_rimed_ice"
    ].default == 1
    assert inspect.signature(launch_refl10cm_morrison).parameters[
        "morr_rimed_ice"
    ].default == 1

    rows = _csv_rows("morrison-levels.csv")
    by_level = {
        (int(row["case"]), int(row["k"]), int(row["mode"])): row
        for row in rows
    }
    differing_reflectivity = 0
    for case in range(1, 15):
        for k in range(1, 33):
            graupel_row = by_level[(case, k, 0)]
            hail_row = by_level[(case, k, 1)]
            # WRF received identical atmospheric inputs; only INIT mode differs.
            for name in LEVEL_INPUTS:
                assert np.float32(graupel_row[f"{name}_in"]).view(
                    np.uint32
                ) == np.float32(hail_row[f"{name}_in"]).view(np.uint32)
            differing_reflectivity += (
                np.float32(graupel_row["refl_10cm_out"]).view(np.uint32)
                != np.float32(hail_row["refl_10cm_out"]).view(np.uint32)
            )
    assert differing_reflectivity == 150


def test_the_bitwise_xfail_states_its_cause_and_the_cause_is_still_true():
    """The acceptance xfail must record WHY, not only that max_ulp != 0.

    Both halves are load-bearing.  The first reads the reason string off
    the marker and requires it to name libm dispatch, FMAD contraction and
    the reason the kernel loader's ``_EXTRA_HEADERS`` allow-list is not the
    fix -- a reason string that states only the symptom passes nothing on.
    The second re-measures the claim against the shipped kernel source, so
    the day somebody routes Morrison through ``glibc_flt32.cuh`` (or trades
    ``cbrtf`` for ``powf``) this goes red instead of leaving a stale cause
    on a marker nobody rereads.
    """
    from gpuwm.core.kernels import EXTRA_HEADERS

    marks = [mark for mark in test_morrison_wrf_acceptance_is_bitwise.pytestmark
             if mark.name == "xfail"]
    assert len(marks) == 1
    reason = marks[0].kwargs["reason"]
    assert reason == BITWISE_XFAIL_REASON
    for cause in ("powf", "tgammaf", "cbrtf", "log10f", "--fmad=true",
                  "_EXTRA_HEADERS", "module_mp_morr_two_moment.F:4153-4384",
                  "glibc_flt32.cuh"):
        assert cause in reason, cause

    source = (Path(__file__).resolve().parents[1] / "gpuwm" / "core"
              / "kernels" / "morrison.cu").read_text(encoding="utf-8")
    measured = {name: source.count(f"{name}(")
                for name in ("powf", "tgammaf", "cbrtf", "expf", "log10f")}
    assert measured == {"powf": 84, "tgammaf": 45, "cbrtf": 10,
                        "expf": 7, "log10f": 4}
    for name, count in measured.items():
        assert f"{count} {name}" in reason, name
    # The allow-list decision the reason string explains: morrison is NOT
    # routed through the glibc transcriptions, and the two kernels that are
    # named there so a silent addition cannot leave the reason stale.
    assert "morrison" not in EXTRA_HEADERS
    assert EXTRA_HEADERS["gf"] == ("glibc_flt32.cuh",)
    assert EXTRA_HEADERS["ntiedtke"] == ("glibc_flt32.cuh",)


# ---------------------------------------------------------------------------
# GPU measurement and acceptance contract.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu
def test_measured_morrison_wrf_residual_cannot_drift_silently():
    report = _measurement()
    observed_max_ulp = {
        name: int(entry["max_ulp"])
        for name, entry in report["fields"].items()
    }
    assert observed_max_ulp in MEASURED_PLATFORM_MAX_ULP
    expected_counts = {
        name: (28 if name in {
            "graupelnc", "graupelncv", "rainnc", "rainncv",
            "snownc", "snowncv", "sr",
        } else 896)
        for name in observed_max_ulp
    }
    assert {
        name: int(entry["value_count"])
        for name, entry in report["fields"].items()
    } == expected_counts
    assert report["overall"]["max_ulp"] == max(observed_max_ulp.values())
    assert report["overall"]["worst_field"] == "sr"
    assert report["overall"]["value_count"] == 10948
    assert report["overall"]["bitwise"] is False


@pytest.mark.gpu
@requires_gpu
@pytest.mark.xfail(
    strict=True,
    reason=BITWISE_XFAIL_REASON,
)
def test_morrison_wrf_acceptance_is_bitwise():
    report = _measurement()
    assert report["overall"]["max_ulp"] == 0
    assert report["overall"]["bitwise"] is True


@pytest.mark.gpu
@requires_gpu
def test_measured_platform_signatures_reject_a_latent_heat_perturbation(monkeypatch):
    """A measured compiler signature must still reject changed physics."""
    import cupy as cp
    from gpuwm.core import kernels, morrison

    source = kernels.module_source("morrison")
    coefficient = "real xlv = 3.1484e6f - 2370.0f * temp;"
    assert source.count(coefficient) == 1
    changed = source.replace(coefficient,
                             "real xlv = 3.1494e6f - 2370.0f * temp;")
    module = cp.RawModule(code=changed, options=("-std=c++17",))
    original_get_kernel = morrison.get_kernel

    def perturbed_kernel(name, entry_point):
        if name == "morrison":
            return module.get_function(entry_point)
        return original_get_kernel(name, entry_point)

    monkeypatch.setattr(morrison, "get_kernel", perturbed_kernel)
    report = measure_oracle()
    observed = {name: int(entry["max_ulp"])
                for name, entry in report["fields"].items()}
    assert observed not in MEASURED_PLATFORM_MAX_ULP
