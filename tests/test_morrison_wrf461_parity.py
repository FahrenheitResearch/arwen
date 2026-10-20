"""Morrison against the byte-unmodified WRF v4.6.1 module.

The fixture comes from the public ``MP_MORR_TWO_MOMENT`` wrapper and
includes its radar diagnostic: every expected value was written by WRF.
``kernels/morrison.cu`` is the literal transcription graded here.  Built
with ``MORR_ROW37_OFF`` (WRF's arithmetic alone) it is bitwise on every
returned word. Row-37 corrections are measured in isolation below; production
also corrects negative sedimentation mass, which WRF can export as negative
surface snowfall. Production is graded for finite, nonnegative water too.
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

#: The production build's named row-37 exceptions on the fixture: the only
#: fixture column where the nucleation cap fires is case 13 (205 K, ice
#: supersaturated above the 1.08 Cooper threshold), under both modes.  The
#: cap shrinks the nucleated number with the vapour it may consume; mass,
#: vapour and temperature move by the FUDGEF rounding of the capped rate.
ROW37_FIXTURE_CASES = [[13, 0], [13, 1]]
ROW37_FIXTURE_FIELDS = {"ni", "qi", "qv"}


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


def test_the_unit_is_routed_through_wrfs_libm_and_no_contraction():
    """The causes the former xfail named are gone, and stay gone."""
    import re
    from gpuwm.core.kernels import EXTRA_HEADERS, module_options

    assert EXTRA_HEADERS["morrison"] == (
        "glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_flt64.cuh",
        "glibc_trig_flt32.cuh", "morrison_constants.cuh")
    assert module_options("morrison") == ("-std=c++17", "--fmad=false")
    source = re.sub(r"//[^\n]*", "", (
        Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
        / "morrison.cu").read_text(encoding="utf-8"))
    # No CUDA libm entry point: WOOF's own libm (gfk_*, glibc_*) and the
    # module's own GAMMA (mo_gamma).
    bare = re.findall(r"(?<![A-Za-z0-9_])(powf|expf|logf|log10f|cbrtf|"
                      r"tgammaf|sinf|sqrtf|pow|exp|log|log10)\(", source)
    assert bare == []
    assert "__device__ float mo_gamma(float x)" in source
    # Every literal constant division is the IEEE quotient.
    assert not re.search(r"[A-Za-z0-9_)\]] / [0-9]", source)


# ---------------------------------------------------------------------------
# GPU measurement and acceptance contract.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _measurement_row37_off() -> dict[str, object]:
    return measure_oracle(row37=False)


@pytest.mark.gpu
@requires_gpu
def test_wrf_measurement_retains_complete_comparison_and_finite_outputs():
    report = _measurement()
    assert set(report["fields"]) == {
        "graupelnc", "graupelncv", "ng", "ni", "nr", "ns", "qc", "qg", "qi",
        "qr", "qs", "qv", "rainnc", "rainncv", "refl_10cm", "snownc",
        "snowncv", "sr", "theta"}
    for name, entry in report["fields"].items():
        assert entry["nonfinite_count"] == 0, name
    expected_counts = {
        name: (28 if name in {
            "graupelnc", "graupelncv", "rainnc", "rainncv",
            "snownc", "snowncv", "sr",
        } else 896)
        for name in report["fields"]
    }
    assert {
        name: int(entry["value_count"])
        for name, entry in report["fields"].items()
    } == expected_counts
    assert report["overall"]["value_count"] == 10948


@pytest.mark.gpu
@requires_gpu
def test_morrison_wrf_acceptance_is_bitwise():
    """WRF's arithmetic alone (MORR_ROW37_OFF): every returned word."""
    report = _measurement_row37_off()
    assert report["row37"] is False
    assert report["overall"]["value_count"] == 10948
    assert report["overall"]["mismatch_count"] == 0
    assert report["overall"]["max_ulp"] == 0
    assert report["overall"]["bitwise"] is True


@pytest.mark.gpu
@requires_gpu
def test_isolated_row37_corrections_retain_their_pinned_differences():
    """With transport correction isolated, row 37 retains its pinned scope."""
    report = measure_oracle(nonnegative_sedimentation=False)
    assert report["row37"] is True
    differing = {name for name, entry in report["fields"].items()
                 if entry["mismatch_count"]}
    assert differing == ROW37_FIXTURE_FIELDS
    for name in differing:
        assert sorted(report["fields"][name]["mismatch_cases"]) == (
            ROW37_FIXTURE_CASES), name


@pytest.mark.gpu
@requires_gpu
def test_production_transport_does_not_export_negative_water():
    report = _measurement()
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "rainnc", "rainncv",
                 "snownc", "snowncv", "graupelnc", "graupelncv", "sr"):
        assert report["fields"][name]["minimum"] >= 0.0, name
    # This fixture independently reproduces the WRF defect corrected by the
    # forecast fix: case 6 exported -1.13e-9 mm snowfall in both ice modes.
    snow = report["fields"]["snowncv"]
    assert sorted(snow["mismatch_cases"]) == [[6, 0], [6, 1]]
    assert snow["worst"]["want"] < 0.0
    assert snow["worst"]["got"] == 0.0


@pytest.mark.gpu
@requires_gpu
def test_wrf_comparison_detects_a_latent_heat_perturbation(monkeypatch):
    """A known wrong latent heat must break the actual WRF theta comparison."""
    import cupy as cp
    from gpuwm.core import kernels, morrison

    baseline = _measurement_row37_off()
    assert baseline["fields"]["theta"]["mismatch_count"] == 0
    source = kernels.module_source("morrison")
    coefficient = "const float xxlv = 3.1484E6f - t3d * 2370.0f;"
    assert source.count(coefficient) == 1
    changed = source.replace(coefficient,
                             "const float xxlv = 3.1494E6f - t3d * 2370.0f;")
    module = cp.RawModule(code="#define MORR_ROW37_OFF 1" + chr(10) + changed,
                          options=kernels.module_options("morrison"))
    original_get_kernel = morrison.get_kernel

    def perturbed_kernel(name, entry_point):
        if name == "morrison":
            return module.get_function(entry_point)
        return original_get_kernel(name, entry_point)

    monkeypatch.setattr(morrison, "get_kernel", perturbed_kernel)
    report = measure_oracle()
    assert report["fields"]["theta"]["mismatch_count"] > 0


@pytest.fixture(autouse=True, scope="module")
def _strict_source_contract():
    from gpuwm.core import kernels
    import gpuwm.wrf_exact as exact

    # A preceding default test may already have compiled Morrison. The source
    # selector is process-wide, while these caches are keyed by module name.
    kernels.get_kernel.cache_clear()
    kernels.load_module.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exact, "ENABLED", True)
        try:
            yield
        finally:
            kernels.get_kernel.cache_clear()
            kernels.load_module.cache_clear()
            _measurement.cache_clear()
            _measurement_row37_off.cache_clear()
