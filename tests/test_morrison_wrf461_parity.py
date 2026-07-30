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

# Per-field max_ulp measured on both authorized RTX 5090s.  The Windows driver
# and Linux/CUDA 12.9 NVRTC produce slightly different transcendental results,
# although both cards have the same architecture.  Pin both complete signatures
# rather than making platform-specific mismatch counts part of the physics
# contract.  fp32_ulp_distance in gpuwm.core.fp32_ulp is the sole ULP owner.
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
MEASURED_PLATFORM_MAX_ULP = (
    MEASURED_WINDOWS_MAX_ULP,
    MEASURED_LINUX_MAX_ULP,
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
    reason=(
        "WRF acceptance contract is max_ulp 0; the measured production "
        "kernel is pinned by test_measured_morrison_wrf_residual_cannot_"
        "drift_silently"
    ),
)
def test_morrison_wrf_acceptance_is_bitwise():
    report = _measurement()
    assert report["overall"]["max_ulp"] == 0
    assert report["overall"]["bitwise"] is True
