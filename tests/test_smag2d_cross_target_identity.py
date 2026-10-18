"""Randomized smag kernels preserve words across NVRTC architecture targets.

The fixtures include the TKE source tendency and all four exported budgets.

The linker emits all replay cubins for the available physical card. This test
isolates NVRTC target arithmetic; physical ptxas and hardware differences need
the paired architecture forecast proof. It rewrites a diagnostic PTX header,
never the engine's runtime compilation route.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from conftest import requires_gpu
from tools.cuda_cross_target_replay import (
    SCENARIOS, TARGETS, compiler_options, sources,
    compile_modules, load_modules, replay_scenario,
)


@pytest.fixture(scope="module")
def blackwell_word_fixtures():
    """Keep original answers except the measured, merged WRF TKE corrections.

    Regenerate the extension with tools/smag_wrf_word_extension.py using
    cuda_cross_target_replay artifacts on a certified card.
    """
    path = Path(__file__).with_name("fixtures") / "smag2d_blackwell_words.json"
    frozen = json.loads(path.read_text())
    assert frozen["schema"] == "gpuwm-smag2d-blackwell-random-fixtures-v1"
    assert frozen["reference_physical_target"] == 120
    assert frozen["runtime_ftz"] is True
    answers = {tuple(case["scenario"]): {
        (row["variant"], row["kernel"]): row for row in case["kernels"]
    } for case in frozen["scenarios"]}
    extension_path = path.with_name("smag2d_wrf_word_extension.json")
    assert hashlib.sha256(extension_path.read_bytes()).hexdigest() == "e1a6203d9a4d8a7c3510e801313082a5205b2e0d2544b6a14c0a1f3d15e8b705"
    extension = json.loads(extension_path.read_text())
    assert extension["schema"] == "smag-compiled-wrf-word-extension-v1"
    assert extension["original_fixture_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    generator = path.parents[2] / "tools/smag_wrf_word_extension.py"
    assert extension["generator_sha256"] == hashlib.sha256(generator.read_bytes()).hexdigest()
    assert extension["module_source_sha256"] == {
        name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources().items()}
    assert extension["options"] == list(compiler_options(120))
    assert [case["scenario"] for case in extension["scenarios"]] == [case["scenario"] for case in frozen["scenarios"]]
    corrected = set()
    for case in extension["scenarios"]:
        expected = answers[tuple(case["scenario"])]
        for row in case["kernels"]:
            key = (row["variant"], row["kernel"])
            assert row["kernel"] in {"wrf_tke_rhs", "wrf_calc_n2", "wrf_tke_km"}
            assert row["input_sha256"] == expected[key]["input_sha256"]
            assert row["output_arg_indices"] == expected[key]["output_arg_indices"]
            corrected.add(row["kernel"])
            expected[key] = row
    assert corrected == {"wrf_tke_rhs", "wrf_calc_n2", "wrf_tke_km"}
    return answers


@pytest.fixture(scope="module")
def cross_target_modules(tmp_path_factory):
    import cupy as cp

    supplied = os.environ.get("GPUWM_CROSS_TARGET_REPLAY_DIR")
    if supplied:
        return load_modules(Path(supplied))[0]
    outdir = tmp_path_factory.mktemp("smag-cross-target")
    try:
        compile_modules(outdir, int(cp.cuda.Device().compute_capability))
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return load_modules(outdir)[0]


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.gpu
@requires_gpu
def test_smag2d_compiler_targets_preserve_output_words(
        cross_target_modules, blackwell_word_fixtures, scenario):
    rows = replay_scenario(cross_target_modules, scenario)
    different = [row for row in rows if any(row["different_words"])
                 and (row["comparison"] == "original_blackwell"
                      or row["compute_target"] in TARGETS)]
    assert not different, different
    expected = blackwell_word_fixtures[scenario]
    actual = {(row["variant"], row["kernel"]): row for row in rows
              if row["comparison"] == "compiler_target" and row["compute_target"] == 120}
    assert actual.keys() == expected.keys(), "Every smag entry needs an original Blackwell word fixture"
    for key, reference in expected.items():
        row = actual[key]
        assert row["input_sha256"] == reference["input_sha256"], (
            key, "Fixture inputs changed; re-record from the original Blackwell answer")
        assert list(row["output_arg_indices"]) == reference["output_arg_indices"]
        assert row["output_sha256"] == reference["output_sha256"], (
            key, "Pinned arithmetic changed the original Blackwell words", row, reference)
