"""Randomized smag kernels preserve words across NVRTC architecture targets.

The fixtures include the TKE source tendency and all four exported budgets.

The linker emits all replay cubins for the available physical card. This test
isolates NVRTC target arithmetic; physical ptxas and hardware differences need
the paired architecture forecast proof. It rewrites a diagnostic PTX header,
never the engine's runtime compilation route.
"""
import json
import os
from pathlib import Path

import pytest

from conftest import requires_gpu
from tools.cuda_cross_target_replay import (
    SCENARIOS, TARGETS,
    compile_modules, load_modules, replay_scenario,
)


@pytest.fixture(scope="module")
def blackwell_word_fixtures():
    """Original sm_120 answers protect the rounding baseline without old code."""
    path = Path(__file__).with_name("fixtures") / "smag2d_blackwell_words.json"
    frozen = json.loads(path.read_text())
    assert frozen["schema"] == "gpuwm-smag2d-blackwell-random-fixtures-v1"
    assert frozen["reference_physical_target"] == 120
    assert frozen["runtime_ftz"] is True
    return {tuple(case["scenario"]): {
        (row["variant"], row["kernel"]): row for row in case["kernels"]
    } for case in frozen["scenarios"]}


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
