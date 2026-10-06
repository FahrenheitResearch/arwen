"""Blackwell advection compiler targets preserve words for every boundary path."""
import os
import json
from pathlib import Path

import pytest

from conftest import requires_gpu
from tools.advection_cross_target_replay import (
    SCENARIOS, compile_advection, load_advection, replay_scenario, validate_artifacts,
)

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.fixture(scope="module")
def advection_target_artifacts(tmp_path_factory):
    import cupy as cp

    supplied = os.environ.get("GPUWM_ADVECTION_TARGET_REPLAY_DIR")
    if supplied:
        return Path(supplied)
    outdir = tmp_path_factory.mktemp("advection-cross-target")
    try:
        compile_advection(outdir, int(cp.cuda.Device().compute_capability))
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return outdir


@pytest.fixture(scope="module")
def advection_target_modules(advection_target_artifacts):
    return load_advection(advection_target_artifacts)[0]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_advection_compiler_targets_preserve_words(advection_target_modules, scenario):
    rows = replay_scenario(advection_target_modules, scenario)
    different = [row for row in rows if row["variant"] in (
                     "candidate_100", "candidate_120")
                 and row["different_vs_candidate120"]]
    different += [row for row in rows if row["variant"] == "original_100"
                  and row["different_vs_original_compute120"]]
    if "native_original_120" in advection_target_modules:
        different += [row for row in rows if row["variant"] == "native_candidate_120"
                      and row["different_vs_original120"]]
    assert not different, different


@pytest.mark.parametrize("damage", ("source", "target", "options", "cubin"))
def test_advection_rejects_stale_or_damaged_artifacts(advection_target_artifacts, tmp_path, damage):
    receipt = json.loads((advection_target_artifacts / "compile-receipt.json").read_text())
    candidate = None
    if damage == "source":
        candidate = tmp_path / "changed.cu"
        candidate.write_text("// A different source must reject the saved cubins.\n")
        expected = "different candidate source"
    elif damage == "target":
        del receipt["variants"]["candidate_100"]
        expected = "required advection compiler targets"
    elif damage == "options":
        first = next(row for row in receipt["variants"].values() if row["status"] == "linked")
        first["options"].remove("-ftz=true")
        expected = "effective runtime options"
    else:
        first = next(row for row in receipt["variants"].values() if row["status"] == "linked")
        (tmp_path / first["cubin"]).write_bytes((advection_target_artifacts / first["cubin"]).read_bytes() + b"damaged")
        expected = "cubin checksum changed"
    (tmp_path / "compile-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match=expected):
        validate_artifacts(tmp_path, candidate_source=candidate)
