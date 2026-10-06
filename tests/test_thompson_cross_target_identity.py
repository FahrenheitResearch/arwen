"""Classic Thompson Blackwell compiler targets preserve words on frozen inputs."""
import os
import json
from pathlib import Path

import pytest

from conftest import requires_gpu
from tools.thompson_cross_target_replay import compile_thompson, load_thompson, replay, validate_artifacts

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.fixture(scope="module")
def thompson_target_artifacts(tmp_path_factory):
    import cupy as cp

    supplied = os.environ.get("GPUWM_THOMPSON_TARGET_REPLAY_DIR")
    if supplied:
        return Path(supplied)
    outdir = tmp_path_factory.mktemp("thompson-cross-target")
    try:
        compile_thompson(outdir, int(cp.cuda.Device().compute_capability))
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return outdir


@requires_gpu
def test_thompson_compiler_targets_preserve_words(thompson_target_artifacts):
    modules, _ = load_thompson(thompson_target_artifacts)
    rows = replay(modules)
    different = [row for row in rows if row["variant"] in (
                     "candidate_100", "candidate_120")
                 and any(row["different_vs_reference"])]
    different += [row for row in rows if row["variant"] == "original_100"
                  and any(row["different_vs_original_compute120"])]
    if "native_original_120" in modules:
        different += [row for row in rows if row["variant"] == "native_candidate_120"
                      and any(row["different_vs_original120"])]
    assert not different, different


@pytest.mark.parametrize("damage", ("source", "target", "options", "cubin"))
def test_thompson_rejects_stale_or_damaged_artifacts(thompson_target_artifacts, tmp_path, damage):
    receipt = json.loads((thompson_target_artifacts / "compile-receipt.json").read_text())
    candidate = None
    if damage == "source":
        candidate = tmp_path / "changed.cu"
        candidate.write_text("// A different source must reject the saved cubins.\n")
        expected = "source or runtime FTZ does not match"
    elif damage == "target":
        del receipt["variants"]["candidate_100"]
        expected = "required compiler target artifacts"
    elif damage == "options":
        first = next(row for row in receipt["variants"].values() if row["status"] == "linked")
        first["options"].remove("-ftz=true")
        expected = "effective runtime options"
    else:
        first = next(row for row in receipt["variants"].values() if row["status"] == "linked")
        (tmp_path / first["cubin"]).write_bytes((thompson_target_artifacts / first["cubin"]).read_bytes() + b"damaged")
        expected = "cubin checksum changed"
    (tmp_path / "compile-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match=expected):
        validate_artifacts(tmp_path, candidate_source=candidate)
