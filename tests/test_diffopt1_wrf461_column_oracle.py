"""diff_opt=1 (coordinate-surface mixing, km_opt 2 and 4) against WRF v4.6.1.

The sealed set (tests/data/diffopt1_wrf461, built by
tools/diffopt1_wrf461_oracle/oracle.py seal) holds six 8 x 7 column windows
of real WRF v4.6.1 history -- convective, stable night, snow over ocean,
high terrain with snow, warm cloud over ocean on a periodic domain, and the
signed-zero edge case -- each with the words of the unmodified WRF Fortran
and of the periodic-seam control, for five diff_opt=1 configurations.

Breakage these gates prevent: the production diff_opt=1 path
(``_compute_wrf_smag_tendencies`` -> ``_compute_coordinate_tendencies``)
drifting from WRF's compiled words in any coefficient or tendency under the
strict build, including the TKE horizontal mixing and WRF's t_init theta
reference, without anything failing.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/diffopt1_wrf461"
ORACLE = ROOT / "tools/diffopt1_wrf461_oracle/oracle.py"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"


def _manifest():
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def test_sealed_words_are_pinned_to_wrf461_and_hash_sealed():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diffopt1_wrf461_oracle_build", ORACLE.with_name("build.py"))
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    manifest = _manifest()
    assert manifest["schema"] == "diffopt1-wrf461-sealed-v1"
    assert manifest["columns"] >= 64
    for tag, control in (("wrf", None), ("ctl", "periodic-top-slope")):
        receipt = manifest["builds"][tag]
        assert receipt["wrf_commit"] == WRF_COMMIT
        assert receipt["source_sha256"] == build.SOURCE_PINS
        assert receipt["flags"] == "strict" and receipt["control"] == control
        assert "-O0" in receipt["commands"][0] and "-ffp-contract=off" in receipt["commands"][0]
    arms = set(manifest["arms"])
    assert arms == {"km4", "km4_pbl0", "km2", "km2_isotropic", "km2_seed_pbl0"}
    regimes = set()
    for row in manifest["cases"]:
        path = DATA / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        regimes.add(row["case"].split("-")[0])
        with np.load(path) as data:
            words = [k for k in data.files if k.startswith("ctl__")]
            assert {k.split("__")[1] for k in words} == arms
            moved = sum(int(np.count_nonzero(data[k].view(np.uint32)
                                             != data["ctl__" + k[5:]].view(np.uint32)))
                        for k in data.files if k.startswith("wrf__"))
        # The seam control edits periodic-seam loops only.
        assert moved == row["words_moved_by_seam_control"]
        assert (moved > 0) == (row["boundary"] == "periodic")
    assert len(regimes) == len(manifest["cases"])


def _replay(tmp_path, **env_extra):
    env = os.environ.copy()
    for key in ("GPUWM_WRF_EXACT", "GPUWM_WRF_EXACT_DIFFUSION"):
        env.pop(key, None)
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")])
    out = tmp_path / "replay.json"
    # A separate process: the strict build changes the compiled kernel
    # sources and must not leak into this process's kernel cache.
    subprocess.run([sys.executable, str(ORACLE), "replay", "--sealed", str(DATA),
                    "--json", str(out)], check=True, env=env, cwd=ROOT)
    return json.loads(out.read_text())


def _differences(result, arms, reference="ctl"):
    return {f"{case_arm}:{field}": row[reference]
            for case_arm, fields in result["cases"].items()
            if case_arm.split("__")[1] in arms
            for field, row in fields.items() if row[reference]}


def _seam_moved():
    moved = {}
    for row in _manifest()["cases"]:
        with np.load(DATA / row["file"]) as data:
            for key in data.files:
                if key.startswith("wrf__"):
                    _, arm, field = key.split("__")
                    moved[f"{row['case']}__{arm}:{field}"] = int(np.count_nonzero(
                        data[key].view(np.uint32) != data["ctl__" + key[5:]].view(np.uint32)))
    return moved


KM4 = ("km4", "km4_pbl0")
KM2 = ("km2", "km2_isotropic", "km2_seed_pbl0")


@pytest.mark.gpu
@requires_gpu
def test_strict_km4_coordinate_mixing_every_wrf461_word(tmp_path):
    """Deformation, K, and every u/v/w/theta/moisture tendency, bitwise.

    Against unmodified WRF the only differing words are the ones WRF's
    periodic-seam slope defect moves (compute_diff_metrics leaves the
    model-top zx/zy at the seam unwritten); WOOF computes that slope.
    """
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    result = _replay(tmp_path, GPUWM_WRF_EXACT="1", GPUWM_WRF_EXACT_DIFFUSION="1")
    assert _differences(result, KM4) == {}
    against_wrf = _differences(result, KM4, "wrf")
    assert against_wrf == {k: v for k, v in _seam_moved().items()
                           if k.split("__")[1].split(":")[0] in KM4 and v}
    assert against_wrf, "the periodic case must exercise WRF's seam defect"


@pytest.mark.gpu
@requires_gpu
def test_strict_km2_coordinate_mixing_every_wrf461_word(tmp_path):
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    result = _replay(tmp_path, GPUWM_WRF_EXACT="1", GPUWM_WRF_EXACT_DIFFUSION="1")
    assert _differences(result, KM2) == {}


@pytest.mark.gpu
@requires_gpu
def test_strict_build_alone_selects_wrf_diffusion_order(tmp_path):
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    result = _replay(tmp_path, GPUWM_WRF_EXACT="1")
    assert _differences(result, KM4) == {}
