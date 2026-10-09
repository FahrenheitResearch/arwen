"""km_opt=4 / diff_opt=2 against compiled WRF v4.6.1: every output word.

The fixtures (tests/data/smag2d_wrf461) come from tools/smag2d_wrf461_oracle:
27 cases, 4,536 columns, real 12 km and 3 km WRF v4.6.1 states plus named
edge cases, under WRF's specified, open and periodic lateral boundaries.
Each holds the input words both sides receive and the words compiled WRF
v4.6.1 wrote for them (deformation D11/D22/D12/D13/D23, xkmh/xkmv/xkhh and
the u, v, w, theta and eight water-species horizontal-diffusion tendencies).

The GPU test runs the production launchers under the strict build and asserts
0 differing words.  Periodic cases are held to a second reference, the WRF
build with one WRF defect corrected (compute_diff_metrics leaves the model-top
terrain slope at the periodic seam unwritten); against unmodified WRF they may
differ only at the top two mass levels, where that defect acts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import importlib.util

import numpy as np
import pytest
from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/smag2d_wrf461"
TOOL = ROOT / "tools/smag2d_wrf461_oracle"
MANIFEST = json.loads((DATA / "manifest.json").read_text())


def test_fixtures_are_sealed_and_cover_the_regimes():
    rows = MANIFEST["cases"]
    for row in rows:
        assert hashlib.sha256((DATA / row["file"]).read_bytes()).hexdigest() == row["sha256"], row["file"]
    assert len(rows) == 27 and MANIFEST["columns"] == sum(r["columns"] for r in rows) == 4536
    assert {r["boundary"] for r in rows} == {"specified", "open", "periodic"}
    names = {r["case"].rsplit("-", 1)[0] for r in rows}
    for regime in ("rockies_snow_wave", "plains_stable_night", "pacific_snow_storm", "gulf_cloud_ocean",
                   "atlantic_open_ocean", "steepest_terrain", "convective_plains_3km", "updraft_core",
                   "zero_flow", "signed_zero_flow", "near_zero_flow", "map_extremes", "strong_shear_cap"):
        assert regime in names, regime


def test_reference_is_pinned_unmodified_wrf_461_strict_build():
    build = MANIFEST["wrf_build"]
    assert build["wrf_release"] == "4.6.1" and build["control"] is None
    assert build["wrf_commit"] == "d66e442fccc04111067e29274c9f9eaccc3cef28"
    spec = importlib.util.spec_from_file_location("smag2d_wrf461_build", TOOL / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert build["source_sha256"] == module.SOURCE_PINS
    assert hashlib.sha256((TOOL / "pipeline.F90").read_bytes()).hexdigest() == build["pipeline_sha256"]
    for command in build["commands"][:-1]:
        for flag in ("-O0", "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds"):
            assert flag in command
    control = MANIFEST["wrfctl_build"]
    assert control["control"] == "periodic-top-slope"
    # The control differs from the reference only in compute_diff_metrics.
    changed = {name for name, span in control["routines"].items()
               if span["sha256"] != build["routines"][name]["sha256"]}
    assert changed == set()  # slices are hashed before the control edit
    assert control["generated_sha256"] != build["generated_sha256"]
    # mix_full_fields only subtracts a zero base wind in real-data WRF.
    assert not any(MANIFEST["mix_full_fields_false_differs"].values())


def test_wrf_leaves_the_periodic_seam_top_slope_unwritten():
    """The WRF v4.6.1 defect the engine does not copy, read from WRF's own words."""
    for row in MANIFEST["cases"]:
        with np.load(DATA / row["file"]) as data:
            zx, zy = data["wrf__zx"], data["wrf__zy"]
        top = zx.shape[0] - 1
        if row["boundary"] == "periodic":
            assert not zx[top, :, 0].any() and not zy[top, 0, :].any(), row["case"]
            if row["case"].startswith("rockies"):
                assert zx[top - 1, :, 0].any() and zx[top, :, 1].any()
        else:
            # Non-periodic WRF zeroes the boundary slope at every level.
            assert not zx[:, :, 0].any() and not zy[:, 0, :].any(), row["case"]


@pytest.mark.gpu
@requires_gpu
def test_strict_build_matches_every_compiled_wrf_461_word(tmp_path):
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPUWM_WRF_EXACT")}
    env["GPUWM_WRF_EXACT"] = "1"   # the strict build; its diffusion order is on by default
    for reference in ("wrf", "wrfctl"):
        subprocess.run([sys.executable, str(TOOL / "oracle.py"), "compare", "--cases", str(DATA),
                        "--reference", reference, "--receipt", str(tmp_path / f"{reference}.json"),
                        "--save", str(tmp_path / f"{reference}.npz")],
                       cwd=ROOT, env=env, check=True)
    receipt = json.loads((tmp_path / "wrf.json").read_text())
    assert receipt["arithmetic"]["strict_build"] and receipt["arithmetic"]["diffusion_order_selected"]
    assert "-DGPUWM_WRF_EXACT_C_DIFFUSION=1" in receipt["arithmetic"]["strict_options"]
    engine = np.load(tmp_path / "wrf.npz")
    periodic_words = 0
    for row in MANIFEST["cases"]:
        result = receipt["cases"][row["case"]]
        if row["boundary"] != "periodic":
            assert all(m["different_words"] == 0 for k, m in result.items() if not k.startswith("_")), row["case"]
            continue
        with np.load(DATA / row["file"]) as data:
            for field in MANIFEST["compared_fields"]:
                got = engine[f"{row['case']}__{field}"]
                assert np.array_equal(got.view(np.uint32), data["wrfctl__" + field].view(np.uint32)), (row["case"], field)
                bad = np.argwhere(got.view(np.uint32) != data["wrf__" + field].view(np.uint32))
                periodic_words += len(bad)
                top = got.shape[0] - (2 if field == "tw" else 1)
                assert (bad[:, 0] >= top - 1).all(), (row["case"], field, sorted(set(bad[:, 0])))
    assert periodic_words > 0   # the WRF seam defect is present and measured, not hidden
    control = json.loads((tmp_path / "wrfctl.json").read_text())
    assert control["total_different_words"] == 0 and control["total_words"] == receipt["total_words"]
