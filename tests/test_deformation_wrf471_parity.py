"""Full-word gates against compiled WRF v4.7.1 deformation and K routines.

The measured GPU words and their distances are pinned together. There is no
allclose gate and no skipped boundary strip. Numerical attribution is recorded
in the no-FMA and native-metric diagnostic receipts beside the fixtures.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from conftest import requires_gpu

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"tests/data/wrf471_diffusion"
TOOLS=ROOT/"tools/wrf_diffusion_oracle"
sys.path.insert(0,str(TOOLS))


def _manifest():
    return json.loads((DATA/"deformation-manifest.json").read_text())


def _case_names():
    return [entry["file"] for entry in _manifest()["cases"]]


@lru_cache(maxsize=None)
def _measure(name):
    from deformation_compare import measure_fixture
    return measure_fixture(DATA/name)


def _baseline(name, measured):
    from gpuwm.core.kernels import module_source
    source_sha = hashlib.sha256(module_source("smag2d").encode()).hexdigest()
    pins = []
    for path in sorted(DATA.glob("deformation-gpu*receipt.json")):
        receipt = json.loads(path.read_text())
        if (receipt["kernel_source_sha256"] == source_sha
                and not receipt.get("diagnostic_no_fma", False)
                and not receipt.get("diagnostic_reference_metrics", False)
                and not receipt.get("diagnostic_reference_order", False)):
            pins.append((path.name, receipt["cases"][name]))
    assert pins, "No production deformation receipt captures the current kernel source"
    for _filename, expected in pins:
        # One complete case must equal one captured platform variant. No
        # branch or field can choose a different pin or distance tolerance.
        if measured == expected:
            return expected
    raise AssertionError(f"{name}: complete output case differs from every current-source pin "
                         f"{[filename for filename, _expected in pins]}")


def test_deformation_fixtures_pin_compiled_wrf_sources_and_all_array_words():
    for line in (DATA/"deformation-sha256sums.txt").read_text().splitlines():
        digest,relative=line.split("  ",1)
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    manifest=_manifest()
    assert manifest["wrf_release"]=="4.7.1"
    assert len(manifest["cases"])==14
    assert len({entry["sha256"] for entry in manifest["cases"]})==14
    for entry in manifest["cases"]:
        path=DATA/entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"]
        with np.load(path) as fixture:
            meta=json.loads(str(fixture["meta_json"]))
            assert meta["nz"]==49
            assert set(k.removeprefix("ref__km4_iso0__") for k in fixture.files
                       if k.startswith("ref__km4_iso0__"))=={
                "div","d11","d22","d33","d12","d13","d23","kmh","kmv","khh","khv","bn2"}
            for branch in ("km2_iso0","km2_iso1","km3_iso0","km3_iso1"):
                assert set(k.removeprefix(f"ref__{branch}__") for k in fixture.files
                           if k.startswith(f"ref__{branch}__"))=={"kmh","kmv","khh","khv","bn2"}
    receipt=json.loads((DATA/"deformation-build-receipt.json").read_text())
    assert receipt["wrf_commit"]=="f52c197ed39d12e087d02c50f412d90d418f6186"
    assert receipt["source_sha256"]=="a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54"
    assert receipt["constants_sha256"]=="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
    assert set(receipt["routines"])>={"cal_deform_and_div","calculate_km_kh","calculate_N2",
                                     "smag2d_km","tke_km","smag_km","compute_diff_metrics"}
    assert "-ffp-contract=off" in receipt["commands"][0]
    assert "-fcheck=bounds" in receipt["commands"][0]


def test_deformation_fixtures_include_real_vertical_motion_and_both_hemispheres():
    cases=[]
    for name in _case_names():
        with np.load(DATA/name) as f:
            cases.append((name,float(np.max(np.abs(f["input__w"]))),
                          float(f["input__lat"].min()),float(f["input__lat"].max())))
    assert any(name.startswith("deformation-evolved_") and speed>0 for name,speed,_,_ in cases)
    assert any(low>0 for _,_,low,_ in cases)
    assert any(high<0 for _,_,_,high in cases)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",_case_names())
def test_cal_deform_and_div_matches_compiled_wrf_word_receipt(name):
    case = _measure(name)
    measured=case["km4_iso0"]
    expected=_baseline(name, case)["km4_iso0"]
    for field in ("div","d11","d22","d33","d12","d13","d23"):
        assert measured[field]==expected[field],field


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",_case_names())
@pytest.mark.parametrize("branch",["km4_iso0","km2_iso0","km2_iso1","km3_iso0","km3_iso1"])
def test_calculate_km_kh_matches_compiled_wrf_word_receipt(name,branch):
    case = _measure(name)
    measured=case[branch]
    expected=_baseline(name, case)[branch]
    for field in ("kmh","kmv","khh","khv","bn2"):
        assert measured[field]==expected[field],field


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",[name for name in _case_names() if name.endswith("open.npz")])
def test_deformation_open_tensor_copies_and_n2_excluded_rows(name):
    from deformation_compare import port_outputs
    with np.load(DATA/name) as fixture:
        arrays={k.removeprefix("input__"):fixture[k] for k in fixture.files if k.startswith("input__")}
        meta=json.loads(str(fixture["meta_json"]))
    got=port_outputs(arrays,meta)
    for field in ("d12","d13","d23"):
        values=got[field].view(np.uint32)
        assert np.array_equal(values[:,:,0],values[:,:,1]),field
        assert np.array_equal(values[:,0,:],values[:,1,:]),field
    bn2=got["bn2"].view(np.uint32)
    assert (bn2[:,:,[0,-1]]==0).all()
    assert (bn2[:,[0,-1],:]==0).all()
