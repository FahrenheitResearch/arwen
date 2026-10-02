"""Every sixth-order tendency word against compiled, unmodified WRF v4.7.1.

The oracle is WRF's routine, not a Python transcription. Its source byte slice,
constants, compiler settings, driver, state, and resulting fixtures are pinned.
Five field staggerings cover specified, nested, and four-sided open boundaries.
No tolerance or relative-error comparison is used.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/wrf471_diffusion"
TOOLS = ROOT / "tools/wrf_diffusion_oracle"


def oracle():
    return json.loads((DATA / "diff6-wrf471.json").read_text()), np.load(DATA / "diff6-wrf471.npz")


def inputs(fixture, case):
    prefix=f"c{case:03d}__"
    return {key[len(prefix):]:fixture[key] for key in fixture.files if key.startswith(prefix)}


def launch(settings, values, *, identity=False):
    from gpuwm.verify.diffusion_oracle import launch_diff6_fixture
    return launch_diff6_fixture(settings, values, identity=identity)


def test_fixture_is_the_pinned_wrf_driver_and_its_receipts():
    metadata,_=oracle()
    for line in (DATA / "diff6-oracle-sha256sums.txt").read_text().splitlines():
        digest,filename=line.split()
        assert hashlib.sha256((DATA/filename).read_bytes()).hexdigest()==digest, filename
    for filename,digest in metadata["tools_sha256"].items():
        assert hashlib.sha256((TOOLS/filename).read_bytes()).hexdigest()==digest, filename
    assert metadata["wrf_version"]=="4.7.1"
    assert metadata["wrf_commit"]=="f52c197ed39d12e087d02c50f412d90d418f6186"
    assert metadata["source_sha256"]=="bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
    assert metadata["constants_sha256"]=="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
    assert metadata["routine_slice_sha256"]=="a4534919fdb15c91789f6d0dc504fe12f8eac6ee711f51053b3900999a055567"
    assert metadata["routine_source_lines"]==[6220,6636]
    assert metadata["slice_is_byte_unmodified"]
    assert metadata["real_kind_bytes"]==4
    assert "-ffp-contract=off" in metadata["commands"][0]
    assert "-fcheck=bounds" in metadata["commands"][0]
    assert metadata["raw_decoder"]=="rw_netcdf dump --raw"
    assert len(metadata["real_state_sha256"])==64


def test_fixture_discriminates_staggers_switches_and_realistic_edges():
    metadata,fixture=oracle()
    cases=metadata["cases"]
    assert len(cases)==200
    assert {r["name"] for r in cases}==set("uvwtq")
    assert {r["boundary_mode"] for r in cases}=={0,1,2,3}
    assert {r["opt"] for r in cases}=={1,2}
    assert {r["slopeopt"] for r in cases}=={0,1}
    assert any(r["dx"]!=r["dy"] for r in cases)
    assert any(r["dt"]==3.7 for r in cases)
    values=[inputs(fixture,r["case"]) for r in cases]
    assert all(np.isfinite(v["reference"]).all() for v in values)
    assert any((v["c1"]>0).any() and (v["c1"]<1).any() and (v["c2"]>0).any() for v in values)
    assert any(np.max(v["mapm"])>2 and np.min(v["mapm"])<.5 for v in values)
    assert any((v["latitude"]<0).all() for v in values)
    assert any((v["latitude"]>0).all() for v in values)
    zero=[v for r,v in zip(cases,values) if r["scenario"]=="zero"]
    assert all(not v["reference"].view(np.uint32).any() for v in zero)
    near=[v for r,v in zip(cases,values) if r["scenario"]=="near_zero"]
    assert all((v["reference"]!=0).any() for v in near)
    for r,v in zip(cases,values):
        ref=v["reference"]
        if r["boundary_mode"]==0:
            if r["name"]=="w":assert np.array_equal(ref[[0,-1]],v["tendency"][[0,-1]])
            continue
        assert not ref[:,:,:3].view(np.uint32).any()
        assert not ref[:,:,-3:].view(np.uint32).any()
        assert not ref[:,:3,:].view(np.uint32).any()
        assert not ref[:,-3:,:].view(np.uint32).any()
        if r["name"]=="w":assert not ref[[0,-1]].view(np.uint32).any()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",list("uvwtq"))
def test_sixth_order_diffusion_is_bit_identical_to_compiled_wrf(name):
    metadata,fixture=oracle()
    words=0
    for settings in metadata["cases"]:
        if settings["name"]!=name:continue
        values=inputs(fixture,settings["case"])
        got=launch(settings,values)
        want=values["reference"]
        changed=got.view(np.uint32)!=want.view(np.uint32)
        assert not changed.any(), (settings,int(changed.sum()),np.argwhere(changed)[:5].tolist())
        words+=want.size
    print(f"sixth_order_diffusion name={name} cases=40 words={words} mismatches=0 max_ulp=0")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",["u","v"])
def test_true_boundary_seam_faces_are_compiled_wrf_words(name):
    metadata,fixture=oracle()
    seen=0
    for settings in metadata["cases"]:
        if settings["name"]!=name or settings["boundary_mode"]==0:continue
        values=inputs(fixture,settings["case"])
        got=launch(settings,values)
        want=values["reference"]
        seam=(slice(None),slice(3,-3),-4) if name=="u" else (slice(None),-4,slice(3,-3))
        assert np.array_equal(got[seam].view(np.uint32),want[seam].view(np.uint32)),settings
        seen+=int(np.count_nonzero(want[seam]))
    assert seen>0, "boundary fixture no longer discriminates the true boundary datum"


@pytest.mark.gpu
@requires_gpu
def test_projected_fixture_rejects_omitted_tendency_map_factors():
    metadata,fixture=oracle()
    changed=0
    for settings in metadata["cases"]:
        if settings["scenario"]!="real_lower_projected" or settings["slopeopt"]!=0:continue
        values=inputs(fixture,settings["case"])
        mutant=launch(settings,values,identity=True)
        changed+=int(np.count_nonzero(mutant.view(np.uint32)!=values["reference"].view(np.uint32)))
    assert changed>1000, "real projection fixture would allow the original missing-map defect"
    print(f"omitted-tendency-map mutation rejected: different_words={changed}")
