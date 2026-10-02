"""Full-word comparison for the explicitly different constant-K operator.

These are direct compiled WRF diff_opt=1 routines. Every tendency word and
every mismatch are retained. Product word pins preserve the declared operator
contract and are not WRF parity claims. No tolerance accepts a difference.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"tests/data/wrf471_diffusion"
TOOLS=ROOT/"tools/wrf_diffusion_oracle"
ROUTINES=("horizontal_diffusion","horizontal_diffusion_3dmp","vertical_diffusion_u","vertical_diffusion_v","vertical_diffusion","vertical_diffusion_3dmp")


def fixture():
    return json.loads((DATA/"constant-wrf471.json").read_text()),np.load(DATA/"constant-wrf471.npz"),np.load(DATA/"constant-product-words.npz")


def values(fx,case):
    prefix=f"c{case:03d}__"
    return {k[len(prefix):]:fx[k] for k in fx.files if k.startswith(prefix)}


def launch(case,fx):
    from gpuwm.verify.diffusion_oracle import launch_constant_fixture
    return launch_constant_fixture(case,fx)


def test_constant_fixture_is_the_compiled_wrf_driver_and_its_receipts():
    meta,fx,pins=fixture()
    for line in (DATA/"constant-oracle-sha256sums.txt").read_text().splitlines():
        digest,name=line.split()
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    assert meta["wrf_commit"]=="f52c197ed39d12e087d02c50f412d90d418f6186"
    assert meta["source_sha256"]=="bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
    assert meta["constants_sha256"]=="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
    assert meta["real_kind_bytes"]==4
    assert set(meta["slices"])==set(ROUTINES)
    assert "-fcheck=bounds" in meta["flags"]
    assert len(meta["cases"])==56
    for name,digest in meta["tools_sha256"].items():
        assert hashlib.sha256((TOOLS/name).read_bytes()).hexdigest()==digest,name
    receipt=json.loads((DATA/"constant-product-comparison.json").read_text())
    for case in meta["cases"]:
        v=values(fx,case["case"]);prefix=f'c{case["case"]:03d}__'
        got=pins[prefix+"gpu"];want=v["reference"]
        different=got.view(np.uint32)!=want.view(np.uint32)
        assert np.array_equal(different,pins[prefix+"different_mask"]),case
        row=receipt["rows"][case["case"]]
        assert int(different.sum())==row["different_words"],case
        assert int(fp32_ulp_distance(got,want).max(initial=0))==row["max_ulp"],case


def test_constant_operator_divergences_are_declared_in_source():
    module=(ROOT/"gpuwm/core/diffusion.py").read_text()
    kernel=(ROOT/"gpuwm/core/kernels/diffusion.cu").read_text()
    assert "on *every* RK stage" in module
    assert "khdif/prandtl" in module
    assert "3x the momentum" in module
    assert "base-state dz" in module
    assert "zero-flux top/bottom boundaries" in kernel
    assert "all stencil" in kernel and "wrap over the core" in kernel


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("routine",ROUTINES)
def test_constant_operator_reports_every_compiled_wrf_difference(routine):
    from gpuwm.core.kernels import module_source
    meta,fx,pins=fixture()
    receipt=json.loads((DATA/"constant-product-comparison.json").read_text())
    assert hashlib.sha256(module_source("diffusion").encode()).hexdigest()==receipt["assembled_cuda_sha256"]
    total=changed=maximum=0
    for case in meta["cases"]:
        if case["routine"]!=routine:continue
        v=values(fx,case["case"]);prefix=f'c{case["case"]:03d}__'
        got=launch(case,v);want=v["reference"]
        # Both the entire product output and its difference trace are exact
        # pins. A changed output needs a new explained oracle assessment.
        assert np.array_equal(got.view(np.uint32),pins[prefix+"gpu"].view(np.uint32)),case
        different=got.view(np.uint32)!=want.view(np.uint32)
        assert np.array_equal(different,pins[prefix+"different_mask"]),case
        total+=got.size;changed+=int(different.sum())
        maximum=max(maximum,int(fp32_ulp_distance(got,want).max(initial=0)))
    assert changed>0, "DIFFERENT is the declared result; it must not silently become a parity claim"
    print(f"{routine}: DIFFERENT, words={total}, different_words={changed}, max_ulp={maximum}")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",["u","w","m"])
def test_constant_cartesian_checker_positive_control_is_bit_identical(name):
    meta,fx,_=fixture()
    case=next(c for c in meta["cases"] if c["scenario"]=="pure_meridional_checker" and c["name"]==name and c["op"]==0)
    v=values(fx,case["case"])
    assert np.array_equal(launch(case,v).view(np.uint32),v["reference"].view(np.uint32))


@pytest.mark.gpu
@requires_gpu
def test_wrf_meridional_v_flux_omits_mass_and_is_not_imitated():
    meta,fx,_=fixture()
    case=next(c for c in meta["cases"] if c["scenario"]=="pure_meridional_checker" and c["name"]=="v" and c["op"]==0)
    v=values(fx,case["case"]);got=launch(case,v);want=v["reference"]
    corrected=(want*v["coupling_mass"][None]).astype(np.float32)
    assert np.array_equal(got.view(np.uint32),corrected.view(np.uint32))
    assert not np.array_equal(got.view(np.uint32),want.view(np.uint32))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",["u","v","m"])
def test_declared_physical_zero_flux_boundary_differs_from_wrfs_copy_flux(name):
    meta,fx,_=fixture()
    case=next(c for c in meta["cases"] if c["scenario"]=="linear_shear" and c["name"]==name and c["op"]==1)
    v=values(fx,case["case"]);got=launch(case,v)
    assert not v["reference"][0].view(np.uint32).any()
    assert (got[0]!=0).all()
