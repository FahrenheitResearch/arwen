"""All output words from the compiled WRF diffusion outer drivers."""
from functools import lru_cache
import hashlib
import importlib
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

def _manifest(folder):return json.loads((DATA/folder/"manifest.json").read_text())

def test_diffusion_driver_producing_tools_and_receipts_are_sealed():
    for line in (DATA/"diffusion-driver-sha256sums.txt").read_text().splitlines():
        digest,relative=line.split("  ",1)
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative

@pytest.mark.parametrize("folder,wrapper,cases",[("horizontal-driver","horizontal_driver_wrapper.F90",28),
    ("vertical-driver","vertical_driver_complete.F90",14),("km-mutations","deformation_wrappers.F90",14)])
def test_diffusion_driver_and_mutable_fixtures_have_compiled_fortran_pins(folder,wrapper,cases):
    manifest=_manifest(folder)
    assert len(manifest["cases"])==cases
    for name,digest in manifest["files"].items():
        assert hashlib.sha256((DATA/folder/name).read_bytes()).hexdigest()==digest,name
    for key in ("fortran_build","preparation_build"):
        if key not in manifest:continue
        receipt=manifest[key]
        assert receipt["wrf_commit"]=="f52c197ed39d12e087d02c50f412d90d418f6186"
        assert receipt["source_sha256"]=="a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54"
        assert receipt["constants_sha256"]=="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
        assert "-ffp-contract=off" in receipt["commands"][0]
        assert "-fcheck=bounds" in receipt["commands"][0]
    assert hashlib.sha256((TOOLS/wrapper).read_bytes()).hexdigest()==manifest["fortran_build"]["wrapper_sha256"]
    for entry in manifest["cases"]:
        with np.load(DATA/folder/entry["file"]) as fixture:
            for key in fixture.files:
                if key=="meta_json":continue
                assert np.isfinite(fixture[key]).all(),(entry["file"],key)
        if "source_fixture_sha256" in entry:
            source=DATA/entry["file"] if folder=="km-mutations" else DATA/"vertical"/("vertical-"+entry["name"]+".npz")
            assert hashlib.sha256(source.read_bytes()).hexdigest()==entry["source_fixture_sha256"]

@lru_cache(maxsize=None)
def _measure(folder,name):
    if folder=="horizontal-driver":
        return importlib.import_module("horizontal_driver").compare_case(DATA/folder/name)
    if folder=="vertical-driver":
        return importlib.import_module("vertical_driver").compare_case(DATA/folder/name)
    return importlib.import_module("deformation_mutations").compare_case(DATA/name,DATA/folder/name)

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("folder",["horizontal-driver","vertical-driver","km-mutations"])
def test_diffusion_driver_and_mutable_every_output_word_receipt(folder):
    expected=[json.loads(p.read_text()) for p in sorted((DATA/folder).glob("gpu-receipt*.json"))]
    from gpuwm.core.kernels import module_source
    assert expected
    assert all(hashlib.sha256(module_source("smag2d").encode()).hexdigest()==pin["kernel_source_sha256"] for pin in expected)
    for case in _manifest(folder)["cases"]:
        key=case["file"] if folder=="km-mutations" else case["name"]
        got=_measure(folder,case["file"])
        # Float32 division and contraction can produce different named
        # platform words. Each complete case must equal one recorded full
        # word variant; a distance threshold does not accept a new result.
        assert any(got==pin["cases"][key] for pin in expected),(key,got)

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name",["real_periodic.npz","evolved_real_periodic.npz","map_extremes.npz","evolved_map_extremes.npz",
                                "steep_open.npz","southern_open.npz"])
def test_vertical_driver_prescribed_heat_refreshes_hfx_output(name):
    measured=_measure("vertical-driver",name)
    pins=[json.loads(p.read_text()) for p in sorted((DATA/"vertical-driver").glob("gpu-receipt*.json"))]
    for km in (2,4):
        for flux in (0,2):
            result=measured[f"km{km}_flux{flux}"]["hfx_after"]
            # The Fortran HFX writer uses the same staged float32 CP and
            # moisture products. Only the independently formed density
            # remains different; the exact measured words are gated above.
            # This regression independently pins every HFX word, including
            # when it is selected without the complete-driver test.
            assert any(result==pin["cases"][name.removesuffix(".npz")][f"km{km}_flux{flux}"]["hfx_after"]
                       for pin in pins)

@requires_gpu
@pytest.mark.gpu
def test_calculate_km_kh_preserves_or_updates_every_mutable_word():
    for entry in _manifest("km-mutations")["cases"]:
        got=_measure("km-mutations",entry["file"])
        for arm,fields in got.items():
            for key,result in fields.items():
                assert result["different_words"]==0,(entry["file"],arm,key,result)

@requires_gpu
@pytest.mark.gpu
def test_prescribed_heat_with_no_hfx_output_preserves_dummy_mass():
    import cupy as cp
    from gpuwm.core.dycore import launch_wrf_smag2d_vertical
    from gpuwm.verify.diffusion_oracle import device_state
    from vertical_gpu import _config
    with np.load(DATA/"vertical-driver/real_periodic.npz") as fixture:
        arrays={k.removeprefix("input_"):fixture[k] for k in fixture.files if k.startswith("input_")}
        meta=json.loads(str(fixture["meta_json"]))
        km=cp.asarray(fixture["km4_kmh"])
    state=device_state(arrays,cf=tuple(meta[k] for k in ("cf1","cf2","cf3")))
    cfg=_config(meta,km_opt=4,isfflx=0,tke_heat_flux=.24)
    state.mup0.fill(.875)
    before=cp.asnumpy(state.mup0).view(np.uint32).copy()
    outputs=[cp.zeros_like(getattr(state,k)) for k in ("u","v","w","thp","qv")]
    launch_wrf_smag2d_vertical(state,cfg,km,ru=outputs[0],rv=outputs[1],rw=outputs[2],
        rth=outputs[3],rqv=outputs[4],time_t=False)
    np.testing.assert_array_equal(cp.asnumpy(state.mup0).view(np.uint32),before)
