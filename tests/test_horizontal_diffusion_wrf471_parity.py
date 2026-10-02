"""Every horizontal tendency word compared with compiled WRF v4.7.1.

The finite differences from WRF are measurements, asserted for equality to
the recorded full-output hashes and distances. No tolerance hides a change.
Tools-only controls isolate metric rounding and FMA; the gate uses the
unmodified production launchers with their normal compiler options.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import numpy as np
import pytest
from conftest import requires_gpu

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"tests/data/wrf471_diffusion/horizontal"

def test_horizontal_diffusion_wrf471_fixture_and_compiler_pins():
    for line in (DATA/"oracle-sha256sums.txt").read_text().splitlines():
        digest,relative=line.split("  ",1)
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    manifest=json.loads((DATA/"manifest.json").read_text())
    assert len(manifest["cases"])==14
    for file,digest in manifest["files"].items():
        assert hashlib.sha256((DATA/file).read_bytes()).hexdigest()==digest
    for name,digest in manifest["tools_sha256"].items():
        assert hashlib.sha256((ROOT/"tools/wrf_diffusion_oracle"/name).read_bytes()).hexdigest()==digest,name
    for key in ("fortran_build","preparation_build"):
        receipt=manifest[key]
        assert receipt["wrf_commit"]=="f52c197ed39d12e087d02c50f412d90d418f6186"
        assert receipt["source_sha256"]=="a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54"
        assert receipt["constants_sha256"]=="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
        assert "-ffp-contract=off" in receipt["commands"][0]
        assert "-fcheck=bounds" in receipt["commands"][0]
    assert hashlib.sha256((ROOT/"tools/wrf_diffusion_oracle/horizontal_wrapper.F90").read_bytes()).hexdigest()==manifest["fortran_build"]["wrapper_sha256"]
    for case in manifest["cases"]:
        with np.load(DATA/case["file"]) as fixture:
            for key in fixture.files:
                assert np.isfinite(fixture[key]).all(),(case["name"],key)
    with np.load(DATA/"evolved_real_open.npz") as fixture:
        assert np.any(fixture["input_w"]!=0.)
        assert np.any(fixture["wrf_thp_w"]!=0.)

def _compare_tool():
    directory=str(ROOT/"tools/wrf_diffusion_oracle")
    if directory not in sys.path: sys.path.insert(0,directory)
    return importlib.import_module("horizontal_compare")

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("routine",["u","v","w","s"])
def test_horizontal_diffusion_wrf471_every_output_word_is_pinned(routine):
    manifest=json.loads((DATA/"manifest.json").read_text())
    receipts=[json.loads(path.read_text()) for path in sorted(DATA.glob("gpu*receipt.json"))]
    assert receipts
    from gpuwm.core.kernels import module_source
    source_hash=hashlib.sha256(module_source("smag2d").encode()).hexdigest()
    receipts=[receipt for receipt in receipts if receipt["kernel_source_sha256"]==source_hash]
    assert receipts,"No current-source horizontal word receipt"
    words=0
    for case in manifest["cases"]:
        measured=_compare_tool().compare_case(DATA/case["file"],case)
        # A complete case must match one observed platform variant. A device
        # name never selects a tolerance or substitutes another field's pin.
        assert any(measured==receipt["cases"][case["name"]] for receipt in receipts),(case["name"],measured)
        fields=("thp","qv") if routine=="s" else ("thp",)
        for field in fields:
            key=field+"_"+routine
            # The hash checks all GPU words; the statistics compare all words
            # with the independently compiled WRF output in the fixture.
            words+=measured[key]["words"]
    print(f"horizontal_diffusion_{routine}: cases={len(manifest['cases'])*len(fields)} words={words}")
