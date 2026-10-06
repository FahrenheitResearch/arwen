"""All output words from the compiled WRF diffusion outer drivers."""
import ast
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import warnings
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.verify.advect_oracle import UncertifiedReceiptStale, advect_gpu_identity, architecture_is_certified

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

# Byte identity is certified on Blackwell and newer only (ruling 2026-10-04),
# the rule gpuwm.verify.advect_oracle applies to the advection receipts.  A
# receipt here records only its device name, so each card that measured one
# is declared with its compute capability; a receipt from an undeclared card
# fails, so a new receipt states whether it gates.
#
# The breakage this prevents: the RTX 4090 receipts holding the driver word
# test red on a certified card after a smag2d move.  1f625334f moved smag2d
# from b60a0290 to 400d89d0, and every pin had to wait for an RTX 4090 to
# recapture uncertified receipts (89860a701), the same way the H100 advection
# receipt held the advection pin test red after b70a94a48.
RECEIPT_CAPABILITY={"NVIDIA GeForce RTX 5090":(12,0),"NVIDIA GeForce RTX 4090":(8,9)}
_HEX64=re.compile(r"[0-9a-f]{64}")

def _receipt_device(pin):
    device=pin["device"]
    return ast.literal_eval(device).decode() if device.startswith(("b'",'b"')) else device

def _smag2d_source():
    from gpuwm.core.kernels import module_source
    return hashlib.sha256(module_source("smag2d").encode()).hexdigest()

def _word_pins(folder,root=None,source=None):
    """The receipts whose words this tree accepts, as ``[(device, receipt)]``.

    A Blackwell or newer receipt must pin the smag2d source the loader
    compiles, and at least one such receipt must exist.  An older card's
    receipt that trails the source is reported (UncertifiedReceiptStale) and
    its words are not accepted, since they were measured on other source.
    """
    source=source or _smag2d_source()
    accepted,certified=[],0
    paths=sorted((Path(root or DATA)/folder).glob("gpu-receipt*.json"))
    assert paths,folder
    for path in paths:
        pin=json.loads(path.read_text())
        device=_receipt_device(pin)
        assert device in RECEIPT_CAPABILITY,(
            f"{folder}/{path.name} was measured on {device!r}, which RECEIPT_CAPABILITY does not declare")
        assert _HEX64.fullmatch(str(pin.get("kernel_source_sha256"))) and pin.get("cases"),(folder,path.name)
        major,minor=RECEIPT_CAPABILITY[device]
        if pin["kernel_source_sha256"]==source:
            accepted.append((device,pin))
            certified+=architecture_is_certified((major,minor))
        elif architecture_is_certified((major,minor)):
            raise AssertionError(
                f"{folder}/{path.name} ({device}) pins smag2d {pin['kernel_source_sha256'][:8]}, not the "
                f"source the loader compiles ({source[:8]}): the kernel moved and that certified card "
                "has not reproduced its words at it")
        else:
            warnings.warn(UncertifiedReceiptStale(
                f"{folder}/{path.name} ({device}, sm_{major}{minor}) is informational, not a gate: byte "
                "identity is certified on Blackwell and newer only (ruling 2026-10-04). It trails the "
                f"tree: smag2d {pin['kernel_source_sha256'][:8]} -> {source[:8]}. A capture on that card "
                "brings it current; its words are not accepted until then."))
    assert certified,f"{folder} has no Blackwell or newer receipt at the smag2d source the loader compiles"
    return accepted

def _device_word_pins(folder):
    """The accepted receipts for this card, resolved before any launch.

    An older card with no receipt of its own at the current source is not
    certified, so its word comparison is skipped and says why; a Blackwell
    or newer card always compares.
    """
    accepted=_word_pins(folder)
    gpu,capability=advect_gpu_identity()
    if not architecture_is_certified(capability) and gpu not in {device for device,_ in accepted}:
        pytest.skip(f"{gpu} (sm_{capability[0]}{capability[1]}) has no {folder} receipt at the current "
                    "smag2d source and is not certified: byte identity is certified on Blackwell and "
                    "newer only (ruling 2026-10-04)")
    return [pin for _,pin in accepted]

def test_driver_receipts_gate_blackwell_and_report_older_cards(tmp_path,monkeypatch):
    """The 2026-10-04 ruling for the driver word receipts, without a card.

    Both ways: a stale RTX 4090 receipt must not hold the gate red, and a
    stale or missing Blackwell receipt, or an undeclared card, must.
    """
    for folder in ("horizontal-driver","vertical-driver","km-mutations"):
        # The committed tree: a certified receipt is current (a stale older
        # receipt is only reported, so it does not fail here either).
        assert any(architecture_is_certified(RECEIPT_CAPABILITY[device]) for device,_ in _word_pins(folder))
    blackwell=json.loads((DATA/"km-mutations/gpu-receipt.json").read_text())
    source=blackwell["kernel_source_sha256"]
    # Laid out current, whatever state the committed RTX 4090 receipt is in.
    ada=dict(json.loads((DATA/"km-mutations/gpu-receipt-4090.json").read_text()),kernel_source_sha256=source)
    folder=tmp_path/"km-mutations"

    def lay(*pins):
        shutil.rmtree(folder,ignore_errors=True)
        folder.mkdir()
        for index,pin in enumerate(pins):
            (folder/f"gpu-receipt-{index}.json").write_text(json.dumps(pin))

    lay(blackwell,dict(ada,kernel_source_sha256="0"*64))
    with pytest.warns(UncertifiedReceiptStale,match="informational, not a gate.*00000000 -> "+source[:8]):
        assert [device for device,_ in _word_pins("km-mutations",tmp_path,source)]==["NVIDIA GeForce RTX 5090"]
    lay(dict(blackwell,kernel_source_sha256="0"*64),ada)
    with pytest.raises(AssertionError,match="certified card has not reproduced"):
        _word_pins("km-mutations",tmp_path,source)
    lay(ada)
    with pytest.raises(AssertionError,match="no Blackwell or newer receipt"):
        _word_pins("km-mutations",tmp_path,source)
    lay(blackwell,dict(ada,device="b'NVIDIA H100 80GB HBM3'"))
    with pytest.raises(AssertionError,match="does not declare"):
        _word_pins("km-mutations",tmp_path,source)
    module=sys.modules[__name__]
    monkeypatch.setattr(module,"_smag2d_source",lambda:source)
    monkeypatch.setattr(module,"DATA",tmp_path)

    def card(name,capability):
        monkeypatch.setattr(module,"advect_gpu_identity",lambda:(name,capability))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UncertifiedReceiptStale)
            return _device_word_pins("km-mutations")

    lay(blackwell,dict(ada,kernel_source_sha256="0"*64))
    assert card("NVIDIA GeForce RTX 5090",(12,0))==[blackwell]
    assert card("unmeasured Blackwell",(10,0))==[blackwell]
    with pytest.raises(pytest.skip.Exception,match="not certified"):
        card("NVIDIA GeForce RTX 4090",(8,9))
    lay(blackwell,ada)
    assert card("NVIDIA GeForce RTX 4090",(8,9))==[blackwell,ada]

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("folder",["horizontal-driver","vertical-driver","km-mutations"])
def test_diffusion_driver_and_mutable_every_output_word_receipt(folder):
    expected=_device_word_pins(folder)
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
    pins=_device_word_pins("vertical-driver")
    measured=_measure("vertical-driver",name)
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
