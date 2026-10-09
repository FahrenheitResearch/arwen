"""The compiled oracle cannot silently transpose fields or accept wrong buffers."""
import json
import hashlib
from pathlib import Path
import numpy as np
from gpuwm.verify.smallstep_oracle import ORACLE_DIR, WRFOracle, load_cases, word_metrics


def _layout(periodic=False):
    oracle = WRFOracle.__new__(WRFOracle)
    oracle.nx,oracle.ny,oracle.nz,oracle.periodic=8,7,49,periodic
    return oracle


def test_real_state_fixture_retains_staggering_and_all_levels():
    cases=list(load_cases())
    assert {name for name,_,_ in cases} == {"real_initial","real_evolved","steep_terrain",
                                           "map_extremes","zero_near_zero","southern_hemisphere"}
    for _,raw,meta in cases:
        nx,ny,nz=meta["nx"],meta["ny"],meta["nz"]
        assert raw["T"].shape==(nz,ny,nx)==(49,7,8)
        assert raw["U"].shape==(nz,ny,nx+1)
        assert raw["V"].shape==(nz,ny+1,nx)
        assert raw["PH"].shape==raw["W"].shape==(nz+1,ny,nx)
        assert all(a.dtype==np.float32 for a in raw.values())
    initial=cases[0][1]
    assert np.ptp(initial["PB"])>50000
    assert np.ptp(initial["U"])>1
    assert np.all(initial["ALT"]>0)


def test_wrf_array_layout_round_trip_and_clamped_ghosts():
    oracle=_layout()
    _,raw,_=next(load_cases())
    for key in ("U","V","W","T","MU","C1F"):
        original=raw[key]
        packed=oracle.array(original)
        assert packed.flags.f_contiguous
        assert np.array_equal(oracle.extract(packed,original.shape).view(np.uint32),original.view(np.uint32))
        if original.ndim==3:
            assert np.array_equal(packed[0],packed[1])
            assert np.array_equal(packed[:,:,0],packed[:,:,1])


def test_periodic_staggered_halos_use_the_mass_grid_period():
    oracle=_layout(True)
    u=np.broadcast_to(np.arange(9,dtype=np.float32),(49,7,9)).copy()
    u[:,:,-1]=u[:,:,0]
    packed=oracle.array(u)
    assert np.all(packed[0]==7)
    assert np.all(packed[9]==0)
    assert np.all(packed[10]==1)
    v=np.broadcast_to(np.arange(8,dtype=np.float32)[None,:,None],(49,8,8)).copy()
    v[:,-1]=v[:,0]
    packed=oracle.array(v)
    assert np.all(packed[:,:,0]==6)
    assert np.all(packed[:,:,8]==0)
    assert np.all(packed[:,:,9]==1)


def test_word_comparison_preserves_signed_zero_and_nonfinite_mismatches():
    measured=word_metrics(np.array([0.0,np.inf,np.nan],np.float32),
                          np.array([-0.0,-np.inf,1.0],np.float32))
    assert measured["different_words"]==3
    assert measured["nonfinite_different_words"]==2
    assert measured["max_ulp"]==0


def test_real_wrf_argument_schema_retains_every_output():
    schema=json.loads((ORACLE_DIR/"wrf-schema.json").read_text())
    assert len(schema["routines"] )==8
    assert set(schema["routines"]["calc_p_rho"]["args"]) >= {"al","p","ph","pm1","t0","ids","ims","its"}
    assert set(schema["routines"]["advance_mu_t"]["args"]) >= {"muave","muts","mudf","t_ave","ww_1"}
    assert schema["routines"]["advance_w"]["declarations"]["t_2ave"]["intent"]=="inout"


def test_smallstep_fixture_builder_and_kernel_receipt_hashes():
    from assembled_legacy_proofs import validate_published_acoustic

    root = Path(__file__).resolve().parents[1]
    manifest = ORACLE_DIR / "oracle-sha256sums.json"
    # The published 170-entry manifest and all native fixture/tool pins stay
    # immutable. Attribution below is a finite source overlay, not a new map.
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == '46b58132a246c3f855a545239d1492d7a9f42f3f4ec067c21470ecf91fe4fe90'
    pins = json.loads(manifest.read_text())
    assert len(pins) >= 160
    assert "tools/smallstep_wrf471_oracle/build.py" in pins
    assert "gpuwm/core/kernels/acoustic.cu" in pins
    original_source_pins = {'gpuwm/core/kernels/acoustic.cu': 'bc2ce86bd35e1017868e8711465ec38a32890eea0dd08a64cc837bcd0790262d',
     'gpuwm/verify/smallstep_bookkeeping_oracle.py': 'cacc6417dd93e6ffdea0bb5ecfd98f6ce81e7b49d28613068b6e43aa3dba30a3',
     'gpuwm/verify/smallstep_vertical_oracle.py': 'dbf2d377f72588e7f0ef27a69ab6aa03a9c9f03751fbe411964278bd0e29a0f4',
     'tests/test_smallstep_vertical_wrf471_parity.py': '545bdccfe08b831090f0e8140850f6adb79f61b3bbf184c4dc881bceffd5930c',
     'tools/smallstep_wrf471_oracle/vertical_workspace.py': '96e7a8d967e11d9c6b14e59b4a21f9e14ca115fed348245ac2b35e99f5cd69b0',
     'tests/test_smallstep_oracle_contract.py': '6dbadf1c006a61b186567c1ab8652af1953425833fcfc250af9ae3c27a7e2d40'}
    assert {name: pins[name] for name in original_source_pins} == original_source_pins

    # 9190ec5f2 repairs observer parameters on conditional signatures without
    # changing the kernel expressions. The vertical test adds CPU contracts
    # for the actual witness/compiler boundary and selected default route.
    # Neither metadata overlay admits another source byte.
    source_byte_overlay = {'gpuwm/verify/smallstep_bookkeeping_oracle.py': '7b7301d90a85b1432de7ce00432060d1656f5cf5a9c6dffc773510ca12c13f6e',
     'tests/test_smallstep_vertical_wrf471_parity.py': 'd58cc590338977134514a09ce32f50db37f86b2c596c0fdb31fb2e5bd581e522'}

    # These three executable kernel/witness sources are accepted ONLY by the
    # sealed real f66 default replay: all 132 published arrays / 305632 words
    # retain their original hashes, with the complete current raw source,
    # compiler options and certified card identity independently checked.
    proof = validate_published_acoustic(root)
    numeric_sources = ('gpuwm/core/kernels/acoustic.cu', 'gpuwm/verify/smallstep_vertical_oracle.py', 'tools/smallstep_wrf471_oracle/vertical_workspace.py')
    for name in numeric_sources:
        source_byte_overlay[name] = proof["source_inputs"][name]["sha256"]
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == source_byte_overlay[name], name

    # A current self-hash would be circular. Reconstruct this exact historical
    # final function under the unchanged file prefix, then check its ORIGINAL
    # manifest hash. All earlier layout/schema/word assertions remain sealed;
    # the new proof validator has its own negative/tamper contracts.
    historical_function = 'def test_smallstep_fixture_builder_and_kernel_receipt_hashes():\n    root=Path(__file__).resolve().parents[1]\n    pins=json.loads((ORACLE_DIR/"oracle-sha256sums.json").read_text())\n    assert len(pins)>=160\n    assert "tools/smallstep_wrf471_oracle/build.py" in pins\n    assert "gpuwm/core/kernels/acoustic.cu" in pins\n    for name,digest in pins.items():\n        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n'
    marker = "def test_smallstep_fixture_builder_and_kernel_receipt_hashes():\n"
    current_text = Path(__file__).read_text(encoding="utf-8")
    historical_self = (current_text.split(marker, 1)[0] + historical_function).encode("utf-8")
    self_name = "tests/test_smallstep_oracle_contract.py"
    assert hashlib.sha256(historical_self).hexdigest() == pins[self_name]
    assert set(source_byte_overlay) == set(original_source_pins) - {self_name}
    for name, digest in pins.items():
        if name == self_name:
            continue  # Its immutable historical bytes were checked above.
        expected = source_byte_overlay.get(name, digest)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name


def test_published_acoustic_proof_refuses_changed_words_options_and_sources(monkeypatch):
    import copy
    import pytest
    import assembled_legacy_proofs as proofs

    root = Path(__file__).resolve().parents[1]
    original = proofs._load(root, "acoustic-published.json")
    for mutation in ("words", "options", "source"):
        proof = copy.deepcopy(original)
        if mutation == "words":
            proof["changed_published_words"] = 1
        elif mutation == "options":
            proof["capture_identity"]["module_options"]["acoustic"].append("--ftz=false")
        else:
            proof["source_inputs"]["gpuwm/core/kernels/acoustic.cu"]["sha256"] = "0" * 64
        monkeypatch.setattr(proofs, "_load", lambda root, name: proof)
        with pytest.raises(AssertionError):
            proofs.validate_published_acoustic(root)
