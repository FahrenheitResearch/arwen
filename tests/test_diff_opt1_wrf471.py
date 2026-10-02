"""Device words from compiled WRF and unchanged diff_opt=2 baseline."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
DATA=Path(__file__).parent/"data/wrf471_diff_opt1"


def test_merged_coordinate_receipt_is_sealed_to_native_and_production_source():
    import hashlib
    from gpuwm.core.kernels import module_source
    receipt=json.loads((DATA/"merged-gpu.json").read_text())
    proof=json.loads((DATA/"merged-attribution.json").read_text())
    assert receipt["archive_sha256"]==hashlib.sha256((DATA/"merged-gpu.npz").read_bytes()).hexdigest()
    assert receipt["native_archive_sha256"]==hashlib.sha256((DATA/"wrf471.npz").read_bytes()).hexdigest()
    assert proof["production_archive_sha256"]==receipt["archive_sha256"]
    assert proof["prior_archive_sha256"]==hashlib.sha256((DATA/"km2-gpu.npz").read_bytes()).hexdigest()
    assert len(receipt["cases"])==88
    assert sum(len(row["fields"]) for row in receipt["cases"])==272
    for name,digest in receipt["module_source_sha256"].items():
        assert hashlib.sha256(module_source(name).encode()).hexdigest()==digest

@pytest.mark.gpu
@requires_gpu
def test_coordinate_flux_every_compiled_wrf_output_word():
    import cupy as cp
    from gpuwm.core.dycore import launch_coordinate_horizontal
    cases=json.loads((DATA/"wrf471.json").read_text())["cases"]
    with np.load(DATA/"wrf471.npz") as values:
        for case in cases:
            if case["family"]!="horizontal":continue
            get=lambda key: values[case["name"]+"_"+key]
            dev=lambda key: cp.asarray(get(key))
            result=dev("seed")
            launch_coordinate_horizontal(dev("field"),dev("km"),dev("mu"),dev("c1"),dev("c2"),
                dev("mt"),dev("mfu"),dev("mfv"),case["dx"],case["dy"],result,
                stagger=("","x","y","z")[case["stag"]],
                boundary_x=case["bx"],boundary_y=case["by"],
                theta_initial=dev("base") if case["perturb"] else None)
            np.testing.assert_array_equal(cp.asnumpy(result).view("u4"),get("expected").view("u4"),err_msg=case["name"])


@pytest.mark.gpu
@requires_gpu
def test_coordinate_km4_every_compiled_wrf_output_word():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    cases=json.loads((DATA/"wrf471.json").read_text())["cases"]
    with np.load(DATA/"wrf471.npz") as values:
        for case in cases:
            if case["family"]!="km4":continue
            get=lambda key: values[case["name"]+"_"+key]
            dev=lambda key: cp.asarray(get(key))
            nz,ny,nx=get("d11").shape
            km=cp.zeros((nz,ny,nx),"f4");kh=cp.zeros_like(km)
            get_kernel("diff_opt1","wrf_diff_opt1_km4")(
                ((nx+127)//128,ny,nz),(128,1,1),
                (dev("d11"),dev("d22"),dev("d12"),dev("mt"),np.float32(case["dx"]),
                 np.float32(case["dy"]),np.float32(.25),np.float32(1./3.),km,kh,
                 np.int32(nz),np.int32(ny),np.int32(nx),np.int32(case["bx"]),np.int32(case["by"])))
            for field,value in (("km",km),("kh",kh)):
                np.testing.assert_array_equal(cp.asnumpy(value).view("u4"),get("expected_"+field).view("u4"),err_msg=case["name"]+field)


@pytest.mark.gpu
@requires_gpu
def test_coordinate_km2_compiled_wrf_horizontal_coefficients_and_tke_words():
    import cupy as cp
    from tools.wrf_diffopt1_oracle.model_case import km2_case
    from tools.wrf_diffopt1_oracle.capture import stats
    from gpuwm.core.dycore import launch_wrf_tke_km
    receipt=json.loads((DATA/"merged-gpu.json").read_text())
    measured={row["name"]:row["fields"] for row in receipt["cases"]}
    with np.load(DATA/"wrf471.npz") as values,np.load(DATA/"merged-gpu.npz") as pinned:
        for case in json.loads((DATA/"wrf471.json").read_text())["cases"]:
            if case["family"]!="km2":continue
            cfg,state=km2_case(case,values)
            km=state.scratch(state.p.shape,"smag_km");kh=state.scratch(state.p.shape,"smag_kh")
            rtke=state.scratch(state.p.shape,"smag_rtke");rtke[:]=np.float32(.125)
            launch_wrf_tke_km(state,cfg,km,kh,time_t=False)
            bn2=state._scratch["diff6_x"].reshape(-1)[:state.p.size].reshape(state.p.shape)
            for field,value in (("km",km),("kh",kh),("tke",state.tke),("bn2",bn2),
                                ("kmv",state._scratch["smag_kmv"]),
                                ("khv",state._scratch["smag_khv"])):
                actual=cp.asnumpy(value);key=case["name"]+"_"+field
                np.testing.assert_array_equal(actual.view("u4"),pinned[key].view("u4"),err_msg=key)
                assert stats(actual,values[case["name"]+"_expected_"+field])==measured[case["name"]][field]
                if field in ("km","kh","tke"):assert measured[case["name"]][field]["max_ulp"]<=5
                if field in ("bn2","tke"):assert measured[case["name"]][field]["different"]==0
            assert bool(cp.all(rtke==np.float32(.125)))


@pytest.mark.gpu
@requires_gpu
def test_coordinate_deformation_matches_compiled_word_receipt():
    import cupy as cp
    from tools.wrf_diffopt1_oracle.model_case import km2_case
    from tools.wrf_diffopt1_oracle.capture import stats
    from gpuwm.core.dycore import launch_wrf_smag2d_km
    measured={row["name"]:row["fields"] for row in json.loads((DATA/"merged-gpu.json").read_text())["cases"]}
    with np.load(DATA/"wrf471.npz") as values,np.load(DATA/"merged-gpu.npz") as pinned:
        for case in json.loads((DATA/"wrf471.json").read_text())["cases"]:
            if case["family"]!="deform":continue
            cfg,state=km2_case(case,values)
            km=state.scratch(state.p.shape,"smag_km");kh=state.scratch(state.p.shape,"smag_kh")
            tensors=launch_wrf_smag2d_km(state,cfg,km,kh,time_t=False)
            for field,value in zip(("d11","d22","d12"),tensors):
                actual=cp.asnumpy(value)
                np.testing.assert_array_equal(actual.view("u4"),pinned[case["name"]+"_"+field].view("u4"))
                assert stats(actual,values[case["name"]+"_expected_"+field])==measured[case["name"]][field]
                expected=values[case["name"]+"_expected_"+field]
                if field in ("d11","d22"):
                    np.testing.assert_array_equal(actual.view("u4"),expected.view("u4"))
                else:
                    assert float(np.max(np.abs(actual-expected)))<=2.**-31
                    # cal_deform_and_div copies the evaluated tensor at physical
                    # faces. Preserve the donor rule independently of the pin.
                    if case["bx"]:
                        np.testing.assert_array_equal(actual[:,:,0].view("u4"),actual[:,:,1].view("u4"))
                    if case["by"]:
                        np.testing.assert_array_equal(actual[:,0,:].view("u4"),actual[:,1,:].view("u4"))


@pytest.mark.gpu
@requires_gpu
def test_merged_coordinate_moved_words_are_only_the_tensor_donor_fix(tmp_path):
    import subprocess
    import sys
    capture=Path(__file__).parents[1]/"tools/wrf_diffopt1_oracle/capture.py"
    output=tmp_path/"donor-reverted"
    # Separate process keeps this arithmetic diagnostic out of the production
    # kernel loader and its cached function wrappers for the other tests.
    subprocess.run([sys.executable,str(capture),str(DATA),str(output),"--mode","km2",
                    "--revert-deformation-donor"],check=True)
    with np.load(DATA/"km2-gpu.npz") as prior,np.load(output.with_suffix(".npz")) as control:
        assert set(control.files)==set(prior.files)
        for key in prior.files:
            np.testing.assert_array_equal(control[key].view("u4"),prior[key].view("u4"),err_msg=key)
    proof=json.loads((DATA/"merged-attribution.json").read_text())
    with np.load(DATA/"km2-gpu.npz") as prior,np.load(DATA/"merged-gpu.npz") as merged:
        observed={}
        for key in prior.files:
            mask=prior[key].view("u4")!=merged[key].view("u4")
            for index in np.argwhere(mask):
                location=tuple(int(value) for value in index)
                observed[(key,location)]=(int(prior[key].view("u4")[location]),int(merged[key].view("u4")[location]))
        recorded={(row["receipt"]+"_"+row["field"],tuple(row["index"])):
                  (int(row["prior_word"],16),int(row["merged_word"],16))
                  for row in proof["moved_word_attribution"]}
        assert observed==recorded
        assert len(observed)==proof["moved_words"]==492


@pytest.mark.gpu
@requires_gpu
def test_metric_diffusion_matches_merged_oracle_fix_baseline():
    """The merged native-oracle fixes have an exact, separately named pin.

    The original a417 words remain intact in diff2-baseline.npz. Every moved
    word is attributed by controlled default-source transitions in the merge
    receipt rather than accepted by a larger numeric tolerance.
    """
    from tools.wrf_diffopt1_oracle.model_case import configuration,model_state,word_arrays
    from gpuwm.core.dycore import step
    with np.load(DATA/"diff2-merged-baseline.npz") as expected:
        for km in (2,4):
            for boundary in (False,True):
                for moist in (False,True):
                    # Preserve the sealed pre-CQ trajectory and legacy checkpoint physics.
                    cfg=replace(configuration(km=km,diff=2,mix=True,boundary=boundary,moist=moist),moist_cq=False)
                    state=model_state(cfg)
                    for _ in range(3):step(state,cfg)
                    name=f"diff2_k{km}_b{int(boundary)}_m{int(moist)}"
                    for field,actual in word_arrays(state).items():
                        np.testing.assert_array_equal(actual.view("u4"),expected[name+"_"+field].view("u4"),err_msg=name+field)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("km",[2,4])
@pytest.mark.parametrize("mix",[False,True])
def test_coordinate_restart_retains_original_theta_and_exact_trajectory(tmp_path,km,mix):
    import cupy as cp
    from tools.wrf_diffopt1_oracle.model_case import configuration,model_state,word_arrays
    from gpuwm.core.dycore import step,initialize_coordinate_reference
    from gpuwm.io.restart import write_restart,restore_restart
    cfg=configuration(km=km,mix=mix)
    straight=model_state(cfg);split=model_state(cfg)
    initialize_coordinate_reference(straight,cfg)
    original=cp.asnumpy(straight._scratch["diff1_theta_initial"]).copy()
    for _ in range(4):step(straight,cfg)
    for _ in range(2):step(split,cfg)
    split.elapsed_seconds=2*cfg.dt
    path=write_restart(tmp_path/"split.npz",split,cfg)
    with np.load(path) as stored:
        np.testing.assert_array_equal(stored["scratch/diff1_theta_initial"].view("u4"),original.view("u4"))
    resumed=model_state(cfg)
    restore_restart(path,resumed,cfg)
    for _ in range(2):step(resumed,cfg)
    np.testing.assert_array_equal(cp.asnumpy(resumed._scratch["diff1_theta_initial"]).view("u4"),original.view("u4"))
    assert not np.array_equal(cp.asnumpy(resumed.thp),cp.asnumpy(model_state(cfg).thp))
    for field,actual in word_arrays(straight).items():
        np.testing.assert_array_equal(actual.view("u4"),word_arrays(resumed)[field].view("u4"),err_msg=field)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("km",[2,4])
def test_legacy_metric_checkpoint_resumes_exact_trajectory(km):
    from dataclasses import replace
    from tools.wrf_diffopt1_oracle.model_case import configuration,model_state,word_arrays
    from gpuwm.core.dycore import step
    from gpuwm.io.restart import restore_restart,read_restart_header,RestartMismatchError
    from tools.wrf_diffopt1_oracle.merged_metric_capture import checkpoint_payload_state
    # Preserve the sealed pre-CQ trajectory and legacy checkpoint physics.
    cfg=replace(configuration(km=km,diff=2,mix=True),moist_cq=False)
    path=DATA/f"diff2-legacy-k{km}.npz"
    assert "diff_opt" not in read_restart_header(path)["config"]
    assert "mix_full_fields" not in read_restart_header(path)["config"]
    # Independent direct payload seeding advances the genuine old state with
    # the merged operator. It does not call the compatibility restore path.
    straight=checkpoint_payload_state(path,cfg)
    resumed=model_state(cfg);restore_restart(path,resumed,cfg)
    for _ in range(2):
        step(resumed,cfg)
        step(straight,cfg)
    straight_words=word_arrays(straight)
    with np.load(DATA/"diff2-merged-legacy.npz") as expected:
        for field,actual in word_arrays(resumed).items():
            np.testing.assert_array_equal(actual.view("u4"),expected[f"k{km}_"+field].view("u4"),err_msg=field)
            np.testing.assert_array_equal(actual.view("u4"),straight_words[field].view("u4"),err_msg=field)
    with pytest.raises(RestartMismatchError,match="mix_full_fields"):
        restore_restart(path,model_state(cfg),replace(cfg,mix_full_fields=False))


def test_metric_merge_receipt_preserves_original_words_and_seals_attribution():
    import hashlib
    from tools.wrf_diffopt1_oracle.capture import stats
    receipt=json.loads((DATA/"diff2-merged-attribution.json").read_text())
    digest=lambda name:hashlib.sha256((DATA/name).read_bytes()).hexdigest()
    assert receipt["original_baseline_sha256"]==digest("diff2-baseline.npz")
    assert receipt["original_legacy_sha256"]==digest("diff2-legacy.npz")
    assert receipt["merged_baseline_sha256"]==digest("diff2-merged-baseline.npz")
    assert receipt["merged_legacy_sha256"]==digest("diff2-merged-legacy.npz")
    for km in (2,4):
        assert receipt["original_checkpoints"][f"k{km}"]==digest(f"diff2-legacy-k{km}.npz")
    moved=sum(row["merged_vs_original"]["different"] for row in receipt["fields"])
    assert moved==receipt["moved_words"]
    assert receipt["original_commit"]=="a4177ebbf342252405df3f6ed8309704daee94fc"
    assert {row["commit"] for row in receipt["control_groups"]}==set(receipt["native_provenance"])
    assert len(receipt["word_attribution_sha256"])==64
    assert len(receipt["controls"])==5
    assert all(row["runtime_sources"] for row in receipt["controls"])
    with np.load(DATA/"diff2-baseline.npz") as old,np.load(DATA/"diff2-merged-baseline.npz") as current:
        measured={key:stats(current[key],old[key]) for key in old.files}
    with np.load(DATA/"diff2-legacy.npz") as old,np.load(DATA/"diff2-merged-legacy.npz") as current:
        measured.update({"legacy_"+key:stats(current[key],old[key]) for key in old.files})
    assert {row["name"]:row["merged_vs_original"] for row in receipt["fields"]}==measured
