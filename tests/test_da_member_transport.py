"""Independent bit, identity and tamper checks for member transport on CPU."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da import member_transport as transport
from gpuwm.da import member_wave as wave
from tools.da_member_leg import MemberLegContext, MemberLegResult
from tools.da_ensemble_state import EnsembleIdentity


@dataclass
class Perturbation:
    amplitude: float = 0.25
    fields: tuple = ("u", "v")


@dataclass
class HotStart:
    active: bool = False


def make_context(tmp_path):
    source = tmp_path/"proof.json"
    source.write_text('{"source":"public synthetic metadata"}')
    identity = EnsembleIdentity(members=2,nx=3,ny=4,nz=5,dt_s=15.0,mp_physics=6,
                                physics_profile="synthetic-public",prepared_content_sha256="0"*64)
    args = argparse.Namespace(prepared_root=tmp_path,authority_dir=None,obs=[],grid_wrfout=[],
                              radar_tten=False,members=2,physics_profile=identity.physics_profile,
                              prepared_content_sha256=identity.prepared_content_sha256)
    cfg=SimpleNamespace(nx=3,ny=4,nz=5,dt=15.0,mp_physics=6)
    inputs=SimpleNamespace(proof_path=source,cache_identity={"case_sha256":"0"*64},
                           experiment=SimpleNamespace(root=SimpleNamespace(run=cfg)))
    ctx=MemberLegContext(args=args,inputs=inputs,identity=identity,cfg_perturb=Perturbation(),hot_cfg=HotStart(),
                         leg=0,absolute_leg_number=0,t_start=0.0,t_end=60.0,stage_root=tmp_path/"stage",out=tmp_path/"out",
                         setup_arrays={"c1h":np.arange(5,dtype=np.float64)},thb_host=np.array(300.0,dtype=np.float32))
    return ctx


def request(tmp_path,ctx):
    common=transport.write_common(tmp_path/"common.json",ctx)
    job=transport.write_request(tmp_path/"job",common,ctx,0)
    return Path(job["argv"][job["argv"].index("--request")+1]),job


def patch_bindings(monkeypatch,ctx):
    import tools.da_member_leg as owner
    monkeypatch.setattr(owner,"build_member_context",lambda args:SimpleNamespace(
        args=args,inputs=ctx.inputs,cfg_perturb=ctx.cfg_perturb,hot_cfg=ctx.hot_cfg))


def change_json(path,change):
    doc=json.loads(path.read_text());change(doc);path.write_text(json.dumps(doc));return doc


def test_array_transport_preserves_dtype_shape_and_actual_bytes(tmp_path):
    bits=np.array([0x00000000,0x80000000,0x7fc01234,0x7f800000],dtype=np.uint32)
    fields={"f32":bits.view(np.float32),"big_endian":np.array([1,-2],dtype=">i4"),
            "noncontiguous":np.arange(24,dtype=np.float64).reshape(4,6)[:,::2],
            "scalar":np.array(-0.0,dtype=np.float32),"complex":np.array([1+2j],np.complex128),
            "boolean":np.array([[True,False]],bool)}
    record=transport.array_file(tmp_path/"arrays.npz",fields)
    loaded=transport.arrays(record)
    for key,expected in fields.items():
        assert loaded[key].dtype == expected.dtype
        assert loaded[key].shape == expected.shape
        assert loaded[key].tobytes(order="C") == expected.tobytes(order="C")


def test_object_arrays_refused_before_file_creation(tmp_path):
    path=tmp_path/"object.npz"
    with pytest.raises(ValueError,match="object|pickle"):
        transport.array_file(path,{"state":np.array([{"not":"numeric"}],dtype=object)})
    assert not path.exists()


def test_pickle_and_changed_payload_are_refused(tmp_path):
    path=tmp_path/"pickle.npz"
    path.write_bytes(pickle.dumps({"x":np.ones(1)}))
    with pytest.raises(ValueError,match="pickle|pickled"):
        transport.arrays(transport.file_record(path))
    plain=tmp_path/"plain.npz"
    record=transport.array_file(plain,{"state":np.ones(3)})
    plain.write_bytes(plain.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="changed"):
        transport.arrays(record)


def test_npz_object_payload_cannot_pass_forecast_barrier(tmp_path):
    path=tmp_path/"objects.npz"
    np.savez(path,x=np.array([{"x":1}],object))
    with pytest.raises(wave.MemberWaveFailed,match="object|pickle"):
        wave._plain_array_archive(path)


def test_npy_v3_utf8_plain_array_headers_supported(tmp_path):
    path=tmp_path/"utf8.npz"
    with pytest.warns(UserWarning):
        np.savez(path,x=np.array([(1.0,)],dtype=[("temperature_\u03bc","<f8")]))
    wave._plain_array_archive(path)
    assert transport.arrays(transport.file_record(path))["x"].dtype.names == ("temperature_\u03bc",)


def test_public_parser_is_pure_and_metadata_round_trip_rebinds(tmp_path,monkeypatch):
    from tools.da_cycle_prepared import build_parser
    parser=build_parser()
    assert parser is not None
    ctx=make_context(tmp_path);patch_bindings(monkeypatch,ctx)
    path,job=request(tmp_path,ctx)
    loaded,doc=transport.read_context(path,expected_hash=job["request_hash"])
    assert loaded.identity == ctx.identity
    assert loaded.setup_arrays["c1h"].tobytes() == ctx.setup_arrays["c1h"].tobytes()
    assert loaded.thb_host.shape == () and loaded.thb_host.tobytes() == ctx.thb_host.tobytes()
    assert "--request-sha256" in job["argv"]
    assert doc["member"] == 0


@pytest.mark.parametrize("kind",["request","common","source","identity","perturbation","prepared"])
def test_request_common_source_and_rebinding_tamper_are_refused(tmp_path,monkeypatch,kind):
    ctx=make_context(tmp_path);patch_bindings(monkeypatch,ctx)
    path,job=request(tmp_path,ctx)
    expected=job["request_hash"]
    if kind=="request":
        change_json(path,lambda d:d.update(t_end=120.0))
    elif kind=="common":
        (tmp_path/"common.json").write_text("changed")
    elif kind=="source":
        ctx.inputs.proof_path.write_text("changed source")
    else:
        key={"identity":"identity","perturbation":"perturbation","prepared":"prepared_cache_identity"}[kind]
        change_json(tmp_path/"common.json",lambda d:d[key].update({"members":3} if kind=="identity" else {"amplitude":99} if kind=="perturbation" else {"case_sha256":"f"*64}))
        # Re-seal the changed common file to test independent public binding,
        # rather than having the outer byte checksum stop this case first.
        change_json(path,lambda d:d.update(common=transport.file_record(tmp_path/"common.json")))
        expected=wave.sha256(path)
    with pytest.raises(ValueError):
        transport.read_context(path,expected_hash=expected)


def test_alternate_callback_field_refused_even_with_matching_request_digest(tmp_path,monkeypatch):
    ctx=make_context(tmp_path);patch_bindings(monkeypatch,ctx)
    path,_=request(tmp_path,ctx)
    change_json(path,lambda d:d.update(wire_override="arbitrary.module"))
    with pytest.raises(ValueError,match="callbacks"):
        transport.read_context(path,expected_hash=wave.sha256(path))


def test_observed_file_is_sealed_and_matches_the_controller_leg(tmp_path,monkeypatch):
    ctx=make_context(tmp_path);patch_bindings(monkeypatch,ctx)
    observed,grid=tmp_path/"obs.nc",tmp_path/"grid.nc"
    observed.write_bytes(b"bound radar observation");grid.write_bytes(b"bound grid")
    ctx.args.obs=[observed];ctx.args.grid_wrfout=[grid];ctx.obs_path=observed
    ctx.document={"schema":"synthetic-radar","variables":{"z_obs":np.ones((1,2,2),np.float32)}}
    path,job=request(tmp_path,ctx)
    observed.write_bytes(b"changed radar observation")
    with pytest.raises(ValueError,match="artifact changed"):
        transport.read_context(path,expected_hash=job["request_hash"])


def test_worker_observation_must_match_already_read_parent_document(tmp_path,monkeypatch):
    from gpuwm.da import obs_radar
    from gpuwm.obs.target_grid import TargetGrid
    ctx=make_context(tmp_path);patch_bindings(monkeypatch,ctx)
    observed,grid=tmp_path/"obs.nc",tmp_path/"grid.nc"
    observed.write_bytes(b"sealed-radar");grid.write_bytes(b"sealed-grid")
    ctx.args.obs=[observed];ctx.args.grid_wrfout=[grid];ctx.obs_path=observed
    ctx.document={"schema":"synthetic-radar","variables":{"z_obs":np.ones((1,2,2),np.float32)}}
    path,job=request(tmp_path,ctx)
    monkeypatch.setattr(TargetGrid,"from_wrfout",classmethod(lambda cls,path:object()))
    monkeypatch.setattr(obs_radar,"read_document",lambda *a,**k:{"schema":"synthetic-radar","variables":{"z_obs":np.full((1,2,2),2.0,np.float32)}})
    # The worker reads the controller's sealed arrays, not the file; arrays
    # sealed from a different document are refused against the identity.
    sealed=json.loads(path.read_text())["observation_arrays"]
    np.savez(sealed["path"]+".tmp.npz",z_obs=np.full((1,2,2),2.0,np.float32))
    Path(sealed["path"]+".tmp.npz").replace(sealed["path"])
    change_json(path,lambda d:d["observation_arrays"].update(transport.file_record(sealed["path"])))
    job["request_hash"]=wave.sha256(path)
    with pytest.raises(ValueError,match="already-read document"):
        transport.read_context(path,expected_hash=job["request_hash"])


def test_complete_tree_result_round_trip_owns_all_host_products(tmp_path):
    from test_da_member_wave import native_restart
    ctx=make_context(tmp_path)
    root=tmp_path/"job";root.mkdir()
    restart=root/"gpuwmrst_d01_end.npz"
    native_restart(restart,0)
    from gpuwm.da.radar_assimilation import read_checkpoint_state
    mirror=read_checkpoint_state(restart)
    result=MemberLegResult(restart=restart,snapshot=mirror,
        H_Z=np.arange(6,dtype=np.float64).reshape(2,3),H_surface={"t2":np.array(300.0,np.float32)},
        hot_pending={"qv":np.zeros(3,np.float32)},setup_arrays={"c1h":np.ones(2,np.float64)},
        thb_host=np.array(300,np.float32),nest_birth=None,record={"elapsed_seconds":60.0},
        consume_restart=None,pending_consumed=False)
    doc=transport.write_result(root/"result.json",ctx,0,result,"a"*64)
    loaded=transport.load_result(doc)
    # The whole-state mirror stays in the worker; its restart carries it.
    assert loaded.snapshot is None
    with np.load(doc["arrays"]["path"]) as payload:
        assert not any(key.startswith("snapshot/") for key in payload.files)
    for before,after in [(result.H_Z,loaded.H_Z),
                         (result.H_surface["t2"],loaded.H_surface["t2"]),(result.thb_host,loaded.thb_host)]:
        assert before.dtype==after.dtype and before.shape==after.shape and before.tobytes()==after.tobytes()
    assert loaded.restart==restart and loaded.record==result.record


def test_leg_requests_written_together_match_one_by_one(tmp_path):
    """write_requests seals the leg's observation once and writes members
    concurrently; every request it writes is the request write_request
    writes alone, byte for byte, apart from the job directory it names."""
    import dataclasses
    from test_da_member_wave import native_restart

    observed, grid = tmp_path/"obs.nc", tmp_path/"grid.nc"
    observed.write_bytes(b"bound radar observation"); grid.write_bytes(b"bound grid")
    base = make_context(tmp_path)
    base.args.obs = [observed]; base.args.grid_wrfout = [grid]
    document = {"schema": "synthetic-radar",
                "variables": {"z_obs": np.ones((1, 2, 2), np.float32)}}
    contexts = {}
    for member in range(2):
        restart = tmp_path/f"r{member}"/"gpuwmrst_d01_end.npz"
        restart.parent.mkdir()
        native_restart(restart, member)
        contexts[member] = dataclasses.replace(
            base, obs_path=observed, document=document, restart=restart,
            pending={"qv": np.full((5, 4, 3), 1e-4 * (member + 1))})
    common = transport.write_common(tmp_path/"common.json", base)
    together = transport.write_requests(tmp_path/"together", common, contexts, [0, 1])
    alone = [transport.write_request(tmp_path/"alone"/f"m{m:03d}", common, contexts[m], m)
             for m in (0, 1)]
    for left, right in zip(together, alone, strict=True):
        lreq = json.loads(Path(left["argv"][left["argv"].index("--request")+1]).read_text())
        rreq = json.loads(Path(right["argv"][right["argv"].index("--request")+1]).read_text())
        for doc in (lreq, rreq):
            for key in ("output_root",):
                doc.pop(key)
            for key in ("pending", "setup", "observation_arrays"):
                if doc.get(key):
                    doc[key].pop("path")
        assert lreq == rreq
        assert left["member"] == right["member"]
        assert left["expected_trajectory_fingerprint"] == right["expected_trajectory_fingerprint"]


def test_parallel_digests_are_the_serial_digests(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path/f"f{index}.bin"
        path.write_bytes(np.random.default_rng(index).bytes(3_000_000 + index))
        paths.append(path)
    digests = wave.sha256_many(paths + [tmp_path/"missing.bin"], threads=4)
    assert set(digests) == {str(p) for p in paths}
    for path in paths:
        assert digests[str(path)] == wave.sha256(path)


def test_a_restart_that_differs_from_its_mirror_is_refused(tmp_path):
    """The packed analysis reads the leg-end restart, not the mirror; a
    worker whose restart and mirror disagree must not publish."""
    from test_da_member_wave import native_restart
    from gpuwm.da.radar_assimilation import read_checkpoint_state
    root=tmp_path/"job";root.mkdir()
    restart=root/"gpuwmrst_d01_end.npz"
    native_restart(restart,0)
    mirror=read_checkpoint_state(restart)
    assert mirror
    assert transport.verify_restart_matches_mirror(restart,mirror)==sum(
        np.asarray(v).nbytes for v in mirror.values())
    key=sorted(mirror)[0]
    moved=dict(mirror); moved[key]=np.array(mirror[key],copy=True)
    moved[key].reshape(-1)[0]+=1
    with pytest.raises(transport.RestartMirrorMismatch,match=key):
        transport.verify_restart_matches_mirror(restart,moved)
    with pytest.raises(transport.RestartMirrorMismatch,match="no state/extra"):
        transport.verify_restart_matches_mirror(
            restart,dict(mirror,extra=np.zeros(2,np.float32)))
    assert transport.verify_restart_matches_mirror(restart,None)==0


def test_a_request_linking_the_generations_pending_file_is_the_same_request(tmp_path):
    """The next leg's request links the pending file the recovery generation
    wrote instead of writing the same arrays again; the worker reads the
    same arrays, and futures sealed on the caller's pool are the same jobs."""
    import dataclasses
    from concurrent.futures import ThreadPoolExecutor
    from test_da_member_wave import native_restart
    from tools import da_ensemble_state as ens_state

    base = make_context(tmp_path)
    contexts = {}
    for member in range(2):
        restart = tmp_path/f"r{member}"/"gpuwmrst_d01_end.npz"
        restart.parent.mkdir()
        native_restart(restart, member)
        contexts[member] = dataclasses.replace(
            base, restart=restart, leg=1, absolute_leg_number=1,
            t_start=60.0, t_end=120.0,
            pending={"qv": np.full((5, 4, 3), 1e-4 * (member + 1)),
                     "u": np.linspace(-1, 1, 64).reshape(4, 4, 4)})
    generation = tmp_path/"gen"
    files = {}
    for member in range(2):
        path = ens_state.pending_path(generation, member)
        ens_state._atomic_savez(path, dict(contexts[member].pending))
        files[member] = path
    common = transport.write_common(tmp_path/"common.json", base)
    own = transport.write_requests(tmp_path/"own", common, contexts, [0, 1])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = transport.write_requests(
            tmp_path/"linked", common, contexts, [0, 1],
            pending_files=files.__getitem__, pool=pool)
        linked = [future.result() for future in futures]
    for left, right, member in zip(own, linked, (0, 1)):
        lreq = json.loads(Path(left["argv"][left["argv"].index("--request")+1]).read_text())
        rreq = json.loads(Path(right["argv"][right["argv"].index("--request")+1]).read_text())
        assert Path(rreq["pending"]["path"]).samefile(files[member])
        a, b = transport.arrays(lreq["pending"]), transport.arrays(rreq["pending"])
        assert sorted(a) == sorted(b)
        for key in a:
            assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
        for doc in (lreq, rreq):
            doc.pop("output_root")
            for key in ("pending", "setup", "observation_arrays"):
                if doc.get(key):
                    doc[key].pop("path")
        assert lreq == rreq


def test_the_worker_reads_the_controllers_sealed_arrays_not_the_radar_file(tmp_path,monkeypatch):
    """A worker no longer decodes the radar file: it reads the arrays the
    controller sealed once for the leg, checked against the identity of the
    controller's own document, and refuses nothing it should accept."""
    from gpuwm.da import obs_radar
    ctx=make_context(tmp_path);patch_bindings(monkeypatch,ctx)
    observed,grid=tmp_path/"obs.nc",tmp_path/"grid.nc"
    observed.write_bytes(b"sealed-radar");grid.write_bytes(b"sealed-grid")
    ctx.args.obs=[observed];ctx.args.grid_wrfout=[grid];ctx.obs_path=observed
    z_obs=np.arange(4,dtype=np.float32).reshape(1,2,2)
    z_mask=np.array([[[1,0],[0,1]]],np.int8)
    ctx.document={"schema":"synthetic-radar","dims":{"level":1},
                  "variables":{"z_obs":z_obs,"z_mask":z_mask,"vr_obs":np.zeros((3,1,2,2),np.float32)}}
    path,job=request(tmp_path,ctx)
    def no_decode(*a,**k):
        raise AssertionError("the worker decoded the radar file")
    monkeypatch.setattr(obs_radar,"read_document",no_decode)
    context,_=transport.read_context(path,expected_hash=job["request_hash"])
    assert sorted(context.document["variables"])==["z_mask","z_obs"]
    for key,value in (("z_obs",z_obs),("z_mask",z_mask)):
        got=context.document["variables"][key]
        assert got.dtype==value.dtype and got.tobytes()==value.tobytes()
    assert context.document["schema"]=="synthetic-radar"
