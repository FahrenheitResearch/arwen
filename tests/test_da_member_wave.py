"""Owned CPU processes verify packed scheduling and the immutable DA barrier."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from gpuwm.da import member_wave as wave


def uuid(index=0):
    return f"GPU-{index:08x}-0000-0000-0000-000000000000"


def jobs(tmp_path, count=4):
    return [{"member": member, "argv": [sys.executable, "-c", "pass"],
             "result_path": str(tmp_path/f"m{member:03d}"/"result.json"),
             "output_root": str(tmp_path/f"m{member:03d}"),
             "request_hash": f"{member:064x}", "t_start": 0.0, "t_end": 60.0,
             "expected_domain_ids": [1], "ensemble_members": count}
            for member in range(count)]


def native_restart(path, member, *, domains=(1,), seconds=60.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"format_version": 6, "domain_ids": list(domains), "elapsed_ticks": int(seconds),
              "tick_den": 1, "elapsed_seconds": seconds, "step_count": int(seconds//15),
              "domain_start_ticks": 0, "domain_lifecycle": "ACTIVE", "dtbc_fp32_bits": 0,
              "experiment_fingerprint_components": {"trajectory": str(member)}, "is_stub": True}
    np.savez(path, __gpuwm_restart_header__=np.frombuffer(json.dumps(header).encode(), np.uint8),
             **{"state/theta": np.array([300.0], np.float32)})
    return header


def file_record(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": wave.sha256(path)}


def result_for(job):
    root = Path(job["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    restart = root/"gpuwmrst_d01_end.npz"
    header = native_restart(restart, job["member"], seconds=job["t_end"])
    arrays = root/"arrays.npz"
    np.savez(arrays, **{"snapshot/theta": np.array([300.0], np.float32), "H_Z": np.array([35.0], np.float64)})
    payload = file_record(arrays)
    doc = {"schema": wave.SCHEMA, "complete": True, "member": job["member"],
           "request_hash": job["request_hash"], "t_start": job["t_start"], "t_end": job["t_end"],
           "output_root": str(root.resolve()), "restart": str(restart.resolve()),
           "arrays": payload, "files": [file_record(restart), payload],
           "clocks": {"1": {key: header[key] for key in wave.CLOCK_FIELDS}}}
    wave.atomic_json(job["result_path"], doc)
    return doc


def mutate(job, change):
    path = Path(job["result_path"])
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


def test_plan_32_members_on_eight_uuid_cards_requires_four_way_mps(tmp_path):
    roster = jobs(tmp_path, 32)
    cards = [{"uuid": uuid(i), "total_bytes": 32*wave.GIB, "free_bytes": 32*wave.GIB} for i in range(8)]
    plan = wave.plan_wave(roster, cards, 4, member_peak_bytes=6*wave.GIB, host_available_bytes=251*wave.GIB)
    assert plan["mps_required"]
    assert plan["decoder_threads"] == 12
    assert len(plan["waves"]) == 1
    assert plan["decoder_ram_bound_bytes"] == 160*wave.GIB
    for card in cards:
        placed = [job for job in plan["waves"][0] if job["gpu_uuid"] == card["uuid"]]
        assert len(placed) == 4
        assert {job["card_slot"] for job in placed} == {0, 1, 2, 3}


def test_two_way_fallback_fits_vram_and_uses_two_complete_waves(tmp_path):
    roster = jobs(tmp_path, 32)
    cards = [{"uuid": uuid(i), "total_bytes": 32*wave.GIB, "free_bytes": 32*wave.GIB} for i in range(8)]
    with pytest.raises(ValueError, match="capacity"):
        wave.plan_wave(roster, cards, 4, member_peak_bytes=10*wave.GIB, host_available_bytes=251*wave.GIB)
    plan = wave.plan_wave(roster, cards, 2, member_peak_bytes=10*wave.GIB, host_available_bytes=251*wave.GIB)
    assert not plan["mps_required"]
    assert [len(part) for part in plan["waves"]] == [16, 16]
    assert [job["member"] for part in plan["waves"] for job in part] == list(range(32))
    assert plan["decoder_ram_bound_bytes"] == 80*wave.GIB


@pytest.mark.parametrize("mutator", [lambda j: j.pop(), lambda j: j.__setitem__(1, dict(j[1],member=0)),
                                    lambda j: j.__setitem__(1, dict(j[1],t_start=60,t_end=120))])
def test_roster_substitution_or_mixed_clocks_cannot_clear_barrier(tmp_path, mutator):
    roster = jobs(tmp_path)
    for job in roster:
        result_for(job)
    mutator(roster)
    with pytest.raises((ValueError, wave.MemberWaveFailed)):
        wave.validate_results(roster)


def test_host_admission_uses_actual_headroom_and_keeps_48_gib_reserve(tmp_path):
    roster = jobs(tmp_path, 32)
    plan = wave.plan_wave(roster, [uuid(i) for i in range(8)], 4, host_available_bytes=120*wave.GIB)
    assert plan["decoder_ram_bound_bytes"] <= 72*wave.GIB
    assert plan["host_reserve_bytes"] == 48*wave.GIB
    with pytest.raises(ValueError, match="host RAM"):
        wave.plan_wave(roster, [uuid(i) for i in range(8)], 4, decoder_budget_bytes=5*wave.GIB, host_available_bytes=120*wave.GIB)
    with pytest.raises(ValueError, match="48 GiB"):
        wave.plan_wave(roster, [uuid(i) for i in range(8)], 4, host_available_bytes=251*wave.GIB, host_reserve_bytes=4*wave.GIB)


def test_missing_mps_refuses_before_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(wave.MemberWaveFailed, match="MPS"):
        wave.run_wave(jobs(tmp_path), [uuid()], 4, tmp_path/"wave", 5,
                      host_available_bytes=251*wave.GIB)
    receipt = json.loads((tmp_path/"wave"/"forecast-leg-receipt.json").read_text())
    assert not receipt["analysis_permitted"] and receipt["inputs_deleted"] is False


def test_host_headroom_is_rechecked_before_each_actual_wave(tmp_path,monkeypatch):
    available=iter([251*wave.GIB,20*wave.GIB])
    monkeypatch.setattr(wave,"available_host_memory_bytes",lambda:next(available))
    with pytest.raises(wave.MemberWaveFailed,match="host RAM fell"):
        wave.run_wave(jobs(tmp_path),[uuid()],2,tmp_path/"wave",5)
    assert not list((tmp_path/"wave").glob("member-*.log"))
    assert not json.loads((tmp_path/"wave"/"forecast-leg-receipt.json").read_text())["analysis_permitted"]


def test_native_restart_member_substitution_is_rejected_even_after_rehash(tmp_path):
    roster=jobs(tmp_path,1);result_for(roster[0])
    restart=Path(roster[0]["output_root"])/"gpuwmrst_d01_end.npz"
    native_restart(restart,1)
    mutate(roster[0],lambda d:d["files"].__setitem__(0,file_record(restart)))
    with pytest.raises(wave.MemberWaveFailed,match="trajectory identity"):
        wave.validate_results(roster)


@pytest.mark.parametrize("kind", ["request", "clock", "arrays_outside", "array_digest", "payload", "child_missing"])
def test_result_tamper_is_rejected_before_analysis(tmp_path, kind):
    roster = jobs(tmp_path, 1)
    result_for(roster[0])
    if kind == "request":
        mutate(roster[0], lambda d: d.update(request_hash="f"*64))
    elif kind == "clock":
        mutate(roster[0], lambda d: d["clocks"]["1"].update(elapsed_ticks=59))
    elif kind == "arrays_outside":
        foreign = tmp_path/"foreign.npz"
        np.savez(foreign, x=np.zeros(1))
        mutate(roster[0], lambda d: d.update(arrays=file_record(foreign)))
    elif kind == "array_digest":
        mutate(roster[0], lambda d: d["arrays"].update(sha256="f"*64))
    elif kind == "payload":
        Path(roster[0]["output_root"]).joinpath("arrays.npz").write_bytes(b"changed")
    else:
        root = Path(roster[0]["output_root"])
        native_restart(root/"gpuwmrst_d01_end.npz", 0, domains=(1,2))
        mutate(roster[0], lambda d: d["files"].__setitem__(0,file_record(root/"gpuwmrst_d01_end.npz")))
    with pytest.raises(wave.MemberWaveFailed):
        wave.validate_results(roster)


FAKE_WORKER = r'''
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
task=json.loads(Path(sys.argv[1]).read_text())
root=Path(task['output_root']); root.mkdir(parents=True,exist_ok=True)
(root/'started.json').write_text(json.dumps({'pid':os.getpid(),'uuid':os.environ.get('CUDA_VISIBLE_DEVICES'), 'threads':os.environ.get('GPUWM_MAPPED_ENGINE_THREADS'),'budget':os.environ.get('GPUWM_MAPPED_ENGINE_MEMORY_BUDGET_BYTES')}))
if task.get('behavior')=='fail':time.sleep(.2);sys.exit(7)
if task.get('behavior')=='sleep':time.sleep(30)
header={'format_version':6,'domain_ids':[1],'elapsed_ticks':60,'tick_den':1,'elapsed_seconds':60.0,'step_count':4,'domain_start_ticks':0,'domain_lifecycle':'ACTIVE','dtbc_fp32_bits':0,'experiment_fingerprint_components':{'trajectory':str(task['member'])},'is_stub':True}
restart=root/'gpuwmrst_d01_end.npz';np.savez(restart,__gpuwm_restart_header__=np.frombuffer(json.dumps(header).encode(),np.uint8))
arrays=root/'arrays.npz';np.savez(arrays,**{'snapshot/theta':np.array([300.0],np.float32)})
def record(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
payload=record(arrays)
doc={'schema':'gpuwm-da.member-leg-result.v1','complete':True,'member':task['member'],'request_hash':task['request_hash'],'t_start':0.0,'t_end':60.0,'output_root':str(root.resolve()),'restart':str(restart.resolve()),'arrays':payload,'files':[record(restart),payload],'clocks':{'1':{k:header[k]for k in ('elapsed_ticks','tick_den','elapsed_seconds','step_count','domain_start_ticks','domain_lifecycle','dtbc_fp32_bits')}}}
Path(task['result_path']).write_text(json.dumps(doc))
'''


def fake_jobs(tmp_path, behaviors):
    script = tmp_path/"fake_worker.py"
    script.write_text(FAKE_WORKER)
    source = tmp_path/"immutable-input.npz"
    source.write_bytes(b"checkpoint survives incomplete roster")
    roster = jobs(tmp_path,len(behaviors))
    for job,behavior in zip(roster,behaviors):
        spec = tmp_path/f"task{job['member']}.json"
        spec.write_text(json.dumps(dict(job, behavior=behavior)))
        job["argv"] = [sys.executable,str(script),str(spec)]
    return roster,source


def test_owned_cpu_workers_use_uuid_and_decode_caps_and_clear_complete_barrier(tmp_path):
    roster,source = fake_jobs(tmp_path,["ok"]*5)
    results = wave.run_wave(roster,[uuid(),uuid(1)],2,tmp_path/"wave",10,
                            host_available_bytes=251*wave.GIB)
    assert [row["member"] for row in results] == list(range(5))
    assert source.read_bytes() == b"checkpoint survives incomplete roster"
    for job in roster:
        record=json.loads(Path(job["output_root"]).joinpath("started.json").read_text())
        assert record["uuid"] in {uuid(),uuid(1)} and record["threads"] == "12"
        assert int(record["budget"]) <= 5*wave.GIB


@pytest.mark.parametrize("behavior,timeout", [("fail",10),("sleep",.7)])
def test_owned_worker_failure_or_timeout_cancels_peers_and_retains_input(tmp_path,monkeypatch,behavior,timeout):
    roster,source=fake_jobs(tmp_path,["sleep",behavior])
    processes=[]
    original=wave.subprocess.Popen
    def spawn(*args,**kwargs):
        process=original(*args,**kwargs);processes.append(process);return process
    monkeypatch.setattr(wave.subprocess,"Popen",spawn)
    with pytest.raises(wave.MemberWaveFailed):
        wave.run_wave(roster,[uuid()],2,tmp_path/"wave",timeout,
                      host_available_bytes=251*wave.GIB)
    assert processes and all(p.poll() is not None for p in processes)
    assert source.read_bytes() == b"checkpoint survives incomplete roster"
    receipt=json.loads((tmp_path/"wave"/"forecast-leg-receipt.json").read_text())
    assert receipt["complete"] is False and receipt["analysis_permitted"] is False
    assert receipt["inputs_deleted"] is False


FAKE_SLOW_WORKER = FAKE_WORKER.replace(
    "if task.get('behavior')=='sleep':time.sleep(30)",
    "if task.get('behavior')=='sleep':time.sleep(30)\n"
    "if task.get('behavior')=='slow':time.sleep(1.5)")


def slow_jobs(tmp_path, behaviors, members=None):
    script = tmp_path/"fake_slow_worker.py"
    script.write_text(FAKE_SLOW_WORKER)
    roster = jobs(tmp_path, len(behaviors))
    for job, behavior in zip(roster, behaviors):
        spec = tmp_path/f"task{job['member']}.json"
        spec.write_text(json.dumps(dict(job, behavior=behavior)))
        job["argv"] = [sys.executable, str(script), str(spec)]
    return roster


def receipt_records(tmp_path):
    receipt = json.loads((tmp_path/"wave"/"forecast-leg-receipt.json").read_text())
    return {row["member"]: row for row in receipt["records"]}


def test_a_free_slot_starts_the_next_member_without_waiting_for_the_wave(tmp_path):
    """Two cards, one slot each: member 2 starts on the card member 1 left
    while member 0 is still running, instead of after the whole wave."""
    roster = slow_jobs(tmp_path, ["slow", "ok", "ok"])
    results = wave.run_wave(roster, [uuid(), uuid(1)], 1, tmp_path/"wave", 30,
                            host_available_bytes=251*wave.GIB)
    assert [row["member"] for row in results] == [0, 1, 2]
    rows = receipt_records(tmp_path)
    assert rows[2]["launched_seconds"] < rows[0]["wall_seconds"]
    assert rows[2]["gpu_uuid"] == uuid(1)


def test_sealing_futures_are_launched_in_roster_order(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    roster = slow_jobs(tmp_path, ["ok"]*4)

    def seal(job):
        time.sleep(0.2 * (4 - job["member"]))   # later members seal first
        return job

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(seal, job) for job in roster]
        results = wave.run_wave(futures, [uuid(), uuid(1)], 1, tmp_path/"wave",
                                30, host_available_bytes=251*wave.GIB)
    assert [row["member"] for row in results] == list(range(4))
    rows = receipt_records(tmp_path)
    assert [rows[m]["launch"] for m in range(4)] == list(range(4))


def test_an_occupying_job_holds_its_card_until_it_ends(tmp_path):
    roster = slow_jobs(tmp_path, ["ok", "ok"])
    owned_dir = tmp_path/"control"
    spec = tmp_path/"control-task.json"
    control = {"member": "control", "result_path": str(owned_dir/"result.json"),
               "output_root": str(owned_dir), "request_hash": "c"*64,
               "t_start": 0.0, "t_end": 60.0, "expected_domain_ids": [1]}
    spec.write_text(json.dumps(dict(control, behavior="slow")))
    control["argv"] = [sys.executable, str(tmp_path/"fake_slow_worker.py"), str(spec)]
    owned = wave.OwnedJob(control, uuid(), tmp_path/"control.log")
    results = wave.run_wave(roster, [uuid()], 1, tmp_path/"wave", 30,
                            host_available_bytes=251*wave.GIB,
                            occupied={uuid(): owned})
    assert [row["member"] for row in results] == [0, 1]
    assert owned.poll() == 0
    rows = receipt_records(tmp_path)
    assert rows[0]["launched_seconds"] >= 1.0      # after the control ended
    checked = owned.wait(5)
    assert checked["member"] == "control"


def test_a_failed_occupying_job_fails_the_barrier(tmp_path):
    roster = slow_jobs(tmp_path, ["ok"])
    spec = tmp_path/"control-task.json"
    control = {"member": "control", "result_path": str(tmp_path/"c"/"result.json"),
               "output_root": str(tmp_path/"c"), "request_hash": "c"*64,
               "t_start": 0.0, "t_end": 60.0}
    spec.write_text(json.dumps(dict(control, behavior="fail")))
    control["argv"] = [sys.executable, str(tmp_path/"fake_slow_worker.py"), str(spec)]
    owned = wave.OwnedJob(control, uuid(), tmp_path/"control.log")
    with pytest.raises(wave.MemberWaveFailed, match="occupying"):
        wave.run_wave(roster, [uuid()], 1, tmp_path/"wave", 30,
                      host_available_bytes=251*wave.GIB,
                      occupied={uuid(): owned})


def _smi(stdout):
    class Done:
        def __init__(self):
            self.stdout = stdout
    return lambda *a, **k: Done()


def test_query_cards_defaults_to_the_cards_cuda_visible_devices_names(monkeypatch):
    # Box S 2026-10-06 18:13Z and 18:20Z: a cycle granted two cards priced
    # its packed members against the tightest of all eight and refused
    # itself while its own two cards were empty.
    rows = "\n".join(f"{uuid(i)}, 32607, {free}" for i, free in
                     enumerate([32000, 4900, 32000, 31000]))
    monkeypatch.setattr(wave.subprocess, "run", _smi(rows + "\n"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{uuid(2)},{uuid(3)}")
    cards = wave.query_cards()
    assert [card["uuid"] for card in cards] == [uuid(2), uuid(3)]
    assert min(card["free_bytes"] for card in cards) == 31000*1024**2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert len(wave.query_cards()) == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert len(wave.query_cards()) == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{uuid(1)},0")
    with pytest.raises(ValueError, match="mixes"):
        wave.query_cards()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert [c["uuid"] for c in wave.query_cards([uuid(1)])] == [uuid(1)]
