"""The fork Thompson table lock is taken once per process, never per step.

Every fork Thompson microphysics call (thompson_version = "wrf_39_noaa")
enters gpuwm.thompson_fork_assets.ensure_thompson_fork_tables through
gpuwm.core.microphysics_aerosol._wrf39_table_root.  It used to take the
cross-process "fetch-tables" lock before looking at its in-process verified
cache, so every model step of every process locked one shared file.  On
2026-10-06 about 70 DA member processes sharing one TMPDIR and table root
queued on that file at its 0.25 s poll with their GPUs near 0%.

These tests pin the repaired order: a process verifies the set under the
lock once, then steps lock-free on a stat-only signature, and a changed file
still falls through to the locked SHA-256 verification.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from gpuwm import fetch_guard
from gpuwm import table_assets
from gpuwm import thompson_fork_assets as fork
from gpuwm.core.thompson_contract import TableAsset

_REPO = Path(__file__).resolve().parents[1]
_PROCESSES = 6
_STEPS = 200


def _payloads():
    # Small transport fixtures exercise the lock, not scientific coefficients.
    return {f"part-{i}.dat": (f"pinned fork table {i}\n" * 64)
            for i in range(4)}


def _assets(payloads):
    return tuple(TableAsset(name, len(text.encode()),
                            hashlib.sha256(text.encode()).hexdigest())
                 for name, text in payloads.items())


@pytest.fixture
def staged(tmp_path, monkeypatch):
    payloads = _payloads()
    root = tmp_path / "thompson-wrf39-noaa"
    root.mkdir()
    for name, text in payloads.items():
        (root / name).write_bytes(text.encode())
    monkeypatch.setattr(fork, "FORK_TABLE_ASSETS", _assets(payloads))
    monkeypatch.setattr(fork, "_packaged_source", lambda: None)
    monkeypatch.setenv(fork.FORK_RELEASE_BASE_ENV, "")
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(tmp_path / "locks"))
    fork._VERIFIED_ROOTS.clear()
    holds = []
    real_hold = fetch_guard.hold

    def counting_hold(kind, target, **kwargs):
        holds.append(kind)
        return real_hold(kind, target, **kwargs)

    monkeypatch.setattr(fetch_guard, "hold", counting_hold)
    yield root, payloads, holds
    fork._VERIFIED_ROOTS.clear()


def test_a_verified_root_steps_without_the_lock(staged):
    root, _payloads_, holds = staged
    for _ in range(_STEPS):
        assert fork.ensure_thompson_fork_tables(root) == root
    assert holds == ["fetch-tables"]


def test_a_changed_file_is_verified_again_under_the_lock(staged):
    root, payloads, holds = staged
    fork.ensure_thompson_fork_tables(root)
    name = next(iter(payloads))
    # Same bytes, new file: the signature moves, the pins still hold.
    replacement = root / (name + ".new")
    replacement.write_bytes(payloads[name].encode())
    os.replace(replacement, root / name)
    assert fork.ensure_thompson_fork_tables(root) == root
    assert holds == ["fetch-tables", "fetch-tables"]
    assert fork.ensure_thompson_fork_tables(root) == root
    assert len(holds) == 2


def test_wrong_bytes_after_verification_are_still_refused(staged):
    root, payloads, holds = staged
    fork.ensure_thompson_fork_tables(root)
    bad = root / next(iter(payloads))
    replacement = root / "wrong.tmp"
    replacement.write_bytes(b"x" * bad.stat().st_size)
    os.replace(replacement, bad)
    with pytest.raises(table_assets.TableAssetError, match="without overwrite"):
        fork.ensure_thompson_fork_tables(root)
    assert len(holds) == 2


def test_the_per_step_path_is_the_function_these_tests_drive():
    # gpuwm.core.microphysics_aerosol imports cupy at module scope, so the
    # CPU tests below drive the same call it makes on every step: the
    # no-argument ensure, which resolves the root from the environment.
    source = (_REPO / "gpuwm" / "core" / "microphysics_aerosol.py").read_text(
        encoding="utf-8")
    body = source.split("def _wrf39_table_root", 1)[1].split("\ndef ", 1)[0]
    assert "return str(ensure_thompson_fork_tables())" in body
    step = source.split("def _apply_thompson_aerosol_call", 1)[1]
    assert "_wrf39_table_root(), version=version)" in step


_CHILD = textwrap.dedent('''
    import hashlib, json, os, sys, time
    from pathlib import Path

    from gpuwm import fetch_guard
    from gpuwm import thompson_fork_assets as fork
    from gpuwm.core.thompson_contract import TableAsset

    mode, steps, go = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    if mode == "cpu":
        payloads = json.loads(os.environ["FORK_LOCK_TEST_PAYLOADS"])
        fork.FORK_TABLE_ASSETS = tuple(
            TableAsset(name, len(text.encode()),
                       hashlib.sha256(text.encode()).hexdigest())
            for name, text in payloads.items())
    holds = []
    real_hold = fetch_guard.hold

    def counting_hold(kind, target, **kwargs):
        holds.append(kind)
        return real_hold(kind, target, **kwargs)

    fetch_guard.hold = counting_hold
    if mode == "gpu":
        import cupy as cp
        sys.path.insert(0, os.environ["FORK_LOCK_TEST_DIR"])
        import test_thompson_aerosol_adapter as harness
        from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol
        state, cfg, dt, *_ = harness._build_case(cp, harness._FIXTURES[0])
        cfg.thompson_version = "wrf_39_noaa"
        cfg.thompson_fork_snow_fall = "wrf_39_noaa"

        def step():
            _apply_thompson_aerosol(state, cfg, dt)
            cp.cuda.Stream.null.synchronize()
    else:
        def step():
            fork.ensure_thompson_fork_tables()
    deadline = time.monotonic() + 120
    while not go.exists():
        if time.monotonic() > deadline:
            raise SystemExit("start barrier never opened")
        time.sleep(0.005)
    started = time.monotonic()
    for _ in range(steps):
        step()
    print(json.dumps({"holds": len(holds),
                      "kinds": sorted(set(holds)),
                      "seconds": time.monotonic() - started}))
''')


def _run_children(tmp_path, mode, processes, steps, env_extra):
    script = tmp_path / "fork_lock_child.py"
    script.write_text(_CHILD, encoding="utf-8")
    go = tmp_path / "go"
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["FORK_LOCK_TEST_DIR"] = str(_REPO / "tests")
    children = [subprocess.Popen(
        [sys.executable, str(script), mode, str(steps), str(go)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(processes)]
    time.sleep(0.5)
    go.write_text("go")
    results = []
    for child in children:
        out, err = child.communicate(timeout=900)
        assert child.returncode == 0, err[-4000:]
        results.append(json.loads(out.strip().splitlines()[-1]))
    return results


def test_concurrent_processes_take_the_lock_at_most_once_each(tmp_path):
    # Every process shares one TMPDIR, HOME, lock root and table root, the
    # shape of the DA member fleet that stalled.
    payloads = _payloads()
    root = tmp_path / "home" / ".gpuwm" / "tables" / "thompson-wrf39-noaa"
    root.mkdir(parents=True)
    for name, text in payloads.items():
        (root / name).write_bytes(text.encode())
    shared_tmp = tmp_path / "tmp"
    shared_tmp.mkdir()
    results = _run_children(tmp_path, "cpu", _PROCESSES, _STEPS, {
        "FORK_LOCK_TEST_PAYLOADS": json.dumps(payloads),
        "HOME": str(tmp_path / "home"),
        "TMPDIR": str(shared_tmp),
        fetch_guard.LOCK_ROOT_ENV: str(shared_tmp / "gpuwm" / "locks"),
        fork.FORK_RELEASE_BASE_ENV: "",
        "GPUWM_THOMPSON_FORK_TABLE_ROOT": str(root),
    })
    assert [result["holds"] for result in results] == [1] * _PROCESSES
    assert {kind for result in results for kind in result["kinds"]} == {
        "fetch-tables"}
