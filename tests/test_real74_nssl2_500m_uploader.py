"""Integration contract for the detached real74 NSSL-2 R2 uploader."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UPLOADER = REPOSITORY_ROOT / "tools" / "upload_real74_nssl2_wrfout_r2.sh"


def _calendar() -> dict[str, list[datetime]]:
    start = datetime(1974, 4, 3, 12)
    return {
        "d01": [start + timedelta(hours=index) for index in range(13)],
        "d02": [start + timedelta(hours=index) for index in range(13)],
        "d03": [start + timedelta(hours=index) for index in range(13)],
        "d04": [start + timedelta(minutes=30 * index) for index in range(25)],
    }


@pytest.mark.skipif(os.name == "nt", reason="production uploader requires Linux")
def test_uploader_parallelizes_only_stable_files_and_verifies_remote_sha(tmp_path):
    if shutil.which("bash") is None or shutil.which("flock") is None:
        pytest.skip("bash and flock are required")

    run_dir = tmp_path / "run"
    state_dir = tmp_path / "state"
    remote_dir = tmp_path / "remote"
    run_dir.mkdir()
    remote_dir.mkdir()

    outputs: dict[str, list[dict[str, object]]] = {}
    for domain, valid_times in _calendar().items():
        records = []
        for index, valid in enumerate(valid_times):
            name = f"wrfout_{domain}_{valid:%Y-%m-%d_%H_%M_%S}"
            payload = f"{domain}:{index}:{valid.isoformat()}\n".encode()
            path = run_dir / name
            path.write_bytes(payload)
            records.append({
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        outputs[domain] = records
    (run_dir / "completion.json").write_text(
        json.dumps({"outputs": outputs}), encoding="utf-8")
    metadata = run_dir / "metadata"
    metadata.mkdir()
    for name in (
        "launch-manifest.json",
        "preflight-receipt.json",
        "real74_nssl2_500m.effective.toml",
    ):
        (metadata / name).write_text(f"fixture {name}\n", encoding="utf-8")

    fake_rclone = tmp_path / "fake-rclone.py"
    fake_rclone.write_text(
        """#!/usr/bin/env python3
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

root = Path(os.environ["FAKE_R2_ROOT"])
counter = Path(os.environ["FAKE_R2_COUNTER"])
command = sys.argv[1]

def target_path(value):
    return root / value.rsplit("/", 1)[-1]

def change_active(delta):
    counter.touch(exist_ok=True)
    with counter.open("r+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        fields = stream.read().strip().split()
        active, maximum = map(int, fields) if fields else (0, 0)
        active += delta
        maximum = max(maximum, active)
        stream.seek(0)
        stream.truncate()
        stream.write(f"{active} {maximum}\\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)

if command == "copyto":
    change_active(1)
    try:
        # Leave enough overlap to prove four-worker scheduling even on a
        # heavily loaded CPU reference controller.
        time.sleep(0.25)
        shutil.copyfile(sys.argv[2], target_path(sys.argv[3]))
    finally:
        change_active(-1)
elif command == "lsl":
    path = target_path(sys.argv[2])
    if path.is_file():
        print(f"{path.stat().st_size} 1970-01-01 00:00:00.000000000 {path.name}")
elif command == "lsf":
    for path in sorted(root.iterdir()):
        if path.is_file():
            print(path.name)
elif command == "cat":
    sys.stdout.buffer.write(target_path(sys.argv[2]).read_bytes())
else:
    raise SystemExit(f"unsupported fake rclone command: {command}")
""",
        encoding="utf-8",
    )
    fake_rclone.chmod(0o755)

    counter = tmp_path / "concurrency.txt"
    environment = dict(os.environ)
    environment.update({
        "RCLONE_BIN": str(fake_rclone),
        "PYTHON_BIN": sys.executable,
        "FAKE_R2_ROOT": str(remote_dir),
        "FAKE_R2_COUNTER": str(counter),
        "GPUWM_R2_POLL_SECONDS": "1",
        "GPUWM_R2_MINIMUM_AGE_SECONDS": "0",
        "GPUWM_R2_PARALLEL_FILES": "4",
    })
    completed = subprocess.run(
        ["bash", str(UPLOADER), str(run_dir), str(state_dir)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(list((state_dir / "uploaded").glob("wrfout_d0*"))) == 64
    assert "COMPLETE uploaded=64 remote=64 sha256=64 manifests=5" in (
        state_dir / "uploader.log").read_text(encoding="utf-8")
    active, maximum = map(int, counter.read_text(encoding="utf-8").split())
    assert active == 0
    assert maximum == 4
