"""Replay a printed production forecast command and compare real restart bytes.

Run under the GPU mutex, with the normal Rust I/O libraries on the environment.
The command file is stdout from ``gpuwm sim ... --print-command``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import numpy as np


def compare(reference, resumed):
    with np.load(reference, allow_pickle=False) as left, np.load(resumed, allow_pickle=False) as right:
        assert left.files == right.files, "restart field inventory differs"
        rows = {}
        for name in left.files:
            if name == "__gpuwm_restart_header__":
                continue
            a, b = left[name], right[name]
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert a.tobytes() == b.tobytes(), name
            rows[name] = dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes,
                              sha256=hashlib.sha256(a.tobytes()).hexdigest())
    return dict(status="PASS", fields=len(rows), seconds=3600,
                restart_seconds=1800, full_field_hashes=rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--command-file", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    command = next(shlex.split(line) for line in a.command_file.read_text().splitlines()
                   if " -m gpuwm.prepared_single_domain_forecast " in line)
    assert command[1:3] == ["-m", "gpuwm.prepared_single_domain_forecast"]
    a.scratch.mkdir(parents=True, exist_ok=True)
    root = a.scratch / uuid.uuid4().hex[:8]
    root.mkdir()
    runner = Path(__file__).with_name("mynn_tkeadvect_real_run.py")
    for phase in ("reference", "resume"):
        run = [sys.executable, str(runner), "--receipt", str(root / f"{phase}.json"),
               "--final-checkpoint", str(root / f"{phase}.npz")]
        if phase == "reference":
            run += ["--split-checkpoint-dir", str(root / "split")]
        args = command[3:].copy()
        args[args.index("--outdir") + 1] = str(root / phase)
        if phase == "resume":
            split = json.loads((root / "reference.json").read_text())["split_checkpoint"]
            assert split is not None, "the real split checkpoint was not published"
            args += ["--restart", split]
        run += ["--"] + args
        print(shlex.join(run), flush=True)
        subprocess.run(run, check=True)
    result = compare(root / "reference.npz", root / "resume.npz")
    for phase in ("reference", "resume"):
        result[phase] = json.loads((root / f"{phase}.json").read_text())
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("REAL_RESTART_PASS", result["fields"], flush=True)


if __name__ == "__main__":
    main()
