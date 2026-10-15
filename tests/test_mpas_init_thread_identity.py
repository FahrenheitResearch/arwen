"""Real binary regression: workers do not change any emitted init byte.

The opt-in fixture supplies a small grid/static, a real WPS intermediate,
and a constructed vertical capsule. No CUDA context is created. A failure
means parallel column scheduling moved arithmetic or serialization.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


def test_real_init_one_vs_many_threads(tmp_path):
    manifest = os.environ.get("GPUWM_HEX_INIT_IDENTITY_MANIFEST")
    if not manifest:
        pytest.skip("set GPUWM_HEX_INIT_IDENTITY_MANIFEST to real small-mesh inputs")
    conf = json.loads(Path(manifest).read_text())
    output = tmp_path / "init.nc"
    hashes = []
    # The same binary and output path keep provenance and file-id inputs fixed.
    for workers in (1, 16):
        env = dict(os.environ, RAYON_NUM_THREADS=str(workers), CUDA_VISIBLE_DEVICES="")
        command = [conf["binary"], *conf["argv"], "--out", str(output)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        with output.open("rb") as stream:
            hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
    assert hashes[0] == hashes[1], f"whole-file bytes changed: {hashes}"
    (tmp_path / "identity.json").write_text(json.dumps({"workers": [1, 16],
        "sha256": hashes, "bytes": output.stat().st_size, "command": command}, indent=2))
