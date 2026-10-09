"""Actual RK forecast continuation through the production restart writer/reader.

Invoke each phase in a fresh process under the GPU mutex. Checkpoint payloads
stay on the box. The check phase records every serialized field's SHA-256.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--version", choices=("wrf_461", "gsd_41"), required=True)
    p.add_argument("--mixlength", type=int, choices=(1, 2), required=True)
    p.add_argument("--phase", choices=("reference", "split", "resume", "check"), required=True)
    p.add_argument("--cycling", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    if a.phase == "check":
        arrays = []
        headers = []
        for name in ("reference", "resume"):
            with np.load(a.folder / f"{name}.npz", allow_pickle=False) as data:
                headers.append(json.loads(data["__gpuwm_restart_header__"].tobytes()))
                arrays.append({key: data[key] for key in data.files
                               if key != "__gpuwm_restart_header__"})
        assert arrays[0].keys() == arrays[1].keys()
        rows = {}
        for key, ref in arrays[0].items():
            got = arrays[1][key]
            assert got.dtype == ref.dtype and got.shape == ref.shape, key
            assert np.isfinite(ref).all() and np.isfinite(got).all(), key
            left, right = ref.tobytes(), got.tobytes()
            assert left == right, key
            rows[key] = dict(shape=list(ref.shape), dtype=str(ref.dtype),
                             bytes=len(left), sha256=hashlib.sha256(left).hexdigest())
        assert np.any(arrays[0]["fields/qke"] > np.float32(0.01))
        assert np.any(arrays[0]["fields/exch_h"] > np.float32(0.0))
        for key in ("elapsed_seconds", "setup_fingerprint", "physics_setup_fingerprint",
                    "physics_setup", "driver", "array_manifest"):
            assert headers[0][key] == headers[1][key], key
        a.out.write_text(json.dumps(dict(version=a.version, mixlength=a.mixlength,
            cycling=a.cycling, steps=40, restart_after=20,
            elapsed_seconds=headers[1]["elapsed_seconds"],
            fields=len(rows), payload_bytes=sum(v["bytes"] for v in rows.values()),
            full_field_hashes=rows), indent=2) + "\n", encoding="utf-8")
        return
    import test_mynn_pbl_runtime as fixture
    from gpuwm.core.dycore import run_steps
    from gpuwm.io.restart import write_restart, restore_restart
    ctor = fixture.RunConfig
    fixture.RunConfig = lambda **kw: ctor(**kw, bl_mynn_version=a.version,
        bl_mynn_mixlength=a.mixlength, cycling=a.cycling)
    state, cfg, driver = fixture._build()
    if a.cycling:
        driver.fields["qke"].fill(np.float32(0.0002))
        driver.fields["qc_bl"].fill(np.float32(0.0003))
        driver.fields["cldfra_bl"].fill(np.float32(0.6))
    if a.phase == "resume":
        restore_restart(a.folder / "split.npz", state, cfg)
    run_steps(state, cfg, 40 if a.phase == "reference" else 20)
    write_restart(a.folder / f"{a.phase}.npz", state, cfg)
    print(a.phase, state.elapsed_seconds, flush=True)


if __name__ == "__main__":
    main()
