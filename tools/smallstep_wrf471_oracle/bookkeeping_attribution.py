"""Reproduce the named emdiv reciprocal rounding cause from frozen inputs."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.smallstep_bookkeeping_oracle import BOOKKEEPING_DIR, load_bookkeeping
from gpuwm.verify.smallstep_oracle import word_metrics


def main():
    manifest = json.loads((BOOKKEEPING_DIR / "manifest.json").read_text())
    rows = {}
    for name, metadata in manifest["files"].items():
        if metadata["routine"] != "emdiv":
            continue
        f = load_bookkeeping(BOOKKEEPING_DIR / name)
        gy = np.float32(-float(f["emdiv"]) * float(f["dy"])) * (
            f["in_mudf"] - np.roll(f["in_mudf"], 1, axis=0))
        if bool(f["periodic"]):
            gy = np.concatenate((gy, gy[:1]), axis=0)
            mapfactor = f["in_msfv"]
            target = f["in_v_pp"].copy()
            expected = f["ref_v_pp"]
        else:
            gy = gy[1:]
            mapfactor = f["in_msfv"][1:-1]
            target = f["in_v_pp"][:, 1:-1].copy()
            expected = f["ref_v_pp"][:, 1:-1]
        native_gradient = gy * (np.float32(1) / mapfactor)
        engine_gradient = gy / mapfactor
        native_increment = f["in_c1h"][:, None, None] * native_gradient[None]
        engine_increment = f["in_c1h"][:, None, None] * engine_gradient[None]
        rows[name] = {
            "gradient": word_metrics(engine_gradient, native_gradient),
            "increment": word_metrics(engine_increment, native_increment),
            "native_trace": word_metrics(target + native_increment, expected)}
    assert all(row["native_trace"]["different_words"] == 0 for row in rows.values())
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
