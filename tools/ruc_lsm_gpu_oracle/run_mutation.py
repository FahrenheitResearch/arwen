"""Prove run_case.sh fails on a real kernel HFX bit flip, then restores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    header = root / "gpuwm/core/kernels/ruc_fused_driver.cuh"
    original = header.read_bytes()
    needle = b"((float*)cp[13])[i]=D_F(hfx);"
    replacement = b"((float*)cp[13])[i]=i==0 ? __uint_as_float(__float_as_uint(D_F(hfx))^1u) : D_F(hfx);"
    assert original.count(needle) == 1
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for plant in (True, False):
            header.write_bytes(original.replace(needle, replacement) if plant else original)
            for mode in ("strict", "default"):
                case = args.out / f"{'planted' if plant else 'restored'}-{mode}"
                case.mkdir()
                command = ["bash", str(root / "tools/ruc_lsm_gpu_oracle/run_case.sh"),
                           str(case), mode, "--nzs", "9", "--steps", "4", "--dt", "20"]
                with (case / "shell.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            env=os.environ | {"GPUWM_WRF_EXACT": "1"})
                report = json.loads((case / "comparison-defined.json").read_text())
                exact = json.loads((case / "case.json").read_text())["wrf_exact"]
                assert exact == ("1" if mode == "strict" else "0"), (mode, exact)
                rows.append({"mode": mode, "planted": plant, "exit_code": result.returncode,
                             "differing_words": report["differing_words"], "fields": report["fields"],
                             "command": command, "log": str(case / "shell.log")})
                assert result.returncode == (1 if plant else 0), rows[-1]
                assert report["differing_words"] == (4 if plant else 0), rows[-1]
    finally:
        header.write_bytes(original)
        receipt = {"source_restored": header.read_bytes() == original,
                   "header_sha256": hashlib.sha256(original).hexdigest(), "runs": rows}
        (args.out / "mutation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"source_restored": True, "runs": [{k: r[k] for k in ("mode", "planted", "exit_code", "differing_words")} for r in rows]}))


if __name__ == "__main__":
    main()
