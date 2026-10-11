"""Run native-cell comparisons and Rust product sheets for one matched hour.

All field reading, normalization, gridding, comparison and rendering are Rust.
This file selects artifacts, invokes those tools and writes command receipts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "history", "reference", "mesh", "init", "bin-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cycle", required=True)
    p.add_argument("--hour", type=int, required=True)
    p.add_argument("--start-time", required=True)
    p.add_argument("--reference-model", required=True)
    p.add_argument("--candidate-label", required=True)
    p.add_argument("--native-label", required=True)
    args = p.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    commands = (json.loads((root / "commands.json").read_text())
                if (root / "commands.json").exists() else [])

    def run(argv, name):
        argv = [str(item) for item in argv]
        with (root / (name + ".log")).open("w") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        commands.append({"command": argv, "exit_code": result.returncode,
                         "log": str(root / (name + ".log"))})
        (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {root / (name + '.log')}")

    normalized = root / ("native-normalized." + args.reference.name.partition(".")[2])
    previous = root / "native-normalized.nc"
    if previous.exists() and not normalized.exists():
        previous.rename(normalized)
    if not normalized.exists():
        run([root.parent / "normalize_history", args.reference, args.init, normalized], "normalize")
    if not (root / "surface-comparison.json").exists():
        run([root.parent / "native_compare", args.mesh, args.history, args.reference,
             args.init, args.init, "-", "-", root / "surface-comparison.json",
             "--raw", "t2m,mslp,rainnc,rainc"], "surface-comparison")
    frames = {}
    for name, history, label in (("candidate", args.history, args.candidate_label),
                                  ("native", normalized, args.native_label)):
        directory = root / name
        if not list(directory.glob("wrfout*")):
            run([args.bin_dir / "rw_mpas_convert", "--history", history, "--mesh", args.mesh,
                 "--init", args.init, "--out-dir", directory, "--window", "mesh",
                 "--field-set", "surface", "--format", "cdf2", "--simulation-start",
                 args.start_time, "--model-label", label, "--json", root / (name + "-convert.json")],
                name + "-convert")
        files = sorted(directory.glob("wrfout*"))
        if len(files) != 1:
            raise RuntimeError(f"expected one converted frame in {directory}, found {len(files)}")
        frames[name] = files[0]
    run([args.bin_dir / "rw_compare", "--fetch-reference", "--store-root", root / "store",
         "--reference", args.reference_model, "--cycle", args.cycle, "--forecast-hour", args.hour], "fetch-reference")
    run([args.bin_dir / "rw_compare", "--store-root", root / "store", "--out-dir", root / "sheets",
         "--reference", args.reference_model, "--cycle", args.cycle, "--products", "refc,qpf1h,t2m,wspd10",
         "--difference", "off", "--width", "1200", "--height", "900", "--source-label",
         args.candidate_label, "--run-label", "hex", frames["candidate"]], "reference-sheets")
    run([args.bin_dir / "rw_compare", "--store-root", root / "native-store", "--out-dir", root / "sheets",
         "--panel", args.candidate_label + "=" + str(frames["candidate"]), "--panel", args.native_label + "=" + str(frames["native"]),
         "--products", "t2m,qpf1h", "--width", "1200", "--height", "900", "--sheet-name", "native-pair"],
        "native-sheets")
    pictures = [{"path": str(path), "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted((root / "sheets").rglob("*.png"))]
    (root / "pictures.json").write_text(json.dumps(pictures, indent=2) + "\n")
    print(json.dumps({"pictures": len(pictures), "root": str(root)}))


if __name__ == "__main__":
    main()
