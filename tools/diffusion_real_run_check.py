"""Four real one-hour CLI runs: default timing and strict field identity.

Run this whole command under one card-queue hold. Each forecast is a fresh
process, and the original source is restored after each baseline run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--regression-receipt", type=Path,
                   help="reuse a completed passing focused-test receipt")
    a = p.parse_args()
    workspace = ROOT.parent.resolve()
    a.out.resolve().relative_to(workspace)
    a.out.mkdir(exist_ok=True)
    manifest = workspace / "MANIFEST.txt"
    with manifest.open("a") as f:
        f.write(str(a.out.resolve()) + "\n")
    tests = [sys.executable, "-m", "pytest", "-q", "--basetemp", str(a.out / "pytest"),
             "tests/test_diffusion_default_words.py", "tests/test_diffopt1_wrf461_column_oracle.py",
             "tests/test_diff6_wrf461_column_oracle.py", "tests/test_smag2d_wrf461_column_oracle.py"]
    if a.regression_receipt:
        prior = json.loads(a.regression_receipt.read_text())
        if prior.get("exit_code") != 0:
            raise ValueError("the supplied regression receipt did not pass the focused word tests")
        (a.out / "focused-tests.json").write_text(json.dumps(
            {**prior, "reused_receipt": str(a.regression_receipt)}) + "\n")
    else:
        with (a.out / "focused-tests.log").open("w") as log:
            test_run = subprocess.run(tests, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        (a.out / "focused-tests.json").write_text(json.dumps({"command": tests, "exit_code": test_run.returncode}) + "\n")
        if test_run.returncode:
            return 1

    def checkout(commit):
        subprocess.run(["git", "checkout", "--detach", commit], cwd=ROOT, check=True)

    results = {}
    for arm, commit, strict in (("before-default", a.before, False),
                               ("before-strict", a.before, True),
                               ("after-default", a.after, False),
                               ("after-strict", a.after, True)):
        checkout(commit)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GPUWM_WRF_EXACT")}
        if strict:
            env["GPUWM_WRF_EXACT"] = "1"
        target = a.out / arm
        command = [sys.executable, "-m", "gpuwm.wrfinput_forecast", "--wrfinput", str(a.input),
                   "--outdir", str(target), "--run-seconds", "3600", "--_worker",
                   "--products", "none", "--progress-every", "1"]
        with (a.out / f"{arm}.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        checkout(a.after)
        receipt = a.out / f"{arm}.json"
        observer = subprocess.run([sys.executable, str(ROOT / "tools/diffusion_step_receipt.py"),
                                   str(target), str(receipt), "--commit", commit], cwd=ROOT)
        results[arm] = {"command": command, "forecast_exit": process.returncode,
                        "observer_exit": observer.returncode, "receipt": receipt.name}
        (a.out / "status.json").write_text(json.dumps(results, indent=2) + "\n")
        if process.returncode or observer.returncode:
            return 1
        # The receipt now holds every field hash and the step timings.
        # Remove only raw files the forecast created in this owned run.
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            raw = (path.name.startswith(("wrfout_d", "gpuwmrst")) or
                   path.suffix.lower() in (".nc", ".npy", ".npz", ".bin", ".grib", ".grb", ".zst"))
            if raw:
                path.parent.resolve().relative_to(target.resolve())
                size = path.lstat().st_size
                with (a.out / "deleted-raw.jsonl").open("a") as log:
                    log.write(json.dumps({"path": str(path), "bytes": size}) + "\n")
                path.unlink()
    before = json.loads((a.out / "before-default.json").read_text())
    after = json.loads((a.out / "after-default.json").read_text())
    strict_before = json.loads((a.out / "before-strict.json").read_text())
    strict_after = json.loads((a.out / "after-strict.json").read_text())
    old_frames, new_frames = list(strict_before["frames"].values()), list(strict_after["frames"].values())
    if len(old_frames) != len(new_frames) or not old_frames:
        raise ValueError("strict frame counts differ or no frame was written")
    moved = [(i, name) for i, (old, new) in enumerate(zip(old_frames, new_frames))
             for name in set(old) | set(new) if old.get(name) != new.get(name)]
    result = {"default_median_before": before["median_after_first"],
              "default_median_after": after["median_after_first"],
              "default_step_delta_percent": 100 * (after["median_after_first"] / before["median_after_first"] - 1),
              "strict_changed_fields": moved, "strict_frames": len(old_frames),
              "strict_fields_compared": sum(len(x) for x in old_frames),
              "pass": not moved}
    (a.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return int(bool(moved))


if __name__ == "__main__":
    raise SystemExit(main())
