#!/usr/bin/env python3
"""Drive the four-command transcript from a fresh venv and a built wheel.

    pip install gpuwm[all]
    gpuwm setup
    gpuwm domain ...
    gpuwm go <config>

The bar this exists to hold is that those four commands are the whole
product for a new reader: no clone, no ``tools/`` on disk, no PYTHONPATH
pointing back at a checkout.  Until the forecast runners moved into the
package, the fourth command could not work from an install at all --
its last two stages were script paths only a git checkout has -- so
this script is also the regression gate for that.

It is deliberately a script rather than only a unit test.  Building a
wheel, making a venv and installing CuPy is minutes of work and tens of
megabytes; a suite that did it on every run would be turned off.  What
the suite carries is the fast, non-vacuous half
(``tests/test_pip_install_reachability.py``): a venv, the wheel, and
the two entry points that did not exist before.

Usage:

    python tools/acceptance_pip_install_chain.py --workdir DIR [--extras gpu,render]
                                                 [--point LAT,LON] [--card 12gb]
                                                 [--stop-after STAGE]

Every step's command and outcome is printed and recorded in
``DIR/transcript.json``.  A stage that cannot run in this environment
(no network, no card) is reported as the named stopping point, never
as a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(label: str, command: list[str], *, cwd: Path,
         env: dict[str, str] | None = None,
         timeout: float | None = None) -> dict:
    """Run one command; return a record of exactly what happened."""

    print(f"\n=== {label} ===\n$ {' '.join(str(c) for c in command)}",
          flush=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(c) for c in command], cwd=str(cwd), env=env,
            capture_output=True, text=True, errors="replace",
            timeout=timeout)
        code, out, err = (completed.returncode, completed.stdout,
                          completed.stderr)
    except subprocess.TimeoutExpired as expired:
        code = None
        out = (expired.stdout or b"").decode("utf-8", "replace") \
            if isinstance(expired.stdout, bytes) else (expired.stdout or "")
        err = f"timed out after {timeout} s"
    elapsed = time.monotonic() - started
    tail = "\n".join((out + err).splitlines()[-40:])
    print(tail, flush=True)
    print(f"--- {label}: exit {code} in {elapsed:.1f} s", flush=True)
    return {"stage": label, "command": [str(c) for c in command],
            "exit_code": code, "seconds": round(elapsed, 1),
            "stdout_tail": "\n".join(out.splitlines()[-60:]),
            "stderr_tail": "\n".join(err.splitlines()[-60:])}


def _venv_bin(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


def _clean_env(venv: Path) -> dict[str, str]:
    """An environment with no path back to the checkout.

    This is the whole point of the exercise, so it is enforced rather
    than assumed: PYTHONPATH is cleared, the venv leads PATH, and the
    working directory (set by every caller below) is outside the repo.
    """

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = str(_venv_bin(venv)) + os.pathsep + env.get("PATH", "")
    return env


def build_wheel(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(REPO),
                    "--no-deps", "--no-build-isolation", "-w", str(outdir)],
                   check=True)
    wheels = sorted(outdir.glob("gpuwm-*.whl"))
    if not wheels:
        raise SystemExit("no wheel was built")
    return wheels[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, default=None,
                        help="a prebuilt wheel; built from the tree if absent")
    parser.add_argument("--extras", default="gpu,render")
    parser.add_argument("--point", default="35.3,-97.5")
    parser.add_argument("--card", default="12gb")
    parser.add_argument("--hours", default="6")
    parser.add_argument("--cycle", default="latest")
    parser.add_argument("--stop-after", default=None,
                        choices=("install", "setup", "domain", "go"))
    args = parser.parse_args(argv)

    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    wheel = (args.wheel.resolve() if args.wheel
             else build_wheel(work / "wheelhouse"))
    print(f"wheel: {wheel}")

    venv = work / "venv"
    if venv.exists():
        shutil.rmtree(venv)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    env = _clean_env(venv)
    python = _venv_bin(venv) / ("python.exe" if os.name == "nt" else "python")
    gpuwm = _venv_bin(venv) / ("gpuwm.exe" if os.name == "nt" else "gpuwm")

    # The run directory is outside the checkout, so nothing can resolve
    # `tools/...` by accident.
    run_dir = work / "run"
    run_dir.mkdir(exist_ok=True)

    spec = f"{wheel}[{args.extras}]" if args.extras else str(wheel)
    records.append(_run("1. pip install gpuwm[all]",
                        [python, "-m", "pip", "install", spec],
                        cwd=run_dir, env=env, timeout=3600))

    # Before anything else: the thing that could not be done at all from
    # an install before this wave.  A failure here is the regression.
    records.append(_run(
        "1a. runner reachable from the install",
        [python, "-m", "gpuwm.prepared_single_domain_forecast",
         "--show-capabilities"], cwd=run_dir, env=env, timeout=600))
    records.append(_run("1b. no checkout is visible",
                        [python, "-c",
                         "import gpuwm, pathlib, sys;"
                         "print(pathlib.Path(gpuwm.__file__).parent);"
                         "print('cwd', pathlib.Path.cwd())"],
                        cwd=run_dir, env=env, timeout=300))

    if args.stop_after != "install":
        records.append(_run("2. gpuwm setup", [gpuwm, "setup"],
                            cwd=run_dir, env=env, timeout=3600))
    if args.stop_after not in ("install", "setup"):
        records.append(_run(
            "3. gpuwm domain",
            [gpuwm, "domain", f"--point={args.point}", "--card", args.card,
             "--ladder", "12", "--source", "gfs", "--cycle", args.cycle,
             "--hours", args.hours, "--physics-profile",
             "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
             "--out", "area.toml"],
            cwd=run_dir, env=env, timeout=1800))
    if args.stop_after is None:
        records.append(_run("4. gpuwm go", [gpuwm, "go", "area.toml"],
                            cwd=run_dir, env=env, timeout=14400))

    transcript = {
        "schema": "gpuwm-pip-install-acceptance-v1",
        "platform": platform.platform(),
        "wheel": str(wheel),
        "extras": args.extras,
        "workdir": str(work),
        "stages": records,
        "reached": next((r["stage"] for r in reversed(records)
                         if r["exit_code"] == 0), None),
        "stopped_at": next((r["stage"] for r in records
                            if r["exit_code"] not in (0,)), None),
    }
    (work / "transcript.json").write_text(
        json.dumps(transcript, indent=2), encoding="utf-8")
    print(f"\ntranscript: {work / 'transcript.json'}")
    print(f"stopped at: {transcript['stopped_at'] or 'nothing -- all stages passed'}")
    return 0 if transcript["stopped_at"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
