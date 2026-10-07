#!/usr/bin/env python3
"""Run the Windows cpu job's Stage 1 selection with no native artifact reachable.

The breakage it catches, named: the windows-2025 cpu job of public CI
builds no natives and runs Stage 1 minus the rows of
``tools/battery/stage1_native_files.txt``.  A Stage 1 file that needs a
compiled artifact but is missing from that partition passes the private
Linux gate, where every native is built, and fails only on the public
runner.  Public CI 37665304517 failed that way on 2.8.7: 81 failures and
19 errors in 12 files, none of them seen before the push.  This runs the
same selection on any host and makes such a file fail there too.

What it does, in order:

1. clones the checkout (its HEAD, with uncommitted edits to tracked files
   copied over) into a work directory, so no ``tools/*/target`` build and
   no ``libexec/bridges`` directory exists in the tree under test;
2. makes a fresh virtual environment and installs the clone the way
   ci.yml's cpu job installs its checkout (editable ``gpuwm-data`` and
   ``gpuwm[dev]``, numpy at the pin ci.yml sets for this host's cpu leg),
   so no editable install of a built checkout can answer an import;
3. clears every ``GPUWM_*``, ``RUSTWX_*`` and ``ARWEN_*`` variable,
   ``PYTHONPATH``, ``PYTHONHOME``, ``CARGO_TARGET_DIR`` and the ``XDG_*``
   directories, sets ``GPUWM_NO_LOCAL_GPU=1`` and an empty
   ``CUDA_VISIBLE_DEVICES``, and points ``HOME`` (and ``USERPROFILE``) at
   an empty directory, so no ``~/.gpuwm/bridges`` estate is reachable;
4. stages the Thompson tables as the cpu job does (``python -m gpuwm.cli
   fetch-tables``), or copies a staged table tree with ``--tables-from``
   for an offline host; tables are data, never natives;
5. runs ``tools/battery/run_stage1.py --platform windows`` with the cpu
   job's pytest arguments, and refuses first if any native artifact
   directory exists in the clone.

It is not a Stage 1 entry: Stage 1 running it would run Stage 1 again.
The integrator runs it on the release candidate on a Linux box, beside
the Stage 1 leg, from the candidate checkout:

    python tools/battery/emulate_windows_cpu_partition.py --work-dir <scratch>/winpart

Arguments after ``--`` are appended to the pytest command.  The exit
status is pytest's; 2 is a refusal before pytest ran.  The work directory
is removed at the end unless ``--keep`` is given.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
CI_WORKFLOW = ".github/workflows/ci.yml"

#: Variable families that can name a native artifact, a staged estate or a
#: different tree.  The cpu job sets none of them.
CLEARED_PREFIXES = ("GPUWM_", "RUSTWX_", "ARWEN_", "XDG_")
CLEARED_NAMES = ("PYTHONPATH", "PYTHONHOME", "CARGO_TARGET_DIR", "VIRTUAL_ENV")

#: Directories in a tree where a built or staged native can be found by a
#: resolution ladder (gpuwm.bridges.artifact_candidates and its siblings).
NATIVE_DIRECTORY_GLOBS = ("tools/*/target", "tools/*/*/target", "libexec/bridges",
                          "gpuwm/libexec/bridges")

#: The cpu job's pytest arguments after the worker count (ci.yml).
CPU_JOB_MARKERS = "not gpu and not slow and not network and not static_platform_qualification"


class Refusal(RuntimeError):
    """The emulation cannot run as the cpu job would; nothing was tested."""


def host_platform() -> str:
    return "windows" if os.name == "nt" else "linux"


def cpu_leg(root: pathlib.Path, platform: str) -> dict:
    """ci.yml's cpu matrix row for ``platform`` (workers, numpy pin)."""
    import yaml

    workflow = yaml.safe_load((root / CI_WORKFLOW).read_text(encoding="utf-8"))
    rows = [row for row in workflow["jobs"]["cpu"]["strategy"]["matrix"]["include"]
            if row.get("platform") == platform]
    if len(rows) != 1:
        raise Refusal(f"{CI_WORKFLOW} has {len(rows)} cpu rows for platform {platform}, not one")
    return rows[0]


def emulated_environment(environ: dict[str, str], home: pathlib.Path,
                         venv: pathlib.Path) -> dict[str, str]:
    """The cpu job's environment on a Windows runner with no native build."""
    old_bins = set()
    if environ.get("VIRTUAL_ENV"):
        old_bins = {str(pathlib.Path(environ["VIRTUAL_ENV"]) / name) for name in ("bin", "Scripts")}
    env = {key: value for key, value in environ.items()
           if not key.startswith(CLEARED_PREFIXES) and key not in CLEARED_NAMES}
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    path = [entry for entry in environ.get("PATH", "").split(os.pathsep)
            if entry and entry not in old_bins]
    env.update({
        "HOME": str(home),
        "USERPROFILE": str(home),
        "VIRTUAL_ENV": str(venv),
        "PATH": os.pathsep.join([str(bin_dir), *path]),
        "GPUWM_NO_LOCAL_GPU": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONUTF8": "1",
    })
    return env


def reachable_native_directories(tree: pathlib.Path) -> list[pathlib.Path]:
    """Native artifact directories present in ``tree`` (none in a fresh clone)."""
    found = []
    for pattern in NATIVE_DIRECTORY_GLOBS:
        found.extend(path for path in tree.glob(pattern) if path.is_dir())
    return sorted(found)


def _run(command, **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


def _git(root: pathlib.Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                          capture_output=True, text=True).stdout


def clone_checkout(root: pathlib.Path, destination: pathlib.Path) -> list[str]:
    """A clone of ``root``'s HEAD carrying its uncommitted tracked edits."""
    head = _git(root, "rev-parse", "HEAD").strip()
    _run(["git", "clone", "--quiet", "--shared", "--no-checkout", root, destination])
    _run(["git", "-C", destination, "checkout", "--quiet", "--detach", head])
    changed = [name for name in _git(root, "diff", "--no-renames", "--name-only", "-z", "HEAD").split("\0") if name]
    for name in changed:
        source, target = root / name, destination / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif target.is_file():
            target.unlink()
    return changed


def make_environment(clone: pathlib.Path, venv: pathlib.Path, python: str,
                     numpy: str, uv: str | None) -> pathlib.Path:
    """A fresh environment holding the clone exactly as the cpu job installs it."""
    packages = ["-e", "./gpuwm-data", "-e", ".[dev]", f"numpy=={numpy}"]
    if uv:
        _run([uv, "venv", "-q", "--seed", "--python", python, venv], cwd=clone)
        interpreter = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run([uv, "pip", "install", "-q", "--python", interpreter, *packages], cwd=clone)
    else:
        _run([python, "-m", "venv", venv], cwd=clone)
        interpreter = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run([interpreter, "-m", "pip", "install", "-q", *packages], cwd=clone)
    return interpreter


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT, help="checkout to emulate")
    parser.add_argument("--work-dir", type=pathlib.Path,
                        help="empty or absent scratch directory (default: a new temporary one)")
    parser.add_argument("--python", default=sys.executable, help="base interpreter for the fresh environment")
    parser.add_argument("--uv", help="uv executable; builds the environment with uv instead of venv and pip")
    parser.add_argument("--tables-from", type=pathlib.Path,
                        help="a staged ~/.gpuwm/tables tree to copy instead of running fetch-tables")
    parser.add_argument("--workers", type=int, help="pytest-xdist workers (default: the Windows cpu leg's)")
    parser.add_argument("--keep", action="store_true", help="keep the work directory")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    work = args.work_dir or pathlib.Path(tempfile.mkdtemp(prefix="gpuwm-winpart-"))
    work = work.resolve()
    # Refused before anything is created: the cleanup below removes the
    # work directory whole, so it must only ever hold what this run made.
    if work.exists() and any(work.iterdir()):
        print(f"emulate_windows_cpu_partition: the work directory {work} is not empty", file=sys.stderr)
        return 2
    try:
        windows_leg, host_leg = cpu_leg(root, "windows"), cpu_leg(root, host_platform())
        clone, venv, home = work / "checkout", work / "venv", work / "home"
        home.mkdir(parents=True)
        changed = clone_checkout(root, clone)
        if changed:
            print(f"copied {len(changed)} uncommitted tracked change(s) into the clone", flush=True)
        interpreter = make_environment(clone, venv, args.python, str(host_leg["numpy"]), args.uv)
        env = emulated_environment(dict(os.environ), home, venv)
        located = subprocess.run([str(interpreter), "-c", "import gpuwm; print(gpuwm.__file__)"],
                                 cwd=work, env=env, check=True, capture_output=True,
                                 text=True).stdout.strip()
        if not pathlib.Path(located).resolve().is_relative_to(clone):
            raise Refusal(f"gpuwm imports from {located}, not from the clone {clone}")
        if args.tables_from:
            shutil.copytree(args.tables_from, home / ".gpuwm" / "tables")
        else:
            _run([interpreter, "-m", "gpuwm.cli", "fetch-tables"], cwd=clone, env=env)
        staged = home / ".gpuwm" / "bridges"
        reachable = reachable_native_directories(clone) + ([staged] if staged.exists() else [])
        if reachable:
            raise Refusal("native artifacts are reachable, so this would not be the Windows cpu job: "
                          + ", ".join(str(path) for path in reachable))
        workers = args.workers if args.workers is not None else int(windows_leg["workers"])
        command = [interpreter, "tools/battery/run_stage1.py",
                   "--manifest", "tools/battery/stage1_files.txt",
                   "--native-manifest", "tools/battery/stage1_native_files.txt",
                   "--platform", "windows", "--minimum", "389", "--",
                   "-q", "-p", "no:cacheprovider", "-n", str(workers), "--dist", "loadfile",
                   "-m", CPU_JOB_MARKERS, f"--basetemp={work / 'b'}", *passthrough]
        print("+ " + " ".join(str(part) for part in command), flush=True)
        return subprocess.call([str(part) for part in command], cwd=clone, env=env)
    except (Refusal, OSError, subprocess.CalledProcessError) as error:
        print(f"emulate_windows_cpu_partition: {error}", file=sys.stderr)
        return 2
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
