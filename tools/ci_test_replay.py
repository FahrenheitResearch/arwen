"""Replay the publish workflow's ``test`` job against any source tree.

The job this replays (.github/workflows/publish.yml, job ``test``) is the
package/contract surface a public cut can break on a GPU-less runner, and
it runs against the PUBLIC source tree -- which lawfully lacks the
size-externalized Thompson tables the private tree carries.  Nothing else
in this repository executes that job's exact motions locally, so a public
checkout that fails it is discovered on the first real cut instead of
here.  This script is the local twin:

    fresh venv
      -> python -m pip install build setuptools
      -> python -m build --outdir <work>/companion-dist <tree>/gpuwm-data
      -> python -m pip install <that wheel>
      -> GPUWM_ALLOW_UNPINNED_WHEEL=1 python -m pip install -e "<tree>[dev]"
      -> python -m pytest -q -m "<marker filter>" <the job's test files>

The test-file list and the ``-m`` filter are PARSED OUT OF publish.yml,
never restated, so this replay cannot drift behind the job it replays.

Two properties of the CI runner are recreated deliberately, because each
one changed a test's verdict when it was missing:

* **A clean home.**  The runner's home directory carries no ``~/.gpuwm``;
  a developer box does, and a stale staged bridge there turned
  ``doctor``'s engine probe from a skip into a failure that names this
  machine, not the tree.  The whole replay runs under a fresh
  ``HOME``/``USERPROFILE`` inside the workdir.
* **A git checkout.**  ``actions/checkout`` always delivers a repository;
  a plain export is not one, and ``doctor``'s run-provenance check
  rightly reports an identityless tree as ``missing``.  A tree handed in
  without ``.git`` is ``git init``-ed and committed in place, locally,
  with a throwaway identity and no remote.
* **An interpreter that carries setuptools.**  CI pins 3.11, whose
  environment ships it; ``python -m venv`` on 3.12+ does not (PEP 632).
  Two of the job's own test files drive setuptools to measure what the
  wheel would contain and refuse when it is absent, so on a 3.12+ box the
  replay went red on its own venv while the job it replays was fine.
  See :data:`VENV_SEEDS`.

The tree is mutated exactly as the job mutates its checkout (the
editable install writes ``gpuwm.egg-info/`` into it) plus that local
``git init``; to replay against a pristine snapshot, copy the snapshot
first and hand the copy in.

Usage:
    python tools/ci_test_replay.py <source-tree> [--workdir DIR] [--keep]
        [--python EXE]

Exit status is pytest's, so 0 means the job's test leg would have passed.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"

#: Seeded into the replay venv before anything else, because the CI
#: interpreter has them and a fresh venv on 3.12+ does not.
#:
#: ``build`` is the job's own first step.  ``setuptools`` is the one this
#: harness has to add back: ``setup-python`` 3.11 ships it (ensurepip
#: bundled it through 3.11), ``python -m venv`` on 3.12+ creates pip alone
#: (PEP 632), and two of the job's own test files --
#: ``tests/test_package_data_coverage.py`` and
#: ``tests/test_configs_are_packaged.py`` -- drive setuptools directly to
#: measure what the wheel would contain, REFUSING when it is missing
#: because a silent skip of a packaging gate reports green.  Without this
#: seed a replay on 3.12+ went red on its own venv while the job it
#: replays was fine, so the verdict described the harness rather than the
#: tree -- and this harness exists to be pointed at a public snapshot from
#: whatever interpreter the box has.
#: CI pins 3.11, whose environment ships setuptools; ``python -m venv``
#: on 3.12+ does not (PEP 632).  Pinned and joined by ``wheel`` because
#: a bare 3.14 venv sent tests/test_package_data_coverage.py and
#: tests/test_configs_are_packaged.py -- which call setuptools directly
#: to measure what the wheel would contain -- to 15 failures that said
#: nothing about the tree.
VENV_SEEDS = ("build", "setuptools>=77", "wheel")


def venv_python(venv_dir: Path) -> Path:
    """The interpreter inside a venv, on either platform."""

    return venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def bootstrap_venv(venv_dir: Path, *, python: str,
                   env: dict[str, str]) -> Path:
    """Create the replay venv, seeded like the CI interpreter, and return it.

    Separated from :func:`main` so the property that broke -- "the venv the
    replay builds can run the job's packaging tests" -- is reachable from a
    test without replaying a whole publish job.
    """

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    if python == sys.executable:
        venv.create(venv_dir, with_pip=True)
    else:
        subprocess.run([python, "-m", "venv", str(venv_dir)], check=True)
    interpreter = venv_python(venv_dir)
    run([str(interpreter), "-m", "pip", "install", *VENV_SEEDS],
        cwd=venv_dir, env=env,
        label="seed the venv with what CI's interpreter already has "
              f"({', '.join(VENV_SEEDS)})")
    return interpreter


def parse_test_job(workflow_text: str) -> tuple[str, list[str]]:
    """The ``-m`` marker expression and test-file list of the ``test`` job.

    Parsed from the one pytest invocation inside the job named ``test``,
    because a hand-maintained copy of that list is exactly the drift this
    script exists to rule out.
    """

    match = re.search(
        r"python -m pytest -q -m \"([^\"]+)\"((?:\s*\\?\s*tests/\S+\.py)+)",
        workflow_text)
    if not match:
        raise SystemExit(
            "cannot find the test job's pytest command in "
            f"{WORKFLOW}; the workflow's shape changed, update this parser")
    marker = match.group(1)
    files = re.findall(r"tests/\S+\.py", match.group(2))
    if len(files) < 5:
        raise SystemExit(
            f"parsed only {len(files)} test files out of {WORKFLOW}; "
            "that cannot be the whole job, update this parser")
    return marker, files


def run(argv: list[str], *, cwd: Path, env: dict[str, str],
        label: str) -> None:
    print(f"\n=== {label}\n    $ {' '.join(argv)}\n    (cwd {cwd})",
          flush=True)
    completed = subprocess.run(argv, cwd=str(cwd), env=env)
    if completed.returncode != 0:
        raise SystemExit(
            f"replay step failed (exit {completed.returncode}): {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tree", type=Path,
                        help="source tree to replay the test job against")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="where the venv and companion dists go "
                             "(default: a fresh temp dir)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the workdir instead of deleting it")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter to build the venv from "
                             "(CI pins 3.11; default: this one)")
    args = parser.parse_args()

    tree = args.tree.resolve()
    if not (tree / "pyproject.toml").is_file() \
            or not (tree / "gpuwm-data" / "pyproject.toml").is_file():
        raise SystemExit(f"{tree} is not a gpuwm source tree "
                         "(pyproject.toml or gpuwm-data/ missing)")

    marker, test_files = parse_test_job(WORKFLOW.read_text(encoding="utf-8"))
    print(f"replaying publish.yml test job against {tree}")
    print(f"  marker: -m {marker!r}")
    print(f"  files:  {len(test_files)}")

    workdir = args.workdir.resolve() if args.workdir else Path(
        tempfile.mkdtemp(prefix="gpuwm-ci-replay-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"  work:   {workdir}")

    venv_dir = workdir / "venv"

    # The whole CI step runs under this env block, so every motion below
    # gets it -- most relevantly the editable install, whose setup.py
    # refuses an unpinned tree without it.
    env = dict(os.environ)
    env["GPUWM_ALLOW_UNPINNED_WHEEL"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # The runner's home is clean; this box's is not.  ~/.gpuwm staging
    # from unrelated work must not leak into the replay's verdict.
    home = workdir / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)

    # ...but pip keeps the box's REAL cache.  With the home redirected,
    # pip's cache resolution loses its usual anchor (measured: a
    # ./pip/cache dropping in the tree, and a PyPI 502 blip failing a
    # replay that every earlier run had the bytes for).  Cache reuse
    # cannot change a test verdict; re-downloading 400 MB can fail one.
    if "PIP_CACHE_DIR" not in env:
        if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
            env["PIP_CACHE_DIR"] = str(
                Path(os.environ["LOCALAPPDATA"]) / "pip" / "Cache")
        else:
            env["PIP_CACHE_DIR"] = os.path.expanduser("~/.cache/pip")

    # actions/checkout delivers a git repository; a plain export handed
    # in gets one, locally, so run-provenance answers as it does in CI.
    if not (tree / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=str(tree), env=env,
                       check=True)
        subprocess.run(["git", "add", "-A"], cwd=str(tree), env=env,
                       check=True)
        subprocess.run(
            ["git", "-c", "user.name=ci-replay",
             "-c", "user.email=ci-replay@localhost",
             "commit", "-q", "-m", "ci replay snapshot"],
            cwd=str(tree), env=env, check=True)
        print("  tree had no .git; initialized a local repository in it")

    # Built here rather than above so it runs under the redirected HOME
    # and the recovered pip cache, like every other pip motion.
    python = bootstrap_venv(venv_dir, python=args.python, env=env)

    companion_dist = workdir / "companion-dist"
    run([str(python), "-m", "build", "--outdir", str(companion_dist),
         "gpuwm-data"],
        cwd=tree, env=env, label="build the companion from the tree")
    wheels = glob.glob(str(companion_dist / "*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one companion wheel, got {wheels}")
    run([str(python), "-m", "pip", "install", wheels[0]],
        cwd=tree, env=env, label="install the companion wheel")
    run([str(python), "-m", "pip", "install", "-e", ".[dev]"],
        cwd=tree, env=env, label="editable dev install of the tree")

    print(f"\n=== pytest -q -m {marker!r} ({len(test_files)} files)",
          flush=True)
    completed = subprocess.run(
        [str(python), "-m", "pytest", "-q", "-m", marker, *test_files],
        cwd=str(tree), env=env)

    if not args.keep and args.workdir is None and completed.returncode == 0:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"removed {workdir}")
    else:
        print(f"kept {workdir}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
