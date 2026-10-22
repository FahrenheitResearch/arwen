"""Replay the actual publish.yml test job in a fresh Linux checkout.

No host binaries or caller environment are accepted. Workflow run steps,
including native builds, execute in order under a new home and Python venv.
The supplied Python must match setup-python. Unsupported job shapes refuse.
The hosted image, credentials and upload services are outside this rehearsal.
Legacy parser and venv helpers remain for callers; main uses the step runner.
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
    """Execute the tree's actual job, including its native build steps."""
    sys.path.insert(0, str(REPO_ROOT))
    from tools.workflow_job_replay import main as replay

    parser = argparse.ArgumentParser(description="Replay publish.yml test in a clean Linux checkout")
    parser.add_argument("tree", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--keep", action="store_true", help="receipts are always retained")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--release-proof")
    parser.add_argument("--release-version")
    parser.add_argument("--publication-commit")
    parser.add_argument("--rustup-home")
    parser.add_argument("--cargo-home")
    parser.add_argument("--pip-cache")
    args = parser.parse_args()
    work = args.workdir or Path(tempfile.mkdtemp(prefix="gpuwm-ci-replay-"))
    options = ["--tree", str(args.tree), "--work", str(work), "--python", args.python]
    for name in ("rustup_home", "cargo_home", "pip_cache", "release_proof", "release_version", "publication_commit"):
        value = getattr(args, name)
        if value:
            options.extend(["--" + name.replace("_", "-"), value])
    return replay(options)


if __name__ == "__main__":
    raise SystemExit(main())
