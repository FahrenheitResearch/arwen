#!/usr/bin/env python3
"""Run one job of a workflow file on this box, step by step, as its YAML says.

    publish_test_job_289.py --tree CLONE --work DIR --python PY311
        [--workflow .github/workflows/publish.yml] [--job test]
        [--rustup-home DIR] [--cargo-home DIR] [--pip-cache DIR]

Job P of ci-repro-289.sh: publish.yml's `test` job, in a fresh clone of the
public tree, before anything is pushed.

Breakage it prevents: v2.8.9. The chain reproduced every ci.yml job and never
this one. After main and the tag were public and the draft release made, run
38124857348 failed in `test` on five tests of tests/test_table_fetch.py
("rw_fetch is missing"): the job built no natives, and every box its file
list had been run on carried a built tools/rustwx/target.

How it stays the job and not a copy of it:
  * Every step is read from the workflow file in the clone. `run` steps are
    executed with the job's and the step's `env`, GITHUB_ENV and GITHUB_PATH
    carried between steps. Nothing is restated here: no install line, no test
    list, no cargo command.
  * A step this cannot reproduce is a refusal, never a skip: an action other
    than checkout, setup-python and upload-artifact, an `if:`, or an
    expression other than `${{ runner.temp }}`.
  * The environment is built from nothing (no variable of the caller is
    inherited), HOME and RUNNER_TEMP are new folders under --work, and the
    interpreter is a new venv of --python. A clone that already holds a cargo
    target folder or an rw_fetch, or a PATH that reaches an rw_fetch or a
    gpuwm, is refused before the first step: the only natives the tests can
    find are the ones the job's own steps build.

What it does not reproduce: the hosted runner image (its preinstalled tools
and their versions), GitHub's checkout (ci-repro-289.sh clones at depth 1
without tags), and artifact upload (only "the path exists" is checked).
Exit status: 0 when every step passed, else the failing step's status.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

EXPRESSION = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
SYSTEM_PATH = ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]


class Unreproducible(SystemExit):
    def __init__(self, why: str) -> None:
        super().__init__(f"REFUSED: {why}")


def say(text: str) -> None:
    print(text, flush=True)


def render(value, context: dict) -> str:
    def one(match):
        name = match.group(1)
        if name not in context:
            raise Unreproducible(f"expression ${{{{ {name} }}}} is not one this reproduction evaluates")
        return context[name]
    return EXPRESSION.sub(one, str(value))


def env_value(value) -> str:
    # YAML booleans reach a step's environment as GitHub writes them.
    return {True: "true", False: "false"}.get(value, value) if isinstance(value, bool) else str(value)


def read_env_file(path: Path) -> dict:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "<<" in line.split("=", 1)[0] or "=" not in line:
            raise Unreproducible(f"GITHUB_ENV line this reproduction cannot read: {line!r}")
        name, value = line.split("=", 1)
        found[name] = value
    return found


def refuse_existing_natives(tree: Path, path: list[str]) -> None:
    # Vendored cc has src/target containing Rust source, not cargo output.
    built = [str(p) for p in [tree / "target", *tree.glob("tools/*/target")] if p.exists()]
    built += [str(p.parent) for p in tree.rglob(".rustc_info.json")]
    built += [str(p) for name in ("rw_fetch", "rw_fetch.exe") for p in tree.rglob(name) if p.is_file()]
    if built:
        raise Unreproducible("the clone already holds built natives: " + ", ".join(built[:5]))
    for name in ("rw_fetch", "gpuwm"):
        reached = shutil.which(name, path=os.pathsep.join(path))
        if reached:
            raise Unreproducible(f"PATH reaches {reached} before the job has run a step")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--python", required=True, help="the interpreter setup-python would install")
    parser.add_argument("--workflow", default=".github/workflows/publish.yml")
    parser.add_argument("--job", default="test")
    parser.add_argument("--rustup-home", default=None)
    parser.add_argument("--cargo-home", default=None)
    parser.add_argument("--pip-cache", default=None)
    parser.add_argument("--cargo-jobs", type=int, default=16, help="CPU build cap; no GPU is used")
    args = parser.parse_args(argv)

    tree, work = args.tree.resolve(), args.work.resolve()
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        raise Unreproducible(f"{tree} is not a git checkout")
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain"], capture_output=True, text=True).stdout
    if dirty.strip():
        raise Unreproducible(f"{tree} is not a clean checkout:\n{dirty[:400]}")
    if work.exists() and any(work.iterdir()):
        raise Unreproducible(f"{work} is not empty")
    for name in ("home", "runner-temp", "tmp", "steps"):
        (work / name).mkdir(parents=True)

    document = yaml.safe_load((tree / args.workflow).read_text(encoding="utf-8"))
    job = document["jobs"][args.job]
    if job.get("runs-on") != "ubuntu-24.04" or "strategy" in job or "if" in job:
        raise Unreproducible("this driver supports an unconditional ubuntu-24.04 job without a matrix only")
    if document.get("defaults") or job.get("defaults") or args.cargo_jobs < 1:
        raise Unreproducible("workflow/job defaults or invalid CPU build cap")
    steps = job["steps"]
    context = {"runner.temp": str(work / "runner-temp")}

    path = [str(work / "venv" / "bin")]
    if args.cargo_home:
        path.append(str(Path(args.cargo_home) / "bin"))
    path += SYSTEM_PATH
    refuse_existing_natives(tree, path)

    files = {name: work / "runner-temp" / f"_{name.lower()}" for name in
             ("GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_PATH", "GITHUB_STEP_SUMMARY")}
    base = {"HOME": str(work / "home"), "TMPDIR": str(work / "tmp"), "LANG": "C.UTF-8",
            "RUNNER_TEMP": context["runner.temp"], "RUNNER_OS": "Linux", "GITHUB_WORKSPACE": str(tree),
            "CI": "true", "GITHUB_ACTIONS": "true", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CARGO_BUILD_JOBS": str(args.cargo_jobs)}
    base.update({name: str(target) for name, target in files.items()})
    for name, value in (("RUSTUP_HOME", args.rustup_home), ("CARGO_HOME", args.cargo_home),
                        ("PIP_CACHE_DIR", args.pip_cache)):
        if value:
            base[name] = value
    declared = {}
    for scope in (document.get("env") or {}, job.get("env") or {}):
        declared.update({name: render(env_value(value), context) for name, value in scope.items()})
    exported: dict = {}

    say(f"job {args.job} of {args.workflow} at {head.stdout.strip()} "
        f"({subprocess.run(['git', '-C', str(tree), 'rev-list', '--count', 'HEAD'], capture_output=True, text=True).stdout.strip()} "
        f"commit(s) in the clone), {len(steps)} steps")
    say(f"runs-on {job.get('runs-on')}; here {os.uname().sysname} {os.uname().release}; work {work}")
    (work / "IDENTITY.json").write_text(json.dumps({
        "commit": head.stdout.strip(), "workflow": args.workflow, "job": args.job,
        "native_paths_before": [], "inherited_environment": False,
        "runner": job["runs-on"], "steps": len(steps)}, indent=2) + "\n", encoding="utf-8")

    python_ready = False
    for number, step in enumerate(steps, 1):
        label = step.get("name") or step.get("uses") or step.get("id") or "run"
        say(f"\n=== step {number}/{len(steps)}: {label}")
        unknown = set(step) - {"name", "id", "uses", "with", "run", "env", "shell", "working-directory"}
        if unknown:
            raise Unreproducible(f"step {label!r} uses {sorted(unknown)}, which this reproduction does not evaluate")
        settings = step.get("with") or {}
        action = step.get("uses")
        if action:
            kind = action.split("@", 1)[0]
            if kind == "actions/checkout":
                if set(settings) - {"persist-credentials", "fetch-depth"}:
                    raise Unreproducible(f"checkout with {sorted(settings)}")
                say(f"    the clone stands in for the checkout (fetch-depth {settings.get('fetch-depth', 1)} in the job)")
            elif kind == "actions/setup-python":
                wanted = str(settings["python-version"])
                subprocess.run([args.python, "-m", "venv", str(work / "venv")], check=True, env=base | {"PATH": os.pathsep.join(SYSTEM_PATH)})
                got = subprocess.run([str(work / "venv/bin/python"), "-c",
                                      "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
                                     capture_output=True, text=True, check=True).stdout.strip()
                if not (got == wanted or got.startswith(wanted + ".")):
                    raise Unreproducible(f"the job wants Python {wanted}; --python is {got}")
                say(f"    new venv of Python {got} at {work / 'venv'}")
                python_ready = True
            elif kind == "actions/upload-artifact":
                target = Path(render(settings["path"], context))
                target = target if target.is_absolute() else tree / target
                if not target.exists():
                    say(f"STEP FAILED: {label}: {target} does not exist")
                    return 1
                say(f"    {target} exists (nothing is uploaded here)")
            else:
                raise Unreproducible(f"action {action} is not one this reproduction can stand in for")
            continue
        if "run" not in step:
            raise Unreproducible(f"step {label!r} has neither uses nor run")
        if not python_ready:
            raise Unreproducible(f"step {label!r} runs before any setup-python step")
        shell = step.get("shell")
        if shell not in (None, "bash"):
            raise Unreproducible(f"shell {shell!r}")
        script = work / "steps" / f"{number:02d}.sh"
        script.write_text(render(step["run"], context), encoding="utf-8", newline="\n")
        environment = dict(base)
        environment["PATH"] = os.pathsep.join(path)
        environment.update(declared)
        environment.update(exported)
        environment.update({name: render(env_value(value), context) for name, value in (step.get("env") or {}).items()})
        for name in sorted(set(environment) - set(base)):
            say(f"    env {name}={environment[name]}")
        for target in files.values():
            target.write_text("", encoding="utf-8")
        cwd = tree / render(step.get("working-directory", "."), context)
        command = ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)] if shell else ["bash", "-e", str(script)]
        status = subprocess.run(command, cwd=str(cwd), env=environment).returncode
        if status != 0:
            say(f"\nSTEP FAILED: {label} (exit {status}); job {args.job} of {args.workflow} FAILED")
            return status
        exported.update(read_env_file(files["GITHUB_ENV"]))
        added = [line.strip() for line in files["GITHUB_PATH"].read_text(encoding="utf-8").splitlines() if line.strip()]
        path[:0] = reversed(added)
        outputs = files["GITHUB_OUTPUT"].read_text(encoding="utf-8").strip()
        if outputs:
            say(f"    outputs: {outputs}")
    say(f"\njob {args.job} of {args.workflow}: all {len(steps)} steps PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
