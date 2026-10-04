"""A native binary is reused across commits only when its build inputs are unchanged.

The release cut stopped recompiling twenty-nine binaries for a release that
changed one Python file: a binary built at an earlier commit is accepted
for the commit being released when every path
``gpuwm.bridge_assets.NATIVE_BUILD_INPUTS`` lists for its crate is the same
git object at both, and the earlier commit is an ancestor.  Two things can
break that and each has a test here:

* the table misses a file a crate's build reads, so a reused binary would
  carry a stale embedded copy of it (the terminal binary embeds
  ``gpuwm/tui_worker.py``); the discovery test re-derives every outside
  input from the Cargo manifests and the Rust sources and refuses one the
  table does not cover;
* the proof accepts something it should not: a changed input, a commit
  that is not an ancestor, a commit the clone does not have.
"""

from __future__ import annotations

import os
from pathlib import Path
import posixpath
import re
import subprocess

import pytest

from gpuwm import bridge_assets

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOTS = tuple(bridge_assets.NATIVE_BUILD_INPUTS)
MARKER = bridge_assets.SOURCE_REV_MARKER


def _root_of(path: str) -> str | None:
    for root in ROOTS:
        if path == root or path.startswith(root + "/"):
            return root
    return None


def _covered(path: str, declared: tuple[str, ...]) -> bool:
    return any(path == item or path.startswith(item + "/") for item in declared)


def _manifest_dir(directory: str) -> str:
    while directory and not (REPO_ROOT / directory / "Cargo.toml").is_file():
        directory = posixpath.dirname(directory)
    return directory


def _outside_references() -> dict[str, set[str]]:
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", *ROOTS],
        capture_output=True, check=True).stdout.decode().split("\0")
    found: dict[str, set[str]] = {root: set() for root in ROOTS}
    for name in filter(None, listed):
        if not name.endswith((".toml", ".rs")):
            continue
        root = _root_of(name)
        directory = posixpath.dirname(name)
        text = (REPO_ROOT / name).read_text(encoding="utf-8", errors="replace")
        candidates: list[str] = []
        if name.endswith("Cargo.toml"):
            candidates += [posixpath.join(directory, match) for match in
                           re.findall(r'path\s*=\s*"([^"]+)"', text)]
        if name.endswith(".rs"):
            candidates += [posixpath.join(directory, item) for item in
                           re.findall(r'#\[path\s*=\s*"([^"]+)"\]', text)]
            for body in re.findall(r"include_(?:str|bytes)!\s*\((.*?)\)\s*[;,)]",
                                   text, re.S):
                literals = re.findall(r'"([^"]*)"', body)
                if "CARGO_MANIFEST_DIR" in body:
                    relative = "".join(item for item in literals
                                       if item != "CARGO_MANIFEST_DIR")
                    candidates.append(_manifest_dir(directory) + "/"
                                      + relative.lstrip("/"))
                elif literals:
                    candidates.append(posixpath.join(directory, literals[0]))
            if name.endswith("build.rs"):
                candidates += [posixpath.join(directory, item) for item in
                               re.findall(r'"(\.\./[^"]+)"', text)]
        for candidate in candidates:
            path = posixpath.normpath(candidate)
            if path.startswith("..") or _root_of(path) == root:
                continue
            if (REPO_ROOT / path).exists():
                found[root].add(path)
    return found


def test_every_bundled_crate_declares_its_build_inputs():
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        declared = bridge_assets.NATIVE_BUILD_INPUTS.get(artifact.crate)
        assert declared, (artifact.name, artifact.crate)
        assert declared[0] == artifact.crate


def test_shared_preparation_resources_follow_native_dependency_closures():
    for crate in ("tools/grib1_bridge", "tools/rw_wps", "tools/rustwx", "tools/zarr_bridge"):
        assert "tools/preparation_resources.rs" in bridge_assets.NATIVE_BUILD_INPUTS[crate]


def test_the_table_covers_every_outside_input_the_sources_reach():
    references = _outside_references()
    missing = {}
    for root, paths in references.items():
        # Transitive: reaching into another root reaches its inputs too.
        closure = set(paths)
        for path in paths:
            other = _root_of(path)
            if other is not None:
                closure |= references[other]
        gaps = sorted(path for path in closure
                      if not _covered(path, bridge_assets.NATIVE_BUILD_INPUTS[root]))
        if gaps:
            missing[root] = gaps
    assert not missing, (
        "a native crate reads these tracked paths and NATIVE_BUILD_INPUTS "
        "does not list them, so a reused binary could embed a stale copy",
        missing)
    # The discovery is not vacuous: the terminal embeds its Python worker.
    assert "gpuwm/tui_worker.py" in references["tools/arwen-tui"]


def test_no_rust_source_embeds_from_an_excluded_path():
    """An exclusion is data the build never reads, or it is a stale-copy hole."""
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", *ROOTS],
        capture_output=True, check=True).stdout.decode().split("\0")
    excluded = [path for paths in bridge_assets.NATIVE_BUILD_INPUT_EXCLUSIONS.values()
                for path in paths]
    embedding = []
    for name in filter(None, listed):
        if not name.endswith(".rs") or "/vendor/" in name:
            continue
        text = (REPO_ROOT / name).read_text(encoding="utf-8", errors="replace")
        for body in re.findall(r"include_(?:str|bytes)!\s*\((.*?)\)\s*[;,)]", text, re.S):
            literals = re.findall(r'"([^"]*)"', body)
            if not literals or "CARGO_MANIFEST_DIR" in body:
                continue
            path = posixpath.normpath(posixpath.join(posixpath.dirname(name), literals[0]))
            if _covered(path, tuple(excluded)):
                embedding.append((name, path))
    assert not embedding, embedding


@pytest.fixture()
def history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, env=env,
                              capture_output=True, text=True).stdout.strip()

    def commit(files, message):
        for path, text in files.items():
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        git("add", "-A")
        git("commit", "-q", "-m", message)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    first = commit({"tools/grib1_bridge/src/lib.rs": "a", "gpuwm/other.py": "1",
                    "tools/arwen-tui/src/main.rs": "t", "gpuwm/tui_worker.py": "w"}, "one")
    python_only = commit({"gpuwm/other.py": "2"}, "python only")
    worker = commit({"gpuwm/tui_worker.py": "w2"}, "the embedded worker")
    notice = commit({"tools/rustwx/assets/basemap/NOTICE.txt": "n2"}, "notice refresh")
    git("checkout", "-q", "-b", "side", first)
    side = commit({"gpuwm/side.py": "s"}, "side")
    return dict(repo=repo, first=first, python_only=python_only, worker=worker, side=side,
                notice=notice)


def test_a_packed_data_file_does_not_force_a_rebuild(history):
    assert bridge_assets.native_input_difference(
        history["repo"], "tools/rustwx", history["first"], history["notice"]) is None


def test_unchanged_inputs_at_an_ancestor_are_interchangeable(history):
    repo = history["repo"]
    assert bridge_assets.native_input_difference(
        repo, "tools/grib1_bridge", history["first"], history["worker"]) is None
    payload = b"junk " + MARKER + history["first"].encode() + b" tail"
    bridge_assets.verify_source_revision(
        payload, expected=history["worker"], label="grib1_bridge",
        equivalent=lambda built: bridge_assets.native_input_difference(
            repo, "tools/grib1_bridge", built, history["worker"]))


def test_a_changed_input_refuses_reuse_and_names_the_path(history):
    repo = history["repo"]
    reason = bridge_assets.native_input_difference(
        repo, "tools/arwen-tui", history["python_only"], history["worker"])
    assert reason is not None and "gpuwm/tui_worker.py" in reason
    payload = b"junk " + MARKER + history["python_only"].encode() + b" tail"
    with pytest.raises(bridge_assets.BridgeAssetError, match="tui_worker"):
        bridge_assets.verify_source_revision(
            payload, expected=history["worker"], label="arwen-tui",
            equivalent=lambda built: bridge_assets.native_input_difference(
                repo, "tools/arwen-tui", built, history["worker"]))


def test_a_commit_off_the_released_history_is_refused(history):
    reason = bridge_assets.native_input_difference(
        history["repo"], "tools/grib1_bridge", history["side"], history["worker"])
    assert reason is not None and "not an ancestor" in reason


def test_a_commit_the_clone_does_not_have_is_refused(history):
    reason = bridge_assets.native_input_difference(
        history["repo"], "tools/grib1_bridge", "ef56" * 10, history["worker"])
    assert reason is not None


def test_without_the_proof_a_foreign_stamp_is_still_stale(history):
    payload = b"junk " + MARKER + history["first"].encode() + b" tail"
    with pytest.raises(bridge_assets.BridgeAssetError, match="stale build"):
        bridge_assets.verify_source_revision(
            payload, expected=history["worker"], label="grib1_bridge")


def _workflow_depth():
    """The packet preflight's own reader, loaded from its file (stdlib only)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cut_workflow_depth", REPO_ROOT / "tools/release/workflow_depth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_shallow_clone_cannot_prove_reuse(history, tmp_path):
    """The breakage the full-history rule below prevents, reproduced."""
    source = history["repo"].resolve().as_uri()
    released = history["worker"]
    for name, depth in (("shallow", ["--depth", "1"]), ("full", [])):
        clone = tmp_path / name
        subprocess.run(["git", "init", "-q", str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "-c", "protocol.version=2", "fetch", "-q", *depth,
                        source, released], check=True)
        answer = bridge_assets.native_input_difference(
            clone, "tools/grib1_bridge", history["first"], released)
        if name == "shallow":
            assert answer is not None and "not an ancestor" in answer
        else:
            assert answer is None


def _verifier_jobs_by_yaml(path):
    import yaml
    VERIFIER_CALLS = _workflow_depth().VERIFIER_CALLS
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    found = {}
    for name, job in workflow.get("jobs", {}).items():
        steps = job.get("steps", [])
        if not any(call in step.get("run", "") for step in steps for call in VERIFIER_CALLS):
            continue
        checkouts = [step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")]
        if not checkouts:
            found[name] = None
            continue
        found[name] = int((checkouts[0].get("with") or {}).get("fetch-depth", 1))
    return found


@pytest.mark.parametrize("workflow", sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")),
                         ids=lambda path: path.name)
def test_every_job_that_runs_the_stamp_verifier_checks_out_full_history(workflow):
    """A reused binary is proved by ancestry; a shallow checkout refuses it after the tag is public."""
    verifier_jobs = _workflow_depth().verifier_jobs
    by_yaml = _verifier_jobs_by_yaml(workflow)
    assert verifier_jobs(workflow.read_text(encoding="utf-8")) == by_yaml, \
        "the packet preflight reads this workflow differently from a YAML parser"
    shallow = {job: depth for job, depth in by_yaml.items() if depth != 0}
    assert not shallow, f"{workflow.name}: these jobs run the stamp verifier without full history: {shallow}"


def test_the_publication_workflow_is_read_as_the_preflight_reads_it():
    verifier_jobs = _workflow_depth().verifier_jobs
    jobs = verifier_jobs((REPO_ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8"))
    assert {"prepare", "qualify"} <= set(jobs)
    text = ("jobs:\n  smoke:\n    steps:\n      - uses: actions/checkout@abc\n"
            "        with:\n          persist-credentials: false\n"
            "      - run: python tools/promote_prepared_release.py smoke --repo x\n")
    assert verifier_jobs(text) == {"smoke": 1}
