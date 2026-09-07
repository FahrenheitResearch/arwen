"""The lanes in `.github/workflows/ci.yml` still run what they claim to.

THE BREAKAGE THIS PREVENTS
--------------------------
Two of them, and the second is the one that lasts.

1. At a70ade37 `.github/workflows/` held exactly one file, `publish.yml`,
   whose `test` job runs a hand-typed list of 17 packaging/CLI/asset-fetch
   test files -- 452 of the tree's 14,711 test functions by AST count, 3.1%,
   and not one of them exercises dynamics, physics, advection, the acoustic
   solver, the LSM, data assimilation, nesting or restart round-trips.  The
   `-m "not gpu and not slow and not network"` filter that job passes would
   have admitted ~12,500 CPU-runnable test functions; the filename list was
   the only thing that cut it to 452 (audit xc-06-01).

2. Nothing gated that list.  `tools/ci_test_replay.py:123` already parses it,
   and the only consumer of the parse checks that the replay venv seeds
   setuptools -- no test anywhere asserts anything about the list's content,
   size, or non-shrinkage.  Which is F20 of the project's own fault-injection
   audit ("delete ONE line and the suite stops running and nothing says so"),
   closed for the battery lists by `list_census.json` and left open for the
   one list a public contributor's PR is judged by (audit xc-06-08).

So the lane being wide is worth nothing on its own; what is worth something
is that narrowing it has to be a deliberate, reviewable edit to a gated
property.  These tests are those properties: the CPU lane names DIRECTORIES,
the oracle lane reads its decks from the manifest rather than from a copy,
and the GPU lane throws the switch the bundle gates have always documented
and nothing ever threw.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tools.battery.no_silent_skip import parse_manifest
from tools.ci_test_replay import parse_test_job

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
PUBLISH = REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml"
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "must_run_gates.txt"
SHARDS = REPOSITORY_ROOT / "tools" / "battery" / "gpu_shard_files.txt"

#: What "reads this switch" means, pinned as a literal for the reason
#: LEG_MARKER_EXPRESSION below is pinned: the `gpu` lane greps the tree with
#: this rule to build its argument list, and a gate that read the rule out of
#: the thing it is judging could not notice the rule changing.  It is the
#: whole of the coupling between the workflow and this file -- everything
#: else here is derived from the tree.
SWITCH_READER_PATTERN = r"os\.environ.*"

#: The marker expression every CPU battery leg runs under, pinned as a
#: literal for the reason tests/test_battery_list_census.py:53-58 pins the
#: same string: a gate that reads the expression out of the thing it is
#: judging cannot notice the expression changing.
LEG_MARKER_EXPRESSION = "not gpu and not slow and not network"

#: Static Rust parity now exercises the shipped implementation built in CI.
CPU_LANE_MARKER_EXPRESSION = LEG_MARKER_EXPRESSION + " and not static_platform_qualification"


def _lines() -> list[str]:
    return WORKFLOW.read_text(encoding="utf-8").splitlines()


def _job(name: str) -> list[str]:
    """The lines of one job, header to the next job header.

    Line-based rather than through a YAML parser, the way
    tests/test_publish_workflow_state_machine.py:33-56 reads the other
    workflow: PyYAML is in no dependency table, and a gate that needs an
    undeclared import skips on the box that most needs it.
    """

    header = re.compile(r"^  [a-z_]+:$")
    start = None
    for index, line in enumerate(_lines()):
        if line == f"  {name}:":
            start = index
            continue
        if start is not None and header.match(line):
            return _lines()[start:index]
    assert start is not None, (
        f"{WORKFLOW} has no job named {name!r}.  Renaming a lane is allowed; "
        "renaming it here in the same commit is the price.")
    return _lines()[start:]


def _step_run(job: str, step: str) -> str:
    """The literal shell body of one named step of one job."""

    lines = _job(job)
    for index, line in enumerate(lines):
        if line.strip() != f"- name: {step}":
            continue
        for offset in range(index + 1, len(lines)):
            if lines[offset].strip().startswith("- "):
                break
            if lines[offset].strip() == "run: |":
                body: list[str] = []
                for raw in lines[offset + 1:]:
                    if raw.strip() and not raw.startswith(" " * 10):
                        break
                    body.append(raw)
                return "\n".join(body)
            if lines[offset].strip().startswith("run:"):
                return lines[offset].split("run:", 1)[1].strip()
    raise AssertionError(f"job {job!r} has no step named {step!r}")


def test_the_repository_has_a_lane_that_runs_the_model_s_own_code() -> None:
    """The headline, and the only test here that is about existence."""

    assert WORKFLOW.is_file(), (
        f"{WORKFLOW} is missing.  Without it the repository's only workflow "
        "is publish.yml, whose `test` job is 17 packaging files -- 3.1% of "
        "the test functions in the tree and 0% of the science -- and a green "
        "check named `test` on a pull request means the wheel still builds.")
    text = "\n".join(_lines())
    assert re.search(r"^on:\n(  push:\n)?(  pull_request:\n?)?", text,
                     re.MULTILINE), text[:400]
    assert "  push:" in text and "  pull_request:" in text, (
        "the lane must run on push AND on pull_request; a lane that runs "
        "only on one of them leaves the other unmeasured")


def test_the_cpu_lane_is_not_a_hand_typed_file_list() -> None:
    """THE PROPERTY THAT KEEPS xc-06-01 CLOSED.

    A directory argument means a test file joins the lane by existing.  A
    filename list means it joins the lane when somebody remembers, which is
    the state this workflow was added to leave -- and the state a
    well-meaning "trim CI wall clock" commit restores in one edit.
    """

    body = _step_run("cpu", "the CPU-runnable estate")
    named = re.findall(r"\btests?/[\w./-]*test_[\w.-]+\.py", body)
    assert not named, (
        f"the CPU lane names individual test files ({named}).  That is the "
        "shape publish.yml's `test` job has and the reason it runs 3.1% of "
        "the tree: a list only grows when somebody remembers.  Pass the "
        "testpaths directories and let the marker expression do the "
        "selecting.")
    for directory in ("tests", "tilestream"):
        assert re.search(rf"(^|\s){directory}(\s|$)", body, re.MULTILINE), (
            f"the CPU lane does not run {directory}/, which is one of "
            "pyproject.toml's two testpaths")


def test_the_cpu_lane_removes_only_what_it_declares() -> None:
    """Narrowing the lane must be an edit to a pinned string, not a drift.

    The failure this refuses is the cheap one: append `and not slow_acceptance
    and not <whatever went red today>` to the workflow, and the lane silently
    stops measuring a class nobody re-reads the YAML to notice.
    """

    body = _step_run("cpu", "the CPU-runnable estate")
    expressions = re.findall(r'-m "([^"]+)"', body)
    assert expressions == [CPU_LANE_MARKER_EXPRESSION], (
        f"the CPU lane's marker expression is {expressions}; this gate pins "
        f"it to {CPU_LANE_MARKER_EXPRESSION!r}.  Changing it is allowed and "
        "changing this line in the same commit is how the change gets read.")


def test_the_cpu_lane_covers_everything_the_packaging_job_covers() -> None:
    """The two workflows compose rather than compete.

    publish.yml keeps its 17-file packaging gate -- it is the root of the
    publish chain and seven test files parse that file by hand.  What must
    stay true is that the wide lane is a superset: every file the narrow one
    names lives under a directory the wide one runs.
    """

    _marker, packaging = parse_test_job(PUBLISH.read_text(encoding="utf-8"))
    body = _step_run("cpu", "the CPU-runnable estate")
    for name in packaging:
        root = name.split("/", 1)[0]
        assert re.search(rf"(^|\s){root}(\s|$)", body, re.MULTILINE), (
            f"publish.yml's test job runs {name}, and the CPU lane does not "
            f"run {root}/ at all")


def test_the_oracle_lane_reads_its_decks_from_the_manifest() -> None:
    """A second copy of the deck list is the defect, not the convenience.

    ``tools/battery/must_run_gates.txt`` is what
    ``tools/battery/no_silent_skip.py`` enforces against and what
    ``tests/test_must_run_gates.py`` gates.  A list retyped into the YAML
    would go stale in exactly the direction nothing measures.
    """

    body = _step_run("oracles", "bitwise oracle decks whose fixtures ship, "
                                "skips forbidden")
    assert "must_run_gates.txt" in body, body
    assert "tools.battery.no_silent_skip" in body, (
        "the oracle lane does not arm tools/battery/no_silent_skip.py, so a "
        "deck that skips every one of its tests exits 0 -- which is the "
        "RRTMG longwave shape this lane exists to make impossible")
    named = re.findall(r"\btests/test_[\w.-]+\.py", body)
    assert not named, (
        f"the oracle lane names decks literally ({named}) instead of reading "
        "tools/battery/must_run_gates.txt")


def test_the_oracle_lane_does_not_deselect_slow() -> None:
    """The one committed-fixture radiation oracle is `slow`-marked.

    ``tests/test_rrtmg_sw_oracle.py`` ships 13 MB of WRF v4.6.1 fixtures and
    carries ``pytestmark = pytest.mark.slow``, so the battery's own
    ``-m "... and not slow"`` is precisely what keeps it off every leg.  A
    lane that inherits that term inherits the hole.
    """

    body = _step_run("oracles", "bitwise oracle decks whose fixtures ship, "
                                "skips forbidden")
    expressions = re.findall(r'-m "([^"]+)"', body)
    assert expressions, body
    for expression in expressions:
        assert "not slow" not in expression, (
            f"the oracle lane deselects slow ({expression!r}), which "
            "deselects tests/test_rrtmg_sw_oracle.py -- the deck this lane "
            "was built around")


@pytest.mark.parametrize("entry", sorted(parse_manifest(
    MANIFEST.read_text(encoding="utf-8"))), ids=lambda value: value)
def test_every_must_run_gate_is_reachable_from_the_oracle_lane(
        entry: str) -> None:
    """Derived, not typed: the lane's arguments ARE the manifest's entries."""

    assert (REPOSITORY_ROOT / entry).is_file()
    body = _step_run("oracles", "bitwise oracle decks whose fixtures ship, "
                                "skips forbidden")
    assert "parse_manifest" in body, (
        "the oracle lane no longer builds its argument list with the "
        f"manifest's own parser, so {entry} is not provably on it")


def test_the_import_lane_sweeps_the_whole_package() -> None:
    """48% of package LOC was not IMPORTED by anything CI ran (xc-06-01)."""

    body = _step_run("imports", "import every module the wheel ships")
    assert "tools.import_sweep" in body, body


def test_there_is_a_gpu_lane_and_it_is_marked_for_a_self_hosted_runner() -> None:
    """Wired and visible even though nothing in this repository runs it.

    An absent job and an unrun job read the same on a dashboard and are not
    the same thing: one of them names the runner it wants, the shard it would
    run and the switch it would throw.
    """

    job = "\n".join(_job("gpu"))
    assert "self-hosted" in job, job[:400]
    assert "gpu_shard_files.txt" in job, (
        "the GPU lane does not name tools/battery/gpu_shard_files.txt, so it "
        "is a job with no leg")
    assert "SHARD-2-BEGIN" in job, (
        "the GPU lane must split the manifest on its own marker the way "
        "tests/test_gpu_shard_manifest.py does")
    second = _step_run("gpu", "GPU shard 2")
    assert "split('SHARD-2-BEGIN', 1)[1]" in second
    assert 'python -m pytest -q -p no:cacheprovider "${shard2[@]}"' in second


def test_the_gpu_lane_throws_the_switch_the_bundle_gates_document() -> None:
    """xc-06-04, in one assertion.

    ``tests/test_real74_wrfinput_mass_gate.py:62-64`` says "assembly/
    ratification runs set GPUWM_REQUIRE_CASE_GATES=1, which converts the
    environmental skip into a hard failure".  At a70ade37 a tree-wide grep for
    that name returned five hits, all in the two test files that READ it: no
    runner, no script and no workflow set it, so the switch that makes the
    WRF-reference gates binding was never thrown by anything the repository
    ships.  It is thrown here, on the one lane that can carry the bundle.
    """

    gpu = "\n".join(_job("gpu"))
    assert 'GPUWM_REQUIRE_CASE_GATES: "1"' in gpu, (
        "no lane sets GPUWM_REQUIRE_CASE_GATES, so tests/test_horiz.py and "
        "tests/test_real74_wrfinput_mass_gate.py skip their bundle gates "
        "everywhere and the comments promising otherwise are unbacked")
    assert "GPUWM_TEST_WRF74_BUNDLE" in gpu, (
        "the lane that requires the case gates must also say where the "
        "bundle comes from, or it fails for the wrong reason")

    assignment = re.compile(r"^\s*GPUWM_REQUIRE_CASE_GATES\s*:")
    for lane in ("syntax", "imports", "cpu", "oracles"):
        # Assignments, not mentions: the CPU lane's comment names the variable
        # precisely so a reader looking for it finds the reason it is absent.
        assert not [line for line in _job(lane) if assignment.match(line)], (
            f"the {lane} lane runs on a hosted runner that cannot hold the "
            "~50 GB WRF v4.6.1 reference bundle, so requiring the case gates "
            "there makes the lane red on every push for a reason no "
            "contributor can fix.  The CPU lane says so in a comment; if "
            "that changes, change the comment too")


def _gpu_declared_switches() -> list[str]:
    """The environment names the `gpu` job declares, in file order."""

    lines = _job("gpu")
    for index, line in enumerate(lines):
        if line == "    env:":
            break
    else:
        raise AssertionError("the gpu job declares no env: block")
    declared = []
    for line in lines[index + 1:]:
        match = re.match(r"^      ([A-Z][A-Z0-9_]*):", line)
        if not match:
            break
        declared.append(match.group(1))
    return declared


def _shard_one() -> list[str]:
    """Shard 1 of the GPU manifest, split where the lane splits it."""

    entries = []
    for raw in SHARDS.read_text(encoding="utf-8").splitlines():
        if "SHARD-2-BEGIN" in raw:
            break
        line = raw.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def _reader_files(variable: str) -> list[str]:
    """Test files that READ ``variable``, by the lane's own rule."""

    pattern = re.compile(SWITCH_READER_PATTERN + variable)
    here = pathlib.Path(__file__).resolve()
    return sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "tests").glob("test_*.py")
        # This file quotes the reader idiom in a docstring, and a gate that
        # counts itself as the evidence for its own claim is the shape being
        # refused three lines further down.
        if path.resolve() != here
        and pattern.search(path.read_text(encoding="utf-8")))


def test_the_gpu_lane_runs_a_file_that_reads_every_switch_it_declares() -> None:
    """xc-06-04's second half.  A DECLARED switch is not a THROWN one.

    The first version of this job set GPUWM_REQUIRE_CASE_GATES and
    GPUWM_TEST_WRF74_BUNDLE and then ran only shard 1 of
    tools/battery/gpu_shard_files.txt -- 38 entries, and measured, not one of
    them reads either name.  So the switch was declared by a lane that never
    executed a line which could see it, and the only test guarding it asserted
    that the assignment appeared in the job text.  That is this audit's own
    thesis -- the absence of a measurement scored as the presence of agreement
    -- reappearing inside a fix for it.

    This asserts the effect instead: for every switch the job declares, some
    file the job RUNS reads it.  What the job runs is derived here from the
    same two sources the YAML derives it from, so adding a lane step or moving
    a reader onto a shard both keep it true without an edit.
    """

    job = "\n".join(_job("gpu"))
    declared = _gpu_declared_switches()
    assert declared, job[:400]

    executed = set(_shard_one())
    for variable in declared:
        if SWITCH_READER_PATTERN + variable in job:
            executed.update(_reader_files(variable))

    for variable in declared:
        readers = _reader_files(variable)
        assert readers, (
            f"the gpu lane declares {variable} and no file under tests/ "
            "reads it; either every reader was renamed or the declaration is "
            "dead")
        assert set(readers) & executed, (
            f"the gpu lane declares {variable} and runs no file that reads "
            f"it.  Its readers are {readers}; what the lane runs is shard 1 "
            "of tools/battery/gpu_shard_files.txt plus whatever its steps "
            "derive from the tree.  A switch set on a job that executes "
            "nothing which can see it is a comment with a YAML key in front "
            "of it")


def test_the_gpu_lane_refuses_a_switch_that_points_nowhere() -> None:
    """An unset `vars.X` reaches the job as the EMPTY STRING, not as absent.

    Every reader is ``Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
    <default>))``, and an empty value does not fall back to the default -- it
    replaces it with ``Path("")``, which is ``Path(".")``, which is the
    checkout.  Paired with GPUWM_REQUIRE_CASE_GATES=1 the case gates would
    then run against this repository and fail for a missing met_em file: a
    true statement about the wrong directory, and the most expensive kind of
    red to read.
    """

    body = _step_run("gpu", "the bundle the case gates require is named and "
                            "present")
    assert '-z "$GPUWM_TEST_WRF74_BUNDLE"' in body, body
    assert "exit 1" in body, (
        "the lane notices the empty bundle variable and carries on; the "
        f"check has to be fatal to be a check: {body}")
    assert 'test -d "$GPUWM_TEST_WRF74_BUNDLE"' in body, (
        "a named bundle that is not a directory fails just as usefully at "
        f"the door as twenty minutes into the shard: {body}")


def test_every_action_is_pinned_by_commit() -> None:
    """The same discipline publish.yml already keeps, on the new file."""

    unpinned = [line.strip() for line in _lines()
                if "uses:" in line and "uses: ./" not in line
                and not re.search(r"@[0-9a-f]{40}\b", line)]
    assert not unpinned, unpinned


@pytest.mark.parametrize("lane", ["imports", "cpu", "oracles", "native"])
def test_hosted_lanes_cover_both_supported_operating_systems(lane):
    job = "\n".join(_job(lane))
    assert "os: [ubuntu-24.04, windows-2025]" in job
    assert "runs-on: ${{ matrix.os }}" in job
    assert "fail-fast: false" in job
    assert "defaults:\n  run:\n    shell: bash" in "\n".join(_lines())


@pytest.mark.parametrize("lane", ["cpu", "oracles", "gpu"])
def test_native_integration_lanes_build_the_bridges_before_testing(lane):
    job = "\n".join(_job(lane))
    assert "uses: ./.github/actions/build-native" in job
    assert job.index("uses: ./.github/actions/build-native") < job.index("python -m pytest")
    native = (REPOSITORY_ROOT / ".github/actions/build-native/action.yml").read_text(encoding="utf-8")
    for workspace in ("grib1_bridge", "rustwx", "region_global_dealias", "rw_wps"):
        assert f"tools/{workspace}" in native
    assert "cargo build --release --locked --offline" in native


def test_rust_tests_run_the_battery_manifest_without_in_tree_target_mutation():
    body = _step_run("native", "offline Rust package gates with their existing coverage floors")
    assert "tools/battery/run_cargo_gates.py" in body
    assert '--target-dir "$RUNNER_TEMP/' in body
    assert "cargo test --locked --offline" in _step_run("native", "region-global dealiasing tests")
    # The TUI's Rust tests call the installed CLI from their crate cwd.
    job = "\n".join(_job("native"))
    install = 'python -m pip install -e ./gpuwm-data -e ".[dev]"'
    assert install in job
    assert job.index(install) < job.index("tools/battery/run_cargo_gates.py")


def test_self_hosted_gpu_runs_require_an_explicit_trusted_default_branch_dispatch():
    job = "\n".join(_job("gpu"))
    condition = next(line for line in job.splitlines() if line.strip().startswith("if:"))
    for required in ("github.event_name == 'workflow_dispatch'", "inputs.run_gpu",
                     "vars.GPUWM_SELF_HOSTED_GPU == '1'", "github.event.repository.default_branch"):
        assert required in condition
    assert "python -m venv --system-site-packages" in job
    assert '"$environment/bin/python" -m pip install -e ./gpuwm-data -e ".[dev]"' in job
    readiness = _step_run("gpu", "the device is present before anything claims to have used it")
    assert "getDeviceCount() > 0" in readiness
    assert "_cuda_headers_check" in readiness and "c.status == 'verified'" in readiness
