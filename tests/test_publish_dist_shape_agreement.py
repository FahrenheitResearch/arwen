"""The jobs of one cut must agree on how many distributions there are.

``prepare`` builds the Python distributions, hashes them, and hands them
to every later job as the ``python-dists`` artifact.  Three separate jobs
then assert the *exact* shape of that artifact:

* ``prepare`` itself, before it uploads;
* ``publish``, before it pushes to PyPI (job 9 of 10);
* ``release``, before it promotes the draft (job 10 of 10).

Each assertion is written as a literal count in its own Bash block, in a
job that runs on a different runner with different privileges, and nothing
connected them.  On the 2.5.0 integration line they diverged: ``prepare``
was changed to build three wheels (a universal one plus one per platform
carrying prebuilt Rust) and demanded ``-ne 3``, while ``publish`` and
``release`` still demanded exactly one wheel and one sdist.

That divergence is not a failed build.  It is a cut that pushes the tag,
creates the draft, uploads every release asset, and *then* dies at job 9 --
which spends the version number, because tags here are forward-only.  It
is precisely how 2.4.0 was burned.

So this file's job is to make the three counts one contract.  The
structural half reads the counts straight out of the workflow and requires
them to agree; it needs nothing but the standard library, so it cannot go
dark on a runner without Bash.  The executable half then builds the dist
set ``prepare`` says it produces and runs ``publish``'s own verification
block over it, which proves the agreement is real rather than numerically
coincidental.

The shape itself is not this file's call to make, but the record is worth
stating: this project has published 37 versions of a single
``py3-none-any`` wheel plus one sdist, and nothing else.  Per-platform
wheels have never shipped.

Since the 2.5.0 split the same ``python-dists`` artifact also carries the
companion distribution ``gpuwm-data`` (one universal wheel plus one sdist,
built from ``gpuwm-data/``), because ``gpuwm==X.Y.Z`` hard-pins
``gpuwm-data==X.Y.Z``.  Its counts are part of the same three-job
contract: a companion pair that ``prepare`` builds and ``publish``
refuses dies at job 9 with the version spent, exactly like the wheel
divergence above.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"

#: The step in each job that owns that job's dist-shape assertion.
_PREPARE_STEP = "prove distributions, pins, bundles, and Linux artifacts"
_PREPARE_DATA_STEP = "prove the gpuwm-data distributions"
_PUBLISH_STEP = "verify the exact prevalidated distributions"
_PROMOTION_STEP = "publish the fully-proven draft"


def _require_workflow() -> None:
    if not WORKFLOW.is_file():
        pytest.skip("no publish workflow in this tree")


def _step_script(marker: str) -> str:
    """One named step's literal Bash body, extracted from the workflow.

    Deliberately the same extraction the publication state-machine suite
    uses: the workflow keeps these blocks inline so the privileged jobs
    execute no repository code, so reading them literally is the only way
    to measure what will really run.
    """

    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    markers = {f"- name: {marker}", f"- id: {marker}"}
    for start, line in enumerate(lines):
        if not line.startswith("      - ") or line.strip() not in markers:
            continue
        for run_line in range(start + 1, len(lines)):
            candidate = lines[run_line]
            if candidate.startswith("      - "):
                break
            if candidate == "        run: |":
                body: list[str] = []
                for raw in lines[run_line + 1:]:
                    if raw and not raw.startswith("          "):
                        break
                    body.append(raw[10:] if raw else "")
                return "\n".join(body).rstrip() + "\n"
        raise AssertionError(f"workflow step {marker!r} has no literal run block")
    raise AssertionError(f"workflow step {marker!r} was not found")


def _counts(script: str) -> dict[str, int]:
    """Every ``"${#name[@]}" -ne N`` cardinality the block asserts."""

    found: dict[str, int] = {}
    for name, count in re.findall(
            r'"\$\{#([A-Za-z_][A-Za-z0-9_]*)\[@\]\}"\s+-ne\s+(\d+)', script):
        # First occurrence wins; a block that asserted two different counts
        # for one array would be self-contradictory and is caught below.
        if name in found:
            assert found[name] == int(count), (
                f"one Bash block asserts {name} is both {found[name]} and "
                f"{count}")
            continue
        found[name] = int(count)
    return found


def _glob_targets(script: str, array: str) -> list[str]:
    """The glob(s) a ``name=(pattern ...)`` assignment collects."""

    match = re.search(rf"^{array}=\((.*)\)\s*$", script, re.MULTILINE)
    assert match, f"no {array}=(...) assignment in this block"
    return match.group(1).split()


def test_prepare_and_publish_agree_on_the_wheel_count() -> None:
    """THE regression: prepare demanded 3 wheels, publish demanded 1.

    A cut in that state pushes the tag, creates the draft, uploads every
    release asset and dies at job 9 of 10 with the version spent.
    """

    _require_workflow()
    prepare = _counts(_step_script(_PREPARE_STEP))
    publish = _counts(_step_script(_PUBLISH_STEP))

    assert "wheels" in prepare and "wheels" in publish, (
        "one of the two jobs stopped counting wheels at all; a dropped "
        "assertion is the same defect as a disagreeing one")
    assert prepare["wheels"] == publish["wheels"], (
        f"prepare builds and asserts {prepare['wheels']} wheel(s) while "
        f"publish asserts {publish['wheels']}. publish downloads exactly what "
        f"prepare uploaded, so these can never differ: the cut would push the "
        f"tag, create the draft, upload every asset, and then die at job 9 of "
        f"10 with the version number spent. Resolve it toward the shape that "
        f"has actually shipped -- one py3-none-any wheel and one sdist.")
    assert prepare["sdists"] == publish["sdists"], (
        f"prepare asserts {prepare['sdists']} sdist(s) and publish asserts "
        f"{publish['sdists']}")


def test_prepare_and_publish_agree_on_the_companion_dist_count() -> None:
    """gpuwm-data rides the same python-dists artifact, same contract.

    ``gpuwm==X.Y.Z`` hard-pins ``gpuwm-data==X.Y.Z``, so the companion
    pair travels in the artifact every later job re-proves.  A prepare
    that builds companion dists publish refuses is the same burn as the
    wheel divergence: tag pushed, draft cut, dead at job 9.
    """

    _require_workflow()
    prepare = _counts(_step_script(_PREPARE_DATA_STEP))
    publish = _counts(_step_script(_PUBLISH_STEP))

    for array in ("data_wheels", "data_sdists"):
        assert array in prepare and array in publish, (
            f"one of the two jobs stopped counting {array} at all; a "
            f"dropped assertion is the same defect as a disagreeing one")
        assert prepare[array] == publish[array], (
            f"prepare asserts {prepare[array]} {array} while publish "
            f"asserts {publish[array]}; publish downloads exactly what "
            f"prepare uploaded, so these can never differ")


def test_the_published_file_count_is_the_sum_of_its_parts() -> None:
    """``dist/*`` must hold both projects' dists and nothing else.

    publish counts the directory as well as its four globs. A file total
    that does not add up means a fifth kind of file is expected, or that
    one of the numbers was edited without the others.
    """

    _require_workflow()
    publish = _counts(_step_script(_PUBLISH_STEP))
    accounted = (publish["wheels"] + publish["sdists"]
                 + publish["data_wheels"] + publish["data_sdists"])
    assert publish["files"] == accounted, (
        f"publish expects {publish['files']} file(s) in dist/ but only "
        f"{publish['wheels']} gpuwm wheel(s) + {publish['sdists']} gpuwm "
        f"sdist(s) + {publish['data_wheels']} gpuwm-data wheel(s) + "
        f"{publish['data_sdists']} gpuwm-data sdist(s) = {accounted} are "
        f"accounted for")


def test_the_final_promotion_expects_the_same_distributions() -> None:
    """Job 10 re-proves the same bytes, so it must expect the same count."""

    _require_workflow()
    publish = _counts(_step_script(_PUBLISH_STEP))
    promotion = _counts(_step_script(_PROMOTION_STEP))
    assert "dist_files" in promotion, (
        "the promotion step no longer counts the Python distributions it is "
        "about to make public")
    assert promotion["dist_files"] == publish["files"], (
        f"the final promotion expects {promotion['dist_files']} distribution "
        f"file(s) while publish expects {publish['files']}; both re-prove the "
        f"one python-dists artifact prepare uploaded")


def test_prepare_names_the_universal_wheel_it_is_asserting() -> None:
    """The count is only half a contract; the tag has to be pinned too.

    ``-ne 1`` with no name would pass on a lone platform wheel, which is
    the artifact that has never shipped. This asserts the workflow still
    pins ``py3-none-any`` by name -- and, when the count is one, that it
    pins nothing else.
    """

    _require_workflow()
    script = _step_script(_PREPARE_STEP)
    prepare = _counts(script)
    named = re.findall(r"gpuwm-\*-(py3-none-[A-Za-z0-9_]+)\.whl", script)
    assert "py3-none-any" in named, (
        "the prepare job no longer pins the universal wheel filename; a bare "
        "count cannot tell a py3-none-any wheel from a platform one")
    distinct = sorted(set(named))
    assert len(distinct) == prepare["wheels"], (
        f"prepare asserts {prepare['wheels']} wheel(s) but pins "
        f"{len(distinct)} distinct wheel tag(s) by name: {distinct}")


def test_prepare_pins_the_companion_names_to_the_cut_version() -> None:
    """The companion's counts are only half its contract.

    ``gpuwm==X.Y.Z`` pins ``gpuwm-data==X.Y.Z`` to the character, so the
    prepare proof must demand the exact universal filenames the cut tag
    implies -- a companion carrying any other version, or a platform tag,
    publishes a gpuwm no fresh ``pip install gpuwm`` can resolve.
    """

    _require_workflow()
    script = _step_script(_PREPARE_DATA_STEP)
    assert "${RELEASE_TAG#v}" in script, (
        "the companion proof no longer derives the expected version from "
        "the cut tag")
    assert "gpuwm_data-${version}-py3-none-any.whl" in script, (
        "the companion proof no longer pins the exact universal wheel "
        "filename the cut version implies")
    assert "gpuwm_data-${version}.tar.gz" in script, (
        "the companion proof no longer pins the exact sdist filename the "
        "cut version implies")


def test_dist_globs_are_the_same_patterns_in_both_jobs() -> None:
    """Same directory, same globs -- or the counts are about different sets."""

    _require_workflow()
    prepare = _step_script(_PREPARE_STEP)
    prepare_data = _step_script(_PREPARE_DATA_STEP)
    publish = _step_script(_PUBLISH_STEP)
    for array in ("wheels", "sdists"):
        assert _glob_targets(prepare, array) == _glob_targets(publish, array), (
            f"prepare and publish collect {array} from different globs, so "
            f"their counts are not about the same files")
    for array in ("data_wheels", "data_sdists"):
        assert _glob_targets(prepare_data, array) == _glob_targets(
            publish, array), (
            f"prepare and publish collect {array} from different globs, so "
            f"their counts are not about the same files")


# --------------------------------------------------------------------------
# The executable half: publish must accept exactly what prepare produces.
# --------------------------------------------------------------------------

def test_publish_accepts_the_dist_set_prepare_declares(tmp_path: Path) -> None:
    """Build prepare's declared shape, then run publish's own check on it.

    The structural assertions above compare numbers. This one proves the
    agreement is real: it synthesises the dist directory prepare says it
    will upload -- the declared wheel count, wearing the wheel tags prepare
    pins by name, plus the declared sdists -- and runs the literal Bash
    ``publish`` will run against it. Before the fix this exits 1 with
    "python-dists must contain exactly one wheel and one sdist".
    """

    _require_workflow()
    state_machine = pytest.importorskip("test_publish_workflow_state_machine")

    script = _step_script(_PREPARE_STEP)
    prepare = _counts(script)
    tags = sorted(set(re.findall(r"gpuwm-\*-(py3-none-[A-Za-z0-9_]+)\.whl",
                                 script)))
    assert len(tags) == prepare["wheels"], (
        f"cannot synthesise prepare's output: it asserts {prepare['wheels']} "
        f"wheel(s) but names {tags}")

    version = "9.8.7"
    payloads = {f"gpuwm-{version}-{tag}.whl": f"{tag} wheel bytes\n".encode()
                for tag in tags}
    for index in range(prepare["sdists"]):
        suffix = "" if index == 0 else f".{index}"
        payloads[f"gpuwm-{version}{suffix}.tar.gz"] = b"sdist bytes\n"

    # ...plus the companion pair the prepare job's gpuwm-data proof step
    # declares.  Its exact-filename pins make the shape one universal
    # wheel and one sdist by construction, so synthesise exactly that.
    data_counts = _counts(_step_script(_PREPARE_DATA_STEP))
    assert data_counts["data_wheels"] == 1 and data_counts["data_sdists"] == 1, (
        f"cannot synthesise the companion output: the gpuwm-data proof "
        f"asserts {data_counts['data_wheels']} wheel(s) and "
        f"{data_counts['data_sdists']} sdist(s) but pins one exact "
        f"filename of each")
    payloads[f"gpuwm_data-{version}-py3-none-any.whl"] = b"data wheel bytes\n"
    payloads[f"gpuwm_data-{version}.tar.gz"] = b"data sdist bytes\n"

    dist = tmp_path / "dist"
    proof = tmp_path / "release-proof"
    dist.mkdir()
    proof.mkdir()
    for name, payload in payloads.items():
        (dist / name).write_bytes(payload)
    state_machine._write_sums(proof / "python-dists-SHA256SUMS", payloads)

    result = state_machine._run_bash(
        tmp_path, _step_script(_PUBLISH_STEP), {})
    assert result.returncode == 0, (
        "publish refuses the dist set prepare declares it will upload:\n"
        f"{result.stdout}\n{result.stderr}")
