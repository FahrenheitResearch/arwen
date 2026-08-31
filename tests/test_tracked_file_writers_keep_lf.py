"""Every writer in this repository that lands on a TRACKED file writes LF.

WHAT THIS IS NOT: a rule about `Path.write_text`.  There are 1,476 bare
`write_text` calls across 447 tracked non-vendored modules and almost all
of them write a scratch directory, a temp file, or an argparse
destination, where the platform's line ending is the right one.  A
blanket sweep would be 1,455 edits preventing nothing.

WHAT IT IS: the twenty-one call sites, in ten files, whose receiver
actually resolves to a path this repository TRACKS.  There the pathlib
default is a defect, because `write_text` opens in text mode and
translates every "\\n" to `os.linesep` on the way out -- so on Windows a
script that rewrites a tracked file flips its whole line endings, and
`tests/test_line_ending_stability.py` records what that costs: whole-file
merge conflicts with the real semantic conflict invisible underneath, and
split SHA-256 digests where the product hashes its own sources.

THE LIVE INSTANCE, measured on the artifact.  The four Noah-MP mutation
studies do

    original = SOURCE.read_text()
    ...
    finally:
        SOURCE.write_text(original)

against `gpuwm/core/noahmp_{driver,sflx,soilwater,water}.py` -- 2,678
lines of tracked core physics, all LF today and none of them recorded
debt in `_CRLF_DEBT`.  Copying `gpuwm/core/noahmp_water.py` and running
one `read_text()` / `write_text()` round trip through it took the file
from 0 CR bytes to 352, one per line.  The `finally` block whose entire
job is to RESTORE the file is what does it, so any Windows run of any of
those studies would land four whole-file diffs on core physics.

The remaining six write JSON reports and fixtures that are likewise
tracked.  `tools/build_registry.py` made this same fix at cdd08f005 and
`tools/noahmp_wrf461_oracle/mutation_study_snow.py` carries the
`newline=""` spelling; this file pins the rest of them red-on-revert.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: `(script that writes, the tracked file it writes)`.  Every entry was
#: reached by resolving the call's receiver -- through function-scoped and
#: module-scoped assignments -- and then checking the result against
#: `git ls-files`.  Sites whose receiver resolves to a temp directory, a
#: `tmp_path` copy or an untracked sibling are deliberately absent: e.g.
#: `fit_arms.py` also writes `child-{name}.toml` beside itself, and those
#: two files are not tracked.
_WRITERS = (
    ("tools/noahmp_wrf461_oracle/mutation_study_driver.py",
     "gpuwm/core/noahmp_driver.py"),
    ("tools/noahmp_wrf461_oracle/mutation_study_sflx.py",
     "gpuwm/core/noahmp_sflx.py"),
    ("tools/noahmp_wrf461_oracle/mutation_study_soilwater.py",
     "gpuwm/core/noahmp_soilwater.py"),
    ("tools/noahmp_wrf461_oracle/mutation_study_water.py",
     "gpuwm/core/noahmp_water.py"),
    ("evidence/2026-08-25-stale-guards-engine/fit_arms.py",
     "evidence/2026-08-25-stale-guards-engine/fit_arms_report.json"),
    ("evidence/2026-08-25-stale-guards-engine/fit_arms_legacy.py",
     "evidence/2026-08-25-stale-guards-engine/fit_arms_legacy_report.json"),
    ("evidence/2026-08-25-stale-guards-engine/slack_refit.py",
     "evidence/2026-08-25-stale-guards-engine/slack_refit_report.json"),
    ("evidence/2026-08-25-stale-guards-engine/tropical_compare.py",
     "evidence/2026-08-25-stale-guards-engine/tropical_compare_report.json"),
    ("tilestream/make_output_json.py", "tilestream/output-scaling.json"),
    ("tools/rustwx/crates/static-fields/tests/fixtures/highres/"
     "generate_goldens.py",
     "tools/rustwx/crates/static-fields/tests/fixtures/highres/meta.json"),
)

#: Text-mode writes left alone, per script, keyed by the receiver as
#: `ast.unparse` spells it -- because the destination is NOT tracked and
#: the platform's line ending there is nobody's business.  `fit_arms.py`
#: and `fit_arms_legacy.py` write `child-{name}.toml` beside themselves;
#: `git ls-files evidence/2026-08-25-stale-guards-engine/` lists neither.
#: An entry here is a claim that can be checked in one command, which is
#: why the receiver is named rather than the line number.
_UNTRACKED_TEXT_WRITES = {
    "evidence/2026-08-25-stale-guards-engine/fit_arms.py": {"toml_path"},
    "evidence/2026-08-25-stale-guards-engine/fit_arms_legacy.py": {"toml_path"},
}

#: The four whose destination is a Python module they rewrite in place,
#: and which therefore expose `read_source`/`write_source`.
_SOURCE_REWRITERS = tuple(script for script, _dest in _WRITERS
                          if "mutation_study_" in script)


def _is_checkout() -> bool:
    try:
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _load(script: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_writer_under_test_" + Path(script).stem, ROOT / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script,destination", _WRITERS)
def test_the_writer_uses_no_translating_write(script, destination):
    """No `write_text` without an explicit `newline` survives in these ten.

    Scoped to the ten files whose writes actually land on a tracked path,
    not to the repository: a rule over all 1,476 bare `write_text` calls
    would be a style gate with no breakage behind it, and the two
    line-ending ratchets already catch the outcome.  Here the breakage is
    named and it has a measurement: one text-mode round trip over
    `gpuwm/core/noahmp_water.py` turns 0 CR bytes into 352.

    `newline=` is accepted as well as `write_bytes` because
    `mutation_study_snow.py` already writes `newline=""` and both spellings
    say the same thing to the platform.
    """

    allowed = _UNTRACKED_TEXT_WRITES.get(script, set())
    tree = ast.parse((ROOT / script).read_bytes().decode("utf-8"))
    translating = [
        (node.lineno, ast.unparse(node.func.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_text"
        and not any(kw.arg == "newline" for kw in node.keywords)
        and ast.unparse(node.func.value) not in allowed]
    assert not translating, (
        f"{script} writes {destination} and still has a text-mode write "
        f"at (line, receiver) {translating}.  On Windows that translates "
        "every newline on the way out, so a tracked file gains a "
        "whole-file CRLF diff.  Use write_bytes(text.encode('utf-8')) or "
        "pass newline='\\n' -- or, if that receiver is genuinely not a "
        "tracked path, name it in _UNTRACKED_TEXT_WRITES with the "
        "`git ls-files` that shows it.")


@pytest.mark.skipif(not _is_checkout(),
                    reason="the destination list is a checkout property")
@pytest.mark.parametrize("script,destination", _WRITERS)
def test_the_destination_is_tracked_and_still_lf(script, destination):
    """The list stays honest: each destination is tracked, and is LF now.

    If a destination stops being tracked the entry belongs elsewhere --
    a scratch file may carry whatever the platform gives it -- and if one
    is already CRLF then the writer above it has been run on Windows and
    this fix arrived too late to matter for that file.  Either way the
    table is a description of a repository that no longer exists, which
    is how allowlists rot.
    """

    listed = subprocess.check_output(
        ["git", "ls-files", "--error-unmatch", "--", destination],
        cwd=ROOT, stderr=subprocess.DEVNULL, text=True)
    assert listed.strip() == destination
    raw = (ROOT / destination).read_bytes()
    assert b"\r" not in raw, (
        f"{destination} carries {raw.count(chr(13).encode())} CR bytes; "
        f"the writer in {script} has already flipped it")


@pytest.mark.parametrize("script", _SOURCE_REWRITERS)
def test_the_source_round_trip_is_byte_exact(script, tmp_path):
    """`read_source` then `write_source` returns the file unchanged.

    This is the round trip that matters: the mutation study reads the
    port's source once, writes a mutant over it for each of several
    hundred mutants, and restores the original in a `finally`.  Every one
    of those writes has to give back exactly the bytes it was handed, or
    a study that changed nothing leaves a whole-file diff behind.

    Run against the REAL tracked source each study mutates -- 528, 1,177,
    621 and 352 lines -- copied into `tmp_path`, so the assertion is over
    the actual bytes rather than a synthetic sample.
    """

    module = _load(script)
    real = module.SOURCE.read_bytes()
    copy = tmp_path / module.SOURCE.name
    copy.write_bytes(real)
    module.SOURCE = copy

    text = module.read_source()
    module.write_source(text)
    assert copy.read_bytes() == real
    assert b"\r" not in copy.read_bytes()

    # And a mutant-shaped write -- the same text with one line changed,
    # which is what actually happens a few hundred times per study --
    # adds no carriage return either.
    mutant = text.replace("import numpy", "import numpy  # mutant", 1)
    module.write_source(mutant)
    assert b"\r" not in copy.read_bytes()
    assert copy.read_bytes() == mutant.encode("utf-8")


@pytest.mark.skipif(not _is_checkout(),
                    reason="the destination list is a checkout property")
@pytest.mark.parametrize("script,destination", _WRITERS)
def test_the_fixed_spelling_survives_what_the_old_one_did_not(
        script, destination, tmp_path):
    """The two spellings, over the destination's own bytes, side by side.

    The fixed spelling -- `write_bytes(text.encode("utf-8"))` -- has to
    reproduce the file exactly.  The spelling it replaced has to be shown
    doing the damage, on this platform, rather than asserted to: without
    that half, a green test proves only that nothing was ever at stake.
    On a POSIX runner `os.linesep` is "\\n" and there is nothing to
    demonstrate, so that half is skipped there and the round trip stands
    on its own -- Windows is where the failure lives.
    """

    real = (ROOT / destination).read_bytes()
    text = real.decode("utf-8")
    copy = tmp_path / Path(destination).name

    copy.write_bytes(text.encode("utf-8"))
    assert copy.read_bytes() == real

    if os.linesep == "\r\n" and "\n" in text:
        copy.write_text(text, encoding="utf-8")
        assert b"\r" in copy.read_bytes(), (
            "the text-mode writer did not translate, so this platform is "
            "not the one that produced the defect and this half of the "
            "test is proving nothing")
        assert copy.read_bytes() != real
