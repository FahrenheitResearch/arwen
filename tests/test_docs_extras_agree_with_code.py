"""The install lines in the docs, held against the extras that exist.

2.3.2 shipped a headline feature nobody could reach.  The libraries the
high-resolution terrain path imports lived in a `geog` extra; `[all]`
deliberately excluded it; every published quickstart one-liner omitted
it; and the only mention of it anywhere was a source-checkout line in an
older document.  Nothing was broken in the wheel -- `pip install
'gpuwm[geog]'` worked perfectly.  What was broken was that no documented
command led there.

That failure is invisible to every other kind of test, because each half
is individually correct.  It is only visible by holding the two halves
against each other, which is what this file does, in both directions:

1. Every extra a shipped document tells a reader to install must be an
   extra `pyproject.toml` declares.  A doc naming `gpuwm[geog]` after
   that extra was deleted would be a pasteable command that fails.
2. Every extra the CODE tells a reader to install must be declared AND
   must appear in a shipped document.  A remedy string is not
   documentation: a reader only sees it after they have already hit the
   failure, which is exactly the trap 2.3.2 set.

Direction 2 is the one that would have caught this release's defect
before it shipped, and it caught two live instances when it was written
(`gpuwm[obs]` and `gpuwm[dealias]` were named by code remedies and by no
document at all).
"""
from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tracked(*prefixes: str) -> list[Path]:
    """Tracked files under ``prefixes`` -- git is the shipping manifest.

    Enumerated rather than hardcoded so a document added tomorrow is
    covered without anyone remembering to add it here.
    """
    root = _repo_root()
    out = subprocess.run(["git", "ls-files", "-z", *prefixes],
                         cwd=root, capture_output=True, check=True)
    return [root / name
            for name in out.stdout.decode("utf-8").split("\0") if name]


def _docs() -> list[Path]:
    return [path for path in _tracked("docs", "README.md")
            if path.suffix.lower() == ".md"]


def _code() -> list[Path]:
    return [path for path in _tracked("gpuwm", "tools")
            if path.suffix == ".py"]


def _declared_extras() -> dict[str, list[str]]:
    with (_repo_root() / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["optional-dependencies"]


#: `gpuwm[a,b]` and the checkout form `.[a,b]`, in quotes or bare.
_EXTRA_PATTERN = re.compile(r"(?:gpuwm|\.)\[([a-z0-9,._-]+)\]")


def _fenced_lines(text: str) -> list[str]:
    """Lines inside ``` fences: the parts of a doc that are commands."""
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        if inside:
            out.append(line)
    return out


def _extras_named_in(paths: list[Path],
                     only_install_lines: bool) -> dict[str, list[str]]:
    """Map extra -> ["path:lineno", ...] over the given files."""
    found: dict[str, list[str]] = {}
    root = _repo_root()
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if only_install_lines and "pip install" not in line:
                continue
            for hit in _EXTRA_PATTERN.findall(line):
                for name in hit.split(","):
                    name = name.strip()
                    if not name:
                        continue
                    found.setdefault(name, []).append(
                        f"{path.relative_to(root).as_posix()}:{lineno}")
    return found


# --------------------------------------------------------------------------
# Anti-vacuous floors.  A sweep over an empty list passes forever.
# --------------------------------------------------------------------------

def test_the_docs_and_code_trees_are_tracked_and_non_empty():
    docs, code = _docs(), _code()
    assert len(docs) > 5, docs
    assert any(p.name == "HIGHRES-TERRAIN.md" for p in docs), \
        "the terrain document is the one this release is about"
    assert len(code) > 50, len(code)
    assert _declared_extras(), "pyproject declares no extras at all"


def test_the_extractor_finds_the_lines_it_is_meant_to_find():
    """The instrument, against known answers, both directions."""
    pattern = _EXTRA_PATTERN
    assert pattern.findall("pip install 'gpuwm[all-cu12]'") == ["all-cu12"]
    assert pattern.findall('python -m pip install -e ".[dev,geog]"') == \
        ["dev,geog"]
    assert pattern.findall("pip install gpuwm[render]") == ["render"]
    # And does NOT fire on prose that merely contains brackets.
    assert pattern.findall("the [static.highres] table") == []
    assert pattern.findall("see [the docs](x.md)") == []


# --------------------------------------------------------------------------
# Direction 1: docs may only name extras that exist.
# --------------------------------------------------------------------------

def test_every_extra_the_docs_install_is_declared_by_pyproject():
    declared = set(_declared_extras())
    named = _extras_named_in(_docs(), only_install_lines=True)
    offenders = [f"{name} at {', '.join(where)}"
                 for name, where in sorted(named.items())
                 if name not in declared]
    assert not offenders, (
        "shipped docs tell a reader to install an extra pyproject does "
        "not declare, so the command fails when pasted:\n  "
        + "\n  ".join(offenders))


def test_the_docs_do_name_extras_so_direction_one_is_not_vacuous():
    named = _extras_named_in(_docs(), only_install_lines=True)
    assert len(named) >= 5, named


# --------------------------------------------------------------------------
# Direction 2: extras the code names must exist AND be documented.
# --------------------------------------------------------------------------

def _extras_named_by_code() -> dict[str, list[str]]:
    # Only the `gpuwm[...]` form: a remedy tells a reader to install the
    # distribution by name, never the checkout-relative `.[...]` form.
    found: dict[str, list[str]] = {}
    root = _repo_root()
    pattern = re.compile(r"gpuwm\[([a-z0-9,._-]+)\]")
    for path in _code():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for hit in pattern.findall(line):
                for name in hit.split(","):
                    name = name.strip()
                    if name:
                        found.setdefault(name, []).append(
                            f"{path.relative_to(root).as_posix()}:{lineno}")
    return found


def test_every_extra_the_code_names_is_declared_by_pyproject():
    declared = set(_declared_extras())
    offenders = [f"{name} at {', '.join(where)}"
                 for name, where in sorted(_extras_named_by_code().items())
                 if name not in declared]
    assert not offenders, (
        "code prints a remedy naming an extra pyproject does not "
        "declare:\n  " + "\n  ".join(offenders))


def test_every_extra_the_code_names_appears_in_a_shipped_doc():
    """A remedy is not documentation: it is only read after the failure."""
    documented = set(_extras_named_in(_docs(), only_install_lines=False))
    offenders = [f"{name} at {', '.join(where)}"
                 for name, where in sorted(_extras_named_by_code().items())
                 if name not in documented]
    assert not offenders, (
        "code tells a reader to install an extra that no shipped "
        "document mentions -- the reader only ever sees it after they "
        "have already hit the failure:\n  " + "\n  ".join(offenders))


def test_direction_two_is_not_vacuous():
    assert len(_extras_named_by_code()) >= 4, _extras_named_by_code()


# --------------------------------------------------------------------------
# The specific reachability contract this release exists to restore.
# --------------------------------------------------------------------------

def test_geog_extra_still_exists_so_the_old_working_command_still_works():
    """`pip install 'gpuwm[geog]'` was the ONE command that worked in
    2.3.2.  Deleting the extra would have broken it."""
    declared = _declared_extras()
    assert "geog" in declared, (
        "the geog extra was removed; every 'pip install gpuwm[geog]' "
        "written down in the wild now fails at resolution")
    assert declared["geog"] == [], (
        "geog should be empty: its contents moved into the runtime "
        f"dependencies, got {declared['geog']}")


def test_the_geography_libraries_are_runtime_dependencies_not_an_extra():
    with (_repo_root() / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    names = " ".join(project["dependencies"])
    for required in ("rasterio", "pyproj"):
        assert required in names, (
            f"{required} is not a runtime dependency, so a bare "
            "`pip install gpuwm` cannot build high-resolution terrain "
            "and HIGHRES-TERRAIN.md is unreachable again")


def test_the_terrain_doc_carries_a_working_install_line():
    """The doc that had no install line at all in 2.3.2."""
    doc = _repo_root() / "docs" / "public" / "HIGHRES-TERRAIN.md"
    text = doc.read_text(encoding="utf-8")
    assert "pip install gpuwm" in text, \
        "HIGHRES-TERRAIN.md names no install command"
    # It must not send the reader to an extra to get the feature.
    named = _EXTRA_PATTERN.findall(text)
    assert "geog" not in named, (
        "the terrain doc still routes the reader through the geog "
        "extra; the whole point of 2.3.3 is that a bare install works")
    # The commands it prints must be runnable off a pip install, which
    # has no source checkout: `python tools/x.py` is not.  Only fenced
    # blocks are commands; prose may legitimately discuss the broken
    # form in order to warn about it, and does.
    offenders = [line.strip() for line in _fenced_lines(text)
                 if re.search(r"python\s+tools/", line)]
    assert not offenders, (
        "the terrain doc prints source-checkout commands a pip user "
        "cannot run; use `python -m tools.<module>`:\n  "
        + "\n  ".join(offenders))


def test_the_fence_reader_separates_commands_from_prose():
    """The instrument, against a known answer, both directions."""
    sample = ("prose mentioning python tools/x.py in passing\n"
              "```\n"
              "python tools/real_command.py\n"
              "```\n"
              "more prose about python tools/y.py\n")
    inside = _fenced_lines(sample)
    assert inside == ["python tools/real_command.py"], inside
