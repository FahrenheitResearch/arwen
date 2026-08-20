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

2.3.3 EXTENSION -- the same two directions, applied to the COMMAND LINE.

An extra is one way to make a shipped feature unreachable.  A flag is the
other, and the 2.3.2 reachability audit found the command surface in both
failure modes at once:

* Documents named doors that do not exist.  `TILES.md` told a reader to
  run `gpuwm run CONFIG.toml --case-data ...` and `gpuwm plan`; argparse
  answers the first with `unrecognized arguments` and the second with
  `invalid choice: 'plan'`.  Both were born in a documentation commit and
  never existed in code, so no user could ever have run either.
* Code defined flags no document named.  `--parent-namelist` gated the
  whole stock-WRF-parent route that `gpuwm downscale --help` advertises;
  `--tiles` was the only way to stream the prepared route; `--no-memory-gate`
  was the only escape from the pre-fetch memory gate.  Forty-six flags on
  the `gpuwm` subcommands alone appeared in no document at all.

So three more rules, holding the two halves against each other:

3. Every command a user-facing document tells a reader to RUN must name a
   real door and pass only flags that door defines.
4. Every option every documented door defines must appear in
   `docs/public/CLI-OPTIONS.md`, under that door.
5. Every option `CLI-OPTIONS.md` lists must still exist in that door's
   parser, so the page cannot outlive a removed flag.

Rule 4 is the one that makes the class non-recurring: it is not possible
to add a flag and leave it undocumented, because the page is generated
from the parsers (`python -m tools.build_cli_options_doc`) and this test
fails when the committed page and the parsers disagree.

A sixth rule covers the same defect in configuration: every key
`[case_data]` REQUIRES must be named by a user-facing document.
`output_title` was required at load and appeared in no user-facing page,
so a reader hand-authoring the table from `CONFIGURATION.md` was refused
for a key the documentation had never mentioned.
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


def test_high_resolution_terrain_is_reachable_from_a_bare_install():
    """The 2.3.2 breakage, gated at the engine rather than at pip.

    In 2.3.2 the terrain path's only engine was rasterio + pyproj and
    they sat in an extra nobody named, so following HIGHRES-TERRAIN.md
    on a documented install downloaded 160.7 MiB of Copernicus tiles and
    then died on an import.  2.3.3 answered that by making them runtime
    dependencies.  Since the warp substrate flipped onto the Rust
    static-fields library they are the parity FALLBACK, not the engine,
    and the reachability question changed with it: what a bare `pip
    install gpuwm` must carry is the ENGINE.

    So this asserts the engine ships in the bundle a wheel stages, and
    leaves the runtime proof -- rasterio, pyproj and affine made
    unimportable, every default call still answering -- to
    tests/test_static_highres_warp_routing.py.  Two gates, one for the
    packaging fact and one for the behaviour, because 2.3.2 passed
    every packaging check it had.
    """
    from gpuwm import bridge_assets

    staged = {artifact.name for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert "static_fields" in staged, (
        "the static-fields library is not a bundled artifact, so a bare "
        "`pip install gpuwm` stages no high-resolution warp engine and "
        "HIGHRES-TERRAIN.md is unreachable again -- exactly the 2.3.2 "
        "failure, one layer down")

    with (_repo_root() / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    names = " ".join(project["dependencies"])
    for fallback_only in ("rasterio", "pyproj"):
        assert fallback_only not in names, (
            f"{fallback_only} is back in the runtime dependencies. It is "
            "the pure-Python parity fallback's library, which no default "
            "high-resolution run reads, so declaring it makes every bare "
            "install carry a GDAL stack it never imports. If the default "
            "engine really did move back to Python, this test is the "
            "wrong thing to edit -- the port did.")


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


# ==========================================================================
# 2.3.3: the command line, in the same two directions.
# ==========================================================================

# The extractor, the door registry and the resolver are SHARED.  Three
# partial implementations of "hold a document against the code" already
# existed when these rules were written -- VERIFICATION.md's recipe
# parser, CERTIFICATION.md's condition binding, and the extras check
# above -- so the mechanism lives in one module that all of them import
# rather than being written a fourth time here.
from doc_command_parity import (  # noqa: E402
    FLAG as _FLAG,
    MODE_DOORS as _MODE_DOORS,
    code_fragments as _code_fragments,
    door_options as _door_options,
    doors as _doors,
    resolve_door as _resolve_door,
    user_facing_docs as _user_facing_docs,
)


# --------------------------------------------------------------------------
# Anti-vacuous floors and instrument self-tests.
# --------------------------------------------------------------------------

def test_the_user_facing_doc_set_is_real():
    docs = _user_facing_docs()
    names = {p.name for p in docs}
    assert len(docs) > 15, docs
    for required in ("README.md", "TILES.md", "CONFIGURATION.md",
                     "FIRST-LIGHT.md", "CLI-OPTIONS.md"):
        assert required in names, f"{required} is not in the user-facing set"


def test_the_door_registry_is_real():
    doors = _doors()
    assert len(doors) > 25, sorted(doors)
    assert "gpuwm run" in doors and "rw-wps" in doors
    total = sum(len(_door_options(p)) for p in doors.values())
    assert total > 300, total


def test_every_installed_console_script_is_a_documented_door():
    """The door registry is driven by `[project.scripts]`, not by memory.

    Rule 4 makes an undocumented FLAG impossible, and said nothing about
    an undocumented PROGRAM.  `gpuwm-member-prep` -- the ensemble-member
    front door, a console script every `pip install gpuwm` puts on the
    PATH -- appeared nowhere on the reference page, because the door list
    was a hand-written literal and nobody added a line to it.  Every one
    of its eleven options was therefore unreachable from any document,
    which is the 2.3.2 defect wearing a different hat.

    So the generator reads the same table setuptools reads, and this
    holds it to that: a script name in `pyproject.toml` is a door.
    """

    from tools.build_cli_options_doc import ALIASES, console_scripts

    scripts = console_scripts()
    assert len(scripts) >= 8, scripts
    assert "gpuwm-member-prep" in scripts, (
        "the instrument is blind: pyproject declares no member-prep "
        "script, so this test could not see it missing")
    built = _doors()
    offenders = []
    for name in sorted(scripts):
        if name in ALIASES:
            continue
        if name == "gpuwm":
            # The subcommand tree IS this door; it has no options of its
            # own beyond the ones its subcommands carry.
            if not any(door.startswith("gpuwm ") for door in built):
                offenders.append(name)
            continue
        if name not in built:
            offenders.append(name)
    assert not offenders, (
        "these console scripts are installed on every user's PATH and "
        "the generated CLI reference has no section for them, so no "
        "document names a single one of their options:\n  "
        + "\n  ".join(offenders))


def test_the_invocation_reader_tells_commands_from_prose():
    """The instrument, against known answers, BOTH directions."""
    doors = _doors()
    assert _resolve_door("gpuwm run CONFIG.toml", doors)[0] == "gpuwm run"
    assert _resolve_door("$ gpuwm plan --help", doors)[0] == "gpuwm plan"
    assert _resolve_door("rw-wps --version", doors)[0] == "rw-wps"
    assert _resolve_door(
        "python -m gpuwm.prepared_single_domain_forecast --outdir x",
        doors)[0] == "gpuwm-prepared-forecast"
    # ...and NOT on prose that merely CONTAINS the program name.  The
    # rule is positional on purpose: a code span that begins with the
    # program name is a command, and one that mentions it mid-sentence
    # is prose.  These four are the real shapes in the corpus -- every
    # one of them would be a false positive under a "contains" rule.
    assert _resolve_door(
        "bl_mynn_mixlength=2 is outside the admitted MYNN option "
        "identity; gpuwm implements bl_mynn_mixlength=1 only", doors) is None
    assert _resolve_door("the tree at /path/to/gpuwm the way you got it",
                         doors) is None
    assert _resolve_door(
        "--locked --offline ... which gpuwm then finds on its own",
        doors) is None
    # Placeholders are not a claim that a subcommand exists.
    assert _resolve_door("gpuwm <command>", doors) is None
    assert _resolve_door("gpuwm SUBCOMMAND", doors) is None


def test_the_invocation_reader_catches_a_door_that_does_not_exist():
    """The negative control: the rule must REJECT the retired shapes.

    `gpuwm plan` and `gpuwm run --case-data` are the two doors TILES.md
    printed that argparse answers with exit 2.  A guard that only ever
    passes is not evidence of anything, so the instrument is held
    against the exact strings the audit found.
    """

    doors = _doors()
    name, body = _resolve_door("gpuwm plan --help", doors)
    assert name == "gpuwm plan"
    assert name not in doors, (
        "`gpuwm plan` resolves to a real subcommand now; if it was "
        "added deliberately this control needs a different retired name")

    name, body = _resolve_door("gpuwm run imported_v2.toml --case-data foo",
                               doors)
    assert name == "gpuwm run" and name in doors
    assert "--case-data" not in _door_options(doors[name]), (
        "`gpuwm run` defines --case-data now; if it was added "
        "deliberately this control needs a different retired flag")
    assert "--case-data" in _FLAG.findall(body)


def test_the_flag_reader_finds_flags_and_not_other_dashes():
    assert _FLAG.findall("gpuwm run C.toml --case-data foo") == \
        ["--case-data"]
    assert _FLAG.findall("--a --b-c") == ["--a", "--b-c"]
    assert _FLAG.findall("a -- b") == []
    assert _FLAG.findall("value=-3.5 and x--y") == []


# --------------------------------------------------------------------------
# Rule 3: a document may only tell a reader to run a door that exists,
# with flags that door defines.
# --------------------------------------------------------------------------

def test_every_command_the_docs_print_names_a_real_door():
    doors = _doors()
    root = _repo_root()
    offenders: list[str] = []
    for path in _user_facing_docs():
        rel = path.relative_to(root).as_posix()
        for lineno, fragment in _code_fragments(
                path.read_text(encoding="utf-8")):
            resolved = _resolve_door(fragment, doors)
            if resolved is None:
                continue
            name, _ = resolved
            if name not in doors:
                offenders.append(
                    f"{rel}:{lineno}: `{fragment.strip()[:80]}` -> there "
                    f"is no `{name}`")
    assert not offenders, (
        "a user-facing document tells a reader to run a command that "
        "does not exist; running it exits 2 with an argparse usage "
        "dump:\n  " + "\n  ".join(offenders))


def test_every_flag_the_docs_pass_is_defined_by_that_door():
    doors = _doors()
    root = _repo_root()
    offenders: list[str] = []
    for path in _user_facing_docs():
        rel = path.relative_to(root).as_posix()
        for lineno, fragment in _code_fragments(
                path.read_text(encoding="utf-8")):
            resolved = _resolve_door(fragment, doors)
            if resolved is None:
                continue
            name, body = resolved
            if name not in doors:
                continue
            defined = _door_options(doors[name])
            # The mode flag NAMES the door; it is the selector, not one
            # of the options the selected program takes.
            defined |= {mode for mode, door in _MODE_DOORS.items()
                        if door == name}
            for flag in _FLAG.findall(body):
                if flag not in defined:
                    offenders.append(
                        f"{rel}:{lineno}: `{name}` has no {flag}  "
                        f"({fragment.strip()[:70]})")
    assert not offenders, (
        "a user-facing document passes a flag the command does not "
        "define; the pasted command exits 2:\n  "
        + "\n  ".join(offenders))


# --------------------------------------------------------------------------
# Rules 4 and 5: the reference page and the parsers, both directions.
# --------------------------------------------------------------------------

_CLI_OPTIONS_DOC = "docs/public/CLI-OPTIONS.md"


def _documented_options() -> dict[str, set[str]]:
    """door -> flags the committed reference page lists for it."""

    text = (_repo_root() / _CLI_OPTIONS_DOC).read_text(encoding="utf-8")
    out: dict[str, set[str]] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^##\s+`([^`]+)`\s*$", line)
        if heading:
            current = heading.group(1)
            out.setdefault(current, set())
            continue
        if current and line.startswith("| `"):
            cell = line.split("|")[1].strip().strip("`")
            for flag in _FLAG.findall(cell):
                out[current].add(flag)
    return out


def test_every_option_every_door_defines_is_on_the_reference_page():
    doors = _doors()
    documented = _documented_options()
    offenders: list[str] = []
    for name, parser in sorted(doors.items()):
        listed = documented.get(name)
        if listed is None:
            offenders.append(f"{name}: the page has no section for it")
            continue
        for flag in sorted(_door_options(parser)):
            if flag != "--help" and flag not in listed:
                offenders.append(f"{name} {flag}")
    assert not offenders, (
        f"these options exist in argparse and appear nowhere in "
        f"{_CLI_OPTIONS_DOC}, so no document names them and no reader "
        f"can find them.  Regenerate with "
        f"`python -m tools.build_cli_options_doc`:\n  "
        + "\n  ".join(offenders))


def test_every_option_the_reference_page_lists_still_exists():
    doors = _doors()
    documented = _documented_options()
    offenders: list[str] = []
    for name, listed in sorted(documented.items()):
        parser = doors.get(name)
        if parser is None:
            continue
        defined = _door_options(parser)
        offenders += [f"{name} {flag}" for flag in sorted(listed)
                      if flag not in defined]
    assert not offenders, (
        f"{_CLI_OPTIONS_DOC} documents flags that no longer exist; a "
        f"reader pasting them gets exit 2.  Regenerate with "
        f"`python -m tools.build_cli_options_doc`:\n  "
        + "\n  ".join(offenders))


def test_the_reference_page_is_not_stale():
    """The page and the parsers, byte for byte."""

    from tools.build_cli_options_doc import render
    current = (_repo_root() / _CLI_OPTIONS_DOC).read_text(encoding="utf-8")
    assert current == render(), (
        f"{_CLI_OPTIONS_DOC} is out of date with the parsers; run "
        "`python -m tools.build_cli_options_doc`")


def test_the_reference_page_names_no_developer_machine():
    """A help string that interpolates a path must not ship one."""

    text = (_repo_root() / _CLI_OPTIONS_DOC).read_text(encoding="utf-8")
    offenders = [line.strip() for line in text.splitlines()
                 if re.search(r"[A-Za-z]:\\Users\\|/home/[a-z]", line)]
    assert not offenders, (
        "the reference page names a developer's own filesystem:\n  "
        + "\n  ".join(offenders))


# --------------------------------------------------------------------------
# Rule 6: a required config key must be documented.
# --------------------------------------------------------------------------

#: Where a reader hand-authoring a config is sent to learn the keys.
_CONFIG_REFERENCE = "docs/public/CONFIGURATION.md"


def test_every_required_case_data_key_is_documented():
    """In the CONFIG REFERENCE, not merely somewhere in the corpus.

    A key that appears only inside one worked example on another page is
    findable by someone who already knows to look there.  The page a
    reader is sent to in order to author the table is the page that has
    to name every key the loader will refuse them for.
    """

    from gpuwm.case_data import _REQUIRED_KEYS

    assert _REQUIRED_KEYS, "no required keys at all -- instrument is blind"
    reference = (_repo_root() / _CONFIG_REFERENCE).read_text(encoding="utf-8")
    offenders = [key for key in _REQUIRED_KEYS if key not in reference]
    assert not offenders, (
        "[case_data] REQUIRES these keys at load and "
        f"{_CONFIG_REFERENCE} does not name them, so a reader "
        "hand-authoring the table from the configuration reference is "
        "refused for a key that page never mentioned:\n  "
        + "\n  ".join(offenders))


def test_every_optional_case_data_key_is_documented():
    """The optional keys too: an undocumented knob is an unreachable one."""

    from gpuwm.case_data import _OPTIONAL_KEYS

    reference = (_repo_root() / _CONFIG_REFERENCE).read_text(encoding="utf-8")
    offenders = [key for key in _OPTIONAL_KEYS if key not in reference]
    assert not offenders, (
        f"[case_data] accepts these keys and {_CONFIG_REFERENCE} names "
        "none of them, so the feature each one gates is reachable only "
        "by reading the source:\n  " + "\n  ".join(offenders))
