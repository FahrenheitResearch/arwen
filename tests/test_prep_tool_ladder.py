"""The prep front door resolves its GRIB2 tools; the flags are overrides.

The papercut these tests close, measured on the release line at
a4995ee07: ``gpuwm prep --source hrrr-prs`` -- and every other
packaged-profile GRIB2 source, the generic mapped GRIB2 route and the
20CRv3 run route -- refused with ``--grib2-dump is required for mapped
grib2`` on a box where ``gpuwm fetch-bridges`` had already staged both
tools and ``gpuwm doctor`` printed them verified.  The ENGINE
(``mapped_source._build_grib2_tools``) had consulted the shared ladder
since 2.4; the FRONT DOOR never asked it and demanded flags instead.

The contract now, after the compose port.  ``compose`` is declared for
``grib2`` in ``mapped_engine_bridge.ENGINE_CAPABILITIES``, and
``gpuwm.mapped_direct`` asks the engine for ``compose`` on every call
(``MAPPED_ROUTE_SUBCOMMAND``), so the mapped GRIB2 route -- packaged
profiles and the generic spelling alike -- does its byte work in
process on the default engine.  That splits this file's rows in two,
and the split is the contract:

* on the DEFAULT engine the two tool paths are not wanted, not
  resolved, and above all not FORWARDED.  Forwarding a resolved tool is
  not merely redundant: naming a subprocess decoder is the spelling
  that routes a call back to the Python engine
  (``_mapped_engine_choice``), so a door that forwarded what it found
  would un-default the Rust engine on every box that ran
  ``gpuwm fetch-bridges``.  The rows below assert the bare door on an
  estate with NOTHING staged and no crate -- which is what a wheel
  install is -- and again on a staged estate, where the staged copies
  must stay out of the command.
* on the WORKAROUND engine (``GPUWM_MAPPED_ENGINE=python``, the
  documented spelling for routing around a Rust decode defect) the
  Python route runs the two executables, so every rung of the original
  papercut still has to hold there, and it is asserted by EXECUTING the
  door rather than by describing it:

  - omitted, the two tool paths resolve through the same ladder every
    other bridge uses (environment override, checkout build, staged
    ``~/.gpuwm/bridges``, wheel), through the same resolver the engine
    runs, so the pre-flight and the route cannot disagree;
  - passed, the flags override the ladder exactly as the per-tool
    environment variables always have -- and naming one is itself a
    request for the Python engine, so those rows need no engine pin;
  - unresolvable, the refusal is the install-aware remedy doctor prints
    -- never a demand for a flag pointing at a file the user does not
    have;
  - a resolved binary that predates this release's decoder contract is
    refused BY NAME when nothing here can rebuild it, because handing
    the route a stale decoder is the 1.1.0 series-file failure again;
  - usage errors still lead: an argument-vocabulary mistake is reported
    before the estate is consulted, so a reader fixes their command line
    once.

The 20CRv3 run route follows the engine table like every composed door:
``gpuwm.twentycrv3_wrf`` routes through ``decode_composed_source`` and
reads the raw record inventory in process on the bare default, so its
ladder rows below hold BOTH halves -- no tools forwarded bare, both
resolved on the documented Python-engine workaround.  Its old
engine-unconditional row was the last ``ENGINE_GAPS`` entry.

Which SUBCOMMAND the route runs decides whether the tools are wanted at
all, and that question lives in ``mapped_engine_bridge`` -- see
``tests/test_mapped_compose_tool_forwarding.py``, which runs the
composed invocation this door prints, and states the no-forwarding
invariant over the whole source registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm import bridges, mapped_source
from gpuwm.mapped_engine_bridge import ENGINE_ENV, ENGINE_PYTHON
from gpuwm.source_cli import EXIT_CONFIG, EXIT_USAGE, main


@pytest.fixture(autouse=True)
def _clean_tool_environment(monkeypatch):
    """No sealed runtime and no per-tool override unless a test sets one."""

    for variable in ("GPUWM_GRIB2_INVENTORY", "GPUWM_GRIB2_DUMP",
                     "GPUWM_NATIVE_DISTRIBUTION_MANIFEST",
                     "GPUWM_MAPPED_ENGINE"):
        monkeypatch.delenv(variable, raising=False)


def _the_workaround_engine(monkeypatch):
    """Ask for the Python engine the way the docs tell a user to.

    The environment spelling rather than ``--mapped-engine python``
    deliberately: it is the one request every route below accepts,
    including the packaged profiles, and it is what a user reaching for
    the workaround sets.
    """

    monkeypatch.setenv(ENGINE_ENV, ENGINE_PYTHON)


def _staged_tools(tmp_path, monkeypatch):
    """A box where ``gpuwm fetch-bridges`` has run: the ladder answers.

    The fakes carry the REAL contract markers so the resolver's static
    ABI gate passes on genuine evidence rather than being stubbed out.
    """

    staged = {}
    for name in ("grib2_inventory", "grib2_dump"):
        path = tmp_path / bridges.executable_name(name)
        path.write_bytes(b"MZ\0\0" + bridges.BRIDGE_ABI_MARKERS[name])
        staged[name] = path
    monkeypatch.setattr(bridges, "find_bridge",
                        lambda name: staged.get(name))
    return staged


def _bare_estate(tmp_path, monkeypatch):
    """A wheel install with nothing staged: ladder empty, no crate.

    ``subprocess.run`` is stubbed to fail loudly so a regression that
    reintroduces an unconditional cargo build is a fast red test, never
    a two-minute compile inside the suite.
    """

    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    monkeypatch.setattr(mapped_source, "_grib2_tools_crate",
                        lambda: tmp_path / "no-such-crate")

    class _NoCargo:
        returncode = 1
        stderr = "cargo must not run in this test"
        stdout = ""

    monkeypatch.setattr(mapped_source.subprocess, "run",
                        lambda *args, **kwargs: _NoCargo())


#: The flags that name a subprocess GRIB2 decoder in a child command.
#: Present, they pin the Python engine; that is why the default route
#: must emit neither.
_TOOL_FLAGS = ("--grib2-inventory", "--grib2-dump")


def _packaged_profile_argv(*extra):
    """A bare hrrr-prs preparation: no tool flags anywhere."""

    return [
        "--source", "hrrr-prs",
        "--input", "/data/f00.grib2",
        "--input", "/data/f01.grib2",
        "--supplement", "/data/f00.grib2",
        "--supplement", "/data/f01.grib2",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--source-manifest", "/case/input-manifest.json",
        "--source-manifest-sha256", "abc123",
        "--output-root", "/output",
        *extra, "--dry-run",
    ]


def _mapped_argv(*extra):
    return [
        "--source", "mapped",
        "--source-format", "grib2",
        "--mapping", "/case/mapping.json",
        "--composition", "/case/composition.json",
        "--input", "/data/source-f000.grb2",
        "--supplement", "terrain=/data/terrain.grb2",
        "--provenance", "terrain=/data/terrain-provenance.md",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--source-sha256s", "/case/input-manifest.json",
        "--source-sha256s-sha256", "abc123",
        "--output-root", "/output",
        *extra, "--dry-run",
    ]


def _twentycr_argv(*extra):
    return [
        "--source", "20crv3",
        "--source-manifest", "/case/member072.manifest.json",
        "--source-manifest-sha256", "abc123",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output",
        *extra, "--dry-run",
    ]


# ---------------------------------------------------------------------------
# The DEFAULT engine composes in process, so the bare door needs no tools
# ---------------------------------------------------------------------------

def test_bare_packaged_profile_prep_composes_in_the_engine_with_no_tools(
        tmp_path, monkeypatch, capsys):
    """THE deliverable: ``gpuwm prep --source hrrr-prs`` on a bare wheel.

    Asserted on an estate where the ladder answers nothing and there is
    no crate to build in -- the configuration in which the pre-compose
    door refused with ``--grib2-dump is required for mapped grib2`` and
    the configuration every wheel install starts in.  It reaches a
    composed command, and the command names no decoder tool, because the
    engine decodes and composes these bytes itself.
    """

    _bare_estate(tmp_path, monkeypatch)

    assert main(_packaged_profile_argv()) == 0
    command = capsys.readouterr().out
    assert "-m gpuwm.mapped_direct" in command
    # The packaged profile still decides the declarative arguments...
    assert "rw-wps-hrrr-prs-grib2.mapping.json" in command.replace("\\", "/")
    # ...and no subprocess decoder is named at all.
    for flag in _TOOL_FLAGS:
        assert flag not in command, flag


def test_bare_mapped_grib2_prep_composes_in_the_engine_with_no_tools(
        tmp_path, monkeypatch, capsys):
    _bare_estate(tmp_path, monkeypatch)

    assert main(_mapped_argv()) == 0
    command = capsys.readouterr().out
    assert "-m gpuwm.mapped_direct" in command
    for flag in _TOOL_FLAGS:
        assert flag not in command, flag


@pytest.mark.parametrize("argv_for", [_packaged_profile_argv, _mapped_argv],
                         ids=["packaged-profile", "generic-mapped"])
def test_staged_tools_are_not_forwarded_by_the_default_route(
        argv_for, tmp_path, monkeypatch, capsys):
    """Staging bridges must not un-default the engine.

    Named breakage: ``--grib2-inventory``/``--grib2-dump`` in a composed
    command is the spelling ``_mapped_engine_choice`` reads as "this
    caller pinned a subprocess decoder", so it routes the child to the
    PYTHON engine.  A door that forwarded whatever the ladder happened
    to find would therefore give a box that ran ``gpuwm fetch-bridges``
    the Python route and a box that never did the Rust route, from the
    same command line.  The tools exist on disk here and stay out.
    """

    staged = _staged_tools(tmp_path, monkeypatch)

    assert main(argv_for()) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    for flag in _TOOL_FLAGS:
        assert flag not in command, flag
    for name, path in staged.items():
        assert path.resolve().as_posix() not in command, name


# ---------------------------------------------------------------------------
# The WORKAROUND engine still runs the two executables, so the ladder holds
# ---------------------------------------------------------------------------

def test_packaged_profile_prep_resolves_the_tools_from_the_ladder_on_python(
        tmp_path, monkeypatch, capsys):
    """``GPUWM_MAPPED_ENGINE=python`` needs the tools, and gets them bare.

    This is the original papercut's row, kept EXECUTED rather than
    retired with the default: the workaround engine really does launch
    ``grib2_inventory`` and ``grib2_dump``, so a user who reaches for it
    must not be told to go find flags for files ``gpuwm fetch-bridges``
    already staged.
    """

    _the_workaround_engine(monkeypatch)
    staged = _staged_tools(tmp_path, monkeypatch)

    assert main(_packaged_profile_argv()) == 0
    command = capsys.readouterr().out
    assert "-m gpuwm.mapped_direct" in command
    assert "rw-wps-hrrr-prs-grib2.mapping.json" in command.replace("\\", "/")
    for name, path in staged.items():
        assert path.resolve().as_posix() in command.replace("\\", "/"), name


def test_mapped_grib2_prep_resolves_the_tools_from_the_ladder_on_python(
        tmp_path, monkeypatch, capsys):
    _the_workaround_engine(monkeypatch)
    staged = _staged_tools(tmp_path, monkeypatch)

    assert main(_mapped_argv()) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.mapped_direct" in command
    for path in staged.values():
        assert path.resolve().as_posix() in command


def test_bare_twentycr_run_composes_in_the_engine_with_no_tools(
        tmp_path, monkeypatch, capsys):
    """The member door follows the engine table like every composed door.

    This row's ancestor asserted the OPPOSITE -- a bare member prep
    resolving both subprocess tools -- which was the last entry in
    ``mapped_engine_bridge.ENGINE_GAPS``.  The runner now routes through
    ``decode_composed_source`` and reads the raw record inventory in
    process, so a forwarded tool path here would be the un-defaulting
    spelling: ``_mapped_engine_choice`` reads it as an explicit
    Python-engine pin on exactly the boxes that staged the tools.
    """

    staged = _staged_tools(tmp_path, monkeypatch)

    assert main(_twentycr_argv()) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.twentycrv3_wrf" in command
    assert "--grib2-inventory" not in command
    assert "--grib2-dump" not in command
    for path in staged.values():
        assert path.resolve().as_posix() not in command


def test_twentycr_python_engine_run_resolves_the_tools_from_the_ladder(
        tmp_path, monkeypatch, capsys):
    """The workaround half: the Python engine still needs the tools.

    ``GPUWM_MAPPED_ENGINE=python`` decodes the member series through the
    subprocess pair, so the door resolves both from the ladder and
    forwards them -- the same shape every packaged GRIB2 profile keeps on
    that engine, held here on the member door so the workaround cannot
    quietly lose its instrument.
    """

    staged = _staged_tools(tmp_path, monkeypatch)
    monkeypatch.setenv("GPUWM_MAPPED_ENGINE", "python")

    assert main(_twentycr_argv()) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.twentycrv3_wrf" in command
    for path in staged.values():
        assert path.resolve().as_posix() in command


# ---------------------------------------------------------------------------
# The flags stay overrides, per tool
# ---------------------------------------------------------------------------

def test_explicit_tool_flags_override_the_ladder(
        tmp_path, monkeypatch, capsys):
    """No engine pin: naming a tool IS the request for the Python engine."""

    _staged_tools(tmp_path, monkeypatch)

    assert main(_packaged_profile_argv(
        "--grib2-inventory", "/mine/grib2_inventory",
        "--grib2-dump", "/mine/grib2_dump")) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "/mine/grib2_inventory" in command
    assert "/mine/grib2_dump" in command
    assert tmp_path.as_posix() not in command


def test_one_explicit_flag_keeps_the_other_on_the_ladder(
        tmp_path, monkeypatch, capsys):
    """Per-tool override, exactly like the per-tool environment variables."""

    staged = _staged_tools(tmp_path, monkeypatch)

    assert main(_mapped_argv(
        "--grib2-inventory", "/mine/grib2_inventory")) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "/mine/grib2_inventory" in command
    assert staged["grib2_dump"].resolve().as_posix() in command


# ---------------------------------------------------------------------------
# Refusals: install-aware, and never a demand for a flag
# ---------------------------------------------------------------------------

def test_missing_tools_refuse_with_the_install_aware_remedy(
        tmp_path, monkeypatch, capsys):
    """On the engine that really needs them, and only there.

    The default engine resolves nothing on this estate and prepares
    fine (see the rows above), so the refusal is asserted where it is
    still owed: the workaround engine, which cannot run without the two
    executables.  A demand for ``--grib2-dump`` here would be the
    original papercut, one engine over.
    """

    _the_workaround_engine(monkeypatch)
    _bare_estate(tmp_path, monkeypatch)

    assert main(_packaged_profile_argv()) == EXIT_CONFIG
    error = capsys.readouterr().err
    assert "not installed here" in error
    assert "Searched, in order" in error
    # The remedy is doctor's install-aware one, not a flag demand.
    assert "is required for mapped grib2" not in error
    assert "--grib2-inventory" in error  # named as the override it is


def test_a_stale_tool_refuses_by_contract_when_nothing_can_rebuild_it(
        tmp_path, monkeypatch, capsys):
    """A binary predating the decoder contract must not reach the route."""

    _the_workaround_engine(monkeypatch)
    stale = {}
    for name in ("grib2_inventory", "grib2_dump"):
        path = tmp_path / bridges.executable_name(name)
        path.write_bytes(b"MZ\0\0not-the-contract")
        stale[name] = path
    monkeypatch.setattr(bridges, "find_bridge", lambda name: stale.get(name))
    monkeypatch.setattr(mapped_source, "_grib2_tools_crate",
                        lambda: tmp_path / "no-such-crate")

    assert main(_packaged_profile_argv()) == EXIT_CONFIG
    error = capsys.readouterr().err
    assert "predates" in error


def test_usage_errors_lead_even_when_the_tools_cannot_resolve(
        tmp_path, monkeypatch, capsys):
    """A vocabulary mistake is reported before the estate is consulted.

    On the workaround engine deliberately: there the unresolvable
    estate WOULD refuse if consulted, so the leading usage error is
    proved by ordering rather than by the default engine never asking.
    """

    _the_workaround_engine(monkeypatch)
    _bare_estate(tmp_path, monkeypatch)

    result = main(_mapped_argv("--gfs-series", "/wrong/route.tsv"))
    assert result == EXIT_USAGE
    error = capsys.readouterr().err
    assert "--gfs-series is not used by --source mapped" in error
    assert "not installed here" not in error


# ---------------------------------------------------------------------------
# One resolver: the front door and the engine cannot disagree
# ---------------------------------------------------------------------------

def test_the_front_door_resolves_through_the_engines_own_resolver(
        tmp_path, monkeypatch, capsys):
    """The resolver of record is ``mapped_source._build_grib2_tools``.

    Two resolvers with different answers is the defect
    ``bridges.resolve_source_decoder`` exists to prevent; asserting the
    call keeps the front door from growing a private ladder later.

    Both arms, because "which resolver" and "whether to resolve at all"
    are one question here: on the workaround engine the door consults
    the engine's own function and forwards exactly what it returns; on
    the default engine it must not consult it, since resolving would
    fail on a bare wheel and succeeding would pin the Python engine.
    """

    staged = _staged_tools(tmp_path, monkeypatch)
    calls = []
    real = mapped_source._build_grib2_tools

    def _counted():
        calls.append(True)
        return real()

    monkeypatch.setattr(mapped_source, "_build_grib2_tools", _counted)

    assert main(_packaged_profile_argv()) == 0
    assert calls == [], (
        "the default route consulted the GRIB2 ladder; resolving there "
        "either fails on a bare wheel or pins the Python engine")
    command = capsys.readouterr().out.replace("\\", "/")
    for path in staged.values():
        assert path.resolve().as_posix() not in command

    _the_workaround_engine(monkeypatch)
    assert main(_packaged_profile_argv()) == 0
    assert calls, "the front door did not consult the shared resolver"
    command = capsys.readouterr().out.replace("\\", "/")
    for path in staged.values():
        assert path.resolve().as_posix() in command


# ---------------------------------------------------------------------------
# GRIB1 joins the same contract: the engine decodes it, so the door is bare
# ---------------------------------------------------------------------------

def _mapped_grib1_argv(*extra):
    """The mapped GRIB1 route, spelled with no decoder flag at all."""

    return [
        "--source", "mapped",
        "--source-format", "grib1",
        "--mapping", "/case/mapping.json",
        "--composition", "/case/composition.json",
        "--input", "/data/source-f000.grb",
        "--supplement", "terrain=/data/terrain.grb",
        "--provenance", "terrain=/data/terrain-provenance.md",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--source-sha256s", "/case/input-manifest.json",
        "--source-sha256s-sha256", "abc123",
        "--output-root", "/output",
        *extra, "--dry-run",
    ]


def test_bare_mapped_grib1_prep_needs_no_bridge_on_the_default_engine(
        tmp_path, monkeypatch, capsys):
    """THE deliverable for GRIB1, and the reason this test exists.

    Named breakage, measured at the union of the two ERA5 port lanes:
    ``gpuwm prep --source mapped --source-format grib1`` refused with
    ``--bridge is required for mapped grib1`` on the DEFAULT engine,
    which decodes GRIB1 in process and spawns no bridge at all.  The
    flag was not merely redundant -- naming a decoder tool is the
    spelling that routes a run back to the PYTHON engine, so the only
    reachable GRIB1 prep was one that could not use the ported decode.
    An engine-proven format with no bare front door is not shipped.

    Asserted on a bare estate, so a box that never staged a bridge --
    which is now every box, for this route -- gets a working command.
    """

    _bare_estate(tmp_path, monkeypatch)
    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    monkeypatch.delenv("GPUWM_GRIB1_BRIDGE", raising=False)

    assert main(_mapped_grib1_argv()) == 0
    command = capsys.readouterr().out
    assert "-m gpuwm.mapped_direct" in command
    assert "--grib1-bridge" not in command, (
        "the bare GRIB1 door forwarded a bridge pin, which re-routes the "
        "decode to the Python engine and un-defaults the Rust engine")


def test_a_named_bridge_still_pins_the_python_engine_for_grib1(
        tmp_path, monkeypatch, capsys):
    """The override survives: ``--bridge`` still works and still pins.

    The flag is a WORKAROUND, not a mode.  It must keep reaching the
    Python engine for a user routing around a Rust decode defect, so
    this asserts the pin is forwarded rather than quietly dropped.
    """

    _bare_estate(tmp_path, monkeypatch)
    bridge = tmp_path / bridges.executable_name("grib1_bridge")
    bridge.write_bytes(b"MZ\0\0")

    assert main(_mapped_grib1_argv("--bridge", str(bridge))) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--grib1-bridge" in command
    assert bridge.resolve().as_posix() in command


def test_the_grib1_bridge_requirement_follows_the_capability_table(
        tmp_path, monkeypatch, capsys):
    """A gap that reopens brings its flag requirement back with it.

    Breakage prevented: dropping the requirement as a flat literal
    would leave a build whose engine cannot decode GRIB1 composing a
    bare command with no decoder at all, which fails deep in the run
    instead of at the door.  The requirement is therefore read off
    ``ENGINE_CAPABILITIES``, and this test proves it by taking GRIB1
    back out -- the same table the router and the docs read.
    """

    from gpuwm import mapped_engine_bridge as engine_bridge

    _bare_estate(tmp_path, monkeypatch)
    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    monkeypatch.setitem(
        engine_bridge.ENGINE_CAPABILITIES, "decode", frozenset({"grib2"}))

    assert main(_mapped_grib1_argv()) == EXIT_USAGE
    assert "--bridge is required for mapped grib1" in capsys.readouterr().err


def test_an_explicit_python_engine_grib1_prep_still_needs_its_bridge(
        tmp_path, monkeypatch, capsys):
    """The workaround door names what it needs, because it really needs it.

    ``--mapped-engine python`` on GRIB1 runs the out-of-process bridge,
    so composing a bridge-less command there would fail in the run.
    """

    _bare_estate(tmp_path, monkeypatch)
    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)

    assert main(_mapped_grib1_argv("--mapped-engine", "python")) == EXIT_USAGE
    assert "--bridge is required for mapped grib1" in capsys.readouterr().err
