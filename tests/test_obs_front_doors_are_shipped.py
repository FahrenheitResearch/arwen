"""A front door a refusal names must be one the pipeline actually ships.

The ``rw_mpas_convert`` precedent, one wave later: ``rw_mrms``,
``rw_stage4``, ``rw_asos`` and ``rw_goes`` were written, tested,
committed, and resolved out of ``~/.gpuwm/bridges`` by
:mod:`gpuwm.obs.frontdoor` -- and none of them was in
:data:`gpuwm.bridge_assets.BUNDLED_ARTIFACTS`, so ``gpuwm
fetch-bridges``, the command their own refusals led with, staged a
complete bundle and the refusal repeated verbatim.  Committed is not
shipped.

Every test here is a property of the SHIPPING path, not of the source
tree: what the bundle table declares, what the release workflow builds,
what the stamp contract can prove, and whether a refusal names a remedy
that can supply the thing it is refusing about.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm import bridge_assets, bridges
from gpuwm.obs import frontdoor

#: The four doors this test file exists for.
OBS_ARTIFACTS = ("rw_mrms", "rw_stage4", "rw_asos", "rw_goes")


# ---------------------------------------------------------------------------
# 1. the bundle table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", OBS_ARTIFACTS)
def test_every_resolvable_front_door_is_a_bundled_artifact(name):
    """FAILS before the fix: the refusal's remedy could not supply it."""

    names = {a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
    assert name in names, (
        f"{name} is resolved out of ~/.gpuwm/bridges by "
        "gpuwm.obs.frontdoor and refused by name when absent, but "
        "gpuwm fetch-bridges does not stage it -- so the command the "
        "refusal names cannot help.  That is rw_mpas_convert again.")


@pytest.mark.parametrize("name", OBS_ARTIFACTS)
def test_the_bundled_entry_uses_the_resolution_ladders_own_env_var(name):
    """A staged bundle and a hand-built tree, found by one code path."""

    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == name]
    door, = [d for d in frontdoor.FRONT_DOORS.values() if d.name == name]
    assert artifact.env_var == door.env_var
    assert artifact.kind == "executable"
    assert artifact.crate == bridges.RUSTWX_CRATE_RELATIVE


def test_the_front_door_table_and_the_bundle_table_agree():
    """Neither may gain a door the other does not know about."""

    resolved = {d.name for d in frontdoor.FRONT_DOORS.values()}
    bundled = {a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
    assert resolved <= bundled, (
        f"gpuwm.obs.frontdoor resolves {sorted(resolved - bundled)} "
        "which no bundle carries")


# ---------------------------------------------------------------------------
# 2. the build half -- the stamp the release cut refuses without
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("relative", (
    "tools/rustwx/crates/rw-obs/build.rs",
    "tools/rustwx/crates/rw-goes/build.rs",
))
def test_the_new_crates_inject_the_source_revision(relative):
    """FAILS before the fix: neither crate had a build.rs at all.

    Without it the binaries carry no ``GPUWM_BRIDGE_SOURCE_REV`` stamp,
    and ``build_bridge_bundle.py pin`` refuses to pin them -- so adding
    them to the table alone would have produced a cut that could not
    ship them.
    """

    script = REPO_ROOT / relative
    assert script.is_file(), f"{script} is missing"
    text = script.read_text(encoding="utf-8")
    assert "rustc-env=GPUWM_BRIDGE_SOURCE_REV" in text


@pytest.mark.parametrize("relative", (
    "tools/rustwx/crates/rw-obs/src/bin/mrms.rs",
    "tools/rustwx/crates/rw-obs/src/bin/stage4.rs",
    "tools/rustwx/crates/rw-obs/src/bin/asos.rs",
    "tools/rustwx/crates/rw-goes/src/main.rs",
))
def test_every_new_entry_point_embeds_the_stamp(relative):
    """A build script that injects nothing anybody reads proves nothing."""

    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert "GPUWM_BRIDGE_SOURCE_REV_STAMP" in text
    assert "black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP)" in text, (
        f"{relative} declares the stamp but never references it in "
        "main, so the linker may discard it and the cut would refuse "
        "a binary that was built correctly")


# ---------------------------------------------------------------------------
# 3. the remedy: no refusal may name a command that cannot supply it
# ---------------------------------------------------------------------------

def _pinned_bundle(monkeypatch, artifacts):
    """A pins document pinning exactly ``artifacts`` for this host."""

    platform = bridge_assets.host_platform()
    if platform is None:
        pytest.skip("no bundle is published for this platform")
    binaries = tuple(
        bridge_assets.BinaryPin(
            artifact=name, filename=f"{name}.exe", bytes=1, sha256="0" * 64)
        for name in artifacts)
    bundle = bridge_assets.BundlePin(
        platform=platform, filename="bundle.zip", bytes=1,
        sha256="0" * 64, binaries=binaries)
    pins = bridge_assets.BridgePins(release="vTEST",
                                    platforms={platform: bundle})
    monkeypatch.setattr(bridge_assets, "load_pins", lambda *a, **k: pins)
    return pins


def test_the_offer_is_silent_for_an_artifact_the_bundle_lacks(monkeypatch):
    """THE bug, in one assertion.

    ``prebuilt_bundle_offer`` used to answer "yes, run gpuwm
    fetch-bridges" whenever *a* bundle existed for the platform, without
    ever asking whether *this* artifact was in it.  Running the offered
    command then printed "already staged and pin-valid" and the refusal
    repeated word for word.
    """

    _pinned_bundle(monkeypatch, ("rw_fetch",))
    assert bridges.prebuilt_bundle_offer("rw_fetch") is not None
    assert bridges.prebuilt_bundle_offer("rw_mrms") is None, (
        "the bundle does not carry rw_mrms, so offering "
        "`gpuwm fetch-bridges` as its remedy is a remedy that cannot "
        "supply it")


def test_the_offer_fires_when_the_bundle_does_carry_it(monkeypatch):
    """The instrument, validated the other direction."""

    _pinned_bundle(monkeypatch, ("rw_fetch", "rw_mrms"))
    offer = bridges.prebuilt_bundle_offer("rw_mrms")
    assert offer is not None
    assert any("gpuwm fetch-bridges" in line for line in offer)


def _as_a_wheel(monkeypatch, tmp_path):
    """Make the remedy builders answer as they do on a pip install.

    ``artifact_remedy`` branches on whether the crate directory exists
    beside the package, so in a checkout it correctly answers "cargo
    build" and never mentions either wheel route.  A test that asserted
    the wheel wording from a checkout would be measuring the wrong
    branch -- the instrument, not the product.  Pointing the package
    parent at an empty directory is what a wheel install IS.
    """

    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)


def test_a_front_door_remedy_never_leads_with_an_unsuppliable_command(
        monkeypatch, tmp_path):
    """The composed refusal a user actually reads, on a wheel."""

    _pinned_bundle(monkeypatch, ("rw_fetch",))
    _as_a_wheel(monkeypatch, tmp_path)
    remedy = frontdoor.MRMS.remedy()
    assert "gpuwm fetch-bridges" not in remedy, (
        "the MRMS refusal leads with a download that does not contain "
        "rw_mrms")
    assert "git clone" in remedy, "and it must still name a route that works"


def test_the_same_remedy_does_lead_with_the_download_once_it_is_bundled(
        monkeypatch, tmp_path):
    """The instrument, validated the other direction."""

    _pinned_bundle(monkeypatch, ("rw_fetch", "rw_mrms"))
    _as_a_wheel(monkeypatch, tmp_path)
    assert "gpuwm fetch-bridges" in frontdoor.MRMS.remedy()


# ---------------------------------------------------------------------------
# 4. the CPU bridge refusal, which named no remedy at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("as_wheel", (False, True))
def test_the_cpu_preprocess_refusal_now_carries_a_remedy(
        monkeypatch, tmp_path, as_wheel):
    """The one resolver whose message never said how to fix it.

    Checked on both install shapes, because "carries a remedy" has two
    correct answers and the defect -- listing four paths and stopping --
    was the same on both.
    """

    from gpuwm.ingest import cpu_backend

    monkeypatch.setattr(cpu_backend, "cpu_bridge_candidates",
                        lambda: (tmp_path / "absent.dll",))
    monkeypatch.setattr(bridges, "find_artifact", lambda *a, **k: None)
    if as_wheel:
        _pinned_bundle(monkeypatch, ("gpuwm_preprocess_cpu",))
        _as_a_wheel(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        cpu_backend.resolve_cpu_bridge()
    message = str(caught.value)
    assert "searched" in message
    expected = ("gpuwm fetch-bridges" if as_wheel else "cargo build")
    assert expected in message, (
        "the CPU preprocessing bridge refusal lists the paths it "
        f"searched and stops; it is recoverable and never says so "
        f"(expected {expected!r} in: {message})")


# ---------------------------------------------------------------------------
# 5. fetch-bridges --help must name what it actually stages
# ---------------------------------------------------------------------------

def test_the_help_summary_is_derived_from_the_artifact_table():
    """A hand-written inventory drifts on the first addition."""

    summary = bridge_assets.staged_artifact_summary()
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        assert artifact.name in summary, (
            f"fetch-bridges --help does not name {artifact.name}, which "
            "it stages")


def test_the_help_text_names_the_radar_and_observation_front_doors():
    """The audit's H24, in the shipped parser rather than in prose."""

    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    bridge_assets.register_cli(sub)
    # format_help() of the SUBPARSER is what `gpuwm fetch-bridges
    # --help` prints, which is the surface the audit measured.
    help_text = sub.choices["fetch-bridges"].format_help()
    for name in ("rw_nexrad", *OBS_ARTIFACTS, "region_global_dealias"):
        assert name in help_text, (
            f"fetch-bridges --help stages {name} and does not say so")


# ---------------------------------------------------------------------------
# 6. one subcommand tree -- both lanes' invocations, on the product parser
# ---------------------------------------------------------------------------
#
# `gpuwm obs` was defined twice when these lanes were written apart: an
# instrument passthrough (`gpuwm obs mrms ...`) and a decode/grid tree
# (`gpuwm obs radar ...`), each registering the name "obs" on the same
# subparsers object.  Whichever registrar ran second raised at import.
# They are one tree now, and these hold it that way: a future lane that
# rebuilds either half has to keep the other half reachable.

def _subparsers(parser):
    import argparse

    for action in parser._actions:                          # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return action.choices
    return {}


def _obs_parser():
    from gpuwm.cli import build_parser

    obs = _subparsers(build_parser()).get("obs")
    assert obs is not None, "`gpuwm obs` is not a subcommand of the product CLI"
    return obs


@pytest.mark.parametrize("instrument", sorted(frontdoor.FRONT_DOORS))
def test_every_resolvable_instrument_has_its_own_door(instrument):
    """A door in the resolver's table and not on the parser is unreachable."""

    assert instrument in _subparsers(_obs_parser()), (
        f"gpuwm.obs.frontdoor resolves {instrument!r} and `gpuwm obs "
        f"{instrument}` is not a command, so the binary is reachable only "
        "by path")


def test_no_instrument_door_exists_without_a_resolver():
    """The other direction: a parser entry with no table row cannot run."""

    from gpuwm.obs import cli as obs_cli

    assert set(obs_cli._INSTRUMENTS) == set(frontdoor.FRONT_DOORS), (  # noqa: SLF001
        "gpuwm.obs.cli._INSTRUMENTS and gpuwm.obs.frontdoor.FRONT_DOORS "
        "disagree; one of them is naming a door that does not exist")


def test_the_radar_tree_survived_the_instrument_doors():
    """The European radar lane's proven invocations, on the product parser."""

    radar = _subparsers(_obs_parser()).get("radar")
    assert radar is not None, "`gpuwm obs radar` is gone"
    assert set(_subparsers(radar)) >= {
        "doctor", "volumes", "pack", "nyquist", "sites", "grid"}


@pytest.mark.parametrize("argv,expected", (
    (["obs", "radar", "doctor"], "_radar_doctor"),
    (["obs", "mrms"], "_instrument_main"),
    (["obs", "odim"], "_instrument_main"),
    (["obs"], "_obs_estate"),
))
def test_the_product_parser_routes_both_lanes_invocations(argv, expected):
    """Parsing, not just registration: each shape reaches its own handler."""

    from gpuwm.cli import build_parser

    namespace = build_parser().parse_args(argv)
    assert namespace.func.__name__ == expected


def test_an_instrument_door_forwards_its_arguments_untouched():
    """The passthrough contract: gpuwm re-spells none of the binary's grammar."""

    from gpuwm.cli import build_parser

    namespace = build_parser().parse_args(
        ["obs", "mrms", "fetch", "--window", "2026-01-01T00Z", "--out", "x"])
    assert namespace.argv == [
        "fetch", "--window", "2026-01-01T00Z", "--out", "x"]
