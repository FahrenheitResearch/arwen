"""``gpuwm fetch-bridges``: staging, verification, and refusal contract.

The wheel ships no compiled Rust, so for a pip install this fetch path
IS the install path for eight artifacts: every byte must be verified
against the packaged pins before anything lands in ``~/.gpuwm/bridges``,
a mismatch must be refused rather than installed, and the other
platform's bundle must be recognised as such instead of half-staged.

Nothing here fabricates a hash.  The synthetic-payload tests pin bytes
this file just wrote, by hashing them; the integration tests pack a real
bundle out of the Rust artifacts this machine has actually built and pin
*those* bytes with the release tool the release workflow runs.  Every
test but one runs entirely offline, and that one is gated twice.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from gpuwm import bridge_assets, bridges

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = REPO_ROOT / "tools" / "build_bridge_bundle.py"


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**{"from_dir": None, "dest": None,
                                 "keep_bundle": False, "list": False,
                                 **kwargs})


def _pin(payload: bytes, artifact: str, filename: str
         ) -> bridge_assets.BinaryPin:
    """A pin computed from bytes that exist, never a literal."""

    return bridge_assets.BinaryPin(
        artifact=artifact, filename=filename, bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest())


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buffer.getvalue()


def _synthetic_bundle(directory: Path, *, platform: str = "win-x86_64",
                      names: tuple[str, ...] = ("alpha.bin", "beta.bin"),
                      corrupt: str | None = None
                      ) -> tuple[Path, bridge_assets.BundlePin]:
    """A bundle whose pins are computed from the bytes it carries.

    ``corrupt`` names a member whose archived bytes differ from the ones
    its pin was computed over -- the only way to build a bundle that
    must be refused without writing a hash by hand.  The member names
    are deliberately not real artifact names: these tests are about the
    staging contract, and a name in
    :data:`gpuwm.bridges.BRIDGE_ABI_MARKERS` would additionally demand a
    contract marker no synthetic payload carries (the real-build tests
    below cover that check).
    """

    directory.mkdir(parents=True, exist_ok=True)
    payloads = {name: os.urandom(2048) + name.encode() for name in names}
    pins = tuple(_pin(payloads[name], name.split(".")[0], name)
                 for name in names)
    archived = dict(payloads)
    if corrupt is not None:
        # Same length, different bytes: the size gate must not be what
        # catches this, or the SHA-256 gate is never exercised.
        flipped = bytearray(payloads[corrupt])
        flipped[0] ^= 0xFF
        archived[corrupt] = bytes(flipped)
    blob = _zip_bytes(archived)
    archive = directory / f"gpuwm-bridges-v0-test-{platform}.zip"
    archive.write_bytes(blob)
    bundle = bridge_assets.BundlePin(
        platform=platform, filename=archive.name, bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(), binaries=pins)
    return archive, bundle


def _install_pins(monkeypatch, bundle: bridge_assets.BundlePin,
                  release: str = "v0-test") -> bridge_assets.BridgePins:
    """Point the command at pins built from real bytes, on this platform.

    Patching the loader rather than writing into the installed package
    keeps the shipped pins document untouched, and pinning the host
    platform to the bundle's makes these tests say the same thing on a
    Windows box and a Linux one.
    """

    pins = bridge_assets.BridgePins(release=release,
                                    platforms={bundle.platform: bundle})
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: bundle.platform)
    return pins


# ---------------------------------------------------------------------------
# Staging semantics, with synthetic artifacts
# ---------------------------------------------------------------------------

def test_stage_from_bundle_verifies_then_installs(tmp_path):
    archive, bundle = _synthetic_bundle(tmp_path)
    dest = tmp_path / "dest"

    installed = bridge_assets.stage_from_bundle(
        archive, bundle, dest, progress=lambda _line: None)

    assert [p.name for p in installed] == ["alpha.bin", "beta.bin"]
    for pin in bundle.binaries:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)


def test_a_corrupt_member_is_refused_and_never_installed(tmp_path):
    archive, bundle = _synthetic_bundle(tmp_path, corrupt="beta.bin")
    dest = tmp_path / "dest"

    with pytest.raises(bridge_assets.BridgeAssetError, match="SHA-256"):
        bridge_assets.stage_from_bundle(archive, bundle, dest,
                                        progress=lambda _line: None)

    # The refusal is per-file and it stops there: the good member ahead
    # of it installed, the bad one did not, and nothing half-written is
    # left behind for the resolver to find.
    assert (dest / "alpha.bin").is_file()
    assert not (dest / "beta.bin").exists()
    assert not (dest / f"{bridge_assets.ARCHIVE_SUBDIR}-stage").exists()


def test_a_member_of_the_wrong_size_is_refused_before_the_hash(tmp_path):
    """Size and hash are both gates, and the size one names itself."""

    archive, bundle = _synthetic_bundle(tmp_path)
    wrong = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=bundle.bytes, sha256=bundle.sha256,
        binaries=(bridge_assets.BinaryPin(
            bundle.binaries[0].artifact, bundle.binaries[0].filename,
            bundle.binaries[0].bytes + 1, bundle.binaries[0].sha256),))

    with pytest.raises(bridge_assets.BridgeAssetError, match="bytes"):
        bridge_assets.stage_from_bundle(archive, wrong, tmp_path / "dest",
                                        progress=lambda _line: None)


def test_the_other_platforms_bundle_is_refused_by_name(tmp_path):
    """A linux bundle staged against windows pins names both shapes.

    The failure mode this replaces is silent: a bundle whose members are
    all called something else stages nothing, and the user learns about
    it from a decoder that is still missing.
    """

    linux_archive, linux = _synthetic_bundle(
        tmp_path / "linux", platform="linux-x86_64",
        names=("grib1_bridge", "rw_fetch"))
    windows = bridge_assets.BundlePin(
        platform="win-x86_64", filename=linux.filename, bytes=linux.bytes,
        sha256=linux.sha256,
        binaries=tuple(
            bridge_assets.BinaryPin(pin.artifact, f"{pin.filename}.exe",
                                    pin.bytes, pin.sha256)
            for pin in linux.binaries))

    with pytest.raises(bridge_assets.BridgeAssetError) as failure:
        bridge_assets.stage_from_bundle(
            linux_archive, windows, tmp_path / "dest",
            progress=lambda _line: None)
    message = str(failure.value)
    assert "grib1_bridge.exe" in message      # what was pinned
    assert "grib1_bridge" in message          # what the archive holds
    assert "not the win-x86_64 bundle" in message
    assert not (tmp_path / "dest" / "grib1_bridge").exists()


def test_an_existing_pin_valid_file_is_left_alone(tmp_path, monkeypatch):
    archive, bundle = _synthetic_bundle(tmp_path)
    _install_pins(monkeypatch, bundle)
    dest = tmp_path / "dest"
    bridge_assets.stage_from_bundle(archive, bundle, dest,
                                    progress=lambda _line: None)
    before = {pin.filename: (dest / pin.filename).stat().st_mtime_ns
              for pin in bundle.binaries}

    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(tmp_path))) == 0
    after = {pin.filename: (dest / pin.filename).stat().st_mtime_ns
             for pin in bundle.binaries}
    assert after == before


def test_a_stale_artifact_is_replaced_not_refused(tmp_path, monkeypatch,
                                                  capsys):
    """The deliberate divergence from ``fetch-tables``, pinned.

    A physics table that differs is the operator's file and is never
    overwritten.  A bridge executable that differs is yesterday's build,
    and leaving it in place is exactly the skew that made 1.1.0
    preparations fail while `doctor` reported the binary `ok`.
    """

    archive, bundle = _synthetic_bundle(tmp_path)
    _install_pins(monkeypatch, bundle)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "alpha.bin").write_bytes(b"yesterday's build")

    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "do not match this release's pins and will be replaced" in out
    assert bridge_assets.matches_pin(dest / "alpha.bin", bundle.binaries[0])


def test_from_dir_stages_loose_artifacts_under_the_same_pins(tmp_path):
    """What an air-gapped operator has: files, not an archive."""

    archive, bundle = _synthetic_bundle(tmp_path)
    loose = tmp_path / "loose"
    loose.mkdir()
    with zipfile.ZipFile(archive) as zf:
        for pin in bundle.binaries:
            (loose / pin.filename).write_bytes(zf.read(pin.filename))
    dest = tmp_path / "dest"

    installed = bridge_assets.stage_from_dir(loose, bundle, dest,
                                             progress=lambda _line: None)
    assert len(installed) == len(bundle.binaries)
    for pin in bundle.binaries:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)


def test_from_dir_refuses_a_directory_missing_an_artifact(tmp_path):
    archive, bundle = _synthetic_bundle(tmp_path)
    loose = tmp_path / "loose"
    loose.mkdir()
    with zipfile.ZipFile(archive) as zf:
        (loose / "alpha.bin").write_bytes(zf.read("alpha.bin"))

    with pytest.raises(bridge_assets.BridgeAssetError, match="beta.bin"):
        bridge_assets.stage_from_dir(loose, bundle, tmp_path / "dest",
                                     progress=lambda _line: None)


def test_from_dir_refuses_a_bundle_whose_own_bytes_drifted(tmp_path):
    """``--from`` verifies the archive itself, not only its members."""

    archive, bundle = _synthetic_bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"appended")

    with pytest.raises(bridge_assets.BridgeAssetError, match="bytes"):
        bridge_assets.stage_from_dir(tmp_path, bundle, tmp_path / "dest",
                                     progress=lambda _line: None)


# ---------------------------------------------------------------------------
# Download: resume, restart, and the size gate
# ---------------------------------------------------------------------------

class _Response(io.BytesIO):
    def __init__(self, payload: bytes, status: int):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class _RangeServer:
    """A urlopen stand-in that either honours Range or ignores it.

    Two values of one dimension, because the resume path has to be
    correct under both: a server that continues the transfer, and one
    (or a ``file:`` URL) that has never heard of a range and answers
    with the whole object.  Appending to a partial in the second case is
    how a naive resume produces a corrupt archive.
    """

    def __init__(self, payload: bytes, *, honour_range: bool):
        self.payload = payload
        self.honour_range = honour_range
        self.requests: list[str | None] = []

    def __call__(self, request):
        header = request.headers.get("Range")
        self.requests.append(header)
        if header and self.honour_range:
            start = int(header.split("=")[1].split("-")[0])
            return _Response(self.payload[start:], 206)
        return _Response(self.payload, 200)


@pytest.mark.parametrize("honour_range", (False, True))
def test_a_partial_download_finishes_correctly_either_way(tmp_path,
                                                          honour_range):
    payload = os.urandom(64 * 1024)
    dest = tmp_path / "bundle.zip"
    dest.write_bytes(payload[:1024])            # an interrupted run
    server = _RangeServer(payload, honour_range=honour_range)
    notes: list[str] = []

    bridge_assets.download_bundle(
        "https://example.invalid/bundle.zip", dest,
        expected_bytes=len(payload), progress=notes.append,
        urlopen_fn=server)

    assert dest.read_bytes() == payload
    assert server.requests == ["bytes=1024-"]
    if honour_range:
        assert any("resuming at 1,024 B" in note for note in notes)
    else:
        assert any("ignored the resume range" in note for note in notes)


def test_a_partial_longer_than_the_pin_is_discarded(tmp_path):
    payload = os.urandom(8192)
    dest = tmp_path / "bundle.zip"
    dest.write_bytes(payload + b"extra bytes from somewhere else")
    server = _RangeServer(payload, honour_range=True)
    notes: list[str] = []

    bridge_assets.download_bundle(
        "https://example.invalid/bundle.zip", dest,
        expected_bytes=len(payload), progress=notes.append,
        urlopen_fn=server)

    assert dest.read_bytes() == payload
    assert server.requests == [None], "a discarded partial restarts at zero"
    assert any("discarded it and restarted" in note for note in notes)


def test_a_complete_partial_is_verified_rather_than_re_downloaded(tmp_path):
    payload = os.urandom(4096)
    dest = tmp_path / "bundle.zip"
    dest.write_bytes(payload)
    server = _RangeServer(payload, honour_range=True)

    bridge_assets.download_bundle(
        "https://example.invalid/bundle.zip", dest,
        expected_bytes=len(payload), progress=lambda _l: None,
        urlopen_fn=server)
    assert server.requests == [], "nothing left to download, nothing fetched"


def test_a_short_download_is_removed_and_refused(tmp_path):
    payload = os.urandom(4096)
    dest = tmp_path / "bundle.zip"
    server = _RangeServer(payload[:100], honour_range=False)

    with pytest.raises(bridge_assets.BridgeAssetError, match="expected"):
        bridge_assets.download_bundle(
            "https://example.invalid/bundle.zip", dest,
            expected_bytes=len(payload), progress=lambda _l: None,
            urlopen_fn=server)
    assert not dest.exists()


def test_a_bundle_whose_bytes_differ_never_reaches_staging(tmp_path):
    """The whole download path, against a served archive that lies."""

    archive, bundle = _synthetic_bundle(tmp_path)
    evil = bytearray(archive.read_bytes())
    evil[-1] ^= 0xFF
    server = _RangeServer(bytes(evil), honour_range=False)
    dest = tmp_path / "dest"
    pins = bridge_assets.BridgePins(release="v0-test",
                                    platforms={bundle.platform: bundle})

    with pytest.raises(bridge_assets.BridgeAssetError, match="SHA-256"):
        bridge_assets.fetch_bundle(pins, bundle, dest,
                                   progress=lambda _l: None,
                                   urlopen_fn=server)
    assert not (dest / "alpha.bin").exists()
    assert not (dest / bridge_assets.ARCHIVE_SUBDIR
                / bundle.filename).exists()


def test_a_verified_bundle_already_on_disk_is_not_downloaded_again(tmp_path):
    archive, bundle = _synthetic_bundle(tmp_path)
    dest = tmp_path / "dest"
    cache = dest / bridge_assets.ARCHIVE_SUBDIR
    cache.mkdir(parents=True)
    (cache / bundle.filename).write_bytes(archive.read_bytes())
    server = _RangeServer(b"", honour_range=False)
    notes: list[str] = []

    installed = bridge_assets.fetch_bundle(
        pins := bridge_assets.BridgePins(
            release="v0-test", platforms={bundle.platform: bundle}),
        bundle, dest, keep_bundle=True, progress=notes.append,
        urlopen_fn=server)
    assert pins.release == "v0-test"
    assert len(installed) == len(bundle.binaries)
    assert server.requests == []
    assert any("already downloaded and pin-verified" in note
               for note in notes)


def test_the_url_base_environment_override_wins(monkeypatch, tmp_path):
    _archive, bundle = _synthetic_bundle(tmp_path)
    pins = bridge_assets.BridgePins(release="v0-test",
                                    platforms={bundle.platform: bundle})
    assert bridge_assets.bundle_url(pins, bundle).startswith(
        f"{bridges.REPOSITORY_URL}/releases/download/v0-test/")
    monkeypatch.setenv(bridge_assets.ASSET_URL_BASE_ENV,
                       "https://mirror.invalid/gpuwm/")
    assert bridge_assets.bundle_url(pins, bundle) == (
        f"https://mirror.invalid/gpuwm/{bundle.filename}")


# ---------------------------------------------------------------------------
# The pins document
# ---------------------------------------------------------------------------

def test_the_packaged_pins_document_parses():
    """Whatever the release wrote, the runtime must be able to read.

    It is legitimate for a tree that has not been through a release cut
    to declare no platform -- that is what an unpinned document says,
    rather than carrying a hash nobody computed -- but the schema, and
    every field it does declare, are checked either way.
    """

    pins = bridge_assets.load_pins()
    for platform, bundle in pins.platforms.items():
        assert platform in bridge_assets.SUPPORTED_PLATFORMS
        assert pins.release, "a pinned platform needs a release to fetch from"
        expected = {
            bridge_assets.artifact_filename(artifact, platform)
            for artifact in bridge_assets.BUNDLED_ARTIFACTS}
        assert {pin.filename for pin in bundle.binaries} == expected


@pytest.mark.parametrize("mutation,pattern", (
    ({"schema": "something-else"}, "schema"),
    ({"platforms": []}, "platforms must be an object"),
    ({"platforms": {"solaris-sparc": {}}}, "unknown platform"),
    ({"release": ""}, "release must be"),
))
def test_a_malformed_pins_document_is_refused(mutation, pattern):
    payload = {"schema": bridge_assets.PINS_SCHEMA, "release": None,
               "platforms": {}}
    payload.update(mutation)
    with pytest.raises(bridge_assets.BridgeAssetError, match=pattern):
        bridge_assets.parse_pins(payload)


def test_a_pinned_platform_without_a_release_is_refused():
    """Pins with nowhere to download from are not pins."""

    payload = {
        "schema": bridge_assets.PINS_SCHEMA, "release": None,
        "platforms": {"win-x86_64": {
            "bundle": {"filename": "b.zip", "bytes": 1, "sha256": "0" * 64},
            "binaries": [{"artifact": "grib1_bridge",
                          "filename": "grib1_bridge.exe",
                          "bytes": 1, "sha256": "0" * 64}]}}}
    with pytest.raises(bridge_assets.BridgeAssetError, match="release"):
        bridge_assets.parse_pins(payload)


def test_a_hash_that_is_not_a_hash_is_refused():
    payload = {
        "schema": bridge_assets.PINS_SCHEMA, "release": "v0",
        "platforms": {"win-x86_64": {
            "bundle": {"filename": "b.zip", "bytes": 1,
                       "sha256": "not-a-digest"},
            "binaries": [{"artifact": "grib1_bridge",
                          "filename": "grib1_bridge.exe",
                          "bytes": 1, "sha256": "0" * 64}]}}}
    with pytest.raises(bridge_assets.BridgeAssetError, match="sha256"):
        bridge_assets.parse_pins(payload)


# ---------------------------------------------------------------------------
# Platform detection is a capability check
# ---------------------------------------------------------------------------

def test_the_platform_key_names_an_os_and_an_architecture(monkeypatch):
    """Both halves matter, and neither is about who is asking.

    An aarch64 Linux box and an x86-64 Linux box run different bytes,
    and so do the same architecture on Windows and on Linux.  A key that
    dropped either half would hand someone a bundle their machine
    cannot execute.
    """

    cases = {
        ("win32", "AMD64"): "win-x86_64",
        ("win32", "ARM64"): None,
        ("linux", "x86_64"): "linux-x86_64",
        ("linux", "aarch64"): None,
        ("darwin", "arm64"): None,
    }
    for (platform_name, machine), expected in cases.items():
        monkeypatch.setattr(bridge_assets.sys, "platform", platform_name)
        monkeypatch.setattr(bridge_assets.platform_module, "machine",
                            lambda m=machine: m)
        assert bridge_assets.host_platform() == expected, (
            f"{platform_name}/{machine}")


def test_artifact_filenames_agree_with_the_resolvers_on_this_host():
    """The bundle's names and the ladder's names cannot disagree.

    ``artifact_filename`` is parametric in the platform so the release
    tool can inspect both bundles from one machine; on the host it must
    produce exactly what :mod:`gpuwm.bridges` and the CPU backend look
    for, or a perfectly staged bundle resolves to nothing.
    """

    from gpuwm.ingest.cpu_backend import cpu_bridge_candidates

    host = bridge_assets.host_platform()
    if host is None:
        pytest.skip("no bundle platform for this host")
    library = cpu_bridge_candidates()[-1].name
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        produced = bridge_assets.artifact_filename(artifact, host)
        if artifact.kind == "library":
            assert produced == library
        else:
            assert produced == bridges.executable_name(artifact.name)


def test_both_platforms_spell_the_library_the_way_that_platform_does():
    """One dimension, two values: the extension is not a suffix guess."""

    library = next(a for a in bridge_assets.BUNDLED_ARTIFACTS
                   if a.kind == "library")
    executable = next(a for a in bridge_assets.BUNDLED_ARTIFACTS
                      if a.kind == "executable")
    assert bridge_assets.artifact_filename(library, "win-x86_64") == (
        f"{library.name}.dll")
    assert bridge_assets.artifact_filename(library, "linux-x86_64") == (
        f"lib{library.name}.so")
    assert bridge_assets.artifact_filename(executable, "win-x86_64") == (
        f"{executable.name}.exe")
    assert bridge_assets.artifact_filename(executable, "linux-x86_64") == (
        executable.name)


def test_every_bundled_artifact_is_one_the_resolver_searches_for():
    """A bundle must not carry a file nothing looks for, or miss one.

    The five decoders come from doctor's own consumer table, and the
    other three are the artifacts with their own resolution modules.
    """

    from gpuwm import doctor, rustwx, rustwx_fetch
    from gpuwm.ingest.cpu_backend import CPU_BRIDGE_ENV

    names = [artifact.name for artifact in bridge_assets.BUNDLED_ARTIFACTS]
    assert set(bridges.BRIDGE_ENV) <= set(names)
    assert set(doctor._BRIDGE_CONSUMERS) <= set(names)
    assert rustwx.RENDERER_NAME in names
    assert rustwx_fetch.FETCH_NAME in names
    envs = {artifact.name: artifact.env_var
            for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert envs[rustwx.RENDERER_NAME] == rustwx.RENDERER_ENV
    assert envs[rustwx_fetch.FETCH_NAME] == rustwx_fetch.FETCH_ENV
    assert envs["gpuwm_preprocess_cpu"] == CPU_BRIDGE_ENV
    for name, env in bridges.BRIDGE_ENV.items():
        assert envs[name] == env


# ---------------------------------------------------------------------------
# The command: refusals a user actually meets
# ---------------------------------------------------------------------------

def test_an_unsupported_platform_is_told_so_and_sent_to_the_build(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(bridge_assets.sys, "platform", "darwin")
    monkeypatch.setattr(bridge_assets.platform_module, "machine",
                        lambda: "arm64")
    assert bridge_assets.fetch_bridges_main(_args(dest=str(tmp_path))) == 2
    out = capsys.readouterr().out
    assert "darwin/arm64" in out
    assert "build the artifacts from a clone" in out
    assert not bridge_assets.staging_available()


def test_a_tree_with_no_pins_refuses_instead_of_inventing_one(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(bridge_assets, "load_pins",
                        lambda path=None: bridge_assets.BridgePins(
                            release=None, platforms={}))
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: "linux-x86_64")
    assert bridge_assets.fetch_bridges_main(_args(dest=str(tmp_path))) == 2
    assert "carries no bundle pins for linux-x86_64" in capsys.readouterr().out
    assert not bridge_assets.staging_available()


def test_a_corrupt_bundle_makes_the_command_exit_two(tmp_path, capsys,
                                                     monkeypatch):
    archive, bundle = _synthetic_bundle(tmp_path, corrupt="beta.bin")
    _install_pins(monkeypatch, bundle)

    dest = tmp_path / "dest"
    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(tmp_path))) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert not (dest / "beta.bin").exists()


def test_the_listing_reports_state_without_touching_the_network(
        tmp_path, capsys, monkeypatch):
    archive, bundle = _synthetic_bundle(tmp_path)
    _install_pins(monkeypatch, bundle)
    dest = tmp_path / "dest"
    dest.mkdir()
    with zipfile.ZipFile(archive) as zf:
        (dest / "alpha.bin").write_bytes(zf.read("alpha.bin"))

    def _no_network(*_args, **_kwargs):
        raise AssertionError("--list must not open a connection")

    # `urlopen`, not `_default_urlopen`: the latter is bound as a default
    # argument at definition time, so patching it is a guard that cannot
    # fire.  This one is looked up when the call happens.
    monkeypatch.setattr(bridge_assets, "urlopen", _no_network)
    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), list=True)) == 0
    out = capsys.readouterr().out
    assert "staged   alpha.bin" in out
    assert "needed   beta.bin" in out
    assert "1 of 2 artifacts already staged" in out


def test_a_set_environment_override_is_reported_not_silently_shadowed(
        tmp_path, capsys, monkeypatch):
    _archive, raw = _synthetic_bundle(
        tmp_path, names=("grib1_bridge.exe", "rw_fetch.exe"))
    bundle = bridge_assets.BundlePin(
        platform=raw.platform, filename=raw.filename, bytes=raw.bytes,
        sha256=raw.sha256,
        binaries=(
            bridge_assets.BinaryPin("grib1_bridge", "grib1_bridge.exe",
                                    raw.binaries[0].bytes,
                                    raw.binaries[0].sha256),
            bridge_assets.BinaryPin("rw_fetch", "rw_fetch.exe",
                                    raw.binaries[1].bytes,
                                    raw.binaries[1].sha256)))
    _install_pins(monkeypatch, bundle)
    monkeypatch.setenv(bridges.BRIDGE_ENV["grib1_bridge"],
                       str(tmp_path / "elsewhere.exe"))
    # This synthetic payload carries no contract marker; that check has
    # its own test below, against a real build.
    monkeypatch.setattr(bridge_assets, "verify_contract_marker",
                        lambda artifact, path: None)

    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(tmp_path / "dest"), from_dir=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert bridges.BRIDGE_ENV["grib1_bridge"] in out
    assert "wins over the staged copy" in out


def test_the_cli_dispatches_fetch_bridges_for_real(monkeypatch, capsys,
                                                   tmp_path):
    """Through gpuwm.cli.main, not just --help.

    fetch-tables once passed a --help-only probe while the real dispatch
    raised AttributeError; the same dispatch table gained an entry here.
    """

    from gpuwm.cli import main

    monkeypatch.setattr(bridge_assets.sys, "platform", "darwin")
    monkeypatch.setattr(bridge_assets.platform_module, "machine",
                        lambda: "arm64")
    assert main(["fetch-bridges", "--dest", str(tmp_path)]) == 2
    assert "no bundle is published for" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The release tool
# ---------------------------------------------------------------------------

def test_the_bundle_tool_refuses_to_pin_a_partial_bundle(tmp_path):
    """A bundle missing an artifact must not become a wheel's pins."""

    archive = tmp_path / "gpuwm-bridges-v0.0.0-test-linux-x86_64.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("grib1_bridge", b"only one of eight")
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", "v0.0.0-test",
         "--bundle", str(archive), "--out", str(tmp_path / "pins.json")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "refusing to pin a partial bundle" in (
        result.stderr + result.stdout)
    assert not (tmp_path / "pins.json").exists()


def test_the_bundle_tool_refuses_a_bundle_named_for_another_release(tmp_path):
    archive = tmp_path / "gpuwm-bridges-v9.9.9-linux-x86_64.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("grib1_bridge", b"payload")
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", "v0.0.0-test",
         "--bundle", str(archive), "--out", str(tmp_path / "pins.json")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "is not a bundle name for release" in (
        result.stderr + result.stdout)


@pytest.mark.parametrize("platform", bridge_assets.SUPPORTED_PLATFORMS)
def test_the_bundle_tool_writes_a_deterministic_archive(tmp_path, platform):
    """Two packs of the same bytes produce the same archive, per platform.

    A release asset whose hash depends on when it was zipped cannot be
    re-derived by anyone auditing the pins later.
    """

    source = tmp_path / "src"
    source.mkdir()
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        name = bridge_assets.artifact_filename(artifact, platform)
        (source / name).write_bytes(name.encode() * 64)
    digests = []
    for run in ("a", "b"):
        out = tmp_path / run
        result = subprocess.run(
            [sys.executable, str(BUNDLE_TOOL), "pack", "--release",
             "v0.0.0-test", "--platform", platform, "--search", str(source),
             "--out", str(out)],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        archive = out / f"gpuwm-bridges-v0.0.0-test-{platform}.zip"
        digests.append(bridge_assets.sha256_file(archive))
    assert digests[0] == digests[1]


def test_the_bundle_tool_names_the_artifact_it_cannot_find(tmp_path):
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pack", "--release", "v0.0.0-test",
         "--platform", "linux-x86_64", "--search", str(tmp_path),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "grib1_bridge" in (result.stderr + result.stdout)


# ---------------------------------------------------------------------------
# Integration: a real bundle, real pins, the real release tool
# ---------------------------------------------------------------------------

def _built_artifact_dirs() -> list[Path] | None:
    """Directories holding this machine's own built Rust artifacts.

    A checkout that has run ``cargo build --release`` in both vendored
    workspaces, or the user-level directory a previous ``fetch-bridges``
    filled.  None means the machine simply has nothing to pack, which is
    a skip rather than a failure.
    """

    host = bridge_assets.host_platform()
    if host is None:
        return None
    wanted = [bridge_assets.artifact_filename(artifact, host)
              for artifact in bridge_assets.BUNDLED_ARTIFACTS]
    groups = [
        [bridges.crate_dir() / "target" / "release",
         REPO_ROOT / "tools" / "rustwx" / "target" / "release"],
        [bridges.default_bridge_dir()],
    ]
    for group in groups:
        if all(any((directory / name).is_file() for directory in group)
               for name in wanted):
            return group
    return None


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory):
    """Pack and pin a bundle out of this machine's own built artifacts.

    The release tool, run exactly as the release workflow runs it, over
    bytes a Rust compiler on this box produced.  Every pin is computed
    here; nothing is transcribed.
    """

    directories = _built_artifact_dirs()
    if directories is None:
        pytest.skip("no complete set of built Rust artifacts on this machine")
    host = bridge_assets.host_platform()
    work = tmp_path_factory.mktemp("bridge-bundle")
    release = "v0.0.0-test"
    pack = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pack", "--release", release,
         "--platform", host, "--out", str(work)]
        + [arg for directory in directories
           for arg in ("--search", str(directory))],
        capture_output=True, text=True)
    assert pack.returncode == 0, pack.stderr
    archive = work / f"gpuwm-bridges-{release}-{host}.zip"
    assert archive.is_file()
    pins_path = work / "bridge-pins.json"
    manifest_path = work / "bridge-bundle-manifest.json"
    pin = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", release,
         "--bundle", str(archive), "--out", str(pins_path),
         "--manifest", str(manifest_path)],
        capture_output=True, text=True)
    assert pin.returncode == 0, pin.stderr
    return {"work": work, "archive": archive, "pins": pins_path,
            "manifest": manifest_path, "platform": host, "release": release}


def test_the_release_tool_produces_pins_the_runtime_accepts(real_bundle):
    pins = bridge_assets.load_pins(real_bundle["pins"])
    assert pins.release == real_bundle["release"]
    bundle = pins.bundle_for(real_bundle["platform"])
    assert bundle is not None
    assert len(bundle.binaries) == len(bridge_assets.BUNDLED_ARTIFACTS)
    # The pins describe the archive that exists, byte for byte.
    bridge_assets.verify_pinned_file(
        real_bundle["archive"], expected_bytes=bundle.bytes,
        expected_sha256=bundle.sha256, label=bundle.filename)
    manifest = json.loads(real_bundle["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema"] == bridge_assets.BUNDLE_MANIFEST_SCHEMA
    assert manifest["release"] == real_bundle["release"]
    assert real_bundle["platform"] in manifest["platforms"]


def test_a_real_bundle_stages_and_the_binaries_execute(real_bundle, tmp_path,
                                                       capsys, monkeypatch):
    """The artifact, not the mock: stage it, then run what was staged.

    ``--from`` over a bundle the release tool packed, staged into a
    scratch directory, and then probed with doctor's own execution check
    -- the only evidence that what landed on disk is a program rather
    than the right number of bytes.
    """

    from gpuwm import doctor

    pins = bridge_assets.load_pins(real_bundle["pins"])
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    dest = tmp_path / "dest"
    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(real_bundle["work"]))) == 0
    out = capsys.readouterr().out
    assert "matches the packaged bundle pin" in out

    bundle = pins.bundle_for(real_bundle["platform"])
    for pin in bundle.binaries:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)
    probe_name = bridge_assets.artifact_filename(
        bridge_assets.BUNDLED_ARTIFACTS[0], real_bundle["platform"])
    ok, evidence = doctor._exec_probe(dest / probe_name)
    assert ok, evidence

    # Idempotent: a second run touches nothing and says so.
    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(real_bundle["work"]))) == 0
    assert "nothing to fetch" in capsys.readouterr().out


def test_a_real_bundle_with_a_flipped_byte_is_refused(real_bundle, tmp_path):
    """Corruption at file level, in a bundle that is otherwise real."""

    pins = bridge_assets.load_pins(real_bundle["pins"])
    bundle = pins.bundle_for(real_bundle["platform"])
    with zipfile.ZipFile(real_bundle["archive"]) as source:
        members = {name: source.read(name) for name in source.namelist()}
    victim = bundle.binaries[0].filename
    members[victim] = members[victim][:-1] + bytes(
        [members[victim][-1] ^ 0xFF])
    tampered = tmp_path / bundle.filename
    with zipfile.ZipFile(tampered, "w") as sink:
        for name, payload in members.items():
            sink.writestr(name, payload)

    with pytest.raises(bridge_assets.BridgeAssetError, match="SHA-256"):
        bridge_assets.stage_from_bundle(tampered, bundle, tmp_path / "dest",
                                        progress=lambda _l: None)
    assert not (tmp_path / "dest" / victim).exists()


def test_an_artifact_that_predates_the_contract_is_refused_at_staging(
        real_bundle, tmp_path):
    """The static contract marker, checked before install rather than after.

    A decoder assembled from a tree that predates a contract change
    still executes and still prints its usage line; the marker is what
    catches it.  Staging runs that check on the temporary file, so those
    bytes never become the installed copy.
    """

    pins = bridge_assets.load_pins(real_bundle["pins"])
    bundle = pins.bundle_for(real_bundle["platform"])
    marked = [pin for pin in bundle.binaries
              if pin.artifact in bridges.BRIDGE_ABI_MARKERS]
    assert marked, "no bundled artifact declares a contract marker"
    pin = marked[0]
    with zipfile.ZipFile(real_bundle["archive"]) as source:
        payload = source.read(pin.filename)
    marker = bridges.BRIDGE_ABI_MARKERS[pin.artifact]
    stripped = payload.replace(marker, b"\x00" * len(marker))
    assert stripped != payload, "the marker was not in the built binary"
    # Pin the bytes actually being staged, so the only check that can
    # fail is the contract one.
    archive = tmp_path / bundle.filename
    with zipfile.ZipFile(archive, "w") as sink:
        sink.writestr(pin.filename, stripped)
    one = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=archive.stat().st_size,
        sha256=bridge_assets.sha256_file(archive),
        binaries=(bridge_assets.BinaryPin(
            pin.artifact, pin.filename, len(stripped),
            hashlib.sha256(stripped).hexdigest()),))

    with pytest.raises(bridge_assets.BridgeAssetError, match="predates"):
        bridge_assets.stage_from_bundle(archive, one, tmp_path / "dest",
                                        progress=lambda _l: None)
    assert not (tmp_path / "dest" / pin.filename).exists()


# ---------------------------------------------------------------------------
# One opt-in live smoke
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_the_published_bundle_downloads_and_verifies(tmp_path):
    """The only test that touches GitHub, and only when asked to.

    Gated twice -- the ``network`` marker and GPUWM_NETWORK_TESTS=1 --
    because it proves the one thing no offline test can: that the URL
    the packaged pins resolve to serves the bytes those pins describe.
    """

    if os.environ.get("GPUWM_NETWORK_TESTS") != "1":
        pytest.skip("live download smoke needs GPUWM_NETWORK_TESTS=1")
    pins = bridge_assets.load_pins()
    bundle = pins.bundle_for(bridge_assets.host_platform())
    if bundle is None:
        pytest.skip("this build publishes no bundle for this platform")
    dest = tmp_path / "dest"
    installed = bridge_assets.fetch_bundle(pins, bundle, dest, progress=print)
    assert len(installed) == len(bundle.binaries)
    for pin in bundle.binaries:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)
