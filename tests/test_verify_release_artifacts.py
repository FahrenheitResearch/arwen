"""Drive the release-artifact verifier outside a live cut.

``tools/verify_release_artifacts.py`` is the last thing between a built
wheel and PyPI, and until now the only way to run it was to be in the
middle of a release.  That is how a defect in it -- a bundle membership
test that compared the archive against the binary pins alone, and so
refused every bundle the current packer produces -- reached a live 1.4.1
round instead of a CI run.  ``--dry-run`` runs the same assertions, and
this module drives them.

The fixtures are deliberately not this module's own idea of a bundle:
the bundles, pins, and manifest come from ``tools/build_bridge_bundle.py``
-- the real writer the cut runs -- over fixture binaries carrying real
``GPUWM_BRIDGE_SOURCE_REV`` stamps, and, for a vendored artifact, the
real declared contract marker the packer asks it for instead.  A verifier
tested against a fixture its own author shaped proves only that two wrong
ideas agree.
"""

from __future__ import annotations

from base64 import urlsafe_b64encode
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from gpuwm import bridge_assets, bridges
from tools import build_bridge_bundle, verify_release_artifacts

RELEASE = "v9.8.7"
VERSION = "9.8.7"
SOURCE_REV = "b" * 40
PINS_MEMBER = "gpuwm/data/bridges/bridge-pins.json"


def _stamped(name: str, source_rev: str) -> bytes:
    """Fixture bytes carrying exactly one well-formed revision stamp."""

    return b"".join(
        (
            b"fixture native artifact ",
            name.encode("utf-8"),
            b"\n",
            bridge_assets.SOURCE_REV_MARKER,
            source_rev.encode("ascii"),
            b"\n",
        )
    )


def _artifact_bytes(
    artifact: bridge_assets.BundledArtifact, name: str, source_rev: str
) -> bytes:
    """Fixture bytes carrying whatever the packer asks this artifact for.

    A vendored artifact is a verbatim upstream crate frozen at a recorded
    commit, so it does not move with the release checkout and the packer
    never asks it for a revision stamp -- it asks for the contract marker
    declared in ``BRIDGE_ABI_MARKERS``, a property of those exact bytes.
    Stamping such a fixture with a revision would prove nothing the packer
    reads, so it carries the real marker instead.
    """

    if artifact.vendored:
        return b"".join(
            (
                b"fixture vendored artifact ",
                name.encode("utf-8"),
                b"\n",
                bridges.BRIDGE_ABI_MARKERS[artifact.name],
                b"\n",
            )
        )
    return _stamped(name, source_rev)


def _pack_and_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_rev: str = SOURCE_REV,
) -> tuple[Path, Path, Path]:
    """A cut's bundle payload, produced by the packer the workflow runs."""

    asset_source = tmp_path / "asset-source"
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        directory = asset_source / subdir
        directory.mkdir(parents=True)
        # Two files, so a test can drop one pin and still leave a bundle
        # that carries map assets at all -- the refusal for carrying none
        # is a different check, and it would mask this one.
        for index in (1, 2):
            (directory / f"fixture-geometry-{index}.bin").write_bytes(
                f"fixture {subdir} geometry {index}\n".encode("utf-8")
            )
    # The one seam the packer names for the asset tree.  Pointing it at a
    # fixture keeps the real walk, the real member naming, and the real
    # refusals, without carrying 36 MiB of basemaps through a unit test.
    monkeypatch.setattr(
        build_bridge_bundle, "source_asset_dir", lambda: asset_source
    )

    bundles = tmp_path / "dist-bridges"
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        search = tmp_path / f"target-{platform}"
        search.mkdir()
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            filename = bridge_assets.artifact_filename(artifact, platform)
            (search / filename).write_bytes(
                _artifact_bytes(artifact, filename, source_rev)
            )
        assert (
            build_bridge_bundle.main(
                [
                    "pack",
                    "--release",
                    RELEASE,
                    "--platform",
                    platform,
                    "--search",
                    str(search),
                    "--out",
                    str(bundles),
                ]
            )
            == 0
        )

    pins = tmp_path / "bridge-pins.json"
    manifest = bundles / build_bridge_bundle.MANIFEST_ASSET
    bundle_args: list[str] = []
    for archive in sorted(bundles.glob("*.zip")):
        bundle_args += ["--bundle", str(archive)]
    assert (
        build_bridge_bundle.main(
            [
                "pin",
                "--release",
                RELEASE,
                "--source-rev",
                source_rev,
                *bundle_args,
                "--out",
                str(pins),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    return bundles, pins, manifest


def _build_dists(tmp_path: Path, pins: Path) -> tuple[Path, Path]:
    """A wheel and sdist carrying the generated pins the way a cut's do."""

    payload = pins.read_bytes()
    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)

    wheel = dist / f"gpuwm-{VERSION}-py3-none-any.whl"
    digest = (
        urlsafe_b64encode(hashlib.sha256(payload).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    record = (
        f"{PINS_MEMBER},sha256={digest},{len(payload)}\n"
        f"gpuwm-{VERSION}.dist-info/RECORD,,\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(PINS_MEMBER, payload)
        archive.writestr(f"gpuwm-{VERSION}.dist-info/RECORD", record)

    sdist = dist / f"gpuwm-{VERSION}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo(f"gpuwm-{VERSION}/{PINS_MEMBER}")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return wheel, sdist


def _dry_run(
    tmp_path: Path,
    bundles: Path,
    pins: Path,
    manifest: Path,
    wheel: Path,
    sdist: Path,
    *,
    release: str = RELEASE,
    source_rev: str = SOURCE_REV,
) -> Path:
    receipt = tmp_path / "release-artifact-proof.json"
    assert (
        verify_release_artifacts.main(
            [
                "--dry-run",
                "--wheel",
                str(wheel),
                "--sdist",
                str(sdist),
                "--pins",
                str(pins),
                "--manifest",
                str(manifest),
                "--bundles",
                str(bundles),
                "--release",
                release,
                "--source-rev",
                source_rev,
                "--receipt",
                str(receipt),
            ]
        )
        == 0
    )
    return receipt


def test_dry_run_passes_the_packers_own_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    receipt_path = _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "gpuwm-release-artifact-proof-v2"
    assert receipt["status"] == "PASS"
    assert receipt["release"] == RELEASE
    assert receipt["source_rev"] == SOURCE_REV
    assert set(receipt["bundles"]) == set(bridge_assets.SUPPORTED_PLATFORMS)
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        assert receipt["bundles"][platform]["binaries"] == len(
            bridge_assets.BUNDLED_ARTIFACTS)
        assert receipt["bundles"][platform]["assets"] >= 1


def test_the_receipt_names_which_proof_each_binary_got(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2's invariant, stated as a partition rather than a boolean.

    v1 claimed one blanket thing about every binary, and a vendored
    artifact -- correct, pinned, deliberately unstamped -- made that claim
    false.  v2 says instead that every binary was proved by exactly one of
    two means, and names them, so bytes nobody proved would show up as a
    filename missing from both lists rather than as a still-true-looking
    ``True``.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    receipt = json.loads(
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist).read_text(
            encoding="utf-8"
        )
    )

    assert receipt["schema"] == "gpuwm-release-artifact-proof-v2"
    # The retired v1 key is gone, not aliased: a reader written against
    # the old blanket claim must fail loudly rather than silently pass.
    assert "source_rev_stamp_verified_in_every_binary" not in receipt

    stamped = receipt["binaries_proved_by_source_rev_stamp"]
    markered = receipt["binaries_proved_by_contract_marker"]
    assert not set(stamped) & set(markered)

    expected: set[str] = set()
    vendored_names: set[str] = set()
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            label = (
                f"{platform}: "
                f"{bridge_assets.artifact_filename(artifact, platform)}"
            )
            expected.add(label)
            if artifact.vendored:
                vendored_names.add(label)
    # Every binary in both bundles is accounted for by one proof or the
    # other, and the vendored one is accounted for by the marker.
    assert set(stamped) | set(markered) == expected
    assert set(markered) == vendored_names
    assert vendored_names, "fixture no longer covers a vendored artifact"


def test_dry_run_refuses_a_vendored_artifact_missing_its_contract_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stale vendored build the marker check exists to catch.

    A vendored artifact carries no revision stamp, so exempting it from
    the stamp check must not mean exempting it from staleness: the marker
    is the only question those bytes can answer, and it has to bite.

    The bundle here is packed and pinned honestly, and then the release's
    declared contract moves on -- which is exactly what re-vendoring is, a
    commit that advances ``BRIDGE_ABI_MARKERS`` together with the crate.
    Bundles built before that advance carry the older bytes.  The packer
    refuses such a bundle at pin time, but this verifier is the
    independent second reading of already-published artifacts, so it has
    to refuse on its own rather than inherit the packer's verdict.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    vendored = [
        artifact
        for artifact in bridge_assets.BUNDLED_ARTIFACTS
        if artifact.vendored
    ]
    assert vendored, "fixture no longer covers a vendored artifact"
    monkeypatch.setitem(
        bridges.BRIDGE_ABI_MARKERS,
        vendored[0].name,
        b"bw_dealias_rift_v2_not_in_these_bytes",
    )

    with pytest.raises(SystemExit) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    assert "contract marker" in str(refusal.value)


def test_dry_run_refuses_a_vendored_artifact_with_no_declared_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unstamped binary with nothing else to prove it is not proved.

    Skipping the stamp check for a vendored artifact is only sound while
    something else answers the staleness question.  If the marker table
    ever loses that artifact's entry, the exemption must become a refusal
    rather than a silent pass.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    vendored = [
        artifact
        for artifact in bridge_assets.BUNDLED_ARTIFACTS
        if artifact.vendored
    ]
    assert vendored, "fixture no longer covers a vendored artifact"
    monkeypatch.delitem(bridges.BRIDGE_ABI_MARKERS, vendored[0].name)

    with pytest.raises(SystemExit) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    assert "no declared" in str(refusal.value)


def test_a_dry_run_receipt_never_claims_what_it_did_not_prove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode is on the receipt, and so is everything it skipped."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    receipt = json.loads(
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist).read_text(
            encoding="utf-8"
        )
    )

    assert receipt["mode"] == "dry-run"
    assert receipt["not_proven"] == list(verify_release_artifacts.DRY_RUN_SKIPS)
    assert receipt["installed_version"] is None
    assert receipt["host_probes"] == []
    assert receipt["prebuilt_offer_live_lines"] == []


def test_a_live_cut_still_requires_the_installation_it_proves() -> None:
    """Without ``--dry-run`` the cut-only inputs stay mandatory, so the
    mode can never be entered by forgetting an argument."""

    with pytest.raises(SystemExit) as refusal:
        verify_release_artifacts._parse_args(
            [
                "--wheel", "w", "--sdist", "s", "--pins", "p",
                "--manifest", "m", "--bundles", "b", "--release", RELEASE,
                "--source-rev", SOURCE_REV, "--receipt", "r",
            ]
        )
    assert refusal.value.code == 2


def _rewrite_platforms(pins: Path, manifest: Path, mutate) -> None:
    """Apply the same edit to both documents the cut publishes.

    They are cross-checked for equality before membership is looked at,
    so a one-sided edit would only ever prove that check.
    """

    for path, key in ((pins, "platforms"), (manifest, "platforms")):
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document[key])
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


@pytest.mark.parametrize("direction", ("under-declared", "over-declared"))
def test_dry_run_refuses_pins_that_do_not_name_the_bundle_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    """The 1.4.1 defect class, now reachable without a live cut.

    A bundle must contain exactly what the release pinned, so the check
    is an equality and both directions have to bite: a pinned member the
    archive does not carry is a download that 404s member-by-member, and
    an unpinned member the archive does carry is bytes nobody verified.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)

    def mutate(platforms: dict) -> None:
        record = platforms[bridge_assets.SUPPORTED_PLATFORMS[0]]
        if direction == "under-declared":
            record["assets"].pop()
        else:
            record["assets"].append(
                {
                    "path": "assets/basemap/never-packed.bin",
                    "bytes": 3,
                    "sha256": "0" * 64,
                }
            )

    _rewrite_platforms(pins, manifest, mutate)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(AssertionError) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    expected = (
        "unexpected: ['assets/basemap/fixture-geometry-2.bin']"
        if direction == "under-declared"
        else "missing: ['assets/basemap/never-packed.bin']"
    )
    assert expected in str(refusal.value)


def test_dry_run_refuses_binaries_stamped_with_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hash proves the bytes; only the stamp proves their source."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(bridge_assets.BridgeAssetError, match="was built from"):
        _dry_run(
            tmp_path, bundles, pins, manifest, wheel, sdist,
            source_rev="c" * 40,
        )


def test_dry_run_refuses_a_wheel_whose_record_misstates_the_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RECORD binds the pins' hash and size, not merely their name."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    payload = pins.read_bytes()
    rebuilt = wheel.with_name("rebuilt.whl")
    with zipfile.ZipFile(rebuilt, "w") as archive:
        archive.writestr(PINS_MEMBER, payload)
        archive.writestr(
            f"gpuwm-{VERSION}.dist-info/RECORD",
            f"{PINS_MEMBER},sha256={'A' * 43},{len(payload)}\n",
        )

    with pytest.raises(AssertionError):
        _dry_run(tmp_path, bundles, pins, manifest, rebuilt, sdist)


def test_dry_run_refuses_pins_generated_for_another_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(AssertionError):
        _dry_run(
            tmp_path, bundles, pins, manifest, wheel, sdist, release="v0.0.1"
        )


# ---------------------------------------------------------------------------
# The host-staging leg's library probe
# ---------------------------------------------------------------------------
#
# The leg that loads each `kind == "library"` artifact is the one thing
# `--dry-run` cannot run, and it is where 2.1.0 lost a number: it asked
# EVERY library for `gpuwm_preprocess_cpu_abi_version` while the vendored
# dealiasing cdylib exports `bw_abi_version`, so a correct bundle was
# refused in the prepare job with the tag already public.  These cases
# drive `probe_library_abi` through its loader seam, so per-artifact
# dispatch and the refusal for an undeclared library are proved without a
# target-native binary -- on both platforms' filenames, in CI, on every
# commit.


class _FakeAbi:
    """One exported symbol, callable the way ctypes' function objects are."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.argtypes: list[object] | None = None
        self.restype: object | None = None

    def __call__(self) -> int:
        return self.value


class _FakeLibrary:
    """A loaded library that exports exactly the symbols it was given."""

    def __init__(self, path: str, exports: dict[str, int]) -> None:
        self.path = path
        self._exports = {name: _FakeAbi(value)
                         for name, value in exports.items()}

    def __getattr__(self, name: str) -> _FakeAbi:
        try:
            return self.__dict__["_exports"][name]
        except KeyError:
            raise AttributeError(
                f"{self.path}: undefined symbol: {name}") from None


def _loader_for(exports_by_filename: dict[str, dict[str, int]]):
    """A ctypes.CDLL stand-in serving each fixture library its own exports."""

    def load(path: str) -> _FakeLibrary:
        return _FakeLibrary(path, exports_by_filename[Path(path).name])

    return load


def _library_artifacts() -> tuple[bridge_assets.BundledArtifact, ...]:
    return tuple(artifact for artifact in bridge_assets.BUNDLED_ARTIFACTS
                 if artifact.kind == "library")


def test_the_release_ships_two_libraries_with_different_abi_symbols() -> None:
    """The premise of the dispatch: two libraries, two handshakes.

    If this ever collapses to one library the dispatch cases below stop
    proving anything, and this one says so out loud rather than letting
    them pass vacuously.
    """

    artifacts = _library_artifacts()
    assert len(artifacts) == 2, [a.name for a in artifacts]
    symbols = {bridge_assets.library_abi_for(a.name)[0] for a in artifacts}
    assert symbols == {"gpuwm_preprocess_cpu_abi_version", "bw_abi_version"}


@pytest.mark.parametrize("platform", bridge_assets.SUPPORTED_PLATFORMS)
def test_each_library_is_probed_with_its_own_declared_symbol(
    tmp_path: Path, platform: str
) -> None:
    """Two library fixtures, each exporting ONLY its own ABI symbol.

    Neither fixture can answer the other's handshake, so a probe that
    reused one symbol for both -- the 2.1.0 defect -- fails here.
    """

    artifacts = _library_artifacts()
    exports: dict[str, dict[str, int]] = {}
    paths: dict[str, Path] = {}
    for artifact in artifacts:
        filename = bridge_assets.artifact_filename(artifact, platform)
        path = tmp_path / platform / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture library " + artifact.name.encode())
        symbol, version = bridge_assets.library_abi_for(artifact.name)
        exports[filename] = {symbol: version}
        paths[artifact.name] = path

    loader = _loader_for(exports)
    for artifact in artifacts:
        probe = verify_release_artifacts.probe_library_abi(
            paths[artifact.name], artifact, loader=loader
        )
        symbol, version = bridge_assets.library_abi_for(artifact.name)
        assert probe["artifact"] == artifact.name
        assert probe["symbol"] == symbol
        assert probe["abi"] == version
        assert probe["status"] == "PASS"
        assert probe["filename"] == bridge_assets.artifact_filename(
            artifact, platform
        )


def test_the_vendored_library_is_not_probed_with_the_stamped_ones_symbol(
    tmp_path: Path,
) -> None:
    """The exact 2.1.0 refusal, as a test.

    The vendored cdylib exports `bw_abi_version` and nothing else.  A
    probe that reaches for `gpuwm_preprocess_cpu_abi_version` gets an
    AttributeError from the loader; the shipped dispatch gets an answer.
    """

    vendored = next(artifact for artifact in _library_artifacts()
                    if artifact.vendored)
    stamped = next(artifact for artifact in _library_artifacts()
                   if not artifact.vendored)
    filename = bridge_assets.artifact_filename(vendored, "linux-x86_64")
    path = tmp_path / filename
    path.write_bytes(b"fixture vendored library\n")
    loader = _loader_for({filename: {"bw_abi_version": 1}})

    probe = verify_release_artifacts.probe_library_abi(
        path, vendored, loader=loader
    )
    assert probe["symbol"] == "bw_abi_version"

    # The stamped library's artifact record against the vendored bytes:
    # the wrong handshake, which is what the old code applied to both.
    with pytest.raises(AttributeError, match="undefined symbol"):
        verify_release_artifacts.probe_library_abi(
            path, stamped, loader=loader
        )


def test_a_library_with_no_declared_handshake_is_refused(
    tmp_path: Path,
) -> None:
    """Fail closed, the same shape as the vendored contract-marker refusal.

    A library added to BUNDLED_ARTIFACTS without a LIBRARY_ABI entry is
    refused by name; it is never probed with some other library's symbol.
    """

    undeclared = bridge_assets.BundledArtifact(
        "future_thing", "library", "tools/future_thing",
        "GPUWM_FUTURE_THING_BRIDGE", "a library nobody declared")
    assert undeclared.name not in bridge_assets.LIBRARY_ABI
    path = tmp_path / "libfuture_thing.so"
    path.write_bytes(b"fixture undeclared library\n")
    loader = _loader_for(
        {"libfuture_thing.so": {"gpuwm_preprocess_cpu_abi_version": 1}}
    )

    with pytest.raises(SystemExit) as refusal:
        verify_release_artifacts.probe_library_abi(
            path, undeclared, loader=loader
        )
    message = str(refusal.value)
    assert "future_thing" in message
    assert "LIBRARY_ABI" in message


def test_a_library_answering_another_version_is_refused(
    tmp_path: Path,
) -> None:
    """The right symbol answering the wrong number is still a stale build."""

    stamped = next(artifact for artifact in _library_artifacts()
                   if not artifact.vendored)
    filename = bridge_assets.artifact_filename(stamped, "linux-x86_64")
    path = tmp_path / filename
    path.write_bytes(b"fixture library\n")
    symbol, version = bridge_assets.library_abi_for(stamped.name)
    loader = _loader_for({filename: {symbol: version + 7}})

    with pytest.raises(SystemExit, match=symbol):
        verify_release_artifacts.probe_library_abi(
            path, stamped, loader=loader
        )


def test_the_workflow_probe_reads_the_shared_table_not_its_own_copy() -> None:
    """One table, cross-checked: the workflow must not re-declare it.

    `.github/workflows/publish.yml` runs the same per-artifact dispatch on
    each target runner.  It carried its own literal copy of the table
    while the verifier carried the older blanket rule, which is precisely
    how the two came to disagree.  The copy is gone; this case fails if
    one is reintroduced, because a second copy is a second thing to
    forget.
    """

    workflow = (
        Path(verify_release_artifacts.__file__).resolve().parent.parent
        / ".github" / "workflows" / "publish.yml"
    )
    text = workflow.read_text(encoding="utf-8")
    assert "bridge_assets.library_abi_for(" in text
    for symbol, _version in bridge_assets.LIBRARY_ABI.values():
        assert f'"{symbol}"' not in text, symbol
        assert f"'{symbol}'" not in text, symbol
