"""All native tools ship in platform wheels, including the three restored tools."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from _release_export import private_input
from gpuwm import bridge_assets, bridges

stage_wheel_bridges = pytest.importorskip("tools.stage_wheel_bridges")

REPO = Path(__file__).resolve().parents[1]
ON_DEMAND = {"rw_simradar", "rw_mlexport", "rw_globe_pack"}
PLATFORMS = ("linux-x86_64", "win-x86_64")
REVISION = "a" * 40
FETCH = "gpuwm fetch-bridges"


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

def test_three_formerly_omitted_artifacts_are_restored():
    assert ON_DEMAND <= {a.name for a in bridge_assets.wheel_artifacts()}
    assert not hasattr(bridge_assets.BundledArtifact, "on_demand")


def test_the_wheel_carries_every_other_artifact_in_roster_order():
    carried = bridge_assets.wheel_artifacts()
    assert [a.name for a in carried] == [
        a.name for a in bridge_assets.BUNDLED_ARTIFACTS]
    assert len(carried) == len(bridge_assets.BUNDLED_ARTIFACTS)
    assert ON_DEMAND <= {a.name for a in carried}


def test_restored_tools_and_default_paths_are_carried():
    """A decoder, a writer or the renderer left out would make a bare install
    refuse its own forecast; every library is behind a default path."""
    for artifact in restored_artifacts():
        assert artifact.kind == "executable", artifact.name
        assert not artifact.vendored, artifact.name
        assert artifact.name not in bridges.BRIDGE_ENV, artifact.name
    for default in ("rw_wrfbatch", "rw_netcdf", "netcdf_writer", "gpuwm_mapped_engine",
                    "rw_fetch", "static_fields", "rw_nexrad", "arwen-tui"):
        assert default in {a.name for a in bridge_assets.wheel_artifacts()}, default


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _fake_build_tree(root: Path, platform: str) -> None:
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        origin = stage_wheel_bridges.source_path(artifact, platform)
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_bytes(b"artifact bytes for " + artifact.name.encode())


def _packaged(tmp_path, monkeypatch) -> Path:
    """Point the in-package directory at a scratch one, as a wheel build sees it."""
    packaged = tmp_path / "package" / "gpuwm" / "libexec" / "bridges"
    monkeypatch.setattr(stage_wheel_bridges, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(bridges, "packaged_bridge_dir", lambda: packaged)
    return packaged


def _assert_wheel_inventory(directory: Path, platform: str) -> None:
    manifest = json.loads((directory / stage_wheel_bridges.MANIFEST_NAME).read_text("utf-8"))
    carried = {a.name: bridge_assets.artifact_filename(a, platform)
               for a in bridge_assets.wheel_artifacts()}
    files = {path.name for path in directory.iterdir()}
    assert files == set(carried.values()) | {stage_wheel_bridges.MANIFEST_NAME}
    assert manifest["platform"] == platform
    assert {row["artifact"]: row["filename"] for row in manifest["artifacts"]} == carried
    assert len(manifest["artifacts"]) == len(carried)
    for row in manifest["artifacts"]:
        payload = (directory / row["filename"]).read_bytes()
        assert row["bytes"] == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("platform", PLATFORMS)
def test_staging_into_the_package_carries_the_restored_tools(tmp_path, monkeypatch, platform, capsys):
    packaged = _packaged(tmp_path, monkeypatch)
    _fake_build_tree(tmp_path, platform)
    assert stage_wheel_bridges.main(["--platform", platform]) == 0
    _assert_wheel_inventory(packaged, platform)
    said = capsys.readouterr().out
    carried, declared = len(bridge_assets.wheel_artifacts()), len(bridge_assets.BUNDLED_ARTIFACTS)
    assert carried == declared
    assert f"staged {carried} artifacts" in said


@pytest.mark.parametrize("platform", PLATFORMS)
def test_staging_the_package_from_a_bundle_carries_the_restored_tools(tmp_path, monkeypatch, platform):
    packaged = _packaged(tmp_path, monkeypatch)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            bundle.writestr(bridge_assets.artifact_filename(artifact, platform),
                            b"bundle bytes for " + artifact.name.encode())
        bundle.writestr("assets/basemap/coast.shp", b"not a binary")
    stage_wheel_bridges.stage_from_bundle(archive, platform)
    _assert_wheel_inventory(packaged, platform)


def test_a_package_wheel_requires_every_artifact_built(tmp_path, monkeypatch):
    """They are not staged, so a build tree without them still stages a wheel;
    any other missing artifact is still refused by name."""
    packaged = _packaged(tmp_path, monkeypatch)
    platform = "linux-x86_64"
    _fake_build_tree(tmp_path, platform)
    stage_wheel_bridges.stage(platform)
    _assert_wheel_inventory(packaged, platform)
    stage_wheel_bridges.source_path(bridge_assets.wheel_artifacts()[0], platform).unlink()
    with pytest.raises(SystemExit) as refusal:
        stage_wheel_bridges.stage(platform)
    assert bridge_assets.wheel_artifacts()[0].name in str(refusal.value)
    assert f"1 of {len(bridge_assets.wheel_artifacts())} declared" in str(refusal.value)
    _fake_build_tree(tmp_path, platform)
    for artifact in restored_artifacts():
        stage_wheel_bridges.source_path(artifact, platform).unlink()
    with pytest.raises(SystemExit) as refusal:
        stage_wheel_bridges.stage(platform)
    for name in ON_DEMAND:
        assert name in str(refusal.value)


def test_any_other_destination_still_gets_the_whole_roster(tmp_path, monkeypatch, capsys):
    """A user's bridge directory is what the three are fetched INTO."""
    _packaged(tmp_path, monkeypatch)
    platform = "linux-x86_64"
    _fake_build_tree(tmp_path, platform)
    home = tmp_path / "home" / ".gpuwm" / "bridges"
    assert stage_wheel_bridges.main(["--platform", platform, "--dest", str(home)]) == 0
    manifest = json.loads((home / stage_wheel_bridges.MANIFEST_NAME).read_text("utf-8"))
    assert {row["artifact"] for row in manifest["artifacts"]} == {
        a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
    for artifact in restored_artifacts():
        assert (home / bridge_assets.artifact_filename(artifact, platform)).is_file()
    assert f"staged {len(bridge_assets.BUNDLED_ARTIFACTS)} artifacts for" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The bundle and its pins are not narrowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform", PLATFORMS)
def test_the_bundle_and_its_pins_still_carry_every_artifact(tmp_path, monkeypatch, platform):
    from tools import build_bridge_bundle

    built = tmp_path / "built"
    built.mkdir()
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        payload = b"fixture " + artifact.name.encode() + b" "
        if artifact.vendored:
            payload += bridges.BRIDGE_ABI_MARKERS[artifact.name]
        else:
            payload += bridge_assets.SOURCE_REV_MARKER + REVISION.encode()
        (built / bridge_assets.artifact_filename(artifact, platform)).write_bytes(payload)
    assets = tmp_path / "assets"
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        (assets / subdir).mkdir(parents=True)
        (assets / subdir / "shape.bin").write_bytes(b"map " + subdir.encode())
    monkeypatch.setattr(build_bridge_bundle, "source_asset_dir", lambda: assets)

    archive = build_bridge_bundle.pack("v9.9.9", platform, [built], tmp_path / "out")
    every = {bridge_assets.artifact_filename(a, platform) for a in bridge_assets.BUNDLED_ARTIFACTS}
    with zipfile.ZipFile(archive) as bundle:
        assert every <= set(bundle.namelist())
    pinned_platform, record = build_bridge_bundle._pin_bundle(archive, "v9.9.9", REVISION)
    assert pinned_platform == platform
    assert [row["artifact"] for row in record["binaries"]] == [
        a.name for a in bridge_assets.BUNDLED_ARTIFACTS]
    assert ON_DEMAND <= {row["artifact"] for row in record["binaries"]}

    # A bundle without one of the three is still refused: fetch-bridges could not deliver it.
    slim = tmp_path / archive.name
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(slim, "w") as target:
        dropped = bridge_assets.artifact_filename(restored_artifacts()[0], platform)
        for info in source.infolist():
            if info.filename != dropped:
                target.writestr(info, source.read(info.filename))
    with pytest.raises(SystemExit, match="refusing to pin a partial bundle"):
        build_bridge_bundle._pin_bundle(slim, "v9.9.9", REVISION)


# ---------------------------------------------------------------------------
# Readers of the declaration that do not import gpuwm
# ---------------------------------------------------------------------------

def _function(relative: str, name: str):
    """One function of a release driver, compiled alone: the drivers read a
    cut's identities at import, which no test has."""
    path = REPO / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node, = (item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    scope = {"ast": ast, "Path": Path, "require": require}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


DECLARATION = REPO / "gpuwm" / "bridge_assets.py"
STAGE_PUBLIC_DESKTOP = "tools/release/cut/drivers/linux-drivers/stage_public_desktop.py"
BUILD_ACCEPTANCE = "tools/release/cut/drivers/build_acceptance.py"


def _driver(relative: str, *values):
    """One parameter row naming a file to read. The release drivers are
    under ``tools/release/cut/**``, which the public export drops, and Stage 1
    runs on the export: seven cases here raised FileNotFoundError there. Such
    a row is skipped in an export, by the rule that dropped its file, and
    runs in the development tree. A file the export ships is never skipped."""
    return pytest.param(relative, *values, marks=private_input(relative), id=relative)


@pytest.mark.parametrize("relative", [
    _driver(STAGE_PUBLIC_DESKTOP),
    _driver("tools/release/cut/drivers/linux-drivers/common.py"),
])
def test_the_desktop_drivers_read_the_same_declaration(relative):
    rows = _function(relative, "declared_artifacts")(DECLARATION)
    assert rows == [(a.name, a.vendored, False) for a in bridge_assets.BUNDLED_ARTIFACTS]


@private_input(STAGE_PUBLIC_DESKTOP)
def test_the_desktop_staging_counts_what_the_wheel_carries():
    relative = STAGE_PUBLIC_DESKTOP
    rows = _function(relative, "declared_artifacts")(DECLARATION)
    carried = _function(relative, "wheel_carried")(rows)
    assert [name for name, _, _ in carried] == [a.name for a in bridge_assets.wheel_artifacts()]


@private_input(BUILD_ACCEPTANCE)
def test_the_acceptance_driver_reads_the_same_on_demand_set():
    reader = _function(BUILD_ACCEPTANCE, "on_demand_artifacts")
    assert reader(REPO) == set()


@pytest.mark.parametrize("relative, needle", [
    _driver("tools/release/cut/drivers/package_sdk_final.py", "bridge_assets.wheel_artifacts()"),
    _driver("tools/release/cut/drivers/verify_installed_bytes.py", "bridge_assets.wheel_artifacts()"),
    _driver("tools/release/cut/drivers/linux-drivers/qualify_public_runtime.py",
            "bridge_assets.wheel_artifacts()"),
    _driver("tools/build_desktop_runtime.py", "bridge_assets.wheel_artifacts()"),
])
def test_installed_inventory_checks_compare_against_what_the_wheel_carries(relative, needle):
    """An installed wheel's BUNDLE.json compared against the whole roster
    refuses every correct 2.8.9 wheel as incomplete."""
    text = (REPO / relative).read_text(encoding="utf-8")
    assert needle in text
    assert "in bridge_assets.BUNDLED_ARTIFACTS}" not in text.replace(
        "{a.name for a in bridge_assets.BUNDLED_ARTIFACTS}", "")


def test_the_release_verifier_still_holds_the_bundle_to_the_whole_roster():
    text = (REPO / "tools" / "verify_release_artifacts.py").read_text(encoding="utf-8")
    assert "artifact.name: artifact for artifact in bridge_assets.BUNDLED_ARTIFACTS" in text
    assert "assert {pin.artifact for pin in bundle.binaries} == set(by_name)" in text
    assert "wheel_artifacts" not in text


# ---------------------------------------------------------------------------
# A pip install: no Rust sources, a bundle pinned for this platform
# ---------------------------------------------------------------------------

@pytest.fixture
def pip_install(tmp_path, monkeypatch):
    platform = "linux-x86_64"
    binaries = tuple(
        bridge_assets.BinaryPin(a.name, bridge_assets.artifact_filename(a, platform), 8,
                                hashlib.sha256(a.name.encode()).hexdigest())
        for a in bridge_assets.BUNDLED_ARTIFACTS)
    bundle = bridge_assets.BundlePin(platform=platform, filename="gpuwm-bridges-v0-test.zip",
                                     bytes=96 * 1024 * 1024, sha256="b" * 64, binaries=binaries)
    pins = bridge_assets.BridgePins(release="v0-test", platforms={platform: bundle})
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform", lambda: platform)
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path / "site-packages")
    for name in ("GPUWM_RW_SIMRADAR", "GPUWM_RW_MLEXPORT", "GPUWM_RW_GLOBE_PACK"):
        monkeypatch.delenv(name, raising=False)
    return pins


def _first_command(remedy: str) -> str:
    return next(line.strip() for line in remedy.splitlines()
                if line.strip() and not line.lstrip().startswith("#"))


@pytest.mark.parametrize("artifact", sorted(ON_DEMAND))
def test_missing_restored_artifact_still_has_the_fetch_remedy(pip_install, artifact):
    remedy = bridges.artifact_remedy(
        env_var="GPUWM_X", filename=artifact, subject="the thing",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE, artifact=artifact)
    assert _first_command(remedy) == FETCH
    assert "fetched on demand" not in remedy
    for line in remedy.splitlines():
        assert line.lstrip().startswith("#") or line.strip() == FETCH, line


def test_the_remedy_for_any_other_artifact_is_unchanged(pip_install):
    remedy = bridges.artifact_remedy(
        env_var="GPUWM_RW_NEXRAD", filename="rw_nexrad", subject="the radar front door",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE, artifact="rw_nexrad")
    assert _first_command(remedy) == FETCH
    assert "fetched on demand" not in remedy


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------

def _absent_checks(monkeypatch):
    from gpuwm import doctor, ml_export, rustwx

    monkeypatch.setattr(rustwx, "simulated_radar_binary", lambda: None)
    monkeypatch.setattr(ml_export.Binary, "find", lambda self: None)
    monkeypatch.setattr(bridges, "find_artifact", lambda env_var, filename: None)
    return {"rw_simradar": doctor._simulated_radar_check(),
            "rw_mlexport": doctor._ml_export_check(),
            "rw_globe_pack": doctor._globe_pack_check()}


def test_doctor_reports_missing_restored_artifacts_as_unreachable_or_degraded(pip_install, monkeypatch):
    from gpuwm import doctor

    for name, check in _absent_checks(monkeypatch).items():
        assert check.status == "missing", name
        expected = doctor.SEVERITY_DEGRADED if name == "rw_globe_pack" else doctor.SEVERITY_UNREACHABLE
        assert check.severity == expected, (name, check.severity)
        assert check.blocking is False, name
        assert check.action == FETCH, (name, check.action)
        assert _first_command(check.remedy) == FETCH, name
        assert "fetched on demand" not in check.detail, (name, check.detail)
        assert not doctor.blocking_gaps([check]), name


def test_doctor_keeps_the_old_severity_for_an_artifact_that_is_not_on_demand(pip_install, monkeypatch):
    from gpuwm import doctor

    checks = _absent_checks(monkeypatch)
    assert checks["rw_simradar"].severity == doctor.SEVERITY_UNREACHABLE
    assert checks["rw_mlexport"].severity == doctor.SEVERITY_UNREACHABLE
    assert checks["rw_globe_pack"].severity == doctor.SEVERITY_DEGRADED
    for check in checks.values():
        assert "fetched on demand" not in check.detail


def _no_bundle_for_this_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge_assets, "load_pins",
                        lambda path=None: bridge_assets.BridgePins(release="v0-test", platforms={}))


def _a_source_checkout(monkeypatch, tmp_path):
    (tmp_path / "site-packages" / bridges.RUSTWX_CRATE_RELATIVE).mkdir(parents=True)


def _a_bundle_without_the_three(monkeypatch, tmp_path):
    pins = bridge_assets.load_pins()
    platform, = pins.platforms
    bundle = pins.platforms[platform]
    slim = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename, bytes=bundle.bytes,
        sha256=bundle.sha256,
        binaries=tuple(pin for pin in bundle.binaries if pin.artifact not in ON_DEMAND))
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: bridge_assets.BridgePins(
        release=pins.release, platforms={platform: slim}))


@pytest.mark.parametrize("install", [
    _no_bundle_for_this_platform, _a_source_checkout, _a_bundle_without_the_three])
def test_doctor_does_not_name_the_fetch_where_it_cannot_install_the_artifact(
        pip_install, monkeypatch, tmp_path, install):
    """Reading the roster flag alone, doctor called the absence opt-in and
    said `gpuwm fetch-bridges` installs it on an install with no bundle
    pinned for its platform, where that command refuses, and in a source
    checkout, where the remedy printed beside it is the cargo build. There
    the artifact is not built, and the line says so with its old severity."""
    from gpuwm import doctor

    install(monkeypatch, tmp_path)
    checks = _absent_checks(monkeypatch)
    assert checks["rw_simradar"].severity == doctor.SEVERITY_UNREACHABLE
    assert checks["rw_mlexport"].severity == doctor.SEVERITY_UNREACHABLE
    assert checks["rw_globe_pack"].severity == doctor.SEVERITY_DEGRADED
    for name, check in checks.items():
        assert "fetched on demand" not in check.detail, (name, check.detail)
        assert FETCH not in check.detail, (name, check.detail)
        assert _first_command(check.remedy) != FETCH, (name, check.remedy)


def test_doctor_still_reports_a_mispointed_or_stale_on_demand_artifact(pip_install, monkeypatch, tmp_path):
    """On demand excuses absence only. A set override naming nothing, and a
    build older than this release's contract, are faults fetch does not fix."""
    from gpuwm import doctor

    monkeypatch.setenv("GPUWM_RW_SIMRADAR", str(tmp_path / "nothing-here"))
    assert doctor._simulated_radar_check().severity == doctor.SEVERITY_UNREACHABLE
    monkeypatch.setenv("GPUWM_RW_MLEXPORT", str(tmp_path / "nothing-here"))
    assert doctor._ml_export_check().severity == doctor.SEVERITY_UNREACHABLE
    monkeypatch.setenv("GPUWM_RW_GLOBE_PACK", str(tmp_path / "nothing-here"))
    assert doctor._globe_pack_check().severity == doctor.SEVERITY_DEGRADED

    stale = tmp_path / "rw_simradar"
    stale.write_bytes(b"a build that predates this release's contract")
    monkeypatch.setenv("GPUWM_RW_SIMRADAR", str(stale))
    assert doctor._simulated_radar_check().severity == doctor.SEVERITY_BROKEN
    monkeypatch.setenv("GPUWM_RW_GLOBE_PACK", str(stale))
    assert doctor._globe_pack_check().severity == doctor.SEVERITY_DEGRADED


def test_every_on_demand_artifact_has_a_doctor_line_of_its_own():
    """The sweep for artifacts without a check reports absence as degraded,
    which is why each of the three is covered by name."""
    from gpuwm import doctor

    assert ON_DEMAND <= set(doctor._CHECKED_ARTIFACTS)


# ---------------------------------------------------------------------------
# The two doors
# ---------------------------------------------------------------------------

def test_the_simulated_radar_door_refuses_with_the_fetch(pip_install, monkeypatch):
    from gpuwm import rustwx

    monkeypatch.setattr(rustwx, "simulated_radar_binary", lambda: None)
    with pytest.raises(rustwx.SimulatedRadarRefusal) as refusal:
        rustwx.require_simulated_radar_binary()
    said = str(refusal.value)
    assert "rw_simradar is not installed" in said
    assert _first_command(said.split("\n", 1)[1]) == FETCH
    assert "fetched on demand" not in said


def test_a_forecast_asking_for_radar_refuses_with_the_fetch_before_it_starts(pip_install, monkeypatch):
    from types import SimpleNamespace

    from gpuwm import rustwx, simulated_radar_config

    monkeypatch.setattr(rustwx, "simulated_radar_binary", lambda: None)
    experiment = SimpleNamespace(simulated_radar=SimpleNamespace(enabled=True))
    (label, sentence), = simulated_radar_config.door_refusals(experiment)
    assert label == simulated_radar_config.REFUSAL_LABEL
    assert FETCH in sentence and "fetched on demand" not in sentence


def test_the_ml_export_door_refuses_with_the_fetch(pip_install, monkeypatch, tmp_path, capsys):
    from gpuwm import ml_export

    monkeypatch.setattr(ml_export.Binary, "find", lambda self: None)
    args = argparse.Namespace(
        inputs=["wrfout_d01_2026-09-29_00_00_00"], out=tmp_path / "x", levels="wb13",
        variables=None, grid="native", regrid="bilinear", names="wb2", layout="analysis",
        domains=None, every=None, start=None, end=None, config=None, zip=False,
        overwrite=False, skip_unavailable=False, threads=None, append=False,
        finalize=False, list=False, json=False)
    assert ml_export.main(args) == ml_export.EXIT_NOT_STAGED
    said = capsys.readouterr().err
    assert "rw_mlexport" in said and "fetched on demand" not in said
    assert _first_command(said.split("\n", 1)[1]) == FETCH


def restored_artifacts():
    return tuple(a for a in bridge_assets.BUNDLED_ARTIFACTS if a.name in ON_DEMAND)
