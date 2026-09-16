"""The wheel carries the Rust, and carries it in a usable state.

Four defects are pinned here, all of them found by building and
installing rather than by reading:

1. setuptools' ``build/lib*`` copy tree is not pruned between builds, so
   staging a second platform shipped BOTH platforms' binaries -- a
   "manylinux" wheel containing Windows ``.exe`` files, 105.99 MB, over
   PyPI's 100 MB cap.
2. The staged binaries installed 0644 under ``pip`` and the first
   ``subprocess.run`` died with ``PermissionError``.  Measured on a real
   Linux userland: 11/11 artifacts lacked the executable bit after
   ``pip install``.  Defect 4 below is the reason, found later; the
   repair in :func:`gpuwm.bridges.ensure_executable` stays for wheels
   published before that fix.
3. A wheel with an empty ``libexec/bridges`` is indistinguishable from a
   correct one until a door is opened, so the staging tool must refuse
   rather than produce one.
4. The wheel stamped its staged artifacts ``0o755`` with no file-type
   bits, and pip's ``zip_item_is_executable`` requires ``S_ISREG``, so
   pip dropped the execute bits the stamping exists to set.  ``uv``,
   which does not apply that predicate, installed the same wheel 0775
   and hid it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

import pytest

from gpuwm import bridge_assets, bridges

_TOOLS = Path(bridges.__file__).resolve().parent.parent / "tools"
if str(_TOOLS.parent) not in sys.path:
    sys.path.insert(0, str(_TOOLS.parent))

stage_wheel_bridges = pytest.importorskip("tools.stage_wheel_bridges")


def test_packaged_bridge_dir_is_inside_the_package():
    """It must be package data, or the wheel cannot carry it.

    ``<root>/libexec/bridges`` -- the sealed-runtime layout -- sits BESIDE
    the package and can never be swept into a wheel.  This rung is the
    one that can.
    """

    packaged = bridges.packaged_bridge_dir()
    package_root = Path(bridges.__file__).resolve().parent
    assert packaged.is_relative_to(package_root), (
        f"{packaged} is not inside {package_root}, so setuptools cannot "
        f"ship it as package data")
    assert packaged.name == "bridges"


def test_every_ladder_offers_the_in_package_rung():
    """All six resolution ladders, not five.

    A rung added to one door and forgotten on another is exactly how a
    wheel install works for the renderer and refuses for the decoder.
    """

    import gpuwm.rustwx as rustwx
    import gpuwm.rustwx_fetch as rustwx_fetch
    from gpuwm import netcdf_bridge, tui_cli
    from gpuwm.obs import dealias_region, nexrad

    packaged = bridges.packaged_bridge_dir()
    ladders = {
        "bridges.grib2_dump": bridges.bridge_candidates("grib2_dump"),
        "rustwx.renderer": rustwx.renderer_candidates(),
        "rustwx_fetch.fetch": rustwx_fetch.fetch_candidates(),
        "obs.nexrad": nexrad.nexrad_candidates(),
        "obs.dealias_region": dealias_region.region_bridge_candidates(),
        "netcdf_bridge": netcdf_bridge.netcdf_candidates(),
        "tui_cli": tui_cli.tui_candidates(),
    }
    missing = [name for name, candidates in ladders.items()
               if not any(packaged in candidate.parents
                          for candidate in candidates)]
    assert not missing, (
        f"{len(missing)} of {len(ladders)} ladders never look inside the "
        f"package, so a wheel install cannot reach their artifact: {missing}")


def test_staging_refuses_rather_than_producing_an_empty_wheel(tmp_path,
                                                              monkeypatch):
    """A missing build must be a named refusal, not a quiet empty dir."""

    monkeypatch.setattr(stage_wheel_bridges, "_REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit) as raised:
        stage_wheel_bridges.stage("linux-x86_64",
                                  destination=tmp_path / "out")
    message = str(raised.value)
    assert "not built" in message
    assert "cargo build" in message, "the refusal must name the remedy"
    # Every missing artifact is listed, not just the first.
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        assert artifact.name in message, f"{artifact.name} not named"


def test_purge_removes_a_stale_platform_from_the_build_tree(tmp_path):
    """Defect 1: the second platform wheel must not inherit the first.

    Fails without ``purge_build_staging``: the stale copy survives into
    the next wheel.
    """

    stale = (tmp_path / "build" / "lib.win-amd64-cpython-313" / "gpuwm"
             / "libexec" / "bridges")
    stale.mkdir(parents=True)
    (stale / "rw_wrfbatch.exe").write_bytes(b"windows binary")
    (stale / "rw_netcdf.exe").write_bytes(b"windows binary")
    assert stale.exists()

    removed = stage_wheel_bridges.purge_build_staging(tmp_path)

    assert removed, "purge reported nothing removed"
    assert not stale.exists(), (
        "the stale platform's binaries survived into the build tree, so the "
        "next wheel would ship both platforms")


def test_staged_bundle_records_its_platform(tmp_path, monkeypatch):
    """A wheel that carries binaries must say which platform they are for."""

    monkeypatch.setattr(stage_wheel_bridges, "_REPO_ROOT", tmp_path)
    platform = "linux-x86_64"
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        origin = stage_wheel_bridges.source_path(artifact, platform)
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_bytes(b"artifact bytes for " + artifact.name.encode())

    destination = tmp_path / "staged"
    stage_wheel_bridges.stage(platform, destination=destination)

    manifest = json.loads(
        (destination / stage_wheel_bridges.MANIFEST_NAME).read_text("utf-8"))
    assert manifest["platform"] == platform
    assert len(manifest["artifacts"]) == len(bridge_assets.BUNDLED_ARTIFACTS)
    staged_names = {path.name for path in destination.iterdir()}
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        assert bridge_assets.artifact_filename(artifact, platform) in staged_names


@pytest.mark.skipif(os.name == "nt",
                    reason="Windows does not consult an executable mode bit")
def test_resolution_repairs_the_mode_pip_did_not_preserve(tmp_path,
                                                          monkeypatch):
    """Defect 2: pip installs package data 0644; running it needs 0755.

    Fails without ``bridges.ensure_executable``: the resolver hands back a
    path that cannot be executed, which is what a wheel install actually
    produced on Linux.
    """

    packaged = tmp_path / "gpuwm" / "libexec" / "bridges"
    packaged.mkdir(parents=True)
    binary = packaged / "rw_netcdf"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o644)          # exactly what pip leaves behind
    monkeypatch.setattr(bridges, "packaged_bridge_dir", lambda: packaged)

    assert not binary.stat().st_mode & stat.S_IXUSR, "fixture is not 0644"
    repaired = bridges.ensure_executable(binary)
    assert repaired.stat().st_mode & stat.S_IXUSR, (
        "the executable bit was not restored, so a wheel-installed bridge "
        "would die with PermissionError on first use")


@pytest.mark.skipif(os.name == "nt", reason="mode bits are POSIX-only")
def test_mode_repair_leaves_files_outside_the_package_alone(tmp_path,
                                                            monkeypatch):
    """Only what this package ships is re-permissioned."""

    packaged = tmp_path / "packaged"
    packaged.mkdir()
    monkeypatch.setattr(bridges, "packaged_bridge_dir", lambda: packaged)

    foreign = tmp_path / "elsewhere" / "rw_netcdf"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"not ours")
    foreign.chmod(0o644)

    bridges.ensure_executable(foreign)

    assert not foreign.stat().st_mode & stat.S_IXUSR, (
        "a file outside the package was re-permissioned; gpuwm does not own "
        "a checkout build, a ~/.gpuwm copy, or an override")


#: Stamps a synthetic wheel with ``setup.py``'s own code, then asks pip
#: itself whether each member would install executable.
#:
#: It runs in a child interpreter, and that is not caution.  Two of the
#: three imports it needs poison the parent, both measured: ``setup.py``
#: calls ``setup()`` at import, and importing ``pip._internal`` pulls in
#: ``distutils`` before setuptools has replaced it.  Either one splits
#: ``Distribution`` into two classes, after which
#: ``tests/test_sdist_excludes_staged_bridges.py`` fails two tests with
#: ``TypeError: dist must be a Distribution instance`` in the same run.
#:
#: ``zip_item_is_executable`` is the predicate that decides whether an
#: installed file keeps its execute bits, so the real one is the right
#: oracle.  The fallback is that function's body, kept so the probe still
#: measures something in an environment without pip.
_WHEEL_MODE_PROBE = """
import importlib.util, json, pathlib, stat, sys, zipfile

import setuptools
setuptools.setup = lambda *args, **kwargs: None
spec = importlib.util.spec_from_file_location("_gpuwm_setup_probe", sys.argv[1])
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

wheel = pathlib.Path(sys.argv[2])
staged = [setup._STAGED_PREFIX + "rw_mpas_mesh",
          setup._STAGED_PREFIX + "librw_netcdf.so"]
plain = ["gpuwm/__init__.py", setup._STAGED_PREFIX + "BUNDLE.json"]
with zipfile.ZipFile(wheel, "w") as archive:
    for name in staged + plain:
        # 0o100644: what bdist_wheel records for a data file, copied from
        # os.stat, and what a cross-built binary carries as well because
        # Windows has no executable bit to copy.
        info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, b"x")

stamped = setup._force_executable_bits(wheel)

try:
    from pip._internal.utils.unpacking import zip_item_is_executable
    oracle = "pip"
except Exception:
    oracle = "restated"

    def zip_item_is_executable(info):
        mode = info.external_attr >> 16
        return bool(mode and stat.S_ISREG(mode) and mode & 0o111)

with zipfile.ZipFile(wheel) as archive:
    modes = {info.filename: info.external_attr >> 16
             for info in archive.infolist()}
    executable = {info.filename: bool(zip_item_is_executable(info))
                  for info in archive.infolist()}
print(json.dumps({"stamped": stamped, "modes": modes, "oracle": oracle,
                  "executable": executable,
                  "staged": staged, "plain": plain}))
"""


def test_the_wheel_stamps_a_mode_pip_will_honour_on_every_staged_artifact(
        tmp_path):
    """Defect 4: a mode with no file-type bits is dropped by pip.

    ``_force_executable_bits`` wrote ``0o755 << 16`` into each staged
    member's ``external_attr``.  A zip's high half is a whole ``st_mode``,
    file-type bits included, and pip tests it as one, so
    ``S_ISREG(0o755)`` was false and pip installed the bridge binaries
    0644.  Measured on the published 2.7.4 manylinux wheel: the first
    door to reach a bridge refused by name with rc 2, ``chmod +x`` fixed
    it, and ``uv`` -- which does not apply that predicate -- installed
    the same wheel 0775 and never showed it.

    The fixture gives every member the 0o100644 a cross-built wheel
    carries, because Windows has no executable bit for bdist_wheel to
    copy; that is the state the stamping exists to correct.  The archive
    is then read back through pip's own predicate rather than against the
    constant, so a future edit that drops the type bits trips here
    instead of in a release check.
    """

    import subprocess

    setup_py = Path(bridges.__file__).resolve().parent.parent / "setup.py"
    if not setup_py.exists():      # an installed copy carries no setup.py
        pytest.skip("setup.py is not present outside a source tree")
    wheel = tmp_path / "gpuwm-0.0.0-py3-none-manylinux_2_28_x86_64.whl"
    completed = subprocess.run(
        [sys.executable, "-c", _WHEEL_MODE_PROBE, str(setup_py), str(wheel)],
        capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])

    assert report["stamped"] == len(report["staged"]), (
        f"stamped {report['stamped']} artifact(s); the wheel staged "
        f"{len(report['staged'])}")
    for name in report["staged"]:
        mode = report["modes"][name]
        assert stat.S_ISREG(mode), (
            f"{name} carries mode 0o{mode:o}, which has no regular-file type "
            "bits, so pip's zip_item_is_executable rejects it and installs "
            "the binary without its execute bits")
        assert mode & 0o111, f"{name} carries mode 0o{mode:o}"
        assert report["executable"][name], (
            f"pip would not preserve the execute bits on {name} "
            f"(oracle: {report['oracle']})")
    for name in report["plain"]:
        mode = report["modes"][name]
        assert stat.S_ISREG(mode), (
            f"{name} carries mode 0o{mode:o}, which names no file type")
        assert not report["executable"][name], (
            f"{name} is not a staged program and must not install executable")


def test_netcdf_decoder_is_a_declared_bundled_artifact():
    """NetCDF decode moved to Rust, so the Rust must ship with the wheel.

    Without this entry the decode is unreachable on a bare install and
    every NetCDF route refuses -- the exact "engine-proven but not
    shipped" failure this change exists to close.
    """

    names = {artifact.name for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert "rw_netcdf" in names, (
        "rw_netcdf is not in BUNDLED_ARTIFACTS, so no release bundle and no "
        "platform wheel carries it")
    artifact = next(a for a in bridge_assets.BUNDLED_ARTIFACTS
                    if a.name == "rw_netcdf")
    from gpuwm import netcdf_bridge
    assert artifact.env_var == netcdf_bridge.NETCDF_ENV, (
        "the declared override variable and the resolver's disagree")


# ---------------------------------------------------------------------------
# Staging a HOME, not only a wheel
# ---------------------------------------------------------------------------
#
# The measured defect: `~/.gpuwm/bridges` on a box that had been staged
# by hand carried sixteen of the declared artifacts.  The two
# absent ones were `rw_netcdf` and `netcdf_writer` -- the NetCDF decoder
# and the NetCDF WRITER behind the DEFAULT wrfout engine -- so every
# NetCDF source refused to decode and every history write would have
# refused, on a home that looked staged.  A hand-copied directory has no
# declaration to check itself against; this tool has one, and until now
# it could only write into the package.

def _fake_build_tree(root: Path, platform: str) -> None:
    """Every declared artifact, built, as bytes naming itself."""

    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        origin = stage_wheel_bridges.source_path(artifact, platform)
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_bytes(b"artifact bytes for " + artifact.name.encode())


def test_a_clean_home_is_staged_with_every_declared_artifact(tmp_path,
                                                             monkeypatch):
    """`--dest` stages a bridge directory, declaration-complete.

    Table-driven, so the artifact that joins BUNDLED_ARTIFACTS next
    travels here on the same commit -- which is the property the hand
    copy did not have.
    """

    monkeypatch.setattr(stage_wheel_bridges, "_REPO_ROOT", tmp_path)
    platform = "win-x86_64"
    _fake_build_tree(tmp_path, platform)
    home_bridges = tmp_path / "home" / ".gpuwm" / "bridges"

    exit_code = stage_wheel_bridges.main(
        ["--platform", platform, "--dest", str(home_bridges)])

    assert exit_code == 0
    staged = {path.name for path in home_bridges.iterdir()}
    expected = {bridge_assets.artifact_filename(artifact, platform)
                for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert expected <= staged, (
        f"a staged home is missing {sorted(expected - staged)}")
    assert "netcdf_writer.dll" in staged, (
        "the NetCDF writer behind the default wrfout engine was not "
        "staged, so every history write on this home refuses")
    manifest = json.loads(
        (home_bridges / stage_wheel_bridges.MANIFEST_NAME).read_text("utf-8"))
    assert manifest["platform"] == platform
    assert {record["artifact"] for record in manifest["artifacts"]} == {
        artifact.name for artifact in bridge_assets.BUNDLED_ARTIFACTS}


def test_staging_a_home_replaces_artifacts_and_keeps_everything_else(
        tmp_path, monkeypatch):
    """A home is not a wheel: nothing outside the declaration is removed.

    The packaged destination is emptied first, deliberately -- a wheel
    carrying two platforms' binaries is both wrong and over PyPI's cap.
    A user's `~/.gpuwm/bridges` is the opposite case: it holds their
    kept-stale copies and the renderer's `assets/` basemap tree, none of
    which this tool staged and none of which it may delete.
    """

    monkeypatch.setattr(stage_wheel_bridges, "_REPO_ROOT", tmp_path)
    platform = "win-x86_64"
    _fake_build_tree(tmp_path, platform)
    home_bridges = tmp_path / "home" / ".gpuwm" / "bridges"
    (home_bridges / "assets" / "basemap").mkdir(parents=True)
    (home_bridges / "assets" / "basemap" / "coast.shp").write_bytes(b"map")
    kept = home_bridges / "rw_wrfbatch.exe.stale-20260805"
    kept.write_bytes(b"yesterday")
    stale = home_bridges / bridge_assets.artifact_filename(
        bridge_assets.BUNDLED_ARTIFACTS[0], platform)
    stale.write_bytes(b"an older build")

    stage_wheel_bridges.main(
        ["--platform", platform, "--dest", str(home_bridges)])

    assert (home_bridges / "assets" / "basemap" / "coast.shp").read_bytes() \
        == b"map", "the renderer's basemap tree was deleted"
    assert kept.read_bytes() == b"yesterday", (
        "a file this tool never staged was deleted from the user's home")
    assert stale.read_bytes() != b"an older build", (
        "a declared artifact already present was not replaced")
