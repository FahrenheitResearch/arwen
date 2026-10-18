"""The projected MET library has a doctor line, judged the way gpuwm-hex loads it.

2.9.0 shipped no rw_projected_intermediate, so `gpuwm-hex intermediate
--source hrrr` exited 2 on every pip install and nothing in `gpuwm doctor`
said so.  The library now ships in the wheel and the bridge bundle; these
tests hold the doctor row to the same four entries and ABI gpuwm-hex binds.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpuwm import bridge_assets, bridges, doctor


def _artifact():
    return next(a for a in bridge_assets.BUNDLED_ARTIFACTS
                if a.name == "rw_projected_intermediate")


def test_the_library_ships_in_the_bundle_and_the_wheel_with_an_abi():
    artifact = _artifact()
    assert artifact.kind == "library" and artifact.crate == "tools/rw_wps"
    assert artifact.env_var == "GPUWM_PROJECTED_INTERMEDIATE_BRIDGE"
    assert artifact in bridge_assets.wheel_artifacts()
    assert bridge_assets.library_abi_for(artifact.name) == (
        "gpuwm_projected_intermediate_abi_version", 1)
    assert bridges.BRIDGE_ABI_MARKERS[artifact.name] == b"gpuwm_projected_window_v1"
    assert "rw_projected_intermediate" in doctor._CHECKED_ARTIFACTS


def test_the_crate_exports_every_entry_gpuwm_hex_binds():
    source = (Path(__file__).resolve().parents[1] / "tools/rw_wps/crates/"
              "projected-intermediate/src/lib.rs")
    if not source.is_file():
        pytest.skip("needs the source tree")
    text = source.read_text(encoding="utf-8")
    for entry in (*doctor._PROJECTED_INTERMEDIATE_ENTRIES,
                  "gpuwm_projected_intermediate_abi_version"):
        assert f"fn {entry}(" in text, entry


def test_an_override_naming_a_missing_file_is_reported_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_PROJECTED_INTERMEDIATE_BRIDGE", str(tmp_path / "absent"))
    check = doctor._projected_intermediate_check()
    assert check.status == "missing"
    assert "names a missing file" in check.detail
    assert check.blocking is False


def test_a_built_library_is_verified(monkeypatch):
    """Runs where the library is built (a checkout build or a staged bundle)."""
    artifact = _artifact()
    filename = bridge_assets.artifact_filename(
        artifact, bridge_assets.host_platform() or "linux-x86_64")
    root = Path(__file__).resolve().parents[1] / artifact.crate / "target"
    built = next((p for p in (Path(os.environ.get(artifact.env_var, "")),
                              root / "release" / filename, root / "debug" / filename)
                  if p.is_file()), None)
    if built is None:
        pytest.skip(f"{filename} is not built here")
    monkeypatch.setenv(artifact.env_var, str(built))
    check = doctor._projected_intermediate_check()
    assert check.status == "verified", check.detail
    assert "ABI 1" in check.detail
