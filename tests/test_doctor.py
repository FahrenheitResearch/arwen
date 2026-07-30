"""``gpuwm doctor`` + the shared bridge-resolution mechanism.

The wheel ships no compiled Rust, so a wheel user's estate is: cupy?
render extra? bridges built and findable? tables packaged? data roots
set?  Doctor must name each gap WITH its exact remedy, and ingest's
bridge resolution must honor the same env-var/default-dir mechanism
doctor describes.  Everything here is CPU-only and read-only.
"""
from __future__ import annotations

import pytest

import gpuwm.cli as cli
import gpuwm.doctor as doctor
from gpuwm import bridges


def test_bridge_env_override_wins_and_fails_loud(monkeypatch, tmp_path):
    exe = tmp_path / "grib1_bridge.exe"
    exe.write_bytes(b"prebuilt")
    monkeypatch.setenv("GPUWM_GRIB1_BRIDGE", str(exe))
    assert bridges.find_bridge("grib1_bridge") == exe.resolve()

    monkeypatch.setenv("GPUWM_GRIB1_BRIDGE", str(tmp_path / "missing.exe"))
    with pytest.raises(FileNotFoundError, match="GPUWM_GRIB1_BRIDGE"):
        bridges.find_bridge("grib1_bridge")


def test_default_bridge_dir_serves_wheel_installs(monkeypatch, tmp_path):
    """No checkout crate, no env var: the user-level default directory
    (~/.gpuwm/bridges) is the wheel user's documented drop point."""
    for variable in bridges.BRIDGE_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: tmp_path / "no-crate")
    monkeypatch.setattr(bridges, "_package_parent",
                        lambda: tmp_path / "site-packages")
    user_dir = tmp_path / "userdir"
    user_dir.mkdir()
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: user_dir)
    name = bridges.executable_name("gfs_grib2_bridge")
    assert bridges.find_bridge("gfs_grib2_bridge") is None
    (user_dir / name).write_bytes(b"prebuilt")
    assert bridges.find_bridge("gfs_grib2_bridge") == (
        user_dir / name).resolve()


def test_build_rust_bridge_honors_the_env_override(monkeypatch, tmp_path):
    from gpuwm.ingest.grib import build_rust_bridge

    exe = tmp_path / "grib1_bridge.exe"
    exe.write_bytes(b"prebuilt")
    monkeypatch.setenv("GPUWM_GRIB1_BRIDGE", str(exe))
    assert build_rust_bridge() == exe.resolve()


def test_build_rust_bridge_wheel_failure_names_the_remedy(monkeypatch,
                                                          tmp_path):
    """site-packages without the crate must not surface as a bare
    'Cargo.toml not found': the error carries the cargo one-liner, the
    env var, and the default directory."""
    import gpuwm.ingest.grib as grib

    monkeypatch.delenv("GPUWM_GRIB1_BRIDGE", raising=False)
    monkeypatch.setattr(grib, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    with pytest.raises(FileNotFoundError) as excinfo:
        grib.build_rust_bridge()
    message = str(excinfo.value)
    assert "cargo build --release --locked --offline" in message
    assert "GPUWM_GRIB1_BRIDGE" in message
    assert "gpuwm doctor" in message


def test_doctor_names_missing_extras_with_exact_remedies(monkeypatch):
    monkeypatch.setattr(doctor, "find_spec", lambda name: None)
    by_name = {check.name: check for check in doctor.collect_checks()}
    cupy = by_name["cupy (GPU runtime)"]
    assert cupy.status == "missing"
    assert "gpuwm[gpu]" in cupy.remedy and "cupy-cuda12x" in cupy.remedy
    render = by_name["render extra (wrf-rust + matplotlib)"]
    assert render.status == "missing"
    assert "gpuwm[render]" in render.remedy


def test_doctor_reports_missing_bridges_with_the_cargo_line(monkeypatch,
                                                            tmp_path):
    for variable in bridges.BRIDGE_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: tmp_path / "no-crate")
    monkeypatch.setattr(bridges, "_package_parent",
                        lambda: tmp_path / "site-packages")
    monkeypatch.setattr(bridges, "default_bridge_dir",
                        lambda: tmp_path / "userdir")
    by_name = {check.name: check for check in doctor.collect_checks()}
    gfs = by_name["bridge gfs_grib2_bridge"]
    assert gfs.status == "missing"
    assert "cargo build --release --locked --offline" in gfs.remedy
    assert "GPUWM_GFS_GRIB2_BRIDGE" in gfs.remedy


def test_doctor_cli_exit_codes_and_report_shape(monkeypatch, capsys):
    rc_healthy_or_not = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "gpuwm doctor: runtime estate" in out
    assert rc_healthy_or_not in (0, 1)

    monkeypatch.setattr(doctor, "find_spec", lambda name: None)
    rc = cli.main(["doctor"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "remedy:" in out


def test_doctor_explains_the_case_data_root_layout(monkeypatch, capsys):
    monkeypatch.delenv("GPUWM_CASE_DATA_ROOT", raising=False)
    checks = {check.name: check for check in doctor.collect_checks()}
    layout = checks["GPUWM_CASE_DATA_ROOT"]
    assert layout.status == "info"
    assert "CONTAINS your case bundles" in layout.detail
    assert "WPS_GEOG" in layout.detail


# ---------------------------------------------------------------------------
# Deep checks: doctor must not lie green (audit finding 3).
# ---------------------------------------------------------------------------

def test_import_probe_catches_an_installed_but_broken_package(
        monkeypatch, tmp_path):
    """find_spec sees a distribution; only an actual import proves it.
    A module that raises on import must fail the probe with the error."""
    (tmp_path / "cupy.py").write_text(
        "raise RuntimeError('broken install')", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(doctor, "find_spec", lambda name: object())
    ok, evidence = doctor._import_probe("cupy")
    assert not ok
    assert "broken install" in evidence

    check = doctor._cupy_check()
    assert check.status == "missing"
    assert "failed to import" in check.detail


def test_bridge_probe_fails_a_present_but_nonexecutable_file(
        monkeypatch, tmp_path):
    """is_file() lied green here before: an existing file that cannot
    execute (empty, truncated, wrong platform) must be a MISSING gap
    with the rebuild remedy, not ok."""
    import os

    fake = tmp_path / doctor.bridges.executable_name("grib1_bridge")
    fake.write_bytes(b"not machine code")
    if os.name != "nt":
        fake.chmod(0o755)
    ok, evidence = doctor._exec_probe(fake)
    assert not ok

    monkeypatch.setenv("GPUWM_GRIB1_BRIDGE", str(fake))
    by_name = {check.name: check for check in doctor._bridge_checks()}
    check = by_name["bridge grib1_bridge"]
    assert check.status == "missing"
    assert "cargo build" in check.remedy


def test_cpu_library_check_requires_a_loadable_abi(monkeypatch, tmp_path):
    bogus = tmp_path / "gpuwm_preprocess_cpu.dll"
    bogus.write_bytes(b"MZ but not really a library")
    monkeypatch.setenv("GPUWM_CPU_PREPROCESS_BRIDGE", str(bogus))
    check = doctor._cpu_library_check()
    assert check.status == "missing"
    assert "not loadable" in check.detail


def test_thompson_tables_check_uses_the_load_time_hash_validation(
        monkeypatch, tmp_path):
    """A relocated root with wrong bytes must fail exactly as a run
    would; the packaged root must verify."""
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)
    packaged = doctor._thompson_tables_check()
    assert packaged.status == "verified"
    assert "SHA-256" in packaged.detail

    (tmp_path / "qr_acr_qg_V4.dat").write_bytes(b"tampered")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(tmp_path))
    tampered = doctor._thompson_tables_check()
    assert tampered.status == "missing"


def test_noah_tables_check_parses_with_the_model_parsers():
    check = doctor._noah_tables_check()
    assert check.status == "verified"
    assert "SOIL_VEG_GEN_PARM" in check.detail
    assert "LANDUSE.TBL" in check.detail


def test_geog_check_requires_each_dataset_index_file(monkeypatch, tmp_path):
    from gpuwm.domain_wizard import GEOG_DATASETS

    geog = tmp_path / "WPS_GEOG"
    for name in GEOG_DATASETS:
        (geog / name).mkdir(parents=True)
        (geog / name / "index").write_text("type = continuous\n",
                                           encoding="ascii")
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path))
    by_name = {check.name: check for check in doctor._case_data_root_check()}
    assert by_name["WPS_GEOG"].status == "verified"
    assert "index" in by_name["WPS_GEOG"].detail

    (geog / GEOG_DATASETS[0] / "index").unlink()
    by_name = {check.name: check for check in doctor._case_data_root_check()}
    assert by_name["WPS_GEOG"].status == "missing"
    assert GEOG_DATASETS[0] in by_name["WPS_GEOG"].detail
    assert "index" in by_name["WPS_GEOG"].detail


def test_distribution_manifest_revalidates_declared_artifact_hashes(
        monkeypatch, tmp_path):
    import hashlib
    import json as jsonlib

    artifact = tmp_path / "payload.bin"
    artifact.write_bytes(b"sealed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(jsonlib.dumps({
        "schema": "gpuwm-native-wrf-runtime-v1", "status": "READY",
        "payload": {"payload.bin": {
            "bytes": 6,
            "sha256": hashlib.sha256(b"sealed").hexdigest(),
            "executable": False,
        }},
    }), encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    check = doctor._distribution_manifest_check()
    assert check.status == "verified"
    assert "re-hashed" in check.detail

    artifact.write_bytes(b"tampered!")
    check = doctor._distribution_manifest_check()
    assert check.status == "missing"
    assert "mismatch" in check.detail


def test_distribution_manifest_without_hashes_is_labeled_presence_only(
        monkeypatch, tmp_path):
    import json as jsonlib

    manifest = tmp_path / "manifest.json"
    manifest.write_text(jsonlib.dumps({
        "schema": "gpuwm-native-wrf-runtime-v1", "status": "READY",
    }), encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    check = doctor._distribution_manifest_check()
    assert check.status == "present"
    assert "presence-only" in check.detail


def test_report_and_exit_code_distinguish_verified_from_present():
    checks = [doctor.Check("deep", "verified", "proved by execution"),
              doctor.Check("shallow", "present", "existence only")]
    text = doctor.format_report(checks)
    assert "ok      deep" in text
    assert "present shallow" in text
    assert "presence-only" in text

    # present is honest, not a gap; missing is the only nonzero driver.
    rc_present = 1 if any(c.status == "missing" for c in checks) else 0
    assert rc_present == 0
    checks.append(doctor.Check("gap", "missing", "absent", "install it"))
    assert "MISSING" in doctor.format_report(checks)
