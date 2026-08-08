"""``gpuwm doctor`` + the shared bridge-resolution mechanism.

The wheel ships no compiled Rust, so a wheel user's estate is: cupy?
render extra? bridges built and findable? tables packaged? data roots
set?  Doctor must name each gap WITH its exact remedy, and ingest's
bridge resolution must honor the same env-var/default-dir mechanism
doctor describes.  Everything here is CPU-only and read-only.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

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
    # Pin the box away from the test machine's own driver: the CuPy
    # remedy is now a function of the box's CUDA major, so an
    # unpinned run would assert different things on different hosts.
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: None)
    by_name = {check.name: check for check in doctor.collect_checks()}
    cupy = by_name["cupy (GPU runtime)"]
    assert cupy.status == "missing"
    # Undetectable major: BOTH extras, neither presented as the default.
    assert "gpuwm[gpu-cu12]" in cupy.remedy
    assert "gpuwm[gpu-cu13]" in cupy.remedy
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


@pytest.mark.parametrize("explain", [False, True])
def test_doctor_cli_exit_codes_and_report_shape(monkeypatch, capsys,
                                                explain):
    """Both layers report the same estate, and neither changes the code.

    The report shape is now a gate keyed on ``--explain``, so it is
    tested at both of its values.  What each layer owes the reader
    differs -- the terse one owes a next command per gap, the full one
    owes the pasteable remedy block -- but the FINDING and the exit
    code are the same, and that is the property a reader relies on when
    they re-run with the flag after seeing a gap.
    """

    argv = ["doctor", "--explain"] if explain else ["doctor"]
    rc_healthy_or_not = cli.main(argv)
    out = capsys.readouterr().out
    if explain:
        assert "gpuwm doctor: runtime estate" in out
    else:
        # Line 1 is the install headline (which copy of gpuwm produced
        # this); the report's own banner follows it.
        assert "\ngpuwm doctor\n" in out
    assert rc_healthy_or_not in (0, 1)

    monkeypatch.setattr(doctor, "find_spec", lambda name: None)
    rc = cli.main(argv)
    # Warn-not-block: an absent optional extra (cupy, the render pair)
    # is REPORTED as MISSING with its remedy, but does not move the
    # exit code -- only broken/integrity-suspect findings do, and
    # un-importing the extras breaks nothing else in the estate.
    assert rc == rc_healthy_or_not
    out = capsys.readouterr().out
    assert "MISSING" in out
    if explain:
        # The whole pasteable block, exactly where it always was.
        assert "remedy:" in out
    else:
        # One line per gap, each naming THE command that closes it,
        # plus the pointer at the full layer (printed exactly when
        # there are gaps to explain).
        assert "remedy:" not in out
        # The CuPy line names a CUDA-major-specific extra, never the
        # bare `gpu` alias: whichever branch this box takes, the brief
        # line has to say which wheel it is recommending.
        assert "-> pip install 'gpuwm[gpu-cu1" in out
        assert "--explain" in out


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
    would; the packaged root must earn whichever answer it has.

    The packaged root has two correct states and this used to assert
    only one.  A developer tree has staged the externalized assets, so
    the check verifies.  A fresh clone of the PUBLIC repository has
    not: `freezeH2O.dat` is 243 MiB, past GitHub's blob limit, and is
    published as a release asset rather than committed -- so `missing`
    there is the honest answer, and the assertion that it be `verified`
    was a statement about the author's disk.  It passed on every
    developer box and failed on the ubuntu publish runner's clean
    checkout, which is the worst place to learn it.

    Neither branch is a skip.  The incomplete tree is held to MORE than
    the complete one: the gap must be exactly the externalized assets
    (a packaged asset missing is a broken install, not a pending
    download) and the remedy must be the command that stages them.
    """

    from gpuwm import table_assets
    from gpuwm.physics_compat import thompson_table_root

    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)
    root = Path(thompson_table_root())
    _valid, invalid, absent = table_assets.classify_assets(root)
    packaged = doctor._thompson_tables_check()
    if not invalid and not absent:
        assert packaged.status == "verified"
        assert "SHA-256" in packaged.detail
    else:
        assert packaged.status == "missing"
        assert not invalid, (
            f"the packaged root holds files with the wrong bytes: {invalid}")
        assert all(
            asset.filename in table_assets.EXTERNALIZED_TABLE_FILENAMES
            for asset in absent), [asset.filename for asset in absent]
        assert "gpuwm fetch-tables" in (packaged.remedy or "")
        assert "reinstall" not in (packaged.remedy or "")

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

    from conftest import complete_runtime_manifest

    artifact = tmp_path / "payload.bin"
    artifact.write_bytes(b"sealed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(jsonlib.dumps(complete_runtime_manifest({
        "payload.bin": {
            "bytes": 6,
            "sha256": hashlib.sha256(b"sealed").hexdigest(),
            "executable": False,
        }})), encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    check = doctor._distribution_manifest_check()
    assert check.status == "verified"
    assert "re-hashed" in check.detail

    artifact.write_bytes(b"tampered!")
    check = doctor._distribution_manifest_check()
    assert check.status == "missing"
    assert "mismatch" in check.detail


def test_distribution_manifest_missing_backend_inventory_is_named_here(
        monkeypatch, tmp_path):
    """The exact document a field user hand-authored, refused up front.

    It carried ``schema`` and ``status`` and nothing else.  Doctor
    blessed it, ``_source_identity`` accepted it, and the run died
    minutes later inside the preprocessing selector on
    ``contract.platform.backends``.  Every one of those keys is named
    here now, before anything runs.
    """

    import json as jsonlib

    from conftest import complete_runtime_manifest

    manifest = tmp_path / "manifest.json"
    manifest.write_text(jsonlib.dumps({
        "schema": "gpuwm-native-wrf-runtime-v1", "status": "READY",
    }), encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    check = doctor._distribution_manifest_check()
    assert check.status == "missing"
    assert check.blocking
    for field in ("artifact", "source", "contract", "payload"):
        assert field in check.detail
    assert check.action == "unset GPUWM_NATIVE_DISTRIBUTION_MANIFEST"

    # And the field the field user's run actually died on, when the
    # rest of the document is otherwise well formed.
    document = complete_runtime_manifest()
    del document["contract"]["platform"]["backends"]
    manifest.write_text(jsonlib.dumps(document), encoding="utf-8")
    check = doctor._distribution_manifest_check()
    assert check.status == "missing"
    assert "contract.platform.backends" in check.detail


def test_distribution_manifest_without_an_artifact_inventory_is_refused(
        monkeypatch, tmp_path):
    """No per-artifact hashes is a defect, not a "presence-only" pass.

    Doctor used to label such a document ``present`` and move on, which
    is how a manifest nobody could actually bind decoders out of read as
    a healthy line in the report.  A sealed archive's installer always
    writes the inventory; a document without one is not one.
    """

    import json as jsonlib

    from conftest import complete_runtime_manifest

    manifest = tmp_path / "manifest.json"
    document = complete_runtime_manifest()
    del document["payload"]
    manifest.write_text(jsonlib.dumps(document), encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    check = doctor._distribution_manifest_check()
    assert check.status == "missing"
    assert "payload" in check.detail


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


def test_doctor_reports_absent_geography_even_with_the_root_unset(
        tmp_path, monkeypatch, capsys):
    """P5: doctor said "no gaps" on a machine with zero static geography.

    v1.0.0 returned a single `info` when GPUWM_CASE_DATA_ROOT was unset
    and never looked for WPS_GEOG at all -- contradicting the README,
    which says doctor requires each dataset's index file, and
    greenlighting a box on which nothing downstream could run.  The
    default root resolves with or without the variable, so the check
    does too.
    """
    import gpuwm.doctor as doctor_module
    from gpuwm.geog_assets import default_geog_root

    monkeypatch.delenv("GPUWM_CASE_DATA_ROOT", raising=False)
    monkeypatch.setattr(doctor_module, "default_case_data_root",
                        lambda: tmp_path / "absent", raising=False)
    monkeypatch.setattr("gpuwm.case_data.default_case_data_root",
                        lambda: tmp_path / "absent")
    assert not default_geog_root().is_dir()

    checks = doctor_module._case_data_root_check()
    by_name = {c.name: c for c in checks}
    assert by_name["GPUWM_CASE_DATA_ROOT"].status == "info"
    geog = by_name["WPS_GEOG"]
    assert geog.status == "missing", (
        "an absent WPS_GEOG tree is a gap, not silence")
    assert "gpuwm fetch-geog" in (geog.remedy or "")
    assert "Nothing" in geog.detail

    # And a complete tree at the SAME default location verifies.
    staged = tmp_path / "absent" / "WPS_GEOG"
    from gpuwm.domain_wizard import GEOG_DATASETS
    for name in GEOG_DATASETS:
        (staged / name).mkdir(parents=True)
        (staged / name / "index").write_text("type=continuous\n",
                                             encoding="ascii")
    checks = doctor_module._case_data_root_check()
    by_name = {c.name: c for c in checks}
    assert by_name["WPS_GEOG"].status == "verified"


# ---------------------------------------------------------------------------
# Remedies have to be true for the install that prints them
# ---------------------------------------------------------------------------

def test_the_bridge_remedy_is_a_real_bootstrap_on_a_pip_install(
        monkeypatch, tmp_path):
    """A pip machine has no `tools/grib1_bridge` to cd into.

    The wheel ships no Rust, so on a pip-only install NOTHING decodes
    GRIB -- and all six bridge remedies said `cd tools/grib1_bridge &&
    cargo build`, naming a directory that does not exist, with no
    repository URL anywhere in the output.  A first-time user reads
    that as a broken install.
    """

    from gpuwm import bridges

    # site-packages: no crate anywhere under the package parent, which
    # is the single root every artifact path is resolved from.
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: tmp_path / "tools" / "grib1_bridge")
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: True)
    remedy = bridges.bridge_remedy("grib1_bridge")
    assert bridges.REPOSITORY_URL in remedy
    assert f"git clone {bridges.REPOSITORY_URL}" in remedy
    assert "cargo build --release --locked --offline" in remedy
    # It says WHY, so the reader knows this is a missing step and not a
    # broken package.
    assert "no Rust sources" in remedy
    # And it does not send them to a directory that is not there.
    assert "cd tools/grib1_bridge" not in remedy

    # Without cargo, the toolchain install leads -- as a bare command
    # under a `#` comment that explains it, never as a labelled string.
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: False)
    lines = [line.strip() for line in bridges.build_from_clone_hint()]
    assert lines[0].startswith("# Rust is not on PATH")
    commands = [line for line in lines if not line.startswith("#")]
    assert commands[0] == bridges.rust_toolchain_install_command()
    assert commands[1] == bridges.cargo_activation_command()
    assert commands[2].startswith("git clone ")


def test_a_bridge_that_predates_the_contract_is_missing_not_ok(
        monkeypatch, tmp_path):
    """"It launches" is not "it speaks this release's contract".

    The wheel ships no Rust, so upgrading the Python half leaves
    yesterday's binaries on disk untouched.  1.1.0 moved the GFS series
    file from two columns to three; a 1.0.1 `gfs_grib2_bridge` still
    launched and still printed its usage diagnostic, so doctor reported
    it `ok` -- and then every preparation died with `series line 1 must
    be HOUR<TAB>GRIB2`, blaming the series file gpuwm had just written
    correctly.  A node-7 validation run found the cause by diffing two
    git tags, which is not a diagnostic path a product may rely on.
    """

    from gpuwm import bridges

    stale = tmp_path / bridges.executable_name("gfs_grib2_bridge")
    stale.write_bytes(
        b"a real executable, from before the contract changed: "
        b"series line {} must be HOUR<TAB>GRIB2")
    current = tmp_path / "current.bin"
    current.write_bytes(
        b"...\x00" + bridges.BRIDGE_ABI_MARKERS["gfs_grib2_bridge"])

    ok, evidence = bridges.bridge_abi_matches("gfs_grib2_bridge", stale)
    assert not ok
    assert "predates" in evidence and "rebuild" in evidence
    assert bridges.bridge_abi_matches("gfs_grib2_bridge", current)[0]

    # And it reaches the report as a gap with a rebuild remedy, not as a
    # green line -- which is the whole finding.
    monkeypatch.setattr(bridges, "find_bridge",
                        lambda name: stale if name == "gfs_grib2_bridge"
                        else None)
    monkeypatch.setattr(doctor, "_exec_probe",
                        lambda path: (True, "executes"))
    checks = {check.name: check for check in doctor._bridge_checks()}
    skewed = checks["bridge gfs_grib2_bridge"]
    assert skewed.status == "missing"
    assert "predates" in skewed.detail
    assert skewed.remedy and "cargo build" in skewed.remedy

    # Every bridge gpuwm resolves declares a contract marker: an
    # undeclared one would pass this handshake silently forever.
    assert set(bridges.BRIDGE_ENV) == set(bridges.BRIDGE_ABI_MARKERS)


def test_the_sealer_and_the_doctor_share_one_bridge_contract(monkeypatch):
    """One marker table, so the two surfaces cannot drift apart.

    `native_wrf_distribution` has refused stale `grib2_inventory` and
    `grib2_dump` builds since before 1.1; that table simply did not
    cover the GFS bridge, and doctor had no table at all.  Two copies is
    how the series-contract change got caught at sealing time and never
    at report time.
    """

    from gpuwm import bridges
    from gpuwm import native_wrf_distribution

    for name, marker in bridges.BRIDGE_ABI_MARKERS.items():
        assert native_wrf_distribution._BRIDGE_ABI_MARKERS[name] is marker


@pytest.mark.parametrize("windows", (False, True))
def test_the_unpinned_pip_bootstrap_wires_what_it_builds(
        monkeypatch, tmp_path, windows):
    """Running every command must actually close the gap it was for.

    A node-8 validation run pasted the whole report on a pip-only
    machine, ran every line of it, and doctor still reported six MISSING
    bridges: the wiring step -- copy into the default directory, or set
    the environment variable -- was offered as two `#` alternatives,
    because it is a choice.  It is still a choice; the copy is now the
    default and the environment variable the commented alternative, so
    the literal paste finishes.

    This is specifically the unpinned-release arm.  A pinned release is
    separately watched by `test_the_pip_remedy_offers_the_prebuilt_bundle_first`:
    its only live command is `gpuwm fetch-bridges` and this source route is
    deliberately retained as comments.  The destination here is asserted to
    be `default_bridge_dir()` itself, not a lookalike: what makes the copy work
    is that it lands in the exact directory `artifact_candidates` searches.
    """

    from gpuwm import bridges

    _force_shell(monkeypatch, windows)
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: tmp_path / "tools" / "grib1_bridge")
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: True)
    monkeypatch.setattr(bridges, "prebuilt_bundle_offer", lambda: None)
    remedy = bridges.bridge_remedy("grib1_bridge")
    _assert_remedy_lines_are_commands_or_comments(remedy, windows=windows)

    commands = [line.strip() for line in remedy.splitlines()
                if line.strip() and not line.strip().startswith("#")]
    destination = str(bridges.default_bridge_dir())
    copy_first = "Copy-Item" if windows else "cp"
    make_first = "New-Item" if windows else "mkdir"
    assert any(line.startswith(make_first) and f'"{destination}"' in line
               for line in commands), commands
    copies = [line for line in commands if line.startswith(copy_first)]
    assert len(copies) == 1, commands
    assert copies[-1].endswith(f'"{destination}"')
    assert bridges.executable_name("grib1_bridge") in copies[-1]
    # The directory is made before anything is copied into it.
    assert (commands.index(copies[0])
            > next(index for index, line in enumerate(commands)
                   if line.startswith(make_first)))
    # The environment variable survives -- demoted, not deleted: it is a
    # genuine alternative for a reader who does not want a second copy.
    assert f"#   {bridges.BRIDGE_ENV['grib1_bridge']}=" in remedy


def test_the_bridge_remedy_stays_one_line_in_a_checkout(monkeypatch,
                                                        tmp_path):
    """Where the crate exists, the clone would be noise."""

    from gpuwm import bridges

    crate = tmp_path / "tools" / "grib1_bridge"
    crate.mkdir(parents=True)
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    monkeypatch.setattr(bridges, "crate_dir", lambda: crate)
    remedy = bridges.bridge_remedy("grib1_bridge")
    assert bridges.CARGO_BUILD_HINT in remedy
    assert "git clone" not in remedy
    # ...and it names the real destination rather than a <clone> the
    # reader has to expand.
    assert "<" not in remedy and ">" not in remedy
    assert str(crate / "target" / "release") in remedy


def test_every_rust_remedy_is_install_aware(monkeypatch, tmp_path):
    """The renderer and fetch backbone live in tools/rustwx, also absent."""

    from gpuwm import bridges

    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    monkeypatch.setattr(bridges, "crate_dir", lambda: tmp_path / "absent")
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: True)
    hint = bridges.install_aware_build_hint(
        bridges.cargo_build_one_liner(bridges.RUSTWX_CRATE_RELATIVE),
        bridges.RUSTWX_CRATE_RELATIVE)
    assert "git clone" in hint
    assert bridges._shell_path(
        bridges.CLONE_DIR, bridges.RUSTWX_CRATE_RELATIVE) in hint


# ---------------------------------------------------------------------------
# The bootstrap has to BE a bootstrap, not a description of one
# ---------------------------------------------------------------------------
# The previous test here only checked that the word "copy-pasteable" had
# left the closing line.  It never read a remedy, so it passed while the
# emitted bootstrap contained `install Rust: winget ... (or
# https://rustup.rs)` -- prose fused to a command -- omitted the cargo
# activation a freshly-installed toolchain needs in the current shell,
# and joined the Windows build with `&&`, which Windows PowerShell 5.1
# cannot parse.  These assert the structure instead.

#: Every command a remedy is allowed to start a line with.  A whitelist,
#: not a heuristic: `install Rust: winget ...` fails it on the first
#: token, and so does `then EITHER set ...`.
_REMEDY_COMMANDS = frozenset({
    "pip", "python", "git", "cd", "cargo", "curl", "winget", "gpuwm",
    "bash", "sh", ".", "export", "$env:Path",
    # The wiring step, which the pip bootstrap now RUNS rather than
    # describing in a comment: a literal paste that builds six bridges
    # and installs none of them left a node-8 validation run at six
    # MISSING while every line it pasted was, technically, honest.
    "mkdir", "cp", "New-Item", "Copy-Item",
    # The CuPy remedy's first step when the box's CUDA major could not
    # be read: the reader has to look it up before choosing an extra,
    # and the lookup is a command, so it is printed as one.  Ships with
    # every NVIDIA driver on both platforms.
    "nvidia-smi",
})

#: A bare ALL-CAPS word is a placeholder the reader must expand.
_PLACEHOLDER_WORD = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

#: Spellings only Windows PowerShell understands.  The mirror of the
#: `&&` rule below, which is the half that had ever run: these tests
#: only ever executed on Windows, where a POSIX remedy carrying
#: `New-Item` would have been produced by a branch the host never took.
#: On a Linux runner that branch is the one under test.
_POWERSHELL_ONLY = ("New-Item", "Copy-Item", "-ItemType", "$env:")

def _frozen_shell_constants():
    """(module, name, value) for every module-level build hint.

    Found by importing every module in the package and searching it, not
    by naming modules: the trap is a build hint frozen at import, and a
    hand-written list is exactly what does not mention a new one.

    This used to search a three-name tuple, which closed half the trap.
    A new CONSTANT inside `gpuwm.bridges` would have been caught; a new
    MODULE was invisible, and that is the half that fired.  The 1.6 radar
    wave added `gpuwm.obs.nexrad` and `gpuwm.obs.frontdoor`, both of
    which freeze a `tools/rustwx` hint at import.  Neither was re-derived
    when a test forced the other platform's shell, so on Windows the
    frozen `;` spelling passed both directions and on the ubuntu publish
    runner the frozen `&&` spelling was measured against "PowerShell
    cannot parse '&&'" -- eight reds in the first CI job, after the tag.
    Walking the package is what makes a sixth module impossible to miss.
    """

    import importlib
    import pkgutil

    import gpuwm

    found = []
    for info in pkgutil.walk_packages(gpuwm.__path__, prefix="gpuwm."):
        try:
            module = importlib.import_module(info.name)
        except Exception:      # optional deps (cupy, wrf-rust) may be absent
            continue
        for name, value in vars(module).items():
            if (name.isupper() and isinstance(value, str)
                    and "cargo build " in value):
                found.append((info.name, name, value))
    return found


def _force_shell(monkeypatch, windows):
    """Put the remedy generators on ``windows``'s shell, completely.

    ``bridges.WINDOWS_SHELL`` alone is not the whole dimension.  Three
    modules compute a build hint from it *at import*, so a test that
    flips the flag and then reaches one of those constants is judging a
    remedy generated for the host against the rules of the other
    platform.  On a Windows box that mistake is invisible -- the frozen
    hint already spells `;` -- and on the ubuntu publish runner it
    failed the cut: `cd tools/grib1_bridge && cargo build ... && cd
    ../..` measured against "PowerShell cannot parse '&&'".

    One helper, so a forcing site cannot do half the job.
    """

    import importlib

    from gpuwm import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)

    # Re-derive EVERY frozen hint the package actually carries, found by
    # walking it, so a module added later is forced too rather than being
    # judged in the host's spelling against the other platform's rules.
    # Naming the modules here is what let 1.6's two new ones through.
    for module_name, name, value in _frozen_shell_constants():
        crate = (bridges.RUSTWX_CRATE_RELATIVE if "rustwx" in value
                 else bridges.CRATE_RELATIVE)
        monkeypatch.setattr(importlib.import_module(module_name), name,
                            bridges.cargo_build_one_liner(crate))


def _force_bare_estate(monkeypatch):
    """Nothing Rust is built yet: a fresh checkout, or any CI runner.

    The estate is a dimension too, and on a developer box it has only
    ever had one value -- everything built.  `collect_checks()` then
    answers those checks `ok` with no remedy, so a sweep over "every
    remedy doctor can print" sweeps almost nothing, and the shell
    forcing added earlier cannot help: there is no remedy to spell.

    That is how a whitelist missing `cd` survived every local run and
    failed the first bare Linux gate on the first line of the checkout
    build remedy.  Forcing the estate is the missing half.
    """

    from gpuwm import rustwx, rustwx_fetch
    from gpuwm.ingest import cpu_backend

    def _no_library():
        raise FileNotFoundError("gpuwm_preprocess_cpu not found")

    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    monkeypatch.setattr(rustwx, "find_renderer", lambda: None)
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    monkeypatch.setattr(cpu_backend, "CpuPreprocessBackend", _no_library)


@pytest.mark.parametrize("windows", (False, True))
def test_the_shell_forcing_helper_re_derives_every_frozen_hint(
        monkeypatch, windows):
    """Forcing the shell must move EVERY constant built from it.

    The failure this pins is not hypothetical: it is the one that ended
    three release cuts' worth of CI on `&&`.  Both arms run here, so
    whichever platform the suite runs on, the other one is exercised.
    """

    before = _frozen_shell_constants()
    assert before, (
        "the search found no build hints at all; it has stopped looking "
        "where they live, and this guard is now vacuous")

    _force_shell(monkeypatch, windows)

    separator = ";" if windows else "&&"
    stale = "&&" if windows else ";"
    for module_name, name, value in _frozen_shell_constants():
        assert separator in value, (
            f"{module_name}.{name} was not re-derived for "
            f"{'windows' if windows else 'posix'}: {value!r}")
        assert stale not in value, (
            f"{module_name}.{name} still carries the other shell's "
            f"separator: {value!r}")


def _assert_remedy_lines_are_commands_or_comments(remedy, *, windows):
    """EVERY line: a command that runs as printed, or a `#` comment.

    No exemptions -- not even the first line.  The closing report
    claims this of every remedy, so a headline sentence must be a
    `#` comment, never bare prose fused into a paste.

    The shell rules are per-platform and that is deliberate: `&&` is
    correct in sh (and fails fast, which `;` does not) and is a parse
    error in Windows PowerShell 5.1, so one universal separator would
    have to be wrong somewhere.  What the contract requires is that a
    remedy generated FOR a platform obeys THAT platform's rules -- in
    both directions, which is the part that was missing.
    """

    lines = remedy.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        assert "<" not in stripped and ">" not in stripped, (
            f"placeholder in remedy line {index}: {line!r}")
        if stripped.startswith("#"):
            continue
        first = stripped.split()[0]
        assert first in _REMEDY_COMMANDS, (
            f"remedy line {index} is neither a command nor a '#' comment "
            f"(first token {first!r}): {line!r}")
        for token in stripped.split():
            assert not _PLACEHOLDER_WORD.match(token.rstrip(",.")), (
                f"unexpanded placeholder {token!r} in: {line!r}")
        if windows:
            assert "&&" not in stripped, (
                f"Windows PowerShell 5.1 cannot parse '&&': {line!r}")
        else:
            for spelling in _POWERSHELL_ONLY:
                assert spelling not in stripped, (
                    f"a POSIX shell has no {spelling!r}: {line!r}")


@pytest.mark.parametrize("windows", (False, True))
@pytest.mark.parametrize("cargo", (False, True))
def test_the_emitted_bootstrap_is_shell_correct_on_both_platforms(
        monkeypatch, tmp_path, windows, cargo):
    """The real assertion: parse what doctor would actually print."""

    from gpuwm import bridges

    _force_shell(monkeypatch, windows)
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: cargo)
    # A pip install: no crate anywhere.
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: tmp_path / "tools" / "grib1_bridge")

    remedy = bridges.bridge_remedy("grib1_bridge")
    _assert_remedy_lines_are_commands_or_comments(remedy, windows=windows)

    lines = [line.strip() for line in bridges.build_from_clone_hint()]
    commands = [line for line in lines if not line.startswith("#")]

    # The clone and the build are always there, in that order, as
    # separate lines -- never joined by an operator one shell lacks.
    assert any(line.startswith("git clone ") for line in commands)
    assert any(line.startswith("cd ") for line in commands)
    # The build is the last thing that DOES anything, and exactly one
    # command follows it: the walk back out of the crate, so the next
    # remedy block starts where this one started.  (This pinned
    # `commands[-1]` as the build, which could not tell "nothing
    # follows" from "the return was dropped".)
    build = commands.index(next(line for line in commands
                                if line.startswith("cargo build ")))
    assert build == len(commands) - 2, commands
    assert commands[-1] == "cd " + bridges._parent_hops(
        bridges.CLONE_DIR, bridges.CRATE_RELATIVE), commands

    if cargo:
        # Rust is already here; installing it again would be noise.
        assert not any("rustup" in line for line in lines)
        return

    # Rust is absent: install it, THEN make it usable in this shell,
    # and only then call cargo.  A bootstrap that installs cargo and
    # immediately fails to find it is the whole complaint.
    activation = bridges.cargo_activation_command()
    assert activation in commands, commands
    first_cargo = next(i for i, line in enumerate(commands)
                       if line.startswith("cargo "))
    assert commands.index(activation) < first_cargo, (
        f"cargo is called before it is on PATH: {commands}")
    if windows:
        assert activation.startswith("$env:Path")
    else:
        assert activation == '. "$HOME/.cargo/env"'


@pytest.mark.parametrize("windows", (False, True))
def test_the_invalid_override_branches_emit_install_aware_remedies(
        monkeypatch, tmp_path, windows):
    """doctor's `fix ENV ... or unset it and <remedy>` paths.

    Both called the old install-unaware helpers, which kept `<clone>`
    placeholders and told a pip-only install to enter a relative
    `tools/rustwx` it does not have.  Both shells, because these two
    remedies are assembled by the same generator whose separator is
    per-platform.
    """

    from gpuwm import bridges, rustwx, rustwx_fetch

    _force_shell(monkeypatch, windows)
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: tmp_path / "tools" / "grib1_bridge")
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: True)

    for remedy in (rustwx.renderer_remedy(), rustwx_fetch.fetch_remedy()):
        assert "<clone>" not in remedy
        assert bridges.REPOSITORY_URL in remedy, (
            "a pip install has no tools/rustwx to cd into; the remedy "
            "must start from the clone")
        _assert_remedy_lines_are_commands_or_comments(
            remedy, windows=windows)


@pytest.mark.parametrize("estate", ("live", "bare"))
@pytest.mark.parametrize("windows", (False, True))
def test_every_remedy_doctor_can_print_obeys_the_closing_claim(
        monkeypatch, windows, estate):
    """The closing line makes a claim about EVERY remedy.  Check them all.

    Including the non-Rust ones: the pip-extra and fetch-geog hints used
    to trail a parenthetical on the command line itself.

    And on both shells, not just the one this box has.  Reading the real
    ``WINDOWS_SHELL`` meant the sweep over every remedy doctor can
    actually assemble here ran against exactly one platform's spelling
    -- the author's -- so the other platform's version of these same
    blocks was covered by nothing until a release cut ran it.
    """

    from gpuwm import doctor

    _force_shell(monkeypatch, windows)
    if estate == "bare":
        _force_bare_estate(monkeypatch)

    for check in doctor.collect_checks():
        if not check.remedy:
            continue
        _assert_remedy_lines_are_commands_or_comments(
            check.remedy, windows=windows)

    for hint in (doctor.GPU_EXTRA_HINT, doctor.RENDER_EXTRA_HINT,
                 doctor.GEOG_HINT, doctor.REINSTALL_HINT):
        _assert_remedy_lines_are_commands_or_comments(
            hint, windows=windows)
        assert hint.splitlines()[0].split()[0] in _REMEDY_COMMANDS


def test_the_closing_line_states_what_it_can_prove():
    """No "copy-pasteable", and no unqualified "runs as printed"."""

    from gpuwm.doctor import Check, format_report

    text = format_report([
        Check("bridge grib1_bridge", "missing", "not built",
              "build it:\n  cargo build --release --locked --offline")])
    assert "copy-pasteable" not in text
    # The claim now names the two line kinds it actually guarantees.
    assert "either a command" in text and "'#' comment" in text


@pytest.mark.parametrize("windows", (False, True))
@pytest.mark.parametrize("staging", (False, True))
def test_every_conditional_remedy_branch_obeys_the_claim(
        monkeypatch, tmp_path, windows, staging):
    """Force the branches collect_checks() cannot reach on a dev box.

    On a machine with the checkout present and the binaries built,
    collect_checks() never assembles the consumer-note, rebuild or
    invalid-override remedies -- which is exactly how `[needed by:]`
    trailers and `fix ENV ...` headlines survived the first structural
    test.  Each branch assembles its remedy differently; force every
    one and hold it to the same contract.
    """

    from gpuwm import bridges, doctor, rustwx, rustwx_fetch

    checkout = tmp_path / "checkout"
    (checkout / "tools" / "grib1_bridge").mkdir(parents=True)
    (checkout / "tools" / "grib1_bridge" / "Cargo.toml").write_text(
        "[package]\n", encoding="utf-8")
    (checkout / "tools" / "rustwx").mkdir(parents=True)
    (checkout / "tools" / "rustwx" / "Cargo.toml").write_text(
        "[package]\n", encoding="utf-8")
    pip_shape = tmp_path / "pipshape"
    pip_shape.mkdir()

    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("ENV_VAR points at nothing on disk")

    _force_shell(monkeypatch, windows)
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: False)
    _set_staging(monkeypatch, tmp_path, available=staging)

    remedies = []
    for shape in (checkout, pip_shape):
        monkeypatch.setattr(bridges, "_package_parent", lambda s=shape: s)
        monkeypatch.setattr(
            bridges, "crate_dir",
            lambda s=shape: s / "tools" / "grib1_bridge")
        monkeypatch.setattr(
            rustwx, "crate_dir", lambda s=shape: s / "tools" / "rustwx")
        monkeypatch.setattr(
            rustwx_fetch, "crate_dir",
            lambda s=shape: s / "tools" / "rustwx")

        # Branch: the override env var names a missing executable.
        monkeypatch.setattr(bridges, "find_bridge", _missing)
        remedies += [c.remedy for c in doctor._bridge_checks() if c.remedy]
        monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", _missing)
        remedies.append(doctor._fetch_backbone_check().remedy)
        monkeypatch.setattr(rustwx, "find_renderer", _missing)
        remedies.append(doctor._rust_renderer_check().remedy)

        # Branch: found on disk but fails its ABI probe -> rebuild.
        monkeypatch.setattr(
            bridges, "find_bridge", lambda name: tmp_path / "stale.exe")
        monkeypatch.setattr(
            doctor, "_exec_probe", lambda path: (False, "stale ABI"))
        remedies += [c.remedy for c in doctor._bridge_checks() if c.remedy]
        monkeypatch.setattr(
            rustwx_fetch, "find_fetch_bin",
            lambda: tmp_path / "stale.exe")
        monkeypatch.setattr(
            rustwx_fetch, "probe_fetch_bin",
            lambda path: (False, "stale ABI"))
        remedies.append(doctor._fetch_backbone_check().remedy)
        monkeypatch.setattr(
            rustwx, "find_renderer", lambda: tmp_path / "stale.exe")
        monkeypatch.setattr(
            rustwx, "probe_renderer", lambda path: (False, "stale ABI"))
        remedies.append(doctor._rust_renderer_check().remedy)

        # Branch: nothing built at all (crate present or pip shape).
        monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
        remedies += [c.remedy for c in doctor._bridge_checks() if c.remedy]
        monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
        remedies.append(doctor._fetch_backbone_check().remedy)
        monkeypatch.setattr(rustwx, "find_renderer", lambda: None)
        remedies.append(doctor._rust_renderer_check().remedy)

    assert len([r for r in remedies if r]) >= 16, (
        "branch coverage collapsed; a doctor branch stopped emitting "
        f"its remedy (got {len(remedies)})")
    for remedy in remedies:
        assert remedy
        _assert_remedy_lines_are_commands_or_comments(
            remedy, windows=windows)


# ---------------------------------------------------------------------------
# The blocks have to COMPOSE, not merely each be well formed
# ---------------------------------------------------------------------------
# doctor's closing line promises a paste: every remedy above, in the
# order printed, run as one sequence.  Line-by-line well-formedness does
# not deliver that, and it is not what the field report failed on.  What
# only the composition can see: a block that ends two directories inside
# a crate, silently re-pointing every relative path in the block after
# it; a `git clone` printed once per gap, which fails from the second
# one on; and a build hint frozen at import for the OTHER platform,
# which a first-token whitelist waves through.

#: An absolute `cd` would anchor a block rather than return it, which
#: this model cannot follow -- it tracks position relative to where the
#: reader started.  Fail closed and say so rather than pass blind.
_ABSOLUTE_PATH = re.compile(r"^(~|/|[A-Za-z]:[\\/])")


def _shell_steps(line, *, windows):
    """The commands one physical remedy line runs, in order.

    Only `cd` and `git clone` are acted on, so the crude split is
    enough: a `;` inside a quoted string (the PowerShell PATH
    activation has one) yields fragments that are neither, and a shell
    parser here would be a second implementation to keep honest.
    """

    separator = ";" if windows else "&&"
    return [step.strip() for step in line.split(separator) if step.strip()]


class _PastedShell:
    """Where the paste has left the working directory, so far.

    Relative to the directory the reader started in: `cd a/b` is two
    levels down, `cd ..` is one back up, and nothing may climb above the
    start -- the reader's own directory is not doctor's to leave.
    """

    def __init__(self):
        self.at = []

    def run(self, step):
        if step.startswith("cd "):
            self._cd(step[len("cd "):])

    def _cd(self, argument):
        argument = argument.strip().strip('"').strip("'")
        assert not _ABSOLUTE_PATH.match(argument), (
            f"absolute cd in a remedy: {argument!r}.  Anchoring that way "
            "is defensible, but this model follows relative position; "
            "teach it the anchor before switching.")
        for part in re.split(r"[\\/]+", argument):
            if part in ("", "."):
                continue
            if part == "..":
                assert self.at, (
                    "a remedy walks above the directory the reader "
                    f"started in: cd {argument}")
                self.at.pop()
            else:
                self.at.append(part)


def _assert_the_paste_composes(remedies, *, windows):
    """Run the whole printed sequence, in order, in a paper shell."""

    shell = _PastedShell()
    clones = []
    for index, remedy in enumerate(remedies):
        notes = []
        for line in remedy.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                notes.append(stripped.lower())
                continue
            for step in _shell_steps(stripped, windows=windows):
                if step.startswith("git clone"):
                    assert not shell.at, (
                        f"remedy {index} clones into {'/'.join(shell.at)} "
                        "rather than the directory the reader started in, "
                        "so a second block's clone lands somewhere else")
                    assert any("skip" in note and "exist" in note
                               for note in notes), (
                        f"remedy {index} pastes {step!r} with nothing "
                        "saying to skip it when the directory is already "
                        "there -- doctor prints this same clone once per "
                        "gap, and the second one only errors")
                    clones.append(step)
                shell.run(step)
        assert not shell.at, (
            f"remedy block {index} leaves the shell in "
            f"{'/'.join(shell.at)}; every relative path in the block "
            "after it then means something else")
    return clones


def _pin_a_bundle(tmp_path, platform="linux-x86_64"):
    """A pins document whose numbers come from bytes on this disk.

    doctor only reads the platform and the download size out of it, but
    a hash typed by hand is a hash nobody computed, so this hashes an
    archive it just wrote.
    """

    import hashlib
    import zipfile

    from gpuwm import bridge_assets

    archive = tmp_path / f"gpuwm-bridges-v0-doctor-{platform}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            name = bridge_assets.artifact_filename(artifact, platform)
            zf.writestr(name, name.encode() * 32)
    blob = archive.read_bytes()
    binaries = []
    with zipfile.ZipFile(archive) as zf:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            name = bridge_assets.artifact_filename(artifact, platform)
            payload = zf.read(name)
            binaries.append(bridge_assets.BinaryPin(
                artifact.name, name, len(payload),
                hashlib.sha256(payload).hexdigest()))
    bundle = bridge_assets.BundlePin(
        platform=platform, filename=archive.name, bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(), binaries=tuple(binaries))
    return bridge_assets.BridgePins(release="v0-doctor",
                                    platforms={platform: bundle})


def _set_staging(monkeypatch, tmp_path, *, available):
    """Make ``bridges.prebuilt_bundle_offer`` reachable, or not.

    Both are real states of a real install: a release that published a
    bundle for this platform, and one that did not (every tree before
    the first bundled release, plus every platform that has no bundle).
    doctor's report has to be honest and pasteable in both.
    """

    from gpuwm import bridge_assets

    if not available:
        monkeypatch.setattr(bridge_assets, "host_platform", lambda: None)
        return None
    pins = _pin_a_bundle(tmp_path)
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: "linux-x86_64")
    return pins


class _ArrangedSys:
    """Stands in for `sys`, overriding ONLY what an arrangement changes.

    The predecessor was a bare SimpleNamespace carrying two attributes,
    and the third one some code path eventually reached did not exist.
    That cost a release: `_driver_library_names` reads `sys.platform`,
    reaching it depends on GPUWM_NO_LOCAL_GPU -- which the battery sets
    and CI does not -- so the double was complete enough locally and a
    hole on the runner, and the two layers of the test net disagreed for
    the first time.

    A double that enumerates attributes can always be one short of the
    code it stands in for.  This one cannot: anything not deliberately
    arranged DELEGATES to the real module, so the failure mode is a
    correct answer rather than an AttributeError, and adding a
    `sys.<anything>` read to doctor can never again break only on the
    machine that happens to reach it.
    """

    def __init__(self, **arranged):
        self.__dict__.update(arranged)

    def __getattr__(self, name):
        import sys as real_sys
        return getattr(real_sys, name)


def test_the_sys_double_answers_every_attribute_doctor_reads():
    """The double cannot be one attribute short of the module it fakes.

    This is the regression pin for the v1.8.1 CI failure.  The old
    double was a two-attribute SimpleNamespace; doctor grew a
    `sys.platform` read; 24 parametrizations died on the runner while
    the battery stayed green, because whether that read is REACHED
    depends on GPUWM_NO_LOCAL_GPU and the battery sets it.

    Rather than re-listing attributes here (a list is what failed), this
    scans doctor for every `sys.<name>` it actually reads and asks the
    double for each one.  A new read added tomorrow is covered the day
    it lands, with no list to remember to update.
    """
    import ast
    import sys as real_sys

    source = (Path(__file__).parents[1] / "gpuwm"
              / "doctor.py").read_text(encoding="utf-8")
    names = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    }
    assert names, "the scan found no sys reads at all; it has stopped working"

    double = _ArrangedSys(version_info=(3, 10, 4), platform="linux")
    for name in sorted(names):
        getattr(double, name)          # must not raise

    # Arranged values win; everything else is the truth from the module.
    assert double.platform == "linux"
    assert double.version_info == (3, 10, 4)
    assert double.maxsize == real_sys.maxsize


def _force_every_gap(monkeypatch, tmp_path, *, windows, shape, mode,
                     staging=False):
    """Make collect_checks() reach a remedy on (nearly) every check.

    The idiom of the branch-forcing test above, widened to the whole
    report: on the box running these tests most of these checks return
    `verified` with no remedy at all, which is exactly how a fused prose
    tail lived in the CPU-library branch underneath two structural
    tests that both claimed to cover every remedy.
    """

    import json as jsonlib
    import sys as real_sys
    import types

    from gpuwm import bridges, rustwx, rustwx_fetch, table_assets
    from gpuwm.core import noah as noah_core
    from gpuwm.core import thompson_contract
    from gpuwm.ingest import cpu_backend

    _set_staging(monkeypatch, tmp_path, available=staging)

    root = tmp_path / shape
    root.mkdir(parents=True, exist_ok=True)
    if shape == "checkout":
        for crate in ("grib1_bridge", "rustwx"):
            (root / "tools" / crate).mkdir(parents=True, exist_ok=True)
            (root / "tools" / crate / "Cargo.toml").write_text(
                "[package]\n", encoding="utf-8")

    # The shell under test -- flag and the three hints frozen from it at
    # import, which is one indivisible move (_force_shell).
    _force_shell(monkeypatch, windows)
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: False)

    monkeypatch.setattr(bridges, "_package_parent", lambda: root)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: root / "tools" / "grib1_bridge")
    monkeypatch.setattr(rustwx, "crate_dir", lambda: root / "tools" / "rustwx")
    monkeypatch.setattr(rustwx_fetch, "crate_dir",
                        lambda: root / "tools" / "rustwx")

    # python below the floor; cupy and the render extra absent (with
    # find_spec gone, the import probes spawn nothing).
    #
    # ``platform`` is NOT optional here, and leaving it out cost a
    # release.  This namespace stands in for the whole `sys` module for
    # every line of doctor that reads it, so an attribute the real
    # module has and this one does not is a hole that opens the moment
    # some code path reaches for it.  `_driver_library_names` reads
    # `sys.platform`, and whether it is reached depends on
    # GPUWM_NO_LOCAL_GPU -- which the battery sets and CI does not, so
    # the hole was invisible locally and fatal on the runner.  It is set
    # from `windows` so the fake agrees with the shell `_force_shell`
    # just installed, rather than being a third opinion about the
    # platform.
    monkeypatch.setattr(doctor, "sys", _ArrangedSys(
        version_info=(3, 10, 4), executable=real_sys.executable,
        platform=("win32" if windows else "linux")))
    monkeypatch.setattr(doctor, "find_spec", lambda name: None)

    def _raise_missing(*_args, **_kwargs):
        raise FileNotFoundError("the override names nothing on disk")

    stale = tmp_path / "stale.exe"
    if mode == "override":
        monkeypatch.setattr(bridges, "find_bridge", _raise_missing)
        monkeypatch.setattr(rustwx, "find_renderer", _raise_missing)
        monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", _raise_missing)
    elif mode == "stale":
        monkeypatch.setattr(bridges, "find_bridge", lambda name: stale)
        monkeypatch.setattr(doctor, "_exec_probe",
                            lambda path: (False, "stale ABI"))
        monkeypatch.setattr(rustwx, "find_renderer", lambda: stale)
        monkeypatch.setattr(rustwx, "probe_renderer",
                            lambda path: (False, "stale ABI"))
        monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: stale)
        monkeypatch.setattr(rustwx_fetch, "probe_fetch_bin",
                            lambda path: (False, "stale ABI"))
    else:
        monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
        monkeypatch.setattr(rustwx, "find_renderer", lambda: None)
        monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)

    def _no_library():
        raise FileNotFoundError("gpuwm_preprocess_cpu not found")

    def _unloadable():
        raise OSError("found, but not a loadable image")

    monkeypatch.setattr(cpu_backend, "CpuPreprocessBackend",
                        _unloadable if mode == "stale" else _no_library)

    def _invalid(*_args, **_kwargs):
        raise FileNotFoundError("qr_acr_qg_V4.dat is not where it should be")

    # Thompson: the externalized-asset gap (a fetch), or everything else
    # (a reinstall).  Both remedies are only reachable from a broken
    # table root, which the box running the tests does not have.
    monkeypatch.setattr(thompson_contract, "validate_table_assets", _invalid)
    externalized = [types.SimpleNamespace(filename="freezeH2O.dat",
                                          bytes=92 * 1024 * 1024)]
    if mode == "absent":
        monkeypatch.setattr(table_assets, "missing_externalized_assets",
                            lambda where: list(externalized))
        monkeypatch.setattr(table_assets, "classify_assets",
                            lambda where: ([], [], list(externalized)))
    else:
        monkeypatch.setattr(table_assets, "missing_externalized_assets",
                            lambda where: [])
        monkeypatch.setattr(table_assets, "classify_assets",
                            lambda where: ([], list(externalized), []))

    monkeypatch.setattr(noah_core, "load_tables", _invalid)

    # The case-data root: set to a directory that is not there, or set
    # to one that is but carries no WPS_GEOG.
    monkeypatch.setenv(
        "GPUWM_CASE_DATA_ROOT",
        str(tmp_path / "absent" if mode == "override" else root))

    # The sealed-runtime manifest: not a READY document, or READY with
    # an artifact that no longer hashes.
    manifest = tmp_path / f"manifest-{mode}.json"
    if mode == "absent":
        manifest.write_text("not a json document at all", encoding="utf-8")
    else:
        manifest.write_text(jsonlib.dumps({
            "schema": "gpuwm-native-wrf-runtime-v1", "status": "READY",
            "payload": {"decoder.bin": {"bytes": 3, "sha256": "00" * 32}},
        }), encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))


@pytest.mark.parametrize("windows", (False, True))
@pytest.mark.parametrize("shape", ("checkout", "wheel"))
@pytest.mark.parametrize("mode", ("absent", "stale", "override"))
@pytest.mark.parametrize("staging", (False, True))
@pytest.mark.parametrize("driver", (None, 13))
def test_the_whole_printed_report_pastes_as_one_sequence(
        monkeypatch, tmp_path, windows, shape, mode, staging, driver):
    """Every remedy doctor can print, in print order, pasted as a block.

    The contract is not "each block is well formed"; it is "select the
    lot and run it".  Four ways that was false shipped past a per-block
    sweep: `cd tools/rustwx` left the shell in the crate so the next
    block's `cd` went somewhere else, `git clone` came back once per
    bridge, the CPU-library remedy fused prose onto the build command,
    and four remedies were prose with no `#` at all.

    ``staging`` is the fifth: when the release published a bundle for
    this platform, every Rust remedy on a wheel install leads with
    `gpuwm fetch-bridges` and demotes the clone-and-build to comments.
    A paste that ran BOTH would clone 2.5 GB and compile for two
    minutes to obtain files the line above already staged, so the
    composition has to be checked in that arrangement too.
    """

    # THE DRIVER ARRANGEMENT, PINNED RATHER THAN INHERITED.  Both
    # remedies this can print are real: with a major in hand doctor
    # names the ONE matching extra, without one it prints both and
    # defaults to neither.  Until this parameter existed the arrangement
    # was whatever the HOST happened to be -- a driver on Drew's box, no
    # driver on the runner -- so the battery and CI were testing
    # different code and could disagree, which they did.  Both now run
    # everywhere, and the suppression env var cannot decide the outcome.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: driver)

    _force_every_gap(monkeypatch, tmp_path, windows=windows, shape=shape,
                     mode=mode, staging=staging)
    checks = doctor.collect_checks()
    remedies = [check.remedy for check in checks if check.remedy]
    assert len(remedies) >= 12, (
        "the sweep stopped reaching most of the report; it can only "
        f"prove what it forces (got {len(remedies)})")

    # 1. Every physical line of the whole paste, first to last.
    _assert_remedy_lines_are_commands_or_comments(
        "\n".join(remedies), windows=windows)

    # 2. The sequence composes: no block depends on -- or damages -- the
    #    working directory another block left behind.
    clones = _assert_the_paste_composes(remedies, windows=windows)

    # 3. And it survives being pasted twice, which is what happens when
    #    a reader re-runs doctor and pastes the report again.
    _assert_the_paste_composes(remedies * 2, windows=windows)

    if shape == "wheel" and not staging:
        assert len(clones) >= 2, (
            "a wheel gaps every Rust artifact at once; the clone should "
            f"appear in several blocks: {clones}")
        assert len(set(clones)) == 1, (
            f"the blocks clone into different directories: {set(clones)}")
    elif shape == "wheel":
        # The same gaps, one command instead of a clone -- and the build
        # route still printed, because it is the only route on a
        # platform with no bundle.  Commented, so the paste does not run
        # both.
        assert not clones, (
            "with a bundle published, a paste must not also clone and "
            f"compile: {clones}")
        rust_blocks = [remedy for remedy in remedies
                       if "cargo build" in remedy]
        assert len(rust_blocks) >= 6, (
            f"the sweep stopped reaching the Rust remedies: {len(rust_blocks)}")
        for remedy in rust_blocks:
            commands = [line.strip() for line in remedy.splitlines()
                        if line.strip() and not line.strip().startswith("#")]
            assert commands and commands[0] == "gpuwm fetch-bridges", (
                f"the prebuilt bundle must be offered first: {commands[:2]}")
            assert all(command == "gpuwm fetch-bridges"
                       for command in commands), (
                "the source build must be commented out when the bundle "
                f"is offered: {commands}")
    else:
        assert not clones, "a checkout has the sources; cloning is noise"
        assert any("cargo build --release --locked --offline" in remedy
                   for remedy in remedies)

    # 4. The printed report IS that paste: the whole block lines up
    #    under `remedy:`, rather than only its first line.
    report_lines = doctor.format_report(checks).splitlines()
    cursor = 0
    for check in checks:
        if not check.remedy:
            continue
        printed = [line.strip() for line in check.remedy.splitlines()]
        label = "          remedy: " + printed[0]
        assert label in report_lines[cursor:], label
        cursor = report_lines.index(label, cursor)
        for offset, text in enumerate(printed[1:], start=1):
            line = report_lines[cursor + offset]
            assert line[:18].strip() == "" and line[18:] == text, (
                "remedy continuation lines must line up under the label: "
                f"{line!r}")
        cursor += len(printed)


@pytest.mark.parametrize("windows", (False, True))
@pytest.mark.parametrize("staging", (False, True))
def test_the_pip_remedy_offers_the_prebuilt_bundle_first(
        monkeypatch, tmp_path, staging, windows):
    """One gap, two true remedies, decided by what the release published.

    With a bundle for this platform the first thing a wheel user is
    handed is the one command that finishes the job; without one they
    get the clone-and-build bootstrap they have always got.  Neither
    arm may leave the other's route out: the build is the only route on
    a platform with no bundle, so it stays printed -- as comments -- and
    a reader who pastes the report runs exactly one of them.
    """

    from gpuwm import bridges

    _force_shell(monkeypatch, windows)
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path / "wheel")
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: tmp_path / "wheel" / "tools" / "grib1_bridge")
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: True)
    pins = _set_staging(monkeypatch, tmp_path, available=staging)

    remedy = bridges.bridge_remedy("grib1_bridge")
    _assert_remedy_lines_are_commands_or_comments(remedy, windows=windows)
    commands = [line.strip() for line in remedy.splitlines()
                if line.strip() and not line.strip().startswith("#")]

    if not staging:
        assert "gpuwm fetch-bridges" not in remedy, (
            "a platform with no published bundle must not be offered a "
            "command that would refuse")
        assert any(line.startswith("git clone") for line in commands)
        assert any(line.startswith("cargo build") for line in commands)
        return

    assert commands == ["gpuwm fetch-bridges"], commands
    assert "git clone" in remedy and "cargo build" in remedy, (
        "the source route must still be printed; it is the only one on "
        "a platform with no bundle")
    bundle = pins.bundle_for("linux-x86_64")
    assert f"{bundle.bytes / (1024 * 1024):.0f} MiB" in remedy, (
        "the offer should say how large the download is")
    assert str(bridges.default_bridge_dir()) in remedy
    # And the one-line composition, for callers that cannot print a block.
    one_line = bridges.install_aware_one_line_hint(bridges.CARGO_BUILD_HINT)
    assert "gpuwm fetch-bridges" in one_line


def test_the_offer_is_absent_when_the_pins_document_is_unreadable(
        monkeypatch, tmp_path):
    """A broken pins file is not an offer, and not a traceback either.

    doctor runs on machines whose install is already wrong; a remedy
    composer that raised while explaining a gap would take the whole
    report down with it.
    """

    from gpuwm import bridge_assets, bridges

    broken = tmp_path / "bridge-pins.json"
    broken.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(bridge_assets, "packaged_pins_path", lambda: broken)
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: "linux-x86_64")
    assert bridges.prebuilt_bundle_offer() is None
    assert bridge_assets.staging_available() is False


def test_doctor_finds_basemaps_the_way_the_renderer_does(tmp_path,
                                                         monkeypatch):
    """A renderer built from a clone resolves assets from its own tree.

    This is the V-14 field finding: gpuwm doctor probed only its own
    checkout path and announced "NO basemap assets found" on installs
    where rw_wrfbatch was drawing coastlines from the build directory
    beside it.
    """

    from gpuwm import rustwx

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
    # Somewhere with no assets above it, so the cwd walk cannot rescue
    # a probe that fails to look beside the executable.
    working = tmp_path / "elsewhere"
    working.mkdir()
    monkeypatch.chdir(working)

    clone = tmp_path / "clone" / "tools" / "rustwx"
    renderer = clone / "target" / "release" / "rw_wrfbatch"
    renderer.parent.mkdir(parents=True)
    renderer.write_bytes(b"")
    assets = clone / "assets" / "basemap"
    assets.mkdir(parents=True)

    # The renderer walks up from its own directory: target/release ->
    # target -> tools/rustwx, where the assets are.
    assert rustwx.resolve_basemap_dir(renderer) == assets
    # And only by walking up from it: nothing else in this environment
    # can reach that build's assets, which is exactly the situation a
    # pip user is in.
    assert rustwx.resolve_basemap_dir(None) != assets
    assert assets not in rustwx.basemap_candidates(None)


def test_the_basemap_environment_overrides_win_in_the_renderers_order(
        tmp_path, monkeypatch):
    from gpuwm import rustwx

    monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
    override = tmp_path / "override"
    override.mkdir()
    clone = tmp_path / "clone" / "tools" / "rustwx"
    renderer = clone / "target" / "release" / "rw_wrfbatch"
    renderer.parent.mkdir(parents=True)
    renderer.write_bytes(b"")
    (clone / "assets" / "basemap").mkdir(parents=True)

    monkeypatch.setenv("RUSTWX_BASEMAP_DIR", str(override))
    assert rustwx.resolve_basemap_dir(renderer) == override
    assert rustwx.basemap_candidates(renderer)[0] == override

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR")
    monkeypatch.setenv("RUSTWX_ASSETS_DIR", str(tmp_path / "assets_root"))
    assert rustwx.basemap_candidates(renderer)[0] == (
        tmp_path / "assets_root" / "basemap")


def test_the_checkout_path_is_still_the_last_candidate():
    """The vendored crate's own assets remain reachable, just not alone."""

    from gpuwm import rustwx

    assert rustwx.basemap_candidates(None)[-1] == rustwx.basemap_dir()


def test_a_real_doctor_gap_prints_a_comment_only_remedy(monkeypatch,
                                                        tmp_path):
    """The honest case the README now admits, proved from doctor.

    An unset-then-wrong `GPUWM_CASE_DATA_ROOT` is a gap whose fix is a
    path only the user knows, so doctor prints a `#`-comment remedy with
    no runnable command in it. That is exactly the case that makes
    "every gap prints the exact command" a lie -- and the case the
    softened README covers with "a few cannot be".
    """

    missing = tmp_path / "not-a-real-root"
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(missing))
    gap = {check.name: check
           for check in doctor._case_data_root_check()}["GPUWM_CASE_DATA_ROOT"]
    assert gap.status == "missing"
    assert gap.remedy
    remedy_lines = [line.strip() for line in gap.remedy.splitlines()
                    if line.strip()]
    # Every line is a comment: there is genuinely no command to print.
    assert remedy_lines and all(line.startswith("#") for line in remedy_lines), \
        gap.remedy


@pytest.mark.parametrize("estate", ("live", "bare"))
def test_every_remedy_line_doctor_can_emit_is_a_command_or_a_comment(
        monkeypatch, estate):
    """Doctor's own closing contract, over the estate it can reach.

    Not "every remedy is a command" -- some are `#`-comment-only, some
    are multi-step. The invariant the README must state, and does, is
    that every printed remedy LINE is a command or a `#` comment, so the
    block survives being pasted whole.

    Two estates, because one of them was doing all the work and it was
    the empty one.  "live" is whatever this machine happens to have;
    "bare" is a checkout with nothing built, which is every CI runner
    and every new user, and is the only estate that makes doctor emit
    the checkout build remedy this test now covers.
    """

    if estate == "bare":
        _force_bare_estate(monkeypatch)

    swept = 0
    for check in doctor.collect_checks():
        if not check.remedy:
            continue
        for raw in check.remedy.splitlines():
            line = raw.strip()
            if not line:
                continue
            swept += 1
            assert line.startswith("#") or _looks_like_command(line), (
                check.name, line)
    if estate == "bare":
        # A sweep that swept nothing passes vacuously, which is how the
        # missing `cd` lived here for as long as it did.
        assert swept >= 20, (
            f"the bare estate produced only {swept} remedy lines; the "
            "forcing has stopped reaching doctor's gaps")


#: Spellings this sweep accepts on top of :data:`_REMEDY_COMMANDS`.
#: The README scan shares this predicate and quotes shell lines that
#: doctor itself never prints.
_EXTRA_COMMAND_TOKENS = frozenset({
    "set", "rustup", "conda", "uv", "rw-wps",
    "$env:GPUWM_GRIB1_BRIDGE",
})


def _looks_like_command(line: str) -> bool:
    """A command line starts a shell invocation, or continues one.

    The whitelist is :data:`_REMEDY_COMMANDS` -- the one the structural
    contract uses -- plus the spellings above.  It used to be a second,
    hand-written set that claimed to mirror the contract and did not:
    it was missing `cd`, and `cd <crate> && cargo build ... && cd ../..`
    is the FIRST line of the checkout remedy for every Rust artifact.

    That gap survived because it is only reachable on a machine where
    something is unbuilt.  Every developer box here has all eight
    artifacts, so `collect_checks()` returns those checks `ok` with no
    remedy at all, and the sweep below had nothing to sweep.  A bare
    Linux runner has none of them, produces the remedy on its first
    run, and failed the pre-cut gate on it.
    """

    first = line.split()[0] if line.split() else ""
    return (first in _REMEDY_COMMANDS
            or first in _EXTRA_COMMAND_TOKENS
            or line.endswith("\\")
            or "=" in first)


def test_the_readme_states_doctors_contract_not_the_old_overclaim():
    """The README claim is bound to doctor's behaviour, not free prose.

    v1.1.1 shipped "every gap prints the exact command that fixes it",
    which the two comment-only gaps above contradict. The README must
    carry doctor's real contract instead.
    """
    from pathlib import Path

    import gpuwm

    readme = (Path(gpuwm.__file__).resolve().parent.parent
              / "README.md").read_text(encoding="utf-8")
    assert "every gap prints the exact command that fixes it" not in readme
    assert "command or a `#` comment" in readme


# ---------------------------------------------------------------------------
# The paths a RUN resolves.  Doctor said "no gaps" and then a field user's
# first preparation died three separate ways on paths doctor never asked
# about; each of those questions is a check now, and each is tested here.
# ---------------------------------------------------------------------------

def _wheel_shaped_install(monkeypatch, tmp_path):
    """A package parent with no checkout crate and no staged bridges."""

    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    staged = tmp_path / "userdir"
    staged.mkdir()
    for variable in bridges.BRIDGE_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(bridges, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(bridges, "_package_parent", lambda: site_packages)
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: staged)
    return staged


def test_the_route_decoder_check_resolves_what_preparation_will_launch(
        monkeypatch, tmp_path):
    """Doctor and the wrapper ask ONE function, so they cannot disagree.

    The wheel failure: the wrapper resolved
    ``<site-packages>/tools/grib1_bridge/target/release/...`` -- a
    directory a wheel does not have -- while doctor resolved the shared
    ladder and reported the bridge ``gpuwm setup`` had staged.  Green
    report, missing file, one command apart.
    """

    from tools import prepare_hrrr_wrf

    staged = _wheel_shaped_install(monkeypatch, tmp_path)

    check = doctor._decoder_route_check("hrrr")
    assert check.status == "missing"
    assert check.blocking
    assert "hrrr_grib2_bridge" in check.detail
    assert str(staged) in check.detail

    decoder = staged / bridges.executable_name("hrrr_grib2_bridge")
    decoder.write_bytes(
        b"\x7fELF" + bridges.BRIDGE_ABI_MARKERS["hrrr_grib2_bridge"])
    check = doctor._decoder_route_check("hrrr")
    assert check.status == "verified"
    assert str(decoder.resolve()) in check.detail
    # And the wrapper resolves the same file, through the same call.
    assert prepare_hrrr_wrf._decoder({}) == decoder.resolve()


def test_a_stale_route_decoder_is_a_blocking_finding(monkeypatch, tmp_path):
    """Negative control for the ABI half: it exists and still refuses.

    Watched firing -- with the marker present the same file reports
    ``verified`` in the test above.
    """

    staged = _wheel_shaped_install(monkeypatch, tmp_path)
    (staged / bridges.executable_name("hrrr_grib2_bridge")).write_bytes(
        b"\x7fELFusage: hrrr_grib2_bridge OLD ARGUMENTS")
    check = doctor._decoder_route_check("hrrr")
    assert check.status == "missing"
    assert check.blocking
    assert "predates" in check.detail


def test_every_declared_doctor_source_has_route_checks():
    for source in doctor.DOCTOR_SOURCES:
        checks = doctor._source_route_checks(source)
        assert checks
        assert all(check.name.startswith(source) for check in checks)
    with pytest.raises(ValueError, match="unknown doctor source"):
        doctor._source_route_checks("not-a-source")


def test_the_gfs_route_reports_decoder_and_both_transports():
    """The 4090 user-zero report showed doctor naming only the hrrr
    route while gfs_grib2_bridge sat verified with no route around it;
    --source gfs must now answer with the decoder AND the transports."""

    checks = doctor._source_route_checks("gfs")
    names = [check.name for check in checks]
    assert "gfs route decoder" in names
    assert "gfs route fetch transport" in names
    transport = next(check for check in checks
                     if check.name == "gfs route fetch transport")
    assert transport.status == "verified"
    assert "NOMADS grib-filter crop" in transport.detail
    assert "full-file" in transport.detail


def test_the_gfs_transport_line_names_the_engine_it_resolved(monkeypatch):
    """Both installs are healthy; the line says which engine full-file
    gets.  Negative control: with the backbone unresolvable the check
    stays verified but names the stdlib transport and the remedy."""

    from gpuwm import rustwx_fetch

    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    check = doctor._gfs_fetch_path_check()
    assert check.status == "verified"
    assert "stdlib transport" in check.detail
    assert "gpuwm setup" in check.detail
    assert check.brief == "cgi-subset default; full-file via python"


def test_the_default_estate_carries_every_route(monkeypatch):
    """A bare ``gpuwm doctor`` reports the routes, not only the estate.

    Behind a flag they would have been invisible to the exact user this
    was written for: someone who ran ``gpuwm doctor``, read "no gaps",
    and then hit a route path the report had never resolved.
    """

    names = {check.name for check in doctor.collect_checks()}
    for source in doctor.DOCTOR_SOURCES:
        assert f"{source} route decoder" in names
    narrowed = {check.name for check in doctor.collect_checks(("hrrr",))}
    assert "hrrr route decoder" in narrowed


def test_the_provenance_check_reports_the_path_a_run_will_take():
    check = doctor._install_identity_check()
    assert check.status == "verified"
    assert any(source in check.detail
               for source in ("git", "installed-wheel-record",
                              "gpuwm-native-distribution-manifest"))


def test_the_provenance_check_is_a_blocking_gap_when_nothing_can_answer(
        monkeypatch):
    """Negative control, watched firing: no identity is a broken install."""

    from gpuwm.runtime_manifest import IdentityError

    def refuse(_root, **_kwargs):
        raise IdentityError("no provenance to bind")

    monkeypatch.setattr("gpuwm.runtime_manifest.provenance", refuse)
    check = doctor._install_identity_check()
    assert check.status == "missing"
    assert check.blocking


def test_the_route_entry_points_import_from_a_scratch_directory():
    check = doctor._non_git_import_check()
    assert check.status == "verified", check.detail
    for module in doctor._ROUTE_ENTRY_MODULES:
        assert module in check.detail


def test_a_route_entry_point_that_cannot_import_is_a_blocking_gap(
        monkeypatch):
    """Negative control, watched firing: an unimportable module is a gap."""

    monkeypatch.setattr(
        doctor, "_ROUTE_ENTRY_MODULES", ("tools.no_such_entry_point",))
    check = doctor._non_git_import_check()
    assert check.status == "missing"
    assert check.blocking
    assert "no_such_entry_point" in check.detail


def test_the_source_flag_narrows_the_report(monkeypatch, capsys):
    from types import SimpleNamespace

    code = doctor.doctor_main(
        SimpleNamespace(json=True, explain=False, source=["hrrr"]))
    assert code in (0, 1)
    import json as jsonlib

    names = {entry["name"] for entry in jsonlib.loads(capsys.readouterr().out)}
    assert "hrrr route decoder" in names


# ---------------------------------------------------------------------------
# The probe that could never time out.  ``subprocess.run(timeout=...)`` bounds
# the WAIT, not ``CreateProcess``; a file with a corrupt image header can make
# the Windows loader raise a modal dialog inside that call, which no timeout
# reaches and no hidden-window session dismisses.  A release battery froze
# there twice, probing sixteen bytes of ASCII.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload,why", (
    (b"not machine code", "the sixteen ASCII bytes that froze the battery"),
    (b"", "an empty file"),
    (b"MZ", "a truncated MZ stub with no signature offset"),
    (b"MZ" + b"\x00" * 0x3a + b"\xff\xff\xff\xff", "an out-of-range PE offset"),
    (b"#!/bin/sh\necho hello\n", "a shell script, not a native image"),
))
def test_a_file_that_is_not_a_native_image_is_never_launched(
        tmp_path, monkeypatch, payload, why):
    """The fix, at the mechanism: refuse from the bytes, launch nothing.

    ``subprocess.run`` is replaced by a landmine rather than merely
    timed, because "it finished quickly" is compatible with having got
    lucky.  The property under test is that the operating system is
    never asked about a file whose header already answers.
    """

    def landmine(*_args, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError(
            f"a non-executable was handed to CreateProcess: {why}")

    monkeypatch.setattr(doctor.subprocess, "run", landmine)
    fake = tmp_path / doctor.bridges.executable_name("grib1_bridge")
    fake.write_bytes(payload)
    if os.name != "nt":
        fake.chmod(0o755)
    ok, evidence = doctor._exec_probe(fake)
    assert not ok
    assert "corrupt" in evidence


def test_the_probe_of_a_non_image_returns_in_bounded_time(tmp_path):
    """And it does so on THIS box, wall clock, with nothing patched.

    The regression the battery hit was a hang, so the assertion that
    matters is elapsed time against a real call, not a mocked one.  The
    budget is generous by three orders of magnitude and still fails a
    return to the old behaviour, which did not return at all.
    """

    import time

    fake = tmp_path / doctor.bridges.executable_name("grib1_bridge")
    fake.write_bytes(b"not machine code")
    if os.name != "nt":
        fake.chmod(0o755)
    started = time.perf_counter()
    ok, _evidence = doctor._exec_probe(fake)
    elapsed = time.perf_counter() - started
    assert not ok
    assert elapsed < 5.0, f"the probe took {elapsed:.1f} s"


def test_the_header_classifier_accepts_real_images_and_nothing_else(tmp_path):
    """Negative control on the gate itself: a real header still passes.

    Watched firing -- flip the magic bytes and the same file is refused.
    Without this, a gate that refused everything would look like a fix.
    """

    from gpuwm import bridges as bridges_module

    elf = tmp_path / "elf.bin"
    elf.write_bytes(b"\x7fELF" + b"\x00" * 0x40)
    assert bridges_module.native_executable_format(elf)[0] == "elf"

    pe = tmp_path / "pe.bin"
    pe.write_bytes(b"MZ" + b"\x00" * 0x3a + (0x80).to_bytes(4, "little")
                   + b"\x00" * 0x40)
    assert bridges_module.native_executable_format(pe)[0] == "pe"

    broken = tmp_path / "broken.bin"
    broken.write_bytes(b"XX" + b"\x00" * 0x40)
    assert bridges_module.native_executable_format(broken)[0] is None

    assert bridges_module.native_executable_format(
        tmp_path / "absent.bin")[0] is None


def test_launchable_refuses_a_binary_built_for_the_other_platform(tmp_path):
    from gpuwm import bridges as bridges_module

    other = "elf" if os.name == "nt" else "pe"
    path = tmp_path / "other.bin"
    if other == "elf":
        path.write_bytes(b"\x7fELF" + b"\x00" * 0x40)
    else:
        path.write_bytes(b"MZ" + b"\x00" * 0x3a
                         + (0x80).to_bytes(4, "little") + b"\x00" * 0x40)
    ok, evidence = bridges_module.launchable(path)
    assert not ok
    assert "another platform" in evidence


def test_quiet_loader_errors_restores_what_it_found():
    """The error-mode backstop must not leak into the rest of the run."""

    from gpuwm import bridges as bridges_module

    if os.name != "nt":
        with bridges_module.quiet_loader_errors():
            pass
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    before = kernel32.SetErrorMode(0)
    kernel32.SetErrorMode(before)
    with bridges_module.quiet_loader_errors():
        inside = kernel32.SetErrorMode(0)
        kernel32.SetErrorMode(inside)
        assert inside & bridges_module.quiet_loader_errors._FAIL_FAST
    after = kernel32.SetErrorMode(0)
    kernel32.SetErrorMode(after)
    assert after == before


def test_the_other_two_probes_share_the_same_gate(tmp_path, monkeypatch):
    """One mechanism, every probe: the renderer and the fetch backbone.

    Three copies of ``subprocess.run(timeout=...)`` had the same hang,
    and fixing only the one the battery happened to hit would have left
    two loaded guns in the same report.
    """

    from gpuwm import rustwx, rustwx_fetch

    def landmine(*_args, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a non-executable was handed to CreateProcess")

    monkeypatch.setattr(rustwx.subprocess, "run", landmine)
    monkeypatch.setattr(rustwx_fetch.subprocess, "run", landmine)
    fake = tmp_path / "rw_thing"
    fake.write_bytes(b"not machine code")
    if os.name != "nt":
        fake.chmod(0o755)
    for probe in (rustwx.probe_renderer, rustwx_fetch.probe_fetch_bin):
        ok, evidence = probe(fake)
        assert not ok
        assert "corrupt" in evidence


def test_the_gap_count_agrees_with_the_exit_code():
    """The reported ``gpuwm setup`` 1.3.1 -> 1.4.0 exit regression.

    1.3.1 exited 1 on an unfetched WPS_GEOG tree and 1.4.0 exits 0, with
    the gap text unchanged.  Exiting 0 is the right answer -- that ~16 GB
    download is an explicit opt-in and an install that did everything its
    documentation asked must not fail an installer.  What made it read as
    a silent regression is that both reports printed the same "1 gap(s)"
    line, so the only visible difference between the two versions was the
    exit code.  The severity has to be in the report.
    """
    opt_in = doctor.Check(
        "WPS_GEOG", "missing", "the default geog_root does not exist",
        action="gpuwm fetch-geog", brief="not staged", blocking=False)
    broken = doctor.Check(
        "bridge x", "missing", "not staged", action="gpuwm fetch-bridges",
        brief="not staged")

    only_opt_in = [doctor.Check("python", "verified", "3.13"), opt_in]
    assert doctor.blocking_gaps(only_opt_in) == []
    brief = doctor.format_brief(only_opt_in)
    full = doctor.format_report(only_opt_in)
    # The count that matters is stated, and it is the exit code.
    assert "0 of them blocking (exit 0)" in brief
    assert "0 of them blocking (the exit code is 0)" in full
    # ...and the finding itself is not softened away: MISSING stays
    # MISSING and the command that closes it still prints.
    assert "MISSING WPS_GEOG" in brief and "gpuwm fetch-geog" in brief

    both = [doctor.Check("python", "verified", "3.13"), opt_in, broken]
    assert len(doctor.blocking_gaps(both)) == 1
    brief = doctor.format_brief(both)
    assert "2 gap(s), 1 of them blocking (exit 1)" in brief


# --- The CuPy-wheel/box CUDA-major pairing (the rental-box trap) --------
#
# cupy-cuda12x on a CUDA-13-only box imports cleanly, compiles kernels,
# and dies at its first cuBLAS load -- proven on a rented CUDA 13.2 node
# where import probes, certification, and a 57-test GPU slice all passed
# before the first matmul of the campaign killed it.  Doctor performs
# that first load deliberately, and when it fails, the remedy must name
# the extra whose wheel matches the box.  These monkeypatch the probe:
# the trap cannot be staged honestly on a healthy box, and the check's
# judgment is a pure function of the probe's report.


def _pairing(monkeypatch, wheels, probe):
    # These scenarios model boxes where the device IS judged; the
    # suite's own no-device flag must not short-circuit them.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (True, "14.1.1"))
    monkeypatch.setattr(doctor, "_installed_cupy_wheels", lambda: wheels)
    monkeypatch.setattr(doctor, "_cublas_pairing_probe", lambda: probe)
    return doctor._cupy_check()


def test_cu12_wheel_on_a_cuda13_only_box_refuses_and_names_gpu_cu13(
        monkeypatch):
    check = _pairing(
        monkeypatch, [("cupy-cuda12x", 12)],
        {"wheel_runtime": 12090, "driver": 13020, "devices": 1,
         "cublas": "ImportError: libcublas.so.12: cannot open shared "
                   "object file: No such file or directory"})
    assert check.status == "missing"
    assert check.blocking, "a run on this box dies at its first matmul"
    assert "pip uninstall -y cupy-cuda12x" in check.remedy
    assert "gpuwm[gpu-cu13]" in check.remedy
    assert check.action == "pip install 'gpuwm[gpu-cu13]'"
    # The diagnosis states both majors and why import probes missed it.
    assert "CUDA 12" in check.detail and "CUDA 13" in check.detail
    for windows in (False, True):
        _assert_remedy_lines_are_commands_or_comments(
            check.remedy, windows=windows)


def test_cu13_wheel_on_a_cuda12_box_names_the_gpu_extra(monkeypatch):
    check = _pairing(
        monkeypatch, [("cupy-cuda13x", 13)],
        {"wheel_runtime": 13000, "driver": 12080, "devices": 1,
         "cublas": "ImportError: libcublas.so.13: cannot open shared "
                   "object file: No such file or directory"})
    assert check.status == "missing" and check.blocking
    assert "pip uninstall -y cupy-cuda13x" in check.remedy
    # gpu-cu12, not the bare `gpu` alias: a remedy printed because the
    # majors disagree has to say WHICH major it chose.
    assert check.action == "pip install 'gpuwm[gpu-cu12]'"
    for windows in (False, True):
        _assert_remedy_lines_are_commands_or_comments(
            check.remedy, windows=windows)


def test_a_working_cublas_load_is_verified_even_across_majors(monkeypatch):
    """A newer driver serving an older wheel is a WORKING install.

    The reference box runs a 13.3 driver over cupy-cuda12x, so a bare
    major comparison would refuse the machine this release was cut on.
    The load is the judgment; the majors are the diagnosis.
    """
    check = _pairing(
        monkeypatch, [("cupy-cuda12x", 12)],
        {"wheel_runtime": 12090, "driver": 13030, "devices": 1,
         "cublas": "ok"})
    assert check.status == "verified"
    assert "cuBLAS loaded" in check.detail
    assert "12.9" in check.detail and "13.3" in check.detail


def test_cupy_without_a_device_stays_present_and_nonblocking(monkeypatch):
    check = _pairing(
        monkeypatch, [("cupy-cuda12x", 12)],
        {"wheel_runtime": 12090, "driver": 0, "devices": 0,
         "device_error": "CUDARuntimeError: cudaErrorNoDevice"})
    assert check.status == "present"
    assert not check.blocking
    assert "not judged" in check.detail


def test_a_cuda_major_without_a_packaged_extra_gets_the_bare_wheel(
        monkeypatch):
    check = _pairing(
        monkeypatch, [("cupy-cuda12x", 12)],
        {"wheel_runtime": 12090, "driver": 14000, "devices": 1,
         "cublas": "ImportError: libcublas.so.12: cannot open shared "
                   "object file: No such file or directory"})
    assert check.status == "missing"
    assert "pip install cupy-cuda14x" in check.remedy
    for windows in (False, True):
        _assert_remedy_lines_are_commands_or_comments(
            check.remedy, windows=windows)


def _extras() -> dict[str, list[str]]:
    import tomllib

    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["optional-dependencies"]


def _resolve_extra(extras: dict[str, list[str]], name: str) -> set[str]:
    """Every concrete requirement ``pip install 'gpuwm[name]'`` reaches.

    `gpuwm[...]` self-references are followed, because that is what pip
    does and because the whole point of the alias layer is that one
    wheel is pinned in exactly one place.
    """

    seen: set[str] = set()
    out: set[str] = set()
    stack = [name]
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        assert key in extras, f"pyproject declares no [{key}] extra"
        for dep in extras[key]:
            inner = re.fullmatch(r"gpuwm\[([^\]]+)\]", dep.strip())
            if inner:
                stack.extend(part.strip()
                             for part in inner.group(1).split(","))
            else:
                out.add(dep)
    return out


def _cupy_wheels(requirements) -> set[str]:
    return {re.split(r"[<>=!~ ]", dep)[0] for dep in requirements
            if dep.startswith("cupy")}


def test_the_gpu_extras_pin_one_cupy_wheel_per_cuda_major():
    """One extra per CUDA major, and each reaches exactly one wheel.

    The doctor remedy names these extras, so their existence and their
    wheels are part of the check's contract, not just packaging.

    `gpu`/`all` are kept as cu12 ALIASES rather than second definitions:
    every install that already works names them, so they may not change
    meaning, but they also may not be a place a wheel is pinned twice --
    the 1.8.0 table pinned cu12 in `[gpu]` and had no cu13 one-liner at
    all, which is how a CUDA-13 box following the quickstart got a wheel
    that cannot load cuBLAS on it (issue #76).
    """
    extras = _extras()
    assert _cupy_wheels(_resolve_extra(extras, "gpu-cu12")) == {
        "cupy-cuda12x"}
    assert _cupy_wheels(_resolve_extra(extras, "gpu-cu13")) == {
        "cupy-cuda13x"}
    # The aliases: same wheel, no drift, and no cu13 leaking into cu12.
    assert _resolve_extra(extras, "gpu") == _resolve_extra(
        extras, "gpu-cu12")
    assert _resolve_extra(extras, "all") == _resolve_extra(
        extras, "all-cu12")
    # A one-liner per major, each carrying the renderer too.
    for major, wheel in ((12, "cupy-cuda12x"), (13, "cupy-cuda13x")):
        resolved = _resolve_extra(extras, f"all-cu{major}")
        assert _cupy_wheels(resolved) == {wheel}
        assert any(dep.startswith("wrf-rust") for dep in resolved)
    # The demo/gallery renderer's shapefile reader rides the extra the
    # quickstart installs (proven missing on a fresh env without it).
    assert any(dep.startswith("pyshp") for dep in extras["render"])


def test_every_cuda_major_doctor_names_has_an_extra_that_exists():
    """The remedy table and the packaging table cannot drift apart.

    `_GPU_EXTRA_BY_MAJOR` is what doctor prints at the moment a user is
    deciding what to install; an extra named there that pyproject does
    not declare is a remedy that fails when pasted.
    """
    extras = _extras()
    for major, extra in doctor._GPU_EXTRA_BY_MAJOR.items():
        assert _cupy_wheels(_resolve_extra(extras, extra)) == {
            f"cupy-cuda{major}x"}


# --- The extra a box with NO CuPy yet is told to install (issue #76) ----
#
# This is the branch the pairing probe above can never reach: it needs a
# CuPy to interrogate, and the user who most needs the right answer has
# none.  Through 1.8.0 the remedy here was a constant whose only command
# was the cu12 extra, so a CUDA-13-only box asked doctor what to install
# and was told to install the wheel that cannot load cuBLAS on it.


def _absent_cupy(monkeypatch, box_major):
    monkeypatch.setattr(doctor, "find_spec", lambda name: None)
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: box_major)
    return doctor._cupy_check()


def test_a_cuda13_box_with_no_cupy_is_told_to_install_the_cu13_extra(
        monkeypatch):
    check = _absent_cupy(monkeypatch, 13)
    assert check.status == "missing" and not check.blocking
    assert check.action == "pip install 'gpuwm[gpu-cu13]'"
    assert "gpu-cu12" not in check.remedy
    assert "CUDA 13" in check.remedy
    for windows in (False, True):
        _assert_remedy_lines_are_commands_or_comments(
            check.remedy, windows=windows)


def test_a_cuda12_box_with_no_cupy_is_told_to_install_the_cu12_extra(
        monkeypatch):
    check = _absent_cupy(monkeypatch, 12)
    assert check.action == "pip install 'gpuwm[gpu-cu12]'"
    assert "gpu-cu13" not in check.remedy
    for windows in (False, True):
        _assert_remedy_lines_are_commands_or_comments(
            check.remedy, windows=windows)


def test_an_unreadable_cuda_major_names_both_extras_and_defaults_to_neither(
        monkeypatch):
    """No detection is not licence to guess cu12.

    The failure this closes is silent: a wrong default reads as a
    recommendation, and the reader has no way to tell that doctor did
    not actually know.
    """
    check = _absent_cupy(monkeypatch, None)
    assert "pip install 'gpuwm[gpu-cu12]'" in check.remedy
    assert "pip install 'gpuwm[gpu-cu13]'" in check.remedy
    # The one-command brief line still carries the alternative.
    assert "gpu-cu13" in check.action
    for windows in (False, True):
        _assert_remedy_lines_are_commands_or_comments(
            check.remedy, windows=windows)


def test_an_exotic_cuda_major_with_no_extra_falls_back_to_the_choice(
        monkeypatch):
    """A major gpuwm does not package must not be invented as an extra."""
    check = _absent_cupy(monkeypatch, 14)
    assert "gpuwm[gpu-cu14]" not in check.remedy
    assert "pip install 'gpuwm[gpu-cu12]'" in check.remedy
    assert "pip install 'gpuwm[gpu-cu13]'" in check.remedy


def test_the_driver_probe_stays_off_the_device_under_no_local_gpu(
        monkeypatch):
    """GPUWM_NO_LOCAL_GPU suppresses the driver read too.

    cuDriverGetVersion opens no context, but the flag's promise is that
    doctor does not touch the local card at all -- so the probe answers
    "unknown" rather than reaching for nvcuda.dll, and the remedy
    degrades to naming both extras.
    """
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")

    def explode(name):
        raise AssertionError(f"loaded {name!r} under GPUWM_NO_LOCAL_GPU")

    monkeypatch.setattr(doctor.ctypes, "CDLL", explode)
    assert doctor._driver_cuda_major() is None


def test_the_driver_probe_reports_none_when_no_driver_is_installed(
        monkeypatch):
    """A box with no NVIDIA driver is the ordinary case, not an error."""
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)

    def missing(name):
        raise OSError(f"cannot open shared object file: {name}")

    monkeypatch.setattr(doctor.ctypes, "CDLL", missing)
    assert doctor._driver_cuda_major() is None


def test_the_driver_probe_reads_the_major_out_of_cudrivergetversion(
        monkeypatch):
    """13040 is CUDA 13.4, so the major is 13."""
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)

    class _Driver:
        @staticmethod
        def cuDriverGetVersion(out):
            out._obj.value = 13040
            return 0

    monkeypatch.setattr(doctor.ctypes, "CDLL", lambda name: _Driver)
    assert doctor._driver_cuda_major() == 13


def test_the_driver_probe_ignores_a_nonzero_cuda_status(monkeypatch):
    """A driver that refuses the call has told us nothing, not zero."""
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)

    class _Driver:
        @staticmethod
        def cuDriverGetVersion(out):
            out._obj.value = 12080
            return 3          # CUDA_ERROR_NOT_INITIALIZED

    monkeypatch.setattr(doctor.ctypes, "CDLL", lambda name: _Driver)
    assert doctor._driver_cuda_major() is None


def test_no_local_gpu_keeps_doctor_off_the_device(monkeypatch):
    """GPUWM_NO_LOCAL_GPU means NO device contact, doctor included.

    The suite's `-m "not gpu"` guarantee and the rented-GPU workflow
    both rest on this flag; the pairing probe opens a CUDA context, so
    under the flag it must not run at all -- asserted by making any
    call to it the failure.
    """
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (True, "14.1.1"))
    monkeypatch.setattr(doctor, "_installed_cupy_wheels",
                        lambda: [("cupy-cuda12x", 12)])

    def _forbidden():
        raise AssertionError("the pairing probe touched the device "
                             "under GPUWM_NO_LOCAL_GPU")

    monkeypatch.setattr(doctor, "_cublas_pairing_probe", _forbidden)
    check = doctor._cupy_check()
    assert check.status == "present"
    assert not check.blocking
    assert "not judged" in check.detail
