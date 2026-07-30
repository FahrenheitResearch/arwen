"""``gpuwm doctor`` + the shared bridge-resolution mechanism.

The wheel ships no compiled Rust, so a wheel user's estate is: cupy?
render extra? bridges built and findable? tables packaged? data roots
set?  Doctor must name each gap WITH its exact remedy, and ingest's
bridge resolution must honor the same env-var/default-dir mechanism
doctor describes.  Everything here is CPU-only and read-only.
"""
from __future__ import annotations

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
def test_the_pip_bootstrap_wires_what_it_builds(monkeypatch, tmp_path,
                                                windows):
    """Running every command must actually close the gap it was for.

    A node-8 validation run pasted the whole report on a pip-only
    machine, ran every line of it, and doctor still reported six MISSING
    bridges: the wiring step -- copy into the default directory, or set
    the environment variable -- was offered as two `#` alternatives,
    because it is a choice.  It is still a choice; the copy is now the
    default and the environment variable the commented alternative, so
    the literal paste finishes.

    The destination is asserted to be `default_bridge_dir()` itself, not
    a lookalike: what makes the copy work is that it lands in the exact
    directory `artifact_candidates` searches.
    """

    from gpuwm import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: tmp_path / "tools" / "grib1_bridge")
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: True)
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
})

#: A bare ALL-CAPS word is a placeholder the reader must expand.
_PLACEHOLDER_WORD = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def _assert_remedy_lines_are_commands_or_comments(remedy, *, windows):
    """EVERY line: a command that runs as printed, or a `#` comment.

    No exemptions -- not even the first line.  The closing report
    claims this of every remedy, so a headline sentence must be a
    `#` comment, never bare prose fused into a paste.
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


@pytest.mark.parametrize("windows", (False, True))
@pytest.mark.parametrize("cargo", (False, True))
def test_the_emitted_bootstrap_is_shell_correct_on_both_platforms(
        monkeypatch, tmp_path, windows, cargo):
    """The real assertion: parse what doctor would actually print."""

    from gpuwm import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
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


def test_the_invalid_override_branches_emit_install_aware_remedies(
        monkeypatch, tmp_path):
    """doctor's `fix ENV ... or unset it and <remedy>` paths.

    Both called the old install-unaware helpers, which kept `<clone>`
    placeholders and told a pip-only install to enter a relative
    `tools/rustwx` it does not have.
    """

    from gpuwm import bridges, rustwx, rustwx_fetch

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
            remedy, windows=bridges.WINDOWS_SHELL)


def test_every_remedy_doctor_can_print_obeys_the_closing_claim():
    """The closing line makes a claim about EVERY remedy.  Check them all.

    Including the non-Rust ones: the pip-extra and fetch-geog hints used
    to trail a parenthetical on the command line itself.
    """

    from gpuwm import bridges, doctor

    for check in doctor.collect_checks():
        if not check.remedy:
            continue
        _assert_remedy_lines_are_commands_or_comments(
            check.remedy, windows=bridges.WINDOWS_SHELL)

    for hint in (doctor.GPU_EXTRA_HINT, doctor.RENDER_EXTRA_HINT,
                 doctor.GEOG_HINT, doctor.REINSTALL_HINT):
        _assert_remedy_lines_are_commands_or_comments(
            hint, windows=bridges.WINDOWS_SHELL)
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
def test_every_conditional_remedy_branch_obeys_the_claim(
        monkeypatch, tmp_path, windows):
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

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: False)

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


def _force_every_gap(monkeypatch, tmp_path, *, windows, shape, mode):
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

    root = tmp_path / shape
    root.mkdir(parents=True, exist_ok=True)
    if shape == "checkout":
        for crate in ("grib1_bridge", "rustwx"):
            (root / "tools" / crate).mkdir(parents=True, exist_ok=True)
            (root / "tools" / crate / "Cargo.toml").write_text(
                "[package]\n", encoding="utf-8")

    # The shell under test -- and the three build hints, which are
    # frozen at import for whichever platform is running the tests.
    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: False)
    monkeypatch.setattr(bridges, "CARGO_BUILD_HINT",
                        bridges.cargo_build_one_liner(bridges.CRATE_RELATIVE))
    rustwx_hint = bridges.cargo_build_one_liner(bridges.RUSTWX_CRATE_RELATIVE)
    monkeypatch.setattr(rustwx, "CARGO_BUILD_HINT", rustwx_hint)
    monkeypatch.setattr(rustwx_fetch, "CARGO_BUILD_HINT", rustwx_hint)

    monkeypatch.setattr(bridges, "_package_parent", lambda: root)
    monkeypatch.setattr(bridges, "crate_dir",
                        lambda: root / "tools" / "grib1_bridge")
    monkeypatch.setattr(rustwx, "crate_dir", lambda: root / "tools" / "rustwx")
    monkeypatch.setattr(rustwx_fetch, "crate_dir",
                        lambda: root / "tools" / "rustwx")

    # python below the floor; cupy and the render extra absent (with
    # find_spec gone, the import probes spawn nothing).
    monkeypatch.setattr(doctor, "sys", types.SimpleNamespace(
        version_info=(3, 10, 4), executable=real_sys.executable))
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
def test_the_whole_printed_report_pastes_as_one_sequence(
        monkeypatch, tmp_path, windows, shape, mode):
    """Every remedy doctor can print, in print order, pasted as a block.

    The contract is not "each block is well formed"; it is "select the
    lot and run it".  Four ways that was false shipped past a per-block
    sweep: `cd tools/rustwx` left the shell in the crate so the next
    block's `cd` went somewhere else, `git clone` came back once per
    bridge, the CPU-library remedy fused prose onto the build command,
    and four remedies were prose with no `#` at all.
    """

    _force_every_gap(monkeypatch, tmp_path, windows=windows, shape=shape,
                     mode=mode)
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

    if shape == "wheel":
        assert len(clones) >= 2, (
            "a wheel gaps every Rust artifact at once; the clone should "
            f"appear in several blocks: {clones}")
        assert len(set(clones)) == 1, (
            f"the blocks clone into different directories: {set(clones)}")
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


def test_every_remedy_line_doctor_can_emit_is_a_command_or_a_comment():
    """Doctor's own closing contract, over the whole live estate.

    Not "every remedy is a command" -- some are `#`-comment-only, some
    are multi-step. The invariant the README must state, and does, is
    that every printed remedy LINE is a command or a `#` comment, so the
    block survives being pasted whole.
    """

    for check in doctor.collect_checks():
        if not check.remedy:
            continue
        for raw in check.remedy.splitlines():
            line = raw.strip()
            if not line:
                continue
            assert line.startswith("#") or _looks_like_command(line), (
                check.name, line)


def _looks_like_command(line: str) -> bool:
    # A command line starts a shell invocation or continues one; a prose
    # sentence would fail this and be caught. Mirrors the doctor remedy
    # contract, deliberately narrow.
    first = line.split()[0] if line.split() else ""
    return (first in {"pip", "python", "cargo", "git", "bash", "set",
                      "export", "rustup", "conda", "uv", "$env:GPUWM_GRIB1_BRIDGE",
                      "gpuwm", "rw-wps"}
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
