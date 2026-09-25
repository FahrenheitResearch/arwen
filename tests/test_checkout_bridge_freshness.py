"""A checkout's own bridge build, measured against the tree it serves.

Reported from the field against 2.7.4/2.7.5: after `git pull` the
binary at `tools/rustwx/target/release/rw_netcdf` was the previous
release's, and the first thing that said so was a decode failure inside
`gpuwm run`.  Nothing on the resolution ladder had asked whether the
build was current: the contract marker (`BRIDGE_ABI_MARKERS`) only
answers when somebody remembers to change it, and it is identical
across 2.7.4, 2.7.5 and 2.7.6 while the Rust under it moved.

The judgement here is cargo's own and needs no cooperation from the
build that already happened: is any file the binary is compiled from
newer than the binary?  It is asked about one KIND OF FILE rather than
about one rung: a binary under this installation's own
`target/{release,debug}`, which is a checkout build whether the ladder
walked to it or `GPUWM_RW_NETCDF` named it.  A wheel, a `libexec`
directory, a fetched bundle and an override naming a built copy
anywhere else have no sources here and are untouched by it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpuwm import bridges


MANIFEST = """[package]
name = "{name}"
version = "0.1.0"

[[bin]]
name = "{binary}"
path = "src/main.rs"

[dependencies]
{deps}

[dev-dependencies]
{dev}
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A source checkout shaped like this one, with one built bridge.

    Two crates and one vendored path dependency, because the question
    "which sources build THIS binary" is the whole precision of the
    check: an edit to the other crate must not condemn this binary, and
    an edit to the dependency it compiles in must.
    """

    root = tmp_path / "checkout"
    workspace = root / "tools" / "rustwx"
    _write(workspace / "Cargo.toml", "[workspace]\nmembers = [\"crates/*\"]\n")
    _write(workspace / "crates" / "rw-netcdf" / "Cargo.toml",
           MANIFEST.format(
               name="rw-netcdf", binary="rw_netcdf",
               deps='netcrust = { path = "../../vendor/netcrust" }',
               dev='netcdf-writer = { path = "../netcdf-writer" }'))
    _write(workspace / "crates" / "rw-netcdf" / "src" / "main.rs", "fn main() {}\n")
    _write(workspace / "crates" / "rw-mpas" / "Cargo.toml",
           MANIFEST.format(name="rw-mpas", binary="rw_mpas_mesh", deps="", dev=""))
    _write(workspace / "crates" / "netcdf-writer" / "Cargo.toml",
           "[package]\nname = \"netcdf-writer\"\nversion = \"0.1.0\"\n")
    _write(workspace / "crates" / "netcdf-writer" / "src" / "lib.rs", "// writer\n")
    _write(workspace / "crates" / "rw-mpas" / "src" / "main.rs", "fn main() {}\n")
    _write(workspace / "vendor" / "netcrust" / "Cargo.toml",
           "[package]\nname = \"netcrust\"\nversion = \"0.1.0\"\n")
    _write(workspace / "vendor" / "netcrust" / "src" / "lib.rs", "// reader\n")
    binary = workspace / "target" / "release" / bridges.executable_name("rw_netcdf")
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x7fELF GPUWM_BRIDGE_SOURCE_REV=" + b"ab12" * 10)
    _touch(binary, 2_000_000_000)
    monkeypatch.setattr(bridges, "_package_parent", lambda: root)
    bridges._CRATE_INDEX.clear()
    yield workspace, binary
    bridges._CRATE_INDEX.clear()


def _touch(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def test_a_build_newer_than_its_sources_is_handed_over_unchanged(checkout):
    workspace, binary = checkout
    assert bridges.checkout_build_status(binary).current is True
    assert bridges.require_current_checkout_build(binary) == binary


def test_a_pull_that_moves_this_crate_refuses_with_the_numbers_and_the_command(checkout):
    workspace, binary = checkout
    source = workspace / "crates" / "rw-netcdf" / "src" / "main.rs"
    _touch(source, 2_000_086_400)
    with pytest.raises(bridges.StaleCheckoutBuildError) as refused:
        bridges.require_current_checkout_build(binary)
    text = str(refused.value)
    assert "rw_netcdf" in text and str(binary) in text
    # The binary names the revision it was built from, and the refusal
    # names the file that moved past it and the command that ends it.
    assert "ab12" * 10 in text
    assert "main.rs" in text
    assert "cargo build --release --locked --offline" in text
    assert "24.0 hours" in text


def test_a_dependency_this_binary_compiles_in_counts_as_its_own_source(checkout):
    workspace, binary = checkout
    _touch(workspace / "vendor" / "netcrust" / "src" / "lib.rs", 2_000_086_400)
    assert bridges.checkout_build_status(binary).current is False


def test_a_crate_only_the_suite_compiles_does_not_condemn_this_binary(checkout):
    """Dev dependencies build the tests, never the shipped binary."""

    workspace, binary = checkout
    _touch(workspace / "crates" / "netcdf-writer" / "src" / "lib.rs", 2_000_086_400)
    assert bridges.checkout_build_status(binary).current is True


def test_another_crate_moving_does_not_condemn_this_binary(checkout):
    workspace, binary = checkout
    _touch(workspace / "crates" / "rw-mpas" / "src" / "main.rs", 2_000_086_400)
    assert bridges.checkout_build_status(binary).current is True
    assert bridges.require_current_checkout_build(binary) == binary


def test_the_lock_and_the_workspace_manifest_are_build_inputs(checkout):
    workspace, binary = checkout
    _touch(_write(workspace / "Cargo.lock", "# locked\n"), 2_000_086_400)
    assert bridges.checkout_build_status(binary).current is False


def test_only_a_checkouts_own_target_directory_is_judged(tmp_path, monkeypatch):
    """A wheel, a libexec directory and a fetched bundle are not narrowed.

    They arrived built, there is no tree here that makes them, and the
    staged rung already has its own judgement (:func:`require_release_pin`).
    """

    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    staged = tmp_path / "gpuwm" / "libexec" / "bridges" / "rw_netcdf"
    _write(staged, "not really a binary")
    assert bridges.checkout_workspace_of(staged) is None
    assert bridges.checkout_build_status(staged) is None
    assert bridges.require_current_checkout_build(staged) == staged


def test_an_override_into_this_installations_own_target_is_judged(
        checkout, monkeypatch):
    """`GPUWM_RW_NETCDF` naming this target names a checkout build.

    The override is the FIRST candidate of every ladder in the package,
    and the override a developer sets is most often the binary they
    just built in this tree.  Exempting it would leave the reported
    defect reachable through the rung a developer uses most, and the
    file is the same file the walked rung would have found.
    """

    from gpuwm import netcdf_bridge

    workspace, binary = checkout
    _touch(workspace / "crates" / "rw-netcdf" / "src" / "main.rs", 2_000_086_400)
    monkeypatch.setenv(netcdf_bridge.NETCDF_ENV, str(binary))
    with pytest.raises(bridges.StaleCheckoutBuildError) as refused:
        netcdf_bridge.find_netcdf_bin()
    text = str(refused.value)
    assert str(binary) in text
    assert "cargo build --release --locked --offline" in text


def test_an_override_naming_a_build_outside_this_installation_is_handed_over(
        checkout, tmp_path, monkeypatch):
    """The remedies that say "point at a built copy" keep working.

    A copy built somewhere else has no sources under this installation
    to be measured against, so it is handed over whatever its age --
    including a copy whose own workspace travelled with it.
    """

    workspace, binary = checkout
    elsewhere = tmp_path / "prebuilt"
    _write(elsewhere / "Cargo.toml", "[workspace]\nmembers = []\n")
    _write(elsewhere / "crates" / "rw-netcdf" / "Cargo.toml",
           MANIFEST.format(name="rw-netcdf", binary="rw_netcdf", deps="", dev=""))
    _write(elsewhere / "crates" / "rw-netcdf" / "src" / "main.rs",
           "fn main() {}\n")
    copy = elsewhere / "target" / "release" / bridges.executable_name("rw_netcdf")
    _write(copy, "built elsewhere")
    _touch(copy, 2_000_000_000)
    _touch(elsewhere / "crates" / "rw-netcdf" / "src" / "main.rs", 2_000_086_400)
    monkeypatch.setenv("GPUWM_GRIB1_BRIDGE", str(copy))
    assert bridges.checkout_workspace_of(copy) is None
    found = bridges.find_artifact(
        "GPUWM_GRIB1_BRIDGE", bridges.executable_name("rw_netcdf"))
    assert found is not None and found.resolve() == copy.resolve()


def test_a_build_outside_this_installation_is_not_judged(checkout, monkeypatch):
    workspace, binary = checkout
    _touch(workspace / "crates" / "rw-netcdf" / "src" / "main.rs", 2_000_086_400)
    monkeypatch.setattr(bridges, "_package_parent", lambda: Path(__file__).parent)
    assert bridges.checkout_build_status(binary) is None


def test_a_report_reads_the_same_judgement_without_dying_on_it(checkout):
    """Doctor resolves inside ``inspection_only`` and must come back.

    The staged-pin defect this repeats: the report said the estate was
    wrong while the ladder kept handing the same bytes to the routes.
    Here the two read one function and differ only in what they do.
    """

    workspace, binary = checkout
    _touch(workspace / "crates" / "rw-netcdf" / "src" / "main.rs", 2_000_086_400)
    with bridges.inspection_only():
        assert bridges.require_current_checkout_build(binary) == binary
    assert bridges.checkout_build_status(binary).current is False


def test_the_accept_step_every_ladder_ends_at_applies_it(checkout):
    workspace, binary = checkout
    _touch(workspace / "crates" / "rw-netcdf" / "src" / "main.rs", 2_000_086_400)
    with pytest.raises(bridges.StaleCheckoutBuildError):
        bridges.accept_resolved(binary)


def test_the_doctor_line_names_it_rather_than_reporting_the_decoder_staged(
        checkout, monkeypatch):
    from gpuwm import doctor, netcdf_bridge

    workspace, binary = checkout
    _touch(workspace / "crates" / "rw-netcdf" / "src" / "main.rs", 2_000_086_400)
    monkeypatch.setattr(netcdf_bridge, "find_netcdf_bin", lambda: binary)
    check = doctor._netcdf_decoder_check()
    assert check.status == "missing"
    assert "older than" in check.detail or "behind the sources" in check.detail
    assert "cargo build --release --locked --offline" in (check.remedy or "")


def test_a_shared_library_is_refused_by_the_name_the_workspace_declares(
        tmp_path, monkeypatch):
    """The refusal prints the artifact, so it prints it spelled properly.

    A shared library lands on disk as ``libgpuwm_preprocess_cpu.so``.
    The crate index is asked twice for exactly that reason, because the
    name the workspace declares has no platform prefix, and the refusal
    used the raw file stem: it told a reader that "the Rust bridge
    `libgpuwm_preprocess_cpu`" was stale, which is not a name that
    appears in the manifest, in the remedy, or anywhere else the reader
    could follow it to.
    """
    root = tmp_path / "checkout"
    workspace = root / "tools" / "rustwx"
    _write(workspace / "Cargo.toml", "[workspace]\nmembers = [\"crates/*\"]\n")
    _write(workspace / "crates" / "preprocess" / "Cargo.toml",
           "[package]\nname = \"preprocess\"\nversion = \"0.1.0\"\n"
           "\n[lib]\nname = \"gpuwm_preprocess_cpu\"\n"
           "crate-type = [\"cdylib\"]\npath = \"src/lib.rs\"\n")
    _write(workspace / "crates" / "preprocess" / "src" / "lib.rs", "// cpu\n")
    binary = workspace / "target" / "release" / "libgpuwm_preprocess_cpu.so"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x7fELF")
    _touch(binary, 2_000_000_000)
    monkeypatch.setattr(bridges, "_package_parent", lambda: root)
    bridges._CRATE_INDEX.clear()
    try:
        _touch(workspace / "crates" / "preprocess" / "src" / "lib.rs",
               2_000_086_400)
        status = bridges.checkout_build_status(binary)
        assert status is not None and status.current is False
        assert status.artifact == "gpuwm_preprocess_cpu"
        sentence = status.refusal().splitlines()[0]
        assert "`gpuwm_preprocess_cpu`" in sentence
        assert "libgpuwm_preprocess_cpu" not in sentence
        # The file on disk keeps its real name, which is what the
        # `what:` line reports; only the SENTENCE names the artifact.
        assert "libgpuwm_preprocess_cpu.so" in status.refusal()
    finally:
        bridges._CRATE_INDEX.clear()


def test_a_binary_whose_own_name_begins_with_lib_keeps_it(
        tmp_path, monkeypatch):
    """Strip only when the stripped spelling is the declared one.

    The prefix comes off because the workspace answers to the stripped
    name, never because the characters are there.
    """
    root = tmp_path / "checkout"
    workspace = root / "tools" / "rustwx"
    _write(workspace / "Cargo.toml", "[workspace]\nmembers = [\"crates/*\"]\n")
    _write(workspace / "crates" / "libra" / "Cargo.toml",
           MANIFEST.format(name="libra", binary="libra", deps="", dev=""))
    _write(workspace / "crates" / "libra" / "src" / "main.rs", "fn main() {}\n")
    binary = workspace / "target" / "release" / bridges.executable_name("libra")
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x7fELF")
    _touch(binary, 2_000_000_000)
    monkeypatch.setattr(bridges, "_package_parent", lambda: root)
    bridges._CRATE_INDEX.clear()
    try:
        status = bridges.checkout_build_status(binary)
        assert status is not None
        assert status.artifact == "libra"
    finally:
        bridges._CRATE_INDEX.clear()
