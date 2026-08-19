"""The Rust workspaces' battery list, kept in sync with the workspaces.

``tools/battery/cargo_gates.txt`` is how the release battery finds the
Cargo test suites.  Drew's 2026-08-12 ruling ("probably we would want
tests right") made them a leg; this suite is the gate on the list, the
same kind of gate ``tests/test_stage1_manifest.py`` is on the stage-1
list and ``tests/test_tiles_gate_manifest.py`` is on the ``[tiles]``
gates.

It carries one property those two do not, and it is the point of the
file: the list is checked against the workspace manifests in BOTH
directions.  Every entry has to name a real member package, and every
member package has to have an entry.  The second direction is the
accepted-implies-builds pattern (``tests/test_mp_accepted_builds.py``)
applied to a battery list rather than to a physics menu: a crate that
joins ``tools/rustwx/Cargo.toml``'s ``members`` turns THIS suite red
until somebody writes its line in the list, so a new crate cannot land
already ungated.  That is how a crate arriving from another lane -- the
Rust MPAS work is the live example -- joins the leg, and it needs no
edit here to happen.

Cheap and hermetic on purpose: it reads three text files and two
``Cargo.toml``s, needs no Rust toolchain and no card, so it runs on
every commit rather than at a cut.  What it CANNOT check is that the
suites pass -- that is the leg itself,
``tools/battery/run_cargo_gates.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tomllib

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "cargo_gates.txt"
RUNNER = REPOSITORY_ROOT / "tools" / "battery" / "run_cargo_gates.py"


_RUNNER_MODULE = "battery_run_cargo_gates"


def _runner():
    """Import the runner by path -- ``tools`` is not an importable package.

    Importing it rather than restating its parser is deliberate: a file
    this suite accepts has to be a file the runner can read, and the
    only way to keep those two statements from drifting is to have one
    of them.

    It is registered in ``sys.modules`` before execution because
    ``@dataclass`` resolves annotations through
    ``sys.modules[cls.__module__]``, and a module that executes without
    being registered has none.
    """

    if _RUNNER_MODULE in sys.modules:
        return sys.modules[_RUNNER_MODULE]
    spec = importlib.util.spec_from_file_location(_RUNNER_MODULE, RUNNER)
    assert spec and spec.loader, f"{RUNNER} is not importable"
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUNNER_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _entries():
    return _runner().read_manifest(MANIFEST)


def _lines() -> list[str]:
    return MANIFEST.read_text(encoding="utf-8").splitlines()


def _reasons() -> dict[str, str]:
    """Each entry line mapped to the comment block above its group.

    A blank line ends a block, so the file header does not attach itself
    to the first entry; a block covers the whole contiguous run beneath
    it, so a package split across two entries by target can share one
    rationale.
    """

    reasons: dict[str, str] = {}
    block: list[str] = []
    for raw in _lines():
        line = raw.strip()
        if not line:
            block = []
        elif line.startswith("#"):
            block.append(line.lstrip("#").strip())
        else:
            reasons[" ".join(line.split())] = " ".join(p for p in block if p)
    return reasons


def _path_dependency_members(root: Path, manifest: dict) -> set[str]:
    """Path dependencies that live inside the workspace directory.

    Cargo makes these workspace members automatically -- "all path
    dependencies residing in the workspace directory become members" --
    which is why ``cargo test -p grib-core`` works from
    ``tools/grib1_bridge`` even though nothing lists it.  The gate has to
    know that, or a crate the release ships and the whole tree decodes
    GRIB through is invisible to the both-directions check.
    """

    names = set()
    for table in ("dependencies", "dev-dependencies", "build-dependencies"):
        for spec in manifest.get(table, {}).values():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            candidate = (root / spec["path"]).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue  # outside the workspace directory: not a member
            member_toml = candidate / "Cargo.toml"
            if member_toml.is_file():
                names.add(tomllib.loads(
                    member_toml.read_text(encoding="utf-8"))["package"]["name"])
    return names


def _members(workspace: str) -> set[str]:
    """The package names a workspace directory declares.

    A ``[workspace]`` table means the members' own manifests carry the
    names; a bare ``[package]`` (``tools/grib1_bridge``) is a one-package
    workspace, is its own member, and additionally owns every path
    dependency vendored beneath it.
    """

    root = REPOSITORY_ROOT / workspace
    manifest = tomllib.loads(
        (root / "Cargo.toml").read_text(encoding="utf-8"))
    if "workspace" not in manifest:
        return {manifest["package"]["name"]} | _path_dependency_members(
            root, manifest)
    names = set()
    for member in manifest["workspace"].get("members", []):
        member_toml = root / member / "Cargo.toml"
        # A member cargo cannot open is a broken workspace, and it would
        # otherwise surface here as a bare FileNotFoundError during
        # collection -- an error that reads like this suite is broken
        # rather than like the tree is.
        assert member_toml.is_file(), (
            f"{workspace}/Cargo.toml lists member {member!r} but "
            f"{member}/Cargo.toml does not exist; cargo refuses the whole "
            "workspace in that state, so the cargo leg would fail on every "
            "entry")
        names.add(tomllib.loads(
            member_toml.read_text(encoding="utf-8"))["package"]["name"])
    return names


def _workspaces() -> list[str]:
    seen: list[str] = []
    for entry in _entries():
        if entry.workspace not in seen:
            seen.append(entry.workspace)
    return seen


def test_the_manifest_exists_and_lists_something() -> None:
    assert MANIFEST.is_file(), f"{MANIFEST} is missing"
    assert RUNNER.is_file(), f"{RUNNER} is missing"
    assert _entries(), (
        "tools/battery/cargo_gates.txt parsed to zero entries; the battery "
        "would compile no Rust and still report a clean leg")


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e.label)
def test_every_entry_names_a_real_workspace_and_package(entry) -> None:
    """Listed implies exists: the first direction of the sync.

    An entry naming a package cargo does not know fails the WHOLE leg at
    the first invocation, because ``cargo test -p <name>`` refuses
    rather than skipping.  Catching it here costs no toolchain.
    """

    root = REPOSITORY_ROOT / entry.workspace
    assert (root / "Cargo.toml").is_file(), (
        f"{entry.workspace}/Cargo.toml does not exist")
    assert (root / ".cargo" / "config.toml").is_file(), (
        f"{entry.workspace}/.cargo/config.toml does not exist, so "
        "`cargo test --offline` has no vendored source to resolve against; "
        "the leg would fail on a box with no crates.io cache and pass on "
        "one that had it")
    members = _members(entry.workspace)
    assert entry.package in members, (
        f"{entry.package} is listed in tools/battery/cargo_gates.txt but is "
        f"not a member of {entry.workspace}: {sorted(members)}")


@pytest.mark.parametrize(
    ("workspace", "package"),
    [(w, p) for w in _workspaces() for p in sorted(_members(w))],
    ids=lambda v: v.replace("/", "-"))
def test_every_workspace_member_is_listed(workspace: str,
                                          package: str) -> None:
    """Exists implies listed: the direction that makes this a gate.

    Without it the list is a snapshot -- correct on the day it was
    written and quietly incomplete afterwards.  A crate added to
    ``members`` is a crate the release ships and nothing compiles, which
    is exactly the state every crate in this tree was in before this
    leg existed.

    Adding the entry is the whole remedy: one line naming its shard, its
    floor and what it covers.  ``tools/battery/run_cargo_gates.py``
    prints the count to use as the floor.
    """

    listed = {(e.workspace, e.package) for e in _entries()}
    assert (workspace, package) in listed, (
        f"{package} is a member of {workspace} but has no entry in "
        "tools/battery/cargo_gates.txt, so the release battery never "
        "compiles or tests it.  Add a line -- SHARD WORKSPACE PACKAGE "
        "MIN_TESTS -- with its reason on the lines above, per the file's "
        "amendment discipline")


def test_no_invocation_is_listed_twice() -> None:
    """Identity is package PLUS cargo arguments.

    One package listed twice under different arguments is two
    invocations covering different targets, which is what the argument
    tail is for; the same package under the same arguments is a doubled
    row whose counts would overstate the coverage.
    """

    invocations = [(e.workspace, e.package, e.args) for e in _entries()]
    duplicates = sorted({i for i in invocations if invocations.count(i) > 1})
    assert not duplicates, (
        f"tools/battery/cargo_gates.txt lists {duplicates} more than once")


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e.label)
def test_every_entry_carries_its_reason_inline(entry) -> None:
    line = " ".join((entry.shard, entry.workspace, entry.package,
                     str(entry.min_tests)) + entry.env + entry.args)
    assert _reasons().get(line, ""), (
        f"{entry.label} has no comment block above it in "
        "tools/battery/cargo_gates.txt; the amendment discipline the file "
        "states is that an addition names what the package covers")


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e.label)
def test_every_floor_is_a_real_floor(entry) -> None:
    """``MIN_TESTS`` of zero would make the contract vacuous.

    ``cargo test`` on a package whose test modules stopped compiling
    under some ``#[cfg]`` exits 0 and reports ``0 passed``.  The floor is
    the only part of the contract that can tell those apart, so a floor
    that any run satisfies is the same as having none.  The parser
    refuses it; this is the statement of why.
    """

    assert entry.min_tests >= 1


def test_the_battery_shard_covers_both_workspaces() -> None:
    """A leg that compiled only the renderer would prove half of it.

    The two workspaces fail in opposite directions: tools/rustwx is
    everything model-output-to-pixels, tools/grib1_bridge is everything
    data-to-model-input.  A release ships binaries from both.
    """

    runner = _runner()
    shard = {e.workspace for e in _entries()
             if e.shard == runner.BATTERY_SHARD}
    for required in ("tools/rustwx", "tools/grib1_bridge"):
        assert required in shard, (
            f"no {runner.BATTERY_SHARD}-shard entry for {required} in "
            "tools/battery/cargo_gates.txt; that workspace ships in the "
            "release and the battery would not compile it")


def test_the_decoder_and_the_batch_renderer_are_both_on_the_battery_shard(
) -> None:
    """The two packages whose defects reach a user as wrong output.

    ``grib1_bridge`` decodes every GFS and HRRR field a run initialises
    from -- a quantization defect there is a wrong initial condition, not
    a crash.  ``rw-wrfbatch`` draws every product PNG a run publishes and
    is the one crate in tools/rustwx whose upstream is this project.
    Pinned by name so a shard reshuffle has to argue with this docstring
    rather than move a line.
    """

    runner = _runner()
    battery = {e.package for e in _entries()
               if e.shard == runner.BATTERY_SHARD}
    for required in ("grib1_bridge", "rw-wrfbatch"):
        assert required in battery, (
            f"{required} is not on the {runner.BATTERY_SHARD} shard of "
            "tools/battery/cargo_gates.txt")


def test_the_fixture_dependent_satellite_suite_is_still_listed() -> None:
    """Deferred coverage stays visible, or it is not deferred.

    ``rw-sat``'s ``tests/cloud_products.rs`` decodes real GOES-19
    granules the repository does not carry and PANICS by name when they
    are absent, so it cannot ride the offline leg.  The honest form of
    that is two entries -- ``--lib`` on the battery shard and the
    integration target on ``fixtures`` -- and the dishonest form is one
    ``--lib`` entry with nothing saying what was left out.  This test
    pins the honest form: dropping the fixtures row is allowed, but it
    has to be an argument in a commit message, not a line that
    disappears while every gate stays green.
    """

    rows = {(e.shard, e.package, e.args) for e in _entries()}
    assert ("fixtures", "rw-sat", ("--test", "cloud_products")) in rows, (
        "the fixtures-shard row for rw-sat's cloud_products suite is gone "
        "from tools/battery/cargo_gates.txt.  The battery shard runs rw-sat "
        "with --lib, so without that row nothing in the tree records that "
        "the GOES L2 cloud-product decode has coverage at all")
    battery = {(e.package, e.args) for e in _entries() if e.shard == "cpu"}
    assert ("rw-sat", ("--lib",)) in battery, (
        "rw-sat's --lib row is gone from the battery shard, so the offline "
        "half of the satellite crate is no longer compiled by the leg")


def test_the_file_is_lf_only() -> None:
    """``.gitattributes`` says ``* -text``: disk bytes are committed bytes."""

    assert b"\r" not in MANIFEST.read_bytes(), (
        "tools/battery/cargo_gates.txt contains CR bytes; the repository "
        "commits LF and .gitattributes does no conversion")


def test_the_runner_refuses_an_in_tree_target_directory(monkeypatch) -> None:
    """The mandate, exercised rather than asserted in a comment.

    A cut hashes the worktree-local renderer at
    ``tools/rustwx/target/release`` before it believes any leg that
    renders, and refuses to assemble from a dirty source tree.  A test
    leg sharing that directory can replace the binary the cut is
    adjudicating.  Both refusals are checked here -- unset, and set
    inside the tree -- because a default would have hidden the mistake
    instead of stopping it.
    """

    runner = _runner()
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    with pytest.raises(RuntimeError, match="CARGO_TARGET_DIR is not set"):
        runner.resolve_target_dir(None)
    inside = REPOSITORY_ROOT / "tools" / "rustwx" / "target"
    with pytest.raises(RuntimeError, match="inside the repository"):
        runner.resolve_target_dir(str(inside))
    with pytest.raises(RuntimeError, match="inside the repository"):
        runner.resolve_target_dir(str(REPOSITORY_ROOT))


def test_the_runner_builds_offline_locked_invocations(monkeypatch) -> None:
    """``--locked --offline`` is the leg, not a preference.

    Both workspaces vendor their dependency closure with a
    ``.cargo/config.toml`` source replacement.  A leg that resolved from
    the network would pass on a developer box with a warm registry and
    fail on the release box, and would stop proving the vendored closure
    is complete -- which is the only reason a clean clone of this tree
    can build at all.
    """

    for entry in _entries():
        argv = entry.invocation()
        assert argv[:4] == ["cargo", "test", "--locked", "--offline"], argv
        assert argv[4:6] == ["-p", entry.package], argv
        assert argv[6:] == list(entry.args), argv


def test_an_external_target_directory_is_accepted(tmp_path) -> None:
    """The other direction of the refusal, so it is not refuse-everything."""

    runner = _runner()
    resolved = runner.resolve_target_dir(str(tmp_path / "cargo-target"))
    assert resolved == (tmp_path / "cargo-target").resolve()
