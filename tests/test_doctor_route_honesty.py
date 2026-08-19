"""``gpuwm doctor`` may not report a component ``ok`` it never exercised.

Two findings from the 2026-08-14 reachability audit share one root:

* ``grib2_inventory`` and ``grib2_dump`` are staged by ``gpuwm
  fetch-bridges``, probe-executed by doctor, and printed ``ok`` -- on a
  box where ``gpuwm-mapped-inspect`` dies with ``NotADirectoryError:
  [WinError 267]`` because ``mapped_source._build_grib2_tools()`` shells
  ``cargo build`` into a directory a wheel does not contain and never
  looks at the staged copies.  Two true statements about the FILES, and
  a false one about the ROUTE.
* ``rw_mrms``, ``rw_stage4`` and ``rw_asos`` are resolved and refused by
  ``gpuwm/obs/frontdoor.py`` and are in NO bundle, so doctor -- which
  audited exactly the bundled set -- printed a fully green estate on a
  box where all three observation front doors were absent.

The rule these tests hold: where a check cannot exercise something, it
says ``untested`` and its detail opens with "not tested".  Never ``ok``,
and never silence.
"""

from __future__ import annotations

import subprocess

import pytest

from gpuwm import bridges, doctor


def _wheel_layout(monkeypatch, tmp_path, *, staged: bool):
    """A wheel install: no crate anywhere, bridges under ~/.gpuwm.

    ``staged`` decides whether ``gpuwm fetch-bridges`` has already run,
    which is the configuration difference the audit measured: staged
    binaries plus a green doctor plus a route that cannot start.
    """

    bridge_dir = tmp_path / "userdir"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    for variable in bridges.BRIDGE_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(bridges, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(bridges, "_package_parent",
                        lambda: tmp_path / "site-packages")
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: bridge_dir)
    monkeypatch.setattr(doctor, "_grib2_route_crate",
                        lambda: tmp_path / "site-packages" / "tools"
                        / "grib1_bridge")
    if staged:
        for name in doctor._GRIB2_ROUTE_TOOLS:
            # The real contract markers, because the route check applies
            # the resolver's static ABI gate: a fake that lacks them is
            # a STALE staged pair, which is its own test below.
            (bridge_dir / bridges.executable_name(name)).write_bytes(
                b"MZ\0\0" + bridges.BRIDGE_ABI_MARKERS[name])
        monkeypatch.setattr(
            bridges, "find_bridge",
            lambda name: bridge_dir / bridges.executable_name(name))
    else:
        monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    return bridge_dir


# ---------------------------------------------------------------------------
# H6 / H18: staged, probe-green, and unreachable by the route that needs them
# ---------------------------------------------------------------------------

def test_staged_grib2_tools_are_reported_reachable_now_that_they_are(
        monkeypatch, tmp_path):
    """The exact configuration the audit measured, and its verdict TODAY.

    Through 2.3.3 this asserted ``missing``: the tools were staged, the
    files probed green, and ``mapped_source._build_grib2_tools()`` shelled
    cargo into a directory a wheel does not contain, so the route died
    before reading a byte.  That function consults the resolution ladder
    first now, so the staged copies are exactly what the routes that
    still launch these executables use -- and a report calling that a gap
    would be the same defect pointed the other way: a red line over a
    working route teaches a reader to stop believing the report.

    The green is earned rather than assumed: it is asserted through the
    SAME ``find_bridge`` the route calls, on a layout with no crate
    anywhere, which is what a wheel install is.

    The DOORS the line credits move with the code, and the assertions
    below hold them to it: since ``compose`` was declared for grib2 the
    packaged-profile GRIB2 preparations do their byte work inside
    ``gpuwm_mapped_engine`` and name neither tool, so this line may
    enumerate them only as the sources that resolve neither.
    """

    bridge_dir = _wheel_layout(monkeypatch, tmp_path, staged=True)
    check = doctor._mapped_grib2_route_check()

    assert check.status == "verified", (
        "the mapped route resolves both tools through the ladder now; a "
        "gap here is a report that has fallen behind the code")
    assert str(bridge_dir / bridges.executable_name("grib2_inventory")) \
        in check.detail
    # WHICH doors this line vouches for, and it may not overclaim.  Until
    # the compose port it could say "every packaged-profile GRIB2
    # source", because every mapped prep ran the Python engine and
    # launched these two executables.  `compose` is declared for grib2
    # now, so a bare `gpuwm prep --source hrrr-prs` composes in
    # `gpuwm_mapped_engine` and names neither tool -- and a report still
    # crediting these tools with that source's decode would tell a
    # reader their bytes are decoded somewhere they are not, which is
    # this file's whole subject pointed at a newer hole.
    for door in ("gpuwm adapt", "gpuwm-member-prep",
                 "--mapped-engine python"):
        assert door in check.detail, door
    # The 20CRv3 member route left the credited list when it moved onto
    # `decode_composed_source`: a bare member prep composes in the engine
    # and reads the raw record inventory in process, so a report still
    # crediting these tools with that door's decode would tell a reader
    # their bytes are decoded somewhere they are not.
    assert "gpuwm prep --source 20crv3 route" not in check.detail
    assert "20crv3" in check.detail, (
        "the member route must still be named -- as a source that "
        "resolves neither, or a reader is left to guess")
    # The packaged profiles are still enumerated -- a reader asks "does
    # MY source need these?" -- but as the sources that resolve NEITHER.
    packaged = doctor._packaged_grib2_source_ids()
    assert "hrrr-prs" in packaged, (
        "the packaged GRIB2 profile enumeration went empty, so this row "
        "would pass while the report named no source at all")
    for source in packaged:
        assert source in check.detail, source
    assert "compose in the engine on a bare run and resolve neither" \
        in check.detail
    assert doctor.blocking_gaps([check]) == []


def test_a_stale_staged_pair_is_not_reported_as_resolving(
        monkeypatch, tmp_path):
    """The resolver refuses a pre-contract binary; the report must too.

    ``_build_grib2_tools`` gates every ladder hit on the static contract
    marker now, so a staged pair predating the converged decoder is a
    pair the route will NOT launch.  A ``verified`` line over it would
    be the original audit finding again, one rung further up.
    """

    bridge_dir = _wheel_layout(monkeypatch, tmp_path, staged=True)
    for name in doctor._GRIB2_ROUTE_TOOLS:
        (bridge_dir / bridges.executable_name(name)).write_bytes(
            b"MZ\0\0predates-the-contract")
    check = doctor._mapped_grib2_route_check()

    assert check.status == "missing"
    assert "predates" in check.detail
    assert check.action == "gpuwm fetch-bridges"


def test_a_stale_staged_pair_in_a_checkout_reads_untested(
        monkeypatch, tmp_path):
    """With a crate present, the route rebuilds; doctor cannot judge it."""

    crate = tmp_path / "site-packages" / "tools" / "grib1_bridge"
    crate.mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    bridge_dir = _wheel_layout(monkeypatch, tmp_path, staged=True)
    for name in doctor._GRIB2_ROUTE_TOOLS:
        (bridge_dir / bridges.executable_name(name)).write_bytes(
            b"MZ\0\0predates-the-contract")
    check = doctor._mapped_grib2_route_check()

    assert check.status == "untested"
    assert check.detail.startswith("not tested")
    assert "predates" in check.detail


def test_the_same_route_reports_not_tested_where_a_build_could_succeed(
        monkeypatch, tmp_path):
    """The other direction: a checkout, where doctor must NOT claim.

    Doctor never runs cargo (its own docstring is the contract), so on
    a tree that has the crate it cannot know whether the build works.
    The one thing it may not do is fall back to ``ok``.
    """

    crate = tmp_path / "site-packages" / "tools" / "grib1_bridge"
    crate.mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    # staged=False: with the tools staged the ladder answers and the
    # cargo fallback is never reached, so this arm has to be the one
    # where the ladder finds nothing.  That is the only configuration in
    # which the build is what decides, and the only one doctor cannot
    # judge without running cargo.
    _wheel_layout(monkeypatch, tmp_path, staged=False)
    check = doctor._mapped_grib2_route_check()

    assert check.status == "untested"
    assert check.detail.startswith("not tested")
    assert check.status != "verified"


def test_an_unstaged_route_names_fetch_bridges_and_then_the_flags(
        monkeypatch, tmp_path):
    _wheel_layout(monkeypatch, tmp_path, staged=False)
    check = doctor._mapped_grib2_route_check()
    assert check.status == "missing"
    assert check.action == "gpuwm fetch-bridges"
    # Staging IS enough now, and the remedy must not pretend otherwise:
    # the second half it used to carry -- "then pass them by path" -- was
    # correct in 2.3.3 and is busywork today.
    assert check.remedy.lstrip().startswith("gpuwm fetch-bridges")
    assert "does not consult the staged copies" not in check.remedy
    assert "override" in check.remedy


def test_the_crate_doctor_names_is_the_one_the_mapped_route_builds_in(
        monkeypatch):
    """Bind the two spellings of one path across a lane boundary.

    ``mapped_source._build_grib2_tools`` computes its cargo working
    directory inline.  Doctor computes the same path from its own
    ``__file__`` (the two modules live in one package directory, so the
    expressions are equal by construction).  "By construction" is a
    claim, and this is the test that makes it one that can fail: it
    captures the ``cwd`` that function would actually hand to cargo.

    ``find_bridge`` is stubbed to find nothing on purpose.  That function
    now consults the resolution ladder BEFORE cargo -- the cross-lane fix
    that stopped a wheel install dying on a NotADirectoryError while
    ``grib2_inventory`` sat staged and pin-valid -- so on any box with a
    staged bundle it returns early and cargo is never reached.  Stubbing
    the ladder is what keeps this measuring the cargo branch instead of
    silently measuring nothing.
    """

    mapped_source = pytest.importorskip("gpuwm.mapped_source")
    from gpuwm import bridges as bridges_module

    captured = {}

    class _Completed:
        returncode = 1
        stderr = "cargo did not run (test double)"
        stdout = ""

    def _capture(command, cwd=None, **kwargs):
        captured["cwd"] = cwd
        return _Completed()

    monkeypatch.setattr(bridges_module, "find_bridge", lambda name: None)
    monkeypatch.setattr(mapped_source.subprocess, "run", _capture)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        mapped_source._build_grib2_tools()
    if "cwd" not in captured:
        pytest.skip("this install has no tools/grib1_bridge crate, so the "
                    "cargo branch does not exist here")
    assert captured["cwd"] == doctor._grib2_route_crate()


def test_the_bridge_lines_say_the_mapped_route_resolves_them_again():
    """The consumer label, corrected twice, and this is the second time.

    "20CRv3/mapped GRIB2 routes" was misleading in 2.3.3 because it read
    as "this staged file is what those routes use" while they built their
    own.  The doctor lane replaced it with "takes it by explicit path",
    which is now misleading in the other direction: the route resolves it
    from here, and the flags are overrides.  A label is not allowed to be
    a fossil of either fix.
    """

    for tool in doctor._GRIB2_ROUTE_TOOLS:
        consumer = doctor._BRIDGE_CONSUMERS[tool]
        assert "resolve it from here" in consumer or \
            "resolve" in consumer and "from here" in consumer
        assert "rather than from here" not in consumer
        assert "overrides" in consumer


# ---------------------------------------------------------------------------
# H16: the three front doors no bundle carries
# ---------------------------------------------------------------------------

def test_the_obs_front_doors_are_reported_when_absent(monkeypatch, tmp_path):
    """A green estate over three dead observation routes, ended."""

    frontdoor = pytest.importorskip("gpuwm.obs.frontdoor")
    for door in frontdoor.FRONT_DOORS.values():
        monkeypatch.delenv(door.env_var, raising=False)
    monkeypatch.setattr(frontdoor, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(frontdoor, "_repo_root", lambda: tmp_path / "pkg")
    monkeypatch.setattr(frontdoor, "default_bridge_dir",
                        lambda: tmp_path / "userdir")

    checks = {check.name: check for check in doctor._obs_front_door_checks()}
    assert len(checks) == len(frontdoor.FRONT_DOORS) == 6
    for name, check in checks.items():
        assert check.status == "missing", name
        assert check.severity == doctor.SEVERITY_UNREACHABLE
        # THE remedy, now that every obs front door is a bundled
        # artifact: one command that can actually supply it.  Through
        # 2.3.3 this asserted the opposite, because the resolver led with
        # `gpuwm fetch-bridges` while the bundle carried none of these --
        # so a reader ran it, was told every pinned artifact was staged,
        # and met the identical refusal.  The line was never hard-coded
        # either way: it is a function of the bundle manifest, and the
        # manifest is what changed.
        assert check.remedy.lstrip().startswith("gpuwm fetch-bridges")
        assert "not a bundled artifact" not in check.remedy
        assert "IS in this release's prebuilt bundle" in check.detail
    # Every door names the product subcommand a reader can type after
    # `pip install gpuwm`, which is the door that did not exist when
    # these checks were written.
    for instrument in ("mrms", "stage4", "asos", "goes", "opera", "odim"):
        assert any(f"gpuwm obs {instrument}" in check.detail
                   for check in checks.values()), instrument
    assert any("tools.obs_fetch_mrms" in check.detail
               for check in checks.values())
    assert any("tools.obs_fetch_stage4" in check.detail
               for check in checks.values())
    assert any("tools.obs_fetch_asos" in check.detail
               for check in checks.values())


def test_an_obs_front_door_that_is_present_and_current_is_verified(
        monkeypatch, tmp_path):
    """The other direction: the check must go quiet on a good box."""

    frontdoor = pytest.importorskip("gpuwm.obs.frontdoor")
    staged = tmp_path / "userdir"
    staged.mkdir()
    for door in frontdoor.FRONT_DOORS.values():
        monkeypatch.delenv(door.env_var, raising=False)
        (staged / f"{door.name}.exe").write_bytes(b"MZ\0\0")
        (staged / door.name).write_bytes(b"MZ\0\0")
    monkeypatch.setattr(frontdoor, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(frontdoor, "_repo_root", lambda: tmp_path / "pkg")
    monkeypatch.setattr(frontdoor, "default_bridge_dir", lambda: staged)
    monkeypatch.setattr(frontdoor.FrontDoor, "probe",
                        lambda self, path: (True, "rw 0.1.0 -- --abi matches"))

    checks = doctor._obs_front_door_checks()
    assert len(checks) == len(frontdoor.FRONT_DOORS) == 6
    assert all(check.status == "verified" for check in checks)
    assert doctor.blocking_gaps(checks) == []


def test_the_obs_front_doors_are_in_the_bundle_doctor_audits():
    """The premise the remedy rests on, asserted rather than assumed.

    Through 2.3.3 this asserted the opposite, and that WAS the finding:
    every obs front door was resolved by name, refused by name, and
    carried by no bundle, so the command the refusal named could not
    supply it.  Both halves landed in 2.4.1 -- the artifacts joined the
    table, and the crates gained the source-revision stamp a release cut
    needs in order to pin them -- so the honest premise is now this one.
    If a door ever leaves the bundle, this fails and the remedy gets
    revisited deliberately rather than going quietly wrong.
    """

    frontdoor = pytest.importorskip("gpuwm.obs.frontdoor")
    from gpuwm.bridge_assets import BUNDLED_ARTIFACTS

    bundled = {artifact.name for artifact in BUNDLED_ARTIFACTS}
    for door in frontdoor.FRONT_DOORS.values():
        assert door.name in bundled, door.name


# ---------------------------------------------------------------------------
# The sweep that stops the count drifting a third time
# ---------------------------------------------------------------------------

def test_every_bundled_artifact_is_covered_by_a_check():
    """The estate's completeness must not depend on someone remembering.

    ``rw_nexrad`` shipped outside both audited sets and doctor passed on
    boxes where every radar route was dead.  The three observation front
    doors did it again.  This asserts the map is complete TODAY; the
    sweep in doctor reports anything it is not, so a fourteenth artifact
    is loud rather than invisible either way.
    """

    from gpuwm.bridge_assets import BUNDLED_ARTIFACTS

    uncovered = sorted(artifact.name for artifact in BUNDLED_ARTIFACTS
                       if artifact.name not in doctor._CHECKED_ARTIFACTS)
    assert not uncovered, (
        "these bundled artifacts have no doctor check; give them one and "
        f"add them to _CHECKED_ARTIFACTS: {uncovered}")

    checks = doctor._bundle_coverage_checks()
    assert len(checks) == 1
    assert checks[0].status == "verified"
    assert f"all {len(BUNDLED_ARTIFACTS)} artifact(s)" in checks[0].detail


def test_an_artifact_added_to_the_bundle_is_reported_not_ignored(
        monkeypatch, tmp_path):
    """The other direction, and the one that matters for the future.

    A bundle that grows an artifact doctor has never heard of must not
    produce silence -- and must not produce ``ok`` either, since doctor
    knows no contract to probe it against.
    """

    from gpuwm import bridge_assets

    newcomer = bridge_assets.BundledArtifact(
        "rw_invented", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_INVENTED", "a route added after doctor was written")
    monkeypatch.setattr(
        bridge_assets, "BUNDLED_ARTIFACTS",
        tuple(bridge_assets.BUNDLED_ARTIFACTS) + (newcomer,))

    # Not staged: a gap with the one command that stages it.
    monkeypatch.setattr(bridges, "find_artifact", lambda env, filename: None)
    absent = [check for check in doctor._bundle_coverage_checks()
              if "rw_invented" in check.name]
    assert len(absent) == 1
    assert absent[0].status == "missing"
    assert absent[0].action == "gpuwm fetch-bridges"

    # Staged: present, unprobed, and saying exactly that.
    staged = tmp_path / "rw_invented.exe"
    staged.write_bytes(b"MZ\0\0")
    monkeypatch.setattr(bridges, "find_artifact",
                        lambda env, filename: staged)
    present = [check for check in doctor._bundle_coverage_checks()
               if "rw_invented" in check.name]
    assert len(present) == 1
    assert present[0].status == "untested"
    assert present[0].detail.startswith("not tested")
    assert present[0].status != "verified"


def test_an_obs_front_door_that_the_bundle_carries_gets_the_one_command(
        monkeypatch, tmp_path):
    """The remedy has to follow the bundle, not a hard-coded assumption.

    Today no bundle carries these, so the remedy is a clone and a build.
    The moment one does, `gpuwm fetch-bridges` becomes the true answer
    and this line has to say so without an edit.
    """

    frontdoor = pytest.importorskip("gpuwm.obs.frontdoor")
    from gpuwm import bridge_assets

    for door in frontdoor.FRONT_DOORS.values():
        monkeypatch.delenv(door.env_var, raising=False)
    monkeypatch.setattr(frontdoor, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(frontdoor, "_repo_root", lambda: tmp_path / "pkg")
    monkeypatch.setattr(frontdoor, "default_bridge_dir",
                        lambda: tmp_path / "userdir")

    bundled = tuple(bridge_assets.BUNDLED_ARTIFACTS) + tuple(
        bridge_assets.BundledArtifact(
            door.name, "executable", bridges.RUSTWX_CRATE_RELATIVE,
            door.env_var, door.subject)
        for door in frontdoor.FRONT_DOORS.values())
    monkeypatch.setattr(bridge_assets, "BUNDLED_ARTIFACTS", bundled)

    checks = doctor._obs_front_door_checks()
    assert len(checks) == len(frontdoor.FRONT_DOORS) == 6
    for check in checks:
        assert check.status == "missing"
        assert check.action == "gpuwm fetch-bridges"
        assert "IS in this release's prebuilt bundle" in check.detail
        assert "not a bundled artifact" not in (check.remedy or "")


def test_an_obs_front_door_the_bundle_lacks_still_gets_the_build_route(
        monkeypatch, tmp_path):
    """The branch no shipped door takes today, kept proven for the next one.

    Every obs front door is bundled as of 2.4.1, so the live report never
    reaches the clone-and-build remedy.  A branch nothing exercises is a
    branch that rots, and the next front door to be written will land on
    it before it lands in a bundle -- which is exactly the state that
    produced this finding the first time.
    """

    frontdoor = pytest.importorskip("gpuwm.obs.frontdoor")
    from gpuwm import bridge_assets

    for door in frontdoor.FRONT_DOORS.values():
        monkeypatch.delenv(door.env_var, raising=False)
    monkeypatch.setattr(frontdoor, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(frontdoor, "_repo_root", lambda: tmp_path / "pkg")
    monkeypatch.setattr(frontdoor, "default_bridge_dir",
                        lambda: tmp_path / "userdir")

    doors = {door.name for door in frontdoor.FRONT_DOORS.values()}
    monkeypatch.setattr(bridge_assets, "BUNDLED_ARTIFACTS", tuple(
        artifact for artifact in bridge_assets.BUNDLED_ARTIFACTS
        if artifact.name not in doors))

    checks = doctor._obs_front_door_checks()
    assert len(checks) == len(frontdoor.FRONT_DOORS) == 6
    for check in checks:
        assert check.status == "missing"
        assert check.action == doctor._OBS_FRONT_DOOR_ACTION
        assert "not a bundled artifact" in check.remedy
        assert not check.remedy.lstrip().startswith("gpuwm fetch-bridges")


# ---------------------------------------------------------------------------
# The status contract itself
# ---------------------------------------------------------------------------

def test_untested_never_reads_as_ok_and_always_says_not_tested():
    """One vocabulary, in both report layers."""

    assert "untested" in doctor._LABELS
    assert doctor._LABELS["untested"] != doctor._LABELS["verified"]
    check = doctor.Check("x", "untested", "not tested -- because reasons",
                         brief="not tested")
    text = doctor.format_report([check])
    assert "UNTESTED" in text
    assert "ok " not in text.splitlines()[1]
    for finding in doctor.collect_checks():
        if finding.status == "untested":
            assert finding.detail.startswith("not tested"), finding.name


def test_a_probe_execution_still_verifies_a_real_bridge(monkeypatch,
                                                        tmp_path):
    """The other direction for the bridge lines themselves.

    Downgrading the ROUTE's claim must not have downgraded the FILE's:
    a bridge that launches and matches its ABI is still ``verified``.
    """

    bridge = tmp_path / bridges.executable_name("gfs_grib2_bridge")
    bridge.write_bytes(b"MZ\0\0")
    monkeypatch.setattr(bridges, "find_bridge", lambda name: bridge)
    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "PE image"))
    monkeypatch.setattr(
        bridges, "bridge_abi_matches",
        lambda name, path: (True, "speaks this release's contract"))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, "gfs_grib2_bridge 1.0", ""))
    checks = {check.name: check for check in doctor._bridge_checks()}
    assert checks["bridge gfs_grib2_bridge"].status == "verified"
