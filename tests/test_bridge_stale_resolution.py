"""A staged bridge that is not this release's binary never reaches a door.

The defect these bind, measured on 2026-09-01 against the published
2.6.1: ``~/.gpuwm/bridges`` held ``rw_mpas_mesh.exe`` stamped
``c8883473...`` (the v2.5.3 tag, staged in August), the wheel's own pins
declared 2,679,808 B stamped ``8b9f89f1...``, ``gpuwm doctor`` printed
the mismatch as a BROKEN line -- and every door's resolution ladder went
on handing the August binary to the routes, because nothing between the
ladder and the door read the verdict.  The mesh door's gradient
predicate then came back false against a binary with no gradient meter
in it, and ten gates in the hex battery reported skipped rather than
failed.

What is bound here is the shape of the answer, not one artifact:

* a STAGED artifact whose bytes are not the pinned ones is refreshed
  (default) or refused BY NAME (offline), at the ladder;
* an environment override, an in-tree build and both ``libexec`` rungs
  are never judged, because none of them is the rung a ``pip install
  -U`` leaves behind;
* a matching artifact resolves exactly as before, and touches nothing;
* ``gpuwm doctor`` reports the same verdict the doors act on, and
  reporting it never fetches or refuses.

Nothing here fabricates a hash: every pin is computed from bytes these
tests just wrote, and every refresh stages a real zip built here.
"""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from gpuwm import bridge_assets, bridges

#: A staged artifact carries an embedded revision stamp, which is what
#: lets a refusal say WHICH release the file on disk came from.
_OLD_REV = "c8883473bd64ce3f1776927c6a17c4698f2f766c"
_NEW_REV = "8b9f89f11028fc4e272f27ab5b75006f1ff71efd"

#: A real bundled artifact, because the refusal names its environment
#: variable and its ABI marker is verified before a refreshed copy is
#: installed -- both are properties of the roster, not of a fixture.
_ARTIFACT = "rw_mpas_mesh"


def _payload(rev: str, *, filler: bytes = b"") -> bytes:
    """Bytes that look enough like a bridge to be judged like one."""

    return (b"MZ" + filler
            + bridge_assets.SOURCE_REV_MARKER + rev.encode("ascii")
            + b"\x00" + bridges.BRIDGE_ABI_MARKERS[_ARTIFACT])


def _pin(payload: bytes, filename: str) -> bridge_assets.BinaryPin:
    return bridge_assets.BinaryPin(
        artifact=_ARTIFACT, filename=filename, bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest())


@pytest.fixture
def estate(tmp_path, monkeypatch):
    """A staged directory, a pinned release, and a mirror to refresh from.

    ``home/.gpuwm/bridges`` holds the OLD binary; the pins declare the
    NEW one; ``mirror`` holds a bundle carrying the new bytes, served
    over ``file://`` so the default arm is a real verified download with
    no network in it.
    """

    filename = bridges.executable_name(_ARTIFACT)
    home = tmp_path / "home"
    staged = home / ".gpuwm" / "bridges"
    staged.mkdir(parents=True)
    old = _payload(_OLD_REV)
    (staged / filename).write_bytes(old)

    new = _payload(_NEW_REV, filler=b"\x90" * 64)
    pin = _pin(new, filename)
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr(filename, new)
    body = blob.getvalue()
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    bundle_name = "gpuwm-bridges-v9.9.9-test.zip"
    (mirror / bundle_name).write_bytes(body)

    bundle = bridge_assets.BundlePin(
        platform="win-x86_64", filename=bundle_name, bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(), binaries=(pin,))
    pins = bridge_assets.BridgePins(release="v9.9.9-test",
                                    platforms={bundle.platform: bundle})

    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: bundle.platform)
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: staged)
    monkeypatch.setenv(bridge_assets.ASSET_URL_BASE_ENV,
                       mirror.resolve().as_uri())
    monkeypatch.delenv(bridge_assets.STALE_POLICY_ENV, raising=False)
    monkeypatch.setattr(bridges, "_REFRESH_ATTEMPTED", False)
    monkeypatch.setattr(bridges, "_REFRESH_FAILURE", None)
    monkeypatch.setattr(bridges, "_STALE_ALLOWED", set())
    bridge_assets.forget_pin_memo()
    return {"staged": staged, "path": staged / filename, "pin": pin,
            "old": old, "new": new, "pins": pins, "tmp": tmp_path,
            "filename": filename}


# ---------------------------------------------------------------------------
# The judgement itself
# ---------------------------------------------------------------------------

def test_a_staged_artifact_from_another_release_is_not_a_match(estate):
    status = bridge_assets.staged_pin_status(estate["path"])

    assert status is not None
    assert status.matches is False
    assert status.observed_revision == _OLD_REV
    assert status.release == "v9.9.9-test"
    assert str(estate["path"]) in status.describe()
    assert _OLD_REV in status.describe()


def test_a_staged_artifact_that_is_this_release_matches(estate):
    estate["path"].write_bytes(estate["new"])
    bridge_assets.forget_pin_memo()

    status = bridge_assets.staged_pin_status(estate["path"])

    assert status is not None and status.matches is True


@pytest.mark.parametrize("where", ["override", "in-tree", "libexec"])
def test_only_the_staged_rung_is_judged(estate, tmp_path, where):
    """An override, a dev build and a libexec copy are never stale.

    A cargo build cannot match a release pin by construction, so a rule
    that judged those rungs would refuse the developer path on every
    run.  The stale bytes here are IDENTICAL to the ones refused above;
    only the directory differs.
    """

    elsewhere = tmp_path / where
    elsewhere.mkdir()
    copy = elsewhere / estate["filename"]
    copy.write_bytes(estate["old"])

    assert bridge_assets.staged_pin_status(copy) is None
    assert bridges.require_release_pin(copy) == copy


def test_a_staged_file_this_release_does_not_pin_is_not_judged(estate):
    """A name the pins do not carry has nothing to be measured against."""

    stray = estate["staged"] / "not_a_pinned_artifact.exe"
    stray.write_bytes(estate["old"])

    assert bridge_assets.staged_pin_status(stray) is None
    assert bridges.require_release_pin(stray) == stray


def test_a_source_checkout_declares_no_release_and_judges_nothing(
        estate, monkeypatch):
    """The packaged pins of an unreleased tree say nothing about a rev.

    ``tools/build_bridge_bundle.py pin`` stamps the release at cut time;
    before that the document declares ``release: null`` and no
    platforms, which is what every checkout and every worktree carries.
    """

    monkeypatch.setattr(
        bridge_assets, "load_pins",
        lambda path=None: bridge_assets.BridgePins(release=None,
                                                   platforms={}))

    assert bridge_assets.staged_pin_status(estate["path"]) is None
    assert bridges.require_release_pin(estate["path"]) == estate["path"]


# ---------------------------------------------------------------------------
# What a resolution does about it
# ---------------------------------------------------------------------------

def test_the_default_is_to_refresh_and_carry_on(estate):
    """Fixed means a bare run stops showing the defect.

    No flag, no command: the resolution re-fetches this release's
    bundle through the same verified path ``gpuwm fetch-bridges``
    walks, and hands the door the pinned bytes.
    """

    resolved = bridges.require_release_pin(estate["path"])

    assert resolved == estate["path"]
    assert estate["path"].read_bytes() == estate["new"]
    assert bridge_assets.matches_pin(estate["path"], estate["pin"])


def test_offline_the_same_resolution_refuses_by_name(estate, monkeypatch):
    """The offline arm names the file, both revisions, and the remedy."""

    monkeypatch.setenv(bridge_assets.ASSET_URL_BASE_ENV,
                       "https://127.0.0.1:9/unreachable")

    with pytest.raises(bridges.StaleBridgeError) as raised:
        bridges.require_release_pin(estate["path"])

    message = str(raised.value)
    assert str(estate["path"]) in message
    assert _OLD_REV in message                  # what is on disk
    assert "v9.9.9-test" in message             # what should be
    assert "gpuwm fetch-bridges" in message     # the one command
    assert "GPUWM_RW_MPAS_MESH" in message      # the override that wins
    # The bytes are left alone: a refusal is not a deletion.
    assert estate["path"].read_bytes() == estate["old"]


def test_the_refusal_is_not_a_missing_file(estate, monkeypatch):
    """`StaleBridgeError` is not a `FileNotFoundError`, deliberately.

    Doors fall back or offer a build one-liner when an artifact is
    absent.  This one is present and wrong, and a fallback is the
    silent degradation the refusal exists to prevent.
    """

    monkeypatch.setenv(bridge_assets.ASSET_URL_BASE_ENV,
                       "https://127.0.0.1:9/unreachable")

    with pytest.raises(bridges.StaleBridgeError):
        bridges.require_release_pin(estate["path"])
    assert not issubclass(bridges.StaleBridgeError, FileNotFoundError)


def test_the_network_is_tried_once_per_run(estate, monkeypatch):
    """Twenty-six stale artifacts must not mean twenty-six downloads."""

    attempts = []
    real = bridge_assets.refresh_staged_bundle

    def counted(**kwargs):
        attempts.append(kwargs)
        raise bridge_assets.BridgeAssetError("no route to the release")

    monkeypatch.setattr(bridge_assets, "refresh_staged_bundle", counted)
    for _ in range(3):
        with pytest.raises(bridges.StaleBridgeError):
            bridges.require_release_pin(estate["path"])

    assert len(attempts) == 1
    assert real is not counted


def test_refuse_never_touches_the_network(estate, monkeypatch):
    """`refuse` is the arm for a box that must not dial out."""

    def forbidden(**kwargs):
        raise AssertionError("refuse must not fetch")

    monkeypatch.setattr(bridge_assets, "refresh_staged_bundle", forbidden)
    monkeypatch.setenv(bridge_assets.STALE_POLICY_ENV, "refuse")

    with pytest.raises(bridges.StaleBridgeError) as raised:
        bridges.require_release_pin(estate["path"])
    assert "gpuwm fetch-bridges" in str(raised.value)


def test_allow_runs_the_staged_file_and_reports_it_as_a_workaround(
        estate, monkeypatch):
    warnings = []
    monkeypatch.setenv(bridge_assets.STALE_POLICY_ENV, "allow")
    monkeypatch.setattr("gpuwm.explain.warn",
                        lambda action, why="": warnings.append(action))

    resolved = bridges.require_release_pin(estate["path"])

    assert resolved == estate["path"]
    assert estate["path"].read_bytes() == estate["old"]
    assert warnings and _OLD_REV in warnings[0]


def test_an_unknown_policy_is_reported_not_obeyed(estate, monkeypatch):
    warnings = []
    monkeypatch.setenv(bridge_assets.STALE_POLICY_ENV, "reFUSE ")
    monkeypatch.setattr(bridge_assets, "warn",
                        lambda action, why="": warnings.append(action))
    assert bridge_assets.stale_policy() == "refuse"

    monkeypatch.setenv(bridge_assets.STALE_POLICY_ENV, "ignore")
    assert bridge_assets.stale_policy() == "refresh"
    assert warnings and "ignore" in warnings[0]


def test_a_matching_artifact_resolves_without_touching_anything(
        estate, monkeypatch):
    estate["path"].write_bytes(estate["new"])
    bridge_assets.forget_pin_memo()

    def forbidden(**kwargs):
        raise AssertionError("a current estate must not be refreshed")

    monkeypatch.setattr(bridge_assets, "refresh_staged_bundle", forbidden)

    assert bridges.require_release_pin(estate["path"]) == estate["path"]
    assert estate["path"].read_bytes() == estate["new"]


def test_inspection_only_resolves_without_acting(estate, monkeypatch):
    """A reader sees what a door would see, and changes nothing."""

    def forbidden(**kwargs):
        raise AssertionError("reading the estate must not refresh it")

    monkeypatch.setattr(bridge_assets, "refresh_staged_bundle", forbidden)

    with bridges.inspection_only():
        assert bridges.require_release_pin(estate["path"]) == estate["path"]
    assert estate["path"].read_bytes() == estate["old"]


# ---------------------------------------------------------------------------
# The doors, and the report that has to agree with them
# ---------------------------------------------------------------------------

def test_the_mesh_door_ladder_refuses_the_stale_binary(
        estate, monkeypatch, tmp_path):
    """The door the defect was measured on, through its own resolver.

    The checkout and libexec rungs sit above the staged rung and are
    exempt from the judgement by design, so on a tree that carries its own
    built ``rw_mpas_mesh`` the door would resolve that binary and never
    reach the staged one; this test passed only on a checkout without a
    build until those rungs were pointed at an empty directory.
    """

    from gpuwm import mpas_mesh

    empty = tmp_path / "no-checkout-build"
    empty.mkdir()
    monkeypatch.setattr("gpuwm.mpas_mesh.crate_dir", lambda: empty)
    monkeypatch.setattr("gpuwm.mpas_mesh._repo_root", lambda: empty)
    monkeypatch.setenv(bridge_assets.ASSET_URL_BASE_ENV,
                       "https://127.0.0.1:9/unreachable")
    monkeypatch.setattr("gpuwm.mpas_mesh.default_bridge_dir",
                        lambda: estate["staged"])
    monkeypatch.delenv(mpas_mesh.MESH.env_var, raising=False)

    with pytest.raises(bridges.StaleBridgeError) as raised:
        mpas_mesh.MESH.find()
    assert _OLD_REV in str(raised.value)


def test_an_environment_override_still_wins_and_is_never_judged(
        estate, monkeypatch, tmp_path):
    """Explicit configuration is a declaration, not a discovery."""

    from gpuwm import mpas_mesh

    mine = tmp_path / "mine" / estate["filename"]
    mine.parent.mkdir()
    mine.write_bytes(estate["old"])
    monkeypatch.setenv(mpas_mesh.MESH.env_var, str(mine))
    monkeypatch.setattr("gpuwm.mpas_mesh.default_bridge_dir",
                        lambda: estate["staged"])

    assert mpas_mesh.MESH.find() == mine.resolve()


def test_doctor_reports_the_verdict_the_doors_act_on(estate, monkeypatch):
    """One judgement, two consumers, and the reader is not the actor.

    The shipped defect was exactly this pair disagreeing: doctor said
    BROKEN and every door resolved the same bytes anyway.
    """

    from gpuwm import doctor

    monkeypatch.setattr(bridges, "default_bridge_dir",
                        lambda: estate["staged"])
    monkeypatch.setattr("gpuwm.mpas_mesh.default_bridge_dir",
                        lambda: estate["staged"])

    def forbidden(**kwargs):
        raise AssertionError("doctor must not repair what it measures")

    monkeypatch.setattr(bridge_assets, "refresh_staged_bundle", forbidden)

    check = doctor._staged_estate_check()

    assert check.status == "missing"
    assert estate["filename"] in check.detail
    assert check.action == "gpuwm fetch-bridges"
    # And the estate is untouched by having been reported on.
    assert estate["path"].read_bytes() == estate["old"]


def test_fetch_bridges_is_reachable_from_the_state_it_repairs(
        estate, monkeypatch):
    """The remedy a refusal names must not be able to hit that refusal."""

    import argparse

    monkeypatch.setenv(bridge_assets.STALE_POLICY_ENV, "refuse")
    arguments = argparse.Namespace(from_dir=None, dest=str(estate["staged"]),
                                   keep_bundle=False, list=True)

    assert bridge_assets.fetch_bridges_main(arguments) == 0
