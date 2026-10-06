"""``gpuwm doctor`` judges each rustwx engine by the probe its door refuses by.

Breakage these tests prevent.  The first doctor lines for ``rw_verify``
and ``rw_compare`` judged both binaries by the static
:data:`gpuwm.bridges.BRIDGE_ABI_MARKERS` byte search, while ``gpuwm
render --compare`` judged ``rw_compare`` by its ``--abi`` line
(:func:`gpuwm.rustwx_lanes.probe_compare_bin`).  The two contracts landed
on parallel branches, so a build can carry one and not the other: such a
build was reported ``verified`` here and then refused at the door, or the
reverse.  The fix is one probe per door, imported by doctor rather than
restated, so these tests build exactly those mismatched binaries and hold
the report and the door to the same verdict.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from gpuwm import bridges, doctor, rustwx, rustwx_lanes

REFERENCE_LITERAL = bridges.BRIDGE_ABI_MARKERS["rw_compare"]
VERIFY_LITERAL = bridges.BRIDGE_ABI_MARKERS["rw_verify"]
#: The ``--abi`` answer of a build that predates the theme line: the
#: contract ``render --compare`` refuses (see test_rustwx_lanes).
OLD_COMPARE_ABI = rustwx_lanes.COMPARE_ABI_MARKER.split(
    "\tgpuwm-rw-compare-presentation-v1", 1)[0]


@pytest.fixture
def engines(monkeypatch, tmp_path):
    """Overrides pointing at planted files; launch and accept stubbed.

    ``answer`` is what the planted ``rw_compare`` prints for ``--abi``.
    The static probes read the planted bytes for real.
    """

    monkeypatch.setattr(bridges, "accept_resolved", lambda path, **_: path)
    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))
    answer = {"abi": rustwx_lanes.COMPARE_ABI_MARKER}
    launched: list[list[str]] = []

    def run(command, **kwargs):
        launched.append([str(part) for part in command])
        return SimpleNamespace(returncode=0, stdout=answer["abi"] + "\n",
                               stderr="")

    monkeypatch.setattr(rustwx_lanes.subprocess, "run", run)
    compare = tmp_path / "rw_compare"
    verify = tmp_path / "rw_verify"
    monkeypatch.setenv(rustwx_lanes.COMPARE_ENV, str(compare))
    monkeypatch.setenv(rustwx.VERIFICATION_ENV, str(verify))
    return SimpleNamespace(compare=compare, verify=verify, answer=answer,
                           launched=launched)


def _door_refuses(door) -> bool:
    try:
        door()
    except RuntimeError:
        return True
    return False


# -- rw_compare: the mismatch the skeptic found, both ways round ------------

def test_reference_literal_with_an_old_abi_line_is_stale_in_doctor_and_refused_at_render(
        engines):
    """The skeptic's case: static marker present, ``--abi`` old.  The
    static search alone called this verified; ``render --compare``
    refuses it."""

    engines.compare.write_bytes(b"build " + REFERENCE_LITERAL)
    engines.answer["abi"] = OLD_COMPARE_ABI

    check = doctor._comparison_engine_check()

    assert check.status != "verified"
    assert check.severity == doctor.SEVERITY_BROKEN
    assert "gpuwm render --compare: launches, but --abi does not match" in check.detail
    assert "1 of 2 door(s)" in check.detail
    assert _door_refuses(rustwx_lanes.require_compare_bin)
    # The other door accepts it, and the line says so rather than hiding it.
    assert "the reference verification panels: speaks" in check.detail
    assert rustwx_lanes.require_compare_reference_bin() == engines.compare


def test_current_abi_line_without_the_reference_literal_is_stale_in_doctor_and_refused_at_the_panels(
        engines, tmp_path):
    """The reverse: ``--abi`` current, ``--fetch-reference`` absent.  The
    ``--abi`` probe alone would call this verified; the reference panels
    cannot use it, and now refuse it before launching it."""

    engines.compare.write_bytes(b"build without the reference receipt")

    check = doctor._comparison_engine_check()

    assert check.status != "verified"
    assert check.severity == doctor.SEVERITY_BROKEN
    assert "the reference verification panels: was built from a checkout" in check.detail
    assert rustwx_lanes.require_compare_bin() == engines.compare
    assert _door_refuses(rustwx_lanes.require_compare_reference_bin)
    engines.launched.clear()
    with pytest.raises(RuntimeError, match="predates this release's rw_compare contract"):
        rustwx.verification_reference(
            cache=tmp_path / "cache", reference="reference", hour=1,
            cycle=datetime.datetime(2026, 1, 1))
    assert not any("--fetch-reference" in command for command in engines.launched)


def test_a_build_speaking_both_contracts_is_verified_and_admitted_by_both_doors(engines):
    """The negative control: the line does not fire on a current build."""

    engines.compare.write_bytes(b"build " + REFERENCE_LITERAL)

    check = doctor._comparison_engine_check()

    assert check.status == "verified", check.detail
    assert rustwx_lanes.require_compare_bin() == engines.compare
    assert rustwx_lanes.require_compare_reference_bin() == engines.compare


# -- rw_verify --------------------------------------------------------------

def test_a_verifier_without_the_request_contract_is_stale_in_doctor_and_refused_at_its_door(
        engines):
    engines.verify.write_bytes(b"build without the request contract")

    check = doctor._verification_engine_check()

    assert check.status != "verified"
    assert check.severity == doctor.SEVERITY_BROKEN
    assert "predates this release's rw_verify contract" in check.detail
    with pytest.raises(RuntimeError, match="predates this release's rw_verify contract"):
        rustwx.verification_binary()


def test_a_current_verifier_is_verified_and_admitted(engines):
    engines.verify.write_bytes(b"build " + VERIFY_LITERAL)

    check = doctor._verification_engine_check()

    assert check.status == "verified", check.detail
    assert rustwx.verification_binary() == engines.verify


# -- one probe, never a second copy -----------------------------------------

@pytest.mark.parametrize("module, probe, check", [
    (rustwx_lanes, "probe_compare_bin", "_comparison_engine_check"),
    (rustwx_lanes, "probe_compare_reference_bin", "_comparison_engine_check"),
    (rustwx, "probe_verification_binary", "_verification_engine_check"),
])
def test_doctor_reports_the_verdict_of_the_door_s_own_probe(
        engines, monkeypatch, module, probe, check):
    """Replace one door's probe and doctor's line follows it.  A doctor
    that kept its own copy of the judgement would not see the change."""

    engines.compare.write_bytes(b"build " + REFERENCE_LITERAL)
    engines.verify.write_bytes(b"build " + VERIFY_LITERAL)
    seen = []

    def sentinel(path):
        seen.append(path)
        return False, "SENTINEL VERDICT"

    monkeypatch.setattr(module, probe, sentinel)
    result = getattr(doctor, check)()

    assert seen, f"doctor never asked {probe}"
    assert result.status != "verified"
    assert "SENTINEL VERDICT" in result.detail


# -- the held-out GRIB2 exporter --------------------------------------------

def test_every_checked_artifact_is_one_this_release_bundles():
    """Breakage: 2.8.6 holds the GRIB2 exporter out (it returns in
    2.8.7), yet a doctor line for it imported the held-out module and
    reported a door this release does not offer.  A check for an artifact
    the bundle does not carry is that defect in general form."""

    from gpuwm.bridge_assets import BUNDLED_ARTIFACTS

    bundled = {artifact.name for artifact in BUNDLED_ARTIFACTS}
    orphans = sorted(set(doctor._CHECKED_ARTIFACTS) - bundled)
    assert not orphans, orphans
    assert not hasattr(doctor, "_grib2_export_check")
