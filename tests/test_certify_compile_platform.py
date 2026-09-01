"""``gpuwm certify`` sees the two things it used to be blind to.

Two conditions were written, shipped, and never called: the compile-platform
fingerprint that separates one NVRTC build from another, and the guard that
asks whether a capsule's kernel manifest recorded anything at all.  With both
unwired, ``gpuwm certify`` printed ``PASS`` over a capsule compiled by a
different compiler than the one now installed, and over a capsule whose
kernel manifest was empty.  A false green on the certification path launders
drift as verified, which is worse than having no instrument.

Every refusal here is measured through :func:`gpuwm.cli.main`, so what is
asserted is the exit code and the sentence a reader actually gets.

**Both directions, on the instrument itself.**  A check that only ever passes
is the bug being fixed here, so each condition is exercised on a good input
and on a bad one, and the decision function is additionally held to the trap
that makes this class of instrument lie: two fingerprints that measured
nothing are *equal*, so an unguarded comparison reports "no drift" over an
unverified platform.  ``compile_platform_agreement`` must refuse to reach a
verdict at all in that case rather than compute one from what is there.
"""

from __future__ import annotations

import copy
import json

import pytest

import certification_fixtures as fixtures
from gpuwm.certify.compile_platform import (FINGERPRINT_KEYS, UNRESOLVED,
                                            compile_platform_agreement,
                                            compile_platform_fingerprint,
                                            describe_drift,
                                            recorded_compile_platform,
                                            unresolved_fingerprint_items)
from gpuwm.certify.kernel_manifest import manifest_is_empty
from gpuwm.certify.verdict import CONDITIONS, certify
from gpuwm.cli import main

CAN_WITNESS = fixtures.this_process_can_witness_the_compile_platform()

needs_a_witness = pytest.mark.skipif(
    not CAN_WITNESS,
    reason="this interpreter resolves no complete compile-platform "
           "fingerprint, so no capsule matches it and the passing direction "
           "cannot be measured here")


def _run_certify(paths) -> int:
    return main(["certify",
                 "--run-capsule", str(paths["capsule"]),
                 "--metrics-csv", str(paths["metrics"]),
                 "--band", str(paths["band"]),
                 "--wrf-reference-manifest", str(paths["wrf_reference"])])


def _refusal(tmp_path, capsys, **kwargs) -> str:
    paths = fixtures.matched_set(tmp_path, **kwargs)
    assert _run_certify(paths) != 0
    return capsys.readouterr().err


def _verdict(tmp_path, **kwargs) -> dict:
    paths = fixtures.matched_set(tmp_path, **kwargs)
    return certify(capsule_path=paths["capsule"],
                   metrics_csv=paths["metrics"],
                   band_path=paths["band"],
                   wrf_reference_manifest=paths["wrf_reference"])


def _condition(verdict: dict, name: str) -> dict:
    for item in verdict["conditions"]:
        if item["condition"] == name:
            return item
    raise AssertionError(f"the verdict declares no condition {name!r}: "
                         f"{[i['condition'] for i in verdict['conditions']]}")


# --------------------------------------------------------------------------
# H35 -- the empty kernel manifest
# --------------------------------------------------------------------------

def test_certify_refuses_a_capsule_whose_kernel_manifest_recorded_nothing(
        tmp_path, capsys):
    """The 'compared nothing, reported identical' shape, on the capsule side.

    ``gpuwm dual-run`` already refuses an empty screen.  Certification did
    not: a capsule whose ``kernel_manifest`` recorded no compiled module at
    all reached ``certify: PASS``.
    """
    def empty(capsule):
        capsule["kernel_manifest"] = {}

    message = _refusal(tmp_path, capsys, capsule_edit=empty)
    assert "kernel_manifest_records_a_compiled_module" in message
    assert "recorded no compiled module" in message


def test_the_empty_manifest_guard_reads_the_capsule_not_this_process(
        tmp_path):
    """The guard must judge the document, never the certifying process.

    ``manifest_is_empty`` used to default to the *live* manifest of whatever
    process asked.  A certifier compiles nothing, so that default would have
    made every capsule look empty -- and, worse, a certifier that had
    compiled something would have made every capsule look full.
    """
    with pytest.raises(TypeError):
        manifest_is_empty()  # type: ignore[call-arg]
    assert manifest_is_empty({}) is True
    assert manifest_is_empty({"gpuwm.core.kernels:diff6": {}}) is False


@needs_a_witness
def test_a_capsule_that_recorded_a_module_satisfies_the_manifest_condition(
        tmp_path):
    """The other direction: the guard is not simply always-refusing."""
    verdict = _verdict(tmp_path)
    item = _condition(verdict, "kernel_manifest_records_a_compiled_module")
    assert item["satisfied"] is True
    # Positive evidence: the detail names what was recorded, so a satisfied
    # row cannot be read as "nothing was checked".
    assert "gpuwm.core.kernels:diff6" in item["detail"]


# --------------------------------------------------------------------------
# H34 -- NVRTC drift
# --------------------------------------------------------------------------

@needs_a_witness
def test_certify_refuses_a_capsule_whose_compile_platform_moved(
        tmp_path, capsys):
    """The pip-NVRTC-shadow failure mode, end to end.

    A capsule compiled under one NVRTC build, certified on a box now running
    another, used to print PASS.  The refusal must name the item that moved
    and both of its values -- a bare "drift" would leave the reader exactly
    where the 2026-08-04 incident left one.
    """
    def shadow(capsule):
        toolkit = capsule["numerical_stack"]["cuda_toolkit_nvrtc"]["value"]
        toolkit["nvrtc_build"] = "0.0.0"

    message = _refusal(tmp_path, capsys, capsule_edit=shadow)
    assert "compile_platform_matches_this_process" in message
    assert "nvrtc_build" in message
    assert "0.0.0" in message


@needs_a_witness
@pytest.mark.parametrize("item", FINGERPRINT_KEYS)
def test_every_fingerprint_item_can_refuse_on_its_own(tmp_path, capsys, item):
    """No item of the fingerprint is carried without being compared.

    Witness-gated (see certification_fixtures.py): the perturbation
    fixture matches every OTHER item to the live measurement, so on a
    process that cannot witness the platform the capsule is born
    un-witnessed and the refusal collapses to recorded-missing instead
    of naming the perturbed item.
    """
    def perturb(capsule):
        stack = capsule["numerical_stack"]
        moved = "moved-by-the-test"
        if item in ("nvrtc_build", "nvrtc_build_id", "nvrtc_library_sha256"):
            stack["cuda_toolkit_nvrtc"]["value"][item] = moved
        elif item == "cuda_driver_version":
            stack["cuda_driver_version"]["value"] = moved
        elif item == "device_compute_capability":
            stack["gpu_identity"]["value"]["compute_capability"] = moved
        else:
            stack[item]["value"] = moved

    message = _refusal(tmp_path, capsys, capsule_edit=perturb)
    assert "compile_platform_matches_this_process" in message
    assert item in message


def test_certify_refuses_a_capsule_that_records_no_compile_platform(
        tmp_path, capsys):
    """A capsule that cannot say which compiler built it cannot be certified.

    This is the half that needs no card: the published pin row used to report
    only ``nvrtc.getVersion()``, which cannot separate 12.9.41 from 12.9.86,
    and a capsule can still carry a *resolved* toolkit pin whose four-part
    build went unmeasured.  Certification refuses that capsule rather than
    comparing against a value nothing measured.
    """
    def blank(capsule):
        toolkit = capsule["numerical_stack"]["cuda_toolkit_nvrtc"]["value"]
        toolkit["nvrtc_build"] = UNRESOLVED
        toolkit["nvrtc_build_id"] = UNRESOLVED
        toolkit["nvrtc_library_sha256"] = UNRESOLVED

    message = _refusal(tmp_path, capsys, capsule_edit=blank)
    assert "compile_platform_recorded" in message
    assert "nvrtc_build" in message


def test_an_unresolved_toolkit_pin_does_not_read_as_a_recorded_platform(
        tmp_path, capsys):
    """A pin marked unavailable projects to unresolved, never to its payload."""
    def unresolve(capsule):
        entry = capsule["numerical_stack"]["cuda_toolkit_nvrtc"]
        entry["status"] = "unavailable"
        entry["reason"] = "the fixture declares this pin unmeasured"

    message = _refusal(tmp_path, capsys, capsule_edit=unresolve)
    assert "compile_platform_recorded" in message


@needs_a_witness
def test_a_matched_capsule_satisfies_both_platform_conditions(tmp_path):
    """The passing direction, with the measurement visible in the verdict."""
    verdict = _verdict(tmp_path)
    recorded = _condition(verdict, "compile_platform_recorded")
    matches = _condition(verdict, "compile_platform_matches_this_process")
    assert recorded["satisfied"] is True
    assert matches["satisfied"] is True

    block = verdict["compile_platform"]
    assert block["comparable"] is True
    assert block["drift"] == []
    assert block["agreed"] is True
    live = compile_platform_fingerprint()
    assert block["measured"] == {key: str(live[key])
                                 for key in FINGERPRINT_KEYS}
    # A satisfied row states what it measured.  A bare boolean would be
    # indistinguishable from a row nobody evaluated.
    assert live["nvrtc_build"] in matches["detail"]


@needs_a_witness
def test_the_platform_block_is_bound_into_the_verdict_digest(tmp_path):
    """Editing the recorded platform moves the binding and the re-derivation."""
    from gpuwm.certify.verdict import (capsule_binding_sha256,
                                       rederive_verdict)

    verdict = _verdict(tmp_path)
    assert verdict["passed"] is True
    baseline = verdict["capsule_binding_sha256"]
    tampered = copy.deepcopy(verdict)
    tampered["compile_platform"]["recorded"]["nvrtc_build"] = "0.0.0"
    assert capsule_binding_sha256(tampered) != baseline
    # And laundering the condition row alone does not buy a pass either.
    tampered["compile_platform"]["drift"] = []
    assert rederive_verdict(tampered) is False


# --------------------------------------------------------------------------
# The trap: two fingerprints that measured nothing are equal
# --------------------------------------------------------------------------

def test_two_unresolved_fingerprints_do_not_read_as_agreement():
    """``describe_drift`` returning [] is not evidence of a verified platform.

    This is the exact-zero-delta trap in fingerprint form.  ``describe_drift``
    is honest -- these two dicts really are identical -- so the guard must sit
    in the caller, and it must suppress the verdict rather than compute one.
    """
    nothing = {key: UNRESOLVED for key in FINGERPRINT_KEYS}
    assert describe_drift(nothing, nothing) == []

    agreement = compile_platform_agreement(nothing, nothing)
    assert agreement["agreed"] is False
    assert agreement["comparable"] is False
    assert agreement["drift"] is None, (
        "a drift list computed over unmeasured items is a verdict from no "
        "evidence; it must be suppressed, not empty")
    assert list(agreement["recorded_unresolved"]) == list(FINGERPRINT_KEYS)
    assert list(agreement["measured_unresolved"]) == list(FINGERPRINT_KEYS)


def test_a_half_measured_side_still_suppresses_the_comparison():
    """One unresolved item is enough to stop the comparison being reported."""
    complete = {key: f"value-{key}" for key in FINGERPRINT_KEYS}
    partial = dict(complete)
    partial["nvrtc_library_sha256"] = UNRESOLVED
    agreement = compile_platform_agreement(complete, partial)
    assert agreement["drift"] is None
    assert agreement["agreed"] is False
    assert agreement["measured_unresolved"] == ["nvrtc_library_sha256"]


def test_the_agreement_function_answers_yes_on_two_complete_equal_sides():
    """The control: with real evidence on both sides it does say agreed."""
    complete = {key: f"value-{key}" for key in FINGERPRINT_KEYS}
    agreement = compile_platform_agreement(complete, dict(complete))
    assert agreement["comparable"] is True
    assert agreement["drift"] == []
    assert agreement["agreed"] is True


def test_the_agreement_function_names_what_moved():
    complete = {key: f"value-{key}" for key in FINGERPRINT_KEYS}
    moved = dict(complete, nvrtc_build="0.0.0")
    agreement = compile_platform_agreement(complete, moved)
    assert agreement["comparable"] is True
    assert agreement["agreed"] is False
    assert len(agreement["drift"]) == 1
    assert "nvrtc_build" in agreement["drift"][0]
    assert "0.0.0" in agreement["drift"][0]


@pytest.mark.parametrize("empty", [None, {}, {"nvrtc_build": ""},
                                   {"nvrtc_build": None}])
def test_unresolved_items_reports_the_whole_key_set_for_an_empty_side(empty):
    assert set(unresolved_fingerprint_items(empty)) >= {"nvrtc_build"}
    assert set(unresolved_fingerprint_items(empty)) <= set(FINGERPRINT_KEYS)


# --------------------------------------------------------------------------
# The projection out of a capsule
# --------------------------------------------------------------------------

@needs_a_witness
def test_the_projection_reads_the_pins_the_published_table_names(tmp_path):
    """The capsule carries the fingerprint as pins; nothing is duplicated."""
    band = fixtures.shipped_band()
    capsule = fixtures.matched_capsule(band["config_sha256"])
    recorded = recorded_compile_platform(capsule)
    assert set(recorded) == set(FINGERPRINT_KEYS)
    assert unresolved_fingerprint_items(recorded) == ()
    assert recorded == compile_platform_fingerprint()
    # The driver pin is an int in the capsule and text in the fingerprint;
    # the projection is what reconciles them, and it must not be by luck.
    assert isinstance(
        capsule["numerical_stack"]["cuda_driver_version"]["value"], int)
    assert isinstance(recorded["cuda_driver_version"], str)


def test_the_projection_never_invents_a_key():
    """A capsule with no numerical stack projects to a complete unresolved."""
    recorded = recorded_compile_platform({})
    assert set(recorded) == set(FINGERPRINT_KEYS)
    assert set(recorded.values()) == {UNRESOLVED}


# --------------------------------------------------------------------------
# The declared condition list
# --------------------------------------------------------------------------

def test_the_new_conditions_are_declared_and_always_reported(tmp_path):
    """Every condition is evaluated on every invocation, fired or not."""
    for name in ("kernel_manifest_records_a_compiled_module",
                 "compile_platform_recorded",
                 "compile_platform_matches_this_process"):
        assert name in CONDITIONS
    verdict = _verdict(tmp_path, out_of_band=True)
    reported = [item["condition"] for item in verdict["conditions"]]
    assert reported == list(CONDITIONS)


def test_a_verdict_document_round_trips_with_its_platform_block(tmp_path):
    paths = fixtures.matched_set(tmp_path)
    out = tmp_path / "verdict.json"
    main(["certify",
          "--run-capsule", str(paths["capsule"]),
          "--metrics-csv", str(paths["metrics"]),
          "--band", str(paths["band"]),
          "--wrf-reference-manifest", str(paths["wrf_reference"]),
          "--out-verdict", str(out)])
    document = json.loads(out.read_text(encoding="utf-8"))
    assert set(document["compile_platform"]) == {
        "recorded", "measured", "recorded_unresolved", "measured_unresolved",
        "drift", "comparable", "agreed"}
