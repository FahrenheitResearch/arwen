"""The two evidence axes, walked exhaustively and made to fail.

The walk here is structural: it descends the whole registry document
looking for every location that carries a ``maturity`` key rather than
consulting a list of surfaces someone remembered to write down.  A check
that reported only ``components/.../options/...`` would prove the options
were fixed and say nothing about the twenty templates and nest edges, which
is the shape of the finding this file closes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gpuwm.physics_registry import (
    MATURITY_RANK,
    _enforce_evidence_axes,
    iter_maturity_surfaces,
    physics_registry,
)

MODEL = Path(__file__).resolve().parents[1]
WALK_RECEIPT = (
    MODEL / "docs" / "public" / "receipts" / "F5-maturity-surface-walk.json"
)


def _surface_class(path: str) -> str:
    if path.startswith("components."):
        return "component-option"
    if path.startswith("templates."):
        return "template"
    if path.startswith("transitions."):
        return "transition-cross-option"
    return "other"


def walk_report(registry: dict) -> dict:
    """The AC1 receipt: what the walk found, broken down by surface class."""

    surfaces = sorted(iter_maturity_surfaces(registry))
    counts: dict[str, int] = {}
    for path, _maturity in surfaces:
        klass = _surface_class(path)
        counts[klass] = counts.get(klass, 0) + 1
    return {
        "schema": "gpuwm-f5-maturity-surface-walk-v1",
        "maturity_bearing_locations": len(surfaces),
        "by_surface_class": dict(sorted(counts.items())),
        "maturity_values_in_use": sorted({m for _p, m in surfaces}),
        "rungs": sorted(registry["maturity_ladder"]["rungs"]),
        "scientific_evidence_in_use": sorted({
            option.get("scientific_evidence")
            for component in registry["components"].values()
            for option in component["options"].values()
            if option.get("implemented") is True
        }),
        "implemented_option_count": sum(
            1
            for component in registry["components"].values()
            for option in component["options"].values()
            if option.get("implemented") is True
        ),
        "component_option_count": sum(
            len(component["options"])
            for component in registry["components"].values()
        ),
    }


def test_the_walk_is_exhaustive_and_every_value_is_a_rung() -> None:
    """AC1: walk the document, not a hardcoded surface list."""

    registry = physics_registry()
    rungs = registry["maturity_ladder"]["rungs"]
    surfaces = list(iter_maturity_surfaces(registry))
    assert surfaces, "the walk found no maturity-bearing location at all"

    classes = {_surface_class(path) for path, _ in surfaces}
    assert "template" in classes and "transition-cross-option" in classes, (
        "the walk reached only component options, which would prove the "
        f"options were fixed and nothing else; classes found: {classes}")

    for path, maturity in surfaces:
        assert maturity in rungs, f"{path} carries {maturity!r}"

    scientific = registry["evidence_axes"]["scientific"]["enum"]
    for component_id, component in registry["components"].items():
        for option_id, option in component["options"].items():
            if option.get("implemented") is not True:
                continue
            assert option.get("scientific_evidence") in scientific, (
                f"components.{component_id}.options.{option_id}")


def test_the_walk_receipt_regenerates() -> None:
    from gpuwm.physics_registry import canonical_json

    regenerated = (
        canonical_json(walk_report(physics_registry())) + "\n"
    ).encode("utf-8")
    assert regenerated == WALK_RECEIPT.read_bytes(), (
        "docs/public/receipts/F5-maturity-surface-walk.json is stale; the "
        "walk's counts are the receipt and land with the registry")


def test_path_a_left_no_validated_maturity_value_anywhere() -> None:
    """AC2: the substring nobody should be able to find in a label."""

    offenders = [
        (path, maturity)
        for path, maturity in iter_maturity_surfaces(physics_registry())
        if "validated" in maturity
    ]
    assert offenders == [], (
        "Path A renamed the vocabulary; these maturity VALUES still claim "
        f"validation: {offenders}")


def test_the_rung_set_equals_the_set_in_use() -> None:
    """AC3: no decorative rungs, no undeclared values."""

    registry = physics_registry()
    in_use = {maturity for _p, maturity in iter_maturity_surfaces(registry)}
    assert in_use == set(registry["maturity_ladder"]["rungs"])
    assert registry["evidence_axes"][
        "conformance_implies_scientific_validation"] is False
    assert set(registry["evidence_axes"]) >= {"maturity", "scientific"}


def test_the_ladder_is_the_only_ordering_of_maturity_names() -> None:
    """F2: MATURITY_RANK and both warning tiers come off the document."""

    registry = physics_registry()
    order = registry["maturity_ladder"]["order"]
    assert MATURITY_RANK == {name: rank for rank, name in enumerate(order)}

    rungs = registry["maturity_ladder"]["rungs"]
    policy = registry["warning_policy"]
    assert policy["warn_maturities"] == [
        name for name in order if rungs[name]["warning_tier"] == "warn"]
    assert policy["nonwarning_maturities"] == [
        name for name in order if rungs[name]["warning_tier"] == "nonwarning"]

    source = (MODEL / "gpuwm" / "physics_registry.py").read_text(
        encoding="utf-8")
    for name in order:
        assert source.count(f'"{name}"') == 0, (
            f"gpuwm/physics_registry.py names the maturity {name!r} "
            "literally; the ladder in the generated document is the only "
            "place a maturity name is allowed to be written down")


# --- the perturbation controls ------------------------------------------
#
# Each mutates an in-memory copy and requires the load-time enforcement to
# refuse it.  Without these the enforcement is unproven: a check that can
# never fail is not a check.


def _refuses(registry: dict) -> str:
    with pytest.raises(RuntimeError) as excinfo:
        _enforce_evidence_axes(registry)
    return str(excinfo.value)


def test_control_option_maturity_off_the_ladder() -> None:
    registry = physics_registry()
    registry["components"]["microphysics"]["options"]["thompson-mp8"][
        "maturity"] = "model-validated"
    assert "names no rung" in _refuses(registry)


def test_control_template_maturity_off_the_ladder() -> None:
    registry = physics_registry()
    registry["templates"]["wsm6-ysu-mm5-noah-no-radiation-v1"][
        "maturity"] = "validation-candidate"
    assert "names no rung" in _refuses(registry)


def test_control_transition_cross_option_maturity_off_the_ladder() -> None:
    registry = physics_registry()
    edges = registry["transitions"]["microphysics-one-way-v1"]["cross_options"]
    labelled = next(edge for edge in edges if "maturity" in edge)
    labelled["maturity"] = "model-validated"
    message = _refuses(registry)
    assert "names no rung" in message and "transitions" in message


def test_control_deleting_a_rung_that_is_in_use() -> None:
    registry = physics_registry()
    for block in (registry["maturity_ladder"],
                  registry["evidence_axes"]["maturity"]):
        block["rungs"].pop("experimental-runtime")
    registry["maturity_ladder"]["order"].remove("experimental-runtime")
    assert "names no rung" in _refuses(registry)


def test_control_a_decorative_rung_nothing_uses() -> None:
    registry = physics_registry()
    for block in (registry["maturity_ladder"],
                  registry["evidence_axes"]["maturity"]):
        block["rungs"]["observationally-verified"] = {
            "rank": 7, "warning_tier": "nonwarning", "definition": "unused"}
    registry["maturity_ladder"]["order"].append("observationally-verified")
    assert "unused rungs" in _refuses(registry)


def test_control_scientific_evidence_off_the_enum() -> None:
    registry = physics_registry()
    registry["components"]["microphysics"]["options"]["thompson-mp8"][
        "scientific_evidence"] = "obs-validated"
    assert "scientific.enum" in _refuses(registry)


def test_control_conformance_implying_scientific_validation() -> None:
    registry = physics_registry()
    registry["evidence_axes"][
        "conformance_implies_scientific_validation"] = True
    assert "agreement with a model" in _refuses(registry)


def test_control_a_warning_tier_that_disagrees_with_the_ladder() -> None:
    registry = physics_registry()
    registry["warning_policy"]["warn_maturities"] = [
        name for name in registry["warning_policy"]["warn_maturities"]
        if name != "wrf-matched-run-candidate"]
    assert "single source" in _refuses(registry)


def test_the_shipped_document_passes_the_enforcement_it_declares() -> None:
    """The positive half: the controls above must not be passing vacuously."""

    _enforce_evidence_axes(physics_registry())


def test_no_case_token_reaches_the_new_vocabulary() -> None:
    """Token hygiene with its own positive control."""

    import re

    from tools.check_case_token_leakage import CASE_TOKENS

    registry = physics_registry()
    subject = json.dumps({
        "maturity_ladder": registry["maturity_ladder"],
        "evidence_axes": registry["evidence_axes"],
    }, ensure_ascii=False)
    pattern = re.compile("|".join(re.escape(t) for t in CASE_TOKENS), re.I)
    assert pattern.search(subject) is None, pattern.findall(subject)

    injected = copy.deepcopy(registry["maturity_ladder"])
    injected["rungs"][f"{sorted(CASE_TOKENS)[0]}-matched"] = {
        "rank": 99, "warning_tier": "warn", "definition": "injected"}
    assert pattern.search(json.dumps(injected, ensure_ascii=False)) is not None
