"""The composition rule, its blast radius, and the ground truth beneath it.

Two receipts are the subject here.  ``F2-ground-truth.json`` records what
the production resolvers return for the default template and for the
legacy-RRTMG run of record; ``F2-composition-blast-radius.json`` enumerates
every registered template under all three candidate composition axes.  Both
are regenerated here and compared byte for byte, so a registry edit that
changes either one has to land the receipt with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import report_registry_ground_truth as ground_truth
from tools import report_template_evidence_consistency as blast_radius

MODEL = Path(__file__).resolve().parents[1]
GROUND_TRUTH = MODEL / "docs" / "public" / "receipts" / "F2-ground-truth.json"
BLAST_RADIUS = (
    MODEL / "docs" / "public" / "receipts" / "F2-composition-blast-radius.json"
)


def test_ground_truth_receipt_regenerates_byte_for_byte() -> None:
    from gpuwm.physics_registry import physics_registry

    regenerated = ground_truth.render(ground_truth.build(physics_registry()))
    assert regenerated == GROUND_TRUTH.read_bytes(), (
        "docs/public/receipts/F2-ground-truth.json no longer matches what "
        "the production resolvers return; regenerate it with "
        "tools/report_registry_ground_truth.py and land both together")


def test_the_ground_truth_receipt_regenerates_on_a_machine_with_no_case_data(
        tmp_path, monkeypatch) -> None:
    """The same bytes on a machine that has never fetched the run of record.

    The receipt resolves PHYSICS: the registry's view of the default
    template, and the selector integers the run of record's four domains
    emit.  None of that reads a GRIB.  But the tool loaded the run of
    record with the existence check on, so regenerating it required the
    1974 reference bundle staged under ``$GPUWM_CASE_DATA_ROOT``
    (``~/.local/share/gpuwm`` by default) and died with "forcing file ...
    does not exist" everywhere else -- a fresh HOME, a clean-venv replay,
    a reviewer's checkout.

    The roots are pointed at an empty directory rather than merely
    unset, so this is the test's decision on a developer machine that
    HAS staged the bundle.
    """

    empty = tmp_path / "no-case-data"
    empty.mkdir()
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(empty))
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setenv("USERPROFILE", str(empty))
    monkeypatch.setenv("XDG_DATA_HOME", str(empty))

    from gpuwm.physics_registry import physics_registry

    regenerated = ground_truth.render(ground_truth.build(physics_registry()))
    assert regenerated == GROUND_TRUTH.read_bytes()


def test_blast_radius_receipt_regenerates_byte_for_byte() -> None:
    from gpuwm.physics_registry import physics_registry

    regenerated = blast_radius.render(
        blast_radius.evaluate(physics_registry()))
    assert regenerated == BLAST_RADIUS.read_bytes(), (
        "docs/public/receipts/F2-composition-blast-radius.json no longer "
        "matches the checker's output; regenerate it with "
        "tools/report_template_evidence_consistency.py")


def test_every_registered_template_appears_under_every_axis() -> None:
    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    receipt = json.loads(BLAST_RADIUS.read_text(encoding="utf-8"))
    assert {row["template_id"] for row in receipt["templates"]} == set(
        registry["templates"]), (
        "the blast-radius receipt does not cover every registered template")
    for row in receipt["templates"]:
        assert set(row["axes"]) == set(blast_radius.AXES), row["template_id"]


def test_the_ground_truth_receipt_shows_the_selector_engine_blindness() -> None:
    """The finding beneath the whole package, restated as a measurement.

    The default template selects the substitution radiation engine and the
    run of record selects the exact legacy port, and both emit the same
    ``RA_LW_PHYSICS``.  This is not an assertion about which is better; it
    is the reason a maturity label attached to a registry option cannot be
    read as a statement about the radiation code that ran.
    """

    receipt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    default_engine = receipt["default_template"][
        "radiation_engine_parameters"]["wrf_rrtmg_compatibility"]
    per_domain = receipt["run_of_record"]["per_domain"]
    assert per_domain, "the run of record resolved no domains"
    record_engines = {
        domain["radiation_engine_fields"]["wrf_rrtmg_compatibility"]
        for domain in per_domain
    }
    assert record_engines != {default_engine}, (
        "the two configurations select the same radiation engine, so this "
        "receipt no longer demonstrates anything")
    selectors = {
        domain["wrf_physics_selector_attrs"]["RA_LW_PHYSICS"]
        for domain in per_domain
    }
    assert len(selectors) == 1, (
        f"the run of record emits more than one RA_LW_PHYSICS: {selectors}")


def test_the_ratified_rule_leaves_no_silent_violation() -> None:
    """Post-remediation self-consistency under the ratified axis.

    Every template the rule flags is discharged either by a resolvable
    evidence pointer or by an exemption naming the owner decision that
    granted it.  The set of exemptions is whatever was ratified; what this
    gates is that none of them is silent, and that none is left over for a
    template the rule no longer flags.
    """

    from gpuwm.physics_registry import physics_registry

    report = blast_radius.evaluate(physics_registry())["ratified"]
    assert report["axis"] == "A-strict-min"
    assert report["enforcement_point"] == "registry-document-invariant"
    assert report["undischarged"] == [], (
        "the ratified composition rule flags templates that carry neither a "
        "resolvable evidence pointer nor a committed exemption: "
        f"{report['undischarged']}")
    assert report["unused_exemptions"] == [], (
        "these exemptions no longer cover any violation and must be "
        f"withdrawn: {report['unused_exemptions']}")
    for entry in report["discharged"]:
        assert entry["detail"], entry


def test_control_raising_one_template_by_one_rank_produces_a_violation() -> (
        None):
    """Mutation control: the rule must be able to fire on a new template.

    Raising exactly one template that the unmutated registry does NOT flag
    -- by one rung, in an in-memory copy -- has to produce a violation the
    shipped document does not produce.  Without this the rule could be
    passing because it never looks at anything.
    """

    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    order = registry["maturity_ladder"]["order"]
    before = blast_radius.evaluate(registry)
    flagged = set(before["violating_template_ids"]["A-strict-min"])
    exempt = set(
        registry["maturity_ladder"]["composition_rule"][
            "composition_exemptions"])

    candidate = next(
        template_id
        for template_id in sorted(registry["templates"])
        if template_id not in flagged
        and order.index(registry["templates"][template_id]["maturity"]) + 1
        < len(order)
    )
    registry["templates"][candidate]["maturity"] = order[
        order.index(registry["templates"][candidate]["maturity"]) + 1]

    after = blast_radius.evaluate(registry)
    new = set(after["violating_template_ids"]["A-strict-min"]) - flagged
    assert candidate in new, (
        f"raising {candidate} by one rung produced no new violation; the "
        "composition rule cannot fire")
    assert candidate not in exempt, (
        "the control chose a template that is already exempt, so its "
        "violation would be discharged and prove nothing")
    assert after["ratified"]["undischarged"], (
        "the mutated registry has an undischarged violation and the "
        "self-consistency check above must therefore be able to fail")


@pytest.mark.parametrize("axis", blast_radius.AXES)
def test_each_candidate_axis_reports_a_verdict_for_every_template(
    axis: str,
) -> None:
    receipt = json.loads(BLAST_RADIUS.read_text(encoding="utf-8"))
    verdicts = {
        row["axes"][axis]["verdict"] for row in receipt["templates"]
    }
    assert verdicts <= {"ok", "violation"}
    assert receipt["violation_counts"][axis] == sum(
        1 for row in receipt["templates"]
        if row["axes"][axis]["verdict"] == "violation")


def test_historical_maturity_aliases_cannot_bypass_composition_checks():
    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    current = blast_radius.evaluate(registry)
    aliases = {new: old for old, new in registry["maturity_ladder"]["aliases"].items()}
    for template in registry["templates"].values():
        template["maturity"] = aliases.get(template["maturity"], template["maturity"])
    for component in registry["components"].values():
        for option in component["options"].values():
            option["maturity"] = aliases.get(option["maturity"], option["maturity"])
    assert blast_radius.evaluate(registry) == current


def test_selected_tuple_membership_matches_the_cartesian_reference():
    """Avoid millions of startup tuples without weakening expert admission."""
    import itertools
    import random
    from gpuwm.physics_compat import _tree_tuple_registry_governance, PhysicsCapabilityError

    def reference(registry, selected, modes):
        normal, expert = set(), {}
        key = lambda value: tuple(sorted(value.items()))
        for route in registry["runner_routes"].values():
            if route["implemented"] is not True or route["mode"] not in modes:
                continue
            sets = {name: set(registry["components"][name]["options"])
                    for name in route["allowed_component_overrides"]}
            for name, values in route["allowed_component_options"].items():
                sets.setdefault(name, set()).update(values)
            for kind in ("source", "expert"):
                for template_id in route[f"{kind}_template_ids"]["source"]:
                    base = registry["templates"][template_id]["components"]
                    candidates = [dict(base)]
                    for name, values in sets.items():
                        choices = values | ({base[name]} if isinstance(base.get(name), str) else set())
                        candidates = [{**item, name: value} for item in candidates for value in choices]
                    for candidate in candidates:
                        if kind == "source":
                            normal.add(key(candidate))
                        else:
                            expert.setdefault(key(candidate), set()).add(route["expert_acknowledgement_id"])
        wanted = key(selected)
        if wanted in normal:
            return "registry-reachable", None
        if wanted in expert:
            acknowledgements = sorted(expert[wanted])
            if len(acknowledgements) > 1:
                return "error", acknowledgements
            return "registry-expert-template", acknowledgements[0]
        return "outside-registry-declared-reachability", "outside"

    rng = random.Random(294)
    for _ in range(24):
        components = {name: {"options": {option: {"implemented": True}
                       for option in rng.sample(["zero", "one"], rng.randrange(3))}}
                      for name in ("a", "b", "c")}
        templates = {f"t{i}": {"components": {name: rng.choice(["own", "zero", "one"])
                     for name in ("a", "b", "c") if rng.randrange(3)}} for i in range(3)}
        routes = {}
        for i in range(3):
            routes[f"r{i}"] = {
                "implemented": bool(rng.randrange(4)), "mode": rng.choice(["fixed", "tree"]),
                "allowed_component_overrides": [name for name in components if rng.randrange(2)],
                "allowed_component_options": {name: rng.sample(list(spec["options"]),
                    rng.randrange(len(spec["options"]) + 1)) for name, spec in components.items()
                    if rng.randrange(2)},
                "source_template_ids": {"source": rng.sample(list(templates), rng.randrange(3))},
                "expert_template_ids": {"source": rng.sample(list(templates), rng.randrange(3))},
                "expert_acknowledgement_id": f"ack{i % 2}",
            }
        registry = {"components": components, "templates": templates, "runner_routes": routes,
                    "authority": {"unnamed_tree_outside_reachability_acknowledgement_id": "outside"}}
        for values in itertools.product([None, "own", "zero", "one", "missing"], repeat=3):
            selected = {name: value for name, value in zip(components, values) if value is not None}
            for modes in (("tree",), ("tree", "fixed")):
                expected = reference(registry, selected, modes)
                if expected[0] == "error":
                    with pytest.raises(PhysicsCapabilityError, match="ambiguous acknowledgements"):
                        _tree_tuple_registry_governance(registry, selected, route_modes=modes)
                else:
                    assert _tree_tuple_registry_governance(registry, selected, route_modes=modes) == expected
