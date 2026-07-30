"""``implemented`` and reachable are independent, so both are declared.

Three schemes reached bitwise parity and had ``implemented: true`` written on
them, and a user could select none of them.  Every template pinned the same
triple (``pbl: ysu``, ``surface_layer: classic-mm5``, ``land_surface: noah``),
the tree route allowed only ``microphysics`` and ``cumulus`` to vary, and the
benchmark route allowed nothing -- so ``implemented: true`` was necessary and
nowhere near sufficient, and nothing said so.

The naive gate would be the biconditional ``implemented <=> reachable``.  It is
the wrong gate, because the two properties are legitimately independent in both
directions:

* ``pbl: off``, ``surface_layer: revised-mm5``, ``radiation: off`` and the
  analytic clear-sky proxy are all implemented and deliberately not offered to
  a source-driven forecast; and
* ``land_surface: ruc-lsm`` is implemented to the point of forecasting,
  restarting and writing output, and is still unreachable because no route can
  produce the nine-level soil state it starts from.

So the intended relationship is DECLARED, per option, in
``reachability.state``, and this file recomputes every state from the templates
and the routes and fails on a difference.  What that buys over the naive gate:
an option that becomes reachable by accident -- a template edit, a route
widening, a source gaining a template list -- fails here with the state it
actually has, and an option that quietly stops being reachable fails the same
way.  The half of the biconditional that is not negotiable is still enforced:
an option that is not implemented must be declared unreachable, and an
implemented one declared unreachable must name its blocker.

The blockers are not prose.  ``test_the_ruc_blocker_is_still_true`` re-runs the
ingest refusal the RUC row cites, so the declaration cannot outlive the reason
for it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gpuwm.physics_registry import physics_registry

#: Easiest first.  An option reachable more than one way is declared by its
#: easiest path, because that is the one a user will find.
_STATE_ORDER = ("template", "component-override", "expert-template",
               "unreachable")

_ROOT = Path(__file__).resolve().parents[1]


def _computed_states(registry: dict) -> dict[tuple[str, str], str]:
    """Recompute every option's reachability from templates and routes.

    Mirrors :func:`gpuwm.physics_registry.validate_physics_plan`'s own rules,
    including the one that used to make this question unanswerable: a route
    that declares any template list must be exhaustive, and a source it
    registers with no list reaches no template.
    """

    components = registry["components"]
    templates = registry["templates"]
    routes = registry["runner_routes"]

    reached: dict[tuple[str, str], set[str]] = {
        (component_id, option_id): set()
        for component_id, component in components.items()
        for option_id in component["options"]
    }

    for route in routes.values():
        if route.get("implemented") is not True:
            continue
        normal = route.get("source_template_ids", {}) or {}
        expert = route.get("expert_template_ids", {}) or {}
        declares = bool(normal) or bool(expert)
        per_domain = route.get("mode") == "experiment-per-domain"
        overridable = set(route.get("allowed_component_overrides", []) or [])
        expert_selector_keys = set(
            route.get("allowed_expert_selector_keys", []) or [])

        for source_id in route.get("source_ids", []) or []:
            if declares:
                normal_ids = list(normal.get(source_id, []) or [])
                expert_ids = list(expert.get(source_id, []) or [])
            else:
                # No declaration at all: every registered template is legal.
                normal_ids, expert_ids = list(templates), []
            for template_id in normal_ids:
                for component_id, option_id in templates[
                        template_id]["components"].items():
                    reached[(component_id, option_id)].add("template")
            for template_id in expert_ids:
                for component_id, option_id in templates[
                        template_id]["components"].items():
                    reached[(component_id, option_id)].add("expert-template")
            if not (normal_ids or expert_ids):
                # An override still needs a base template to override.
                continue
            for component_id, component in components.items():
                selector_keys = set(component.get("selector_keys", []) or [])
                by_override = per_domain and component_id in overridable
                by_selector = bool(selector_keys) and selector_keys <= (
                    expert_selector_keys)
                if not (by_override or by_selector):
                    continue
                for option_id in component["options"]:
                    reached[(component_id, option_id)].add(
                        "component-override")

    states: dict[tuple[str, str], str] = {}
    for (component_id, option_id), ways in reached.items():
        option = components[component_id]["options"][option_id]
        if option.get("implemented") is not True:
            # Nameable is not reachable.  ``microphysics: sase`` sits inside the
            # tree route's allowed_component_overrides, so a plan CAN name it --
            # and gpuwm/physics_registry.py refuses it with
            # ``unimplemented-option`` on every route, from every template, for
            # every source.  Reachable here means "a plan naming it can be
            # launchable", so an unimplemented option is unreachable by
            # construction, and
            # test_an_unimplemented_option_is_refused_by_the_resolver is what
            # holds that construction to the resolver's actual behaviour rather
            # than to this comment.
            states[(component_id, option_id)] = "unreachable"
            continue
        states[(component_id, option_id)] = next(
            (state for state in _STATE_ORDER if state in ways), "unreachable")
    return states


def _declared_states(registry: dict) -> dict[tuple[str, str], dict]:
    return {
        (component_id, option_id): option.get("reachability")
        for component_id, component in registry["components"].items()
        for option_id, option in component["options"].items()
    }


def test_every_option_declares_its_reachability() -> None:
    registry = physics_registry()
    missing = sorted(
        f"{component_id}.{option_id}"
        for (component_id, option_id), declaration
        in _declared_states(registry).items()
        if not isinstance(declaration, dict)
        or declaration.get("state") not in _STATE_ORDER
    )
    assert missing == [], (
        "these options declare no usable reachability.state; a new option must "
        f"say how a user reaches it, or that nobody can: {missing}")


def test_declared_reachability_matches_the_templates_and_routes() -> None:
    registry = physics_registry()
    computed = _computed_states(registry)
    declared = _declared_states(registry)
    wrong = sorted(
        f"{component_id}.{option_id}: declared "
        f"{declared[(component_id, option_id)]['state']!r}, actually "
        f"{state!r}"
        for (component_id, option_id), state in computed.items()
    if declared[(component_id, option_id)]["state"] != state)
    assert wrong == [], (
        "the registry's reachability declarations disagree with what its own "
        "templates and routes make selectable:\n  " + "\n  ".join(wrong))


def test_an_unimplemented_option_is_refused_by_the_resolver() -> None:
    """The half of the biconditional that is not negotiable.

    ``_computed_states`` calls an unimplemented option unreachable.  That is a
    claim about ``validate_physics_plan``, so it is checked against
    ``validate_physics_plan``: name each one on a plan built from the route and
    template that would otherwise reach it, and require the refusal.
    """

    from gpuwm.physics_registry import (PLAN_SCHEMA, registry_sha256,
                                        validate_physics_plan)

    registry = physics_registry()
    unimplemented = [
        (component_id, option_id)
        for component_id, component in sorted(registry["components"].items())
        for option_id, option in sorted(component["options"].items())
        if option.get("implemented") is not True
    ]
    assert unimplemented, (
        "the registry publishes no unimplemented option, so this gate "
        "measured nothing")

    route_id = "tools.prepared_domain_tree_forecast"
    route = registry["runner_routes"][route_id]
    template_id = route["source_template_ids"]["hrrr"][0]
    for component_id, option_id in unimplemented:
        plan = {
            "schema": PLAN_SCHEMA,
            "plan_id": "unimplemented-option-probe-v1",
            "registry_sha256": registry_sha256(),
            "context": {"source_id": "hrrr", "runner_id": route_id,
                        "topology_id": "one-way-nested-v1"},
            "domains": [
                {"domain_id": "d01", "template_id": template_id,
                 "components": {component_id: option_id}},
                {"domain_id": "d02", "template_id": template_id,
                 "components": {component_id: option_id},
                 "parameters": {"spec_exp": 0.0}},
            ],
            "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
        }
        report = validate_physics_plan(plan)
        assert report["launchable"] is False, (
            f"{component_id}.{option_id} is not implemented and a plan naming "
            "it is launchable")
        codes = {error["code"] for error in report["errors"]}
        assert codes & {"unimplemented-option", "component-override-route"}, (
            f"{component_id}.{option_id} is refused, but for none of the "
            f"reasons that make it unreachable: {sorted(codes)}")


def test_an_unreachable_implemented_option_names_its_blocker() -> None:
    """The other half, which a biconditional would have got wrong.

    An implemented option nobody can select is not automatically a defect --
    but it is always a claim that owes a reason, and the reason has to be
    written down where the reader of ``implemented: true`` will see it.
    """

    registry = physics_registry()
    declared = _declared_states(registry)
    missing, spurious = [], []
    for (component_id, option_id), declaration in sorted(declared.items()):
        blocker = declaration.get("blocker")
        if declaration["state"] == "unreachable":
            if not isinstance(blocker, str) or len(blocker.strip()) < 40:
                missing.append(f"{component_id}.{option_id}")
        elif blocker is not None:
            spurious.append(f"{component_id}.{option_id}")
    assert missing == [], (
        f"unreachable options with no stated blocker: {missing}")
    assert spurious == [], (
        f"reachable options carrying a blocker: {spurious}")


def test_an_expert_only_option_is_warned_and_acknowledged() -> None:
    """Expert-only means the route says why and demands consent."""

    registry = physics_registry()
    computed = _computed_states(registry)
    expert_options = {key for key, state in computed.items()
                      if state == "expert-template"}
    assert expert_options, (
        "no option is expert-only, so this gate measured nothing")

    templates = registry["templates"]
    for route_id, route in registry["runner_routes"].items():
        expert = route.get("expert_template_ids", {}) or {}
        if not expert:
            continue
        acknowledgement = route.get("expert_acknowledgement_id")
        assert isinstance(acknowledgement, str) and acknowledgement.strip(), (
            f"{route_id} offers expert templates with no "
            "expert_acknowledgement_id, so consent cannot be required")
        warnings = route.get("expert_warnings") or []
        assert warnings and all(isinstance(text, str) and text.strip()
                                for text in warnings), (
            f"{route_id} offers expert templates with no expert_warnings; an "
            "expert path that does not say what is wrong is just a hidden one")
        for template_ids in expert.values():
            for template_id in template_ids:
                assert template_id in templates, (
                    f"{route_id} offers unregistered expert template "
                    f"{template_id!r}")


def _perturbed(mutate) -> list[str]:
    registry = copy.deepcopy(physics_registry())
    mutate(registry)
    computed = _computed_states(registry)
    declared = _declared_states(registry)
    return sorted(
        f"{component_id}.{option_id}"
        for (component_id, option_id), state in computed.items()
        if declared[(component_id, option_id)]["state"] != state)


def test_the_gate_fails_when_a_template_makes_something_reachable() -> None:
    """A gate nobody has seen fail is not evidence, part one.

    Swap the revised MM5 surface layer into a registered template -- exactly
    the kind of one-word edit that made three ported schemes invisible in the
    first place, run backwards -- and the recomputation must notice.
    """

    def mutate(registry: dict) -> None:
        registry["templates"]["wsm6-ysu-mm5-noah-no-radiation-v1"][
            "components"]["surface_layer"] = "revised-mm5"

    wrong = _perturbed(mutate)
    assert "surface_layer.revised-mm5" in wrong, wrong
    # classic-mm5 does NOT flip: four other templates still select it, which is
    # the asymmetry a per-option declaration captures and a per-template one
    # would not.
    assert "surface_layer.classic-mm5" not in wrong, wrong


def test_the_gate_fails_when_a_route_widens() -> None:
    """Part two: a route override that nobody declared."""

    def mutate(registry: dict) -> None:
        route = registry["runner_routes"]["tools.prepared_domain_tree_forecast"]
        route["allowed_component_overrides"] = sorted(
            set(route["allowed_component_overrides"]) | {"land_surface"})

    wrong = _perturbed(mutate)
    assert "land_surface.noah-mp" in wrong, wrong
    assert "land_surface.off" in wrong, wrong
    # ``ruc-lsm`` is deliberately NOT probed here.  It reaches ``template``
    # through the RUC template on the era5/gfs single-domain routes, and
    # ``_STATE_ORDER`` declares an option by its easiest path, so widening a
    # different route cannot move it.  An option that is already reachable is
    # the one case this control cannot speak about.
    assert "land_surface.ruc-lsm" not in wrong, wrong


def test_the_gate_fails_when_an_expert_template_loses_its_acknowledgement():
    """Part three: an expert path that stops being one.

    Drop the route's acknowledgement id and the resolver must refuse the expert
    template outright -- ``expert-route-policy`` -- rather than quietly promote
    it to a peer of the base templates.  That is the failure mode that matters:
    a warned expert path decaying into an unwarned normal one.
    """

    from gpuwm.physics_registry import (PLAN_SCHEMA, registry_sha256,
                                        validate_physics_plan)

    registry = copy.deepcopy(physics_registry())
    route = registry["runner_routes"]["tools.prepared_domain_tree_forecast"]
    template_id = route["expert_template_ids"]["hrrr"][0]
    route.pop("expert_acknowledgement_id")

    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": "expert-policy-probe-v1",
        "registry_sha256": registry_sha256(registry),
        "context": {"source_id": "hrrr",
                    "runner_id": "tools.prepared_domain_tree_forecast",
                    "topology_id": "one-way-nested-v1"},
        "domains": [
            {"domain_id": "d01", "template_id": template_id},
            {"domain_id": "d02", "template_id": template_id,
             "parameters": {"spec_exp": 0.0}},
        ],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is False
    assert "expert-route-policy" in {
        error["code"] for error in report["errors"]}

    # ...and the shipped route does publish one, so the same plan only needs the
    # consent rather than being structurally broken.
    shipped = validate_physics_plan({
        **plan,
        "registry_sha256": registry_sha256(),
        "expert_acknowledgements": [
            physics_registry()["runner_routes"][
                "tools.prepared_domain_tree_forecast"][
                    "expert_acknowledgement_id"]],
    })
    assert shipped["launchable"] is True, shipped["errors"]


def test_the_ruc_soil_contract_refusal_is_still_true() -> None:
    """Re-run the refusal the RUC row cites, wherever it is now recorded.

    This began as ``test_the_ruc_blocker_is_still_true``: ``ruc-lsm`` was
    declared ``unreachable`` and cited ``gpuwm/ingest/soil_contract.py`` as
    the blocker.  The RUC lane has since registered a RUC template on the
    era5 and gfs single-domain routes, so the option IS reachable and the
    recomputation above is the authority on that -- but the declarative soil
    contract still admits exactly one target, Noah's four layers, so the
    refusal itself did not go away.  It moved from a ``blocker`` to the
    option's warnings, and the point of this test is unchanged: the claim
    must not outlive the code it rests on.  If a soil lane lands a RUC
    target, this fails and the warning has to be revisited.
    """

    from gpuwm.ingest.soil_contract import validate_soil_layer_contract

    composition = json.loads(
        (_ROOT / "configs" / "rw-wps-era5-netcdf-terrain.composition.json"
         ).read_text(encoding="utf-8"))
    contract = copy.deepcopy(composition["soil_layers"])
    validate_soil_layer_contract(contract)          # the Noah target is fine

    # RUC's nine-level grid, as layer bounds rather than WRF's level depths.
    ruc = [0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 2.2, 3.0]
    contract["target_layers"] = [
        {"top": top, "bottom": bottom}
        for top, bottom in zip(ruc, ruc[1:])
    ]
    assert len(contract["target_layers"]) == 9
    with pytest.raises(ValueError, match="target_layers differ"):
        validate_soil_layer_contract(contract)

    option = physics_registry()["components"]["land_surface"][
        "options"]["ruc-lsm"]
    assert option["reachability"]["state"] == "template"
    assert "blocker" not in option["reachability"], (
        "ruc-lsm is reachable through a registered template, so a blocker "
        "here would be a declaration contradicting the routes")
    assert any("soil_contract.py" in warning
               for warning in option.get("warnings", [])), (
        "the soil-contract refusal above still fires, so the option must "
        "still say so somewhere a user reads")


def test_the_mynn_suite_is_reachable_as_a_pair() -> None:
    """The lane's own deliverable, asserted rather than assumed."""

    registry = physics_registry()
    computed = _computed_states(registry)
    assert computed[("pbl", "mynn")] == "template"
    assert computed[("surface_layer", "mynn")] == "template"

    # ...and only ever together, because half a suite is a different model.
    for template_id, template in registry["templates"].items():
        pair = (template["components"]["pbl"] == "mynn",
                template["components"]["surface_layer"] == "mynn")
        assert pair[0] == pair[1], (
            f"template {template_id!r} selects half the MYNN suite; "
            "gpuwm/physics_compat.py refuses that at runtime, so a template "
            "offering it would be a launchable-looking dead end")
