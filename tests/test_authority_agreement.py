"""The two gatekeepers must not disagree about any configuration.

GPUWM has two independent authorities over "may this physics suite run", and
until this file existed nothing compared them:

* the registry's ``constraints`` machinery -- ``required_settings``,
  ``forbidden_setting_values`` and ``requires_components`` -- evaluated by
  :func:`gpuwm.physics_registry.validate_physics_plan`, which is what a
  launcher shows a user *before* anything is prepared; and
* :func:`gpuwm.config.validate_run_config`, which every RunConfig passes
  through on both the legacy and the experiment path, and which is therefore
  what actually decides whether a run starts.

They disagreed.  For the MYNN 5/5 suite with Noah the registry raised
``component-dependency: option 'noah' requires surface_layer in
[revised-mm5, classic-mm5]`` while ``validate_run_config`` returned OK -- and
that pair is the exact configuration the MYNN PBL option's own warning cites
as its runtime evidence (``tests/test_mynn_pbl_runtime.py:49``).  WRF was the
tiebreaker and the registry was wrong: ``share/module_check_a_mundo.F`` has no
``sf_sfclay_physics`` constraint for ``sf_surface_physics=2``, and
``phys/module_surface_driver.F:2386-2390`` hands ``SFCLAY_mynn`` the same
``chs/chs2/cqs2/cpm/flhc/flqc/qgh/qsfc`` exchange coefficients
``CASE (LSMSCHEME)`` consumes.  The constraint is gone; this file is what stops
the next one.

Two layers, because they answer different questions:

**Exhaustive component cross-product.**  All 4,032 combinations of the six
components' registered options, run through the shipped constraint evaluator
and through ``validate_run_config`` on the settings that evaluator resolved.
This is exhaustion, not sampling: the property is proven over the whole space
rather than over examples.  It needs a permissive runner route, because the
shipped routes deliberately do not let a user vary the PBL or the land-surface
model per domain -- so the route is synthesised here, and ONLY the route.  The
constraints, the options, the parameter specs and both authorities' code are
the shipped ones.  This is the layer the MYNN/Noah bug lived in, and note that
a reachable-only enumeration would have MISSED it: MYNN was unreachable, so
the disagreement was invisible from the selectable surface.

**Reachable surface.**  Every plan a user can actually build from the shipped
templates, routes, allowed component overrides and allowed per-domain
parameters.  Here the direction that matters is absolute: a plan the registry
calls launchable must survive the runtime's own battery, or the GUI is
offering something that will not start.

Scope, stated so it cannot drift: the two authorities do not have the same
field of view.  A route, a topology, a template registration or a tree-edge
transition is a question a single per-domain RunConfig cannot answer, and
``_REGISTRY_ONLY_CODES`` names those.  ``_SHARED_CODES`` names the ones both
authorities own, and a refusal carrying any of those must be a refusal on both
sides.  An error code in neither set fails the test rather than being ignored,
so a new code has to be classified by whoever adds it.
"""

from __future__ import annotations

import dataclasses
import itertools
from copy import deepcopy
from pathlib import Path

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.physics_compat import NOAHMP_MEASURED_COLUMN_CEILING
from gpuwm.physics_registry import (
    PLAN_SCHEMA,
    physics_registry,
    registry_sha256,
    validate_physics_plan,
)

#: Questions a per-domain RunConfig cannot answer, so the registry may refuse
#: them alone: they are about routing, topology, plan shape or tree edges.
_REGISTRY_ONLY_CODES = frozenset({
    "component-override-route",
    "expert-acknowledgement-required",
    "expert-route-policy",
    "fixed-template-components",
    "graph-setting-constraint",
    "nonuniform-base-template",
    "parameter-route",
    # This asks whether a real-source plan explicitly opted into a moist
    # carrier while MP is off.  RunConfig sees only the resolved boolean and
    # cannot distinguish an explicit parameter from the registry default.
    "real-source-mp-off-requires-explicit-moist",
    "template-route",
    "transition-required-setting",
    # An unimplemented option is refused BEFORE its selectors are projected:
    # gpuwm/physics_registry.py stops at ``unimplemented-option`` and
    # ``continue``s without the ``settings.update(selectors)`` an implemented
    # option gets, so the resolved settings never carry the choice and no
    # RunConfig built from them can see it.  The runtime authority is not
    # excused from the question, it is handed a different one --
    # ``test_an_unimplemented_option_is_refused_when_its_selectors_are_forced``
    # asks it directly, and finding that it answered WRONG for the 1/1
    # radiation pair is what put the ra_lw_physics=1 refusal into
    # gpuwm/config.py.
    "unimplemented-option",
    "unknown-expert-acknowledgement",
    "unsupported-component-transition",
})

#: The physics-suite question both authorities own.  A registry refusal
#: carrying any of these must also be a ``validate_run_config`` refusal.
_SHARED_CODES = frozenset({
    "component-dependency",
    "component-forbidden-setting",
    "component-required-setting",
    "parameter-value",
    "unimplemented-selector",
    "unknown-selector-combination",
})

_PERMISSIVE_RUNNER = "test.exhaustive-component-cross-product"

#: Small enough that the Noah-MP column rail is not what is being measured
#: here (``tests/test_noahmp_column_budget.py`` owns that), and legal for every
#: other invariant in the battery.
_COLUMNS_NX = 16
_COLUMNS_NY = 16
_GEOMETRY = {
    "nx": _COLUMNS_NX, "ny": _COLUMNS_NY, "nz": 20,
    "dx": 3000.0, "dy": 3000.0, "ztop": 20000.0,
    "dt": 6.0, "run_seconds": 60.0,
}

_RUN_CONFIG_FIELDS = frozenset(
    field.name for field in dataclasses.fields(RunConfig))


def test_the_cross_product_geometry_stays_inside_the_noahmp_rail() -> None:
    """Keep this file measuring coupling, not grid width."""
    assert _COLUMNS_NX * _COLUMNS_NY <= NOAHMP_MEASURED_COLUMN_CEILING


def _base_template_id(registry: dict) -> str:
    """The template every synthesised plan starts from.

    Any template works -- every component is overridden explicitly -- so the
    choice is the one whose parameters are the plainest: no RRTMGP
    compatibility token, no per-domain override columns.
    """

    return "wsm6-ysu-mm5-noah-no-radiation-v1"


def _permissive_registry() -> dict:
    """The shipped registry plus one route that permits every override.

    Nothing else is touched.  In particular no option, constraint, parameter
    spec or warning is modified, so the constraint evaluation exercised below
    is the shipped one.
    """

    registry = physics_registry()
    registry["runner_routes"][_PERMISSIVE_RUNNER] = {
        "allowed_component_overrides": sorted(registry["components"]),
        "allowed_expert_selector_keys": [],
        "allowed_expert_setting_keys": [],
        "allowed_parameter_keys": [],
        "implemented": True,
        "mode": "experiment-per-domain",
        "require_explicit_components": True,
        "source_ids": ["*"],
        "source_template_ids": {},
        "topology_ids": ["single-domain-v1"],
    }
    return registry


def _single_domain_plan(registry: dict, runner_id: str, source_id: str,
                        template_id: str, components: dict | None = None,
                        parameters: dict | None = None,
                        acknowledgements: list | None = None) -> dict:
    domain: dict = {"domain_id": "d01", "template_id": template_id}
    if components:
        domain["components"] = dict(components)
    if parameters:
        domain["parameters"] = dict(parameters)
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": "authority-agreement-probe-v1",
        "registry_sha256": registry_sha256(registry),
        "context": {"source_id": source_id, "runner_id": runner_id,
                    "topology_id": "single-domain-v1"},
        "domains": [domain],
        "edges": [],
    }
    if acknowledgements:
        plan["acknowledgements"] = list(acknowledgements)
    return plan


def _tree_plan(registry: dict, runner_id: str, source_id: str,
               template_id: str, components: dict | None = None,
               parameters: dict | None = None,
               acknowledgements: list | None = None) -> dict:
    def domain(domain_id: str, root: bool) -> dict:
        entry: dict = {"domain_id": domain_id, "template_id": template_id}
        if components:
            entry["components"] = dict(components)
        merged = dict(parameters or {})
        if not root:
            merged["spec_exp"] = 0.0
        if merged:
            entry["parameters"] = merged
        return entry

    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": "authority-agreement-tree-probe-v1",
        "registry_sha256": registry_sha256(registry),
        "context": {"source_id": source_id, "runner_id": runner_id,
                    "topology_id": "one-way-nested-v1"},
        "domains": [domain("d01", True), domain("d02", False)],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }
    if acknowledgements:
        plan["acknowledgements"] = list(acknowledgements)
    return plan


def _run_config(settings: dict, *, nested: bool) -> RunConfig:
    """Build the RunConfig the resolved settings describe.

    Only registry settings that name a real RunConfig field are carried;
    experiment-scope knobs (``p_top``, ``blend_width``, ``co2_vmr``) belong to
    the experiment schema and unimplemented knobs never reach ``settings`` at
    all (``tests/test_physics_registry_declarations.py`` owns both).
    """

    keywords = {name: value for name, value in settings.items()
                if name in _RUN_CONFIG_FIELDS}
    keywords.update(_GEOMETRY)
    keywords["nested"] = nested
    return RunConfig(**keywords)


def _config_refusal(settings: dict, *, nested: bool) -> str | None:
    try:
        validate_run_config(_run_config(settings, nested=nested))
    except (ValueError, NotImplementedError, TypeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _declared_environment(report: dict) -> dict[str, str]:
    """Environment the plan's OWN asset requirements say a launch needs.

    The registry may publish an external table set as an asset requirement
    with ``enable_environment`` and ``root_environment`` rather than as an
    error, the same way it publishes bundled package data: a prerequisite a
    launcher checks, not a malformed choice.  So the honest comparison is
    "given the prerequisites the plan itself declares, do the two authorities
    agree".  Since the mp8 promotion (packaged Thompson tables, product/v1
    packaging lane 2026-07-28) no shipped requirement declares either key --
    Thompson's entry is a packaged-table-set with only an optional
    root-override env -- so this helper currently returns {} for every plan;
    the mechanism stays because it is the declared contract for any future
    external table set.
    """

    environment: dict[str, str] = {}
    for entry in report.get("asset_requirements", []):
        requirement = entry.get("requirement", {})
        enable = requirement.get("enable_environment", {})
        if isinstance(enable, dict):
            for name, value in enable.items():
                environment[str(name)] = str(value)
        root = requirement.get("root_environment")
        if isinstance(root, str) and root:
            # Any non-empty path satisfies the readiness authority; the BYTES
            # are validated at launch, which is not this gate's question.
            environment[root] = str(Path(__file__).resolve().parent)
    return environment


def _compare(label: str, report: dict, disagreements: list[str],
             registry: dict | None = None,
             monkeypatch=None) -> None:
    """Record any way in which the two authorities differ about ``report``."""

    registry = physics_registry() if registry is None else registry
    codes = {error["code"] for error in report["errors"]}
    unclassified = sorted(codes - _REGISTRY_ONLY_CODES - _SHARED_CODES)
    if unclassified:
        disagreements.append(
            f"{label}: registry raised unclassified error code(s) "
            f"{unclassified}; classify them in _REGISTRY_ONLY_CODES or "
            f"_SHARED_CODES so this gate keeps its meaning")
        return

    shared = sorted(codes & _SHARED_CODES)
    environment = _declared_environment(report)
    for index, resolved in enumerate(report["resolved_domains"]):
        nested = index > 0
        if environment and monkeypatch is not None:
            for name, value in environment.items():
                monkeypatch.setenv(name, value)
            try:
                refusal = _config_refusal(resolved["settings"], nested=nested)
            finally:
                for name in environment:
                    monkeypatch.delenv(name, raising=False)
        else:
            refusal = _config_refusal(resolved["settings"], nested=nested)
        if report["launchable"]:
            if refusal is not None:
                disagreements.append(
                    f"{label} [{resolved['domain_id']}]: registry says "
                    f"LAUNCHABLE, validate_run_config REFUSES -- {refusal}")
        elif shared:
            if refusal is None:
                disagreements.append(
                    f"{label} [{resolved['domain_id']}]: registry REFUSES on "
                    f"{shared}, validate_run_config says OK")


def _component_option_ids(registry: dict) -> dict[str, list[str]]:
    return {
        component_id: sorted(component["options"])
        for component_id, component in sorted(registry["components"].items())
    }


def test_exhaustive_component_cross_product_agrees_on_every_combination(
        monkeypatch):
    """Every combination of every registered option, both authorities."""

    registry = _permissive_registry()
    template_id = _base_template_id(registry)
    options = _component_option_ids(registry)
    component_ids = sorted(options)

    disagreements: list[str] = []
    combinations = 0
    for values in itertools.product(*(options[c] for c in component_ids)):
        components = dict(zip(component_ids, values))
        combinations += 1
        plan = _single_domain_plan(
            registry, _PERMISSIVE_RUNNER, "any-source", template_id,
            components=components)
        report = validate_physics_plan(plan, registry=registry)
        _compare(repr(components), report, disagreements, registry=registry,
                 monkeypatch=monkeypatch)

    expected = 1
    for component_id in component_ids:
        expected *= len(options[component_id])
    assert combinations == expected, (
        f"the cross-product enumerated {combinations} of {expected} "
        "combinations")
    assert combinations >= 4032, (
        "the registry should carry at least the component options this gate "
        f"was written against; enumerated {combinations}")
    assert disagreements == [], (
        f"{len(disagreements)} of {combinations} component combinations are "
        "decided differently by the registry and by validate_run_config:\n  "
        + "\n  ".join(disagreements[:40]))


def _selectable_parameter_values(registry: dict, name: str) -> list:
    """Values of a route-allowed per-domain parameter the registry declares.

    Enumerable specs are enumerated exhaustively; a continuous one contributes
    only its declared default, because a numeric range is not a selection and
    ``_parameter_error`` already gates its bounds.
    """

    spec = registry["parameters"].get(name)
    if not isinstance(spec, dict):
        return []
    if spec.get("type") == "boolean":
        return [False, True]
    enum = spec.get("enum")
    if isinstance(enum, list):
        return list(enum)
    return [spec["default"]] if "default" in spec else []


def _reachable_plans(registry: dict):
    """Every plan a user can build from the shipped templates and routes."""

    routes = registry["runner_routes"]
    for runner_id, route in sorted(routes.items()):
        if route.get("implemented") is not True:
            continue
        acknowledgement = route.get("expert_acknowledgement_id")
        expert = route.get("expert_template_ids", {}) or {}
        normal = route.get("source_template_ids", {}) or {}
        builder = (_tree_plan if "one-way-nested-v1" in route["topology_ids"]
                   else _single_domain_plan)
        overridable = (
            sorted(route.get("allowed_component_overrides", []))
            if route.get("mode") == "experiment-per-domain" else [])
        parameter_axes = {
            name: _selectable_parameter_values(registry, name)
            for name in sorted(route.get("allowed_parameter_keys", []))
        }
        parameter_axes = {name: values
                          for name, values in parameter_axes.items() if values}

        for source_id in sorted(route.get("source_ids", [])):
            template_ids = list(normal.get(source_id, []))
            expert_ids = list(expert.get(source_id, []))
            for template_id in template_ids + expert_ids:
                acknowledgements = (
                    [acknowledgement]
                    if template_id in expert_ids and acknowledgement else None)
                # The template alone, then one axis at a time: a full Cartesian
                # product over overrides AND parameters would multiply into
                # tens of thousands of plans without exercising a coupling the
                # exhaustive layer above has not already decided.
                yield (runner_id, source_id, template_id,
                       builder(registry, runner_id, source_id, template_id,
                               acknowledgements=acknowledgements))
                for component_id in overridable:
                    for option_id in sorted(
                            registry["components"][component_id]["options"]):
                        yield (runner_id, source_id, template_id,
                               builder(registry, runner_id, source_id,
                                       template_id,
                                       components={component_id: option_id},
                                       acknowledgements=acknowledgements))
                for name, values in parameter_axes.items():
                    for value in values:
                        yield (runner_id, source_id, template_id,
                               builder(registry, runner_id, source_id,
                                       template_id,
                                       parameters={name: value},
                                       acknowledgements=acknowledgements))


def test_every_reachable_plan_the_registry_calls_launchable_actually_validates(
        monkeypatch):
    """The direction a user feels: offered means startable."""

    registry = physics_registry()
    disagreements: list[str] = []
    plans = 0
    launchable = 0
    for runner_id, source_id, template_id, plan in _reachable_plans(registry):
        plans += 1
        report = validate_physics_plan(plan)
        if report["launchable"]:
            launchable += 1
        _compare(f"{runner_id} {source_id} {template_id} "
                 f"{plan['domains'][0].get('components', {})}"
                 f"{plan['domains'][0].get('parameters', {})}",
                 report, disagreements, registry=registry,
                 monkeypatch=monkeypatch)

    assert plans > 100, f"the reachable surface enumerated only {plans} plans"
    assert launchable > 0, "no reachable plan is launchable at all"
    assert disagreements == [], (
        f"{len(disagreements)} of {plans} reachable plans are decided "
        "differently by the registry and by validate_run_config:\n  "
        + "\n  ".join(disagreements[:40]))


def test_the_gate_fails_when_an_authority_is_perturbed():
    """A gate nobody has seen fail is not evidence.

    Reintroduce the exact constraint that was wrong -- Noah requiring an MM5
    surface layer -- and the comparison must report the MYNN 5/5 pair.  The
    perturbation is applied to a deep copy, so the shipped registry is not
    touched.
    """

    registry = _permissive_registry()
    noah = registry["components"]["land_surface"]["options"]["noah"]
    noah["constraints"]["requires_components"] = {
        "surface_layer": ["revised-mm5", "classic-mm5"]}

    components = {
        "cumulus": "off", "land_surface": "noah", "microphysics": "wsm6-mp6",
        "pbl": "mynn", "radiation": "dudhia-shortwave",
        "surface_layer": "mynn",
    }
    plan = _single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry), components=components)
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is False
    assert "component-dependency" in {
        error["code"] for error in report["errors"]}

    disagreements: list[str] = []
    _compare("perturbed-noah", report, disagreements)
    assert len(disagreements) == 1, disagreements
    assert "registry REFUSES" in disagreements[0]
    assert "component-dependency" in disagreements[0]

    # ...and the shipped registry still agrees about that same pair.
    shipped = _permissive_registry()
    unperturbed = validate_physics_plan(
        _single_domain_plan(shipped, _PERMISSIVE_RUNNER, "any-source",
                            _base_template_id(shipped),
                            components=components),
        registry=shipped)
    assert unperturbed["launchable"] is True, unperturbed["errors"]
    agreed: list[str] = []
    _compare("shipped-noah", unperturbed, agreed)
    assert agreed == []


def test_the_gate_fails_when_the_runtime_authority_is_the_wrong_one():
    """The other direction: launchable but unstartable is also a failure."""

    registry = _permissive_registry()
    components = {
        "cumulus": "off", "land_surface": "noah", "microphysics": "wsm6-mp6",
        "pbl": "ysu", "radiation": "dudhia-shortwave",
        "surface_layer": "classic-mm5",
    }
    report = validate_physics_plan(
        _single_domain_plan(registry, _PERMISSIVE_RUNNER, "any-source",
                            _base_template_id(registry),
                            components=components),
        registry=registry)
    assert report["launchable"] is True, report["errors"]

    settings = dict(report["resolved_domains"][0]["settings"])
    # km_opt=1 with the template's zero khdif/kvdif is fine; km_opt=4 with a
    # nonzero one is exactly what validate_run_config refuses, and the registry
    # spec for khdif is a bare non-negative number, so it cannot see it.
    settings["khdif"] = 100.0
    assert _config_refusal(settings, nested=False) is not None
    forged = deepcopy(report)
    forged["resolved_domains"][0]["settings"] = settings
    disagreements: list[str] = []
    _compare("forged-khdif", forged, disagreements)
    assert len(disagreements) == 1, disagreements
    assert "registry says LAUNCHABLE" in disagreements[0]


def test_an_unimplemented_option_is_refused_when_its_selectors_are_forced():
    """The question the cross-product cannot ask, asked directly.

    ``validate_physics_plan`` refuses an unimplemented option without ever
    projecting its selectors into the resolved settings, so the cross-product
    above can only see the registry's half of that refusal.  A namelist import
    or a hand-written TOML has no such courtesy: it puts the selector value
    straight into a RunConfig.  So force each unimplemented option's selectors
    onto an otherwise valid configuration and require the runtime authority to
    refuse them too.

    This found a real hole.  ``ra_lw_physics=1`` -- WRF's RRTM longwave, the
    ``wrf-rrtm-dudhia`` pair -- was inside ``validate_run_config``'s accepted
    set ``(0, 1, 4, 90)`` and no readiness blocker covered it, so a config
    naming it validated cleanly and then raised NotImplementedError from
    ``gpuwm/core/physics.py`` initialize_physics at driver construction.
    """

    registry = physics_registry()
    report = validate_physics_plan(
        _single_domain_plan(registry, "tools.prepared_single_domain_forecast",
                            "gfs", _base_template_id(registry)))
    assert report["launchable"] is True, report["errors"]
    baseline = dict(report["resolved_domains"][0]["settings"])
    assert _config_refusal(baseline, nested=False) is None

    exercised = []
    for component_id, component in sorted(registry["components"].items()):
        for option_id, option in sorted(component["options"].items()):
            if option.get("implemented") is True:
                continue
            selectors = option.get("selectors") or {}
            if not selectors:
                # No projection exists at all; the reachability declaration is
                # what carries this option's honesty
                # (tests/test_registry_reachability.py).
                continue
            forced = dict(baseline)
            forced.update(selectors)
            refusal = _config_refusal(forced, nested=False)
            exercised.append((f"{component_id}.{option_id}", refusal))
            assert refusal is not None, (
                f"{component_id}.{option_id} is registered as NOT implemented "
                f"and its selectors {selectors} are accepted by "
                "validate_run_config; a namelist or TOML naming them would "
                "validate and then fail at driver construction")
    assert exercised, (
        "no unimplemented option declares selectors, so this gate measured "
        "nothing; if that is genuinely true, delete it rather than leave it "
        "passing vacuously")


@pytest.mark.parametrize("code", sorted(_SHARED_CODES | _REGISTRY_ONLY_CODES))
def test_every_classified_code_is_a_code_the_validator_can_emit(code: str):
    """Classification cannot drift into naming codes that no longer exist."""

    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "gpuwm"
              / "physics_registry.py").read_text(encoding="utf-8")
    assert f'"{code}"' in source, (
        f"{code!r} is classified here but gpuwm/physics_registry.py never "
        "emits it")
