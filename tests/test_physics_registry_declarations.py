"""The registry declares every knob GPUWM implements, and only honest ones.

These pins guard the declaration contract itself rather than plan resolution
(``tests/test_physics_registry.py`` owns that):

* every declared IMPLEMENTED parameter names a real GPUWM configuration
  field, so the registry cannot advertise a control that does not exist;
* every parameter value the registry itself ships -- in a component option
  or a template -- satisfies its own declared spec, so the document cannot
  contradict itself;
* implementation is proven by a resolving ``consuming_read`` citation, never
  by an assertion -- ``tools/check_parameter_claims.py`` is the gate;
* a parameter declared ``implemented: false`` is published for discovery but
  can never be set and never becomes a resolved setting;
* a parameter carrying no citation claims nothing, which is how each
  scheme's owner turns their own knobs on without editing another's rows;
* per-domain template columns resolve by depth below the tree root.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from gpuwm.config import RunConfig
from gpuwm.physics_registry import (
    MORRISON_TEMPLATE_ID,
    canonical_json,
    parameter_is_implemented,
    physics_registry,
    validate_physics_plan,
)

from test_physics_registry import _single_plan, _uniform_tree

#: Knobs GPUWM reads from the experiment schema rather than from RunConfig
#: (gpuwm/experiment.py and gpuwm/case_data.py).  They are whole-experiment
#: values, never per-domain.
_EXPERIMENT_SCHEMA_PARAMETERS = frozenset({
    "p_top", "blend_width", "co2_vmr", "feedback"})

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parameters() -> dict[str, dict]:
    return physics_registry()["parameters"]


def test_every_implemented_parameter_names_a_real_configuration_field():
    """A declared knob must exist in the configuration GPUWM actually reads."""
    run_config_fields = {f.name for f in dataclasses.fields(RunConfig)}
    known = run_config_fields | _EXPERIMENT_SCHEMA_PARAMETERS
    undeclarable = sorted(
        name
        for name, spec in _parameters().items()
        if parameter_is_implemented(spec) and name not in known
    )
    assert undeclarable == [], (
        "registry claims these parameters are implemented but GPUWM has no "
        f"configuration field for them: {undeclarable}"
    )


def test_implementation_is_proven_by_citation_not_by_assertion():
    """Every claim resolves against the tree, checked by the shipped gate."""
    from tools.check_parameter_claims import check

    assert check(_REPO_ROOT) == []

    cited = [
        name for name, spec in _parameters().items()
        if parameter_is_implemented(spec)
    ]
    assert cited, "the registry should claim some implemented knobs"
    for name in cited:
        citation = _parameters()[name].get("consuming_read")
        assert isinstance(citation, str) and citation.strip(), (
            f"{name} is implemented, so it must cite the read that proves it")


def test_a_declaration_without_a_citation_claims_nothing():
    """Uncited rows publish a type; they never assert an implementation.

    This is what lets each scheme's owner turn their own knobs on without
    anyone editing their entries: no citation means no claim, and the knob
    stays unsettable until its owner cites the read that makes it true.
    """
    uncited = sorted(
        name for name, spec in _parameters().items()
        if spec.get("implemented") is not False
        and "consuming_read" not in spec
    )
    for name in uncited:
        spec = _parameters()[name]
        assert not parameter_is_implemented(spec), name
        assert "default" not in spec, (
            f"{name} claims no implementation, so it must not seed a setting")

    if uncited:
        plan = _uniform_tree()
        plan["domains"][1]["parameters"] = {uncited[0]: 1}
        report = validate_physics_plan(plan)
        assert report["launchable"] is False


def test_selector_keys_are_not_duplicated_as_parameters():
    """Scheme selection belongs to selector_keys; two channels would race."""
    registry = physics_registry()
    selectors: set[str] = set()
    for component in registry["components"].values():
        selectors |= set(component.get("selector_keys", ()))
    overlap = sorted(selectors & set(registry["parameters"]))
    assert overlap == [], (
        f"selector keys must not also be declared as parameters: {overlap}"
    )


def test_registry_never_ships_a_value_its_own_spec_rejects():
    """Option and template values must satisfy the declared parameter specs."""
    registry = physics_registry()
    specs = registry["parameters"]
    offenders: list[str] = []

    def check(where: str, values: object, container_implemented: bool = True) -> None:
        if not isinstance(values, dict):
            return
        for name, value in values.items():
            spec = specs.get(name)
            if not isinstance(spec, dict):
                continue
            if not parameter_is_implemented(spec):
                # An unimplemented option may pin the identity of its own
                # unimplemented knobs; that is the port's specification.  A
                # selectable one may not, because the value would go nowhere.
                if container_implemented:
                    offenders.append(
                        f"{where}.{name} sets an unimplemented knob")
                continue
            enum = spec.get("enum")
            if enum is not None and value not in enum:
                offenders.append(f"{where}.{name}={value!r} not in {enum!r}")
            for bound, worse in (("minimum", value.__lt__),
                                 ("maximum", value.__gt__)):
                limit = spec.get(bound)
                if limit is not None and isinstance(value, (int, float)):
                    if worse(limit):
                        offenders.append(
                            f"{where}.{name}={value!r} violates {bound}={limit!r}")

    for component_id, component in registry["components"].items():
        for option_id, option in component["options"].items():
            check(f"components.{component_id}.{option_id}",
                  option.get("parameters"),
                  option.get("implemented") is True)
    for template_id, template in registry["templates"].items():
        check(f"templates.{template_id}", template.get("parameters"))
        for index, column in enumerate(template.get("per_domain_overrides", ())):
            column = {k: v for k, v in column.items() if k != "nominal_dx_m"}
            check(f"templates.{template_id}.per_domain_overrides[{index}]",
                  column)

    assert offenders == [], offenders


def test_unimplemented_parameters_are_published_but_unsettable():
    """implemented=false is a roadmap entry, never a control.

    Scoped to the EXPLICIT roadmap form.  A row that simply carries no
    citation is a different state -- it claims nothing and owes no reason --
    and ``test_a_declaration_without_a_citation_claims_nothing`` covers it.
    """
    unimplemented = sorted(
        name for name, spec in _parameters().items()
        if spec.get("implemented") is False
    )
    assert unimplemented, "the registry should publish the porting roadmap"

    for name in unimplemented:
        spec = _parameters()[name]
        assert not parameter_is_implemented(spec), name
        reason = spec.get("unimplemented_reason")
        assert isinstance(reason, str) and reason.strip(), (
            f"{name} must say why it is not implemented")
        assert "default" not in spec, (
            f"{name} is not implemented and must not seed a resolved setting")

    plan = _uniform_tree()
    plan["domains"][1]["parameters"] = {unimplemented[0]: 1}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "parameter-value" in {error["code"] for error in report["errors"]}


def test_no_unimplemented_parameter_reaches_resolved_settings():
    report = validate_physics_plan(_single_plan())
    assert report["launchable"] is True
    resolved = report["resolved_domains"][0]["settings"]
    leaked = sorted(
        name for name, spec in _parameters().items()
        if not parameter_is_implemented(spec) and name in resolved
    )
    assert leaked == [], leaked


def test_per_domain_template_columns_resolve_by_depth():
    """The verified nest chain varies these knobs down the tree."""
    plan = _uniform_tree(MORRISON_TEMPLATE_ID)
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]

    root, child = report["resolved_domains"]
    assert root["domain_id"] == "d01" and child["domain_id"] == "d02"
    assert root["settings"]["diff_6th_factor"] == 0.12
    assert child["settings"]["diff_6th_factor"] == 0.10
    assert root["settings"]["radt"] == 12.0
    assert child["settings"]["radt"] == 3.0
    assert root["settings"]["epssm"] == 0.5
    assert child["settings"]["epssm"] == 0.1


def test_per_domain_provenance_is_never_a_resolved_setting():
    report = validate_physics_plan(_uniform_tree(MORRISON_TEMPLATE_ID))
    for domain in report["resolved_domains"]:
        assert "nominal_dx_m" not in domain["settings"]


def test_single_domain_plan_takes_the_root_column():
    report = validate_physics_plan(_single_plan(MORRISON_TEMPLATE_ID))
    assert report["launchable"] is True, report["errors"]
    settings = report["resolved_domains"][0]["settings"]
    assert settings["diff_6th_factor"] == 0.12
    assert settings["radt"] == 12.0


def test_registry_file_stays_canonical_after_declaration_growth():
    registry = physics_registry()
    assert canonical_json(registry) == canonical_json(physics_registry())
