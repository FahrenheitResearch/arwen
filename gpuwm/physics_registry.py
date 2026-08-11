"""Data-driven GPUWM physics inventory and per-domain plan validation.

This module is intentionally usable with the Python standard library alone.
It describes executable components; it does not import or initialize NumPy,
CuPy, CUDA, forecast drivers, or physics implementations.  The v2 contract is
additive: existing v1 launchers continue to own execution until a caller
explicitly consumes a validated v2 plan.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


REGISTRY_SCHEMA = "gpuwm-physics-registry-v2"
PLAN_SCHEMA = "gpuwm-physics-plan-v2"
VALIDATION_SCHEMA = "gpuwm-physics-plan-validation-v2"

WSM6_TEMPLATE_ID = "wsm6-ysu-mm5-noah-no-radiation-v1"
THOMPSON_TEMPLATE_ID = "thompson-mp8-ysu-mm5-noah-validation-v1"
THOMPSON_KF_TEMPLATE_ID = "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1"
MORRISON_TEMPLATE_ID = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"
NSSL2_TEMPLATE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1"
)
NSSL2_LEGACY_RRTMG_TEMPLATE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1"
)

#: The registered template whose suite `gpuwm domain` emits by default
#: (product decision, 2026-07-29): Thompson mp8 in the certified
#: real-data set.  Morrison stays registered and fully selectable at its
#: own maturity label; a test binds the wizard's emitted selectors to
#: this template's components so the two cannot drift apart.
DEFAULT_TEMPLATE_ID = THOMPSON_KF_TEMPLATE_ID

_PINNED_REGISTRY_ENV = "GPUWM_PINNED_PHYSICS_REGISTRY"
_PINNED_REGISTRY_SHA256_ENV = "GPUWM_PINNED_PHYSICS_REGISTRY_SHA256"
_pinned_path = os.environ.get(_PINNED_REGISTRY_ENV)
_pinned_sha256 = os.environ.get(_PINNED_REGISTRY_SHA256_ENV)
if (_pinned_path is None) != (_pinned_sha256 is None):
    raise RuntimeError(
        "pinned physics registry path and SHA-256 must be supplied together")
_REGISTRY_PATH = Path(
    _pinned_path or Path(__file__).with_name("physics_registry_v2.json")
).resolve()
try:
    _REGISTRY_BYTES = _REGISTRY_PATH.read_bytes()
except OSError as exc:
    raise RuntimeError(
        f"GPUWM physics registry is unreadable: {_REGISTRY_PATH}") from exc
if _pinned_sha256 is not None:
    if (len(_pinned_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in _pinned_sha256)
            or hashlib.sha256(_REGISTRY_BYTES).hexdigest()
            != _pinned_sha256):
        raise RuntimeError(
            "pinned physics registry raw SHA-256 differs before JSON decode")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def iter_maturity_surfaces(registry: Mapping[str, object]):
    """Yield ``(path, maturity)`` for every surface carrying a maturity.

    The walk is structural rather than a list of known surfaces, so a
    maturity introduced on a surface nobody thought of is still checked.
    ``maturity_ladder`` and ``evidence_axes`` are the vocabulary itself and
    are skipped: their rung names are definitions, not uses.
    """

    def walk(node: object, path: str):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "maturity" and isinstance(value, str):
                    yield f"{path}.maturity", value
                else:
                    yield from walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(value, f"{path}[{index}]")

    for key, value in registry.items():
        if key in ("maturity_ladder", "evidence_axes"):
            continue
        yield from walk(value, key)


def _enforce_evidence_axes(registry: Mapping[str, object]) -> None:
    """Load-time two-axis membership enforcement (owner decision D-22).

    Enforced here, at the loader, because the repository's own ethos is
    that a malformed document does not get to reach a caller.  What is
    enforced is MEMBERSHIP on both axes -- every maturity value at every
    surface names a rung, every implemented option's scientific_evidence
    names an enum entry -- which cannot take a working template out of
    service: a template only fails it by carrying a word the vocabulary
    does not define.  The composition rule is deliberately NOT enforced
    here (D-16): clause C2 flags six shipped templates, and a blocking
    loader would take WSM6 and both NSSL-2 profiles out of service, so it
    is a registry-document invariant that a test enforces instead.
    """

    ladder = registry.get("maturity_ladder")
    if not isinstance(ladder, dict):
        raise RuntimeError("bundled GPUWM physics registry lacks the "
                           "maturity_ladder block")
    order = ladder.get("order")
    rungs = ladder.get("rungs")
    if not isinstance(order, list) or not isinstance(rungs, dict):
        raise RuntimeError("maturity_ladder must carry an order list and a "
                           "rungs map")
    if list(rungs) and set(order) != set(rungs):
        raise RuntimeError(
            "maturity_ladder.order and maturity_ladder.rungs disagree: "
            f"{sorted(set(order) ^ set(rungs))}")

    axes = registry.get("evidence_axes")
    if not isinstance(axes, dict):
        raise RuntimeError("bundled GPUWM physics registry lacks the "
                           "evidence_axes block")
    for axis_key in ("maturity", "scientific"):
        if not isinstance(axes.get(axis_key), dict):
            raise RuntimeError(f"evidence_axes lacks the {axis_key!r} axis")
    if axes.get("conformance_implies_scientific_validation") is not False:
        raise RuntimeError(
            "evidence_axes.conformance_implies_scientific_validation must be "
            "present and false: agreement with WRF is agreement with a "
            "model and never implies scientific validation")
    if axes["maturity"].get("rungs") != rungs:
        raise RuntimeError(
            "evidence_axes.maturity.rungs and maturity_ladder.rungs are two "
            "orderings of the same names; they must be identical")

    in_use = set()
    for path, maturity in iter_maturity_surfaces(registry):
        if maturity not in rungs:
            raise RuntimeError(
                f"{path} carries maturity {maturity!r}, which names no rung "
                f"of maturity_ladder; rungs are {sorted(rungs)}")
        in_use.add(maturity)
    if in_use != set(rungs):
        raise RuntimeError(
            "maturity_ladder.rungs must equal the maturity values in use; "
            f"unused rungs {sorted(set(rungs) - in_use)}, "
            f"undeclared values {sorted(in_use - set(rungs))}")

    scientific_enum = axes["scientific"].get("enum")
    if not isinstance(scientific_enum, dict) or not scientific_enum:
        raise RuntimeError("evidence_axes.scientific.enum is missing")
    components = registry.get("components")
    for component_id, component in (components or {}).items():
        if not isinstance(component, dict):
            continue
        for option_id, option in component.get("options", {}).items():
            if not isinstance(option, dict) or option.get(
                    "implemented") is not True:
                continue
            value = option.get("scientific_evidence")
            if value not in scientific_enum:
                raise RuntimeError(
                    f"components.{component_id}.options.{option_id}"
                    f".scientific_evidence is {value!r}, which names no "
                    f"entry of evidence_axes.scientific.enum")

    policy = registry.get("warning_policy")
    policy = policy if isinstance(policy, dict) else {}
    for tier, key in (("nonwarning", "nonwarning_maturities"),
                      ("warn", "warn_maturities")):
        computed = [
            name for name in order
            if isinstance(rungs.get(name), dict)
            and rungs[name].get("warning_tier") == tier
        ]
        if policy.get(key) != computed:
            raise RuntimeError(
                f"warning_policy.{key} is {policy.get(key)!r} but the ladder "
                f"computes {computed!r}; the ladder is the single source")


def _load_registry() -> dict[str, object]:
    try:
        text = _REGISTRY_BYTES.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("GPUWM physics registry is not UTF-8") from exc
    value = json.loads(text, parse_constant=_reject_json_constant)
    if not isinstance(value, dict) or value.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("bundled GPUWM physics registry v2 schema drift")
    if value.get("plan_schema") != PLAN_SCHEMA:
        raise RuntimeError("bundled GPUWM physics plan schema drift")
    if value.get("validation_schema") != VALIDATION_SCHEMA:
        raise RuntimeError("bundled GPUWM physics validation schema drift")
    _enforce_evidence_axes(value)
    return value


_REGISTRY = _load_registry()

#: The one ordering of maturity names in this module, read from the
#: generated document rather than restated here.  Every consumer that needs
#: to compare two maturities -- the warning tiers below, the composition
#: ceiling in tests/test_physics_registry_composition.py -- reads this.
MATURITY_RANK: dict[str, int] = {
    name: rank
    for rank, name in enumerate(_REGISTRY["maturity_ladder"]["order"])
}


def maturity_rank(maturity: object) -> int | None:
    """Rank of a maturity on the conformance ladder, or None if unknown."""

    return MATURITY_RANK.get(maturity) if isinstance(maturity, str) else None


def _ladder_tier(registry: Mapping[str, object], tier: str) -> list[str]:
    """Return the ladder-ordered maturities in one warning tier."""

    ladder = registry.get("maturity_ladder")
    ladder = ladder if isinstance(ladder, dict) else {}
    order = ladder.get("order")
    rungs = ladder.get("rungs")
    if not isinstance(order, list) or not isinstance(rungs, dict):
        return []
    return [
        name for name in order
        if isinstance(rungs.get(name), dict)
        and rungs[name].get("warning_tier") == tier
    ]


def canonical_json(value: object) -> str:
    """Return the one byte-stable JSON rendering used by registry hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    """Hash canonical UTF-8 JSON rather than Python insertion order."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def physics_registry() -> dict[str, object]:
    """Return an independent copy of the immutable built-in registry."""

    return deepcopy(_REGISTRY)


def registry_sha256(registry: Mapping[str, object] | None = None) -> str:
    """Return the semantic hash of the selected registry document."""

    return canonical_sha256(physics_registry() if registry is None else registry)


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _is_json_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def parameter_is_implemented(spec: Mapping[str, object]) -> bool:
    """Whether a declared parameter reaches a GPUWM runtime component.

    Implementation is proven by ``consuming_read`` -- a repo-relative file
    that actually reads the knob -- and never by an unbacked assertion.  A
    declaration carrying no citation makes no claim: it publishes the knob's
    type for discovery while leaving it unsettable, which is what lets each
    scheme's owner turn their own knobs on, by citing the read that makes it
    true, without editing anyone else's rows.

    ``implemented: false`` is the explicit roadmap form and additionally
    carries ``unimplemented_reason`` for a user interface to display.

    ``tools/check_parameter_claims.py`` is the gate: it re-resolves every
    citation, so a claim cannot outlive the code that justified it.
    """

    if spec.get("implemented") is False:
        return False
    citation = spec.get("consuming_read")
    return isinstance(citation, str) and bool(citation.strip())


def _parameter_error(spec: Mapping[str, object], value: object) -> str | None:
    if not parameter_is_implemented(spec):
        reason = spec.get("unimplemented_reason")
        detail = f": {reason}" if isinstance(reason, str) and reason else ""
        return f"is declared but not implemented{detail}"
    expected = spec.get("type")
    if expected == "boolean":
        valid = isinstance(value, bool)
    elif expected == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if valid:
            try:
                valid = math.isfinite(float(value))
            except (OverflowError, TypeError, ValueError):
                valid = False
    elif expected == "string":
        valid = isinstance(value, str)
    else:
        valid = _is_json_scalar(value)
    if not valid:
        return f"must have registry type {expected}"
    if "enum" in spec and value not in spec["enum"]:
        return f"must be one of {spec['enum']!r}"
    if "minimum" in spec:
        try:
            below_minimum = value < spec["minimum"]  # type: ignore[operator]
        except (TypeError, ValueError):
            return f"must be comparable with minimum {spec['minimum']}"
        if below_minimum:
            return f"must be >= {spec['minimum']}"
    if "maximum" in spec:
        try:
            above_maximum = value > spec["maximum"]  # type: ignore[operator]
        except (TypeError, ValueError):
            return f"must be comparable with maximum {spec['maximum']}"
        if above_maximum:
            return f"must be <= {spec['maximum']}"
    return None


def _same_value(left: object, right: object) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return False


def _conditional_refusals(constraints: Mapping[str, object]) -> list[dict]:
    """``constraints.refused_when``, the CONDITIONAL refusal rules.

    The other three constraint kinds each ask ONE question:
    ``required_settings`` about a setting, ``forbidden_setting_values``
    about a setting's value, ``requires_components`` about a sibling
    component.  A real coupling is sometimes a conjunction across those
    kinds, and until 1.9 the registry could not say one: mp_physics=9 is
    refused with RTE+RRTMGP only when the RTE+RRTMGP ADAPTER is selected,
    and the adapter is chosen by the ``ra_rrtmg_variant`` PARAMETER over
    the same radiation component option that the legacy adapter uses.  The
    coupling was therefore written as prose in ``extensions`` -- which
    nothing evaluates -- and the registry told launchers a configuration
    was startable that ``validate_run_config`` refused, 320 combinations
    of it (tests/test_authority_agreement.py).

    A rule is a mapping with a human ``reason`` and any of:

    ``components``
        ``{component_id: [option_id, ...]}`` -- every named component must
        resolve to one of the listed options.
    ``settings``
        ``{setting: [value, ...]}`` -- every named setting must resolve to
        one of the listed values.  A setting the plan does not carry falls
        back to the registry's own declared default for that parameter, so
        a rule cannot be dodged by leaving the knob at its default.

    An empty rule fires on everything, which is never what anyone means,
    so it is treated as no rule at all rather than as an unconditional
    refusal.
    """

    rules = constraints.get("refused_when", [])
    if not isinstance(rules, list):
        return []
    return [
        rule for rule in rules
        if isinstance(rule, dict)
        and isinstance(rule.get("reason"), str)
        and (isinstance(rule.get("components"), dict)
             or isinstance(rule.get("settings"), dict))
    ]


def _effective_setting(
    name: str,
    settings: Mapping[str, object],
    parameter_specs: Mapping[str, object],
) -> object:
    """A setting's resolved value, or the registry's declared default."""

    if name in settings:
        return settings[name]
    spec = parameter_specs.get(name)
    return spec.get("default") if isinstance(spec, Mapping) else None


def _conditional_refusal_fires(
    rule: Mapping[str, object],
    resolved_components: Mapping[str, str],
    settings: Mapping[str, object],
    parameter_specs: Mapping[str, object],
) -> bool:
    """Does every clause of one :func:`_conditional_refusals` rule hold?"""

    required_components = rule.get("components")
    if isinstance(required_components, Mapping):
        for component_id, option_ids in required_components.items():
            if not isinstance(option_ids, list):
                return False
            if resolved_components.get(component_id) not in option_ids:
                return False
    required_settings = rule.get("settings")
    if isinstance(required_settings, Mapping):
        for name, values in required_settings.items():
            if not isinstance(values, list):
                return False
            observed = _effective_setting(name, settings, parameter_specs)
            if not any(_same_value(observed, value) for value in values):
                return False
    return True


def _empty_validation(
    registry: Mapping[str, object], plan: object
) -> dict[str, object]:
    try:
        plan_hash = canonical_sha256(plan)
    except (TypeError, ValueError):
        plan_hash = None
    return {
        "schema": VALIDATION_SCHEMA,
        "launchable": False,
        "errors": [],
        "warnings": [],
        "registry_sha256": registry_sha256(registry),
        "plan_sha256": plan_hash,
        "plan_id": None,
        "context": None,
        "acknowledgements": [],
        "acknowledgement_provenance": {},
        "resolved_domains": [],
        "asset_requirements": [],
    }


def validate_physics_plan(
    plan: object,
    *,
    registry: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve a v2 plan without importing or executing any physics code."""

    selected_registry = physics_registry() if registry is None else deepcopy(registry)
    result = _empty_validation(selected_registry, plan)
    errors: list[dict[str, str]] = result["errors"]  # type: ignore[assignment]
    warnings: list[dict[str, str]] = result["warnings"]  # type: ignore[assignment]

    if selected_registry.get("schema") != REGISTRY_SCHEMA:
        errors.append(
            _issue(
                "registry-schema",
                "registry.schema",
                f"expected {REGISTRY_SCHEMA!r}",
            )
        )
        return result
    if not isinstance(plan, dict):
        errors.append(_issue("plan-type", "$", "physics plan must be a JSON object"))
        return result
    if plan.get("schema") != PLAN_SCHEMA:
        errors.append(
            _issue("plan-schema", "schema", f"expected {PLAN_SCHEMA!r}")
        )

    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id.strip():
        errors.append(
            _issue("plan-id", "plan_id", "must be a non-empty string")
        )
    else:
        result["plan_id"] = plan_id

    bound_registry_sha256 = plan.get("registry_sha256")
    actual_registry_sha256 = result["registry_sha256"]
    if not isinstance(bound_registry_sha256, str) or not bound_registry_sha256:
        errors.append(
            _issue(
                "registry-binding",
                "registry_sha256",
                "must bind the exact GPUWM physics registry semantic SHA-256",
            )
        )
    elif bound_registry_sha256 != actual_registry_sha256:
        errors.append(
            _issue(
                "stale-registry-binding",
                "registry_sha256",
                "plan registry SHA-256 differs from the active GPUWM registry",
            )
        )

    context = plan.get("context")
    context_valid = True
    if not isinstance(context, dict):
        errors.append(_issue("context-type", "context", "must be a JSON object"))
        context = {}
        context_valid = False
    normalized_context: dict[str, object] = {
        "source_id": None,
        "runner_id": None,
        "topology_id": None,
        "edges": [],
    }
    for key in ("source_id", "runner_id", "topology_id"):
        value = context.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                _issue("context-field", f"context.{key}", "must be a non-empty string")
            )
            context_valid = False
        else:
            normalized_context[key] = value
    if context_valid:
        result["context"] = normalized_context

    routes = selected_registry.get("runner_routes")
    routes = routes if isinstance(routes, dict) else {}
    runner_id = normalized_context.get("runner_id")
    route = routes.get(runner_id) if isinstance(runner_id, str) else None
    if runner_id is not None and not isinstance(route, dict):
        errors.append(
            _issue(
                "unknown-runner",
                "context.runner_id",
                f"runner {runner_id!r} is not registered",
            )
        )
    elif isinstance(route, dict):
        if route.get("implemented") is not True:
            errors.append(
                _issue(
                    "unimplemented-runner",
                    "context.runner_id",
                    f"runner {runner_id!r} is registered but not implemented",
                )
            )
        source_ids = route.get("source_ids", [])
        source_id = normalized_context.get("source_id")
        if (
            isinstance(source_id, str)
            and isinstance(source_ids, list)
            and "*" not in source_ids
            and source_id not in source_ids
        ):
            errors.append(
                _issue(
                    "unsupported-source-route",
                    "context.source_id",
                    f"runner {runner_id!r} does not register source {source_id!r}",
                )
            )
        topology_ids = route.get("topology_ids", [])
        topology_id = normalized_context.get("topology_id")
        if (
            isinstance(topology_id, str)
            and isinstance(topology_ids, list)
            and topology_id not in topology_ids
        ):
            errors.append(
                _issue(
                    "unsupported-topology-route",
                    "context.topology_id",
                    f"runner {runner_id!r} does not register topology {topology_id!r}",
                )
            )

    domains = plan.get("domains")
    if not isinstance(domains, list) or not domains:
        errors.append(
            _issue("domains-type", "domains", "must be a non-empty JSON array")
        )
        domains = []

    domain_ids: list[str] = []
    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            continue
        domain_id = domain.get("domain_id")
        if isinstance(domain_id, str) and domain_id.strip():
            domain_ids.append(domain_id)
    duplicate_ids = sorted(
        domain_id for domain_id in set(domain_ids) if domain_ids.count(domain_id) > 1
    )
    for domain_id in duplicate_ids:
        errors.append(
            _issue("duplicate-domain", "domains", f"duplicate domain_id {domain_id!r}")
        )

    edges = plan.get("edges", [])
    if not isinstance(edges, list):
        errors.append(_issue("edges-type", "edges", "must be a JSON array"))
        edges = []
    normalized_edges: list[dict[str, str]] = []
    edge_pairs: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges):
        path = f"edges[{index}]"
        if not isinstance(edge, dict):
            errors.append(_issue("edge-type", path, "must be a JSON object"))
            continue
        parent = edge.get("parent_domain_id")
        child = edge.get("child_domain_id")
        if (
            not isinstance(parent, str)
            or not parent.strip()
            or not isinstance(child, str)
            or not child.strip()
        ):
            errors.append(
                _issue(
                    "edge-field",
                    path,
                    "parent_domain_id and child_domain_id must be non-empty strings",
                )
            )
            continue
        if parent not in domain_ids or child not in domain_ids:
            errors.append(_issue("edge-domain", path, "edge references an unknown domain"))
            continue
        if parent == child:
            errors.append(_issue("edge-self", path, "domain cannot parent itself"))
            continue
        pair = (parent, child)
        if pair in edge_pairs:
            errors.append(
                _issue("duplicate-edge", path, f"duplicate edge {parent!r}->{child!r}")
            )
            continue
        edge_pairs.add(pair)
        normalized_edges.append(
            {"parent_domain_id": parent, "child_domain_id": child}
        )
    normalized_context["edges"] = normalized_edges

    topology_id = normalized_context.get("topology_id")
    if topology_id == "single-domain-v1" and len(domains) != 1:
        errors.append(
            _issue(
                "topology-domain-count",
                "domains",
                "single-domain-v1 requires exactly one domain",
            )
        )
    if topology_id == "single-domain-v1" and edges:
        errors.append(
            _issue(
                "single-domain-edges",
                "edges",
                "single-domain-v1 requires zero edges",
            )
        )
    if topology_id == "one-way-nested-v1" and len(domains) < 2:
        errors.append(
            _issue(
                "topology-domain-count",
                "domains",
                "one-way-nested-v1 requires at least two domains",
            )
        )
    if topology_id == "one-way-nested-v1" and domains:
        expected_edges = len(domains) - 1
        if len(normalized_edges) != expected_edges:
            errors.append(
                _issue(
                    "tree-edge-count",
                    "edges",
                    f"one-way tree requires {expected_edges} unique edges",
                )
            )
        incoming = {domain_id: 0 for domain_id in domain_ids}
        children: dict[str, list[str]] = {domain_id: [] for domain_id in domain_ids}
        for edge in normalized_edges:
            parent = edge["parent_domain_id"]
            child = edge["child_domain_id"]
            incoming[child] += 1
            children[parent].append(child)
        for domain_id, count in sorted(incoming.items()):
            if count > 1:
                errors.append(
                    _issue(
                        "multiple-parents",
                        "edges",
                        f"domain {domain_id!r} has {count} parents",
                    )
                )
        roots = sorted(domain_id for domain_id, count in incoming.items() if count == 0)
        if len(roots) != 1:
            errors.append(
                _issue(
                    "tree-root-count",
                    "edges",
                    f"one-way tree requires one root, found {len(roots)}",
                )
            )
        visiting: set[str] = set()
        visited: set[str] = set()
        cycle = False

        def visit(domain_id: str) -> None:
            nonlocal cycle
            if domain_id in visiting:
                cycle = True
                return
            if domain_id in visited:
                return
            visiting.add(domain_id)
            for child_id in children.get(domain_id, []):
                visit(child_id)
            visiting.remove(domain_id)
            visited.add(domain_id)

        for domain_id in domain_ids:
            visit(domain_id)
        if cycle:
            errors.append(_issue("tree-cycle", "edges", "one-way tree is cyclic"))
        if len(roots) == 1:
            reachable: set[str] = set()
            pending = [roots[0]]
            while pending:
                domain_id = pending.pop()
                if domain_id in reachable:
                    continue
                reachable.add(domain_id)
                pending.extend(children.get(domain_id, []))
            missing = sorted(set(domain_ids) - reachable)
            if missing:
                errors.append(
                    _issue(
                        "tree-disconnected",
                        "edges",
                        f"domains are disconnected from root {roots[0]!r}: {missing}",
                    )
                )

    # Depth below the tree root, used to index a template's per-domain
    # override columns.  A single-domain plan is depth 0.  Cyclic graphs are
    # already reported above; the visited guard keeps this terminating.
    parent_of = {
        edge["child_domain_id"]: edge["parent_domain_id"]
        for edge in normalized_edges
    }
    domain_depths: dict[str, int] = {}
    for domain_id in domain_ids:
        depth = 0
        cursor = domain_id
        walked = {cursor}
        while cursor in parent_of:
            cursor = parent_of[cursor]
            if cursor in walked:
                depth = 0
                break
            walked.add(cursor)
            depth += 1
        domain_depths[domain_id] = depth

    components = selected_registry.get("components")
    components = components if isinstance(components, dict) else {}
    templates = selected_registry.get("templates")
    templates = templates if isinstance(templates, dict) else {}
    parameter_specs = selected_registry.get("parameters")
    parameter_specs = parameter_specs if isinstance(parameter_specs, dict) else {}
    warning_policy = selected_registry.get("warning_policy", {})
    warning_policy = warning_policy if isinstance(warning_policy, dict) else {}
    # Both tiers come off the ladder, not out of warning_policy: the policy
    # block is the document's readable echo of them and the loader refuses a
    # registry where the two disagree, so reading the ladder here leaves
    # exactly one ordering of maturity names in this module.
    warn_maturities = _ladder_tier(selected_registry, "warn")
    nonwarning_maturities = _ladder_tier(selected_registry, "nonwarning")
    warn_unknown_maturity = (
        warning_policy.get("unknown_implemented_maturity") == "warn"
    )

    selector_owners: dict[str, str] = {}
    for component_id, component in components.items():
        if not isinstance(component, dict):
            continue
        keys = component.get("selector_keys", [])
        if not isinstance(keys, list):
            continue
        for key in keys:
            if isinstance(key, str):
                selector_owners[key] = component_id

    route_mode = route.get("mode") if isinstance(route, dict) else None
    route_component_overrides = set(
        route.get("allowed_component_overrides", [])
        if isinstance(route, dict)
        else []
    )
    route_component_options: dict[str, set[str]] = {}
    raw_component_options = (
        route.get("allowed_component_options", {})
        if isinstance(route, dict)
        else {}
    )
    if isinstance(raw_component_options, dict):
        route_component_options = {
            component_id: {
                option_id for option_id in option_ids
                if isinstance(option_id, str)
            }
            for component_id, option_ids in raw_component_options.items()
            if isinstance(component_id, str) and isinstance(option_ids, list)
        }
    route_parameter_keys = set(
        route.get("allowed_parameter_keys", []) if isinstance(route, dict) else []
    )
    route_expert_settings = set(
        route.get("allowed_expert_setting_keys", [])
        if isinstance(route, dict)
        else []
    )
    route_expert_selectors = set(
        route.get("allowed_expert_selector_keys", [])
        if isinstance(route, dict)
        else []
    )
    # A route's template lists are the reachability declaration the GUI reads.
    # ``route_declares_templates`` records that the route publishes any list at
    # all: a route that does must be exhaustive, so a source it registers with
    # NO list reaches no template rather than every template.  It used to reach
    # every template -- ``source_template_ids`` named only ``hrrr`` on the tree
    # route, so an era5/gfs/20crv3 tree plan could name any registered
    # template, which made "is this option reachable" unanswerable from the
    # declaration and made an expert-only template reachable without its
    # acknowledgement through a source that simply had no list.
    route_source_templates: set[str] = set()
    route_expert_templates: set[str] = set()
    route_declares_templates = False
    route_expert_acknowledgement = None
    if isinstance(route, dict):
        source_templates = route.get("source_template_ids", {})
        expert_templates = route.get("expert_template_ids", {})
        source_id = normalized_context.get("source_id")
        if isinstance(source_templates, dict):
            route_declares_templates = bool(source_templates)
            if isinstance(source_id, str):
                declared = source_templates.get(source_id, [])
                if isinstance(declared, list):
                    route_source_templates = {
                        value for value in declared if isinstance(value, str)
                    }
        if isinstance(expert_templates, dict):
            route_declares_templates = (
                route_declares_templates or bool(expert_templates))
            if isinstance(source_id, str):
                declared = expert_templates.get(source_id, [])
                if isinstance(declared, list):
                    route_expert_templates = {
                        value for value in declared if isinstance(value, str)
                    }
        acknowledgement = route.get("expert_acknowledgement_id")
        if isinstance(acknowledgement, str) and acknowledgement.strip():
            route_expert_acknowledgement = acknowledgement

    # Consent is plan-level and names the route's own acknowledgement id, so a
    # launcher cannot satisfy it by accident: an unrecognized id is an error
    # rather than a no-op, and consent without an expert template is reported
    # too (it means the plan and the route disagree about what is being run).
    acknowledged: set[str] = set()
    plan_acknowledgements = plan.get("acknowledgements", [])
    if not isinstance(plan_acknowledgements, list):
        errors.append(
            _issue(
                "expert-acknowledgements-type",
                "acknowledgements",
                "must be a JSON array of route acknowledgement ids",
            )
        )
        plan_acknowledgements = []
    for index, value in enumerate(plan_acknowledgements):
        if not isinstance(value, str) or not value.strip():
            errors.append(
                _issue(
                    "expert-acknowledgement-id",
                    f"acknowledgements[{index}]",
                    "must be a non-empty string",
                )
            )
            continue
        if value != route_expert_acknowledgement:
            errors.append(
                _issue(
                    "unknown-expert-acknowledgement",
                    f"acknowledgements[{index}]",
                    f"runner {runner_id!r} does not publish acknowledgement "
                    f"{value!r}",
                )
            )
            continue
        acknowledged.add(value)
    result["acknowledgements"] = sorted(acknowledged)
    result["acknowledgement_provenance"] = {
        value: ["physics-plan.acknowledgements"]
        for value in sorted(acknowledged)
    }

    asset_domains: dict[str, tuple[dict[str, object], list[str]]] = {}
    domain_template_ids: list[str] = []
    resolved_domains: list[dict[str, object]] = result[  # type: ignore[assignment]
        "resolved_domains"
    ]
    for index, domain in enumerate(domains):
        base_path = f"domains[{index}]"
        if not isinstance(domain, dict):
            errors.append(_issue("domain-type", base_path, "must be a JSON object"))
            continue
        domain_id = domain.get("domain_id")
        if not isinstance(domain_id, str) or not domain_id.strip():
            errors.append(
                _issue("domain-id", f"{base_path}.domain_id", "must be a non-empty string")
            )
            domain_id = f"invalid-domain-{index}"

        template_id = domain.get("template_id")
        template: Mapping[str, Any] = {}
        if template_id is not None:
            if not isinstance(template_id, str):
                errors.append(
                    _issue(
                        "template-id",
                        f"{base_path}.template_id",
                        "must be a string",
                    )
                )
            elif not isinstance(templates.get(template_id), dict):
                errors.append(
                    _issue(
                        "unknown-template",
                        f"{base_path}.template_id",
                        f"template {template_id!r} is not registered",
                    )
                )
            else:
                template = templates[template_id]
                domain_template_ids.append(template_id)
                if route_declares_templates and template_id not in (
                        route_source_templates | route_expert_templates):
                    errors.append(
                        _issue(
                            "template-route",
                            f"{base_path}.template_id",
                            f"template {template_id!r} is not registered for this source/runner route",
                        )
                    )
                elif template_id in route_expert_templates:
                    if route_expert_acknowledgement is None:
                        errors.append(
                            _issue(
                                "expert-route-policy",
                                "registry.runner_routes",
                                f"runner {runner_id!r} offers expert templates "
                                "without publishing expert_acknowledgement_id",
                            )
                        )
                    elif route_expert_acknowledgement not in acknowledged:
                        errors.append(
                            _issue(
                                "expert-acknowledgement-required",
                                f"{base_path}.template_id",
                                f"template {template_id!r} is an expert-only "
                                f"template; the plan must carry "
                                f"acknowledgements "
                                f"[{route_expert_acknowledgement!r}]",
                            )
                        )
                    for message in route.get("expert_warnings", []) or []:
                        if isinstance(message, str):
                            warnings.append(
                                _issue(
                                    "expert-template-warning",
                                    f"{base_path}.template_id",
                                    message,
                                )
                            )
                template_maturity = template.get("maturity")
                if template_maturity in warn_maturities or (
                    warn_unknown_maturity
                    and template_maturity not in nonwarning_maturities
                ):
                    warnings.append(
                        _issue(
                            "template-maturity",
                            f"{base_path}.template_id",
                            f"template {template_id!r} maturity is {template_maturity}; warning does not block launch",
                        )
                    )
                template_warnings = template.get("warnings", [])
                if isinstance(template_warnings, list):
                    for message in template_warnings:
                        if isinstance(message, str):
                            warnings.append(
                                _issue(
                                    "template-warning",
                                    f"{base_path}.template_id",
                                    message,
                                )
                            )

        resolved_components: dict[str, str] = {}
        template_components = template.get("components", {})
        if isinstance(template_components, dict):
            resolved_components.update(template_components)
        requested_components = domain.get("components", {})
        if not isinstance(requested_components, dict):
            errors.append(
                _issue(
                    "components-type",
                    f"{base_path}.components",
                    "must be a JSON object",
                )
            )
            requested_components = {}
        if (
            isinstance(route, dict)
            and route.get("require_explicit_components") is True
            and not requested_components
        ):
            errors.append(
                _issue(
                    "explicit-components-required",
                    f"{base_path}.components",
                    "runner route requires a non-empty explicit component map",
                )
            )
        for component_id, option_id in requested_components.items():
            if component_id not in components:
                errors.append(
                    _issue(
                        "unknown-component",
                        f"{base_path}.components.{component_id}",
                        f"component {component_id!r} is not registered",
                    )
                )
                continue
            if not isinstance(option_id, str):
                errors.append(
                    _issue(
                        "option-id",
                        f"{base_path}.components.{component_id}",
                        "option id must be a string",
                    )
                )
                continue
            resolved_components[component_id] = option_id

        expert = domain.get("expert_overrides", {})
        if not isinstance(expert, dict):
            errors.append(
                _issue(
                    "expert-overrides-type",
                    f"{base_path}.expert_overrides",
                    "must be a JSON object",
                )
            )
            expert = {}
        unknown_expert_sections = sorted(set(expert) - {"selectors", "settings"})
        for section in unknown_expert_sections:
            errors.append(
                _issue(
                    "expert-overrides-section",
                    f"{base_path}.expert_overrides.{section}",
                    "only selectors and settings are recognized",
                )
            )
        expert_selectors = expert.get("selectors", {})
        expert_settings = expert.get("settings", {})
        if not isinstance(expert_selectors, dict):
            errors.append(
                _issue(
                    "expert-selectors-type",
                    f"{base_path}.expert_overrides.selectors",
                    "must be a JSON object",
                )
            )
            expert_selectors = {}
        if not isinstance(expert_settings, dict):
            errors.append(
                _issue(
                    "expert-settings-type",
                    f"{base_path}.expert_overrides.settings",
                    "must be a JSON object",
                )
            )
            expert_settings = {}
        for selector in sorted(expert_selectors):
            if selector not in selector_owners:
                errors.append(
                    _issue(
                        "unknown-expert-selector",
                        f"{base_path}.expert_overrides.selectors.{selector}",
                        "selector is not registered and cannot imply implementation",
                    )
                )
        for selector in sorted(set(expert_settings) & set(selector_owners)):
            errors.append(
                _issue(
                    "selector-in-settings",
                    f"{base_path}.expert_overrides.settings.{selector}",
                    "registered selectors must be placed in expert_overrides.selectors",
                )
            )

        for name in sorted(expert_selectors):
            if name not in route_expert_selectors:
                errors.append(
                    _issue(
                        "expert-selector-route",
                        f"{base_path}.expert_overrides.selectors.{name}",
                        "runner route does not accept this expert selector",
                    )
                )
        for name in sorted(expert_settings):
            if name not in route_expert_settings:
                errors.append(
                    _issue(
                        "expert-setting-route",
                        f"{base_path}.expert_overrides.settings.{name}",
                        "runner route does not accept this expert setting",
                    )
                )

        if route_mode == "fixed-template":
            if requested_components:
                errors.append(
                    _issue(
                        "fixed-template-components",
                        f"{base_path}.components",
                        "fixed-template runner accepts only its immutable template_id; component overrides must be empty",
                    )
                )
        elif route_mode == "experiment-per-domain" and isinstance(
            template_components, dict
        ):
            for component_id, option_id in requested_components.items():
                if (
                    template_components.get(component_id) != option_id
                    and component_id not in route_component_overrides
                    and option_id not in route_component_options.get(
                        component_id, set())
                ):
                    errors.append(
                        _issue(
                            "component-override-route",
                            f"{base_path}.components.{component_id}",
                            "runner route does not allow this component option "
                            "to vary per domain",
                        )
                    )

        for component_id in sorted(components):
            component = components[component_id]
            if not isinstance(component, dict):
                continue
            option_id = resolved_components.get(component_id)
            if not isinstance(option_id, str):
                errors.append(
                    _issue(
                        "missing-component",
                        f"{base_path}.components.{component_id}",
                        "select an option directly or through a template",
                    )
                )
                continue
            options = component.get("options", {})
            if not isinstance(options, dict) or not isinstance(options.get(option_id), dict):
                errors.append(
                    _issue(
                        "unknown-option",
                        f"{base_path}.components.{component_id}",
                        f"option {option_id!r} is not registered for {component_id}",
                    )
                )
                continue
            selected_option = options[option_id]
            keys = component.get("selector_keys", [])
            if isinstance(keys, list) and selected_option.get("implemented") is True:
                projection = dict(selected_option.get("selectors", {}))
                touched = False
                for key in keys:
                    if key in expert_selectors:
                        projection[key] = expert_selectors[key]
                        touched = True
                if touched:
                    matches = [
                        (candidate_id, candidate)
                        for candidate_id, candidate in options.items()
                        if isinstance(candidate, dict)
                        and all(
                            key in candidate.get("selectors", {})
                            and _same_value(candidate["selectors"][key], projection.get(key))
                            for key in keys
                        )
                    ]
                    if len(matches) != 1:
                        errors.append(
                            _issue(
                                "unknown-selector-combination",
                                f"{base_path}.expert_overrides.selectors",
                                f"selectors for {component_id} do not identify one registered option",
                            )
                        )
                    else:
                        candidate_id, candidate = matches[0]
                        if candidate.get("implemented") is not True:
                            errors.append(
                                _issue(
                                    "unimplemented-selector",
                                    f"{base_path}.expert_overrides.selectors",
                                    f"selectors resolve to unimplemented {component_id} option {candidate_id!r}",
                                )
                            )
                        else:
                            if candidate_id != option_id:
                                warnings.append(
                                    _issue(
                                        "expert-selector-resolution",
                                        f"{base_path}.expert_overrides.selectors",
                                        f"expert selectors replace {component_id} option {option_id!r} with {candidate_id!r}",
                                    )
                                )
                            resolved_components[component_id] = candidate_id

        settings: dict[str, object] = {
            name: spec["default"]
            for name, spec in parameter_specs.items()
            if isinstance(spec, dict)
            and "default" in spec
            and parameter_is_implemented(spec)
        }
        template_parameters = template.get("parameters", {})
        if isinstance(template_parameters, dict):
            settings.update(template_parameters)
        # Per-domain template values, indexed by depth below the tree root.
        # Several WRF knobs are max_domains in the Registry and were varied
        # down the nest chain by the verified runs (sixth-order diffusion
        # factor, radiation interval, acoustic off-centering).  A template
        # records those columns verbatim rather than deriving them from grid
        # spacing, because they came from a namelist, not from a scaling law.
        overrides = template.get("per_domain_overrides")
        if isinstance(overrides, list):
            depth = domain_depths.get(domain_id)
            if depth is not None and depth < len(overrides):
                column = overrides[depth]
                if isinstance(column, dict):
                    for name, value in column.items():
                        if name == "nominal_dx_m":
                            continue
                        spec = parameter_specs.get(name)
                        message = (
                            _parameter_error(spec, value)
                            if isinstance(spec, dict)
                            else "parameter is not registered"
                        )
                        if message is not None:
                            errors.append(
                                _issue(
                                    "template-per-domain-override",
                                    f"registry.templates.{template_id}"
                                    f".per_domain_overrides[{depth}].{name}",
                                    message,
                                )
                            )
                            continue
                        settings[name] = value
        domain_parameters = domain.get("parameters", {})
        if not isinstance(domain_parameters, dict):
            errors.append(
                _issue(
                    "parameters-type",
                    f"{base_path}.parameters",
                    "must be a JSON object",
                )
            )
            domain_parameters = {}
        for name in sorted(domain_parameters):
            if name not in route_parameter_keys:
                errors.append(
                    _issue(
                        "parameter-route",
                        f"{base_path}.parameters.{name}",
                        "runner route does not accept this per-domain setting",
                    )
                )

        for component_id in sorted(components):
            option_id = resolved_components.get(component_id)
            component = components.get(component_id)
            options = component.get("options", {}) if isinstance(component, dict) else {}
            option = options.get(option_id) if isinstance(options, dict) else None
            if not isinstance(option, dict):
                continue
            if option.get("implemented") is not True:
                errors.append(
                    _issue(
                        "unimplemented-option",
                        f"{base_path}.components.{component_id}",
                        f"option {option_id!r} is registered but not implemented",
                    )
                )
                continue
            maturity = option.get("maturity")
            if maturity in warn_maturities or (
                warn_unknown_maturity and maturity not in nonwarning_maturities
            ):
                warnings.append(
                    _issue(
                        "maturity",
                        f"{base_path}.components.{component_id}",
                        f"{option_id} maturity is {maturity}; warning does not block launch",
                    )
                )
            option_warnings = option.get("warnings", [])
            if isinstance(option_warnings, list):
                for message in option_warnings:
                    if isinstance(message, str):
                        warnings.append(
                            _issue(
                                "component-warning",
                                f"{base_path}.components.{component_id}",
                                message,
                            )
                        )
            selectors = option.get("selectors", {})
            parameters = option.get("parameters", {})
            if isinstance(selectors, dict):
                settings.update(selectors)
            if isinstance(parameters, dict):
                settings.update(parameters)
            requirements = option.get("asset_requirements", [])
            if isinstance(requirements, list):
                for requirement in requirements:
                    if not isinstance(requirement, dict):
                        continue
                    key = canonical_json(requirement)
                    if key not in asset_domains:
                        asset_domains[key] = (deepcopy(requirement), [])
                    if domain_id not in asset_domains[key][1]:
                        asset_domains[key][1].append(domain_id)

        for name, value in domain_parameters.items():
            spec = parameter_specs.get(name)
            if not isinstance(spec, dict):
                errors.append(
                    _issue(
                        "unknown-parameter",
                        f"{base_path}.parameters.{name}",
                        "parameter is not registered; use expert_overrides.settings for passthrough",
                    )
                )
                continue
            message = _parameter_error(spec, value)
            if message is not None:
                errors.append(
                    _issue("parameter-value", f"{base_path}.parameters.{name}", message)
                )
                continue
            settings[name] = value

        for name, value in expert_settings.items():
            if name in selector_owners:
                continue
            if not _is_json_scalar(value):
                errors.append(
                    _issue(
                        "expert-setting-value",
                        f"{base_path}.expert_overrides.settings.{name}",
                        "expert setting must be a finite JSON scalar",
                    )
                )
                continue
            spec = parameter_specs.get(name)
            if isinstance(spec, dict):
                message = _parameter_error(spec, value)
                if message is not None:
                    errors.append(
                        _issue(
                            "expert-setting-value",
                            f"{base_path}.expert_overrides.settings.{name}",
                            message,
                        )
                    )
                    continue
            else:
                warnings.append(
                    _issue(
                        "untyped-expert-setting",
                        f"{base_path}.expert_overrides.settings.{name}",
                        "passed through without claiming registry implementation or validation",
                    )
                )
            settings[name] = value
        for name, value in expert_selectors.items():
            if name in selector_owners and _is_json_scalar(value):
                settings[name] = value

        for component_id, option_id in sorted(resolved_components.items()):
            component = components.get(component_id)
            options = component.get("options", {}) if isinstance(component, dict) else {}
            option = options.get(option_id) if isinstance(options, dict) else None
            constraints = option.get("constraints", {}) if isinstance(option, dict) else {}
            if not isinstance(constraints, dict):
                continue
            required_settings = constraints.get("required_settings", {})
            if isinstance(required_settings, dict):
                for name, expected in required_settings.items():
                    if name not in settings or not _same_value(settings[name], expected):
                        errors.append(
                            _issue(
                                "component-required-setting",
                                f"{base_path}.components.{component_id}",
                                f"option {option_id!r} requires {name}={expected!r}",
                            )
                        )
            forbidden = constraints.get("forbidden_setting_values", {})
            if isinstance(forbidden, dict):
                for name, values in forbidden.items():
                    if not isinstance(values, list) or name not in settings:
                        continue
                    if any(_same_value(settings[name], value) for value in values):
                        errors.append(
                            _issue(
                                "component-forbidden-setting",
                                f"{base_path}.components.{component_id}",
                                f"option {option_id!r} forbids {name}={settings[name]!r}",
                            )
                        )
            required_components = constraints.get("requires_components", {})
            if isinstance(required_components, dict):
                for required_id, allowed_options in required_components.items():
                    if (
                        not isinstance(allowed_options, list)
                        or resolved_components.get(required_id) not in allowed_options
                    ):
                        errors.append(
                            _issue(
                                "component-dependency",
                                f"{base_path}.components.{component_id}",
                                f"option {option_id!r} requires {required_id} in {allowed_options!r}",
                            )
                        )
            for rule in _conditional_refusals(constraints):
                if _conditional_refusal_fires(
                    rule, resolved_components, settings, parameter_specs
                ):
                    errors.append(
                        _issue(
                            "component-conditional-refusal",
                            f"{base_path}.components.{component_id}",
                            f"option {option_id!r} is refused here: "
                            f"{rule['reason']}",
                        )
                    )

        if (
            isinstance(route, dict)
            and route.get("requires_moist_real_initialization") is True
            and resolved_components.get("microphysics") == "off"
            and domain_parameters.get("moist") is not True
        ):
            errors.append(
                _issue(
                    "real-source-mp-off-requires-explicit-moist",
                    f"{base_path}.parameters.moist",
                    "microphysics off on this real-source route requires "
                    "an explicit per-domain parameter moist=true. That "
                    "setting allocates qv/qc/qr carrier fields while "
                    "microphysics remains off; it does not synthesize "
                    "analyzed clouds, so source-absent cloud mass stays "
                    "exact zero. Native HRRR is still refused at "
                    "preparation because MP off cannot faithfully retain "
                    "its analyzed QC/QR/QI/QS/QG and no radiation-only "
                    "analyzed-cloud carrier is implemented.",
                )
            )

        resolved_domains.append(
            {
                "domain_id": domain_id,
                "template_id": template_id,
                "components": dict(sorted(resolved_components.items())),
                "settings": dict(sorted(settings.items())),
            }
        )

    if (
        isinstance(route, dict)
        and route.get("template_policy") == "uniform-base-template"
        and len(set(domain_template_ids)) > 1
    ):
        errors.append(
            _issue(
                "nonuniform-base-template",
                "domains",
                "runner route requires one uniform base template across domains",
            )
        )

    graph_constraints = (
        route.get("graph_setting_constraints", [])
        if isinstance(route, dict)
        else []
    )
    if isinstance(graph_constraints, list):
        child_ids = {edge["child_domain_id"] for edge in normalized_edges}
        root_ids = set(domain_ids) - child_ids
        for constraint in graph_constraints:
            if not isinstance(constraint, dict):
                continue
            setting_key = constraint.get("setting_key")
            scope = constraint.get("scope")
            if not isinstance(setting_key, str) or scope not in {
                "root",
                "non-root",
                "all",
            }:
                errors.append(
                    _issue(
                        "graph-setting-policy",
                        "registry.runner_routes",
                        "malformed graph setting constraint",
                    )
                )
                continue
            expected = constraint.get("required_value")
            for domain_index, resolved in enumerate(resolved_domains):
                domain_id = resolved["domain_id"]
                selected = (
                    scope == "all"
                    or (scope == "root" and domain_id in root_ids)
                    or (scope == "non-root" and domain_id in child_ids)
                )
                if not selected:
                    continue
                observed = resolved["settings"].get(setting_key)
                if not _same_value(observed, expected):
                    errors.append(
                        _issue(
                            "graph-setting-constraint",
                            f"domains[{domain_index}].parameters.{setting_key}",
                            f"{scope} domain requires {setting_key}={expected!r}, got {observed!r}",
                        )
                    )

    transition_policy_id = (
        route.get("transition_policy_id") if isinstance(route, dict) else None
    )
    if isinstance(transition_policy_id, str):
        transitions = selected_registry.get("transitions", {})
        transition_policy = (
            transitions.get(transition_policy_id)
            if isinstance(transitions, dict)
            else None
        )
        if not isinstance(transition_policy, dict):
            errors.append(
                _issue(
                    "transition-policy",
                    "registry.transitions",
                    f"runner route names missing transition policy {transition_policy_id!r}",
                )
            )
        else:
            component_id = transition_policy.get("component_id")
            resolved_by_id = {
                domain["domain_id"]: domain
                for domain in resolved_domains
                if isinstance(domain.get("domain_id"), str)
            }
            for edge_index, edge in enumerate(normalized_edges):
                parent = resolved_by_id.get(edge["parent_domain_id"])
                child = resolved_by_id.get(edge["child_domain_id"])
                if (
                    not isinstance(component_id, str)
                    or parent is None
                    or child is None
                ):
                    continue
                parent_option = parent["components"].get(component_id)
                child_option = child["components"].get(component_id)
                rule: object = None
                if parent_option == child_option:
                    same_option = transition_policy.get("same_option")
                    if isinstance(same_option, dict) and same_option.get("allowed") is True:
                        rule = same_option
                else:
                    cross_options = transition_policy.get("cross_options", [])
                    if isinstance(cross_options, list):
                        matches = [
                            candidate
                            for candidate in cross_options
                            if isinstance(candidate, dict)
                            and candidate.get("parent_option_id") == parent_option
                            and candidate.get("child_option_id") == child_option
                        ]
                        if len(matches) == 1:
                            rule = matches[0]
                edge_path = f"edges[{edge_index}]"
                if not isinstance(rule, dict):
                    errors.append(
                        _issue(
                            "unsupported-component-transition",
                            edge_path,
                            f"transition policy {transition_policy_id!r} does not admit {component_id} {parent_option!r}->{child_option!r}",
                        )
                    )
                    continue
                maturity = rule.get("maturity")
                warning_maturities = _ladder_tier(selected_registry, "warn")
                if maturity in warning_maturities:
                    warnings.append(
                        _issue(
                            "transition-maturity",
                            edge_path,
                            f"{component_id} transition "
                            f"{parent_option!r}->{child_option!r} is "
                            f"{maturity!r}; review the per-species receipt",
                        )
                    )
                for role, resolved, key in (
                    ("parent", parent, "required_parent_settings"),
                    ("child", child, "required_child_settings"),
                ):
                    required = rule.get(key, {})
                    if not isinstance(required, dict):
                        continue
                    for name, expected in required.items():
                        observed = resolved["settings"].get(name)
                        if not _same_value(observed, expected):
                            errors.append(
                                _issue(
                                    "transition-required-setting",
                                    edge_path,
                                    f"{role} domain requires {name}={expected!r}, got {observed!r}",
                                )
                            )

    result["asset_requirements"] = [
        {"requirement": requirement, "domain_ids": sorted(domain_list)}
        for requirement, domain_list in (
            asset_domains[key] for key in sorted(asset_domains)
        )
    ]
    result["launchable"] = not errors
    return result


def load_physics_plan(path: str | Path) -> object:
    """Load strict JSON for CLI validation without accepting NaN/Infinity."""

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )


__all__ = [
    "DEFAULT_TEMPLATE_ID",
    "MORRISON_TEMPLATE_ID",
    "NSSL2_LEGACY_RRTMG_TEMPLATE_ID",
    "NSSL2_TEMPLATE_ID",
    "PLAN_SCHEMA",
    "REGISTRY_SCHEMA",
    "THOMPSON_KF_TEMPLATE_ID",
    "THOMPSON_TEMPLATE_ID",
    "VALIDATION_SCHEMA",
    "WSM6_TEMPLATE_ID",
    "canonical_json",
    "canonical_sha256",
    "load_physics_plan",
    "physics_registry",
    "registry_sha256",
    "validate_physics_plan",
]
