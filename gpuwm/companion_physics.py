"""Read-only repairs for a physics draft, admitted by the installed engine.

Evidence labels never control eligibility. This enumerates registered scheme
replacements and coupled microphysics/radiation and boundary-layer choices;
the ordinary configuration parser owns all combination checks.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import tomllib

SCHEMA = "arwen.companion-physics-repairs.v1"


def draft_digest(action):
    return hashlib.sha256(json.dumps(action, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _summary(error):
    if "mp_physics=9" in error and "cloud-optics" in error:
        return ("Milbrandt-Yau is implemented, but its cloud particles are not yet "
                "coupled to RTE+RRTMGP radiation. Keep Milbrandt-Yau and choose "
                "WRF RRTMG, or select another compatible scheme below.")
    return "These physics settings cannot run together. Choose a compatible replacement below."


def repairs(request):
    from gpuwm.config_authority import read_config_authority
    from gpuwm.companion_domains import _apply, _build, _exact_keys, physics_components, REQUEST_SCHEMA
    from gpuwm.case_catalog import _native_contract

    required = {"schema", "config_path", "expected_sha256", "action"}
    _exact_keys(request, required, required, where="physics repair request")
    if request["schema"] != REQUEST_SCHEMA or request["action"].get("kind") != "set_physics":
        raise ValueError("Physics repairs require a set_physics draft")
    authority = read_config_authority(request["config_path"])
    if authority.sha256 != request["expected_sha256"]:
        raise ValueError("The configuration changed; refresh before checking repairs")
    raw = tomllib.loads(authority.payload.decode("utf-8-sig"))
    original = _build(raw, authority.source)
    action = copy.deepcopy(request["action"])
    grid_id = action["grid_id"]
    # Validate the request's shape and recognized settings even when its
    # combination will be refused. Never reinterpret unknown keys as repairs.
    proposed = copy.deepcopy(raw)
    _apply(proposed, action, authority.source)
    result = {"schema": SCHEMA, "source_path": str(authority.source),
              "source_sha256": authority.sha256, "draft_sha256": draft_digest(action),
              "action": action, "created": False, "forecast_started": False,
              "options": [], "rejected": []}
    try:
        _build(proposed, authority.source)
    except (ValueError, NotImplementedError) as error:
        result.update(valid=False, error=str(error), summary=_summary(str(error)))
    else:
        result.update(valid=True, summary="These settings pass the configuration checks.")
        return result

    components = {c["id"]: c for c in physics_components()}
    _, _, domain_keys = _native_contract()
    runs = [d.run for d in original.domains if grid_id == 0 or d.grid_id == grid_id]
    attempted, seen = set(), set()

    def evaluate(options, replacement=None, label=None):
        settings = dict(action["settings"])
        evidence = []
        for component, option in options:
            settings.update(option["settings"])
            evidence.append({"component": component, "option": option["id"],
                             "maturity": option["maturity"], "warnings": option["warnings"]})
        settings.update(replacement or {})
        if grid_id:
            # Shared values already in effect need no edit. A different shared
            # value is genuinely outside this domain's scope, so parser refusal
            # explains that choice rather than silently widening the edit.
            settings = {k: v for k, v in settings.items() if k in domain_keys
                        or not all(getattr(run, k, None) == v for run in runs)}
        candidate_action = {"kind": "set_physics", "grid_id": grid_id, "settings": settings}
        signature = draft_digest(candidate_action)
        if signature in attempted:
            return
        attempted.add(signature)
        title = label or " + ".join(option["label"] for _, option in options)
        candidate = copy.deepcopy(raw)
        try:
            _apply(candidate, candidate_action, authority.source)
            exp = _build(candidate, authority.source)
        except (ValueError, NotImplementedError) as error:
            result["rejected"].append({"label": title, "reason": str(error)})
            return
        # Deduplicate identical effective configurations, including legacy
        # aggregate aliases. Their scientific state, not label, is authority.
        effective = [(d.grid_id, vars(d.run)) for d in exp.domains]
        identity = json.dumps(effective, sort_keys=True, default=str)
        if identity in seen:
            return
        seen.add(identity)
        changes = [{"field": key, "before": action["settings"].get(key,
                       [getattr(run, key, None) for run in runs]), "after": value}
                   for key, value in settings.items()
                   if any(action["settings"].get(key, getattr(run, key, None)) != value for run in runs)]
        effects = []
        if settings.get("ra_lw_physics") == 0:
            effects.append("Longwave radiation will be disabled.")
        if settings.get("ra_sw_physics") == 0:
            effects.append("Shortwave radiation will be disabled.")
        if settings.get("ra_lw_physics") == 90 or settings.get("ra_sw_physics") == 90:
            effects.append("Uses analytic clear-sky radiation instead of cloud-aware radiation.")
        if "mp_physics" in settings and any(action["settings"].get("mp_physics", run.mp_physics) != settings["mp_physics"] for run in runs):
            effects.append("Changes the forecast's cloud and precipitation scheme.")
        result["options"].append({"id": signature, "label": title, "action": candidate_action,
            "changes": changes, "effects": effects, "evidence": evidence,
            "validation": {"configuration_parser": "passed", "forecast_run": "not_run"}})

    # Preserve both radiation spectra and every other draft setting first.
    if "mp_physics=9" in result["error"] and "cloud-optics" in result["error"]:
        evaluate([], {"ra_rrtmg_variant": "rrtmg_legacy"},
                 "Keep Milbrandt-Yau; use WRF RRTMG radiation")
    for component, spec in components.items():
        for option in spec["options"]:
            evaluate([(component, option)])
    for group in (("microphysics", "radiation"), ("pbl", "surface_layer"),
                  ("pbl", "turbulence"), ("pbl", "surface_layer", "turbulence")):
        for selected in itertools.product(*(components[c]["options"] for c in group)):
            evaluate(list(zip(group, selected)))
    result["options"].sort(key=lambda option: len(option["changes"]))
    if hashlib.sha256(authority.source.read_bytes()).hexdigest() != authority.sha256:
        raise ValueError("The configuration changed while checking repairs; refresh and try again")
    return result
