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
AVAILABILITY_SCHEMA = "arwen.companion-physics-availability.v1"

#: The ways an installed option can be closed to the current draft.
#: Each names a door that already refuses; the sentence a caller shows is
#: that door's own words, never a restatement written here.
#:
#: A kind is a CLAIM, so it is only made where it can be MEASURED, and
#: two different claims are made here.
#:
#: :data:`SHARED_SCOPE` claims the SELECTED SCOPE is what refuses, so it
#: is measured by asking the per-domain application -- the door a save
#: passes through -- to set exactly the run-wide values the option needs
#: on the selected domain.  Measuring it instead by whether All domains
#: is admitted classified three land-surface schemes, whose own sentence
#: reads "These settings apply to all domains ... Select All domains to
#: change them", as ordinary combination refusals, left them selectable,
#: and let a save fail on them.
#:
#: :data:`NOT_IMPLEMENTED` and :data:`SOURCE_ROUTE` claim the refusal
#: belongs to the OPTION rather than to the combination it was asked
#: about, so they are kept only where no installed choice for another
#: component admits the option on this setup at all.  A refusal one such
#: choice clears is :data:`COMBINATION`: the option is reachable, and
#: greying it walls off the repair search that exists to reach it.
#:
#: :data:`PRECONDITION` claims something about the MACHINE rather than
#: about the draft: a dataset or table set this install does not have.
#: It is measured by the same inventory the run door raises from
#: (``gpuwm.config.run_preparation_preconditions``), so a greyed cell and
#: the refusal a run meets cannot say different things, and it fires at
#: :data:`AT_RUN_PREPARATION` because the configuration parser -- the
#: door a save passes through -- deliberately does not ask it.
SHARED_SCOPE = "shared-scope"
SOURCE_ROUTE = "source-route"
NOT_IMPLEMENTED = "not-implemented"
COMBINATION = "combination"
CHECK_FAILED = "check-failed"
PRECONDITION = "precondition"

#: Every kind this door can emit, for callers that want to fail loudly on
#: a kind they do not know rather than invent a label for it.
REASON_KINDS = (SHARED_SCOPE, SOURCE_ROUTE, NOT_IMPLEMENTED, COMBINATION,
                CHECK_FAILED, PRECONDITION)

#: When the door that produced a reason fires.  The configuration parser
#: is the door a save passes through, so its verdict IS the save's
#: verdict.  The emission route's gate fires LATER -- at namelist
#: emission, while a run is being prepared -- and is asked here, at plan
#: review, because a refusal that arrives after the plan was accepted is
#: the breakage the gate law names.  A front end that prints both as
#: "ArWen will refuse these settings" says something untrue about the
#: second, so each reason reports which door it came from instead of
#: leaving a reader to guess.
AT_SAVE = "save"
AT_RUN_PREPARATION = "run-preparation"

#: The one remedy this door can measure: the same option, applied to
#: every domain, is admitted.  Claimed only where that was tried.
ALL_DOMAINS = "all-domains"


def draft_digest(action):
    return hashlib.sha256(json.dumps(action, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


#: What a draft is told when no registry rule owns its refusal.
GENERIC_SUMMARY = ("These physics settings cannot run together. Choose a "
                   "compatible replacement below.")


def _fired_rules(runs, settings):
    """The registry's own ``refused_when`` rules for this draft, and the misses.

    ONE TABLE, NOT A SUBSTRING.  This used to be two branches matching
    "mp_physics=9" and "cloud-optics" in the parser's error text: one
    chose a hand-written summary for that scheme, the other offered the
    minimal repair.  Both were per-scheme code paths in generic code,
    coupled by literal substring to prose in gpuwm/config.py -- a
    reworded refusal loses the repair, and the next scheme refused the
    same way never had one.  The rules carry their reason and their
    remedy as data now, so every scheme is answered by the same loop.

    A rule fires per DOMAIN, so the draft is measured on each run in
    scope; a rule already reported is not reported again.

    A MEASUREMENT THAT FAILED IS REPORTED, NOT DROPPED.  A read-only
    panel must not raise a traceback out of a door a user opened, so a
    surprise on one domain does not cost the other domains their rules --
    but it used to cost the reader any account of itself: the door fell
    back to the generic sentence with no tailored repair and nothing said
    a domain had gone unmeasured, which is the flawed-instrument failure
    (a wrong answer given confidently) rather than an instrument saying
    it could not read.  Returns ``(fired, unmeasured)``; ``unmeasured``
    carries one row per distinct failure, and the caller publishes it.
    """

    from gpuwm.physics_compat import conditional_refusals_for

    fired, seen = [], set()
    unmeasured, failures = [], set()
    for index, run in enumerate(runs):
        try:
            rules = conditional_refusals_for(dict(vars(run), **settings))
        except Exception as error:  # noqa: BLE001 - reported below, never swallowed
            text = f"{type(error).__name__}: {error}"
            if text not in failures:
                failures.add(text)
                unmeasured.append({"domain_index": index, "error": text})
            continue
        for rule in rules:
            key = (rule["component"], rule["option"], rule["reason"])
            if key in seen:
                continue
            seen.add(key)
            fired.append(rule)
    return fired, unmeasured


def _summary(rules):
    """The refusing door's own sentence, or the generic one.

    The registry's ``reason`` already carries the breakage and the way
    out, so a front end prints it rather than a restatement written
    here -- the same rule the rest of this module follows.
    """

    return rules[0]["reason"] if rules else GENERIC_SUMMARY


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
              "options": [], "rejected": [], "unmeasured": []}
    refusal_text = None
    try:
        _build(proposed, authority.source)
    except (ValueError, NotImplementedError) as error:
        refusal_text = str(error)
    if refusal_text is None:
        result.update(valid=True, summary="These settings pass the configuration checks.")
        return result

    components = {c["id"]: c for c in physics_components()}
    _, _, domain_keys = _native_contract()
    runs = [d.run for d in original.domains if grid_id == 0 or d.grid_id == grid_id]
    rules, unmeasured = _fired_rules(runs, action["settings"])
    # A domain the registry table could not be evaluated on is named in the
    # payload, so a generic summary here is distinguishable from a summary
    # that is generic because no rule owns this refusal.
    result.update(valid=False, error=refusal_text, summary=_summary(rules),
                  unmeasured=unmeasured)
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

    # The registry's own way out first, where a rule declares one: it is the
    # smallest edit that clears the refusal and it preserves every other draft
    # setting, including ones the generic per-component options overwrite.
    for rule in rules:
        if rule["remedy_settings"]:
            evaluate([], dict(rule["remedy_settings"]),
                     rule["remedy_label"] or rule["reason"])
    # SEEDED REPAIRS, as a table, after the registry's own remedies.  Each
    # row is a refusal whose own text names a setting change that lifts it,
    # offered before the component sweep so the operator sees "keep what
    # you chose" ahead of "replace a scheme".  The sweep below cannot find
    # these: it varies COMPONENT options, and every row here is a plain
    # setting the components do not carry.  The o3input and use_mp_re rows were the two refusals
    # that named their remedy in prose and left the panel offering only
    # scheme substitutions for it.  (A Milbrandt-Yau row once sat here for
    # the mp=9 against RTE+RRTMGP refusal; that scheme now radiates its
    # own radii under RTE+RRTMGP and the parser no longer emits it.)
    from gpuwm.physics_compat import (
        RRTMG_VARIANT_LEGACY, RRTMG_VARIANT_RTE_RRTMGP)
    legacy_arm = {"ra_rrtmg_variant": RRTMG_VARIANT_LEGACY}
    for markers, replacement, label in (
        (("o3input=", RRTMG_VARIANT_RTE_RRTMGP), legacy_arm,
         "Keep this ozone input; use the legacy WRF RRTMG port"),
        (("use_mp_re=", RRTMG_VARIANT_RTE_RRTMGP), legacy_arm,
         "Keep the diagnosed effective radii; use the legacy WRF RRTMG "
         "port"),
    ):
        if all(marker in result["error"] for marker in markers):
            evaluate([], replacement, label)
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


def availability(request):
    """Why each installed physics option is closed to this draft, or open.

    THE FAILURE THIS CLOSES.  The companion greyed shared-scheme options
    with one general note beneath the whole panel, so an operator who
    reached for land surface, soil layers or a radiation variant while a
    single domain was selected read a sentence that did not name their
    option, and an option the parser would refuse at save looked
    selectable until they saved it.

    Nothing here decides anything.  Every option is APPLIED to the saved
    configuration on top of the current draft and handed to the ordinary
    parser, exactly as saving would; the emission route's own gate is
    then asked about the resolved domains.  Every sentence a caller shows
    is the refusing door's own, so a closed option and the refusal it
    will meet cannot say different things.

    AN OPTION CARRIES A SET OF REASONS, NOT ONE.  A single verdict has to
    choose which door to report, and choosing hid the half that mattered:
    a land-surface scheme one domain cannot set AND that every domain is
    refused reported only the second, so the panel offered the cell and
    the save failed on it.  Both are reported now, scope first, in the
    order a reader meets them.

    WHAT A REASON TAKES AWAY IS MEASURED, NOT READ OFF ITS KIND.  Every
    reason says whether it ``closes`` the option, and a caller greys a
    cell if and only if one of them does.  Two things close one: a
    selected scope that cannot apply these values at all, and a door that
    refuses this option under every installed choice for another
    component.  Everything else is a refusal of the option TOGETHER with
    the rest of the draft; the repair search exists to trade one for the
    other and is only reachable from a draft the reader was allowed to
    build, so those cells stay live and say what will happen instead.

    One option's surprise is not the panel's.  An unexpected exception
    from a single option is caught, reported as :data:`CHECK_FAILED` for
    that option alone, and never allowed to take down the answer for the
    other forty.

    Read-only: no candidate configuration is written and no forecast is
    created, the same contract :func:`repairs` carries.
    """
    from gpuwm.config_authority import read_config_authority
    from gpuwm.companion_domains import _apply, _build, _exact_keys, physics_components, REQUEST_SCHEMA
    from gpuwm.case_catalog import _native_contract
    from gpuwm.config import run_preparation_preconditions
    from gpuwm.explain import split as split_explanation
    from gpuwm.physics_menu import switch_route_blocker

    required = {"schema", "config_path", "expected_sha256", "action"}
    _exact_keys(request, required, required, where="physics availability request")
    if request["schema"] != REQUEST_SCHEMA or request["action"].get("kind") != "set_physics":
        raise ValueError("Physics availability requires a set_physics draft")
    authority = read_config_authority(request["config_path"])
    if authority.sha256 != request["expected_sha256"]:
        raise ValueError("The configuration changed; refresh before checking availability")
    raw = tomllib.loads(authority.payload.decode("utf-8-sig"))
    original = _build(raw, authority.source)
    action = copy.deepcopy(request["action"])
    grid_id = action["grid_id"]
    if not isinstance(action.get("settings"), dict):
        raise ValueError("Physics settings must be an object")
    # The forcing source is the config's own advisory hint, read where the
    # loader reads it. A configuration naming no source is asked no route
    # question rather than being answered against a guessed one.
    forcing_source = (raw.get("fetch") or {}).get("source")
    _, shared_keys, domain_keys = _native_contract()
    installed = physics_components()
    runs = [d.run for d in original.domains if grid_id == 0 or d.grid_id == grid_id]

    def stated(kind, error, closes, at=AT_SAVE, remedy=None):
        """One refusal, layered the way this project prints refusals.

        ``reason`` is the ACTION half -- what was refused and the remedy,
        the half a tooltip has room for -- and ``detail`` is the mechanism
        half, kept whole for a reader who opens it.  The split is
        :mod:`gpuwm.explain`'s own, so no front end has to know a sentinel
        exists or pick its own place to cut a paragraph.
        """

        sentence, detail = split_explanation(str(error))
        return {"kind": kind, "reason": sentence.strip(),
                "detail": detail.strip() or None, "closes": bool(closes),
                "at": at, "remedy": remedy}

    def resolve(settings, scope):
        candidate = copy.deepcopy(raw)
        _apply(candidate, {"kind": "set_physics", "grid_id": scope,
                           "settings": settings}, authority.source)
        return _build(candidate, authority.source)

    def refusal(settings, scope):
        """``(kind, error, at)`` from the first door that refuses, or ``None``.

        The kind here names the DOOR only.  Whether that door's refusal
        belongs to the option or to the combination it was asked about is
        a separate measurement, made in :func:`placed`.
        """

        try:
            experiment = resolve(settings, scope)
        except NotImplementedError as error:
            return NOT_IMPLEMENTED, error, AT_SAVE
        except ValueError as error:
            return COMBINATION, error, AT_SAVE
        except Exception as error:  # one option's surprise is not the panel's
            return CHECK_FAILED, error, AT_SAVE
        try:
            for domain in experiment.domains:
                if scope and domain.grid_id != scope:
                    continue
                blocker = switch_route_blocker(vars(domain.run), forcing_source)
                if blocker is not None:
                    return SOURCE_ROUTE, blocker, AT_RUN_PREPARATION
                # And the doors that are about this MACHINE rather than
                # about the source's route: the run door raises exactly
                # these sentences, from this inventory, so the panel
                # reports what a prepared run will meet instead of
                # discovering it after the save.
                unmet = run_preparation_preconditions(domain.run)
                if unmet:
                    return (PRECONDITION, ValueError(unmet[0]),
                            AT_RUN_PREPARATION)
        except Exception as error:  # one option's surprise is not the panel's
            return CHECK_FAILED, error, AT_SAVE
        return None

    def run_wide(settings):
        """The values here that no single domain owns and that differ."""

        return {key: value for key, value in settings.items()
                if key in shared_keys and key not in domain_keys
                and not all(getattr(run, key, None) == value for run in runs)}

    def scoped(settings):
        """What this edit actually has to change at the selected scope.

        A shared value already in effect needs no edit, so it can never be
        the reason an option is closed.
        """

        if not grid_id:
            return dict(settings)
        return {key: value for key, value in settings.items() if key in domain_keys
                or not all(getattr(run, key, None) == value for run in runs)}

    def scope_refusal(settings):
        """The per-domain application's own words, or ``None``.

        Measured by asking that application -- the door a save passes
        through -- to set exactly the run-wide values this edit needs on
        the selected domain.  Nothing is read off the setting names, and
        nothing is inferred from what All domains would do: a scheme that
        is shared IN FACT is reported as shared whether or not selecting
        All domains would then be admitted.
        """

        needed = run_wide(settings)
        if not (grid_id and needed):
            return None
        try:
            _apply(copy.deepcopy(raw), {"kind": "set_physics", "grid_id": grid_id,
                                        "settings": needed}, authority.source)
        except ValueError as error:
            return error
        except Exception:
            return None
        return None

    def substitutions(option_settings):
        """This option alone, then with one other installed choice added.

        Bounded on purpose, and asked run-wide with the rest of the draft
        cleared: the question is whether the option can run on this setup
        AT ALL, not whether it runs beside what is on screen.  One
        substitution at a time is also what "changing something else would
        clear this" means to the reader about to click the cell.
        """

        yield dict(option_settings)
        for component in installed:
            for other in component["options"]:
                if any(key in option_settings for key in other["settings"]):
                    continue
                yield dict(option_settings, **other["settings"])

    def unreachable(option_settings):
        """Whether NO installed choice for another component admits this
        option on this setup.

        The measurement behind a greyed cell.  It stops at the first
        combination the whole parser and the route gate accept, so an
        option one other pick would open is never walled -- that wall was
        the defect: on a shipped three-domain setup 16 of 19 refusals
        belong to a combination, every one of them was selectable before
        this panel existed, and the repair search is only reachable from a
        draft the reader was allowed to build.
        """

        return all(refusal(candidate, 0) is not None
                   for candidate in substitutions(option_settings))

    #: "Not measured yet", distinct from "measured, and nothing refused".
    unmeasured = object()

    def placed(found, option_settings, everywhere=unmeasured):
        """One door's refusal, placed: this OPTION, or this COMBINATION.

        A wall is claimed only where the SAME door refuses the option on
        its own and no installed choice for another component admits it.
        The closing reason is then that door's words about the OPTION --
        not its words about the draft the option was asked beside, which
        can be a different refusal entirely.  Printing the second over a
        wall the first produced sends the reader to change a setting that
        would not have opened the cell.  Both are reported when they
        differ, the wall first.
        """

        kind, error, at = found
        if kind == PRECONDITION:
            # NEVER A WALL, and never relabelled.  The way out this door
            # names is a [shared] setting about the option being asked
            # about (``mp28_aerosol_source``, which is a tree-wide key --
            # ``gpuwm/experiment.py`` refuses it on a ``[[domain]]``
            # table), and ``unreachable`` measures walls by varying OTHER
            # components, so it cannot see that way out and must not
            # claim one.  Greying a cell a documented one-line setting
            # opens is the defect this panel's wall
            # measurement exists to avoid; the reader keeps the cell, and
            # the sentence says what a prepared run will meet.
            return [stated(PRECONDITION, error, closes=False, at=at)]
        if kind in (NOT_IMPLEMENTED, SOURCE_ROUTE):
            alone = refusal(option_settings, 0)
            if alone is not None and alone[0] == kind and unreachable(option_settings):
                walls = [stated(kind, alone[1], closes=True, at=alone[2])]
                if str(alone[1]) != str(error):
                    walls.append(stated(kind, error, closes=False, at=at))
                return walls
        if kind == CHECK_FAILED:
            return [stated(CHECK_FAILED, error, closes=False, at=at)]
        # THE REMEDY IS ONE MEASUREMENT, MADE ONCE.  "Select All domains to
        # use this" is claimed only where this option, beside the rest of
        # the draft, is admitted run-wide -- refusal(option, 0).  A caller
        # that already made that measurement hands it over instead of
        # paying for a second full parse of the same candidate, so the two
        # can no longer answer differently about one cell.
        if everywhere is unmeasured:
            everywhere = refusal(dict(action["settings"], **option_settings),
                                 0) if grid_id else None
        remedy = ALL_DOMAINS if grid_id and everywhere is None else None
        return [stated(COMBINATION, error, closes=False, at=at, remedy=remedy)]

    def verdict(extra, isolate=True):
        """Every reason this option -- or the draft itself -- is closed.

        ``isolate`` is false for the draft, which is not an option: no
        substitution isolates it, nothing is greyed for it, and the door
        that spoke keeps its own kind.
        """

        settings = dict(action["settings"])
        settings.update(extra)
        outside = scope_refusal(settings)
        if outside is not None:
            everywhere = refusal(settings, 0)
            reasons = [stated(SHARED_SCOPE, outside, closes=True,
                              remedy=None if everywhere else ALL_DOMAINS)]
            # Selecting All domains is the remedy that refusal offers. Where
            # All domains is refused too, the reader is told here rather than
            # after taking the advice and meeting the second door.
            if everywhere is not None:
                reasons.extend(placed(everywhere, extra, everywhere) if isolate
                               else [stated(everywhere[0], everywhere[1], closes=False, at=everywhere[2])])
            return {"available": False, "reasons": reasons}
        found = refusal(scoped(settings), grid_id)
        if found is None:
            return {"available": True, "reasons": []}
        if not isolate:
            return {"available": False,
                    "reasons": [stated(found[0], found[1], closes=False, at=found[2])]}
        return {"available": False, "reasons": placed(found, extra)}

    components = [{"id": component["id"], "label": component["label"],
                   "options": [dict(verdict(option["settings"]), id=option["id"])
                               for option in component["options"]]}
                  for component in installed]
    result = {"schema": AVAILABILITY_SCHEMA, "source_path": str(authority.source),
              "source_sha256": authority.sha256, "draft_sha256": draft_digest(action),
              "action": action, "created": False, "forecast_started": False,
              "forcing_source": forcing_source, "draft": verdict({}, isolate=False),
              "components": components}
    if hashlib.sha256(authority.source.read_bytes()).hexdigest() != authority.sha256:
        raise ValueError("The configuration changed while checking availability; refresh and try again")
    return result
