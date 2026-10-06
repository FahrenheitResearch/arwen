"""Source chain facts shared by input validation and forecast planning."""
from __future__ import annotations
from typing import Any


def local_input_requested(hints) -> bool:
    """An explicit staged root remains usable when a source gains acquisition."""
    from gpuwm import fetch_routes
    source = hints.get("source")
    return bool(drivability_for(source).get("requires_source_root") or
                (hints.get("source_root") and fetch_routes.source_root_layout(str(source)) is not None))


def drivability_for(source: object) -> dict[str, Any]:
    """The drivability verdict for a configuration's own spelling.

    :func:`intent_drivability` is keyed by REGISTRY ID.  A configuration,
    a ``--source`` flag and an emitted ``[fetch]`` table may each spell
    an alias instead, and an alias that missed this lookup read as "no
    verdict": the local-input admission, which lives in the verdict, was
    silently skipped and the plan went on to look for a download route
    that does not exist.  Every door asks through here so an alias
    cannot admit what its registry id refuses.
    """

    from gpuwm.source_adapters import get_source_adapter

    name = str(source or "")
    try:
        canonical = get_source_adapter(name).source_id
    except ValueError:
        # A future source stays a name: selection explains its lack of a
        # native adapter rather than discarding the question.
        canonical = name
    return intent_drivability().get(canonical, {})


def candidate_route_chain(source: object) -> str:
    """The chain a configuration naming SOURCE will dispatch to.

    The prepared route is three chains wearing one name, and which one a
    configuration reaches is decided by its source's registry row rather
    than by a list of model names.  This is that decision, and it lives
    here rather than in the dispatcher because two callers need it: the
    dispatcher itself (``gpuwm.runplan._chain_key``, which turns a plan
    into a chain) and every door that publishes a candidate and has to
    write the files that chain reads beside it
    (:func:`gpuwm.hrrr_route_inputs.candidate_companions`).  One
    function, so the files a candidate is given and the files its run
    reads cannot be decided differently -- and a preprocessing install
    that carries those doors without the forecast dispatcher can still
    ask.

    Unlike a LAUNCH, this refuses nothing: a door has not been asked to
    start anything, and a configuration whose source has no launch route
    still has to be editable.
    """

    name = (str(source) if source is not None else "").strip()
    if name:
        chain = str((drivability_for(name) or {}).get("chain") or "")
        if chain.startswith("prepared:"):
            return chain
    return "prepared:go"


def intent_drivability() -> dict[str, dict[str, Any]]:
    """Derive chain availability from the prep dispatcher and source facts.

    A composed source without a download route is structurally drivable
    from local inputs. The local-root requirement is checked at review,
    before any acquisition or preparation stage.
    """
    from gpuwm import fetch_routes
    from gpuwm.source_adapters import source_adapters, wizard_planable_source_ids
    from gpuwm.source_authorities import packaged_profile
    from gpuwm.source_cli import preparation_runners

    planable = set(wizard_planable_source_ids())
    downloadable = set(fetch_routes.all_fetchable_sources())
    table = set(fetch_routes.route_ids())
    runners = preparation_runners()

    def _one(adapter) -> dict[str, Any]:
        source = adapter.source_id
        def refused(reason):
            return {"routes": [], "chain": None, "refusal": reason}
        if not adapter.runnable:
            return refused(f"{source!r} declares no runnable implementation route "
                           f"(status {adapter.status.value!r}). "
                           "Choose a runnable source from gpuwm sources.")
        if source not in planable:
            return refused(f"{source!r} declares no forcing_interval_seconds. "
                           "Declare its boundary cadence before using gpuwm domain.")
        runner = runners.get(adapter.runner)
        if runner is None:
            return refused(f"{source!r} names runner {adapter.runner!r}, which no "
                           "run-plan chain executes. Supply a supported prepared bundle.")
        if runner.chain == "experiment":
            return {"routes": ["experiment"], "chain": "experiment", "refusal": None}
        if adapter.member_set is not None:
            from gpuwm.forcing_member import member_contract
            try:
                member_contract(source)
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                return {"routes": [], "chain": None, "refusal": (
                    f"{source!r} cannot bind its member selection: {error}")}
        if runner.chain == "prepared:staged":
            if adapter.packaged_profile is None:
                return refused(f"{source!r} ships no packaged profile for its mapped "
                               "preparation. Supply a caller-authored prepared bundle.")
            try:
                state = packaged_profile(adapter.packaged_profile)["composition_state"]
            except (OSError, KeyError, TypeError, ValueError) as error:
                return refused(f"{source!r} has an unreadable preparation profile: "
                               f"{error}. Restore the matching packaged authorities.")
            if state != "composed":
                return refused(f"{source!r} declares composition_state {state!r}. "
                               "Bind the missing composition before preparing it.")
            if source not in downloadable:
                if runner.local_kind is None:
                    return refused(f"{source!r} has no local input preparation contract. "
                                   "Supply a verified prepared bundle.")
                why = fetch_routes.acquisition_refusal_reason(source)
                return {"routes": ["prepared"], "chain": runner.chain,
                        "refusal": None, "requires_source_root": True,
                        "source_root_reason": why}
            if not fetch_routes.publishes_prep_handoff(source):
                return refused(f"{source!r} publishes no bound preparation handoff. "
                               "Supply a verified prepared bundle.")
        elif source not in downloadable:
            return refused(f"{source!r} has no acquisition route or local input contract. "
                           "Supply a verified prepared bundle.")
        return {"routes": ["prepared"], "chain": runner.chain, "refusal": None}

    return {adapter.source_id: _one(adapter) for adapter in source_adapters()}


def drivability_for(source: object) -> dict[str, Any]:
    """Resolve aliases before looking up a source's preparation contract."""
    from gpuwm.source_adapters import get_source_adapter
    name = str(source or "")
    try:
        name = get_source_adapter(name).source_id
    except ValueError:
        pass
    return intent_drivability().get(name, {})
