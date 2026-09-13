"""Source chain facts shared by input validation and forecast planning."""
from __future__ import annotations
from typing import Any

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
