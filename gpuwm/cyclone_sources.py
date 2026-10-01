"""Source contracts shared by cyclone configuration and selection clients."""
from __future__ import annotations

from datetime import datetime
import math


def source_ids() -> tuple[str, ...]:
    """Sources with both an initialization route and an acquisition door."""
    from gpuwm.fetch_routes import all_fetchable_sources
    from gpuwm.source_adapters import wizard_planable_source_ids

    return tuple(sorted(set(all_fetchable_sources()) & set(wizard_planable_source_ids())))


def source_adapter(source: str):
    from gpuwm.source_adapters import get_source_adapter

    adapter = get_source_adapter(source)
    if adapter.source_id not in source_ids():
        raise ValueError(
            f"{adapter.display_title} has no complete acquisition and initialization "
            "route for this setup; supply prepared inputs through gpuwm prep, or "
            "choose a source listed by cyclone-setup --list-sources")
    return adapter


def selected_member(source: str, member: str | None = None) -> str | None:
    """Preserve the fetch route's member vocabulary, including its default."""
    from gpuwm.fetch_routes import route_for, route_ids, resolve_member

    adapter = source_adapter(source)
    if adapter.source_id in route_ids():
        return resolve_member(route_for(adapter.source_id), member)[0] or None
    if member is not None:
        # What is true and what is not: the row carries no member axis on
        # the acquisition route THIS door authors, which is not the same
        # claim as "this source publishes no ensemble".  The menu's own
        # `members` list is the set a selection can come from, so the
        # sentence sends the reader there rather than asserting something
        # about the source's products.
        raise ValueError(
            f"{adapter.display_title} carries no member grammar on the "
            "acquisition route this setup authors; omit --member, or choose "
            "a source whose --list-sources row lists members")
    return None


def resolve_cycle(raw: str, *, source: str, latest: bool = False) -> datetime:
    from gpuwm import domain_wizard as dw
    from gpuwm.source_cycles import cycle_grid_for

    adapter = source_adapter(source)
    text = raw.strip()
    if text.lower() == "latest" and not latest:
        raise ValueError(
            "Select a center on a resolved source map first; creation requires "
            "that map's exact cycle")
    if len(text) == 10 and text.isdecimal():
        text = datetime.strptime(text, "%Y%m%d%H").strftime("%Y-%m-%dT%H")
    moment = dw._resolve_cycle(text, source=adapter.source_id, hours=0)
    grid = cycle_grid_for(adapter.source_id)
    if grid is not None and (moment.hour not in grid.hours or moment.minute
                             or moment.second or moment.microsecond):
        hours = "/".join(f"{hour:02d}" for hour in grid.hours)
        raise ValueError(
            f"{adapter.display_title} initialization must name a published "
            f"{hours} UTC cycle; select one of those hours")
    return moment


def global_alternatives() -> str:
    from gpuwm.source_adapters import get_source_adapter

    return ", ".join(source for source in source_ids()
                     if get_source_adapter(source).coverage_window is None)


def validate_center(source: str, point: tuple[float, float]) -> None:
    from gpuwm.source_coverage import points_outside

    adapter = source_adapter(source)
    if (len(point) != 2 or not all(math.isfinite(v) for v in point)
            or not -85 <= point[0] <= 85 or not -180 <= point[1] <= 180):
        raise ValueError("Select a finite center within latitude +/-85 and longitude +/-180")
    if bool(points_outside(adapter.coverage_window, point[0], point[1])):
        raise ValueError(
            f"The cyclone position ({point[0]:.4f}, {point[1]:.4f}) lies outside "
            f"{adapter.display_title}'s grid; choose a covering source such as "
            f"{global_alternatives()}")


def forecast_horizon(source: str, moment: datetime) -> int | None:
    """The last lead this source publishes for THIS cycle, or ``None``.

    The cycle grid answers first because a horizon is a property of the
    cycle hour on several producers -- one runs 48 h at 00/06/12/18Z and
    18 h otherwise -- and the registry row's ``max_forecast_hour`` is the
    fallback for a source with no grid.  ONE function, because the lead a
    run may begin at and the window it may cover are judged against the
    same number, and two spellings of it would disagree on exactly the
    cycles where it matters.
    """
    from gpuwm.source_cycles import cycle_grid_for

    adapter = source_adapter(source)
    grid = cycle_grid_for(adapter.source_id)
    horizon = grid.horizon(moment) if grid is not None else None
    if horizon is None:
        horizon = adapter.max_forecast_hour or None
    return None if horizon is None else int(horizon)


def resolve_start_hour(start_hour, *, source: str, moment: datetime,
                       hours: int = 0) -> int:
    """The forecast LEAD this setup begins at, judged against the source.

    ONE validator for both surfaces this door has -- the selection map,
    which has no fetch table to be checked by, and the configuration,
    whose fetch table is checked by the route itself -- so the preview
    and the configuration cannot disagree about which leads a cycle has.
    ``hours`` is the window that begins at the lead: zero asks only
    whether the lead itself is published, which is the map's question.
    """
    from gpuwm import domain_wizard as dw

    adapter = source_adapter(source)
    if type(start_hour) is not int or start_hour < 0:
        raise ValueError("--start-hour must be a nonnegative whole forecast lead")
    if start_hour and not dw.source_reaches_forecast_leads(adapter.source_id):
        raise ValueError(
            f"--start-hour {start_hour}: {adapter.display_title} publishes no "
            "forecast leads (its registry row declares max_forecast_hour = 0), "
            "so every time in it is an analysis at its own valid time and there "
            "is no lead to begin at; name the time you want with --cycle")
    horizon = forecast_horizon(adapter.source_id, moment)
    if horizon is not None and start_hour + hours > horizon:
        # The lead AND the window, against the horizon this cycle declares:
        # a run that begins at f186 and covers six hours needs f192, and
        # which of the two walked off the end is what the reader has to know
        # to choose between --start-hour and --hours.
        needed = (f"begins at f{start_hour:03d} and needs f{start_hour + hours:03d}"
                  if hours else f"begins at f{start_hour:03d}")
        raise ValueError(
            f"{adapter.display_title}'s {moment:%H} UTC cycle ends at "
            f"f{horizon:03d}, but this setup {needed}; start at an earlier "
            "lead, shorten --hours, or choose a cycle with a longer forecast")
    return start_hour


def window_cadence_hours(source: str, *, moment: datetime, start_hour: int,
                         hours: int) -> int | None:
    """The file spacing this door fetches a window at, or None.

    The domain door's rule (:func:`gpuwm.domain_wizard._fetch_cadence_h`)
    asked of the one cycle this door names: the source's usual spacing
    where that cycle's ladder publishes it over the whole window, and the
    coarser spacing the window runs into otherwise (IFS past f144, GEFS
    past f240).  Passing the usual spacing, this door refused those
    windows while `gpuwm fetch` and `gpuwm domain` took them.  ONE
    function, because the fetch table and the boundary interval the
    configuration is priced and namelisted with must read the same one.
    """
    from gpuwm import domain_wizard as dw

    return dw._fetch_cadence_h(source_adapter(source).source_id, start_hour,
                               hours, cycle=moment)


def fetch_hints(*, source: str, moment: datetime, hours: int,
                projection: dict, dims: tuple[int, int], dx_m: float,
                member: str | None = None, start_hour: int = 0) -> dict:
    """A source's actual fetch contract, validated before configuration output.

    ``start_hour`` is the forecast LEAD the run begins at, so the window
    this table asks for is f``start_hour`` through f``start_hour + hours``
    of the named cycle rather than that cycle's analysis.  It is the fetch
    route's own grammar -- the same ``forecast_start_hour`` key
    ``gpuwm fetch`` takes and the same ladder :func:`resolve_leads`
    checks -- so a source publishes leads to this door exactly when it
    publishes them to that one, and no source is named here.
    """
    from gpuwm import domain_wizard as dw
    from gpuwm.fetch import fetch_accepts_area, validate_fetch_hints
    from gpuwm.fetch_routes import route_for, route_ids, resolve_leads

    adapter = source_adapter(source)
    source = adapter.source_id
    if type(hours) is not int or hours < 1:
        raise ValueError("Cyclone duration must be a positive integer number of hours")
    selection = selected_member(source, member)
    interval_h = float(adapter.forcing_interval_seconds) / 3600.
    if interval_h < 1 or not interval_h.is_integer():
        raise ValueError(
            f"{adapter.display_title}'s forcing interval is {interval_h:g} hours, "
            "which the hourly acquisition interface cannot express; supply prepared inputs")
    rounded = int(math.ceil(hours / interval_h) * interval_h)
    start_hour = resolve_start_hour(start_hour, source=source, moment=moment,
                                    hours=rounded)
    cadence = window_cadence_hours(source, moment=moment,
                                   start_hour=start_hour, hours=rounded)
    if cadence is not None and rounded % cadence:
        # Whole steps of the spacing the cycle's ladder publishes past the
        # lead where the finer one ends, and that longer window is the one
        # the horizon is judged on.
        rounded = max(cadence, int(math.ceil(rounded / cadence)) * cadence)
        start_hour = resolve_start_hour(start_hour, source=source,
                                        moment=moment, hours=rounded)
    if source in route_ids():
        resolve_leads(route_for(source), moment, rounded, cadence=cadence,
                      start_hour=start_hour)
    refusal = dw.source_coverage_refusal(projection, *dims, source=source, root_dx_m=dx_m)
    if refusal:
        raise ValueError(refusal + "; covering sources: " + global_alternatives())
    hints = {"source": source, "cycle": moment.strftime("%Y-%m-%dT%H"),
             "hours": rounded, "out": f"data/{source}-cyclone-{moment:%Y%m%d%H}"}
    if cadence is not None:
        hints["cadence"] = cadence
    if start_hour:
        # Written only when there IS a lead, so an analysis-initialized
        # setup emits the table every prior release emitted, byte for byte.
        hints["forecast_start_hour"] = start_hour
    if selection is not None:
        hints["member"] = selection
        hints["out"] += "-" + selection
    if fetch_accepts_area(source):
        hints["area"] = dw.fetch_area_hint(projection, *dims, source=source, root_dx_m=dx_m)
    # The run door's own keyless default, asked rather than copied: a
    # reanalysis setup that names no provider reads the analysis-ready
    # store.  Left to the fetch's keyed default, a cyclone configuration
    # authored on any computer without a CDS key failed at acquisition.
    from gpuwm.runplan import _keyless_era5_default
    hints = _keyless_era5_default(hints, source)
    validate_fetch_hints(hints, source="cyclone setup")
    return hints


def follow_statics(source: str) -> dict:
    """How a moving nest's statics arrive on the chain this source runs on.

    The run door's own table, asked here rather than copied: see
    :func:`gpuwm.runplan.source_follow_statics`.  The cyclone door
    authors a following nest for every source, and on a chain whose
    preparation seals no corridor that nest is refused by the run door
    at ITS plan review.  Asking the same function while the setup is
    still on the screen is what keeps the two doors saying the same
    thing about one configuration.
    """
    from gpuwm.runplan import source_follow_statics

    return source_follow_statics(source_adapter(source).source_id)


def moving_nest_sources() -> tuple[str, ...]:
    """Planable sources whose chain can carry this door's following nest."""
    return tuple(source for source in source_ids()
                 if follow_statics(source)["integrates_moving_nest"])


def moving_nest_note(source: str) -> dict:
    """What the run door will do with this configuration's moving nest.

    Stated on every emitted document, in both directions, because the
    reader who picks a source needs the answer before the run door gives
    it: a supported chain says which delivery feeds the nest, and an
    unsupported one says what is missing, what the run door will do
    about it, and which sources do carry a moving nest today.  It is not
    a refusal: the configuration is authored, priced and reviewable on
    every planable source, and this says exactly which part of it the
    chain cannot execute.

    THREE ANSWERS, not two.  A row that reaches no launch chain at all is
    not a statics answer: the run door refuses the source before the nest
    is ever considered, so this says so in the run door's own words and
    offers only the way on that works.  Dropping the follow source would
    leave such a configuration exactly as unlaunchable as it was.
    """
    decision = dict(follow_statics(source))
    adapter = source_adapter(source)
    covering = ", ".join(moving_nest_sources())
    if decision["integrates_moving_nest"]:
        decision["note"] = (
            f"{adapter.display_title} runs on the {decision['chain']} chain, "
            f"whose preparation feeds a moving nest by {decision['delivery']}, "
            "so the following nest in this configuration is integrated as "
            "authored.")
        return decision
    if decision["launch_refusal"] is not None:
        decision["note"] = (
            f"{adapter.display_title} reaches no launch route this release "
            "dispatches to, so the moving nest is not what stops this "
            "configuration: `gpuwm go` refuses the whole of it at its own "
            "plan review, before any fetch, with \"" +
            decision["launch_refusal"] + "\", because "
            f"{decision['reason']}. This setup is authored, priced and "
            "reviewable as it stands, and no change to the nest makes it "
            f"launch. Ways on: author the same centre and cycle with "
            f"--source from {covering}, which both launch and carry a "
            "moving nest.")
        return decision
    decision["note"] = (
        f"{adapter.display_title} runs on the {decision['chain']} chain, "
        "which cannot supply the statics a moving nest travels over: "
        f"{decision['reason']}. This setup is authored, priced and "
        "reviewable as it stands, and `gpuwm go` refuses its following "
        "nest at its own plan review, before any fetch. Ways on: author "
        f"the same centre and cycle with --source from {covering}, which "
        "do carry a moving nest; or run this configuration with a "
        "bounds-only [relocation], which keeps the nest where it was "
        "placed and needs no corridor.")
    return decision


def source_options() -> list[dict]:
    """A client menu derived at request time, with no parallel source list."""
    from gpuwm.fetch_routes import member_tokens, route_for, route_ids
    from gpuwm.source_cycles import cycle_grid_for

    options = []
    for source in source_ids():
        adapter = source_adapter(source)
        grid = cycle_grid_for(source)
        members = (list(member_tokens(route_for(source))) if source in route_ids() else [])
        moving = follow_statics(source)
        options.append({"source": source, "label": adapter.display_title,
                        "members": members, "default_member": selected_member(source),
                        "forcing_interval_seconds": adapter.forcing_interval_seconds,
                        "cycle_hours": list(grid.hours) if grid else [],
                        # THE LEAD A RUN MAY BEGIN AT, on the row where the
                        # source is chosen.  A form that offers --start-hour
                        # has to bound the field before a cycle exists, and
                        # this is the registry's own declared ceiling: 0
                        # means the row publishes analyses only, so the field
                        # does not apply to it at all.  The per-cycle horizon
                        # is still the authority and still refuses
                        # (:func:`forecast_horizon`); this is what a picker
                        # can check without asking the mirrors anything.
                        "max_forecast_hour": int(adapter.max_forecast_hour or 0),
                        "coverage_envelope": (list(adapter.coverage_window.envelope())
                                              if adapter.coverage_window else None),
                        # The run door's answer, on the row where the source
                        # is chosen: the delivery word that feeds a moving
                        # nest on this source's chain, or null where that
                        # chain seals none.  A picker that shows the menu
                        # shows the limit with it, instead of leading the
                        # reader to a launch refusal.
                        "follow_statics": moving["delivery"]})
    return options


def plan_route(source: str) -> str:
    """The run-plan route that runs the configuration this source's cyclone setup writes.

    Read from the source's row, as :func:`declared_case_data` reads it: a
    row that declares a case-data file gets a ``[case_data]`` table, which
    the config-driven route decodes (``experiment``); every other row's
    configuration carries only ``[fetch]``, and the config-driven route
    refuses it as belonging to the prepared route (``prepared``, what
    ``gpuwm go`` runs).  The breakage this prevents (A152): a
    storm-following plan written with one route for every source, which
    ``gpuwm run-plan`` refused for a GFS storm.
    """

    return ("experiment" if source_adapter(source).case_data_file is not None
            else "prepared")


def declared_case_data(source: str, hints: dict, config_path: str) -> dict | None:
    """Bind the config-driven preparation recipe when the source declares one."""
    import os
    from pathlib import Path

    adapter = source_adapter(source)
    if adapter.case_data_file is None:
        return None
    path = Path(config_path).resolve()
    # The file the declared fetch publishes, which depends on the provider
    # it names; the adapter's row is the default provider's name.
    name = adapter.case_data_file
    if "era5_provider" in hints:
        from gpuwm.fetch import era5_combined_name
        name = era5_combined_name(hints["era5_provider"])
    forcing = Path(hints["out"]).resolve() / name
    return {"forcing": [Path(os.path.relpath(forcing, path.parent)).as_posix()],
            "vtable": path.with_suffix(".Vtable").name,
            "forcing_interval_s": adapter.forcing_interval_seconds,
            "wps_namelist": path.with_suffix(".namelist.wps").name,
            "geog_root": "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG",
            "sfcp_to_sfcp": True, "output_domain": 1,
            "output_title": f"gpuwm {adapter.display_title} cyclone"}


def companion_input_files(source: str, config_path) -> tuple:
    from pathlib import Path
    from gpuwm import source_adapters

    adapter = source_adapter(source)
    if adapter.case_data_vtable is None:
        return ()
    table = Path(source_adapters.__file__).parent / adapter.case_data_vtable
    return ((Path(config_path).with_suffix(".Vtable"), table.read_text(encoding="utf-8")),)
