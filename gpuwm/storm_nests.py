"""Storm-following nests on command or automatically: the ``[storm_nests]`` key.

One table on any single-root configuration (the HRRR v4 recipe is the
headline) asks for fine nests that spawn on storms, follow them and stop on
their own.  It is authoring, not a new engine path: the table expands into
dormant ``[[domain]]`` slots that the existing lifecycle machinery runs
(``spawn`` / ``follow`` / ``retire`` / ``rearm``, docs/nest-lifecycle.md),
so every slot is priced before the first step and the prepared tree route
walks the run in legs (gpuwm.runtime.walk_spawn_legs).

Two ways a nest is born:

* **automatic** -- ``slots`` dormant slots watch the parent's 2-5 km updraft
  helicity (the running maximum since the last look, the consumer-owned
  ``uh_spawn_window``).  A slot fires on the strongest rotating storm in
  the parent that no live nest already covers, follows that storm's UH
  centroid, and retires when the UH under it stays quiet (the storm died
  or left the domain) or when it merges into an older nest's storm.
* **on command** -- each ``points`` entry (``{lat, lon, at}``) is one slot
  born at that point at that time (``at`` is an ISO time or seconds into
  the run), which then follows and retires exactly like an automatic one.

The defaults are the convection-allowing verification standard rather than
tuned numbers: 75 m2 s-2 of 2-5 km UH is the 3 km surrogate-severe
threshold (Sobash et al. 2011, 2016; the HRRR/HREF UH guidance), a nest is
kept while its storm stays above 25 m2 s-2, and it is retired after 30 min
below that once it has lived an hour.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

TABLE = "storm_nests"
SCHEMA = "gpuwm-storm-nests.v1"

#: Every key the table takes; any other is refused by name.
KEYS = frozenset({
    "slots", "max_slots", "dx_km", "size_km", "detector",
    "spawn_threshold", "follow_threshold", "fallback_threshold",
    "retire_threshold", "retire_sustained_s", "min_lifetime_s",
    "merge_overlap", "earliest_s", "latest_s", "cadence_s",
    "follow_cadence_s", "history_interval_s", "radt", "points",
    "max_firings", "rearm_cooldown_s", "vram_gib",
})
POINT_KEYS = frozenset({"lat", "lon", "at", "at_s", "name"})

DEFAULTS = {
    "slots": 3,
    "max_slots": 8,
    "dx_km": 1.0,
    "size_km": 240.0,
    "detector": "uh",
    "spawn_threshold": 75.0,
    "follow_threshold": 50.0,
    "fallback_threshold": 25.0,
    "retire_threshold": 25.0,
    "retire_sustained_s": 1800.0,
    "min_lifetime_s": 3600.0,
    "merge_overlap": 0.5,
    "cadence_s": 900.0,
    "follow_cadence_s": 900.0,
    "history_interval_s": 900.0,
    "radt": 5.0,
    "max_firings": 4,
    "rearm_cooldown_s": 1800.0,
}
#: Detector -> (spawn/follow/retire field, units).  ``reflectivity`` reads
#: the composite refl_10cm the parent stashes at its history cadence.
DETECTORS = {"uh": "m2 s-2", "reflectivity": "dBZ"}


class StormNestsRefusal(ValueError):
    """A ``[storm_nests]`` request this module will not author."""


def _number(table, key, *, positive=True):
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or (positive and float(value) <= 0):
        raise StormNestsRefusal(
            f"[{TABLE}] {key} must be a finite {'positive ' if positive else ''}"
            f"number, got {value!r}")
    return float(value)


def resolve_options(table: dict) -> dict:
    """The table with defaults filled in, validated; unknown keys refused."""
    if not isinstance(table, dict):
        raise StormNestsRefusal(f"[{TABLE}] must be a table")
    unknown = sorted(set(table) - KEYS)
    if unknown:
        raise StormNestsRefusal(
            f"[{TABLE}] does not have key(s) {unknown}; it takes {sorted(KEYS)}")
    out = dict(DEFAULTS)
    out.update(table)
    if out["detector"] not in DETECTORS:
        raise StormNestsRefusal(
            f"[{TABLE}] detector must be one of {sorted(DETECTORS)}, got "
            f"{out['detector']!r}")
    slots = out["slots"]
    if slots != "auto" and (isinstance(slots, bool) or not isinstance(slots, int)
                            or slots < 0):
        raise StormNestsRefusal(
            f"[{TABLE}] slots must be a whole number >= 0 or \"auto\" "
            f"(priced from the card), got {slots!r}")
    for key in ("dx_km", "size_km", "spawn_threshold", "follow_threshold",
                "fallback_threshold", "retire_threshold", "cadence_s",
                "follow_cadence_s", "history_interval_s", "radt"):
        out[key] = _number(out, key)
    for key in ("retire_sustained_s", "min_lifetime_s", "rearm_cooldown_s"):
        out[key] = _number(out, key, positive=False)
    if not 0.0 < float(out["merge_overlap"]) <= 1.0:
        raise StormNestsRefusal(
            f"[{TABLE}] merge_overlap must be a fraction in (0, 1], got "
            f"{out['merge_overlap']!r}")
    for key in ("max_slots", "max_firings"):
        if isinstance(out[key], bool) or not isinstance(out[key], int) or out[key] < 1:
            raise StormNestsRefusal(f"[{TABLE}] {key} must be a whole number >= 1")
    points = out.get("points") or []
    if not isinstance(points, list):
        raise StormNestsRefusal(f"[{TABLE}] points must be an array of tables")
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise StormNestsRefusal(f"[{TABLE}] points[{index}] must be a table")
        bad = sorted(set(point) - POINT_KEYS)
        if bad:
            raise StormNestsRefusal(
                f"[{TABLE}] points[{index}] does not have key(s) {bad}; it takes "
                f"{sorted(POINT_KEYS)}")
        for key in ("lat", "lon"):
            if key not in point:
                raise StormNestsRefusal(f"[{TABLE}] points[{index}] needs {key}")
    out["points"] = points
    return out


def _root(raw: dict) -> dict:
    domains = raw.get("domain")
    if isinstance(domains, dict):
        domains = [domains]
    if not isinstance(domains, list) or not domains:
        raise StormNestsRefusal(f"[{TABLE}] needs a configuration with a root [[domain]]")
    roots = [d for d in domains if int(d.get("parent_id", 0)) == 0]
    if len(roots) != 1:
        raise StormNestsRefusal(f"[{TABLE}] needs exactly one root domain")
    if any(d.get("spawn") is not None for d in domains):
        raise StormNestsRefusal(
            f"[{TABLE}] authors the dormant slots itself; this configuration "
            "already declares spawn tables. Use one or the other")
    return roots[0]


def _start_time(raw: dict) -> datetime:
    value = raw.get("experiment", {}).get("start_time")
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "")).replace(tzinfo=None)
    raise StormNestsRefusal(f"[{TABLE}] needs [experiment] start_time to place timed points")


def _root_grid(raw: dict, root: dict):
    from gpuwm.static.projection import projection_class

    projection = raw.get("projection")
    if not isinstance(projection, dict):
        raise StormNestsRefusal(f"[{TABLE}] needs a [projection] table to place points")
    cls = projection_class(str(projection.get("map_proj", "lambert")))
    dx = float(root["dx"])
    return cls(ref_lat=float(projection["ref_lat"]), ref_lon=float(projection["ref_lon"]),
               truelat1=float(projection.get("truelat1", projection["ref_lat"])),
               truelat2=float(projection.get("truelat2", projection.get("truelat1", projection["ref_lat"]))),
               stand_lon=float(projection.get("stand_lon", projection["ref_lon"])),
               dx=dx, dy=float(root.get("dy", dx)),
               e_we=int(root["nx"]) + 1, e_sn=int(root["ny"]) + 1)


def slot_shape(raw: dict, options: dict) -> dict:
    """Ratio, child size and the admissible placement band on the root."""
    root = _root(raw)
    root_dx_km = float(root["dx"]) / 1000.0
    ratio = int(round(root_dx_km / float(options["dx_km"])))
    if ratio < 2 or abs(root_dx_km / ratio - float(options["dx_km"])) > 1e-6 * root_dx_km:
        raise StormNestsRefusal(
            f"[{TABLE}] dx_km = {options['dx_km']:g} is not a whole fraction "
            f"(>= 2) of the root's {root_dx_km:g} km")
    parent_cells = max(2, int(round(float(options["size_km"]) / root_dx_km)))
    child_n = parent_cells * ratio
    experiment = raw.get("experiment", {})
    clear = int(experiment.get("spec_bdy_width", 5)) + int(experiment.get("blend_width", 5)) + 1
    nx, ny = int(root["nx"]), int(root["ny"])
    if parent_cells + 2 * clear >= min(nx, ny):
        raise StormNestsRefusal(
            f"[{TABLE}] a {options['size_km']:g} km nest does not fit inside "
            f"the {nx}x{ny} root with its {clear}-cell boundary clearance")
    return {"ratio": ratio, "parent_cells": parent_cells, "child_n": child_n,
            "clear": clear, "root_nx": nx, "root_ny": ny,
            "root_dt": float(root.get("time_step", 0) or 0),
            "root_dx_km": root_dx_km}


def _placement(shape: dict, i_center: float, j_center: float) -> tuple[int, int]:
    half = shape["parent_cells"] / 2.0
    lo = shape["clear"]
    hi_i = shape["root_nx"] - shape["clear"] - shape["parent_cells"]
    hi_j = shape["root_ny"] - shape["clear"] - shape["parent_cells"]
    i = int(round(i_center - half))
    j = int(round(j_center - half))
    return max(lo, min(hi_i, i)), max(lo, min(hi_j, j))


def _follow(options: dict, shape: dict) -> dict:
    from gpuwm.core.nest_relocation import max_parent_cells_for_overlap

    overlap = 0.7
    bound = max_parent_cells_for_overlap(
        overlap, parent_grid_ratio=shape["ratio"], child_nx=shape["child_n"],
        child_ny=shape["child_n"])
    max_move = max(1, min(8, 8 if bound is None else int(bound)))
    return {
        "field": options["detector"],
        "threshold": float(options["follow_threshold"]),
        "fallback_threshold": float(options["fallback_threshold"]),
        "search_margin_cells": 12,
        "min_shift_cells": 2,
        "max_shift_cells": max_move,
        "cooldown_seconds": float(options["follow_cadence_s"]),
        "cadence_seconds": float(options["follow_cadence_s"]),
        "max_move_parent_cells": max_move,
        "min_overlap_fraction": overlap,
    }


def _retire(options: dict) -> dict:
    return {"trigger": options["detector"],
            "threshold": float(options["retire_threshold"]),
            "sustained_s": float(options["retire_sustained_s"]),
            "min_lifetime_s": float(options["min_lifetime_s"]),
            "merge_overlap": float(options["merge_overlap"])}


def _on_lattice(seconds: float, step: float) -> float:
    if step <= 0:
        return float(seconds)
    return float(math.ceil(round(seconds / step, 9)) * step)


def _slot_domain(grid_id: int, shape: dict, placement, options: dict, *, spawn: dict,
                 rearm: dict | None) -> dict:
    domain = {
        "grid_id": int(grid_id), "parent_id": 1,
        "i_parent_start": int(placement[0]), "j_parent_start": int(placement[1]),
        "parent_grid_ratio": shape["ratio"], "parent_time_step_ratio": shape["ratio"],
        "nx": shape["child_n"], "ny": shape["child_n"],
        "specified": False, "nested": True,
        "history_interval_s": float(options["history_interval_s"]),
        "radt": float(options["radt"]), "cu_physics": 0,
        # A slot's ground is its whole-parent corridor, whose steepest slope
        # (the Rockies at 1 km: 35 degrees on the HRRR grid) needs more
        # acoustic off-centering than the parent's epssm; "auto" takes the
        # measured floor for that slope (gpuwm.acoustic_adaptation).
        "epssm": "auto",
        # The nest rebuild at a spawn or a move does not carry the CLM lake
        # column (water, ice, snow, sediment heat), which
        # gpuwm.experiment._refuse_rebuilt_nest_lake refuses; storm nests run
        # without the lake model and say so in the receipt.
        "sf_lake_physics": 0,
        # A child's statics carry no sub-grid orographic statistics, which
        # the GSL drag suite reads (gpuwm.experiment._refuse_child_terrain_drag);
        # at 1 km the resolved terrain carries the drag.
        "gwd_opt": 0,
        "spawn": spawn, "follow": _follow(options, shape), "retire": _retire(options),
    }
    if rearm is not None:
        domain["rearm"] = rearm
    return domain


def author_slots(raw: dict, options: dict, *, slots: int | None = None) -> list[dict]:
    """The dormant ``[[domain]]`` slots the table asks for, in grid order."""
    shape = slot_shape(raw, options)
    run_seconds = float(raw.get("experiment", {}).get("run_seconds", 0.0) or 0.0)
    if run_seconds <= 0:
        raise StormNestsRefusal(f"[{TABLE}] needs [experiment] run_seconds")
    count = int(options["slots"] if slots is None else slots)
    count = min(count, int(options["max_slots"]))
    domains: list[dict] = []
    grid_id = 2
    if options["points"]:
        grid = _root_grid(raw, _root(raw))
        start = _start_time(raw)
    for point in options["points"]:
        x, y = grid.latlon_to_ij(float(point["lat"]), float(point["lon"]))
        placement = _placement(shape, float(x), float(y))
        if "at_s" in point:
            at_s = float(point["at_s"])
        elif "at" in point:
            when = point["at"]
            when = (when if isinstance(when, datetime)
                    else datetime.fromisoformat(str(when).replace("Z", "")))
            at_s = (when.replace(tzinfo=None) - start).total_seconds()
        else:
            at_s = 0.0
        # A spawn is a leg-boundary operation: at least one root step in,
        # on the root-step lattice.
        at_s = _on_lattice(max(at_s, shape["root_dt"] or 1.0), shape["root_dt"])
        if not 0.0 < at_s < run_seconds:
            raise StormNestsRefusal(
                f"[{TABLE}] point {point} spawns at {at_s:g} s, outside the "
                f"{run_seconds:g} s run")
        domains.append(_slot_domain(
            grid_id, shape, placement, options,
            spawn={"trigger": "time", "at_s": at_s}, rearm=None))
        grid_id += 1
    centre = (shape["root_nx"] / 2.0, shape["root_ny"] / 2.0)
    box = [shape["clear"] + 1, shape["clear"] + 1,
           shape["root_nx"] - shape["clear"], shape["root_ny"] - shape["clear"]]
    earliest = float(options.get("earliest_s", shape["root_dt"] or 0.0))
    latest = float(options.get("latest_s", run_seconds))
    cadence = _on_lattice(float(options["cadence_s"]), shape["root_dt"])
    for _ in range(count):
        domains.append(_slot_domain(
            grid_id, shape, _placement(shape, *centre), options,
            spawn={"trigger": options["detector"],
                   "threshold": float(options["spawn_threshold"]),
                   "earliest_s": earliest, "latest_s": latest,
                   "search_box": box, "cadence_s": cadence},
            rearm={"max_firings": int(options["max_firings"]),
                   "cooldown_s": float(options["rearm_cooldown_s"])}))
        grid_id += 1
    return domains


def slot_bytes(raw: dict, options: dict) -> int:
    """Device bytes one slot reserves, from the planner's own estimate.

    The difference between the planner's estimate of the tree with one
    slot and without any: the cost a slot's reservation adds.
    """
    import copy

    from gpuwm.core.preflight import estimate_experiment
    from gpuwm.domain_wizard import _render_table, experiment_from_text

    def priced(n):
        doc = copy.deepcopy(raw)
        doc.pop(TABLE, None)
        root = _root(doc)
        _align_parent_history(root, options)
        doc["domain"] = [root] + author_slots(doc, {**options, "points": []}, slots=n)
        text = _emit(doc, _render_table)
        return estimate_experiment(experiment_from_text(text, source="storm-nests"))

    # Two slots against one: a lone root prices on the single-domain basis
    # (no shared arena), so the first slot's difference is not a slot's cost.
    return max(0, int(priced(2).alloc_estimate_bytes)
               - int(priced(1).alloc_estimate_bytes))


def _align_parent_history(root: dict, options: dict):
    """Lower the parent's history interval to the follow cadence if needed."""
    parent_history = float(root.get("history_interval_s", 0.0) or 0.0)
    follow_cadence = float(options["follow_cadence_s"])
    if parent_history > 0.0:
        ratio = follow_cadence / parent_history
        if abs(ratio - round(ratio)) <= 1e-9 and round(ratio) >= 1:
            return None
    root["history_interval_s"] = follow_cadence
    return {"from_s": parent_history, "to_s": follow_cadence}


def _emit(doc: dict, render_table=None) -> str:
    """The canonical TOML of ``doc`` (inline lifecycle tables included)."""
    del render_table
    from gpuwm.toml_document import emit_experiment_toml
    return emit_experiment_toml(doc)


def priced_slot_count(raw: dict, options: dict, *, free_bytes: int,
                      base_bytes: int) -> int:
    """How many automatic slots fit what the card has left after the parent."""
    per = slot_bytes(raw, options)
    if per <= 0:
        return int(options["max_slots"])
    spare = int(free_bytes) - int(base_bytes)
    return max(0, min(int(options["max_slots"]), spare // per))


def expand(raw: dict, *, slots_override=None, points_override=None,
           free_bytes: int | None = None) -> tuple[dict, dict]:
    """``raw`` with ``[storm_nests]`` replaced by its dormant slots, plus a receipt."""
    import copy

    table = dict(raw.get(TABLE) or {})
    if slots_override is not None:
        table["slots"] = slots_override
    if points_override:
        table["points"] = list(table.get("points") or []) + list(points_override)
    options = resolve_options(table)
    doc = copy.deepcopy(raw)
    doc.pop(TABLE, None)
    root = _root(doc)
    count = options["slots"]
    pricing = None
    if count == "auto":
        budget = (int(float(options["vram_gib"]) * 2**30) if "vram_gib" in options
                  else free_bytes)
        if budget is None:
            raise StormNestsRefusal(
                f"[{TABLE}] slots = \"auto\" prices slots from the card's free "
                "memory, and none was measured or given; set vram_gib or a number")
        from gpuwm.core.preflight import estimate_experiment
        from gpuwm.domain_wizard import _render_table, experiment_from_text
        bare = copy.deepcopy(doc)
        bare["domain"] = [copy.deepcopy(root)]
        base = estimate_experiment(experiment_from_text(_emit(bare, _render_table),
                                                        source="storm-nests"))
        base_bytes = int(base.alloc_estimate_bytes)
        per = slot_bytes(doc, options)
        # A parent too big to sit resident streams through tiles, and the
        # planner decides its working set later; the slots are then priced
        # against a quarter of the card, and the receipt says so.
        streams = base_bytes >= int(budget)
        spare = int(budget) // 4 if streams else int(budget) - base_bytes
        count = max(0, min(int(options["max_slots"]), spare // max(1, per)))
        pricing = {"budget_bytes": int(budget), "parent_bytes": base_bytes,
                   "slot_bytes": per, "slots": count,
                   "basis": ("parent streams: slots priced against a quarter of the card"
                             if streams else "card budget minus the resident parent")}
    slots = author_slots(doc, options, slots=int(count))
    # THE BREAKAGE THIS CLOSES: this line was `[root] + slots`, so a
    # configuration with its own fixed nests lost them silently whenever the
    # expansion ran (`gpuwm go CONFIG --storm-nests off / N / --nest-at`, or a
    # [storm_nests] table): a 3-domain tree was re-stated as 1 domain.  The
    # fixed nests are kept; slots beside them are refused by name, because
    # slots are numbered from grid 2 and lifecycle slots under a fixed tree
    # are not a shape this door authors.
    domains = doc.get("domain")
    domains = [domains] if isinstance(domains, dict) else list(domains or [])
    fixed = [d for d in domains if d is not root]
    if slots and fixed:
        raise StormNestsRefusal(
            f"[{TABLE}] would add {len(slots)} storm-nest slot(s) beside this "
            "configuration's own fixed nest(s) (grid "
            + ", ".join(str(int(d.get("grid_id", 0))) for d in fixed)
            + "); slots are numbered from grid 2 and are authored only under "
            "a single root domain. Run with --storm-nests off to keep the "
            "fixed nests, or drop them to use storm nests")
    doc["domain"] = [root] + fixed + slots
    # The follower's echo fallback reads the parent's composite reflectivity,
    # which the microphysics stashes at the parent's HISTORY cadence
    # (gpuwm.experiment._refuse_unservable_follow_cadence), so the parent
    # writes history at least as often as a nest looks.
    history_changed = _align_parent_history(root, options) if slots else None
    # A dormant slot can fire anywhere, so its statics corridor spans the
    # whole parent, and the 30 m terrain overlay over that span is a
    # continental 1-arc-second mosaic (~80 GB, most of an hour on one core
    # for CONUS; a rented box, 2026-10-21).  The nests take the WPS 30-arc-second
    # terrain instead, about their own 1 km spacing, unless the
    # configuration declares its own [static.highres].
    terrain = None
    static = doc.get("static")
    if slots and (static is None or (isinstance(static, dict) and "highres" not in static)):
        from gpuwm.static.highres_production import default_highres_cache_root
        doc["static"] = {**(static or {}), "highres": {
            "enabled": False, "cache_root": str(default_highres_cache_root())}}
        terrain = "WPS 30-arc-second (gmted2010_30s); [static.highres] off for the whole-parent corridor"
    receipt = {"schema": SCHEMA, "options": {k: v for k, v in options.items() if k != "points"},
               "points": options["points"], "automatic_slots": int(count),
               "slot_grid_ids": [int(d["grid_id"]) for d in slots],
               "shape": slot_shape(raw, options), "pricing": pricing,
               "parent_history_interval": history_changed,
               "nest_terrain": terrain}
    return doc, receipt


def parse_point(text: str) -> dict:
    """``LAT,LON`` or ``LAT,LON@TIME`` (ISO time or seconds into the run)."""
    where, _, when = str(text).partition("@")
    try:
        lat, lon = (float(v) for v in where.split(","))
    except ValueError:
        raise StormNestsRefusal(
            f"--nest-at takes LAT,LON or LAT,LON@TIME, got {text!r}") from None
    point = {"lat": lat, "lon": lon}
    when = when.strip()
    if when:
        try:
            point["at_s"] = float(when)
        except ValueError:
            point["at"] = when
    return point


def expand_config_file(config: Path, out: Path, *, slots_override=None,
                       points_override=None, free_bytes=None) -> tuple[Path, dict]:
    """Write ``config`` expanded to ``out`` (the copy ``gpuwm go`` runs)."""
    import json
    import tomllib

    from gpuwm.toml_document import emit_experiment_toml

    raw = tomllib.loads(Path(config).read_text(encoding="utf-8-sig"))
    doc, receipt = expand(raw, slots_override=slots_override,
                          points_override=points_override, free_bytes=free_bytes)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(emit_experiment_toml(doc), encoding="utf-8")
    receipt["source_config"] = str(Path(config).resolve())
    receipt["expanded_config"] = str(out.resolve())
    (out.parent / "storm-nests.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out, receipt


__all__ = ["DEFAULTS", "KEYS", "StormNestsRefusal", "author_slots", "expand",
           "expand_config_file", "parse_point", "priced_slot_count", "resolve_options",
           "slot_bytes", "slot_shape"]


from gpuwm.nest_lives import nest_lives, write_nest_lives  # noqa: E402,F401  (re-export)
