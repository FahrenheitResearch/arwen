"""Every key a config table accepts has a declared type a front end can read.

A config key reaches a front end's form only if the form can learn what it
takes.  For most keys that is the typed field the loader stores the value
in: a ``RunConfig`` field, or a field of the dataclass the table builds.
Some keys were accepted through a name-only set and typed only inside the
parsing code -- a front end could send them but not learn their type or
default, and a hosted contract generated from the engine listed them as
untyped.  Each of those now has one declared row (``gpuwm.config_keys``)
beside the table that owns it, the owning loader checks values against the
row, and ``gpuwm.config.declared_key_rows`` exports every row.
"""
from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import json
import pathlib
import sys
import types
import typing

import pytest

import gpuwm.config as config_module
from gpuwm import case_data as C
from gpuwm import experiment as E
from gpuwm import fetch as F
from gpuwm.config import RunConfig
from gpuwm.core import (nest_lifecycle, nest_spawn, storm_track_writer,
                        storm_tracking, streaming)
from gpuwm.ingest import soil_downscale as S
from gpuwm.io import history_selection as H


def _rows() -> dict[str, dict[str, dict]]:
    export = getattr(config_module, "declared_key_rows", None)
    return {} if export is None else export()


def _toml_type(annotation) -> str | None:
    """The TOML type a field annotation names, or None when it names none."""

    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        kinds = {_toml_type(arg) for arg in typing.get_args(annotation)
                 if arg is not type(None)}
        return None if None in kinds or not kinds else "|".join(sorted(kinds))
    if origin is typing.Literal:
        kinds = {type(arg) for arg in typing.get_args(annotation)}
        return _toml_type(kinds.pop()) if len(kinds) == 1 else None
    if origin in (list, tuple, set, frozenset, collections.abc.Sequence):
        return "array"
    if origin in (dict, collections.abc.Mapping):
        return "table"
    simple = {bool: "boolean", int: "integer", float: "number", str: "string",
              datetime.datetime: "datetime", pathlib.Path: "string",
              list: "array", tuple: "array", dict: "table"}
    if annotation in simple:
        return simple[annotation]
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return "table"
    return None


def _field_types(*classes) -> dict[str, str | None]:
    namespace: dict = {}
    for module in (storm_track_writer, storm_tracking, nest_lifecycle,
                   nest_spawn, streaming, H, E, C):
        namespace.update(vars(module))
    found: dict[str, str | None] = {}
    for cls in classes:
        hints = typing.get_type_hints(
            cls, globalns={**namespace, **vars(sys.modules[cls.__module__])})
        for field in dataclasses.fields(cls):
            found.setdefault(field.name, _toml_type(hints.get(field.name)))
    return found


#: Every loader key table a config document reaches, the dataclasses its
#: values land in, and the keys that are themselves walked sub-tables.
def _loader_tables():
    run_fields = {field.name for field in dataclasses.fields(RunConfig)}
    shared_extras = ({"e_vert", "eta_levels", "p_top"}
                     | set(E._GUARD_DEFAULTS)) - run_fields
    return {
        "experiment": (E._EXPERIMENT_KEYS, (E.ExperimentConfig,), {}),
        # Every RunConfig field is a [shared] key typed by RunConfig; the
        # extras beside them are what this table adds.
        "shared": (shared_extras, (E.VerticalConfig,), {}),
        "projection": (E._PROJECTION_KEYS, (E.ProjectionConfig,), {}),
        "perturbation": (E._PERTURBATION_KEYS, (),
                         {"bubbles": "perturbation.bubbles"}),
        "perturbation.bubbles": (E._BUBBLE_KEYS, (E.BubbleConfig,), {}),
        "domain": (E._DOMAIN_KEYS, (E.DomainConfig, RunConfig),
                   {key: f"domain.{key}" for key in
                    ("spawn", "retire", "rearm", "follow", "tiles",
                     "output")}),
        "relocation": (E._RELOCATION_KEYS, (E.RelocationConfig,),
                       {key: f"relocation.{key}" for key in
                        ("follow", "move", "containment", "track")}),
        "relocation.containment": (E._RELOCATION_CONTAINMENT_KEYS,
                                   (E.ContainmentConfig,), {}),
        "relocation.move": (E._RELOCATION_MOVE_KEYS,
                            (E.ScheduledRelocationMove,), {}),
        "relocation.follow": (storm_tracking.FOLLOW_KEYS,
                              (storm_tracking.FollowConfig,), {}),
        "relocation.track": (storm_track_writer.TRACK_KEYS,
                             (storm_track_writer.TrackConfig,), {}),
        "domain.spawn": (nest_spawn.SPAWN_KEYS, (nest_spawn.SpawnConfig,),
                         {}),
        "domain.retire": (nest_lifecycle.RETIRE_KEYS,
                          (nest_lifecycle.RetireConfig,), {}),
        "domain.rearm": (nest_lifecycle.REARM_KEYS,
                         (nest_lifecycle.RearmConfig,), {}),
        "domain.follow": (storm_tracking.FOLLOW_KEYS
                          | nest_lifecycle.DOMAIN_FOLLOW_EXTRA_KEYS,
                          (storm_tracking.FollowConfig,
                           nest_lifecycle.DomainFollowConfig), {}),
        "domain.tiles": (streaming.STREAMING_KEYS,
                         (streaming.StreamingOptions,), {}),
        "domain.output": (H.OUTPUT_KEYS, (H.HistorySelection,), {}),
        "tiles": (streaming.STREAMING_KEYS, (streaming.StreamingOptions,),
                  {}),
        "output": (H.OUTPUT_KEYS, (H.HistorySelection,), {}),
        "case_data": (C._KNOWN_KEYS, (C.CaseDataConfig,), {}),
        "fetch": (F.FETCH_HINT_KEYS, (), {}),
        "ingest": (S.INGEST_TABLE_KEYS, (), {}),
    }


def test_every_loader_key_table_declares_each_keys_type_default_and_doc():
    """Walk every table; a key with no declaration is listed by name.

    A key is declared when the table's declared rows carry it (type,
    default and one-line doc), when its value lands in a typed field of
    the dataclass the table builds, or when it is itself a sub-table
    walked here.
    """

    rows = _rows()
    tables = _loader_tables()
    undeclared = []
    for table, (keys, classes, subtables) in tables.items():
        typed = _field_types(*classes) if classes else {}
        declared = rows.get(table, {})
        for key in sorted(keys):
            if key in declared:
                row = declared[key]
                assert row["type"] and row["doc"].strip(), (table, key)
                assert "default" in row, (table, key)
                continue
            if key in subtables:
                assert subtables[key] in tables, (table, key)
                continue
            if typed.get(key):
                continue
            undeclared.append(f"{table}.{key}")
    assert undeclared == [], (
        "keys a loader accepts with no declared type: "
        + ", ".join(undeclared))


#: The keys the hosted contract found typed only in parsing code.
FRONT_DOOR_AUDIT_KEYS = (
    ("experiment", "physics_mode"), ("experiment", "patchset"),
    ("experiment", "patches"), ("shared", "e_vert"),
    ("domain", "start_time"), ("relocation", "track"),
    ("case_data", "water_temperature_overlay"),
    ("ingest", "soil_texture_downscale"),
    ("fetch", "member"), ("fetch", "radius_km"), ("fetch", "retrieve"),
    ("fetch", "source_root"),
)


@pytest.mark.parametrize("table,key", FRONT_DOOR_AUDIT_KEYS)
def test_each_audited_key_has_one_declared_row(table, key):
    rows = _rows()
    assert key in rows.get(table, {}), f"[{table}] {key} has no declared row"
    row = rows[table][key]
    assert row["type"]
    assert "default" in row
    assert row["doc"].strip() and "\n" not in row["doc"]


def test_the_rows_are_exported_as_json_beside_the_run_config_fields():
    rows = _rows()
    assert rows, "gpuwm.config.declared_key_rows exports nothing"
    assert json.loads(json.dumps(rows)) == rows
    # Every row names a key its table accepts: a row for a key no loader
    # reads would advertise a setting that does nothing.
    tables = _loader_tables()
    for table, declared in rows.items():
        keys = tables[table][0]
        assert set(declared) <= set(keys), (table, set(declared) - set(keys))


def _experiment_text(*, experiment="", shared="", domain="", tail=""):
    return (
        "[experiment]\n"
        'name = "rows"\n'
        "start_time = 2024-05-03T12:00:00\n"
        "run_seconds = 3600.0\n"
        "restart_interval_s = 0.0\n"
        f"{experiment}\n"
        "[shared]\n"
        "nz = 30\nztop = 20000.0\n"
        f"{shared}\n"
        "[[domain]]\n"
        "grid_id = 1\nparent_id = 0\ni_parent_start = 1\nj_parent_start = 1\n"
        "parent_grid_ratio = 1\nparent_time_step_ratio = 1\n"
        "history_interval_s = 3600.0\n"
        "nx = 40\nny = 40\ndx = 12000.0\ndy = 12000.0\ntime_step = 60\n"
        f"{domain}\n"
        f"{tail}")


@pytest.mark.parametrize("fragment,where,phrase", [
    ({"experiment": 'physics_mode = "arwen-patched"\npatches = "L3"'},
     "patches", "divergence-ledger entry ids"),
    ({"shared": 'e_vert = "31"'}, "e_vert", "full-level count"),
    ({"domain": "e_we = 41.0"}, "e_we", "staggered west-east"),
])
def test_the_loader_refuses_a_wrong_type_in_the_rows_own_words(
        fragment, where, phrase):
    """The loader's type check IS the row: its refusal carries the row's
    declared type and doc, so the two cannot say different things."""

    import tomllib

    raw = tomllib.loads(_experiment_text(**fragment))
    with pytest.raises(ValueError) as refusal:
        E.build_experiment(raw, source="rows.toml")
    text = str(refusal.value)
    assert where in text and phrase in text, text


@pytest.mark.parametrize("spelling", [
    "input_from_hires = 0", "interp_method_type = 2.0",
    "nest_interp_coord = 0.0", "vert_refine_method = false"])
def test_a_spelling_of_the_one_implemented_guard_value_runs(spelling):
    """The nest guard keys are compared with the value that runs, as they
    were before their rows existed.  ``input_from_hires = 0`` IS the
    implemented value, so a type refusal of it would name no breakage."""

    import tomllib

    raw = tomllib.loads(_experiment_text(shared=spelling))
    E.build_experiment(raw, source="rows.toml")


def test_a_guard_value_that_cannot_run_is_refused_with_its_reason():
    import tomllib

    raw = tomllib.loads(_experiment_text(shared="input_from_hires = true"))
    with pytest.raises(ValueError, match="not implemented"):
        E.build_experiment(raw, source="rows.toml")
    raw = tomllib.loads(_experiment_text(shared="interp_method_type = 1"))
    with pytest.raises(ValueError, match="only SINT"):
        E.build_experiment(raw, source="rows.toml")


def test_a_quoted_number_on_a_number_row_reads_as_the_number():
    """``radius_km`` was read through float() before its row existed, so a
    quoted radius cropped; the row keeps reading it, and refuses only a
    value that is not one finite number, in the row's own words."""

    row = F.FETCH_HINT_ROWS["radius_km"]
    assert row.check("250", where="x") == 250.0
    F.validate_fetch_hints({"source": "gfs", "radius_km": "250",
                            "point": "35,-97"}, source="rows.toml")
    for bad in ("250 km", "nan", "wide"):
        with pytest.raises(ValueError) as refusal:
            F.validate_fetch_hints({"source": "gfs", "radius_km": bad,
                                    "point": "35,-97"}, source="rows.toml")
        assert "radius_km" in str(refusal.value)
        assert "crop radius" in str(refusal.value)
    # A row that is not a number gains nothing: an integer row still
    # refuses a quoted integer.
    with pytest.raises(ValueError):
        E._DOMAIN_KEY_ROWS["e_we"].check("41", where="x")


def test_the_fetch_and_ingest_loaders_read_their_rows():
    with pytest.raises(ValueError) as refusal:
        F.validate_fetch_hints({"source": "gfs", "radius_km": True,
                                "point": "35,-97"}, source="rows.toml")
    assert "radius_km" in str(refusal.value)
    assert "crop radius" in str(refusal.value)
    with pytest.raises(ValueError, match="must be true or false"):
        S.parse_ingest_table({"soil_texture_downscale": "off"},
                             source="rows.toml")
    # Silence is the row's default, read from the row.
    assert S.declared_soil_texture_downscale({}) is (
        S.INGEST_TABLE_ROWS["soil_texture_downscale"].default)
    assert set(F.FETCH_HINT_KEYS) == set(F.FETCH_HINT_ROWS)
    assert tuple(S.INGEST_TABLE_KEYS) == tuple(S.INGEST_TABLE_ROWS)


def test_a_nest_start_time_absent_inherits_and_present_is_checked():
    import tomllib

    base = _experiment_text()
    exp = E.build_experiment(tomllib.loads(base), source="rows.toml")
    assert exp.domain_start_time(1) == exp.start_time
    raw = tomllib.loads(_experiment_text(domain='start_time = "noon"'))
    with pytest.raises(ValueError) as refusal:
        E.build_experiment(raw, source="rows.toml")
    assert "start_time" in str(refusal.value)
    assert "date-time" in str(refusal.value)


def test_a_key_row_refuses_a_default_of_the_wrong_type():
    from gpuwm.config_keys import KeyRow

    with pytest.raises(ValueError):
        KeyRow("x", "integer", "seven", "a count")
    with pytest.raises(ValueError):
        KeyRow("x", "whole", None, "not a TOML type")
    with pytest.raises(ValueError):
        KeyRow("x", "string", None, "")
