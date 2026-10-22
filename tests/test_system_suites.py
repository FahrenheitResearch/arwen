"""The system-suite tables: rows that load, hold to the registry, and run nothing by default."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import system_suites as ss
from gpuwm.physics_registry import (
    REGISTRY_BLOCKS_OUTSIDE_PHYSICS_IDENTITY, physics_registry,
    registry_physics_parts, registry_physics_sha256)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ss.load_tables()


def test_every_table_holds_to_the_registry():
    registry = physics_registry()
    assert TABLES, "no system-suite table is installed"
    for doc in TABLES.values():
        assert ss.table_problems(doc, registry) == []


def test_the_registry_block_equals_the_tables():
    # A merge that keeps another builder's registry line drops the block
    # silently; this is the test that notices (tools/apply_system_suites.py).
    assert physics_registry()[ss.REGISTRY_BLOCK] == json.loads(
        json.dumps(ss.registry_rows(TABLES)))


def test_the_block_is_outside_every_physics_identity():
    assert ss.REGISTRY_BLOCK in REGISTRY_BLOCKS_OUTSIDE_PHYSICS_IDENTITY
    without = physics_registry()
    without.pop(ss.REGISTRY_BLOCK)
    assert registry_physics_sha256(without) == registry_physics_sha256()
    assert registry_physics_parts(without) == registry_physics_parts()


@pytest.mark.parametrize("system,profile", [
    (system, profile) for system, doc in TABLES.items()
    for profile in doc.get("profiles") or ()])
def test_every_profile_carries_its_suite_exactly(system, profile):
    doc = TABLES[system]
    experiment = tomllib.loads((ROOT / profile["config"]).read_text("utf-8"))
    assert ss.profile_mismatches(doc, profile, experiment) == []


def test_a_profile_that_drifts_from_its_suite_is_named():
    doc = TABLES["rrfs-v1"]
    profile = doc["profiles"][0]
    experiment = tomllib.loads((ROOT / profile["config"]).read_text("utf-8"))
    experiment["shared"]["cu_physics"] = 3
    experiment["domain"][0]["cu_physics"] = 3
    assert any("cu_physics" in p
               for p in ss.profile_mismatches(doc, profile, experiment))


def test_missing_schemes_are_declared_and_never_substituted():
    doc = TABLES["rrfs-v1"]
    det = ss.resolved_suite(doc, "rrfs-sas-deterministic")
    packages = {e["part"]: e["package"] for e in det["declared_missing"]}
    assert packages["deep-convection"] == "C1"
    assert "C9" in packages["smoke-and-dust"]
    assert packages["ozone-and-water-vapour-photochemistry"] == "C4"
    # The missing deep scheme runs as no scheme at all.
    assert det["components"]["cumulus"]["option"] == "off"
    assert det["settings"]["cu_physics"] == 0
    refused = {e["setting"]: e["package"] for e in det["declared_refused"]}
    assert refused["bl_mynn_tkeadvect"] == "C2"
    assert det["settings"]["bl_mynn_tkeadvect"] is False


def test_a_null_component_cannot_be_runnable():
    registry = physics_registry()
    doc = copy.deepcopy(TABLES["rrfs-v1"])
    doc["suites"]["rrfs-ensemble-member-1"]["runnable"] = True
    problems = ss.suite_problems(doc, "rrfs-ensemble-member-1", registry)
    assert any("cannot be runnable" in p for p in problems)


def test_a_quiet_substitution_is_refused():
    registry = physics_registry()
    doc = copy.deepcopy(TABLES["rrfs-v1"])
    row = doc["suites"]["rrfs-sas-deterministic"]
    row["components"]["cumulus"] = {"option": "grell-freitas",
                                    "source": "samfdeepcnv",
                                    "relation": "in kind"}
    problems = ss.suite_problems(doc, "rrfs-sas-deterministic", registry)
    assert any("cu_physics" in p for p in problems)


def test_the_cycle_table():
    doc = TABLES["rrfs-v1"]
    assert [ss.forecast_length_hours(doc, "deterministic", h)
            for h in (0, 6, 12, 18, 1, 13, 23)] == [84, 84, 84, 84, 18, 18, 18]
    assert ss.forecast_length_hours(doc, "ensemble", 0) == 60
    assert ss.forecast_length_hours(doc, "ensemble", 3) == 0
    assert ss.forecast_length_hours(doc, "fire-weather-nest", 6) == 36


def test_no_other_configuration_names_a_system_suite():
    # Default-off: the rows reach a run only through a profile that carries
    # them.  No registry template, route or other shipped configuration
    # refers to a suite id.
    ids = {suite for doc in TABLES.values() for suite in doc["suites"]}
    registry = physics_registry()
    outside = {k: v for k, v in registry.items() if k != ss.REGISTRY_BLOCK}
    text = json.dumps(outside)
    assert not [i for i in ids if i in text]
    profiles = {p["config"] for doc in TABLES.values() for p in doc["profiles"]}
    for path in (ROOT / "configs").rglob("*.toml"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in profiles:
            continue
        body = path.read_text("utf-8", errors="replace")
        assert not [i for i in ids if i in body], rel


# RRFS v1's own settings, suite by suite, transcribed from each suite's own
# namelist at rrfs-workflow e4a100b (det/, enkf/, firewx/ and
# ensf/input.nml_restart_stoch_ensphyN); an unset key is the CCPP default.
# Kept apart from the table on purpose: the table states these values with
# line numbers, and this test holds it to a second transcription.
_RRFS_V1_SOURCE = {
    #                             rrfs_sd mix_chem tkeadv h2o   hail  light  iovr iaer  isncovr spp
    "rrfs-sas-deterministic":     (True,  True,  True,  True, True,  True,  3, 1011, 3, False),
    "rrfs-sas-enkf-members":      (False, True,  True,  True, True,  True,  3, 1011, 3, False),
    "rrfs-sas-nogwd-fire-weather": (False, False, True, True, False, True,  3, 1011, 3, False),
    "rrfs-ensemble-member-1":     (False, False, False, True, True,  True,  3, 1011, 3, True),
    "rrfs-ensemble-member-2":     (False, False, True,  True, True,  True,  3, 1011, 3, True),
    "rrfs-ensemble-member-3":     (False, False, True,  True, True,  True,  3, 1011, 3, True),
    "rrfs-ensemble-member-4":     (False, False, False, True, True,  True,  3, 1011, 3, True),
    "rrfs-ensemble-member-5":     (False, False, True,  True, True,  True,  3, 1011, 3, True),
}
_SOURCE_KEYS = ("rrfs_sd", "mix_chem", "bl_mynn_tkeadvect", "h2o_phys",
                "do_hailcast", "lightning_threat", "iovr", "iaer",
                "isncovr_opt", "do_spp")
_C4_C10 = {"ozone-and-water-vapour-photochemistry", "hail-diagnostic",
           "lightning-threat", "hourly-maxima"}
_RUNNABLE_DECLARATIONS = {
    "rrfs-sas-deterministic": (
        {"deep-convection", "smoke-and-dust"} | _C4_C10,
        {"bl_mynn_tkeadvect", "mix_chem", "isncovr_opt", "iovr", "iaer"}),
    "rrfs-sas-enkf-members": (
        {"deep-convection"} | _C4_C10,
        {"bl_mynn_tkeadvect", "mix_chem", "isncovr_opt", "iovr", "iaer"}),
    "rrfs-sas-nogwd-fire-weather": (
        {"deep-convection", "ozone-and-water-vapour-photochemistry",
         "lightning-threat", "hourly-maxima"},
        {"bl_mynn_tkeadvect", "isncovr_opt", "iovr", "iaer"}),
    "rrfs-ensemble-member-2": (
        {"deep-convection", "stochastic-physics"} | _C4_C10,
        {"bl_mynn_tkeadvect", "isncovr_opt", "iovr", "iaer"}),
}


def test_every_suite_states_its_own_source_settings():
    doc = TABLES["rrfs-v1"]
    defaults = doc["source_rules"]["defaults"]
    assert set(doc["suites"]) == set(_RRFS_V1_SOURCE)
    for suite_id, expected in _RRFS_V1_SOURCE.items():
        row = ss.resolved_suite(doc, suite_id)
        got = tuple(ss._effective_source_value(row, defaults, key)
                    for key in _SOURCE_KEYS)
        assert got == expected, suite_id


def test_every_suite_declares_what_its_own_source_runs():
    doc = TABLES["rrfs-v1"]
    for suite_id in doc["suites"]:
        assert ss.source_declaration_problems(doc, suite_id) == [], suite_id


def test_the_runnable_suites_declare_exactly_their_absences():
    doc = TABLES["rrfs-v1"]
    runnable = {s for s in doc["suites"]
                if ss.resolved_suite(doc, s)["runnable"]}
    assert runnable == set(_RUNNABLE_DECLARATIONS)
    for suite_id, (parts, settings) in _RUNNABLE_DECLARATIONS.items():
        row = ss.resolved_suite(doc, suite_id)
        assert {e["part"] for e in row["declared_missing"]} == parts, suite_id
        assert {e["setting"] for e in row["declared_refused"]} == settings, suite_id


def test_a_silent_absence_on_a_derived_row_is_named():
    doc = copy.deepcopy(TABLES["rrfs-v1"])
    doc["suites"]["rrfs-ensemble-member-2"]["declared_missing_remove"].append(
        "lightning-threat")
    problems = ss.source_declaration_problems(doc, "rrfs-ensemble-member-2")
    assert any("lightning-threat" in p and "not declared missing" in p
               for p in problems)


def test_a_declared_value_the_source_does_not_set_is_named():
    doc = copy.deepcopy(TABLES["rrfs-v1"])
    doc["suites"]["rrfs-sas-nogwd-fire-weather"].pop("declared_refused_remove")
    problems = ss.source_declaration_problems(doc, "rrfs-sas-nogwd-fire-weather")
    assert any("mix_chem" in p and "refused" in p for p in problems)


def test_a_derived_row_never_replaces_its_parents_declarations():
    doc = copy.deepcopy(TABLES["rrfs-v1"])
    doc["suites"]["rrfs-ensemble-member-2"]["declared_missing"] = []
    with pytest.raises(ss.SystemSuiteError, match="wholesale"):
        ss.resolved_suite(doc, "rrfs-ensemble-member-2")


def test_a_profile_whose_start_is_not_its_arms_is_named():
    doc = copy.deepcopy(TABLES["rrfs-v1"])
    profile = next(p for p in doc["profiles"] if p["arm"] == "GS")
    experiment = tomllib.loads((ROOT / profile["config"]).read_text("utf-8"))
    relabelled = dict(profile, arm="G")
    problems = ss.profile_mismatches(doc, relabelled, experiment)
    assert any("starts from 'gfs'" in p and "'rrfs'" in p for p in problems)
    arm_r = next(p for p in doc["profiles"] if p["arm"] == "R")
    experiment = tomllib.loads((ROOT / arm_r["config"]).read_text("utf-8"))
    problems = ss.profile_mismatches(doc, dict(arm_r, arm="GS"), experiment)
    assert any("lead f000" in p and "6 h older" in p for p in problems)
    assert any("boundary source 'gfs'" in p for p in problems)


def test_an_inheritance_cycle_is_reported_without_recursing():
    doc = {"system": "synthetic", "suites": {
        "a": {"derives_from": "b"}, "b": {"derives_from": "a"}}}
    with pytest.raises(ss.SystemSuiteError, match="inheritance cycle a -> b -> a"):
        ss.resolved_suite(doc, "a")
