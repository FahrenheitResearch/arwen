"""A declared companion extends the primary ladder above its top.

The capability is generic composition metadata: a binding row's
``companion`` declaration lets a run of the same grid and cycle stack its
levels above the primary's top onto the primary's own columns.  These
tests hold the join and the grammar on synthetic collections; the packaged
rrfs-ens profile is held in tests/test_ensemble_source_routes.py and the
Rust twin in tools/rw_wps/crates/mapped-engine/src/join.rs.
"""

from __future__ import annotations

import copy
from datetime import datetime
from types import MappingProxyType
import json

import numpy as np
import pytest

from gpuwm.mapped_composition import (
    _compose_bound_fields,
    companion_ladder,
    load_composition,
)
from gpuwm.mapped_source import _DecodedCollection, _DirectValue


CYCLE = datetime(2026, 10, 5, 12)
LATER = datetime(2026, 10, 5, 13)


def _value(name, time, member, values, *, axes=("vertical", "y", "x"),
           cycle=CYCLE, ref="ref"):
    values = np.asarray(values, dtype=np.float64)
    return _DirectValue(
        name=name, valid_time=time, member=member, source_cycle=cycle,
        axes=tuple(axes), values=values,
        missing_count=int(np.isnan(values).sum()), references=(ref,))


def _collection(levels, values, *, member, times=(CYCLE,), cycle=CYCLE):
    direct = {}
    for time in times:
        for value in values(time):
            direct[(time, member, value.name)] = value
    return _DecodedCollection(
        latitude=np.array([30.0, 31.0]), longitude=np.array([260.0, 261.0, 262.0]),
        vertical_values=np.asarray(levels, dtype=np.float64),
        direct=MappingProxyType(direct),
        source_cycles=MappingProxyType({(time, member): cycle for time in times}),
        grid_fingerprint="fixture")


def _column(levels, base):
    return np.stack([np.full((2, 3), base + index) for index in range(len(levels))])


MEMBER_LEVELS = [25000.0, 50000.0, 100000.0]
DONOR_LEVELS = [200.0, 5000.0, 20000.0, 25000.0, 50000.0, 70000.0, 100000.0]


def _member(times=(CYCLE,)):
    return _collection(MEMBER_LEVELS, lambda time: [
        _value("air_temperature", time, "1", _column(MEMBER_LEVELS, 100.0), ref="member"),
        _value("surface_pressure", time, "1", np.full((2, 3), 9e4),
               axes=("y", "x"), ref="member"),
    ], member="1", times=times)


def _donor(times=(CYCLE,), cycle=CYCLE, member=None):
    return _collection(DONOR_LEVELS, lambda time: [
        _value("air_temperature", time, member, _column(DONOR_LEVELS, 500.0),
               cycle=cycle, ref="companion"),
        _value("land_fraction", time, member, np.ones((2, 3)), axes=("y", "x"),
               cycle=cycle, ref="companion"),
    ], member=member, times=times, cycle=cycle)


BINDING = {
    "source_id": "deterministic", "fields": ["land_fraction"],
    "grid_alignment": "exact_coordinate_subset",
    "time_alignment": "valid_time_exact",
    "companion": {"vertical_extension": "above_primary_top",
                  "fields": ["air_temperature"]},
}


def test_the_companion_stacks_only_its_levels_above_the_primary_top():
    composed, receipt = _compose_bound_fields(
        _member(), _donor(), binding_name="companion", binding=BINDING)
    assert composed.vertical_values.tolist() == [
        200.0, 5000.0, 20000.0, 25000.0, 50000.0, 100000.0]
    value = composed.direct[(CYCLE, "1", "air_temperature")]
    # Companion levels 0..2 (values 500..502) sit above the member's own
    # three levels (100..102); the member's 250 hPa stays the member's.
    assert value.values[:, 0, 0].tolist() == [500.0, 501.0, 502.0, 100.0, 101.0, 102.0]
    assert value.references == ("member", "companion")
    assert value.member == "1"
    extension = receipt["companion"]
    assert extension["primary_levels"] == MEMBER_LEVELS
    assert extension["companion_levels"] == [200.0, 5000.0, 20000.0]
    assert extension["composed_levels"] == composed.vertical_values.tolist()
    assert extension["fields"] == ["air_temperature"]
    assert extension["primary_member"] == "1"
    assert extension["companion_member"] is None
    assert set(extension["companion_slab_sha256"]) == {"air_temperature"}
    assert receipt["fields"] == ["land_fraction"]
    # The land state rides the same binding as an ordinary borrow.
    assert (CYCLE, "1", "land_fraction") in composed.direct


def test_a_descending_primary_ladder_keeps_its_own_order():
    descending = MEMBER_LEVELS[::-1]
    composed, ladder = companion_ladder(
        np.array(descending), np.array(DONOR_LEVELS), binding_name="c")
    assert composed.tolist() == descending + [20000.0, 5000.0, 200.0]
    assert ladder == [2, 1, 0]


def test_nothing_above_the_top_is_refused_rather_than_claimed():
    with pytest.raises(ValueError, match="carries no level above it"):
        companion_ladder(np.array([200.0, 1000.0]), np.array([500.0, 1000.0]),
                         binding_name="c")


def test_a_companion_from_another_cycle_is_refused():
    with pytest.raises(ValueError, match="from another cycle"):
        _compose_bound_fields(
            _member(), _donor(cycle=datetime(2026, 10, 5, 6)),
            binding_name="companion", binding=BINDING)


def test_a_companion_missing_a_valid_time_is_refused():
    with pytest.raises(ValueError, match="lacks '(land_fraction|air_temperature)' at primary valid time"):
        _compose_bound_fields(
            _member(times=(CYCLE, LATER)), _donor(times=(CYCLE,)),
            binding_name="companion", binding=BINDING)


def test_an_ordinary_binding_still_refuses_a_different_ladder():
    binding = {key: value for key, value in BINDING.items() if key != "companion"}
    binding["fields"] = ["air_temperature", "land_fraction"]
    member = _collection(MEMBER_LEVELS, lambda time: [
        _value("surface_pressure", time, "1", np.full((2, 3), 9e4), axes=("y", "x"))],
        member="1")
    with pytest.raises(ValueError, match="different vertical ladder"):
        _compose_bound_fields(member, _donor(), binding_name="donor", binding=binding)


def test_only_a_declared_companion_may_lend_another_members_values():
    binding = {key: value for key, value in BINDING.items() if key != "companion"}
    donor = _collection(MEMBER_LEVELS, lambda time: [
        _value("land_fraction", time, "2", np.ones((2, 3)), axes=("y", "x"))],
        member="2")
    with pytest.raises(ValueError, match="mix two trajectories"):
        _compose_bound_fields(_member(), donor, binding_name="donor", binding=binding)
    # A deterministic donor (no member) lends land state as the hybrids do.
    deterministic = _collection(MEMBER_LEVELS, lambda time: [
        _value("land_fraction", time, None, np.ones((2, 3)), axes=("y", "x"))],
        member=None)
    composed, receipt = _compose_bound_fields(
        _member(), deterministic, binding_name="donor", binding=binding)
    assert receipt["donor_member"] is None
    assert "companion" not in receipt


# -- grammar, on the packaged rrfs-ens documents --------------------------

def _packaged():
    from gpuwm.source_authorities import packaged_authorities

    paths = packaged_authorities("rrfs-ens-member-companion-grib2-v1")
    return (paths["composition"], paths["mapping"],
            json.loads(paths["composition"].read_text(encoding="utf-8")),
            json.loads(paths["mapping"].read_text(encoding="utf-8")))


def _binding(raw):
    return raw["field_sources"]["deterministic_companion"]


@pytest.mark.parametrize("change,match", [
    (lambda raw, mapping: _binding(raw)["companion"].update(fields=[
        "air_temperature", "specific_humidity", "eastward_wind",
        "northward_wind"]),
     "on the primary's ladder alone"),
    (lambda raw, mapping: _binding(raw)["companion"]["fields"].append(
        "surface_pressure"),
     "does not decode directly on its vertical ladder"),
    (lambda raw, mapping: _binding(raw).update(
        time_alignment="cycle_invariant_broadcast"),
     "freeze the top of a moving column"),
    (lambda raw, mapping: _binding(raw)["companion"].update(
        vertical_extension="below_primary_bottom"),
     "vertical_extension must be one of"),
    (lambda raw, mapping: raw["field_sources"].update(
        second={**copy.deepcopy(_binding(raw)), "mapping_role": "m2",
                "data_role": "d2", "provenance_role": "p2", "fields": []}),
     "both declare a companion"),
])
def test_companion_grammar_refusals_name_their_breakage(change, match):
    composition, mapping_path, raw, mapping = _packaged()
    change(raw, mapping)
    with pytest.raises(ValueError, match=match):
        load_composition(composition, mapping_path, _raw=raw,
                         _mapping_raw=mapping)


def test_a_companion_needs_a_pressure_ladder_to_know_what_is_above():
    from gpuwm.mapped_composition import _validate_companion

    _composition, _mapping, raw, mapping = _packaged()
    mapping["coordinates"]["vertical"]["kind"] = "hybrid_sigma_pressure"
    with pytest.raises(ValueError, match="only a pressure ladder"):
        _validate_companion(_binding(raw)["companion"], mapping, label="b",
                            time_alignment="valid_time_exact")


def test_a_companion_binding_may_lend_levels_alone():
    composition, mapping_path, raw, mapping = _packaged()
    load_composition(composition, mapping_path, _raw=raw, _mapping_raw=mapping)
    lone = copy.deepcopy(raw)
    _binding(lone)["fields"] = []
    with pytest.raises(ValueError, match="have no contributing source"):
        # The land gaps the mapping declares are now unbound: the grammar
        # still holds every composition_bound field to a provider.
        load_composition(composition, mapping_path, _raw=lone,
                         _mapping_raw=mapping)
