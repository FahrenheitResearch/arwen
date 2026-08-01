# tests/test_feedback_reference_pair.py  (WP-5 vs-WRF nest tier: scoring a
# stock-WRF feedback A/B pair as a state difference)
"""Synthetic A/B fixtures for the feedback reference scorer.

Every fixture here is synthetic: a provenance record, an operator map and
two field sets built in the test, so the scorer's behaviour is pinned
without a 5.9 GB reference bank present.  What the fixtures deliberately
reproduce from a real pair is its GEOMETRY, because the zone derivation
is the part that cannot be checked by inspection --
:func:`test_derived_zones_reproduce_a_reference_pairs_own_zones` scores
the derivation against the zones a real pair computed and shipped, and
that check fails if the derivation drifts by a single cell.

The controls, named:

* single-variable guarantee -- non-identical inputs, a non-pristine
  build, and a pair that is not 0-vs-1 each refuse to load;
* operator routing -- a name-based router (the mutant) and the shipped
  staggering-based router are both evaluated on the same map and the
  fields they disagree on are committed;
* attribution, both directions -- difference injected only inside the
  feedback zone must not appear in the rim, and difference injected only
  outside the footprint must read exactly zero inside the zone;
* null -- run A scored against itself reads exactly 0.0 on every metric,
  so a scorer that always returned zeros is distinguishable from a null
  result.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from gpuwm.verify.feedback_reference import (
    DIFFERENCE_METRICS, OPERATOR_CLASSES, ReferencePairError, Zone,
    build_feedback_ab_tier, feedback_zone, load_operator_map,
    load_reference_pair_layout, nest_footprint, operator_class,
    operator_class_census, score_difference, score_field_regions, score_frame,
)

#: Synthetic pair geometry: small enough to build arrays for, with a
#: 4:1 ratio so the even-ratio operator split is exercised.
_FIXTURE_GEOMETRY = {
    "parent": {"e_we": 21, "e_sn": 17, "e_vert": 10, "dx_m": 12000},
    "nest": {"e_we": 21, "e_sn": 17, "e_vert": 10, "dx_m": 3000,
             "parent_grid_ratio": 4, "i_parent_start": 5, "j_parent_start": 4},
}
_FIXTURE_NY = _FIXTURE_GEOMETRY["parent"]["e_sn"] - 1
_FIXTURE_NX = _FIXTURE_GEOMETRY["parent"]["e_we"] - 1


def _provenance(**overrides):
    record = {
        "purpose": "synthetic feedback A/B fixture",
        "experiment": {
            "case": "synthetic",
            "domains": {"d01": dict(_FIXTURE_GEOMETRY["parent"]),
                        "d02": dict(_FIXTURE_GEOMETRY["nest"])},
            "only_difference": "domains.feedback  (A=0, B=1)",
        },
        "build": {"pristine": True, "git_commit": "0" * 40,
                  "wrf_exe_sha256": "a" * 64},
        "inputs": {"identical": True},
        "execution": {
            "runs": {
                "A_feedback0": {"dir": "/pair/runA/run", "feedback": 0},
                "B_feedback1": {"dir": "/pair/runB/run", "feedback": 1},
            },
        },
    }
    for path, value in overrides.items():
        target = record
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return record


def _write_provenance(tmp_path, **overrides):
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(_provenance(**overrides), indent=1),
                    encoding="utf-8")
    return path


def _layout(tmp_path, **overrides):
    return load_reference_pair_layout(_write_provenance(tmp_path, **overrides))


#: A miniature operator map in the shipped shape.  ``wind_10m_x`` and
#: ``wind_10m_y`` are the trap: diagnostics carried at mass points whose
#: names look like the prognostic winds.
_FIXTURE_OPERATOR_MAP = {
    "theta": {"op": "copy_fcn", "xstag": False, "ystag": False},
    "wind_x": {"op": "copy_fcn", "xstag": True, "ystag": False},
    "wind_y": {"op": "copy_fcn", "xstag": False, "ystag": True},
    "wind_10m_x": {"op": "copy_fcn", "xstag": False, "ystag": False},
    "wind_10m_y": {"op": "copy_fcn", "xstag": False, "ystag": False},
    "skin_temperature": {"op": "copy_fcnm", "xstag": False, "ystag": False},
    "land_category": {"op": "copy_fcni", "xstag": False, "ystag": False},
    "nest_marker": {"op": "mark_domain", "xstag": False, "ystag": False},
}


def _write_operator_map(tmp_path):
    path = tmp_path / "feedback_operator_map.json"
    path.write_text(json.dumps(_FIXTURE_OPERATOR_MAP, indent=1),
                    encoding="utf-8")
    return path


def _field(value=0.0, *, ny=_FIXTURE_NY, nx=_FIXTURE_NX):
    return np.full((ny, nx), value, dtype=np.float64)


# ---- single-variable guarantee controls -----------------------------------

def test_layout_loads_from_the_pairs_own_provenance(tmp_path):
    layout = _layout(tmp_path)
    assert layout.feedback_a == 0 and layout.feedback_b == 1
    assert layout.parent_grid_ratio == 4
    assert layout.inputs_identical and layout.build_pristine
    assert str(layout.run_a_dir).endswith("runA/run")
    assert str(layout.run_b_dir).endswith("runB/run")
    assert len(layout.provenance_sha256) == 64


def test_non_identical_inputs_refuse_to_load(tmp_path):
    with pytest.raises(ReferencePairError, match="inputs are not identical"):
        _layout(tmp_path, **{"inputs.identical": False})


def test_non_pristine_binary_refuses_to_load(tmp_path):
    with pytest.raises(ReferencePairError, match="not recorded pristine"):
        _layout(tmp_path, **{"build.pristine": False})


def test_a_pair_without_a_feedback_difference_refuses_to_load(tmp_path):
    runs = {"A": {"dir": "/pair/runA/run", "feedback": 1},
            "B": {"dir": "/pair/runB/run", "feedback": 1}}
    with pytest.raises(ReferencePairError, match="no difference to attribute"):
        _layout(tmp_path, **{"execution.runs": runs})


def test_a_pair_that_is_not_zero_versus_one_refuses_to_load(tmp_path):
    runs = {"A": {"dir": "/pair/runA/run", "feedback": 0},
            "B": {"dir": "/pair/runB/run", "feedback": 2}}
    with pytest.raises(ReferencePairError, match="feedback 0 vs 1"):
        _layout(tmp_path, **{"execution.runs": runs})


# ---- operator routing + its name-routing mutant ---------------------------

def test_operator_routing_is_by_staggering_not_by_name(tmp_path, capsys):
    """MUTATION CONTROL: the name router, run beside the shipped one.

    Neither router is asserted to be "right" by fiat -- the shipped map
    states each field's staggering, the staggering router follows it, and
    the fields the name router puts elsewhere are printed and committed.
    """
    classes, digest = load_operator_map(_write_operator_map(tmp_path))
    assert len(digest) == 64

    def _name_routed(name):                      # the mutant
        lowered = name.lower()
        if lowered.startswith("wind") and lowered.endswith("_x"):
            return "u_face"
        if lowered.startswith("wind") and lowered.endswith("_y"):
            return "v_face"
        return classes[name]

    disagreements = sorted(name for name in classes
                           if _name_routed(name) != classes[name])
    print(f"name-routing mutant disagrees on {disagreements}; "
          f"shipped classes "
          f"{ {n: classes[n] for n in disagreements} }")
    assert disagreements == ["wind_10m_x", "wind_10m_y"]
    assert classes["wind_10m_x"] == classes["wind_10m_y"] == "mass"
    assert classes["wind_x"] == "u_face"
    assert classes["wind_y"] == "v_face"
    assert classes["skin_temperature"] == "masked"
    assert classes["land_category"] == "integer"
    assert classes["nest_marker"] == "mark_domain"


def test_a_doubly_staggered_field_has_no_operator():
    with pytest.raises(ReferencePairError, match="doubly-staggered"):
        operator_class({"op": "copy_fcn", "xstag": True, "ystag": True})


def test_an_unknown_operator_refuses_rather_than_defaulting():
    with pytest.raises(ReferencePairError, match="unknown feedback operator"):
        operator_class({"op": "copy_everything", "xstag": False,
                        "ystag": False})


def test_census_covers_every_known_class(tmp_path):
    classes, _ = load_operator_map(_write_operator_map(tmp_path))
    census = operator_class_census(classes)
    assert set(census) == set(OPERATOR_CLASSES)
    assert census["mass"] == 3          # theta + the two 10 m diagnostics
    assert census["u_face"] == census["v_face"] == 1
    assert sum(census.values()) == len(_FIXTURE_OPERATOR_MAP)


# ---- zone derivation ------------------------------------------------------

#: Geometry and shipped zones of a real stock-WRF feedback pair, kept as
#: the derivation's ground truth.  The zones were computed by that pair's
#: own analyzer from the WRF loop bounds and shipped beside its runs.
_REFERENCE_GEOMETRY = {
    "parent": {"e_we": 251, "e_sn": 201, "e_vert": 50, "dx_m": 12000},
    "nest": {"e_we": 501, "e_sn": 401, "e_vert": 50, "dx_m": 3000,
             "parent_grid_ratio": 4, "i_parent_start": 63,
             "j_parent_start": 51},
}
_REFERENCE_ZONES_0BASED = {
    "mass": {"i": [63, 185], "j": [51, 148]},
    "masked": {"i": [63, 185], "j": [51, 148]},
    "u_face": {"i": [63, 186], "j": [51, 148]},
    "v_face": {"i": [63, 185], "j": [51, 149]},
}
_REFERENCE_FOOTPRINT_0BASED = {"i": [62, 186], "j": [50, 149]}


def test_derived_zones_reproduce_a_reference_pairs_own_zones(tmp_path):
    domains = {"d01": dict(_REFERENCE_GEOMETRY["parent"]),
               "d02": dict(_REFERENCE_GEOMETRY["nest"])}
    layout = _layout(tmp_path, **{"experiment.domains": domains})
    for class_name, expected in _REFERENCE_ZONES_0BASED.items():
        zone = feedback_zone(layout, class_name)
        assert [zone.i0, zone.i1] == expected["i"], class_name
        assert [zone.j0, zone.j1] == expected["j"], class_name
    footprint = nest_footprint(layout)
    assert [footprint.i0, footprint.i1] == _REFERENCE_FOOTPRINT_0BASED["i"]
    assert [footprint.j0, footprint.j1] == _REFERENCE_FOOTPRINT_0BASED["j"]


def test_the_feedback_zone_is_inside_the_footprint_on_every_side(tmp_path):
    layout = _layout(tmp_path)
    footprint = nest_footprint(layout)
    zone = feedback_zone(layout, "mass")
    assert zone.i0 == footprint.i0 + 1 and zone.i1 == footprint.i1 - 1
    assert zone.j0 == footprint.j0 + 1 and zone.j1 == footprint.j1 - 1


def test_face_zones_extend_by_one_along_their_own_stagger_only(tmp_path):
    layout = _layout(tmp_path)
    mass = feedback_zone(layout, "mass")
    u_face = feedback_zone(layout, "u_face")
    v_face = feedback_zone(layout, "v_face")
    assert (u_face.i0, u_face.i1) == (mass.i0, mass.i1 + 1)
    assert (u_face.j0, u_face.j1) == (mass.j0, mass.j1)
    assert (v_face.i0, v_face.i1) == (mass.i0, mass.i1)
    assert (v_face.j0, v_face.j1) == (mass.j0, mass.j1 + 1)


def test_an_unknown_class_has_no_zone(tmp_path):
    layout = _layout(tmp_path)
    with pytest.raises(ReferencePairError, match="unknown operator class"):
        feedback_zone(layout, "cell_average")


# ---- null control ---------------------------------------------------------

def test_a_run_scored_against_itself_reads_exactly_zero(tmp_path, capsys):
    layout = _layout(tmp_path)
    rng = np.random.default_rng(20260731)
    field = rng.normal(size=(_FIXTURE_NY, _FIXTURE_NX))
    scored = score_field_regions(field, field.copy(), layout=layout,
                                 class_name="mass")
    for region in ("fb_zone", "footprint", "full"):
        for metric in DIFFERENCE_METRICS:
            assert scored[region][metric] == 0.0, (region, metric)
    print("null control: every metric exactly 0.0 over all three regions")


# ---- attribution control, both directions ---------------------------------

def test_difference_inside_the_zone_is_attributed_to_the_zone(tmp_path,
                                                              capsys):
    layout = _layout(tmp_path)
    zone = feedback_zone(layout, "mass")
    a = _field()
    b = _field()
    b[zone.j0:zone.j1 + 1, zone.i0:zone.i1 + 1] = 5.0
    scored = score_field_regions(a, b, layout=layout, class_name="mass")

    footprint = nest_footprint(layout)
    rim_mask = footprint.mask(_FIXTURE_NY, _FIXTURE_NX) & ~zone.mask(
        _FIXTURE_NY, _FIXTURE_NX)
    rim_max = float(np.max(np.abs(b - a)[rim_mask]))
    print(f"zone-only injection: fb_zone max_abs "
          f"{scored['fb_zone']['max_abs']}, frac_nonzero "
          f"{scored['fb_zone']['frac_nonzero']}, rim max_abs {rim_max}")

    assert scored["fb_zone"]["max_abs"] == 5.0
    assert scored["fb_zone"]["frac_nonzero"] == 1.0
    assert rim_max == 0.0
    assert scored["full"]["frac_nonzero"] < scored["fb_zone"]["frac_nonzero"]


def test_difference_outside_the_footprint_leaves_the_zone_at_zero(tmp_path,
                                                                  capsys):
    layout = _layout(tmp_path)
    footprint = nest_footprint(layout)
    a = _field()
    b = _field()
    outside = ~footprint.mask(_FIXTURE_NY, _FIXTURE_NX)
    b[outside] = 3.0
    scored = score_field_regions(a, b, layout=layout, class_name="mass")
    print(f"outside-only injection: fb_zone max_abs "
          f"{scored['fb_zone']['max_abs']}, full max_abs "
          f"{scored['full']['max_abs']}, full frac_nonzero "
          f"{scored['full']['frac_nonzero']}")

    assert scored["fb_zone"]["max_abs"] == 0.0
    assert scored["fb_zone"]["frac_nonzero"] == 0.0
    assert scored["footprint"]["max_abs"] == 0.0
    assert scored["full"]["max_abs"] == 3.0
    assert scored["full"]["frac_nonzero"] > 0.0


# ---- scoring mechanics ----------------------------------------------------

def test_metrics_are_computed_over_the_zone_and_not_the_grid():
    zone = Zone(i0=1, i1=2, j0=1, j1=2)
    a = np.zeros((4, 4))
    b = np.zeros((4, 4))
    b[1, 1] = 2.0
    scored = score_difference(a, b, zone)
    assert scored["max_abs"] == 2.0
    assert scored["mean_abs"] == pytest.approx(0.5)
    assert scored["rms"] == pytest.approx(1.0)
    assert scored["frac_nonzero"] == pytest.approx(0.25)


def test_a_column_field_scores_over_the_same_horizontal_zone():
    zone = Zone(i0=0, i1=1, j0=0, j1=1)
    a = np.zeros((3, 2, 2))
    b = np.zeros((3, 2, 2))
    b[:, 0, 0] = 1.0
    scored = score_difference(a, b, zone)
    assert scored["max_abs"] == 1.0
    assert scored["frac_nonzero"] == pytest.approx(0.25)


def test_mismatched_shapes_refuse_to_score():
    with pytest.raises(ReferencePairError, match="disagree in shape"):
        score_difference(np.zeros((4, 4)), np.zeros((4, 5)),
                         Zone(i0=0, i1=3, j0=0, j1=3))


def test_a_zone_off_the_grid_refuses_rather_than_clipping():
    with pytest.raises(ReferencePairError, match="does not fit"):
        score_difference(np.zeros((4, 4)), np.zeros((4, 4)),
                         Zone(i0=0, i1=9, j0=0, j1=3))


def test_an_unrouted_field_refuses_rather_than_defaulting(tmp_path):
    layout = _layout(tmp_path)
    classes, _ = load_operator_map(_write_operator_map(tmp_path))
    fields = {"theta": _field(), "unlisted": _field()}
    with pytest.raises(ReferencePairError, match="no operator-map entry"):
        score_frame(fields, {k: v.copy() for k, v in fields.items()},
                    layout=layout, classes=classes)


def test_frames_with_different_field_sets_refuse_to_score(tmp_path):
    layout = _layout(tmp_path)
    classes, _ = load_operator_map(_write_operator_map(tmp_path))
    with pytest.raises(ReferencePairError, match="different field sets"):
        score_frame({"theta": _field()},
                    {"theta": _field(), "wind_10m_x": _field()},
                    layout=layout, classes=classes)


def test_a_frame_routes_every_field_by_its_own_class(tmp_path):
    layout = _layout(tmp_path)
    classes, _ = load_operator_map(_write_operator_map(tmp_path))
    names = ["theta", "wind_10m_x", "skin_temperature"]
    a = {name: _field() for name in names}
    b = {name: _field() for name in names}
    scored = score_frame(a, b, layout=layout, classes=classes)
    assert scored["theta"]["class"] == "mass"
    assert scored["skin_temperature"]["class"] == "masked"
    assert set(scored) == set(names)


# ---- the receipt tier -----------------------------------------------------

def _tier(tmp_path, **overrides):
    layout = _layout(tmp_path)
    classes, digest = load_operator_map(_write_operator_map(tmp_path))
    kwargs = {
        "layout": layout,
        "operator_map_sha256": digest,
        "class_census": operator_class_census(classes),
        "frames": [{"frame": "f0", "fields": {}}],
        "null_control": {"method": "run B scored against run A's own file",
                         "max_abs_over_all_fields": 0.0},
    }
    kwargs.update(overrides)
    return build_feedback_ab_tier(**kwargs)


def test_the_measured_tier_carries_its_provenance_and_zones(tmp_path):
    tier = _tier(tmp_path)
    assert tier["status"] == "measured"
    assert tier["decision"] == "D-8"
    assert tier["reference"]["inputs_identical"] is True
    assert len(tier["operator_map_sha256"]) == 64
    assert set(tier["zones"]) == set(OPERATOR_CLASSES)
    assert tier["metrics"] == list(DIFFERENCE_METRICS)


def test_the_tier_records_the_evidence_form_wrf_cannot_produce(tmp_path):
    tier = _tier(tmp_path)
    assert len(tier["not_available"]) == 1
    entry = tier["not_available"][0]
    assert "coupling tendency" in entry["item"]
    assert entry["reason"] and entry["evidence"] and entry["consequence"]


def test_a_tier_without_its_null_control_refuses_to_build(tmp_path):
    with pytest.raises(ReferencePairError, match="null control"):
        _tier(tmp_path, null_control={})


def test_a_tier_without_a_scored_frame_refuses_to_build(tmp_path):
    with pytest.raises(ReferencePairError, match="at least one scored frame"):
        _tier(tmp_path, frames=[])


def test_a_tier_with_an_incomplete_census_refuses_to_build(tmp_path):
    with pytest.raises(ReferencePairError, match="missing class"):
        _tier(tmp_path, class_census={"mass": 1})


def test_a_tier_without_its_operator_map_digest_refuses_to_build(tmp_path):
    with pytest.raises(ReferencePairError, match="binds the operator map"):
        _tier(tmp_path, operator_map_sha256="")


# ---- token hygiene --------------------------------------------------------

def test_no_case_token_in_the_feedback_reference_module():
    import re

    from gpuwm.verify import feedback_reference

    with open(feedback_reference.__file__, encoding="utf-8") as handle:
        source = handle.read()
    hits = re.findall(r"real74|1974|ohio|hrrr", source, flags=re.IGNORECASE)
    assert hits == [], hits


# ---- the scoring tool's own controls --------------------------------------

def _load_scoring_tool():
    import importlib.util
    import sys
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1]
            / "tools" / "score_feedback_reference_pair.py")
    spec = importlib.util.spec_from_file_location(
        "score_feedback_reference_pair", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_tool_refuses_a_history_variable_it_cannot_route(tmp_path):
    tool = _load_scoring_tool()
    classes, _ = load_operator_map(_write_operator_map(tmp_path))
    assert tool.resolve_registry_key("THETA", classes) == "theta"
    with pytest.raises(ReferencePairError, match="no operator-map entry"):
        tool.resolve_registry_key("UNLISTED", classes)


def test_the_tool_cross_checks_staggering_against_the_files_dimensions():
    """CONTROL: the map and the file must agree about staggering.

    Both directions fail -- a map that claims a face field for an
    unstaggered variable, and a map that claims a mass field for a
    staggered one.  Either disagreement means the file and the map are
    not describing the same run, and scoring would use the wrong region.
    """
    tool = _load_scoring_tool()
    tool.check_staggering("wind_x", "u_face", True, False)     # agreeing
    tool.check_staggering("theta", "mass", False, False)       # agreeing
    with pytest.raises(ReferencePairError, match="carries it unstaggered"):
        tool.check_staggering("wind_x", "u_face", False, False)
    with pytest.raises(ReferencePairError, match="dimensions say"):
        tool.check_staggering("theta", "mass", True, False)
    with pytest.raises(ReferencePairError, match="dimensions say"):
        tool.check_staggering("wind_x", "u_face", False, True)
