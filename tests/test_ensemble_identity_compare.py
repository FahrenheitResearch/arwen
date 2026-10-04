"""Grouped identity may change only a proved selected-roster descriptor."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

import pytest

from gpuwm.ensemble.recipes import RecipeMember, SourceRecipe, SourceTrajectory

_PATH = Path(__file__).resolve().parents[1] / "tools/ensemble_identity_compare.py"
_SPEC = importlib.util.spec_from_file_location("ensemble_identity_compare_relationship", _PATH)
comparator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparator)


def recipe():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    donors = tuple(SourceTrajectory("gefs", start, f"p{index:02d}") for index in (1, 2, 3))
    return SourceRecipe("recentered", SourceTrajectory("gfs", start), start,
        start + timedelta(hours=12),
        tuple(RecipeMember(index, 9000 + index, trajectory) for index, trajectory in zip((0, 7, 19), donors)),
        donors)


def selected_inputs(frozen, indices):
    selected = frozen.select_members(indices)
    request = {"recipe": {**frozen.describe(), "sha256": frozen.sha256}, "member_indices": list(indices)}
    roster = {"recipe": selected.describe(), "recipe_sha256": selected.sha256, "member_order": list(indices),
        "members": [{"member_id": member.index, "seed": member.seed, "trajectory_sha256": member.trajectory.identity,
            "recipe_sha256": selected.sha256, "preparation": {"execution_recipe_sha256": selected.sha256,
                "frozen_recipe_sha256": frozen.sha256}} for member in selected.members]}
    return request, roster


@pytest.fixture
def pair():
    frozen = recipe()
    group = comparator.selected_recipe_authority(*selected_inputs(frozen, (0, 19)))
    independent = comparator.selected_recipe_authority(*selected_inputs(frozen, (19,)))
    relation = {"group": group, "independent": independent, "member_id": 19, "seed": 9019,
                "stochastic_enabled": True}
    header = {"contract": "gpuwm-ensemble-stochastic-binding.v1", "member_id": 19,
        "recipe_sha256": group["execution_recipe_sha256"], "applied_steps": 240,
        "hook": {"enabled": True, "completed_step": 239, "spp_levels": {}, "spp": {}, "skebs": None,
            "sppt": {"completed_step": 239,
                "metadata": {"member_seed": 9019, "config": {"kind": "sppt", "stddev": 0.5},
                    "dt": 15.0, "rng_version": "gpuwm-stochastic-philox4x32-10.v1", "shape": [7, 9]},
                "spectrum": {"array": "stochastic/binding/hook/sppt/spectrum", "dtype": "complex64", "shape": [7, 9]}}}}
    single = deepcopy(header)
    single["recipe_sha256"] = independent["execution_recipe_sha256"]
    return header, single, relation


def test_canonical_selection_retains_original_donor_population_and_raw_headers(pair):
    left, right, relation = pair
    original = deepcopy((left, right))
    result = comparator.stochastic_header_comparison(left, right, relation)
    assert result["status"] == "PASS"
    assert result["scientific_differences"] == []
    assert [row["path"] for row in result["raw_differences"]] == ["/recipe_sha256"]
    assert result["raw_headers"] == {"group": original[0], "independent": original[1]}
    assert (left, right) == original
    assert relation["group"]["donor_population"] == relation["independent"]["donor_population"]
    assert len(relation["independent"]["donor_population"]) == 3


@pytest.mark.parametrize("path,value", [
    (("applied_steps",), 241),
    (("hook", "completed_step"), 240),
    (("hook", "sppt", "completed_step"), 240),
    (("hook", "sppt", "metadata", "config", "stddev"), 0.5000000000000001),
    (("hook", "sppt", "metadata", "dt"), 15.000000000000002),
    (("hook", "sppt", "metadata", "rng_version"), "other-rng"),
    (("hook", "sppt", "spectrum", "dtype"), "complex128"),
    (("hook", "sppt", "spectrum", "shape"), [7, 10]),
    (("hook", "spp_levels"), {"pbl": 49}),
    (("hook", "sppt", "metadata", "new_authority"), "unmatched"),
])
def test_group_relation_never_ignores_numerical_rng_or_unknown_authority(pair, path, value):
    left, right, relation = pair
    target = right
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = comparator.stochastic_header_comparison(left, right, relation)
    assert result["status"] == "FAIL"
    expected = "/" + "/".join(path)
    assert any(row["path"] == expected or row["path"].startswith(expected + "/")
               for row in result["scientific_differences"])


def test_seed_and_selection_hashes_must_match_the_verified_member(pair):
    left, right, relation = pair
    right["hook"]["sppt"]["metadata"]["member_seed"] += 1
    with pytest.raises(ValueError, match="process seed"):
        comparator.stochastic_header_comparison(left, right, relation)
    right["hook"]["sppt"]["metadata"]["member_seed"] -= 1
    left["recipe_sha256"] = right["recipe_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="canonical selected recipe"):
        comparator.stochastic_header_comparison(left, right, relation)


def test_selection_rejects_changed_full_recipe_and_unproved_roster():
    frozen = recipe()
    request, roster = selected_inputs(frozen, (0, 19))
    roster["members"][0]["preparation"]["execution_recipe_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frozen recipe, seed or trajectory"):
        comparator.selected_recipe_authority(request, roster)
    request, roster = selected_inputs(frozen, (0, 19))
    request["recipe"]["donor_population"].pop()
    with pytest.raises(ValueError, match="canonical frozen source recipe"):
        comparator.selected_recipe_authority(request, roster)


def test_group_relation_rejects_missing_active_stochastic_state_or_reduced_donors(pair):
    left, right, relation = pair
    with pytest.raises(ValueError, match="omitted its stochastic"):
        comparator.stochastic_header_comparison(None, None, relation)
    relation["independent"]["donor_population"].pop()
    with pytest.raises(ValueError, match="grouped-to-singleton"):
        comparator.stochastic_header_comparison(left, right, relation)
