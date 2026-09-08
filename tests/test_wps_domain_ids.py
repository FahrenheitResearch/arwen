"""Stable nest identity remains separate from compact WPS array positions."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.wps_domain_ids import domain_ids_from_wps_text, validated_domain_order, with_domain_ids


def test_legacy_bytes_are_unchanged_and_mapping_round_trips():
    text = "&share\n max_dom = 3,\n/\n"
    assert domain_ids_from_wps_text(text, 3) == (1, 2, 3)
    assert with_domain_ids(text, (1, 2, 3)) == text
    declared = with_domain_ids(text, (1, 3, 4))
    assert domain_ids_from_wps_text(declared, 3) == (1, 3, 4)
    assert with_domain_ids(declared, (1, 3, 4)) == declared
    assert with_domain_ids(declared, (1, 2, 3)) == text


@pytest.mark.parametrize("comment", [
    "! GPUWM_DOMAIN_IDS_V1 = 1,3,3", "! GPUWM_DOMAIN_IDS_V1 = 2,3,4",
    "! GPUWM_DOMAIN_IDS_V1 = 1,3", "! GPUWM_DOMAIN_IDS_V1 = 1,0,4",
    "! GPUWM_DOMAIN_IDS_V2 = 1,3,4", "! GPUWM_DOMAIN_IDS_V1 = 1,3,4 trailing",
    "! GPUWM_DOMAIN_IDS_V1 = 1,3,4\n! GPUWM_DOMAIN_IDS_V1 = 1,3,4",
])
def test_malformed_ambiguous_or_mis_sized_identity_refuses(comment):
    with pytest.raises(ValueError, match="identity"):
        domain_ids_from_wps_text(comment + "\n&share\n max_dom = 3,\n/\n", 3)


def test_domain_order_requires_one_root_and_existing_earlier_parents():
    rows = [SimpleNamespace(grid_id=1, parent_id=0), SimpleNamespace(grid_id=3, parent_id=1),
            SimpleNamespace(grid_id=4, parent_id=3)]
    assert validated_domain_order(rows) == (1, 3, 4)
    with pytest.raises(ValueError, match="precede"):
        validated_domain_order([rows[0], rows[2], rows[1]])
    rows[1].parent_id = 0
    with pytest.raises(ValueError, match="precede"):
        validated_domain_order(rows)


def test_tiles_rebase_preserves_slot_identity_and_rejects_bad_metadata(tmp_path):
    from gpuwm.starter_template import _tiles_wps
    from gpuwm.namelist_import import parse_namelist_text
    text = with_domain_ids("&share\n max_dom=3,\n/\n&geogrid\n geog_data_path='GEOG',\n/\n", (1, 3, 4))
    result = _tiles_wps(tmp_path / "source.wps", tmp_path / "next" / "candidate.wps", text=text)
    assert domain_ids_from_wps_text(result, 3) == (1, 3, 4)
    assert parse_namelist_text(result)["geogrid"]["geog_data_path"] == [str((tmp_path / "GEOG").resolve())]
    with pytest.raises(ValueError, match="identity"):
        _tiles_wps(tmp_path / "source.wps", tmp_path / "candidate.wps", text=text.replace("V1", "V2"))


def test_native_artifact_inventory_and_reuse_keep_noncontiguous_ids(tmp_path):
    from gpuwm import stage_reuse
    from gpuwm.wrf_direct import write_domain_artifacts_manifest, _validated_hierarchy
    from test_stage_reuse_hierarchy import _domain
    artifacts = [_domain(tmp_path / f"d{grid_id:02}", grid_id) for grid_id in (1, 3, 4)]
    write_domain_artifacts_manifest(tmp_path / "domain-artifacts.json", list(reversed(artifacts)))
    identity, snapshot, readers, error = stage_reuse._published(tmp_path)
    assert error is None and identity["domain_config"]["grid_id"] == 1
    assert set(snapshot["domains"]) == {"d01", "d03", "d04"}
    assert len(readers) == 3
    domains = tuple(SimpleNamespace(grid_id=grid_id, parent_id=parent,
                        run=SimpleNamespace(specified=grid_id == 1, nested=grid_id != 1))
                    for grid_id, parent in ((1, 0), (3, 1), (4, 3)))
    exp = SimpleNamespace(domains=domains, projection=SimpleNamespace(map_proj="lambert"))
    assert [artifact.grid_id for _, artifact in _validated_hierarchy(exp, list(reversed(artifacts)))] == [1, 3, 4]
    with pytest.raises(ValueError, match="exactly cover"):
        _validated_hierarchy(exp, artifacts[:2])


def test_hrrr_gate_accepts_stable_child_id_without_changing_root_or_physics():
    from gpuwm.hrrr_hierarchy_direct import _supported_hierarchy_slice
    from test_hrrr_hierarchy_direct import _native, _target
    exp = _native()
    child = exp.domains[1]
    child = replace(child, grid_id=3, run=replace(child.run, grid_id=3))
    exp = replace(exp, domains=(exp.domains[0], child))
    _supported_hierarchy_slice(exp, _target(), forcing_hours=tuple(range(13)))
    with pytest.raises(ValueError, match="one-way"):
        _supported_hierarchy_slice(replace(exp, feedback=1), _target(), forcing_hours=tuple(range(13)))
