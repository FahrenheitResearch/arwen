"""CPU checks through the real static preparation and Rust smoother doors."""
from datetime import date
import hashlib
import json

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.static.lambert import LambertGrid
from gpuwm.static.terrain_autosmooth import (
    MAX_PASSES, field_hash, measured_slope_limit, prepare_fields,
    receipt_of, run_line, smooth_to_limit, verify_receipt)
from gpuwm.static.highres_production import apply_prepared_highres
from gpuwm.static.terrain_smoothing import TerrainSmoothing, smooth_terrain
from gpuwm.acoustic_adaptation import steepest_slope


def grid():
    return LambertGrid(25, 0, 20, 30, 0, 500, 500, 25, 25)


def fields(steep=True):
    h = np.zeros((24, 24))
    h[10:14, 10:14] = 2000 if steep else 20
    return {"HGT_M": h, "LANDMASK": np.ones_like(h),
            "SOILTEMP": np.full_like(h, 285),
            "TMN": 285 - .0065 * h,
            "GREENFRAC": np.full((12, 24, 24), .3)}


def hashes(values):
    return {key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in values.items()}


def test_measured_limit_uses_clock_and_acoustic_maps():
    from gpuwm.acoustic_adaptation import STABLE_SLOPE_BY_OFFCENTERING
    limit, arms = measured_slope_limit(grid())
    assert limit == STABLE_SLOPE_BY_OFFCENTERING[-1][2] == .85
    assert arms
    coarse = grid()
    coarse.dx = coarse.dy = 12000
    assert 0 < measured_slope_limit(coarse)[0] < limit


def test_minimal_pass_count_receipt_and_real_rust_result():
    original = fields()["HGT_M"]
    original_hash = field_hash(original)
    result, receipt = smooth_to_limit(original, grid(), domain_id=3)
    assert receipt["smooth_passes"] == 5
    assert receipt["slope_before"] > .85 > receipt["slope_after"]
    assert receipt["terrain_before_sha256"] == original_hash
    assert receipt["terrain_after_sha256"] == field_hash(result)
    assert receipt["domain_id"] == 3
    json.dumps(receipt, allow_nan=False)
    extended = np.pad(original, 1, mode="edge")
    for _ in range(receipt["smooth_passes"] - 1):
        extended = smooth_terrain(extended, TerrainSmoothing("1-2-1", 1))
    factors = dict(msfu=grid().mapfac_u(), msfv=grid().mapfac_v())
    assert steepest_slope(extended[1:-1, 1:-1], 500, 500, **factors).slope >= .85
    extended = smooth_terrain(extended, TerrainSmoothing("1-2-1", 1))
    assert result.tobytes() == extended[1:-1, 1:-1].tobytes()
    assert field_hash(original) == original_hash


def test_prepare_door_smooths_only_the_steep_domain_and_recomputes_tmn(capsys):
    gentle, steep = fields(False), fields()
    old_gentle, old_steep = hashes(gentle), hashes(steep)
    held_receipt = {"cache_key": "existing"}
    held, receipt = apply_prepared_highres(
        gentle, grid(), config=None, domain_id=1, case_date=date(2026, 1, 1),
        landuse_attrs=None, baseline_receipt=held_receipt)
    assert held is gentle and receipt is held_receipt
    assert hashes(held) == old_gentle
    fixed, receipt = apply_prepared_highres(
        steep, grid(), config=None, domain_id=3, case_date=date(2026, 1, 1),
        landuse_attrs=None)
    assert receipt["terrain_autosmooth"] == receipt_of(fixed)
    assert hashes(steep) == old_steep
    assert np.array_equal(fixed["TMN"], 285 - .0065 * fixed["HGT_M"])
    for key in set(steep) - {"HGT_M", "TMN"}:
        assert hashes(fixed)[key] == old_steep[key]
    assert run_line(receipt_of(fixed)) in capsys.readouterr().out
    repeated, repeated_receipt = apply_prepared_highres(
        fixed, grid(), config=None, domain_id=3, case_date=date(2026, 1, 1),
        landuse_attrs=None, baseline_receipt=receipt)
    assert repeated is fixed and repeated_receipt == receipt


def test_bounded_refusal_has_breakage_and_remedy_and_keeps_input():
    h = fields()["HGT_M"]
    before = field_hash(h)
    with pytest.raises(ValueError, match="after 1 1-2-1 passes.*before GPU time.*non-finite.*coarser grid"):
        smooth_to_limit(h, grid(), domain_id=3, max_passes=1)
    assert field_hash(h) == before


@pytest.mark.parametrize("bound", [-1, MAX_PASSES + 1, True, 1.5])
def test_invalid_pass_bound_refused(bound):
    with pytest.raises(ValueError, match="pass bound"):
        smooth_to_limit(fields()["HGT_M"], grid(), domain_id=3, max_passes=bound)


def test_full_static_receipt_binds_the_written_field(tmp_path):
    from types import SimpleNamespace
    from gpuwm.native_domain_artifacts import write_domain_static_files
    fixed = prepare_fields(fields(), grid(), domain_id=3)
    cfg = RunConfig(nx=24, ny=24, nz=4, dx=500, dy=500,
                    ztop=20000, dt=1, run_seconds=1)
    static, geometry = write_domain_static_files(
        tmp_path, domain=SimpleNamespace(run=cfg), grid=grid(), static_fields=fixed)
    assert static["terrain_autosmooth"] == geometry["terrain_autosmooth"] == receipt_of(fixed)
    assert json.loads((tmp_path / "geometry-receipt.json").read_text())["terrain_autosmooth"] == receipt_of(fixed)
    with np.load(tmp_path / "native-static.npz") as payload:
        assert "terrain_autosmooth" not in payload
        assert field_hash(payload["HGT_M"]) == receipt_of(fixed)["terrain_after_sha256"]


def test_tampered_smoothing_record_refuses():
    fixed = prepare_fields(fields(), grid(), domain_id=3)
    receipt = receipt_of(fixed)
    assert verify_receipt(receipt, fixed["HGT_M"], domain_id=3) is receipt
    with pytest.raises(ValueError, match="does not bind"):
        verify_receipt(receipt, fixed["HGT_M"] + 1, domain_id=3)


@pytest.mark.parametrize("steep", [False, True])
def test_written_static_receipt_is_accepted_by_forecast_artifact_reader(tmp_path, steep):
    from types import SimpleNamespace
    from gpuwm.native_domain_artifacts import write_domain_static_files
    from gpuwm.prepared_domain_tree_forecast import _validate_domain_receipt, _sha256
    fixed = prepare_fields(fields(steep), grid(), domain_id=3)
    cfg = RunConfig(nx=24, ny=24, nz=4, dx=500, dy=500,
                    ztop=20000, dt=1, run_seconds=1)
    domain = SimpleNamespace(run=cfg, grid_id=3, parent_id=2)
    static, geometry = write_domain_static_files(
        tmp_path, domain=domain, grid=grid(), static_fields=fixed)
    reader = SimpleNamespace(content_sha256="prepared-fields", payload_bytes=8, arrays=("field",))
    receipt = {"schema": "gpuwm-native-domain-artifact-build-v1", "status": "READY",
        "grid_id": 3, "parent_id": 2, "boundary_mode": "nested-parent-forced",
        "artifacts": {"static_cache": static, "geometry_receipt": {
            "path": "geometry-receipt.json", "sha256": _sha256(tmp_path / "geometry-receipt.json"),
            "geometry": geometry["geometry"]},
            "prepared_cache": {"path": "prepared-cache", "content_sha256": reader.content_sha256,
                "payload_bytes": 8, "array_count": 1}},
        "verification": {"path": "prepared-cache", "status": "PASS",
            "content_sha256": reader.content_sha256, "payload_bytes": 8, "array_count": 1}}
    kwargs = dict(domain=domain, bundle=tmp_path, reader=reader,
        static_path=tmp_path / "native-static.npz", geometry_path=tmp_path / "geometry-receipt.json")
    _validate_domain_receipt(receipt, **kwargs)
    if steep:
        receipt["artifacts"]["static_cache"]["terrain_autosmooth"] = {
            **receipt_of(fixed), "smooth_passes": 1}
        with pytest.raises(ValueError, match="artifact hashes differ"):
            _validate_domain_receipt(receipt, **kwargs)


def test_nonfinite_terrain_is_refused_before_smoothing():
    h = fields()["HGT_M"]
    h[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite heights"):
        smooth_to_limit(h, grid(), domain_id=3)


def test_applied_record_changes_run_identity_only_for_changed_terrain():
    from types import SimpleNamespace
    from gpuwm import prepared_domain_tree_forecast as tree
    from test_acoustic_adaptation import _wizard_experiment
    exp = _wizard_experiment([(40, 40)], (), 500.)
    bundle = SimpleNamespace(grid_id=1, cache_reader=SimpleNamespace(
        content_sha256="same-fields"))
    inputs = SimpleNamespace(experiment=exp, domains=(bundle,),
        authority_sha256={"preparation_receipt": "same-proof"},
        execution_plan={"plan_id": "same-plan"})
    old = tree.tree_restart_identity_components(inputs, {"runtime": "same"})
    assert "terrain_autosmooth" not in old
    bundle.terrain_autosmooth = None
    assert tree.tree_restart_identity_components(inputs, {"runtime": "same"}) == old
    fixed = prepare_fields(fields(), grid(), domain_id=1)
    bundle.terrain_autosmooth = receipt_of(fixed)
    changed = tree.tree_restart_identity_components(inputs, {"runtime": "same"})
    assert changed.pop("terrain_autosmooth") == {"d01": receipt_of(fixed)}
    assert changed == old
    from gpuwm.prepared_single_domain_forecast import _single_checkpoint_identity
    single = SimpleNamespace(source="test", experiment=exp,
        cache_reader=SimpleNamespace(content_sha256="same-fields"),
        file_sha256={"geometry": "same-geometry"}, geometry_receipt={})
    old_single = _single_checkpoint_identity(single, {"runtime": "same"})
    single.geometry_receipt = {"terrain_autosmooth": receipt_of(fixed)}
    changed_single = _single_checkpoint_identity(single, {"runtime": "same"})
    assert changed_single.pop("terrain_autosmooth") == receipt_of(fixed)
    assert changed_single == old_single


@pytest.mark.parametrize("height", [20, 2000])
def test_real_domain_cli_surveys_staged_synthetic_terrain(tmp_path, height):
    from gpuwm.cli import main
    from test_static_build import _write_index, _write_tiles
    topo = tmp_path / "GEOG" / "topo_gmted2010_30s"
    topo.mkdir(parents=True)
    kv = _write_index(topo, dx=.005, dy=.005, known_lat=24.75,
                      known_lon=-.25, tile_x=120, tile_y=120)
    source = np.zeros((1, 120, 120), dtype=np.int16)
    source[0, 43:58, 43:58] = height
    _write_tiles(topo, source, kv)
    output = tmp_path / "domain.toml"
    assert main(["domain", "--point=25,0", "--root-dx", ".5",
                 "--point-extent-km", "16", "--vram-gib", "24",
                 "--hours", "1", "--source", "gfs", "--cycle",
                 "2026-10-07T12", "--geog-root", str(topo.parent),
                 "--out", str(output)]) == 0
    path = output.with_suffix(".terrain-autosmooth.json")
    assert path.exists() == (height == 2000)
    if path.exists():
        rows = json.loads(path.read_text())["domains"]
        assert len(rows) == 1 and rows[0]["domain_id"] == 1
        assert rows[0]["slope_after"] < rows[0]["slope_limit"]


def test_moving_steep_footprint_refuses_before_coordinate_initialization(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from test_acoustic_adaptation import _wizard_experiment
    from test_static_build import _write_index, _write_tiles
    from gpuwm.static.projection import grids_from_projection_config
    from gpuwm.vertical_adaptation import run_terrain_fields
    exp = _wizard_experiment([(40, 40), (30, 30)], (3,), 1500.)
    exp = replace(exp, domains=(exp.domains[0], replace(exp.domains[1], follow=object())))
    geog = tmp_path / "GEOG"
    topo = geog / "topo_gmted2010_30s"
    topo.mkdir(parents=True)
    kv = _write_index(topo, dx=.005, dy=.005,
                      known_lat=exp.projection.ref_lat - .25,
                      known_lon=exp.projection.ref_lon - .25,
                      tile_x=120, tile_y=120)
    data = np.zeros((1, 120, 120), dtype=np.int16)
    data[0, 43:58, 43:58] = 3000
    _write_tiles(topo, data, kv)
    wps = tmp_path / "namelist.wps"
    wps.write_text('&share\nmax_dom=2,\n/\n&geogrid\ngeog_data_res="default","default",\n/\n')
    catalog = SimpleNamespace(files=(
        SimpleNamespace(role="wps_namelist", path=wps),
        SimpleNamespace(role="geog_index", path=topo / "index")))
    with pytest.raises(ValueError, match="requires terrain auto-smoothing and moves.*overlap-statics"):
        run_terrain_fields(exp, grids_from_projection_config(exp),
            root_terrain=np.zeros((40, 40)), static_catalog=catalog, corridors=False)
