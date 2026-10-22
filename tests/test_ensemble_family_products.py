"""Family mass, GPU reducer reuse, masked scores and post-run receipt controls."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from gpuwm.ensemble.batch_products import (FieldProducts, apply_member_weights,
                                           probability_matched_mean)
from gpuwm.ensemble.family_products import family_roster, score_reflectivity, _input_frames, _clock


def test_family_mass_does_not_depend_on_number_of_children():
    rows = [{"member_id": i, "provenance": {"parent_family": name, "family_weight": weight}}
            for i, (name, weight) in enumerate((("ai", .7), ("physics", .3), ("physics", .3)))]
    roster = family_roster(rows)
    assert [r["weight"] for r in roster["members"]] == pytest.approx([.7, .15, .15])
    assert roster["weight_origin"] == "recorded"
    assert roster["members"][0]["provenance"] == rows[0]["provenance"]


def test_legacy_explicit_family_map_and_default_mass():
    roster = family_roster([{"member_id": 10}, {"member_id": 20}, {"member_id": 30}],
                           member_map={"10": "custom-ai", "20": "custom-physics", "30": "custom-physics"})
    assert [r["weight"] for r in roster["members"]] == [.5, .25, .25]
    assert roster["weight_origin"] == "equal-family-default"


def test_zero_family_mass_keeps_its_conditional_products():
    roster = family_roster([{"member_id": 0, "family": "ai", "family_weight": 1., "weight": 1.},
                            {"member_id": 1, "family": "physics", "family_weight": 0., "weight": 0.}])
    assert [row["weight"] for row in roster["members"]] == [1., 0.]
    assert [row["conditional_weight"] for row in roster["members"]] == [1., 1.]


@pytest.mark.parametrize("rows", [
    [{"member_id": 0}],
    [{"member_id": 0, "family": "ai", "family_weight": .5}, {"member_id": 1, "family": "ai", "family_weight": .6}],
    [{"member_id": 0, "family": "../escape"}],
    [{"member_id": 0, "family": "ai", "family_weight": float("nan")}],
    [{"member_id": 0, "family": "ai", "family_weight": .5}, {"member_id": 1, "family": "physics"}],
])
def test_bad_lineage_cannot_publish_probabilities(rows):
    with pytest.raises(ValueError):
        family_roster(rows)


def test_weighted_pmm_matches_replicated_population_and_ties():
    values = np.array([[[0, 5, 20, 40]], [[10, 30, 50, 60]]], np.float32)
    mean = (values[0] + 3 * values[1]) / 4
    replicated = np.concatenate([values[:1], np.repeat(values[1:], 3, axis=0)])
    np.testing.assert_array_equal(probability_matched_mean(values, mean, np, [.25, .75]),
                                  probability_matched_mean(replicated, mean, np))
    tied = np.full_like(mean, mean.mean())
    assert np.unique(probability_matched_mean(values, tied, np, [.25, .75])).size == 1


def test_weighted_statistics_mask_missing_cells_and_keep_probability_mass():
    values = np.array([[[0, 40, 10]], [[40, 50, np.nan]], [[40, 60, 50]]], np.float32)
    request = FieldProducts("refl", "dBZ", (30, 40))
    outputs = {"refl:mean": np.zeros((1, 3), np.float32),
               "refl:spread": np.zeros((1, 3), np.float32),
               "refl:probability": np.zeros((2, 1, 3), np.float32),
               "refl:pmm": np.zeros((1, 3), np.float32)}
    product = SimpleNamespace(_xp=np, members=3, fields={"refl": values},
                              requests=(request,), outputs=outputs)
    apply_member_weights(product, [.75, .125, .125])
    assert outputs["refl:mean"][0, 0] == 10
    assert outputs["refl:probability"][0, 0, 0] == .25
    assert np.isnan(outputs["refl:probability"][:, 0, 2]).all()
    assert np.isnan(outputs["refl:pmm"][0, 2])
    with pytest.raises(ValueError, match="weights"):
        apply_member_weights(product, [1, 2])


def test_scores_use_existing_rust_masks_and_several_neighbourhoods():
    observed = np.zeros((5, 7)); observed[2, 3] = 50
    forecast = observed.copy(); forecast[0, 0] = 70
    valid = np.ones_like(observed, bool); valid[0, 0] = False
    rows = score_reflectivity({("ai", "pmm"): forecast, ("combined", "mean"): observed},
                              observed, valid, dx_m=3000, half_widths=(0, 1, 3))
    assert len(rows) == 12
    assert all(row["fss"] == 1 and row["critical_success_index"] == 1 for row in rows)
    assert {row["box_length_m"] for row in rows} == {3000, 9000, 21000}
    assert all(row["total"] == 34 for row in rows)


def test_postrun_rejects_receipt_without_retained_members(tmp_path):
    manifest = tmp_path / "ensemble-manifest.json"
    manifest.write_text(json.dumps({"member_order": [0], "member_metadata": [{"member_id": 0}],
                                     "member_files": [], "frames": []}))
    with pytest.raises(ValueError, match="no retained"):
        _input_frames(manifest)


def test_no_observed_events_and_no_valid_cells_are_json_safe():
    values = np.zeros((5, 7))
    for mask in (np.ones_like(values, bool), np.zeros_like(values, bool)):
        rows = score_reflectivity({("combined", "mean"): values}, values, mask, dx_m=3000)
        json.dumps(rows, allow_nan=False)
        assert all(row["score_status"] == ("scored" if mask.any() else "no-common-valid-cells") for row in rows)


def test_history_clocks_and_member_map_provenance(tmp_path):
    manifest = tmp_path / "ensemble-manifest.json"
    manifest.write_text(json.dumps({"member_order": [1], "member_metadata": [{"member_id": 1}],
        "member_files": [{"member_id": 1, "path": "wrfout_d01_2022-06-13_23_00_00"}]}))
    _, _, _, frames = _input_frames(manifest)
    assert list(frames) == [(tmp_path.name, "2022-06-13T23:00:00")]
    assert _clock("2022-06-14T01:00:00+02:00") == "2022-06-13T23:00:00"
    roster = family_roster([{"member_id": 1}], member_map={"1": {"family": "physics", "family_weight": 1.}})
    assert roster["weight_origin"] == "member-map"
    assert roster["members"][0]["family_origin"] == "member-map"


def test_run_directory_ignores_hidden_resume_manifest_copies(tmp_path):
    domain = tmp_path / "d01"
    domain.mkdir()
    document = {"member_order": [1], "member_metadata": [{"member_id": 1}],
                "member_files": [{"member_id": 1, "path": "wrfout_d01_2022-06-13_23_00_00"}]}
    (domain / "ensemble-manifest.json").write_text(json.dumps(document))
    hidden = tmp_path / ".ensemble-resume" / "generations" / "old" / "d01"
    hidden.mkdir(parents=True)
    (hidden / "ensemble-manifest.json").write_text(json.dumps(document))
    assert _input_frames(tmp_path) == _input_frames(domain / "ensemble-manifest.json")


def _hybrid_retained_manifest(tmp_path):
    """Receipt control only: no NetCDF weather data or forecast is generated."""
    import hashlib
    order = list(range(36))
    histories = [{"member_id": member, "domain": "d01",
                  "path": f"m{member}/wrfout_d01_2022-06-13_18_00_00"} for member in order]
    histories.append({"member_id": 0, "domain": "d01",
                      "path": "m0/wrfout_d01_2022-06-13_19_00_00"})
    frames = []
    for hour in range(19, 25):
        valid = f"2022-06-13T{hour:02}:00:00" if hour < 24 else "2022-06-14T00:00:00"
        packs = []
        for start in (0, 18):
            members = order[start:start+18]
            payload = json.dumps({"valid_time": valid, "member_ids": members}).encode()
            path = tmp_path / f"hour-{hour}-members-{start}.pack"
            path.write_bytes(payload)
            packs.append({"path": path.name, "member_ids": members, "bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest()})
        frames.append({"valid_time": valid, "domain": "d01", "diagnostic_files": packs})
    receipt = {"member_order": order, "member_metadata": [{"member_id": member} for member in order],
               "member_files": histories, "frames": frames}
    manifest = tmp_path / "ensemble-manifest.json"
    manifest.write_text(json.dumps(receipt))
    return manifest, receipt


def test_retained_histories_do_not_hide_later_diagnostic_hours(tmp_path):
    manifest, receipt = _hybrid_retained_manifest(tmp_path)
    _, _, _, frames = _input_frames(manifest)
    assert len(frames) == 7
    assert all(set(rows) == set(range(36)) for rows in frames.values())
    assert frames[("d01", "2022-06-13T19:00:00")][0][0] == "history"
    assert frames[("d01", "2022-06-13T19:00:00")][1] == (
        "pack", (tmp_path / receipt["frames"][0]["diagnostic_files"][0]["path"], 1))
    assert all(item[0] == "pack" for item in frames[("d01", "2022-06-14T00:00:00")].values())


@pytest.mark.parametrize("defect", ["hash", "duplicate"])
def test_hybrid_manifest_keeps_diagnostic_identity_and_roster_checks(tmp_path, defect):
    manifest, receipt = _hybrid_retained_manifest(tmp_path)
    first = receipt["frames"][0]["diagnostic_files"][0]
    if defect == "hash":
        (tmp_path / first["path"]).write_bytes(b"changed after receipt")
        message = "committed hash"
    else:
        receipt["frames"][0]["diagnostic_files"].append(first)
        manifest.write_text(json.dumps(receipt))
        message = "duplicate diagnostic member"
    with pytest.raises(ValueError, match=message):
        _input_frames(manifest)


def test_gpu_family_outputs_preserve_input_and_weighted_mean():
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("requires admitted GPU")
    import cupy as cp
    from gpuwm.ensemble.family_products import reduce_family_frame
    values = cp.asarray([[[0, 20], [30, 40]], [[40, 40], [50, 60]], [[60, 60], [60, 60]]], dtype=cp.float32)
    before = values.get().tobytes()
    roster = family_roster([{"member_id": i, "family": family, "family_weight": weight}
                            for i, (family, weight) in enumerate((("ai", .75), ("physics", .25), ("physics", .25)))])
    means = {}
    for name, products, attrs in reduce_family_frame({"refl": values}, (FieldProducts("refl", "dBZ", (30, 40)),),
                                                     roster, available_bytes=64 << 20):
        means[name] = products.outputs["refl:mean"].get()
        assert np.isfinite(products.outputs["refl:pmm"].get()).all()
        assert json.loads(attrs["family_weights"]) == {"ai": .75, "physics": .25}
    np.testing.assert_array_equal(means["combined"], .75 * values[0].get() + .25 * values[1:].get().mean(axis=0))
    assert before == values.get().tobytes()


@pytest.mark.parametrize("transport", ["mass", "native", "mass-multigrid"])
def test_gpu_postrun_native_render_and_score_pipeline(tmp_path, monkeypatch, transport):
    """Synthetic transport/control proof, never a weather skill measurement."""
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("requires admitted GPU")
    import os
    from pathlib import Path
    from gpuwm import rustwx
    from gpuwm.obs import mrms_fetch, obspack
    from gpuwm.ensemble.batch_product_output import NativeDiagnosticSpool
    from gpuwm.ensemble.family_products import run_products
    from test_obs_battery_ingest import build_pack, grid_pack_bytes
    from PIL import Image
    renderer = rustwx.find_ensemble_product_renderer(os.environ.get("GPUWM_ENSEMBLE_RENDERER"))
    assert renderer is not None
    shape = (11, 13)
    lat = np.broadcast_to(np.linspace(35., 35.27, shape[0], dtype=np.float32)[:, None], shape).copy()
    lon = np.broadcast_to(np.linspace(-99., -98.60, shape[1], dtype=np.float32), shape).copy()
    j, i = np.indices(shape)
    observed = (60 * np.exp(-((i - 6)**2 + (j - 5)**2) / 12)).astype(np.float32)
    fields = {"refl": np.stack([observed, np.roll(observed, 2, axis=1)]),
              "uh": np.stack([observed * 3, observed * 2]),
              "rain_total": np.stack([observed, observed / 2])}
    members = [{"member_id": index, "provenance": {"parent_family": family, "family_weight": weight}}
               for index, (family, weight) in enumerate((("ai", .75), ("physics", .25)))]
    valid = "2022-06-13T19:00:00"
    if transport.startswith("mass"):
        for index, row in enumerate(members):
            path = tmp_path / f"member-{index}.npz"
            np.savez(path, **{key: value[index] for key, value in fields.items()}, latitude=lat, longitude=lon)
            row["frames"] = [{"grid_id": 1, "episode": 0, "valid_time": valid, "path": str(path)}]
            if transport == "mass-multigrid":
                second = tmp_path / f"member-{index}-grid2.npz"
                np.savez(second, **{key: value[index] for key, value in fields.items()}, latitude=lat+1, longitude=lon+1)
                row["frames"].append({"grid_id": 2, "episode": 0, "valid_time": valid, "path": str(second)})
        run = tmp_path / "ensemble-set.json"
        run.write_text(json.dumps({"completed": members}))
    else:
        import cupy as cp
        from gpuwm.ensemble.batch_products import default_product_requests
        requests, _ = default_product_requests(fields, postage_stamp=False)
        spool = NativeDiagnosticSpool(tmp_path / "native", members=2, requests=requests,
            latitude=lat, longitude=lon, renderer=renderer, member_metadata=members, export_coordinates=True)
        spool.submit({key: cp.asarray(value) for key, value in fields.items()}, member_ids=(0, 1), valid_time=valid)
        run = spool.manifest_path
    geo = tmp_path / "geometry.obspack"
    geo.write_bytes(build_pack({"schema": obspack.GEO_SCHEMA, "status": "READY"},
                              {"latitude": lat, "longitude": lon}))
    second_geo = tmp_path / "geometry2.obspack"
    second_geo.write_bytes(build_pack({"schema": obspack.GEO_SCHEMA, "status": "READY"},
                                     {"latitude": lat+1, "longitude": lon+1}))
    scan = tmp_path / "fixture.obspack"
    scan.write_bytes(grid_pack_bytes(valid_time=valid, values=observed))
    cache_roots = {}
    class FixtureCache:
        def __init__(self, *args, bbox, **kwargs):
            cache_roots[bbox] = Path(args[0])
            self.geometry_path = geo if bbox == mrms_fetch.bbox_around(lat, lon) else second_geo
        def ensure(self, requested):
            assert requested == valid
            return SimpleNamespace(pack_path=scan, record=lambda: {"is_stub": True, "fixture": "synthetic control"})
    monkeypatch.setattr(mrms_fetch, "MrmsCompositeCache", FixtureCache)
    result = run_products(run, tmp_path / "out", renderer=renderer)
    count = 2 if transport == "mass-multigrid" else 1
    assert len(result["frames"]) == count and len(result["scores"]) == 60 * count
    assert all(row["scored_cells"] == observed.size for row in result["scores"])
    if count == 2:
        assert len(set(cache_roots.values())) == 2
        assert len({path.parent for path in cache_roots.values()}) == 1
    assert result["frames"][0]["status"] == "scored"
    assert set(result["frames"][0]["products"]["combined"]) == set(fields)
    assert (tmp_path / "out" / "scores.csv").is_file()
    from gpuwm.netcdf_bridge import open_dataset
    with open_dataset(result["frames"][0]["products"]["combined"]["refl"]) as dataset:
        np.testing.assert_allclose(dataset.variables["refl_mean"][:], .75 * fields["refl"][0] + .25 * fields["refl"][1], atol=1e-5)
    for sheet in result["frames"][0]["sheets"]:
        assert Path(sheet).is_file()
        with Image.open(sheet) as picture:
            assert picture.width >= 3200 and picture.height >= 3600
