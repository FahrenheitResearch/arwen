"""Cache integrity, clock and spatial provenance without model execution."""
import hashlib
import json
from pathlib import Path

import pytest

from gpuwm import earth2_fetch_cache as cache, earth2_run
from gpuwm.earth2_door import build_spec


def test_receipt_cache_checks_identity_coverage_and_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_EARTH2_FETCH_CACHE", str(tmp_path / "index"))
    monkeypatch.delenv("GPUWM_EARTH2_FETCH_RECEIPTS", raising=False)
    monkeypatch.setattr(earth2_run, "publish_handoffs", lambda receipt, out: None)
    spec = build_spec(model="fcn3", cycle="2024-09-24T12", hours=24, storm="auto", nest="2.5km")
    producer = tmp_path / "producer"
    producer.mkdir()
    files = []
    for lead in range(0, 49, 6):
        path = producer / f"frame-{lead}.nc"
        path.write_bytes(str(lead).encode())
        files.append(dict(path=str(path), member=0, lead_hours=lead,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    donor = producer / "donor.nc"
    donor.write_bytes(b"donor")
    parent = producer / "earth2-run.json"
    receipt = dict(schema="gpuwm-earth2-fetch-v2", model="fcn3", cycle=spec["cycle"],
                   model_row=earth2_run.resolve_model("fcn3")[1],
                   initial_condition_source="era5", members=[0], forecast_start_hour=0,
                   model_table_sha256="older-table-with-identical-model-row", hours=48,
                   runner=dict(files=files), donor=dict(supplements={"surface": str(donor)}))
    parent.write_text(json.dumps(receipt))
    request = producer / "earth2-request.json"
    request.write_text(json.dumps(dict(area=None)))
    assert cache.reuse(spec, tmp_path / "miss") is None
    monkeypatch.setenv("GPUWM_EARTH2_FETCH_RECEIPTS", str(parent))
    reused = cache.reuse(spec, tmp_path / "shared")
    assert reused["model_table_sha256"] == "older-table-with-identical-model-row"
    assert reused["hours"] == 24
    assert len(reused["forecast"]) == 5
    assert reused["cache_reuse"]["parent_sha256"] == hashlib.sha256(parent.read_bytes()).hexdigest()
    cache.remember(spec, parent)
    donor.write_bytes(b"modified donor")
    monkeypatch.delenv("GPUWM_EARTH2_FETCH_RECEIPTS")
    with pytest.raises(ValueError, match="donor or forecast hash differs"):
        cache.reuse(spec, tmp_path / "changed-donor")
    donor.write_bytes(b"donor")
    Path(files[0]["path"]).write_bytes(b"modified forecast")
    with pytest.raises(ValueError, match="forecast hash differs"):
        cache.reuse(spec, tmp_path / "changed-frame")
    Path(files[0]["path"]).write_bytes(b"0")
    request.write_text(json.dumps(dict(area=[10, 270, 30, 280])))
    with pytest.raises(ValueError, match="donor or forecast hash differs"):
        cache.reuse(spec, tmp_path / "cropped")
    donor.unlink()
    assert cache.reuse(spec, tmp_path / "retired") is None


def test_legacy_cache_checks_defaults_and_native_clock_without_relabelling(tmp_path, monkeypatch):
    from gpuwm import earth2_legacy_cache, tc_tracks
    monkeypatch.setenv('GPUWM_EARTH2_FETCH_CACHE', str(tmp_path / 'index'))
    spec = build_spec(model='fcn3', cycle='2024-09-24T12', hours=24, storm='auto', nest='2.5km')
    row = earth2_run.resolve_model('fcn3')[1]
    producer = tmp_path / 'legacy'
    producer.mkdir()
    forecast, analysis = producer / 'forecast.nc', producer / 'analysis.nc'
    forecast.write_bytes(b'forecast')
    analysis.write_bytes(b'analysis')
    old_row = {key: row[key] for key in ('earth2studio_class', 'profile', 'step_hours')}
    receipt = dict(schema='gpuwm-earth2-fetch-v1', model='fcn3', model_row=old_row,
        initial_condition_source='era5', cycle=spec['cycle'], hours=72,
        forecast=str(forecast), analysis=str(analysis), profile=row['profile'],
        runner={'earth2studio':'0.19.0'}, wall_seconds=100.)
    parent = producer / 'earth2-run.json'
    parent.write_text(json.dumps(receipt))
    (producer / 'earth2-request.json').write_text(json.dumps(dict(area=None,
        model_class=row['earth2studio_class'], ic_class='ARCO', steps=12)))
    monkeypatch.setenv('GPUWM_EARTH2_FETCH_RECEIPTS', str(parent))
    monkeypatch.setattr(earth2_legacy_cache, 'default_identity', lambda *args: {'seed':333, 'package':row['weights']['uri']})
    metadata = {'variables':{
        'time':{'values':[0], 'attrs':{'units':'hours since 2024-09-24T12:00:00+00:00'}},
        'lead_time':{'values':list(range(0,73,6)), 'attrs':{'units':'hours'}},
        'lat':{'values':[90,-90]}, 'lon':{'values':[0,359.75]}}}
    monkeypatch.setattr(tc_tracks, 'inventory', lambda *args: metadata)
    cache.remember(spec, parent)
    result = cache.reuse(spec, tmp_path / 'reused')
    assert result['schema'] == 'gpuwm-earth2-fetch-v1' and result['model_row'] == old_row
    assert result['legacy_compatibility']['checked_leads_hours'] == [0,6,12,18,24]
    handoff = json.loads((tmp_path / 'reused/prep-arguments.json').read_text())
    assert handoff['argv'].count(str(forecast)) == 1 and handoff['unbound_supplement_roles'] == []
    analysis.write_bytes(b'changed')
    with pytest.raises(ValueError, match='donor or forecast hash differs'):
        cache.reuse(spec, tmp_path / 'changed')
    metadata['variables']['lead_time']['values'] = [0,12,24]
    with pytest.raises(ValueError, match='lead ladder differs'):
        cache.reuse(spec, tmp_path / 'missing')


def test_copied_receipt_rebases_only_retired_paths_and_preserves_hash_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_EARTH2_FETCH_CACHE", str(tmp_path / "index"))
    monkeypatch.setattr(earth2_run, "publish_handoffs", lambda receipt, out: None)
    spec = build_spec(model="fcn3", cycle="2024-09-24T12", hours=6, storm="auto", nest="2.5km")
    copied = tmp_path / "copied"
    copied.mkdir()
    old = tmp_path / "retired"
    entries = []
    for lead in (0, 6):
        data = str(lead).encode()
        (copied / f"frame-{lead}.nc").write_bytes(data)
        entries.append(dict(path=str(old / f"frame-{lead}.nc"), member=0, lead_hours=lead,
                            sha256=hashlib.sha256(data).hexdigest()))
    (copied / "analysis").mkdir()
    (copied / "analysis/donor.nc").write_bytes(b"analysis")
    parent = copied / "earth2-run.json"
    parent.write_text(json.dumps(dict(schema="gpuwm-earth2-fetch-v2", model="fcn3", cycle=spec["cycle"],
        model_row=earth2_run.resolve_model("fcn3")[1], initial_condition_source="era5", members=[0],
        forecast_start_hour=0, model_table_sha256=spec["table"]["sha256"], hours=6,
        runner=dict(files=entries), donor=dict(supplements={"surface":str(old / "analysis/donor.nc")}))))
    (copied / "earth2-request.json").write_text(json.dumps(dict(area=None, output=str(old / "forecast.nc"))))
    original = parent.read_bytes()
    monkeypatch.setenv("GPUWM_EARTH2_FETCH_RECEIPTS", str(parent))
    reused = cache.reuse(spec, tmp_path / "job")
    assert reused["forecast"] == [str(copied / f"frame-{lead}.nc") for lead in (0, 6)]
    assert reused["artifact_relocation"]["paths"][str(old / "analysis/donor.nc")] == str(copied / "analysis/donor.nc")
    assert parent.read_bytes() == original
    (copied / "frame-6.nc").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="forecast hash differs"):
        cache.reuse(spec, tmp_path / "corrupt")


def test_explicit_input_checks_coverage_and_never_falls_back_to_inference(tmp_path, monkeypatch):
    from gpuwm import earth2_job
    monkeypatch.setenv("GPUWM_EARTH2_FETCH_CACHE", str(tmp_path / "index"))
    monkeypatch.setattr(earth2_run, "publish_handoffs", lambda receipt, out: None)
    spec = build_spec(model="fcn3", cycle="2024-09-24T12", hours=24, storm="auto", nest="2.5km")
    producer = tmp_path / "regional"
    producer.mkdir()
    files = []
    for lead in range(0, 73, 6):
        frame = producer / f"f{lead}.nc"
        frame.write_bytes(str(lead).encode())
        files.append(dict(path=str(frame), member=0, lead_hours=lead,
                          sha256=hashlib.sha256(frame.read_bytes()).hexdigest()))
    donor = producer / "analysis.nc"
    donor.write_bytes(b"donor")
    parent = producer / "earth2-run.json"
    receipt = dict(schema="gpuwm-earth2-fetch-v2", model="fcn3", cycle=spec["cycle"],
        model_row=earth2_run.resolve_model("fcn3")[1], initial_condition_source="era5",
        members=[0], forecast_start_hour=0, hours=72, runner=dict(files=files),
        donor=dict(supplements={"surface": str(donor)}))
    parent.write_text(json.dumps(receipt))
    original = parent.read_bytes()
    (producer / "earth2-request.json").write_text(json.dumps(dict(area=[0, 230, 55, 340])))
    # A regional superset covers the configured search in either cache form.
    monkeypatch.setenv("GPUWM_EARTH2_FETCH_RECEIPTS", str(parent))
    implicit = cache.reuse(spec, tmp_path / "implicit")
    reused = cache.reuse(spec, tmp_path / "explicit", producer)
    coverage = reused["cache_reuse"]["coverage"]
    assert coverage["search_region"]["id"] == "forecast-area"
    assert coverage["search_region"]["area"] == [0, 230, 55, 340]
    assert implicit["cache_reuse"]["coverage"] == coverage
    assert coverage["search_region"]["basis"] == "receipt-checked regional forecast area"
    assert len(reused["forecast"]) == 5 and reused["hours"] == 24
    assert parent.read_bytes() == original
    config = tmp_path / "cyclone.toml"
    config.write_text('[fetch]\narea = "15,-92,30,-80"\n')
    cache.check_domain_coverage(reused, config)
    assert reused["cache_reuse"]["coverage"]["checked_domain_area"] == [15, -92, 30, -80]
    config.write_text('[fetch]\narea = "15,-140,30,-80"\n')
    with pytest.raises(ValueError, match="does not cover.*domain/corridor"):
        cache.check_domain_coverage(reused, config)
    assert cache.covers_area([0, 230, 55, 340], [15, -92, 30, -80])
    assert cache.covers_area([-90, 0, 90, 360], [-90, -180, 90, 180])
    assert cache.covers_area([0, 170, 55, 210], [5, 175, 30, -160])
    assert not cache.covers_area([0, 170, 55, 210], [5, -160, 30, 175])
    saved = tmp_path / "spec.json"
    saved.write_text(json.dumps(spec))
    def no_inference(**kwargs):
        pytest.fail("an incompatible explicit fetch must not start a model")
    monkeypatch.setattr(earth2_run, "run_model", no_inference)
    request = producer / "earth2-request.json"
    request.write_text(json.dumps(dict(area=[15, -90, 30, -80])))
    smaller = cache.reuse(spec, tmp_path / "smaller", producer)
    assert smaller["cache_reuse"]["coverage"]["search_region"]["area"] == [15, -90, 30, -80]
    with pytest.raises(ValueError, match="does not cover.*domain/corridor"):
        cache.check_domain_coverage(smaller, config)
    request.write_text(json.dumps(dict(area=[0, 230, 55, 340])))
    for key, invalid, message in (("model", "another-model", "model differs"),
                                  ("cycle", "2024-09-25T12:00:00Z", "cycle differs"),
                                  ("hours", 12, "shorter")):
        parent.write_text(json.dumps(dict(receipt, **{key: invalid})))
        with pytest.raises(ValueError, match=message):
            earth2_job._fetch(saved, tmp_path / key, producer)
    parent.write_bytes(original)


def test_search_policy_is_table_data_and_explicit_area_wins(tmp_path, monkeypatch):
    from gpuwm import cyclone_search
    policy = tmp_path / "cyclone-search.v1.json"
    policy.write_text(json.dumps(dict(default_auto_region="another-basin", regions={
        "another-basin": dict(area=[-30, 170, 30, -150], description="wrapped search")})))
    monkeypatch.setattr(cyclone_search, "TABLE", policy)
    spec = build_spec(model="fcn3", cycle="2024-09-24T12", storm="auto", nest="2.5km")
    region = cache._identity(spec)["search_region"]
    assert region["area"] == [-30, 170, 30, -150]
    assert region["id"] == "another-basin"
    assert not bool(cyclone_search.window(region["area"]).outside(10, -175))
    explicit = cyclone_search.resolve_region([10, -90, 30, -70], automatic=True)
    assert explicit["id"] == "job-area" and "authority" not in explicit


def test_short_simulation_acquires_the_surrounding_model_boundaries(tmp_path, monkeypatch):
    from gpuwm import earth2_job
    spec = build_spec(model="fcn3", cycle="2024-09-24T12", hours=1,
                      storm="auto", nest="2.5km")
    saved = tmp_path / "spec.json"
    saved.write_text(json.dumps(spec))
    calls = []
    monkeypatch.setattr(cache, "reuse", lambda job, out, input_dir: calls.append(job) or {})
    earth2_job._fetch(saved, tmp_path / "output", tmp_path / "completed")
    assert calls[0]["hours"] == earth2_run.resolve_model(spec["model"])[1]["step_hours"]
    assert json.loads(saved.read_text())["hours"] == 1
