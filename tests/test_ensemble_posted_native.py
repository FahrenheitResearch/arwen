"""Common native preparation is shared only when its actual inputs agree."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.posted_native import (
    checked_source_inputs, copy_common_artifacts, relay_source_segment, shared_surface,
    writer_source_wait,
)
from test_ensemble_physical_preparation import snapshot


@pytest.fixture
def prepared():
    met = snapshot()
    met = replace(met, fields={**met.fields, "SKINTEMP": np.full((2, 3), 280., dtype="f4")})
    surface = {"TSLB": np.full((4, 2, 3), 283., dtype="f4"),
               "SMOIS": np.full((4, 2, 3), .2, dtype="f4")}
    # The native prepared head intentionally drops raw soil inputs once the
    # solved canonical surface is written. Its remaining met arrays still
    # have to match the separately captured complete physical frame exactly.
    retained = set(met.fields)-{"ST000010"}
    arrays = {"met/"+key: met.fields[key] for key in retained}
    arrays.update({"surface/"+key: value for key, value in surface.items()})
    repairs = {"repaired_land_columns": 1, "bounding_box": {"rows": [0, 0], "columns": [1, 1]}}
    checked = SimpleNamespace(proof={"soil_texture_downscale": {"enabled": True}},
        cache_reader=SimpleNamespace(
            header={"metadata": {"met_fields": sorted(retained), "surface_fields": sorted(surface),
                                  "user": {"soil_temperature_repair": repairs}}},
            read_array=lambda key: arrays[key].copy()))
    return checked, met, surface, repairs


def test_canonical_surface_reused_for_changed_atmosphere_and_native_repairs(prepared):
    checked, base, expected, repairs = prepared
    member = replace(base, fields={**base.fields, "TT": base.fields["TT"]+1.,
                                   "PSFC": base.fields["PSFC"]+25.})
    result = shared_surface(checked, base_met=base, member_met=member)
    assert all(result.fields[key].tobytes() == value.tobytes() for key, value in expected.items())
    assert not result.fields["TSLB"].flags.writeable
    assert result.soil_texture_downscale == {"enabled": True}
    assert result.soil_temperature_repair == repairs
    result.soil_temperature_repair["repaired_land_columns"] = 99
    assert repairs["repaired_land_columns"] == 1


@pytest.mark.parametrize("field", ["ST000010", "SKINTEMP", "SOURCE_OROGRAPHY"])
def test_changed_surface_ingredients_cannot_reuse_ordinary_solved_surface(prepared, field):
    checked, base, *_ = prepared
    member = replace(base, fields={**base.fields, field: base.fields[field]+1.})
    with pytest.raises(ValueError, match="shared native surface input "+field):
        shared_surface(checked, base_met=base, member_met=member)


@pytest.mark.parametrize("kind", ["array", "receipt", "native_base"])
def test_shared_surface_keeps_water_and_captured_head_authority(prepared, kind):
    checked, base, *_ = prepared
    member = replace(base)
    if kind == "array":
        member = replace(member, water_temperature=np.full((2, 3), 285., "f4"))
    elif kind == "receipt":
        member = replace(member, water_temperature_receipt={"policy": "changed"})
    else:
        base = replace(base, fields={**base.fields, "TT": base.fields["TT"]+1.})
        member = replace(base)
    with pytest.raises(ValueError, match="shared native surface"):
        shared_surface(checked, base_met=base, member_met=member)


def test_checked_source_inputs_uses_original_cadence_and_head_pin(monkeypatch, tmp_path):
    from gpuwm import experiment, prepared_single_domain_forecast
    from datetime import datetime, timedelta, timezone
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    calls = []
    context = SimpleNamespace(prepared_root=tmp_path, prepared_head={"head_sha256": "a"*64},
        physical_stream=SimpleNamespace(times=(start, start+timedelta(hours=6))),
        verify=lambda: calls.append("verify"))
    monkeypatch.setattr(experiment, "load_experiment", lambda _path:
                        SimpleNamespace(root=SimpleNamespace(history_interval_s=900.), run_seconds=19800.))
    monkeypatch.setattr(prepared_single_domain_forecast, "preflight_prepared_forecast", lambda **kwargs: kwargs)
    result = checked_source_inputs(context, source="gfs", experiment_config=tmp_path/"original.toml",
                                   wps_namelist=tmp_path/"original.wps")
    assert result["history_interval_seconds"] == 900.
    assert result["run_seconds"] == 19800.
    assert result["source_manifest_sha256"] is None
    assert result["prepared_head_sha256"] == "a"*64
    assert calls == ["verify", "verify"]


def test_shared_artifacts_are_checked_then_linked_without_geography_work(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    static = source/"native-static.npz"
    static.write_bytes(b"verified static bytes")
    geometry = source/"native-geometry-receipt.json"
    receipt = {"projection": "verified", "static_sha256": digest_file(static)}
    geometry.write_text(json.dumps(receipt))
    calls = []
    context = SimpleNamespace(verify=lambda: calls.append("verify"))
    checked = SimpleNamespace(static_path=static, geometry_receipt_path=geometry,
                              geometry_receipt=receipt, cache_identity={"static_cache_sha256": digest_file(static)})
    result = copy_common_artifacts(context, checked, tmp_path/"member", geometry_name=geometry.name)
    assert result["static"]["path"].read_bytes() == static.read_bytes()
    assert result["geometry"]["path"].name == geometry.name
    assert calls == ["verify", "verify"]
    with pytest.raises(ValueError, match="within the member"):
        copy_common_artifacts(context, checked, tmp_path/"unsafe", geometry_name="../outside")
    static.write_bytes(b"replaced static")
    with pytest.raises(ValueError, match="changed after"):
        copy_common_artifacts(context, checked, tmp_path/"drift")


def test_native_segment_relays_raw_source_authority_through_ordinary_writer():
    marker = {"index": 3, "posted_leads": {"3": "a"*64, "4": "b"*64}}
    seen = []
    context = SimpleNamespace(require_interval=lambda index: marker if index == 3 else None)
    writer = SimpleNamespace(write_segment=lambda index, interval, **kwargs: seen.append((index, interval, kwargs)))
    relay_source_segment(context, 3, writer, "member interval")
    assert seen == [(3, "member interval", {"relay_marker": marker})]


def test_writer_wait_tracks_source_to_preparation_and_arrival():
    seen = []
    writer = SimpleNamespace(check_stop=lambda: seen.append("stop"),
        waiting_for_source=lambda cause: seen.append(("source", cause)),
        source_arrived=lambda: seen.append("producing"))
    observe = writer_source_wait(writer)
    cause = {"source": "gfs", "lead": 3}
    observe({"cause": cause})
    observe({"cause": None})
    observe(None)
    assert seen == ["stop", ("source", cause), "stop", "producing", "stop", "producing"]
