from pathlib import Path

import pytest

from gpuwm.soundings import inventory, point
from gpuwm.ensemble.request import EnsembleRequest


def test_plain_ensemble_inventory_ignores_aggregate_and_render(tmp_path):
    for name in ("members/m000/d01/wrfout_d01_2026-01-01_00_00_00",
                 "members/m001/d01/wrfout_d01_2026-01-01_00_00_00",
                 "mean/d01/wrfout_d01_2026-01-01_00_00_00",
                 ".ensemble-resume/generations/1/members/m000/d01/wrfout_d01_2026-01-01_00_00_00"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    rows = inventory(tmp_path)
    assert {(r["member"], r["domain"]) for r in rows} == {("m000", "d01"), ("m001", "d01")}


def test_store_inventory_and_station_catalogue(tmp_path):
    path = tmp_path / "d02-3km" / "f000.rws"
    path.parent.mkdir()
    path.touch()
    assert inventory(tmp_path)[0]["domain"] == "d02-3km"
    assert point("TEST", {"TEST": [35, -97]}) == {"label": "TEST", "lat": 35., "lon": -97.}
    assert point("site=35,-97")["label"] == "site"
    with pytest.raises(ValueError):
        point("91,0")


def test_ensemble_sounding_product_keeps_profiles_and_positional_api():
    assert EnsembleRequest(2, True).keep_member_files
    request = EnsembleRequest.from_mapping({"members": 2, "sounding_points": ["35,-97"]})
    assert request.keep_member_files
    assert request.receipt()["sounding_points"] == ["35,-97"]


@pytest.mark.parametrize("options", [
    {"sounding_points": [123]},
    {"sounding_points": ["35,-97"], "keep_member_files": "yes"},
])
def test_invalid_sounding_request_values(options):
    with pytest.raises(ValueError):
        EnsembleRequest.from_mapping({"members": 2, **options})


def test_public_sounding_member_listing_and_legend_keep_family_and_model(tmp_path):
    import json
    from gpuwm.soundings import member_listing, lineage_viewer
    from gpuwm.ensemble.lineage import build_lineage
    records = [build_lineage({"member": i, "source": source, "cycle": "2022-06-13T00", "seed": i})
               for i, source in enumerate(("earth2-fcn3", "gefs"))]
    (tmp_path / "ensemble-run.json").write_text(json.dumps({"request": {"provenance": records}}))
    listing = member_listing(tmp_path)
    assert listing[0]["label"] == "member-0000 (ai: earth2-fcn3)"
    assert listing[1]["label"] == "member-0001 (physics: gefs)"
    output = tmp_path / "render"
    output.mkdir()
    pictures = []
    for mode in ("all", "member-0000", "member-0001"):
        path = output / (mode + ".png")
        path.with_suffix(".json").write_text(json.dumps({"mode": mode, "valid": "2022-06-13T06:00Z", "members": [{"member": "member-0000"}]}))
        pictures.append({"png": str(path)})
    (output / "soundings-manifest.json").write_text(json.dumps({"pictures": pictures}))
    lineage_viewer(output, listing)
    page = (output / "index.html").read_text()
    assert 'id="member-picker"' in page
    assert "ai: earth2-fcn3" in page and "physics: gefs" in page
    data = json.loads((output / "all.json").read_text())
    assert data["members"][0]["provenance"] == records[0]
    assert json.loads((output / "soundings-manifest.json").read_text())["member_listing"] == listing
