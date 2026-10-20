"""The shared heating evaluation seam also orchestrates native family products."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tools.heat_eval import score_case, sheets
from tools import ensemble_reflectivity_panels as compatibility


def test_family_frame_uses_native_matcher_and_refuses_wrong_result(tmp_path, monkeypatch):
    request = {"quantity": "composite_reflectivity", "valid_time": "2022-06-13T23:00:00Z"}
    receipt = {**request, "schema": "gpuwm-ensemble-grid-match.v1", "status": "complete"}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert json.loads(Path(command[2]).read_text()) == request
        return SimpleNamespace(stdout=json.dumps(receipt))

    monkeypatch.setattr(score_case.subprocess, "run", run)
    arguments = dict(matcher=tmp_path / "matcher", request_path=tmp_path / "request.json",
                     product_dir=tmp_path / "products")
    assert score_case.score_family_frame(request, **arguments) == receipt
    assert calls == [[str(arguments['matcher']), "match-grid", str(arguments['request_path']),
                      str(arguments['product_dir'])]]
    receipt["valid_time"] = "2022-06-14T00:00:00Z"
    with pytest.raises(ValueError, match="did not complete"):
        score_case.score_family_frame(request, **arguments)
    receipt["valid_time"] = request["valid_time"]
    receipt["quantity"] = "temperature_2m"
    with pytest.raises(ValueError, match="did not complete"):
        score_case.score_family_frame(request, **arguments)


def test_family_sheet_routes_native_products_and_restores_renderer(tmp_path, monkeypatch):
    """Only complete native PNGs reach the compositor, with family order fixed."""
    products = []
    for family in ("physics", "observed"):
        source = tmp_path / f"{family}.nc"
        source.write_bytes(family.encode())  # Orchestration fixture, not a weather array.
        products.append({"family": family, "path": str(source),
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "products": ["prob", "postage"] if family == "physics" else ["mean"]})
    binary = tmp_path / "rw_wrfbatch"
    binary.write_bytes(b"renderer fixture")
    receipt = {"schema": "gpuwm-ensemble-grid-match.v1", "status": "complete",
               "valid_time": "2022-06-13T23:00:00Z", "products": products}
    monkeypatch.setenv("GPUWM_ENSEMBLE_RENDERER", "previous")
    calls = []

    def render(renderer, source, **kwargs):
        import os
        assert os.environ["GPUWM_ENSEMBLE_RENDERER"] == str(binary)
        calls.append((source.stem, kwargs["products"]))
        out = kwargs["out_dir"]
        out.mkdir(parents=True)
        name = "ens_prob_refl_ge_35" if source.stem == "physics" else "ens_mean_refl"
        path = out / f"{name}.png"
        Image.new("RGB", (960, 720), "blue" if source.stem == "physics" else "red").save(path)
        return [path], "native fixture"

    monkeypatch.setattr(sheets, "run_ensemble_product_renderer", render)
    assert compatibility.render_panels is sheets.draw_family_panels
    result = sheets.draw_family_panels(receipt, renderer=binary, out=tmp_path / "sheets")
    assert sorted(calls) == [("observed", ("mean",)), ("physics", ("prob", "postage"))]
    import os
    assert os.environ["GPUWM_ENSEMBLE_RENDERER"] == "previous"
    with Image.open(result["comparison"]["path"]) as image:
        assert image.size == (2880, 760)
        assert image.getpixel((30, 100)) == (255, 0, 0)  # MRMS left
        assert image.getpixel((1950, 100)) == (0, 0, 255)  # physics right
        assert image.getpixel((990, 100)) == (238, 240, 243)  # absent AI blank
    products[0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="native product changed"):
        sheets.draw_family_panels(receipt, renderer=binary, out=tmp_path / "refused")
    assert os.environ["GPUWM_ENSEMBLE_RENDERER"] == "previous"
