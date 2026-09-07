"""A caller's explicit no-downscaling choice must not become autodetection."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from gpuwm.ingest import hrrr_physics, soil_downscale


@pytest.mark.parametrize("choice", ["omitted", "disabled", "supplied", "prepared"])
def test_actual_hrrr_initializer_preserves_the_soil_mesh_choice(monkeypatch, choice):
    planned = object()
    explicit = object()
    surface = object()
    seen = []
    def plan(*args, **kwargs):
        seen.append("plan")
        return planned
    def resolve(met, cfg, static, **kwargs):
        seen.append(kwargs["soil_mesh"])
        return surface
    monkeypatch.setattr(soil_downscale, "soil_mesh_plan_from_case", plan)
    monkeypatch.setattr(hrrr_physics, "resolve_prepared_noah_surface", resolve)
    monkeypatch.setattr(hrrr_physics, "initialize_prepared_physics", lambda *a, **k: "initialized")
    options = ({"soil_mesh": None} if choice == "disabled" else
               {"soil_mesh": explicit} if choice == "supplied" else
               {"surface": surface} if choice == "prepared" else {})
    attrs = {"MMINLU": "USGS", "ISWATER": 16, "ISLAKE": 28, "ISICE": 24, "CEN_LAT": 35.}
    assert hrrr_physics.initialize_hrrr_physics(
        object(), SimpleNamespace(), object(), {}, attrs, object(), datetime(2026, 1, 1),
        **options) == "initialized"
    assert seen == (["plan", planned] if choice == "omitted" else
                    [explicit] if choice == "supplied" else [None])
