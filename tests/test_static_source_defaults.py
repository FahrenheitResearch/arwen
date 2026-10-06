from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.static.source_defaults import (
    align_source_projection, with_source_static_defaults,
    with_source_static_defaults_text)


def test_undeclared_sources_keep_configuration_bytes_and_objects():
    text = '[experiment]\nname = "sample"\n'
    raw = {"experiment": {"name": "sample"}}
    for source in ("gfs", "era5", "rap", "rrfs"):
        assert with_source_static_defaults_text(text, source) is text
        assert with_source_static_defaults(raw, source) is raw
        from gpuwm.source_adapters import get_source_adapter
        declared = get_source_adapter(source).to_dict()
        assert "static_source" not in declared
        assert "runtime_surface_fields" not in declared


def test_metadata_default_preserves_an_explicit_field_subset():
    raw = {"static": {"source": "hrrr-conus-v4", "source_fields": ["soil"]}}
    assert with_source_static_defaults(raw, "hrrr") is raw
    assert with_source_static_defaults({}, "hrrr")["static"] == {
        "source": "hrrr-conus-v4"}


def test_source_grid_emission_is_a_whole_cell_window():
    from gpuwm.static.external_source import crop_window, source_grid, static_source_row
    projection = {"map_proj": "lambert", "ref_lat": 35.3, "ref_lon": -97.5,
                  "truelat1": 25.3, "truelat2": 45.3, "stand_lon": -97.5}
    assert align_source_projection(projection, (190, 152), 12000., "hrrr") is projection
    assert align_source_projection(projection, (190, 152), 3000., "gfs") is projection
    aligned = align_source_projection(projection, (190, 152), 3000., "hrrr")
    row = static_source_row("hrrr-conus-v4")
    geometry = {key: value for key, value in aligned.items() if key != "map_proj"}
    grid = type(source_grid(row))(**geometry, dx=3000., dy=3000., e_we=191, e_sn=153)
    assert crop_window(row, grid) == (804, 335)


def test_analyzed_vegetation_survives_runtime_selection_exactly():
    from gpuwm.ingest.hrrr_physics import initial_vegetation_fraction
    from gpuwm.ingest.vegetation import initial_vegetation_fraction as generic_selection
    assert initial_vegetation_fraction is generic_selection
    vegetation = np.array([[1.0, 50.0], [0.0, 100.0]], dtype=np.float32)
    static = {"LANDMASK": np.ones((2, 2)), "GREENFRAC": np.full((12, 2, 2), 0.25)}
    met = SimpleNamespace(fields={"VEGFRA": vegetation})
    assert initial_vegetation_fraction(met, static, datetime(2026, 10, 2)) is vegetation
    legacy = initial_vegetation_fraction(SimpleNamespace(fields={}), static, datetime(2026, 10, 2))
    np.testing.assert_array_equal(legacy, np.full((2, 2), 25.0))
    met.fields["VEGFRA"] = np.full((2, 2), 100.1)
    with pytest.raises(ValueError, match="percent"):
        initial_vegetation_fraction(met, static, datetime(2026, 10, 2))


def test_source_runtime_declaration_refuses_a_dropped_initial_field():
    from gpuwm.runtime_surface_fetch import require_runtime_surface_fields
    from gpuwm.source_adapters import get_source_adapter
    met = SimpleNamespace(fields={})
    require_runtime_surface_fields(met, get_source_adapter("gfs"))
    with pytest.raises(ValueError, match="different vegetated area"):
        require_runtime_surface_fields(met, get_source_adapter("hrrr"))
    met.fields["VEGFRA"] = np.zeros((2, 2))
    require_runtime_surface_fields(met, get_source_adapter("hrrr"))



SHIPPED_CONFIGS = Path(__file__).resolve().parent.parent / "configs"

#: Shipped configurations that name no static source and sit on their own
#: cone, so the hrrr-conus-v4 row their [fetch] source selects by default
#: is set aside and they build from WPS_GEOG (2.8.6 gate ruling, round 3:
#: a DEFAULTED source falls back on a projection mismatch, a DECLARED one
#: still refuses).  Pinned so a shipped configuration that drifts off the
#: row's cone, or onto it, is seen here rather than silently changing
#: which statics it runs on.
_DEFAULTED_ON_OWN_CONE = frozenset({
    "initdemo_bubble_off", "initdemo_bubble_on",
    "les_nest_250m_grayzone", "les_nest_250m_km3",
    "les_tornado_100m_dodgecity_20160524",
    "les_tornado_100m_mayfield_20211210",
    "les_tornado_100m_mayfield_20211210_attempt2",
    "les_tornado_100m_mayfield_20211210_attempt2b",
    "les_tornado_100m_mayfield_20211210_attempt3",
    "les_tornado_100m_mayfield_20211210_attempt3_fine30s",
    "frozen/les_nest_250m_grayzone", "frozen/les_nest_250m_km3",
    "frozen/les_tornado_100m_mayfield_20211210",
})


def _shipped_name(path):
    return path.relative_to(SHIPPED_CONFIGS).with_suffix("").as_posix()


def _shipped_tables():
    """Every shipped configuration (the wheel ships configs/** whole)."""
    import tomllib
    return [(path, tomllib.loads(path.read_text(encoding="utf-8")))
            for path in sorted(SHIPPED_CONFIGS.rglob("*.toml"))]


def _defaulted_row(raw):
    """The row a configuration's [fetch] source selects when it names none."""
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.static.source_defaults import source_static_defaults
    static = raw.get("static")
    if isinstance(static, dict) and "source" in static:
        return None
    fetch = (raw.get("fetch") or {}).get("source")
    if not isinstance(fetch, str):
        return None
    try:
        adapter = get_source_adapter(fetch)
    except (KeyError, ValueError):
        return None
    return source_static_defaults(adapter.source_id).get("source")


def _shipped_static_source_configs():
    """Every shipped configuration whose static fields come from a pinned
    source file, with that file's row id: its own [static].source, else
    the metadata default its [fetch].source declares (the same reading
    gpuwm.static.highres_production.resolve_static_highres makes)."""
    from gpuwm.static.source_defaults import with_declared_source_static_defaults
    found = []
    for path, raw in _shipped_tables():
        row_id = (with_declared_source_static_defaults(raw).get("static")
                  or {}).get("source")
        if row_id is not None:
            found.append(pytest.param(path, row_id,
                                      id=f"{_shipped_name(path)}-{row_id}"))
    return found


def _shipped_fallback_configs():
    return [pytest.param(path, id=_shipped_name(path))
            for path, raw in _shipped_tables()
            if _shipped_name(path) in _DEFAULTED_ON_OWN_CONE]


def test_the_defaulted_fallback_set_is_exactly_the_pinned_one():
    """Breakage it prevents: a shipped configuration silently moving
    between the pinned HRRR statics and WPS_GEOG.  The set of shipped
    configurations whose defaulted row is set aside is the pinned set."""
    from gpuwm.static.source_defaults import defaulted_source_fallback
    fallen = set()
    for path, raw in _shipped_tables():
        row_id = _defaulted_row(raw)
        if (row_id is not None and isinstance(raw.get("projection"), dict)
                and defaulted_source_fallback(row_id, raw["projection"])):
            fallen.add(_shipped_name(path))
    assert fallen == _DEFAULTED_ON_OWN_CONE


@pytest.mark.parametrize("path", _shipped_fallback_configs())
def test_a_shipped_config_on_its_own_cone_builds_from_wps_geog(path):
    """Breakage it prevents: these 13 shipped configurations stopped at
    "Prepare root static fields" in 2.8.6 staging (the case grid's
    projection differs) because a row they never named was applied by
    default.  The default is set aside, the carrier is the one they had
    before the row existed, and the declared form of the same row still
    refuses their 3 km root grid by name."""
    import tomllib
    from gpuwm.domain_wizard import experiment_from_text
    from gpuwm.static.external_source import (
        StaticSourceError, crop_window, static_source_row)
    from gpuwm.static.highres_production import resolve_static_highres
    from gpuwm.static.projection import grids_from_projection_config
    from gpuwm.static.source_defaults import (
        DEFAULTED_FALLBACK_REASON, defaulted_source_fallback)
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    row_id = _defaulted_row(raw)
    record = defaulted_source_fallback(row_id, raw["projection"])
    assert record["reason"] == DEFAULTED_FALLBACK_REASON
    assert record["used"] == "WPS_GEOG"
    carrier = resolve_static_highres(raw, source=str(path), base_dir=path.parent)
    assert getattr(carrier, "static_source", None) is None
    exp = experiment_from_text(text, source=str(path))
    root = grids_from_projection_config(exp)[0]
    # The builder reads the grid; it sees the same record as the table.
    assert defaulted_source_fallback(row_id, root) == record
    assert abs(root.dx - 3000.0) < 1e-6
    with pytest.raises(StaticSourceError, match="projection differs"):
        crop_window(static_source_row(row_id), root)
    declared = {**raw, "static": {"source": row_id}}
    named = resolve_static_highres(declared, source=str(path), base_dir=path.parent)
    assert named.static_source.id == row_id


def test_a_declared_source_off_its_cone_is_still_refused_by_name():
    """Breakage it prevents: the fallback reaching a source the
    configuration named, which would run WPS_GEOG statics under the
    named row's identity."""
    import tomllib
    from gpuwm.static.external_source import (
        StaticSourceError, crop_window, source_grid)
    from gpuwm.static.highres_production import resolve_static_highres
    from gpuwm.static.source_defaults import with_source_static_defaults
    path = SHIPPED_CONFIGS / "initdemo_bubble_on.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    raw["static"] = {"source": "hrrr-conus-v4"}
    assert with_source_static_defaults(raw, "hrrr") is raw
    carrier = resolve_static_highres(raw, source=str(path), base_dir=path.parent)
    assert carrier.static_source.id == "hrrr-conus-v4"
    row = carrier.static_source.row
    grid = type(source_grid(row))(
        ref_lat=38.5, ref_lon=-99.5, truelat1=28.5, truelat2=48.5,
        stand_lon=-99.5, dx=3000., dy=3000., e_we=101, e_sn=101)
    with pytest.raises(StaticSourceError,
                       match="'hrrr-conus-v4'.*projection differs"):
        crop_window(row, grid)


def test_a_defaulted_source_on_its_own_cone_is_still_taken():
    """Breakage it prevents: the fallback swallowing a configuration on
    the row's own cone, which would quietly drop the pinned statics."""
    import tomllib
    from gpuwm.static.highres_production import resolve_static_highres
    path = SHIPPED_CONFIGS / "hrrr_native_quick_demo.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    raw.pop("static")
    carrier = resolve_static_highres(raw, source=str(path), base_dir=path.parent)
    assert carrier.static_source.id == "hrrr-conus-v4"


@pytest.mark.parametrize("path,row_id", _shipped_static_source_configs())
def test_every_shipped_static_source_config_samples_its_source(path, row_id):
    """Breakage it prevents: configs/hrrr_native_quick_demo shipped on a
    25.3/45.3 cone while the pinned hrrr-conus-v4 source is 38.5/38.5, so
    `gpuwm go` refused it at "Prepare root static fields" (the case grid's
    projection differs) although `gpuwm prep --dry-run` accepted it.  The
    root grid must pass the build's own sampling_window at the row's
    spacing, and a d01-target.json companion must name the same window."""
    from gpuwm.domain_wizard import experiment_from_text
    from gpuwm.ingest.hrrr_target import load_hrrr_target_domain
    from gpuwm.static.external_source import sampling_window, static_source_row
    from gpuwm.static.projection import grids_from_projection_config
    row = static_source_row(row_id)
    exp = experiment_from_text(path.read_text(encoding="utf-8"), source=str(path))
    root = grids_from_projection_config(exp)[0]
    window = sampling_window(row, root)
    if window is None:
        pytest.skip(f"{path.name}: root spacing {root.dx:g} m is not the "
                    "row's; its statics build from WPS_GEOG")
    target = path.with_name(f"{path.stem}.d01-target.json")
    if target.is_file():
        assert sampling_window(row, load_hrrr_target_domain(target).grid()) == window


def _native_root(tmp_path, monkeypatch, **projection):
    """Run tools/hrrr_build_native_static.py on the native fixture's target
    (optionally on another cone), with no configuration file so the
    builder's own hrrr default applies, the field build replaced by the
    fixture's arrays and the selection it was asked for recorded."""
    import json
    import sys
    from dataclasses import replace
    from test_hrrr_native_static import _fixture
    from tools import hrrr_build_native_static as producer
    from gpuwm.static import build
    tmp_path.mkdir(parents=True, exist_ok=True)
    target, cache, _ = _fixture(tmp_path)
    target = replace(target, **projection)
    domain = tmp_path / "domain.json"
    domain.write_text(json.dumps(target.to_payload()), encoding="utf-8")
    with np.load(cache) as stored:
        arrays = {name: stored[name] for name in stored.files}
    geog = tmp_path / "geog"
    for directory in build._DEFAULT_GEOG_DIRS.values():
        (geog / directory).mkdir(parents=True, exist_ok=True)
        (geog / directory / "index").write_text("type=continuous\n")
    selections = []

    def fake_build(grid, root, *, selection, source_coverage_report):
        selections.append(selection)
        return dict(arrays)
    monkeypatch.setattr(producer, "build_static", fake_build)
    out, sealed = tmp_path / "out.npz", tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "native-static", "--geog-root", str(geog), "--domain-spec", str(domain),
        "--output", str(out), "--receipt", str(sealed)])
    producer.main()
    return selections, json.loads(sealed.read_text(encoding="utf-8"))


def test_native_root_receipt_says_a_defaulted_source_was_set_aside(tmp_path, monkeypatch):
    """Breakage it prevents: a root built from WPS_GEOG in place of the
    source its metadata defaults to, with nothing in the static receipt
    saying so.  Off the row's cone the builder takes no source and its
    receipt carries the ruling's sentence; on the cone the default still
    rides the selection and no fallback is recorded."""
    from gpuwm.static.source_defaults import DEFAULTED_FALLBACK_REASON
    # The fixture's 3 km target sits on its own 30/60 cone, as the shipped
    # LES and bubble roots do on theirs.
    (selection,), receipt = _native_root(tmp_path / "off", monkeypatch)
    assert selection.static_source is None
    record = receipt["static_source_fallback"]["d01"]
    assert record["id"] == "hrrr-conus-v4"
    assert record["reason"] == DEFAULTED_FALLBACK_REASON
    assert record["used"] == "WPS_GEOG"
    assert record["projection_mismatch"] == {
        "stand_lon": [-97.0, -97.5], "truelat1": [30.0, 38.5],
        "truelat2": [60.0, 38.5]}
    assert "static_source" not in receipt
    # On the row's cone (at another spacing, so no crop is attempted) the
    # default still rides the selection.
    (selection,), receipt = _native_root(
        tmp_path / "on", monkeypatch, truelat1=38.5, truelat2=38.5,
        stand_lon=-97.5, dx_m=1000.0, dy_m=1000.0)
    assert selection.static_source.id == "hrrr-conus-v4"
    assert "static_source_fallback" not in receipt
