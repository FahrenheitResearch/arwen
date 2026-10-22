"""Composition admission: table selection, immutable artifacts and first physics."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import composition_source as source
from gpuwm.chem_boundary_remap import BoundaryRemapError, boundary_rows, _needed_variables
from gpuwm.chem_table import catalog, load_sets


def test_sources_are_table_selected_and_all_bound_fields_exist():
    for name in ("merra2", "cams-reanalysis"):
        row = catalog().sources[name]
        table = load_sets(("gocart_simple",), (name,))
        selected = boundary_rows(table, name, available=(name,))
        assert {r.name for r in selected} >= {"dust_1", "dust_5", "bc1", "oc2", "seas_4", "sulf"}
        assert set(_needed_variables(row, selected)) <= set(row.variables)
        paths = [(g, Path(f"/{g}.nc")) for g in {v["group"] for v in row.variables.values() if "netcdf_name" in v}]
        manifest = source._manifest(row, paths)
        assert {v["group"] for v in manifest["variables"]} == {g for g, p in paths}
    table = load_sets(("cams_aq",), ("cams-reanalysis",))
    assert {r.name for r in boundary_rows(table, "cams-reanalysis", available=("cams-reanalysis",))} == {"o3", "no2", "co", "so2"}


def test_import_refuses_licence_or_missing_file_groups_before_decoder(tmp_path, monkeypatch):
    def absent():
        pytest.fail("decoder must not run before input admission")
    monkeypatch.setattr(source, "_decoder", absent)
    cams = catalog().sources["cams-reanalysis"]
    limited = replace(cams, acquisition={**cams.acquisition, "research_only_licence": "CC-BY-NC-4.0"})
    with pytest.raises(ValueError, match="--licence CC-BY-NC-4.0"):
        source.import_files(limited, [], tmp_path)
    f = tmp_path / "a.nc"
    f.write_bytes(b"not-decoded")
    with pytest.raises(ValueError, match="exactly these groups"):
        source.import_files(catalog().sources["merra2"], [("aerosol", f)], tmp_path)


def _admitted(tmp_path):
    row = catalog().sources["merra2"]
    raw = tmp_path / "original.nc"
    decoder = tmp_path / "decoder"
    raw.write_bytes(b"original content")
    decoder.write_bytes(b"decoder identity")
    identity = source._identity(row, [("aerosol", raw)], decoder)
    directory = tmp_path / "decoded"
    directory.mkdir()
    field = directory / "dust.f32"
    # These are artifact integrity test bytes, not an alternate data decoder.
    field.write_bytes(np.arange(8, dtype="<f4").tobytes())
    stack = dict(schema="gpuwm-composition-stack-v1", latitude=[40., 39.], longitude=[-4., -3.],
                 outputs=[dict(key="dust_1", valid_time="2022-03-15T00:00:00Z", levels=[1., 2.], file=field.name)])
    (directory / "stack.json").write_text(json.dumps(stack))
    payloads = {p.name: dict(bytes=p.stat().st_size, sha256=source._hash(p)) for p in directory.iterdir()}
    (directory / "authority.json").write_text(json.dumps(dict(identity=identity, payloads=payloads)))
    return row, raw, decoder, directory, identity


def test_artifact_is_readonly_and_payload_or_source_changes_are_refused(tmp_path):
    row, raw, decoder, directory, identity = _admitted(tmp_path)
    frames = source._read(row, directory, identity)
    assert frames.times == (datetime(2022, 3, 15, tzinfo=timezone.utc),)
    field = frames.frames[frames.times[0]]["dust_1"]
    np.testing.assert_array_equal(field.array, np.arange(8, dtype=np.float32).reshape(2, 2, 2))
    assert not field.array.flags.writeable
    raw.write_bytes(b"changed content!")
    newer = source._identity(row, [("aerosol", raw)], decoder)
    with pytest.raises(BoundaryRemapError, match="current files or source table"):
        source._read(row, directory, newer)
    (directory / "dust.f32").write_bytes(bytes(32))
    with pytest.raises(BoundaryRemapError, match="payload changed"):
        source._read(row, directory, identity)


def test_decoder_and_source_table_changes_invalidate_admission(tmp_path):
    from dataclasses import replace
    row, raw, decoder, directory, identity = _admitted(tmp_path)
    decoder.write_bytes(b"different decoder")
    newer = source._identity(row, [("aerosol", raw)], decoder)
    assert newer != identity
    with pytest.raises(BoundaryRemapError, match="current files or source table"):
        source._read(row, directory, newer)
    changed = replace(row, licence="different licence")
    assert source._identity(changed, [("aerosol", raw)], decoder) != newer


def test_coverage_requires_each_variable_at_both_bracketing_times(tmp_path, monkeypatch):
    from gpuwm.chem_boundary_remap import SourceFrames
    row = catalog().sources["merra2"]
    t0 = datetime(2022, 3, 15, tzinfo=timezone.utc)
    from datetime import timedelta
    t1 = t0 + timedelta(hours=3)
    frames = SourceFrames(row.name, {t0: {"dust_1": object()}, t1: {"dust_1": object()}}, ("original.nc",))
    monkeypatch.setattr(source, "registered_frames", lambda *args: frames)
    assert source.covering_files(row, ["dust_1"], t0 + timedelta(hours=1), tmp_path) == {"0": Path("original.nc")}
    with pytest.raises(BoundaryRemapError, match="do not cover"):
        source.covering_files(row, ["dust_2"], t0 + timedelta(hours=1), tmp_path)
    with pytest.raises(BoundaryRemapError, match="do not cover"):
        source.covering_files(row, ["dust_1"], t1 + timedelta(hours=1), tmp_path)


def test_source_mass_initializes_thompson_before_first_step(monkeypatch):
    from gpuwm.core import chem_mp_coupling as coupling
    table = load_sets(("gocart_simple",), ("merra2",))
    state = SimpleNamespace(chem=SimpleNamespace(table=table), nifa=object(), nwfa=object())
    terms = coupling.pack_terms(coupling.rows(table))
    for term in terms:
        setattr(state, table.row(term.name).state_attr, object())
    calls = []
    monkeypatch.setattr(coupling, "launch_niwfa", lambda *args: calls.append(args))
    cfg = SimpleNamespace(aerosol_mp_coupling="diagnose")
    receipt = coupling.initialize_from_sources(state, cfg, [dict(source="merra2", rows=[t.name for t in terms])])
    assert receipt["initialized_fields"] == ["nwfa", "nifa"]
    assert receipt["dataset_used"] and not receipt["synthetic_fallback_in_use"]
    assert len(calls) == 1 and calls[0][2:] == (state.nifa, state.nwfa)
    with pytest.raises(ValueError, match="no populated aerosol source"):
        coupling.initialize_from_sources(state, cfg, [dict(source="merra2", rows=[])])


def test_composition_rows_reach_the_public_source_listing_without_model_dispatch():
    from gpuwm.sources_cli import _resolve, _format_listing
    rows = source.inventory_rows()
    document = dict(sources=[], composition_sources=rows)
    for name in ("merra2", "cams-reanalysis", "cams-global"):
        assert _resolve(document, name)["source_kind"] == "composition"
        assert name in _format_listing(document)
    for name in ("merra2", "cams-reanalysis", "geos-fp"):
        row = _resolve(document, name)
        assert row["maturity"] == dict(status="implemented", runnable=True)
        assert not row["run_plan"]["intent_supported"]


def test_source_backed_thompson_coupling_is_default_and_can_be_disabled():
    from gpuwm.config import RunConfig
    from gpuwm.core.chem_mp_coupling import effective_mode
    cfg = RunConfig(nx=96, ny=96, nz=49, dx=3000., dy=3000., ztop=20000., dt=15., run_seconds=10800., mp_physics=28, chem_sets="gocart_simple", chem_sources="merra2")
    assert effective_mode(cfg) == "diagnose"
    from dataclasses import replace
    cfg = replace(cfg, aerosol_mp_coupling="none")
    assert effective_mode(cfg) == "none"
    assert effective_mode(RunConfig(nx=96, ny=96, nz=49, dx=3000., dy=3000., ztop=20000., dt=15., run_seconds=10800., mp_physics=28)) == "none"


def test_dust_only_source_does_not_claim_or_force_ccn(monkeypatch):
    from gpuwm.core import chem_mp_coupling as coupling
    from gpuwm.boundary_fields import admissible_boundary_inventory
    from gpuwm.chem_source_init import chem_boundary_fields
    table = load_sets(("dust",), ("merra2",))
    state = SimpleNamespace(chem=SimpleNamespace(table=table), nifa=object(), nwfa=object())
    terms = coupling.pack_terms(coupling.rows(table))
    for term in terms:
        setattr(state, table.row(term.name).state_attr, object())
    monkeypatch.setattr(coupling, "launch_niwfa", lambda *args: None)
    cfg = SimpleNamespace(aerosol_mp_coupling="diagnose", chem_sets="dust", chem_sources=("merra2",),
                          mp_physics=28, moist=True, specified=True)
    receipt = coupling.initialize_from_sources(state, cfg, [dict(source="merra2", rows=[t.name for t in terms])])
    assert receipt["initialized_fields"] == ["nifa"]
    assert "synthetic_fallback_in_use" not in receipt
    assert admissible_boundary_inventory(cfg, ("qv", "nifa", *chem_boundary_fields(table, cfg)))


def test_published_organic_matter_is_not_counted_twice_in_optics_or_numbers():
    from gpuwm.core.chem_mp_coupling import pack_terms, rows
    table = load_sets(("gocart_primary",), ("merra2",))
    oc = table.row("oc2")
    boundary = next(b for b in oc.boundary if b["source"] == "merra2")
    assert boundary["weights"][0] * oc.optics["mass_multiplier"] == pytest.approx(1.0)
    term = next(t for t in pack_terms(rows(table)) if t.name == "oc2")
    assert term.to_kg_per_kg * boundary["weights"][0] == pytest.approx(1e-9)
    geos = catalog().sources["geos-fp"]
    selected = boundary_rows(table, "geos-fp", available=("geos-fp",))
    assert len(selected) == 13
    assert set(_needed_variables(geos, selected)) <= set(geos.variables)


def test_native_hybrid_coefficients_are_table_metadata(tmp_path):
    from dataclasses import replace
    row, raw, decoder, directory, identity = _admitted(tmp_path)
    # The same arbitrary decoded payload can use a different hybrid source row.
    cams = catalog().sources["cams-reanalysis"]
    assert cams.grid["decode"]["level"] == "model_level"
    assert len(cams.vertical["half_level_a_pa"]) == len(cams.vertical["half_level_b"]) == 61
    assert cams.vertical["half_level_a_pa"][-1] == 0.
    assert cams.vertical["half_level_b"][-1] == 1.
    hybrid = replace(row, vertical=cams.vertical)
    identity = source._identity(hybrid, [("aerosol", raw)], decoder)
    authority = json.loads((directory / "authority.json").read_text())
    authority["identity"] = identity
    (directory / "authority.json").write_text(json.dumps(authority))
    frames = source._read(hybrid, directory, identity)
    fld = frames.frames[frames.times[0]]["dust_1"]
    assert fld.coordinate_values == tuple(cams.vertical["half_level_a_pa"]) + tuple(cams.vertical["half_level_b"])


def test_native_pressure_reference_and_complete_layer_count_are_declared_in_rows():
    for name in ('merra2', 'geos-fp'):
        row = catalog().sources[name]
        paths = [(group, Path(f'/{group}.nc')) for group in ('aerosol', 'meteorology')]
        constructor = source._manifest(row, paths)['pressure']
        assert constructor['top_pressure_pa'] == 1.0
        assert constructor['expected_layers'] == 72
        assert constructor['order'] == 'top_down'


@pytest.mark.parametrize("name,credential", [
    ("merra2", "NASA Earthdata login"),
    ("cams-reanalysis", "Copernicus ADS key"),
])
def test_missing_composition_names_acquisition_access_and_unpopulated_state(tmp_path, monkeypatch, name, credential):
    monkeypatch.setattr(source, "_decoder", lambda: pytest.fail("missing files must fail before decoding"))
    row = catalog().sources[name]
    for attempt in (
        lambda: source.registered_frames(row, tmp_path),
        lambda: source.import_files(row, [("aerosol", tmp_path / "missing.nc")], tmp_path,
                                    licence=row.acquisition.get("research_only_licence")),
    ):
        with pytest.raises(BoundaryRemapError) as caught:
            attempt()
        message = str(caught.value)
        assert credential in message
        assert "chemistry IC/BC" in message
        assert "initialization never downloads or reads credentials" in message
        assert f"--source {name}" in message
        # CC-BY-4.0 (CAMS) and CC0 (NASA) both permit commercial use, so no
        # remedy asks for a research opt-in.
        assert "--licence" not in message
