"""Native-level boundary conversion, declarations and shared remapping."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.chem_conversions import convert_source
from gpuwm.ingest.native_extras import (
    EXTRA_GATES, gate_extra_fields, inside_native_grid,
    interpolate_boundary_tracers, map_boundary_rows,
)
from gpuwm.verify.chem_conversions_ref import mass_density_inverse
from gpuwm.verify.chem_oracle import ulp_table
from gpuwm.chem_table import catalog

SOURCE = catalog().sources["hrrr-native-smoke"]
PARAMETERS = SOURCE.variables["MASSDEN"]["conversion_parameters"]


#: The CPU preprocessing bridge's symbols these tests drive
#: (tools/grib1_bridge/src/chem_conversions.rs).
_CONVERSION_SYMBOLS = ("gpuwm_kg_m3_to_ug_kg_dry_f32",
                       "gpuwm_boundary_weighted_sum_f32")


@pytest.fixture
def bridge():
    """The CPU bridge, when it is a build that carries the conversions.

    A staged bridge from a release before the chem conversions is skipped
    by name rather than failed: the conversion is Rust (the Python boundary
    law), so there is nothing to test until the tree's own build is loaded.
    """
    import ctypes
    from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
    try:
        path = resolve_cpu_bridge()
        library = ctypes.CDLL(str(path))
    except (OSError, RuntimeError, ValueError) as error:
        pytest.skip(f"CPU preprocessing bridge not loadable: {error}")
    missing = [name for name in _CONVERSION_SYMBOLS if not hasattr(library, name)]
    if missing:
        pytest.skip(f"{path} predates the chem source conversions ({missing}); "
                    "rebuild tools/grib1_bridge to run these")
    return path


def test_conversion_upp_and_hand_arithmetic(bridge):
    mass = np.array([0, 1e-9, 7e-8, 3e-7], np.float32)
    p = np.array([100000, 90000, 50000, 10000], np.float32)
    t = np.array([300, 290, 260, 220], np.float32)
    actual = convert_source("kg_m3_to_ug_kg_dry", mass, {"PRES": p, "TT": t}, parameters=PARAMETERS, cpu_bridge=bridge)
    expected = mass_density_inverse(mass, p, t)
    assert ulp_table(actual, expected) == {"max_ulp": 0, "n_nonzero": 0, "n": 4}
    np.testing.assert_allclose(actual, mass.astype(float) * 287.04 * t / p * 1e9, rtol=2e-7)


def test_conversion_formula_fixture(bridge):
    root = Path(__file__).parents[1] / "tests/data/oracles/chem/smoke/source_conversion"
    from gpuwm.verify.chem_oracle import load_case
    case = load_case(root / "source_cells")
    inputs = case["inputs"].T
    expected = case["inverse"]
    from gpuwm.verify.chem_conversions_ref import effective_density, mass_density_forward
    assert ulp_table(effective_density(inputs[1], inputs[2]), case["density"]) == {"max_ulp": 0, "n_nonzero": 0, "n": 8}
    assert ulp_table(mass_density_forward(case["source_mixing_ratio"], inputs[1], inputs[2]), inputs[0]) == {"max_ulp": 0, "n_nonzero": 0, "n": 8}
    for value in (mass_density_inverse(*inputs), convert_source(
            "kg_m3_to_ug_kg_dry", inputs[0], {"PRES": inputs[1], "TT": inputs[2]}, parameters=PARAMETERS, cpu_bridge=bridge)):
        assert ulp_table(value, expected) == {"max_ulp": 0, "n_nonzero": 0, "n": 8}


def test_invalid_source_prevents_invalid_chemistry(bridge):
    for mass, p, t in ((-1, 100000, 300), (1, 0, 300), (1, 100000, float("nan"))):
        with pytest.raises(ValueError, match="prevents non-finite or negative"):
            convert_source("kg_m3_to_ug_kg_dry", [mass], {"PRES": [p], "TT": [t]}, parameters=PARAMETERS, cpu_bridge=bridge)


def test_density_parameters_come_from_the_source_row(bridge):
    met = {"PRES": [100000.], "TT": [300.]}
    original = convert_source("kg_m3_to_ug_kg_dry", [1e-9], met,
                              parameters=PARAMETERS, cpu_bridge=bridge)
    changed = dict(PARAMETERS, gas_constant_j_kg_k=574.08)
    actual = convert_source("kg_m3_to_ug_kg_dry", [1e-9], met,
                            parameters=changed, cpu_bridge=bridge)
    np.testing.assert_array_equal(actual, np.float32(2) * original)
    with pytest.raises(ValueError, match="prevents using the wrong density"):
        convert_source("kg_m3_to_ug_kg_dry", [1e-9], met, cpu_bridge=bridge)


def test_gate_extras_are_numeric_and_explicit():
    assert gate_extra_fields({}) == ()
    gate = {"extra_fields": "MASSDEN", "extra_MASSDEN": EXTRA_GATES["MASSDEN"]}
    assert gate_extra_fields(gate) == ("MASSDEN",)
    assert EXTRA_GATES["MASSDEN"] == "PASS discipline=0 category=20 parameter=0 level_type=105"
    for changed in (dict(gate, extra_MASSDEN=EXTRA_GATES["PMTF"]), {"extra_fields": "UNKNOWN"}, {"extra_MASSDEN": EXTRA_GATES["MASSDEN"]}):
        with pytest.raises(ValueError, match="prevents"):
            gate_extra_fields(changed)


def test_default_selector_inventory_and_explicit_extras():
    from tools.download_hrrr_native_subset import atmosphere_selectors, HYBRID_FIELDS, SURFACE_FIELDS, field_spellings
    expected = tuple([f"{'|'.join(field_spellings(name))}:{level} hybrid level" for name in HYBRID_FIELDS for level in range(1, 51)] + [f"{'|'.join(field_spellings(name))}:{level}" for name, level in SURFACE_FIELDS])
    assert atmosphere_selectors() == expected
    extra = atmosphere_selectors(("MASSDEN",))
    assert len(extra) == len(expected) + 50
    assert sum("MASSDEN:" in text for text in extra) == 50
    assert not any("PMTF" in text or "PMTC" in text for text in extra)


def test_default_gate_and_snapshot_do_not_add_fields(tmp_path):
    from gpuwm.ingest.hrrr import _read_gate, _ATMOSPHERE_3D
    gate = {"status": "PASS", "atmosphere_selected_per_time": "561", "hybrid_levels": "50", "soil_selected_per_time": "18", "window_shape": "4x4", "window_zero_based_inclusive": "i=0..3 j=0..3", "qice_mapping": "PASS discipline=0 category=1 parameter=82 level_type=105", "cross_time_inventory": "PASS"}
    path = tmp_path / "gate.txt"
    path.write_text("".join(f"{k}\t{v}\n" for k, v in gate.items()))
    assert _read_gate(tmp_path) == gate
    assert _ATMOSPHERE_3D == ("PRES", "QC", "QI", "QR", "QS", "QG", "HGT", "TT", "SPFH", "U_MASS", "V_MASS")
    gate.update(extra_fields="MASSDEN", extra_MASSDEN=EXTRA_GATES["MASSDEN"], atmosphere_selected_per_time="611")
    path.write_text("".join(f"{k}\t{v}\n" for k, v in gate.items()))
    assert _read_gate(tmp_path) == gate


def test_source_row_loads_through_catalog():
    from gpuwm.chem_table import catalog
    row = catalog().sources["hrrr-native-smoke"]
    assert row.kind == "boundary" and row.remap == "native_levels"
    assert row.variables["MASSDEN"]["selector"]["level_type"] == 105


def test_source_conversion_precedes_convex_mapping(bridge):
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    backend = CpuPreprocessBackend(bridge)
    raw = np.array([[[0, 1e-8], [2e-8, 0]]], np.float32)
    source = {"MASSDEN": raw, "PRES": np.array([[[100000, 50000], [80000, 60000]]], np.float32), "TT": np.full_like(raw, 280)}
    plan = backend.indexed_plan((2, 2), np.array([[0.5]]), np.array([[0.5]]))
    row = SimpleNamespace(name="arbitrary_tracer", boundary=({"source": "declared_source", "fields": ("MASSDEN",), "weights": (1,), "conversion": "kg_m3_to_ug_kg_dry"},))
    result = map_boundary_rows(source, (row,), SimpleNamespace(name="declared_source", variables=SOURCE.variables), plan, cpu_bridge=bridge)[row.name]
    expected = plan.apply(mass_density_inverse(raw, source["PRES"], source["TT"]), method="bilinear")
    np.testing.assert_array_equal(result, expected)
    assert np.min(result) >= 0
    zeros = plan.apply(np.zeros_like(raw), method="bilinear")
    assert np.count_nonzero(zeros) == 0


def test_synthetic_column_uses_hydrometeor_vertical_operator(bridge):
    from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend
    engine = resolve_preprocess_backend("cpu", cpu_bridge=bridge, workers=1)
    source_p = np.array([90000, 70000, 50000, 30000], np.float32)[:, None, None]
    target_p = np.array([85000, 60000, 40000], np.float32)[:, None, None]
    surface_p = np.array([[100000]], np.float32)
    zero = np.zeros((1, 1), np.float32)
    plan = engine.prepare_wrf_vertical(source_p, surface_p, target_p)
    field = np.array([2, 4, 6, 8], np.float32)[:, None, None]
    actual = interpolate_boundary_tracers({"arbitrary": field}, plan, engine.float32, zero, 3)["arbitrary"]
    expected = plan.apply(field, zero, interp_in_logp=True, extrap="constant", vboundb=4, values_are_finite=True)
    np.testing.assert_array_equal(actual, expected)
    assert ulp_table(actual, expected) == {"max_ulp": 0, "n_nonzero": 0, "n": 3}
    assert actual.shape == (3, 1, 1) and np.all(actual >= 0)
    # force_sfc_in_vinterp=1 discards the 900 hPa donor below the first
    # target: the bottom target lies between zero at 1000 hPa and 4 at 700.
    expected_bottom = 4 * (np.log(np.float32(85000)) - np.log(np.float32(100000))) / (np.log(np.float32(70000)) - np.log(np.float32(100000)))
    np.testing.assert_allclose(actual[0, 0, 0], expected_bottom, rtol=1e-7)
    assert 4 <= actual[1, 0, 0] <= 6


@pytest.mark.parametrize("axis,edge,inside,outside", [(0, "west", 1, 0.999), (0, "east", 1796.999, 1797), (1, "south", 1, 0.999), (1, "north", 1056.999, 1057)])
def test_coverage_on_both_sides_of_every_edge(axis, edge, inside, outside):
    point = [100., 100.]
    point[axis] = inside
    assert inside_native_grid([point[0]], [point[1]], nx=SOURCE.grid["nx"], ny=SOURCE.grid["ny"])
    point[axis] = outside
    assert not inside_native_grid([point[0]], [point[1]], nx=SOURCE.grid["nx"], ny=SOURCE.grid["ny"])


def test_empty_nonfinite_coverage():
    assert not inside_native_grid([], [], nx=SOURCE.grid["nx"], ny=SOURCE.grid["ny"])
    assert not inside_native_grid([float("nan")], [100], nx=SOURCE.grid["nx"], ny=SOURCE.grid["ny"])


def test_ordered_fallback_and_absent_receipt():
    from gpuwm.ingest.native_extras import choose_boundary_source
    rows = ({"source": "first_source"}, {"source": "next_source"})
    assert choose_boundary_source(rows, {"first_source", "next_source"}) == (rows[0], None)
    assert choose_boundary_source(rows, {"next_source"}) == (rows[1], None)
    assert choose_boundary_source(rows, set()) == (None, "smoke from outside the domain is absent")


def test_geographic_coverage_matches_native_index_halo():
    from gpuwm.ingest.hrrr import hrrr_source_grid, native_domain_coverage
    grid = hrrr_source_grid()
    for x, y, expected in ((2.01, 101., True), (1.99, 101., False),
                           (1797.99, 101., True), (1798.01, 101., False),
                           (101., 2.01, True), (101., 1.99, False),
                           (101., 1057.99, True), (101., 1058.01, False)):
        lat, lon = grid.ij_to_latlon(np.array([x]), np.array([y]))
        assert native_domain_coverage(lat, lon) is expected


def test_weighted_boundary_terms_are_generic(bridge):
    from gpuwm.chem_conversions import weighted_source_fields
    fields = [np.array([2., 0.], np.float32), np.array([4., 6.], np.float32)]
    value = weighted_source_fields(fields, [.25, .5], cpu_bridge=bridge)
    assert ulp_table(value, np.array([2.5, 3.], np.float32)) == {"max_ulp": 0, "n_nonzero": 0, "n": 2}


def test_extra_loader_requires_manifest_binding(tmp_path):
    from gpuwm.ingest.hrrr import _load_verified_hrrr_native_window, _ATMOSPHERE_3D, _ATMOSPHERE_2D, _SOIL_3D
    gate = {"cycle": "2025-08-01 00:00:00", "window_zero_based_inclusive": "i=0..3 j=0..3", "window_shape": "4x4", "extra_fields": "MASSDEN", "extra_MASSDEN": EXTRA_GATES["MASSDEN"]}
    atmosphere = tmp_path / "atmosphere-f00"
    soil = tmp_path / "soil-f00"
    atmosphere.mkdir()
    soil.mkdir()
    for names, root, shape in ((_ATMOSPHERE_3D + ("MASSDEN",), atmosphere, (50, 4, 4)), (_ATMOSPHERE_2D, atmosphere, (4, 4)), (_SOIL_3D, soil, (9, 4, 4))):
        for name in names:
            np.zeros(shape, "<f4").tofile(root / (name + ".f32le"))
    entries = {p.relative_to(tmp_path) for p in tmp_path.rglob("*.f32le")}
    snap = _load_verified_hrrr_native_window(tmp_path, gate, 0, manifest_entries=entries)
    assert snap.extra_fields == ("MASSDEN",) and snap.fields["MASSDEN"].shape == (50, 4, 4)
    entries.remove(Path("atmosphere-f00/MASSDEN.f32le"))
    with pytest.raises(ValueError, match="prevents unauthenticated"):
        _load_verified_hrrr_native_window(tmp_path, gate, 0, manifest_entries=entries)
    default_gate = {k: v for k, v in gate.items() if not k.startswith("extra_")}
    default = _load_verified_hrrr_native_window(tmp_path, default_gate, 0)
    assert default.extra_fields == () and "MASSDEN" not in default.fields
    assert set(default.fields) == set(_ATMOSPHERE_3D + _ATMOSPHERE_2D + _SOIL_3D)


def test_fetch_records_extras_and_refetches_a_changed_selector(tmp_path, monkeypatch):
    from datetime import datetime
    import json
    from gpuwm import fetch
    from tools import download_hrrr_native_subset as transport
    calls = []

    def product(request, *, workers, retries, expected_count=-1, extras=()):
        calls.append((request.kind, tuple(extras), expected_count))
        message = b"GRIB" + b"\x00\x00\x00\x02" + (20).to_bytes(8, "big") + b"7777"
        request.destination.write_bytes(message * expected_count)
        request.index_path.write_text("1:0:fixture\n")
        return {"kind": request.kind}

    monkeypatch.setattr(transport, "_download_product", product)
    options = dict(cycle=datetime(2025, 8, 1), hours=(0, 1), area=None,
                   out=tmp_path / "inputs", progress=lambda _: None, file_workers=1)
    default = json.loads(fetch.fetch_hrrr(**options).read_text())
    assert "native_extras" not in default
    assert sorted(count for kind, extras, count in calls if kind == "atmosphere") == [561, 561]
    calls.clear()
    extra = json.loads(fetch.fetch_hrrr(**options, extras=("MASSDEN",)).read_text())
    assert extra["native_extras"] == ["MASSDEN"]
    assert [(kind, count) for kind, _, count in calls] == [("atmosphere", 611), ("soil", 18)] * 2
    calls.clear()
    fetch.fetch_hrrr(**options, extras=("MASSDEN",))
    assert not calls
    fetch.fetch_hrrr(**options, extras=("PMTF",))
    assert len(calls) == 4 and all(extras == ("PMTF",) for _, extras, _ in calls)


def test_the_receipt_says_when_a_row_takes_no_inflow_from_outside():
    # DESIGN 5.1: with no enabled boundary source, smoke inflow is the row's
    # default (0) and the run receipt says smoke from outside is absent.
    from gpuwm import chem_table
    from gpuwm.core.chem_driver import boundary_inflow
    words = boundary_inflow(chem_table.load_sets(("smoke",), ("rave-3km",)))
    assert words["smoke"].startswith("smoke from outside the domain is absent")
    assert "hrrr-native-smoke" in words["smoke"] and "default 0 ug kg-1" in words["smoke"]
    enabled = boundary_inflow(chem_table.load_sets(("smoke",), ("rave-3km", "hrrr-native-smoke")))
    assert enabled["smoke"] == "smoke inflow from hrrr-native-smoke"
