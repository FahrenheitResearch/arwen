"""Flagged metgrid number analyses reach the exact selected scalar package."""
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS, metgrid_number_targets
from gpuwm.ingest.real import initialize_real


@pytest.mark.parametrize("mp,expected", [
    (0, {}), (1, {}), (6, {}), (8, {"QNI":"ni", "QNR":"nr"}),
    (9, dict(zip(METGRID_NUMBER_FIELDS, ("ni","nc","nr","ns","ng","nh")))),
    (10, {"QNI":"ni", "QNR":"nr", "QNS":"ns", "QNG":"ng"}),
    (16, {"QNC":"nc", "QNR":"nr"}),
    (18, {"QNI":"qni", "QNR":"qnr", "QNS":"qns", "QNG":"qng", "QNH":"qnh"}),
    (28, {"QNI":"ni", "QNC":"nc", "QNR":"nr"}),
    (50, {"QNI":"ni", "QNR":"nr"}),
])
def test_registry_membership_distinguishes_qnc_and_qndrop(mp, expected):
    cfg = _case(mp)[1]
    assert metgrid_number_targets(cfg) == expected


def _case(mp):
    from test_metem_differential import _synthetic
    snapshot, cfg, coord, terrain, orography = _synthetic(2500., 2500.)
    cfg = replace(cfg, mp_physics=mp, mp28_aerosol_source="synthetic")
    # Different vertical and horizontal values expose accidental broadcast,
    # field aliasing and use of an unrelated donor column.
    shape = snapshot.fields["TT"].shape
    profile = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1) * 64
    fields = dict(snapshot.fields)
    for index, name in enumerate(METGRID_NUMBER_FIELDS):
        fields[name] = profile * np.float32(2 ** index)
        fields[name+"_SFC"] = (np.arange(cfg.ny*cfg.nx, dtype=np.float32).reshape(cfg.ny,cfg.nx)+3) * np.float32(2 ** index)
    return replace(snapshot, fields=fields), cfg, coord, terrain, orography


@pytest.mark.parametrize("mp", [8, 9, 10, 16, 18, 28, 50])
def test_native_cpu_initialization_retains_numbers_without_changing_thermodynamics(mp):
    snapshot, cfg, coord, terrain, orography = _case(mp)
    kw = dict(source_orography=orography, p_top=5000., preprocess_backend="cpu",
              state_backend="preprocess", analyzed_species=())
    absent = initialize_real(snapshot, cfg, coord, terrain, **kw)
    supplied = initialize_real(snapshot, cfg, coord, terrain,
                               analyzed_number_fields=METGRID_NUMBER_FIELDS, **kw)
    targets = metgrid_number_targets(cfg)
    receipt = supplied.hydrometeor_initialization["number_moments"]
    assert receipt["retained_correspondence"] == targets
    assert set(receipt["discarded_inactive_package_fields"]) == set(METGRID_NUMBER_FIELDS)-set(targets)
    for name in ("mup", "thp", "php", "qv", "u", "v", "qc", "qr", "pb", "alb"):
        np.testing.assert_array_equal(getattr(supplied.state, name), getattr(absent.state, name))
    for source, target in targets.items():
        actual = getattr(supplied.state, target)
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        assert np.count_nonzero(actual) > 0
        assert np.all(getattr(absent.state, target) == 0)
        assert np.ptp(actual) > 0
    # Every field uses the same existing Q operator. Exact powers of two
    # distinguish number-field routing while keeping a bitwise scale oracle.
    first_source = next(iter(targets))
    base = getattr(supplied.state, targets[first_source])
    first_index = METGRID_NUMBER_FIELDS.index(first_source)
    for source, target in targets.items():
        factor = np.float32(2 ** (METGRID_NUMBER_FIELDS.index(source)-first_index))
        np.testing.assert_array_equal(getattr(supplied.state, target), base * factor)
    if mp == 18:
        np.testing.assert_array_equal(supplied.state.qndrop, absent.state.qndrop)


@pytest.mark.parametrize("bad", [-1., np.inf, np.nan])
def test_bad_number_analysis_refuses_even_when_package_does_not_consume_it(bad):
    snapshot, cfg, coord, terrain, orography = _case(6)
    snapshot.fields["QNI"][1, 1, 1] = bad
    with pytest.raises(ValueError, match="analyzed number field QNI"):
        initialize_real(snapshot, cfg, coord, terrain, source_orography=orography,
            preprocess_backend="cpu", state_backend="preprocess", analyzed_species=(),
            analyzed_number_fields=("QNI",))


@pytest.mark.parametrize("declared", [("QNI","QNI"), ("unknown",), "QNI"])
def test_number_inventory_must_be_explicit_and_unique(declared):
    snapshot, cfg, coord, terrain, orography = _case(8)
    with pytest.raises(ValueError, match="analyzed_number_fields"):
        initialize_real(snapshot, cfg, coord, terrain, source_orography=orography,
            preprocess_backend="cpu", state_backend="preprocess", analyzed_number_fields=declared)


def test_missing_declared_number_field_refuses_before_interpolation():
    snapshot, cfg, coord, terrain, orography = _case(8)
    snapshot = replace(snapshot, fields={k:v for k,v in snapshot.fields.items() if k != "QNI"})
    with pytest.raises(KeyError, match="QNI"):
        initialize_real(snapshot, cfg, coord, terrain, source_orography=orography,
            preprocess_backend="cpu", state_backend="preprocess", analyzed_number_fields=("QNI",))


def test_surface_pseudo_level_can_be_the_only_number_source():
    snapshot, cfg, coord, terrain, orography = _case(8)
    snapshot.fields["QNI"][...] = 0
    kw = dict(source_orography=orography, preprocess_backend="cpu", state_backend="preprocess",
              analyzed_species=(), analyzed_number_fields=("QNI",))
    supplied = initialize_real(snapshot, cfg, coord, terrain, **kw)
    assert np.count_nonzero(supplied.state.ni) > 0
    snapshot.fields["QNI_SFC"][...] = 0
    absent = initialize_real(snapshot, cfg, coord, terrain, **kw)
    assert np.count_nonzero(absent.state.ni) == 0


def _mass_case(mp=18):
    snapshot, cfg, coord, terrain, orography = _case(mp)
    fields = dict(snapshot.fields)
    for name in ("QC", "QH"):
        fields[name] = np.zeros_like(fields["TT"])
        fields[name+"_SFC"] = np.full((cfg.ny,cfg.nx), .003, np.float32)
    return replace(snapshot, fields=fields), cfg, coord, terrain, orography


def test_mass_surface_analysis_reaches_state_before_moist_pressure_recurrence():
    snapshot, cfg, coord, terrain, orography = _mass_case()
    kw = dict(source_orography=orography, preprocess_backend="cpu", state_backend="preprocess",
              analyzed_species=("QC", "QH"), analyzed_surface_fields=("QC", "QH"))
    supplied = initialize_real(snapshot, cfg, coord, terrain, **kw)
    for name in ("qc", "qh"):
        assert np.count_nonzero(getattr(supplied.state, name)) > 0
    for name in ("QC", "QH"):
        snapshot.fields[name+"_SFC"][...] = 0
    empty = initialize_real(snapshot, cfg, coord, terrain, **kw)
    assert np.count_nonzero(empty.state.qc) == np.count_nonzero(empty.state.qh) == 0
    assert np.any(supplied.state.php != empty.state.php)
    receipt = supplied.hydrometeor_initialization
    assert receipt["schema"] == "gpuwm-metgrid-hydrometeor-initialization-v1"
    assert "vertical_disposition" not in receipt
    assert set(receipt["surface_pseudo_levels"]) == {"QC", "QH"}
    assert receipt["retained_correspondence"] == {"QC":"qc", "QH":"qh"}


def test_preparation_prices_original_mass_number_levels_and_surfaces():
    from types import SimpleNamespace
    from pathlib import Path
    from gpuwm.metem_door import metgrid_analysis_shapes
    names = ("QC", "QH", *METGRID_NUMBER_FIELDS)
    shapes = {name:(1,7,11,13) for name in ("TT","GHT","RH",*names)}
    shapes.update(UU=(1,7,11,14), VV=(1,7,12,13), PSFC=(1,11,13), SOILHGT=(1,11,13))
    metadata = SimpleNamespace(path=Path("met_em.test"), nx=13, ny=11,
        global_attributes={"FLAG_"+name:1 for name in names}, variables=shapes)
    actual = metgrid_analysis_shapes(metadata)
    for name in names:
        assert actual[name] == (6,11,13)
        assert actual[name+"_SFC"] == (11,13)


def test_explicit_metgrid_mass_inventory_honors_passive_vapor_package():
    snapshot, cfg, coord, terrain, orography = _mass_case(0)
    kw = dict(source_orography=orography, preprocess_backend="cpu", state_backend="preprocess")
    supplied = initialize_real(snapshot, cfg, coord, terrain, analyzed_species=("QC","QH"),
        analyzed_surface_fields=("QC","QH"), **kw)
    absent = initialize_real(snapshot, cfg, coord, terrain, analyzed_species=(), **kw)
    receipt = supplied.hydrometeor_initialization
    assert receipt["retained_correspondence"] == {}
    assert set(receipt["discarded_source_species"]) == {"QC","QH"}
    for name in ("mup", "php", "thp", "qv", "qc", "qr"):
        np.testing.assert_array_equal(getattr(supplied.state,name), getattr(absent.state,name))
