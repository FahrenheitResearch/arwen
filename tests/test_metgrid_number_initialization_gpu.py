"""CUDA scalar routing and its independent, quantized-geometry Q oracle.

CPU and CUDA vertical interpolation use different FP32 logarithm/compiler
implementations. Their existing contract is numeric parity, while routing
through the same CUDA operator must remain bitwise identical.
"""
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS, metgrid_number_targets
from gpuwm.ingest import real
from gpuwm.ingest.preprocess_backend import CudaPreprocessBackend
from gpuwm.ingest.real import initialize_real
from gpuwm.verify.npref import np_wrf_real_vert_interp
from test_metgrid_number_initialization import _case, _mass_case


def _capture_geometry(monkeypatch):
    """Retain the actual FP32 geometry without replacing CUDA arithmetic."""
    import cupy as cp

    geometries = []
    prepare = CudaPreprocessBackend.prepare_wrf_vertical

    def capture(source, surface, target):
        geometries.append(tuple(cp.asnumpy(value).astype(np.float64)
                                for value in (source, surface, target)))
        return prepare(source, surface, target)

    monkeypatch.setattr(CudaPreprocessBackend, "prepare_wrf_vertical",
                        staticmethod(capture))
    return geometries


def _q_reference(snapshot, name, geometry, nz):
    # These fixtures carry isobaric source levels. Reorder the supplied
    # field independently, then use the float64 WRF transcription on the
    # exact FP32 pressures seen by CUDA (no geometry-rounding ambiguity).
    order = np.argsort(-snapshot.levels_hpa)
    source = np.asarray(snapshot.fields[name], dtype=np.float32)[order]
    surface = np.asarray(snapshot.fields[name + "_SFC"], dtype=np.float32)
    return np_wrf_real_vert_interp(
        source.astype(np.float64), surface.astype(np.float64), *geometry,
        interp_in_logp=True, extrap="constant", vboundb=nz + 1)


def _assert_q_matches_reference(actual, reference):
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all() and np.all(actual >= 0)
    # The existing scalar-Q authority rule is pinned in
    # test_hrrr_vertical_disposition.py; this is not a measured-max bound
    # or the temperature operator's much larger 5e-3 absolute tolerance.
    np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=5e-8)
    np.testing.assert_array_equal(actual == 0, reference == 0)


@pytest.mark.gpu
@pytest.mark.parametrize("mp", [8, 9, 10, 16, 18, 28, 50])
def test_cuda_initial_numbers_match_wrf_authority_with_supplied_surface(mp, monkeypatch):
    import cupy as cp

    geometries = _capture_geometry(monkeypatch)
    snapshot, cfg, coord, terrain, orography = _case(mp)
    kw = dict(source_orography=orography, analyzed_species=())
    supplied = initialize_real(snapshot, cfg, coord, terrain,
        analyzed_number_fields=METGRID_NUMBER_FIELDS, **kw)
    absent = initialize_real(snapshot, cfg, coord, terrain, **kw)
    for source, target in metgrid_number_targets(cfg).items():
        actual = cp.asnumpy(getattr(supplied.state, target))
        reference = _q_reference(snapshot, source, geometries[0], cfg.nz)
        _assert_q_matches_reference(actual, reference)
        assert np.count_nonzero(actual) > 0
        assert cp.count_nonzero(getattr(absent.state, target)).item() == 0
    # Number moments cannot enter total water or change thermodynamics.
    for name in ("mup", "thp", "php", "qv", "u", "v", "qc", "qr", "pb", "alb"):
        cp.testing.assert_array_equal(getattr(supplied.state, name),
                                      getattr(absent.state, name))
    if mp == 18:
        cp.testing.assert_array_equal(supplied.state.qndrop, absent.state.qndrop)


@pytest.mark.gpu
def test_cuda_mass_surface_reaches_first_pressure_recurrence(monkeypatch):
    import cupy as cp

    geometries = _capture_geometry(monkeypatch)
    totals = []
    rebalance = real._rebalance_moist_pressure

    def capture_total(pressure, qtot, *args, **kwargs):
        totals.append(qtot.copy())
        return rebalance(pressure, qtot, *args, **kwargs)

    monkeypatch.setattr(real, "_rebalance_moist_pressure", capture_total)
    snapshot, cfg, coord, terrain, orography = _mass_case()
    kw = dict(source_orography=orography, analyzed_species=("QC", "QH"),
              analyzed_surface_fields=("QC", "QH"))
    supplied = initialize_real(snapshot, cfg, coord, terrain, **kw)
    for name in ("QC", "QH"):
        actual = cp.asnumpy(getattr(supplied.state, name.lower()))
        _assert_q_matches_reference(
            actual, _q_reference(snapshot, name, geometries[0], cfg.nz))
        assert np.count_nonzero(actual) > 0
        snapshot.fields[name + "_SFC"][...] = 0

    first_empty = len(totals)
    empty = initialize_real(snapshot, cfg, coord, terrain, **kw)
    assert cp.count_nonzero(empty.state.qc).item() == 0
    assert cp.count_nonzero(empty.state.qh).item() == 0
    # Before the first recurrence, vapor is identical in these two runs.
    # Its load must already contain both supplied condensate species.
    condensate = (cp.asnumpy(supplied.state.qc).astype(np.float64)
                  + cp.asnumpy(supplied.state.qh).astype(np.float64))
    np.testing.assert_array_equal(totals[0], totals[first_empty] + condensate)
    assert cp.any(supplied.state.php != empty.state.php).item()


@pytest.mark.gpu
def test_cuda_number_route_matches_existing_mass_q_operator_bitwise():
    import cupy as cp

    snapshot, cfg, coord, terrain, orography = _case(8)
    # Use exactly the same source profile for an existing mass-Q path and
    # the new number path, with their common zero surface convention.
    # A power-of-two scale keeps cloud mass realistic and values exact.
    numbers = snapshot.fields["QNI"] * np.float32(2 ** -24)
    snapshot = replace(snapshot, fields={
        **snapshot.fields, "QNI": numbers, "QC": numbers.copy(),
        "QNI_SFC": np.zeros((cfg.ny, cfg.nx), dtype=np.float32)})
    kw = dict(source_orography=orography, analyzed_species=("QC",),
              analyzed_number_fields=("QNI",))
    supplied = initialize_real(snapshot, cfg, coord, terrain, **kw)
    cp.testing.assert_array_equal(supplied.state.ni.view(cp.uint32),
                                  supplied.state.qc.view(cp.uint32))
    assert cp.count_nonzero(supplied.state.ni).item() > 0

    snapshot.fields["QNI"][...] = 0
    snapshot.fields["QC"][...] = 0
    zero = initialize_real(snapshot, cfg, coord, terrain, **kw)
    assert cp.count_nonzero(zero.state.ni).item() == 0
    assert cp.count_nonzero(zero.state.qc).item() == 0
