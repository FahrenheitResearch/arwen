"""The masked surface chain in Rust against its NumPy oracle, byte for byte.

``gpuwm_wps_masked_chain_f64`` replaced the single-core NumPy chain that
mapped soil moisture and temperature, snow, skin temperature and sea ice.
That chain is kept verbatim in :mod:`gpuwm.verify.wps_masked_oracle`, and
every assertion here compares float64 BYTES and the repair tallies in
their key order, never a tolerance: the forecast's land surface starts
from these values, and the prepared trees of every route were
byte-identical before the move.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import pytest

from gpuwm.ingest import horiz
from gpuwm.ingest.cpu_backend import (
    CPU_BRIDGE_ENV,
    CpuPreprocessBackend,
    MaskedChainUnavailable,
    WPS_MASKED_CHAIN_ENTRY,
)
from gpuwm.verify import wps_masked_oracle as oracle

pytestmark = pytest.mark.requires_capability("wps_masked_chain_bridge")

ROOT = Path(__file__).resolve().parents[1]

FULL = horiz._WPS_FULL_CHAIN
SNOW = horiz._WPS_SNOW_CHAIN
SST = horiz._WPS_SST_CHAIN


def _native() -> CpuPreprocessBackend:
    return CpuPreprocessBackend()


def _axes(ny, nx):
    return np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64)


def _field(rng, ny, nx, bounds):
    """A source field with every value class the chain treats specially."""
    if bounds is None:
        low, high = 250.0, 300.0
    else:
        low, high = bounds
    span = high - low
    field = rng.uniform(low, high, (ny, nx))
    # A sharp step makes sixteen_pt overshoot past the range.
    field[:, : nx // 3] = low + 0.002 * span
    picks = rng.integers(0, ny * nx, 40)
    specials = [np.nan, np.inf, -np.inf, -999.0, 9999.0, 0.0, -0.0,
                high + 0.0003 * span, low - 0.0003 * span, high, low,
                1.0e308, -1.0e308, 1.0e-20]
    for index, pick in enumerate(picks):
        field.flat[pick] = specials[index % len(specials)]
    return field


def _land(rng, ny, nx):
    """A coast, a lake, single-cell islands, and fractional-only land."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    fraction = np.clip((xx - nx * 0.45) / 4.0 + 0.5, 0.0, 1.0)
    fraction[(yy - ny // 2) ** 2 + (xx - nx * 3 // 4) ** 2 < 5] = 0.0
    for _ in range(3):
        fraction[rng.integers(0, ny), rng.integers(0, nx // 3)] = 0.9
    fraction[rng.integers(0, ny), rng.integers(0, nx // 3)] = 0.3
    return fraction > 0.5, fraction > 0.0


def _targets(rng, ny, nx, count):
    ty = rng.uniform(0.0, ny - 1.0, count)
    tx = rng.uniform(0.0, nx - 1.0, count)
    # Integer and half-integer coordinates (degenerate four_pt, the
    # coincident sixteen_pt branch, equal-distance search ties), the first
    # and last rows and columns, and points within 2e-10 of them.
    ty[:12] = np.floor(ty[:12])
    tx[6:18] = np.floor(tx[6:18]) + 0.5
    ty[18:22] = [0.0, ny - 1.0, 2.0e-10, ny - 1.0 - 2.0e-10]
    tx[22:26] = [0.0, nx - 1.0, 2.0e-10, nx - 1.0 - 2.0e-10]
    ty[26:30] = np.minimum(np.floor(ty[26:30]) + 1.0e-5, ny - 1.0)
    tx[30:34] = np.maximum(np.floor(tx[30:34]) - 5.0e-6, 0.0)
    return ty.reshape(-1, 1), tx.reshape(-1, 1)


def _same(native, reference):
    native = np.asarray(native, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    assert native.shape == reference.shape
    assert native.tobytes() == reference.tobytes()


CASES = [(mode, chain, bounds, fill)
         for mode in ("plain", "land", "skin")
         for chain in (FULL, SNOW, SST)
         for bounds in (None, (0.0, 1.0), (170.0, 400.0))
         for fill in (0.0, 1.0, 285.0, np.nan)
         if not (mode == "skin" and chain != FULL)]


@pytest.mark.parametrize(("mode", "chain", "bounds", "fill"), CASES)
def test_native_chain_matches_the_oracle_byte_for_byte(mode, chain, bounds,
                                                       fill):
    native = _native()
    seed = abs(hash((mode, chain, bounds, str(fill)))) % (2 ** 32)
    rng = np.random.default_rng(seed)
    ny, nx = 23, 31
    lat, lon = _axes(ny, nx)
    field = _field(rng, ny, nx, bounds)
    land, partial = _land(rng, ny, nx)
    ty, tx = _targets(rng, ny, nx, 260)
    target_land = rng.uniform(size=ty.shape) < 0.55
    for donors_land, donors_partial in (
            (land, partial), (np.zeros_like(land), partial),
            (np.zeros_like(land), np.zeros_like(land)),
            (np.ones_like(land), np.ones_like(land))):
        tallies = ({}, {})
        if mode == "plain":
            active = rng.uniform(size=ty.shape) < 0.8
            got = horiz.wps_masked_field_interpolate(
                field, lat, lon, ty, tx, source_valid=donors_land,
                target_active=active, chain=chain, fill_value=fill,
                physical_range=bounds, tally=tallies[0], native=native)
            want = oracle.wps_masked_field_interpolate(
                field, lat, lon, ty, tx, source_valid=donors_land,
                target_active=active, chain=chain, fill_value=fill,
                physical_range=bounds, tally=tallies[1])
            _same(got, want)
        elif mode == "land":
            got, got_recovered = horiz._land_pass_with_fractional_second_chance(
                field, lat, lon, ty, tx, land_donors=donors_land,
                partial_land_donors=donors_partial, target_active=target_land,
                chain=chain, fill_value=fill, physical_range=bounds,
                tally=tallies[0], native=native)
            want, want_recovered = oracle._land_pass_with_fractional_second_chance(
                field, lat, lon, ty, tx, land_donors=donors_land,
                partial_land_donors=donors_partial, target_active=target_land,
                chain=chain, fill_value=fill, physical_range=bounds,
                tally=tallies[1])
            _same(got, want)
            assert got_recovered == want_recovered
        else:
            got, got_recovered = horiz._skin_temperature_on_both_surfaces(
                field, lat, lon, ty, tx, land_donors=donors_land,
                partial_land_donors=donors_partial, target_land=target_land,
                fill_value=fill, physical_range=bounds, tally=tallies[0],
                native=native)
            want, want_recovered = oracle._skin_temperature_on_both_surfaces(
                field, lat, lon, ty, tx, land_donors=donors_land,
                partial_land_donors=donors_partial, target_land=target_land,
                fill_value=fill, physical_range=bounds, tally=tallies[1])
            _same(got, want)
            assert got_recovered == want_recovered
        assert list(tallies[0].items()) == list(tallies[1].items())


def _plain_pair(field, valid, ty, tx, active, chain, fill, bounds,
                workers=None):
    ny, nx = field.shape
    lat, lon = _axes(ny, nx)
    got_tally, want_tally = {}, {}
    got = horiz.wps_masked_field_interpolate(
        field, lat, lon, ty, tx, source_valid=valid, target_active=active,
        chain=chain, fill_value=fill, physical_range=bounds,
        tally=got_tally, native=_native(), workers=workers)
    want = oracle.wps_masked_field_interpolate(
        field, lat, lon, ty, tx, source_valid=valid, target_active=active,
        chain=chain, fill_value=fill, physical_range=bounds,
        tally=want_tally)
    _same(got, want)
    assert list(got_tally.items()) == list(want_tally.items())
    return got, got_tally


def test_edge_sources_match_the_oracle():
    rng = np.random.default_rng(20260928)
    ny, nx = 12, 14
    ty, tx = _targets(rng, ny, nx, 120)
    active = np.ones(ty.shape, dtype=bool)
    flat = np.full((ny, nx), 0.25)
    everything = np.ones((ny, nx), dtype=bool)
    nothing = np.zeros((ny, nx), dtype=bool)
    # No donors at all, every donor, no active targets, an empty chain.
    _plain_pair(flat, nothing, ty, tx, active, FULL, 1.0, (0.0, 1.0))
    _plain_pair(flat, everything, ty, tx, active, FULL, 1.0, (0.0, 1.0))
    _plain_pair(flat, everything, ty, tx, ~active, FULL, 1.0, (0.0, 1.0))
    _plain_pair(flat, everything, ty, tx, active, (), 1.0, None)
    # Exact zeros and negative zeros take the 1e-20 sentinel.
    zeros = np.where(rng.uniform(size=(ny, nx)) < 0.5, 0.0, -0.0)
    zeros[3:6, 4:9] = 1.0e-20
    _plain_pair(zeros, everything, ty, tx, active, FULL, 7.0, None)
    # Values near the float64 limit overflow an average to inf, which
    # falls through to the next operator.
    huge = np.full((ny, nx), 1.7e308)
    huge[::2, ::3] = -1.7e308
    _plain_pair(huge, rng.uniform(size=(ny, nx)) < 0.6, ty, tx, active,
                FULL, 3.0, None)
    # A saturated ice sheet stored at 1.0003 goes on the bound.
    ice = np.full((ny, nx), 1.0003)
    ice[:, :4] = 0.0
    _, tally = _plain_pair(ice, everything, ty, tx, active, SNOW, 0.0,
                           (0.0, 1.0))
    assert tally["source_roundoff_at_bound"] > 0
    # The search: one donor in a corner answers the whole grid.
    corner = np.zeros((ny, nx), dtype=bool)
    corner[0, 0] = True
    _, tally = _plain_pair(np.full((ny, nx), 0.5), corner, ty, tx, active,
                           ("search",), 1.0, (0.0, 1.0))
    assert tally["search"] > 0


def test_a_donor_beyond_the_search_depth_is_left_at_the_fill():
    # A tenth of a degree per column keeps the axis inside one revolution,
    # so the target columns are the ones written here.
    ny, nx = 5, 1300
    latitude = np.arange(ny, dtype=np.float64)
    longitude = np.arange(nx, dtype=np.float64) * 0.1
    field = np.zeros((ny, nx))
    valid = np.zeros((ny, nx), dtype=bool)
    field[2, 1290] = 0.7
    valid[2, 1290] = True
    ty = np.array([[0.0], [2.0]])
    tx = np.array([[0.0], [125.0]])
    tallies = ({}, {})
    got = horiz.wps_masked_field_interpolate(
        field, latitude, longitude, ty, tx, source_valid=valid,
        target_active=np.ones(ty.shape, bool), chain=FULL, fill_value=1.0,
        physical_range=(0.0, 1.0), tally=tallies[0], native=_native())
    want = oracle.wps_masked_field_interpolate(
        field, latitude, longitude, ty, tx, source_valid=valid,
        target_active=np.ones(ty.shape, bool), chain=FULL, fill_value=1.0,
        physical_range=(0.0, 1.0), tally=tallies[1])
    _same(got, want)
    assert list(tallies[0].items()) == list(tallies[1].items())
    assert got[0, 0] == 1.0 and got[1, 0] == 0.7
    assert tallies[0]["fill"] == 1 and tallies[0]["search"] == 1


def test_a_grid_too_small_for_the_16_point_weights_falls_through():
    """Where the oracle has no defined answer, the native chain has one.

    On a source under four cells across, NumPy's ``wt_average_16pt``
    indexes past the array and raises IndexError.  The stencil leaves the
    array, which is exactly the case metgrid's operator declines, so the
    native chain falls through to the search.
    """
    field = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7], [0.1, 0.2, 0.3]])
    valid = np.array([[True, False, False], [False, False, False],
                      [False, False, True]])
    ty, tx = np.array([[1.2]]), np.array([[1.3]])
    with pytest.raises(IndexError):
        oracle.wps_masked_field_interpolate(
            field, *_axes(3, 3), ty, tx, source_valid=valid,
            target_active=np.array([[True]]), chain=FULL, fill_value=1.0)
    tally = {}
    got = horiz.wps_masked_field_interpolate(
        field, *_axes(3, 3), ty, tx, source_valid=valid,
        target_active=np.array([[True]]), chain=FULL, fill_value=1.0,
        tally=tally, native=_native())
    assert np.isfinite(got).all() and tally["search"] == 1


def test_a_whole_globe_source_maps_its_dateline_columns():
    latitude = np.linspace(-90.0, 90.0, 19)
    longitude = np.arange(0.0, 360.0, 10.0)
    rng = np.random.default_rng(7)
    field = rng.uniform(0.0, 1.0, (19, 36))
    valid = rng.uniform(size=(19, 36)) < 0.5
    target_lat = rng.uniform(-90.0, 90.0, (40, 1))
    target_lon = np.concatenate(
        [rng.uniform(345.0, 350.0, 20), rng.uniform(0.0, 5.0, 20)])[:, None]
    target_lat[:4, 0] = [-90.0, 90.0, 0.0, 45.0]
    tallies = ({}, {})
    got = horiz.wps_masked_field_interpolate(
        field, latitude, longitude, target_lat, target_lon,
        source_valid=valid, target_active=np.ones((40, 1), bool),
        chain=FULL, fill_value=1.0, physical_range=(0.0, 1.0),
        tally=tallies[0], native=_native())
    want = oracle.wps_masked_field_interpolate(
        field, latitude, longitude, target_lat, target_lon,
        source_valid=valid, target_active=np.ones((40, 1), bool),
        chain=FULL, fill_value=1.0, physical_range=(0.0, 1.0),
        tally=tallies[1])
    _same(got, want)
    assert list(tallies[0].items()) == list(tallies[1].items())


def test_the_queue_limited_search_counterexample_matches():
    field = np.zeros((10, 10))
    valid = np.zeros((10, 10), dtype=bool)
    field[4, 0], valid[4, 0] = 11.0, True
    field[6, 7], valid[6, 7] = 22.0, True
    got, _ = _plain_pair(field, valid, np.array([[4.49]]),
                         np.array([[4.49]]), np.array([[True]]),
                         ("search",), -999.0, None)
    assert got[0, 0] == 11.0


#: Targets whose search answer depends on how a distance is squared.  Each
#: pair of donors (value 1 at the first offset from (7, 7), value 2 at the
#: second) is equally far from the target in real arithmetic, and the
#: rounding of the squared distance decides which is strictly nearer.  The
#: shared-walk search squares as ``dx * dx``, which IEEE arithmetic rounds
#: the same on every platform; CPython's ``float ** 2`` (the C ``pow``,
#: which the per-target walk used) gives the other answer on glibc for
#: every row below.
SQUARE_DECIDED_TIES = (
    (7.178984594419762, 7.357969188839524, (1, -2), (-1, 2), 2.0),
    (6.839742026555739, 7.320515946888522, (3, 1), (1, -3), 1.0),
    (6.931512296209771, 6.657561481048855, (2, 3), (3, -2), 1.0),
    (7.199544108446771, 7.332573514077952, (-4, 1), (-1, -4), 1.0),
    (7.087867745857984, 6.824264508284031, (1, 2), (-1, -2), 1.0),
)


@pytest.mark.parametrize(("yy", "xx", "first", "second", "expected"),
                         SQUARE_DECIDED_TIES)
def test_a_search_tie_is_decided_by_the_product_as_in_the_shared_walk(
        yy, xx, first, second, expected):
    field = np.zeros((15, 15))
    valid = np.zeros((15, 15), dtype=bool)
    for (dx, dy), value in ((first, 1.0), (second, 2.0)):
        field[7 + dy, 7 + dx] = value
        valid[7 + dy, 7 + dx] = True
    got, _ = _plain_pair(field, valid, np.array([[yy]]), np.array([[xx]]),
                         np.array([[True]]), ("search",), -1.0, None)
    assert got[0, 0] == expected


def test_an_island_a_coarse_source_holds_as_sea_matches_the_shared_walk():
    """The island tile's shape (tests/test_wps_search_shared_walk.py): a
    whole-globe 0.25 degree source whose only usable land is 60 cells
    away, and 2,100 fine-grid island targets in three source cells.  The
    native search takes its walk once per start cell as the oracle does,
    so it answers in the oracle's bytes and in well under its time."""
    import time

    ny, nx = 721, 1440
    field = np.full((ny, nx), 290.0)
    valid = np.zeros((ny, nx), dtype=bool)
    valid[400:404, 700:704] = True
    field[400:404, 700:704] = 300.0 + np.arange(16).reshape(4, 4)
    rng = np.random.default_rng(0)
    ty = (np.repeat(np.array([402.0, 402.0, 403.0]), 700)
          + rng.uniform(-0.49, 0.49, size=2100)).reshape(3, 700)
    tx = (np.repeat(np.array([760.0, 761.0, 761.0]), 700)
          + rng.uniform(-0.49, 0.49, size=2100)).reshape(3, 700)
    active = np.ones(ty.shape, dtype=bool)
    got, _ = _plain_pair(field, valid, ty, tx, active, ("search",), 285.0,
                         None)
    assert np.all((got >= 300.0) & (got <= 315.0))
    started = time.perf_counter()
    horiz.wps_masked_field_interpolate(
        field, *_axes(ny, nx), ty, tx, source_valid=valid,
        target_active=active, chain=("search",), fill_value=285.0,
        native=_native(), workers=1)
    assert time.perf_counter() - started < 2.0


@pytest.mark.parametrize("workers", [1, 2, 3, 7, 64, None])
def test_worker_count_moves_no_bit_and_no_count(workers):
    rng = np.random.default_rng(3)
    ny, nx = 41, 57
    lat, lon = _axes(ny, nx)
    field = _field(rng, ny, nx, (0.0, 1.0))
    land, partial = _land(rng, ny, nx)
    ty, tx = _targets(rng, ny, nx, 9000)
    target_land = rng.uniform(size=ty.shape) < 0.5
    tallies = ({}, {})
    got, recovered = horiz._skin_temperature_on_both_surfaces(
        field, lat, lon, ty, tx, land_donors=land,
        partial_land_donors=partial, target_land=target_land,
        fill_value=0.0, physical_range=(0.0, 1.0), tally=tallies[0],
        native=_native(), workers=workers)
    serial, serial_recovered = horiz._skin_temperature_on_both_surfaces(
        field, lat, lon, ty, tx, land_donors=land,
        partial_land_donors=partial, target_land=target_land,
        fill_value=0.0, physical_range=(0.0, 1.0), tally=tallies[1],
        native=_native(), workers=1)
    _same(got, serial)
    assert recovered == serial_recovered
    assert list(tallies[0].items()) == list(tallies[1].items())


def test_the_unit_refusal_reads_the_native_counts_with_its_old_words():
    rng = np.random.default_rng(5)
    ny, nx = 10, 12
    land = np.zeros((ny, nx), dtype=bool)
    land[:, 6:] = True
    active = np.ones((4, 4), dtype=bool)
    percent = rng.uniform(5.0, 40.0, (ny, nx))
    for slab, layer in ((percent, 2), (np.full((ny, nx), np.nan), None)):
        messages = []
        for module in (horiz, oracle):
            with pytest.raises(ValueError) as caught:
                module._refuse_land_field_not_in_its_unit(
                    slab, name="SM000007", layer=layer, bounds=(0.0, 1.0),
                    fill=1.0, land_donors=land, partial_land_donors=land,
                    target_active=active)
            messages.append(str(caught.value))
        assert messages[0] == messages[1]
    # In its unit: no refusal from either.
    for module in (horiz, oracle):
        module._refuse_land_field_not_in_its_unit(
            rng.uniform(0.0, 1.0, (ny, nx)), name="SM000007", layer=0,
            bounds=(0.0, 1.0), fill=1.0, land_donors=land,
            partial_land_donors=land, target_active=active)


def test_an_unknown_operator_is_refused_only_when_a_target_reaches_it():
    field = np.full((6, 6), 0.5)
    valid = np.ones((6, 6), dtype=bool)
    ty = np.array([[2.5]])
    tx = np.array([[2.5]])
    # four_pt answers first, so the unknown name is never reached.
    _plain_pair(field, valid, ty, tx, np.array([[True]]),
                ("four_pt", "bogus"), 0.0, None)
    for module in (horiz, oracle):
        kwargs = {"native": _native()} if module is horiz else {}
        with pytest.raises(ValueError,
                           match="unknown WPS interpolation operator 'bogus'"):
            module.wps_masked_field_interpolate(
                field, *_axes(6, 6), ty, tx, source_valid=~valid,
                target_active=np.array([[True]]),
                chain=("four_pt", "bogus"), fill_value=0.0, **kwargs)


def test_a_library_without_the_chain_is_refused_by_name_with_the_remedy():
    stale = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    stale.path = Path("/opt/old/libgpuwm_preprocess_cpu.so")
    stale.wps_masked_chain_entry = False
    with pytest.raises(MaskedChainUnavailable) as caught:
        stale.require_wps_masked_chain()
    message = str(caught.value)
    assert WPS_MASKED_CHAIN_ENTRY in message
    assert "soil moisture and temperature, snow, skin temperature" in message
    assert "/opt/old/libgpuwm_preprocess_cpu.so" in message
    # The remedy every missing CPU library gets, not a bare sentence.
    from gpuwm.bridges import cpu_bridge_remedy
    assert cpu_bridge_remedy(stale.path.name) in message


def test_an_override_naming_a_missing_library_is_refused_naming_it(
        monkeypatch, tmp_path):
    from gpuwm.ingest import cpu_backend

    missing = tmp_path / "absent" / "libgpuwm_preprocess_cpu.so"
    monkeypatch.setenv(CPU_BRIDGE_ENV, str(missing))
    monkeypatch.setattr(cpu_backend, "_SHARED_BACKENDS", {})
    with pytest.raises(FileNotFoundError, match=CPU_BRIDGE_ENV):
        horiz.wps_masked_field_interpolate(
            np.zeros((4, 4)), *_axes(4, 4), np.array([[1.0]]),
            np.array([[1.0]]), source_valid=np.ones((4, 4), bool),
            target_active=np.array([[True]]), chain=FULL, fill_value=0.0)


def test_a_cuda_preparation_says_why_it_needs_a_missing_cpu_library(
        monkeypatch, tmp_path):
    """Under CUDA the masked fields are the one reason the CPU library
    is needed, so the refusal names them before the resolver's text."""
    import gpuwm.bridges
    from gpuwm.bridges import cpu_bridge_remedy
    from gpuwm.ingest import cpu_backend
    from gpuwm.ingest.preprocess_backend import CudaPreprocessBackend

    def refused():
        monkeypatch.setattr(cpu_backend, "_SHARED_BACKENDS", {})
        with pytest.raises(MaskedChainUnavailable) as caught:
            CudaPreprocessBackend(host_workers=2).wps_masked_chain_engine()
        first, rest = str(caught.value).split("\n", 1)
        assert first.startswith(
            "the masked surface fields (soil moisture and temperature, "
            "snow, skin temperature and sea ice) map through the CPU "
            "preprocessing library under every preprocessing backend, the "
            "CUDA backend included")
        assert isinstance(caught.value.__cause__, FileNotFoundError)
        return rest

    # An override naming a file that is not there: the resolver names it.
    missing = tmp_path / "absent" / "libgpuwm_preprocess_cpu.so"
    monkeypatch.setenv(CPU_BRIDGE_ENV, str(missing))
    assert CPU_BRIDGE_ENV in refused()
    # Nothing on the ladder: the resolver's search list and its remedy.
    monkeypatch.delenv(CPU_BRIDGE_ENV)
    monkeypatch.setattr(gpuwm.bridges, "find_artifact", lambda *_: None)
    rest = refused()
    assert rest.startswith("GPUWM parallel CPU preprocessing bridge was "
                           "not found")
    assert cpu_bridge_remedy(cpu_backend._library_names()[0]) in rest


def _snapshot():
    from gpuwm.ingest.grib import Era5Snapshot
    from gpuwm.static.lambert import LambertGrid

    grid = LambertGrid(
        ref_lat=40.0, ref_lon=-85.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-85.0, dx=60_000.0, dy=60_000.0, e_we=9, e_sn=8)
    latitude = np.linspace(34.0, 46.0, 13, dtype=np.float64)
    longitude = np.linspace(267.0, 283.0, 15, dtype=np.float64)
    lon2, lat2 = np.meshgrid(longitude, latitude)
    base = lat2 + 0.1 * lon2
    landsea = np.clip((lon2 - 273.0) / 4.0, 0.0, 1.0)
    landsea[2, 2] = 0.8
    rng = np.random.default_rng(17)
    fields = {
        "LANDSEA": landsea,
        "SKINTEMP": 250.0 + base + rng.uniform(-2.0, 2.0, base.shape),
        "SST": np.where(landsea < 0.5, 270.0 + base, np.nan),
        "SEAICE": np.where(landsea < 0.5,
                           np.clip((275.0 - lon2) / 8.0, 0.0, 1.0003), np.nan),
        "ST000007": 260.0 + base,
        "SM000007": np.clip(0.2 + 0.01 * rng.normal(size=base.shape), 0, 1),
        "SNOW_EC": np.where(lat2 > 42.0, 0.01 * base, 0.0),
    }
    snapshot = Era5Snapshot(
        valid_time=datetime(1974, 4, 3, 12),
        levels_hpa=np.array([500.0, 1000.0], dtype=np.float64),
        latitude=latitude, longitude=longitude, fields=fields)
    return snapshot, grid


def _oracle_mapping(snapshot, grid, target_land):
    """What the NumPy chain made of the snapshot's masked fields."""
    mass_lat, mass_lon = grid.latlon_mass()
    transform, _ = horiz.source_coordinate_transform(snapshot)
    mass_lat, mass_lon = transform(mass_lat, mass_lon)
    fields = snapshot.fields
    landsea = np.asarray(fields["LANDSEA"], np.float32)
    source_land = landsea > 0.5
    partial = landsea > 0.0
    lat, lon = snapshot.latitude, snapshot.longitude
    out, repairs = {}, {}

    def slab(name):
        return np.asarray(np.asarray(fields[name], np.float32), np.float64)

    repairs["SKINTEMP"] = {}
    out["SKINTEMP"], _ = oracle._skin_temperature_on_both_surfaces(
        slab("SKINTEMP"), lat, lon, mass_lat, mass_lon,
        land_donors=source_land, partial_land_donors=partial,
        target_land=target_land, fill_value=0.0,
        physical_range=(170.0, 400.0), tally=repairs["SKINTEMP"])
    for name, fill, bounds in (("ST000007", 285.0, (170.0, 400.0)),
                               ("SM000007", 1.0, (0.0, 1.0)),
                               ("SNOW_EC", 0.0, None)):
        tally = repairs.setdefault(name, {}) if bounds else None
        soil = name != "SNOW_EC"
        passes = {} if tally is not None else None
        combined, _ = oracle._land_pass_with_fractional_second_chance(
            slab(name), lat, lon, mass_lat, mass_lon,
            land_donors=source_land, partial_land_donors=partial,
            target_active=target_land,
            chain=SNOW if name == "SNOW_EC" else FULL,
            fill_value=np.nan if soil else fill,
            physical_range=bounds, tally=passes)
        if soil:
            # interpolate_era5_to_lambert's soil rule: a value the land
            # pass leaves missing is land the source holds no land for,
            # counted as no_source_land, then METGRID.TBL fill_missing.
            starved = target_land & ~np.isfinite(combined)
            if passes is not None:
                moved = int(np.count_nonzero(starved))
                passes["fill"] = passes.get("fill", 0) - moved
                passes["no_source_land"] = moved
            combined = np.where(np.isfinite(combined), combined, fill)
        if tally is not None:
            for key, value in passes.items():
                tally[key] = tally.get(key, 0) + value
        out[name] = combined
    out["SST"] = oracle.wps_masked_field_interpolate(
        slab("SST"), lat, lon, mass_lat, mass_lon,
        source_valid=np.ones(source_land.shape, bool),
        target_active=np.ones(mass_lat.shape, bool), chain=SST,
        fill_value=0.0)
    repairs["XICE"] = {}
    out["XICE"] = oracle.wps_masked_field_interpolate(
        slab("SEAICE"), lat, lon, mass_lat, mass_lon,
        source_valid=~source_land, target_active=~target_land, chain=SNOW,
        fill_value=0.0, physical_range=(0.0, 1.0), tally=repairs["XICE"])
    return ({name: value.astype(np.float32) for name, value in out.items()},
            repairs)


def test_the_cpu_route_maps_every_masked_field_as_the_oracle_did(
        monkeypatch):
    snapshot, grid = _snapshot()
    first = horiz.interpolate_era5_to_lambert(snapshot, grid, backend="cpu")
    target_land = np.asarray(first.fields["LANDSEA"]) >= 0.5
    want, want_repairs = _oracle_mapping(snapshot, grid, target_land)

    def refuse(*args, **kwargs):
        raise AssertionError("the NumPy oracle ran at run time")

    for name in ("wps_masked_field_interpolate",
                 "_land_pass_with_fractional_second_chance",
                 "_skin_temperature_on_both_surfaces", "_wps_search",
                 "_wps_sixteen_pt", "_wps_four_pt", "_wps_wt_average"):
        monkeypatch.setattr(oracle, name, refuse)
    for workers in (1, 5):
        mapped = horiz.interpolate_era5_to_lambert(
            snapshot, grid, backend="cpu", workers=workers)
        for name, value in want.items():
            got = np.asarray(mapped.fields[name])
            assert got.dtype == np.float32, name
            assert got.tobytes() == value.tobytes(), name
        got_repairs = {name: list(counts.items()) for name, counts
                       in mapped.masked_field_repairs.items()}
        assert got_repairs == {name: list(counts.items()) for name, counts
                               in want_repairs.items()}


def test_every_backend_reaches_the_same_native_entry(monkeypatch):
    """The CUDA backend and a custom backend map masks in the same library."""
    from gpuwm.ingest import cpu_backend
    from gpuwm.ingest.preprocess_backend import (
        CudaPreprocessBackend, resolve_preprocess_backend)

    shared = cpu_backend.shared_cpu_backend()
    native, workers = CudaPreprocessBackend().wps_masked_chain_engine()
    assert native is shared
    assert workers == cpu_backend.available_cpu_count()
    _, workers = CudaPreprocessBackend(
        host_workers=3).wps_masked_chain_engine()
    assert workers == 3

    calls = []
    original = CpuPreprocessBackend.wps_masked_chain

    def spy(self, *args, **kwargs):
        calls.append(self.path)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CpuPreprocessBackend, "wps_masked_chain", spy)
    cpu = resolve_preprocess_backend("cpu")

    class Custom:
        """A backend object without its own masked-chain binding."""
        array_module = np

        def __getattr__(self, name):
            if name == "wps_masked_chain_engine":
                raise AttributeError(name)
            return getattr(cpu, name)

    snapshot, grid = _snapshot()
    reference = horiz.interpolate_era5_to_lambert(snapshot, grid, backend=cpu)
    before = len(calls)
    custom = horiz.interpolate_era5_to_lambert(snapshot, grid,
                                               backend=Custom())
    assert len(calls) - before == 6
    assert set(calls[before:]) == {shared.path}
    for name in ("SKINTEMP", "ST000007", "SM000007", "SNOW_EC", "SST",
                 "XICE"):
        assert (np.asarray(custom.fields[name]).tobytes()
                == np.asarray(reference.fields[name]).tobytes())


def test_the_receipts_name_the_masked_chain_library():
    from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend

    receipt = resolve_preprocess_backend("cpu").receipt()
    chain = receipt["masked_surface_chain"]
    assert chain["entry"] == WPS_MASKED_CHAIN_ENTRY
    assert chain["bridge"]["sha256"] == receipt["bridge"]["sha256"]


def test_the_rust_chain_carries_the_python_constants():
    source = (ROOT / "tools" / "grib1_bridge" / "src" / "wps_masked.rs"
              ).read_text(encoding="utf-8")

    def constant(name):
        found = re.search(rf"const {name}: \w+ = ([^;]+);", source)
        assert found, name
        return float(found.group(1))

    assert constant("DONOR_SPAN_TOLERANCE") == horiz._DONOR_SPAN_TOLERANCE
    assert (constant("SOURCE_ROUNDOFF_FRACTION")
            == horiz.SOURCE_ROUNDOFF_FRACTION)
    assert constant("SEARCH_DEPTH") == oracle._WPS_SEARCH_DEPTH


def test_nothing_at_run_time_imports_the_oracle():
    """The NumPy chain is a test oracle, never a silent runtime fallback."""
    offenders = []
    for base in ("gpuwm", "tilestream", "tools"):
        for path in sorted((ROOT / base).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith("gpuwm/verify/"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "wps_masked_oracle" not in text:
                continue
            tree = ast.parse(text)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [
                        alias.name for alias in node.names]
                if any("wps_masked_oracle" in name for name in names):
                    offenders.append(relative)
    assert offenders == []


def test_the_contract_marker_is_the_entry_the_backends_call():
    from gpuwm import bridges

    from gpuwm.noah_init_bridge import NOAH_SH2O_ENTRY

    # The newest default entry initializes host soil liquid water.
    assert (bridges.BRIDGE_ABI_MARKERS["gpuwm_preprocess_cpu"]
            == NOAH_SH2O_ENTRY.encode("ascii"))
    assert _native().masked_nearest_entry
    assert _native().water_blend_missing == ()
    assert _native().wps_masked_chain_entry
    assert _native().masked_stencil_entry
    assert _native().water_blend_entry
    ok, _ = bridges.bridge_abi_matches("gpuwm_preprocess_cpu", _native().path)
    assert ok


def test_the_sealed_distribution_runs_the_chain_at_both_worker_counts():
    from gpuwm import native_wrf_distribution

    receipt = native_wrf_distribution._cpu_masked_chain_self_test(_native())
    assert receipt["status"] == "PASS"
    assert receipt["worker_counts"] == [1, 3]
    assert receipt["cases"] == ["search_queue_limited",
                                "range_roundoff_at_bound"]
    assert len(receipt["output_sha256"]) == 64


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="builds an ELF stand-in")
def test_the_sealed_distribution_refuses_a_library_without_the_chain(
        tmp_path):
    from gpuwm import native_wrf_distribution

    stale = tmp_path / "libgpuwm_preprocess_cpu.so"
    stale.write_bytes(b"\x7fELF" + b"\x00" * 64
                      + b"gpuwm_preprocess_cpu_abi_version")
    with pytest.raises(RuntimeError, match="predates the masked surface chain"):
        native_wrf_distribution.cpu_backend_identity(stale)


def test_the_sealed_contract_declares_the_chain_the_receipts_name():
    from gpuwm.ingest.cpu_backend import WPS_MASKED_CHAIN_IMPLEMENTATION
    from gpuwm.native_wrf_distribution import distribution_contract

    contract = distribution_contract("linux-x86_64")
    assert (contract["preprocess_backends"]["masked_surface_chain"]
            == WPS_MASKED_CHAIN_IMPLEMENTATION)
