"""A grid prepared on one machine is the same grid on the machine that runs it.

Positions that come out of projection arithmetic (a nest's anchor, a centre,
the latitude/longitude extremes) round differently in the last digit on
different math libraries.  Every identity check on a prepared tree must
accept those, and must still refuse any grid that differs in a real way:
moved by a cell, re-spaced, re-sized, or anchored at another given value.
"""

from __future__ import annotations

from datetime import date
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.native_wrf_contract import (
    load_native_static_cache,
    native_geometry_contract,
    native_static_export_fields,
    verify_native_static_receipt,
    write_native_geometry_receipt,
    write_native_static_cache,
)
from gpuwm.static import highres_production as highres_owner
from gpuwm.static.corridor import grid_identity_probes, grid_probe_drift
from gpuwm.static.grid_identity import (
    GRID_POSITION_TOLERANCE_CELLS,
    position_tolerance_deg,
)
from gpuwm.static.lambert import LambertGrid
from gpuwm.wrf_direct import _load_static_geometry_receipt


#: One grid's northern edge and one nest's anchor, each as two math
#: libraries computed it for the grids built by ``_root`` and ``_nest``.
D01_LAT_MAX = (41.73852197541409, 41.7385219754141)
D02_REF_LAT = (40.35613748598461, 40.356137485984604)


def _root(**changes):
    params = dict(
        ref_lat=40.711, ref_lon=-74.08285000000001, truelat1=30.71,
        truelat2=50.71, stand_lon=-74.08285000000001, dx=1000.0,
        dy=1000.0, e_we=227, e_sn=227)
    params.update(changes)
    return LambertGrid(**params)


def _nest(root=None, i_start=75, j_start=75, ratio=2, e_we=157, e_sn=157):
    root = _root() if root is None else root
    return root.nest(i_start, j_start, ratio, e_we, e_sn,
                     resolved_dx=root.dx / ratio,
                     resolved_dy=root.dy / ratio)


def _cfg(grid, nz=59):
    return SimpleNamespace(nx=grid.e_we - 1, ny=grid.e_sn - 1, nz=nz,
                           dx=grid.dx, dy=grid.dy)


def _ulps(value, steps):
    """``value`` moved ``steps`` representable doubles up (or down)."""
    toward = math.inf if steps > 0 else -math.inf
    for _ in range(abs(steps)):
        value = math.nextafter(value, toward)
    return value


def _other_machine(geometry, grid):
    """The same contract as a machine that rounds differently computes it."""
    other = json.loads(json.dumps(geometry))
    other["lat_range"] = [_ulps(other["lat_range"][0], 3),
                          _ulps(other["lat_range"][1], -2)]
    other["lon_range"] = [_ulps(other["lon_range"][0], -4),
                          _ulps(other["lon_range"][1], 1)]
    if not grid.anchor_is_given:
        other["ref_lat"] = _ulps(other["ref_lat"], 1)
        other["ref_lon"] = _ulps(other["ref_lon"], -3)
    if not grid.center_is_given:
        other["center_lat"] = _ulps(other["center_lat"], -2)
        other["center_lon"] = _ulps(other["center_lon"], 2)
    return other


def _sealed(tmp_path, grid, geometry=None):
    """A static cache and its geometry receipt, optionally re-worded."""
    static_path = tmp_path / "native-static.npz"
    receipt_path = tmp_path / "geometry-receipt.json"
    write_native_static_cache(
        static_path, {"HGT_M": np.zeros((grid.e_sn - 1, grid.e_we - 1))})
    receipt = write_native_geometry_receipt(
        receipt_path, grid, _cfg(grid), static_path)
    if geometry is not None:
        receipt["geometry"] = geometry(receipt["geometry"])
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    return receipt_path, static_path


@pytest.mark.parametrize("kind", ["root", "nest", "translated"])
def test_the_same_grid_rounded_by_another_machine_is_accepted(tmp_path, kind):
    grid = {"root": _root, "nest": _nest,
            "translated": lambda: _nest().translated(4, -3)}[kind]()
    receipt_path, static_path = _sealed(
        tmp_path, grid, lambda geometry: _other_machine(geometry, grid))

    verify_native_static_receipt(receipt_path, static_path, grid, _cfg(grid))
    _load_static_geometry_receipt(
        receipt_path, static_path,
        expected_geometry=native_geometry_contract(grid, _cfg(grid)),
        expected_grid=grid)


def test_the_two_measured_spellings_of_one_grid_are_the_same_grid(tmp_path):
    root, nest = _root(), _nest()
    here = native_geometry_contract(root, _cfg(root))["lat_range"][1]
    other = D01_LAT_MAX[1] if here == D01_LAT_MAX[0] else D01_LAT_MAX[0]

    def swap_edge(geometry):
        geometry["lat_range"][1] = other
        return geometry

    (tmp_path / "d01").mkdir()
    receipt_path, static_path = _sealed(tmp_path / "d01", root, swap_edge)
    verify_native_static_receipt(receipt_path, static_path, root, _cfg(root))

    here = native_geometry_contract(nest, _cfg(nest))["ref_lat"]
    other = D02_REF_LAT[1] if here == D02_REF_LAT[0] else D02_REF_LAT[0]

    def swap_anchor(geometry):
        geometry["ref_lat"] = other
        return geometry

    (tmp_path / "d02").mkdir()
    receipt_path, static_path = _sealed(tmp_path / "d02", nest, swap_anchor)
    verify_native_static_receipt(receipt_path, static_path, nest, _cfg(nest))


_DIFFERENT_GRIDS = {
    "nest one parent cell east": (_nest, lambda: _nest(i_start=76)),
    "nest one parent cell north": (_nest, lambda: _nest(j_start=76)),
    "root one cell over": (_root, lambda: _root().translated(1, 0)),
    "translated nest one cell further": (
        lambda: _nest().translated(4, -3), lambda: _nest().translated(5, -3)),
    "different dx": (_root, lambda: _root(dx=999.0, dy=999.0)),
    "different nx": (_root, lambda: _root(e_we=228)),
    "root anchor moved far below a cell": (
        _root, lambda: _root(ref_lat=40.711 + 1.0e-9)),
    "different ratio at the same start": (
        lambda: _nest(ratio=2, e_we=157, e_sn=157),
        lambda: _nest(ratio=3, e_we=157, e_sn=157)),
}


@pytest.mark.parametrize("name", sorted(_DIFFERENT_GRIDS))
@pytest.mark.parametrize("with_definition", [True, False])
def test_a_grid_that_differs_is_still_refused(tmp_path, name,
                                               with_definition):
    sealed_grid, run_grid = (build() for build in _DIFFERENT_GRIDS[name])

    def reword(geometry):
        if not with_definition:
            geometry.pop("definition")
        return geometry

    receipt_path, static_path = _sealed(tmp_path, sealed_grid, reword)
    with pytest.raises(ValueError, match="geometry differs from target"):
        verify_native_static_receipt(
            receipt_path, static_path, run_grid, _cfg(run_grid))
    with pytest.raises(ValueError, match="geometry differs from namelist"):
        _load_static_geometry_receipt(
            receipt_path, static_path,
            expected_geometry=native_geometry_contract(
                run_grid, _cfg(run_grid)),
            expected_grid=run_grid)


def test_a_receipt_written_before_definitions_is_held_to_its_positions(
        tmp_path):
    nest = _nest()

    def older(geometry):
        geometry = _other_machine(geometry, nest)
        geometry.pop("definition")
        return geometry

    receipt_path, static_path = _sealed(tmp_path, nest, older)
    verify_native_static_receipt(receipt_path, static_path, nest, _cfg(nest))


def test_the_tolerance_is_a_small_fraction_of_a_cell():
    tolerance = position_tolerance_deg(500.0, 500.0)
    cell_deg = 500.0 / (6370000.0 * math.pi / 180.0)
    assert GRID_POSITION_TOLERANCE_CELLS == 1.0e-3
    assert tolerance == pytest.approx(1.0e-3 * cell_deg)
    # Three orders below the smallest real move, and still orders above
    # the last-digit disagreement measured on real grids even at 10 m.
    assert position_tolerance_deg(10.0, 10.0) > 1.0e4 * abs(
        D01_LAT_MAX[0] - D01_LAT_MAX[1])


def test_a_grid_definition_holds_only_given_values():
    root, nest = _root(), _nest()
    moved = nest.translated(4, -3)
    assert root.anchor_is_given and root.center_is_given
    assert not nest.anchor_is_given and not nest.center_is_given
    assert not moved.anchor_is_given and not moved.center_is_given
    assert root.translated(2, 2).anchor_is_given
    assert not root.translated(2, 2).center_is_given

    definition = nest.definition()
    assert definition["nest_of"] == root.definition()
    assert (definition["i_parent_start"], definition["j_parent_start"],
            definition["parent_grid_ratio"]) == (75, 75, 2)
    assert "ref_lat" not in definition
    assert moved.definition()["translated_from"] == definition
    assert moved.definition()["offset_cells"] == [4, -3]
    for grid in (root, nest, moved):
        assert json.loads(json.dumps(grid.definition())) == grid.definition()


def _highres_config(tmp_path):
    source = tmp_path / "case.toml"
    source.write_text(
        "[static.highres]\nenabled = true\n"
        'cache_root = "not-created/cache"\nfields = "terrain"\n',
        encoding="utf-8")
    return highres_owner.load_static_highres(source)


def test_a_sealed_high_resolution_grid_record_survives_another_machine(
        tmp_path):
    config, day = _highres_config(tmp_path), date(2026, 9, 27)

    def sealed(grid, domain_id, reword=None):
        record = highres_owner._grid_identity(grid, domain_id)
        if reword is not None:
            record = reword(record)
        return {"highres": {"status": "APPLIED", "config": config.echo(),
                            "case_date": day.isoformat(), "grid": record}}

    def rounded(record):
        record = dict(record)
        record["ref_lat"] = _ulps(record["ref_lat"], 1)
        record["ref_lon"] = _ulps(record["ref_lon"], -1)
        return record

    nest = _nest()
    highres_owner.require_prepared_highres(
        sealed(nest, 2, rounded), nest, config=config, domain_id=2,
        case_date=day)
    # The receipt's file identity does not move with the rounding either.
    assert highres_owner._grid_record_key(
        rounded(highres_owner._grid_identity(nest, 2))
    ) == highres_owner._grid_record_key(highres_owner._grid_identity(nest, 2))

    for run_grid in (_nest(i_start=76), _nest(ratio=3)):
        with pytest.raises(ValueError, match="do not bind"):
            highres_owner.require_prepared_highres(
                sealed(nest, 2), run_grid, config=config, domain_id=2,
                case_date=day)
    # A root's anchor is its given value and is still held to the bit.
    root = _root()
    highres_owner.require_prepared_highres(
        sealed(root, 1), root, config=config, domain_id=1, case_date=day)
    with pytest.raises(ValueError, match="do not bind"):
        highres_owner.require_prepared_highres(
            sealed(root, 1, rounded), root, config=config, domain_id=1,
            case_date=day)


def test_corridor_grid_probes_accept_rounding_and_refuse_a_move():
    corridor = _nest().translated(-6, -6, e_we=170, e_sn=170)
    probes = grid_identity_probes(corridor)
    rounded = {name: [_ulps(lat, 2), _ulps(lon, -2)]
               for name, (lat, lon) in probes.items()}

    assert grid_probe_drift(probes, corridor) == {}
    assert grid_probe_drift(rounded, corridor) == {}
    moved = grid_identity_probes(
        _nest().translated(-5, -6, e_we=170, e_sn=170))
    assert grid_probe_drift(moved, corridor)
    missing = dict(probes)
    missing.pop("center")
    assert grid_probe_drift(missing, corridor)
    assert grid_probe_drift(None, corridor)


class _RoundsDifferently:
    """A grid whose math library puts one value across a float32 midpoint.

    ``regen`` is what this machine computes; the static cache carries the
    preparation machine's bytes two doubles away on the other side of the
    float32 rounding boundary, so the model state (float32) built from
    one differs from the state built from the other.
    """

    def __init__(self, stored):
        self.stored = stored

    def _regen(self, name):
        value = np.array(self.stored[name], dtype=np.float64, copy=True)
        value.flat[0] = _ulps(float(value.flat[0]), -2)
        return value

    def mapfac_m(self):
        return self._regen("MAPFAC_M")

    def mapfac_u(self):
        return self._regen("MAPFAC_U")

    def mapfac_v(self):
        return self._regen("MAPFAC_V")

    def coriolis_m(self):
        return self._regen("F"), self._regen("E")

    def rotation_m(self):
        return self._regen("SINALPHA"), self._regen("COSALPHA")


def test_prepared_geometry_fields_keep_their_bytes_on_another_machine():
    # A value one double above a float32 rounding midpoint: rounds up in
    # float32, while the same value two doubles lower rounds down.
    low = np.float32(1.0000124)
    midpoint = (float(low) + float(np.nextafter(low, np.float32(2)))) / 2.0
    stored_value = _ulps(midpoint, 1)
    stored = {
        name: np.full(shape, stored_value)
        for name, shape in (("MAPFAC_M", (3, 4)), ("MAPFAC_U", (3, 5)),
                            ("MAPFAC_V", (4, 4)), ("F", (3, 4)),
                            ("E", (3, 4)), ("SINALPHA", (3, 4)),
                            ("COSALPHA", (3, 4)))}
    grid = _RoundsDifferently(stored)
    assert np.float32(grid.mapfac_m().flat[0]) != np.float32(stored_value)

    fields = native_static_export_fields(dict(stored), grid)
    for name, value in stored.items():
        assert np.asarray(fields[name]).tobytes() == value.tobytes(), name
        assert np.array_equal(np.asarray(fields[name], dtype=np.float32),
                              value.astype(np.float32)), name


def test_a_static_cache_republishes_its_own_bytes_on_another_machine(
        tmp_path):
    from test_native_wrf_contract import _complete_native_static

    grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=12_000.0, dy=12_000.0, e_we=5, e_sn=4)
    prepared = native_static_export_fields(_complete_native_static(grid), grid)
    # The preparation machine rounded its geometry fields a few doubles
    # away from this one.
    for name in ("MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E"):
        prepared[name] = np.vectorize(lambda v: _ulps(float(v), 2))(
            np.asarray(prepared[name], dtype=np.float64))
    first = tmp_path / "native-static.npz"
    write_native_static_cache(first, prepared)

    loaded = load_native_static_cache(first, grid, 3, 4)
    second = tmp_path / "republished" / "native-static.npz"
    write_native_static_cache(second, native_static_export_fields(loaded, grid))
    assert second.read_bytes() == first.read_bytes()


# -- observation grids --------------------------------------------------------


def _machine_rounded(base, ulps=2, straddle=None):
    """``base``'s projection class, rounding latitudes/longitudes elsewhere.

    ``straddle`` puts the first latitude one double to the given side of
    the float32 rounding midpoint nearest it, which is where two machines'
    stored float32 coordinates come out one step apart.
    """

    class Rounded(type(base)):
        def latlon_mass(self):
            lat, lon = super().latlon_mass()
            lat = np.array(lat, dtype=np.float64, copy=True)
            lon = np.array(lon, dtype=np.float64, copy=True)
            for _ in range(ulps):
                lat = np.nextafter(lat, np.inf)
                lon = np.nextafter(lon, -np.inf)
            if straddle is not None:
                low = np.float32(lat.flat[0])
                high = np.nextafter(low, np.float32(np.inf))
                midpoint = (float(low) + float(high)) / 2.0
                lat.flat[0] = math.nextafter(
                    midpoint, math.inf if straddle > 0 else -math.inf)
            return lat, lon

    grid = Rounded.__new__(Rounded)
    # Everything but a bridge handle, which belongs to ``base`` and is
    # released with it; the copy opens its own.
    grid.__dict__.update({name: value for name, value in base.__dict__.items()
                          if name != "_rust_grid_handle"})
    return grid


def _observation_grid(projection, z_top=12000.0, nz=6):
    from gpuwm.obs.target_grid import TargetGrid

    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, z_top, nz + 1), name="analysis")


def _gridded_document(grid, identity=None):
    """What a reader hands the filter for a file written against ``grid``."""

    from gpuwm.obs.radar_grid import _coordinate_digests

    return {
        "grid_identity_sha256": (grid.identity_sha256() if identity is None
                                 else identity),
        "grid_coordinate_sha256": _coordinate_digests(grid),
        "variables": {"XLAT": grid.lat.astype(np.float32),
                      "XLONG": grid.lon.astype(np.float32)},
    }


@pytest.mark.parametrize("projection", ["root", "nest"])
def test_an_observation_grid_rounded_by_another_machine_is_the_same_grid(
        projection):
    from gpuwm.obs.radar_grid import require_grid_binding

    base = {"root": _root, "nest": _nest}[projection]()
    here = _observation_grid(base)
    for there in (_observation_grid(_machine_rounded(base, straddle=1)),
                  _observation_grid(_machine_rounded(base, straddle=-1))):
        assert here.lat.tobytes() != there.lat.tobytes()
        assert here.legacy_identity_sha256() != there.legacy_identity_sha256()
        assert here.identity_sha256() == there.identity_sha256()
        here.require_identity(there.identity_sha256())
        require_grid_binding(_gridded_document(there), here)
        require_grid_binding(_gridded_document(here), there)
    # The first latitude rounds to float32 one step apart on the two sides
    # of the midpoint, which a digest of the stored coordinates cannot see
    # past and the one-step allowance does.
    up = _observation_grid(_machine_rounded(base, straddle=1))
    down = _observation_grid(_machine_rounded(base, straddle=-1))
    assert np.float32(up.lat.flat[0]) != np.float32(down.lat.flat[0])
    require_grid_binding(_gridded_document(up), down)


def test_a_product_written_before_v2_is_still_read_where_it_was_made():
    from gpuwm.obs.radar_grid import require_grid_binding

    grid = _observation_grid(_nest())
    require_grid_binding(
        _gridded_document(grid, grid.legacy_identity_sha256()), grid)
    grid.require_identity(grid.legacy_identity_sha256())


_DIFFERENT_OBSERVATION_GRIDS = {
    "nest one parent cell east": lambda: _observation_grid(_nest(i_start=76)),
    "root one cell over": lambda: _observation_grid(_root().translated(1, 0)),
    "different dx": lambda: _observation_grid(_root(dx=999.0, dy=999.0)),
    "different nx": lambda: _observation_grid(_root(e_we=228)),
    "different vertical ladder": lambda: _observation_grid(
        _root(), z_top=12500.0),
}


@pytest.mark.parametrize("name", sorted(_DIFFERENT_OBSERVATION_GRIDS))
def test_an_observation_grid_that_differs_is_still_refused(name):
    from gpuwm.obs.radar_grid import require_grid_binding
    from gpuwm.obs.target_grid import GridMismatchError

    here = _observation_grid(
        _nest() if name.startswith("nest") else _root())
    other = _DIFFERENT_OBSERVATION_GRIDS[name]()
    assert not here.matches_identity(other.identity_sha256())
    with pytest.raises(GridMismatchError):
        here.require_identity(other.identity_sha256())
    with pytest.raises(GridMismatchError):
        require_grid_binding(_gridded_document(other), here)


def test_an_observation_grid_holds_its_coordinates_to_its_projection():
    from gpuwm.obs.radar_grid import require_grid_binding
    from gpuwm.obs.target_grid import GridMismatchError, TargetGrid

    grid = _observation_grid(_root())
    fields = {name: getattr(grid, name) for name in (
        "name", "map_proj", "nx", "ny", "nz", "dx_m", "dy_m", "ref_lat",
        "ref_lon", "truelat1", "truelat2", "stand_lon", "lat", "lon", "z_w",
        "terrain_m", "projection", "source")}
    # A cell's width, and a small fraction of one, off the projection.
    for shift in (1000.0 / 111195.0, 0.05 * 1000.0 / 111195.0):
        with pytest.raises(ValueError, match="mass points"):
            TargetGrid(**{**fields, "lat": grid.lat + shift})
    # A file whose stored coordinates sit four float32 steps off (under
    # two metres), with a digest table that names them, is refused too.
    from gpuwm.obs.radar_grid import _digest

    document = _gridded_document(grid)
    moved = grid.lat.astype(np.float32).copy()
    moved[0, 0] += np.float32(4 * np.spacing(np.float32(moved[0, 0])))
    document["variables"]["XLAT"] = moved
    document["grid_coordinate_sha256"]["XLAT"] = _digest(moved, np.float32)
    with pytest.raises(GridMismatchError, match="XLAT"):
        require_grid_binding(document, grid)


def test_a_sealed_hrrr_source_window_survives_another_machine():
    from gpuwm.ingest.hrrr_target import (HrrrTargetDomain,
                                          required_hrrr_source_window)

    window = required_hrrr_source_window(HrrrTargetDomain.legacy_500x500())
    record = json.loads(json.dumps(window.to_dict()))
    rounded = {**record,
               "target_source_i_range": [
                   _ulps(record["target_source_i_range"][0], 3),
                   _ulps(record["target_source_i_range"][1], -2)],
               "target_source_j_range": [
                   _ulps(record["target_source_j_range"][0], -1),
                   _ulps(record["target_source_j_range"][1], 4)]}
    assert window.matches_record(record)
    assert window.matches_record(rounded)

    shifted_crop = {**record, "zero_based_inclusive": {
        "i": [record["zero_based_inclusive"]["i"][0] + 1,
              record["zero_based_inclusive"]["i"][1] + 1],
        "j": record["zero_based_inclusive"]["j"]}}
    moved_extent = {**record, "target_source_i_range": [
        record["target_source_i_range"][0] + 0.01,
        record["target_source_i_range"][1] + 0.01]}
    missing = dict(record)
    missing.pop("surface_fallback_radius_cells")
    for different in (shifted_crop, moved_extent, missing, None):
        assert not window.matches_record(different)
