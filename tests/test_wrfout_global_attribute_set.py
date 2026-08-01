# tests/test_wrfout_global_attribute_set.py
"""Exact global-attribute SET of a frame written through the production path.

Nothing in the suite watches the wrfout global-attribute set as a set.  Field
inventories and individual attribute values are guarded elsewhere; an
attribute ADDED to every emitted file passes all of it.  That is the change a
capsule-versus-sidecar decision would make, so the guard has to exist before
the decision is taken rather than after, or the decision is taken blind.

The golden set is whatever ``gpuwm.runtime._global_wrf_attrs`` +
``WrfoutWriter`` emit at this commit, read back off a real NETCDF4_CLASSIC
file with netCDF4 rather than off the attrs dict: the writer adds its own
header attributes on top of the caller's, and only the file knows the union.
The mutation control adds one synthetic global and requires the guard to fail.
"""

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.io.wrfout import WrfoutWriter
from gpuwm.runtime import _global_wrf_attrs

_GOLDEN_PATH = (Path(__file__).parent / "data"
                / "wrfout_global_attribute_set_v1.json")

_NX, _NY, _NZ = 4, 3, 5


def _grid():
    """Projection identity a real-case domain carries into the writer."""

    return SimpleNamespace(
        truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
        ref_lat=39.0, ref_lon=-96.5,
        wrf_map_proj=1, map_proj_char="Lambert Conformal",
        moad_cen_lat=39.0, cen_lat=39.0, cen_lon=-96.5)


def _domain():
    # A real RunConfig, not a namespace mock: wrf_global_attrs stamps the
    # physics-selector globals off the resolved run configuration, so a
    # hand-listed mock would silently fall behind every selector the
    # production reader grows.  Defaults are the resolved truth here.
    run = RunConfig(nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                    dt=6.0, ztop=10000.0, run_seconds=12.0)
    return SimpleNamespace(
        grid_id=2, parent_id=1, i_parent_start=5, j_parent_start=5,
        parent_grid_ratio=3, run=run)


def _coord():
    return SimpleNamespace(hybrid_opt=2, etac=0.2)


def _write_production_frame(path, *, extra_attrs=None):
    """One frame through the production attribute assembly and writer."""

    attrs = _global_wrf_attrs(
        _grid(), datetime.datetime(2020, 6, 1, 12),
        domain=_domain(), coord=_coord())
    if extra_attrs is not None:
        attrs = {**attrs, **extra_attrs}
    frame = {
        "T2": np.full((_NY, _NX), 290.0, np.float32),
        "PSFC": np.full((_NY, _NX), 98000.0, np.float32),
    }
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs, soil_layers=4) as writer:
        writer.write_frame("2020-06-01_12:00:00", frame)
    return path


def _emitted_global_attribute_set(path):
    with netCDF4.Dataset(path, "r") as dataset:
        return sorted(dataset.ncattrs())


def test_emitted_global_attribute_set_is_exactly_the_golden_set(tmp_path):
    """The SET, not a subset: an added or dropped global is a failure."""

    path = _write_production_frame(tmp_path / "wrfout_d02_frame.nc")
    emitted = _emitted_global_attribute_set(path)
    golden = json.loads(_GOLDEN_PATH.read_text())["global_attributes"]
    assert emitted == sorted(golden), (
        "the wrfout global-attribute set moved; added="
        f"{sorted(set(emitted) - set(golden))} removed="
        f"{sorted(set(golden) - set(emitted))}")


def test_one_synthetic_global_attribute_fails_the_guard(tmp_path):
    """MUTATION CONTROL: the guard can see exactly the change it exists for.

    A single extra global -- the shape an embedded evidence-tier attribute
    would take -- must make the set assertion above fail.  Without this the
    guard could be a subset check that never fires.
    """

    path = _write_production_frame(
        tmp_path / "wrfout_d02_mutated.nc",
        extra_attrs={"GPUWM_SYNTHETIC_CONTROL_ATTRIBUTE": "mutation-control"})
    emitted = _emitted_global_attribute_set(path)
    golden = json.loads(_GOLDEN_PATH.read_text())["global_attributes"]
    assert "GPUWM_SYNTHETIC_CONTROL_ATTRIBUTE" in emitted
    assert emitted != sorted(golden), (
        "the guard did not see a synthetic global attribute; it is not an "
        "exact-set assertion")


def test_dropping_one_global_attribute_fails_the_guard(tmp_path):
    """MUTATION CONTROL, other direction: a removed global must fail too."""

    path = _write_production_frame(tmp_path / "wrfout_d02_frame.nc")
    emitted = _emitted_global_attribute_set(path)
    golden = json.loads(_GOLDEN_PATH.read_text())["global_attributes"]
    shortened = sorted(set(golden) - {sorted(golden)[0]})
    assert emitted != shortened


def test_golden_set_carries_no_case_token():
    """Certification-path data stays generic; the case lives in configs."""

    text = _GOLDEN_PATH.read_text().lower()
    for token in ("real74", "1974", "ohio", "hrrr"):
        assert token not in text, (
            f"case token {token!r} leaked into the wrfout attribute golden set")
