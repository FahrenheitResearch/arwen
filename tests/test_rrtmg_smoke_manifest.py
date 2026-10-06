"""Metadata, coverage and immutable-file guards without device access."""
from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
import pytest

from gpuwm.core.rrtmg_smoke_manifest import (
    BoundSmokeManifest, SCHEMA, validate_smoke_manifest, describe_smoke_source)

START = datetime(2026, 10, 2, 21, tzinfo=timezone.utc)


def _fixture(tmp_path, quantity="dry_mass_mixing_ratio", dtype="<f4"):
    """Tiny metadata fixture. These values are not a weather donor dataset."""
    nz, ny, nx = 2, 3, 4
    units = {"dry_mass_mixing_ratio": "ug/kg-dryair", "layer_aod": "1",
             "posted_mass_concentration": "kg/m3"}[quantity]
    def member(name, shape, unit, value=1):
        path = tmp_path / (name + ".bin")
        np.full(shape, value, dtype=dtype).tofile(path)
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dtype": dtype, "shape": list(shape), "units": unit}
    raw = {"schema": SCHEMA, "quantity": quantity, "units": units,
           "shape": [nz, ny, nx], "vertical_order": "bottom_to_top",
           "time_interpolation": "linear", "start_time": "2026-10-02T21:00:00Z",
           "provenance": {"vertical": {"eta_levels": [1.0, .5, 0.0],
               "hybrid_opt": 2, "etac": .2, "p_top": 1500.0}},
           "geometry": {"latitude": member("lat", (ny, nx), "degrees_north"),
                        "longitude": member("lon", (ny, nx), "degrees_east")},
           "frames": []}
    for number, time in enumerate(("2026-10-02T21:00:00Z", "2026-10-02T23:00:00Z")):
        frame = {"valid_time": time, "value": member(f"value-{number}", (nz, ny, nx), units)}
        if quantity == "posted_mass_concentration":
            frame["donor_p"] = member(f"p-{number}", (nz, ny, nx), "Pa", 95000)
            frame["donor_t"] = member(f"t-{number}", (nz, ny, nx), "K", 290)
        raw["frames"].append(frame)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, raw


@pytest.mark.parametrize("quantity", ("dry_mass_mixing_ratio", "layer_aod", "posted_mass_concentration"))
@pytest.mark.parametrize("dtype", ("<f4", "<f8"))
def test_full_member_metadata_is_bound_without_a_device(tmp_path, quantity, dtype):
    path, raw = _fixture(tmp_path, quantity, dtype)
    result = validate_smoke_manifest(path, START, 2)
    assert result.shape == (2, 3, 4)
    assert result.quantity == quantity
    assert result.frames[0]["value"].identity() == raw["frames"][0]["value"]
    result.verify(deep=True)


@pytest.mark.parametrize("change,error", [
    (lambda r: r.update(units="kg/m2"), "quantity/units"),
    (lambda r: r.update(quantity="column_aotk"), "three-dimensional"),
    (lambda r: r.update(vertical_order="top_to_bottom"), "bottom_to_top"),
    (lambda r: r.update(time_interpolation="hold"), "bounded linear"),
    (lambda r: r.update(start_time="2026-10-02T22:00:00Z"), "actual case start"),
    (lambda r: r.update(start_time="2026-10-02T21:00:00"), "explicitly use UTC"),
    (lambda r: r["frames"][1].update(valid_time=r["frames"][0]["valid_time"]), "strictly increasing"),
    (lambda r: r["frames"][0]["value"].update(dtype=">f4"), "little-endian"),
    (lambda r: r["frames"][0]["value"].update(shape=[1, 3, 4]), "member shape"),
    (lambda r: r["frames"][0]["value"].update(path="../outside.bin"), "remain inside"),
    (lambda r: r["frames"][0]["value"].update(sha256="0" * 64), "SHA256 differs"),
    (lambda r: r["frames"][0]["value"].update(path="missing.bin"), "absent"),
])
def test_bad_or_incomplete_input_is_refused_before_cuda(tmp_path, change, error):
    path, raw = _fixture(tmp_path)
    change(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        validate_smoke_manifest(path, START, 2)


def test_posted_density_requires_its_own_pressure_and_temperature(tmp_path):
    path, raw = _fixture(tmp_path, "posted_mass_concentration")
    del raw["frames"][0]["donor_p"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fields do not match"):
        validate_smoke_manifest(path, START, 2)


def test_missing_layers_and_run_extrapolation_are_not_filled(tmp_path):
    path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="every target vertical layer"):
        validate_smoke_manifest(path, START, 3)
    bound = object.__new__(BoundSmokeManifest)
    bound._manifest = validate_smoke_manifest(path, START, 2)
    bound.require_coverage(7200)
    with pytest.raises(ValueError, match="complete requested run"):
        bound.require_coverage(7201)
    for seconds in (-1, 7201, float("nan")):
        with pytest.raises(ValueError, match="extrapolation|finite elapsed"):
            bound.at(seconds)


def test_identity_guard_rehashes_a_same_stat_member(tmp_path):
    path, _ = _fixture(tmp_path)
    result = validate_smoke_manifest(path, START, 2)
    member = result.frames[0]["value"]
    prior = member.path.stat()
    with member.path.open("r+b") as handle:
        handle.write(np.array([2], dtype="<f4").tobytes())
    os.utime(member.path, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    with pytest.raises(ValueError, match="SHA256 changed"):
        result.verify(deep=True)


def test_vertical_values_are_bound_beyond_layer_count(tmp_path):
    path, _ = _fixture(tmp_path)
    manifest = validate_smoke_manifest(path, START, 2)
    manifest.require_vertical((1.0, .5, 0.0), 2, .2, 1500.0)
    for eta, hybrid, etac, p_top in (
        ((1.0, .4, 0.0), 2, .2, 1500.0),
        ((1.0, .5, 0.0), 1, .2, 1500.0),
        ((1.0, .5, 0.0), 2, .3, 1500.0),
        ((1.0, .5, 0.0), 2, .2, 5000.0),
    ):
        with pytest.raises(ValueError, match="vertical provenance differs"):
            manifest.require_vertical(eta, hybrid, etac, p_top)


def test_descriptor_without_case_context_binds_declared_clock_and_files(tmp_path):
    path, _ = _fixture(tmp_path)
    identity = describe_smoke_source(path)
    assert identity["native_shape"] == [2, 3, 4]
    assert identity["times"][0] == START.isoformat()
    assert identity["provenance"]["vertical"]["hybrid_opt"] == 2
    with pytest.raises(ValueError, match="actual case start"):
        describe_smoke_source(path, start_time=datetime(2026, 10, 2, 22), nz=2)
