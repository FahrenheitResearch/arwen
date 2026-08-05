"""The observation battery's ingest lane: packs, front doors, sources.

Three layers, tested at the level each can be tested honestly:

* the pack reader, against bytes this file builds, including the malformed
  ones a reader must refuse;
* the front-door locator, against the resolution ladder and the built
  binaries' own ``--abi`` when a checkout has them;
* the source classes, against packs built here, for the frame matching and
  the re-hash that the promotion rule's integrity clause turns on.

Where a test needs the scoring lane's contract module it skips rather than
fails: the two lanes land in separate waves, and an ingest suite that goes
red because the scorer is not merged yet reports the wrong thing.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gpuwm.obs import frontdoor, obspack, sources


# --------------------------------------------------------------- fixtures

def build_pack(meta: dict, arrays: dict[str, np.ndarray]) -> bytes:
    """Assemble a GPWMOBS1 pack the way the Rust writer does."""

    payload = b""
    index = {}
    for key, array in arrays.items():
        dtype = "<u1" if array.dtype == np.uint8 else "<f8"
        data = array.astype(np.dtype(dtype)).tobytes(order="C")
        index[key] = {"dtype": dtype, "shape": list(array.shape),
                      "offset": len(payload), "bytes": len(data)}
        payload += data
    meta = dict(meta)
    meta["arrays"] = index
    meta["payload_bytes"] = len(payload)
    meta["content_sha256"] = hashlib.sha256(payload).hexdigest()
    blob = json.dumps(meta).encode()
    header = (obspack.PACK_MAGIC
              + struct.pack("<II", obspack.PACK_VERSION, len(blob))
              + struct.pack("<Q", len(payload)))
    header += b"\0" * (obspack.PACK_HEADER_BYTES - len(header))
    return header + blob + payload


def grid_pack_bytes(valid_time="2024-05-21T21:00:00", *, quantity=None,
                    values=None, uri="obs.grib2.gz", sha256=None,
                    extra=None) -> bytes:
    values = np.array([[10.0, 20.0], [30.0, 40.0]]) if values is None else values
    valid = np.ones(values.shape, dtype=np.uint8)
    meta = {
        "schema": obspack.GRID_SCHEMA,
        "status": "READY",
        "quantity": quantity or sources.QUANTITY_COMPOSITE_REFLECTIVITY,
        "units": "dBZ",
        "valid_time": valid_time,
        "provenance": {
            "source": "mrms", "product": "MergedReflectivityQCComposite_00.50",
            "uri": uri, "sha256": sha256 or ("a" * 64),
            "fetched_at": "2026-08-03T12:00:00",
            "is_stub": False, "stub_reason": "",
        },
    }
    meta.update(extra or {})
    return build_pack(meta, {"values": values, "valid": valid})


def geo_pack_bytes(shape=(2, 2)) -> bytes:
    latitude = np.linspace(41.0, 42.0, shape[0] * shape[1]).reshape(shape)
    longitude = np.linspace(-94.0, -93.0, shape[0] * shape[1]).reshape(shape)
    meta = {"schema": obspack.GEO_SCHEMA, "status": "READY",
            "source_product": "test"}
    return build_pack(meta, {"latitude": latitude, "longitude": longitude})


# ------------------------------------------------------------ pack reader

def test_a_pack_round_trips_with_its_arrays_and_dtypes(tmp_path):
    path = tmp_path / "one.obspack"
    path.write_bytes(grid_pack_bytes())
    pack = obspack.read_grid_pack(path)
    assert pack.schema == obspack.GRID_SCHEMA
    assert pack.meta["valid_time"] == "2024-05-21T21:00:00"
    values = pack.array("values")
    assert values.dtype == np.float64
    assert values.shape == (2, 2)
    assert values[1, 1] == 40.0
    assert pack.array("valid").dtype == np.uint8


def test_a_sweep_pack_is_refused_by_its_magic(tmp_path):
    path = tmp_path / "sweeps.rdrpack"
    path.write_bytes(b"GPWMRDR1" + grid_pack_bytes()[8:])
    with pytest.raises(ValueError, match="magic"):
        obspack.read_pack(path)


def test_a_truncated_pack_is_refused_rather_than_read_short(tmp_path):
    path = tmp_path / "cut.obspack"
    path.write_bytes(grid_pack_bytes()[:-8])
    with pytest.raises(ValueError, match="header declares"):
        obspack.read_pack(path)


def test_a_payload_that_changed_under_its_digest_is_refused(tmp_path):
    raw = bytearray(grid_pack_bytes())
    raw[-1] ^= 0xFF
    path = tmp_path / "rotten.obspack"
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="digest mismatch"):
        obspack.read_pack(path)


def test_a_byteswapped_dtype_is_refused_rather_than_read_backwards(tmp_path):
    # numpy would accept '>f8' and silently return every value byte-swapped,
    # which is the one corruption that produces plausible-looking finite
    # numbers rather than an error.
    raw = grid_pack_bytes().replace(b'"<f8"', b'">f8"')
    path = tmp_path / "swapped.obspack"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="does not read"):
        obspack.read_pack(path)


def test_a_future_pack_version_is_refused_by_number(tmp_path):
    raw = bytearray(grid_pack_bytes())
    raw[8:12] = struct.pack("<I", obspack.PACK_VERSION + 1)
    path = tmp_path / "future.obspack"
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="pack version"):
        obspack.read_pack(path)


def test_a_geo_pack_comes_back_as_two_matching_float64_grids(tmp_path):
    path = tmp_path / "grid.obspack"
    path.write_bytes(geo_pack_bytes())
    latitude, longitude = obspack.read_geo_pack(path)
    assert latitude.dtype == longitude.dtype == np.float64
    assert latitude.shape == longitude.shape == (2, 2)
    # And a geometry pack is not a field: asking for one as the other is
    # refused by schema rather than returning coordinates as observations.
    with pytest.raises(ValueError, match="expected"):
        obspack.read_grid_pack(path)


# ------------------------------------------------------------ front doors

def test_every_front_door_declares_a_distinct_name_env_and_abi():
    doors = list(frontdoor.FRONT_DOORS.values())
    assert len(doors) == 3
    for attribute in ("name", "env_var", "abi_marker"):
        values = [getattr(door, attribute) for door in doors]
        assert len(set(values)) == len(values), f"{attribute} is not distinct"
    for door in doors:
        assert door.env_var.startswith("GPUWM_RW_")
        assert door.abi_marker.startswith("gpuwm-obs.")


def test_the_resolution_ladder_prefers_the_environment_override(tmp_path,
                                                                monkeypatch):
    door = frontdoor.MRMS
    named = tmp_path / frontdoor.executable_name(door.name)
    monkeypatch.setenv(door.env_var, str(named))
    assert door.candidates()[0] == named
    # A named file that does not exist is a hard error, never a fall-through.
    with pytest.raises(FileNotFoundError, match=door.env_var):
        door.find()
    named.write_bytes(b"")
    assert door.find() == named.resolve()


def test_a_front_door_without_a_binary_offers_this_installs_remedy(monkeypatch):
    door = frontdoor.STAGE4
    monkeypatch.delenv(door.env_var, raising=False)
    monkeypatch.setattr(frontdoor.FrontDoor, "find", lambda self: None)
    with pytest.raises(RuntimeError) as caught:
        door.require()
    message = str(caught.value)
    assert door.name in message
    assert "cargo build" in message


@pytest.mark.parametrize("instrument", sorted(frontdoor.FRONT_DOORS))
def test_a_built_front_door_matches_the_abi_this_wrapper_expects(instrument):
    """The wrapper's marker against the binary's own, when one is built.

    This is the check that catches a Python half upgraded past its Rust
    half: both sides carry the contract string, and only running the binary
    can prove they still agree.
    """

    door = frontdoor.FRONT_DOORS[instrument]
    binary = door.find()
    if binary is None:
        pytest.skip(f"{door.name} is not built in this checkout")
    ok, detail = door.probe(binary)
    assert ok, detail
    printed = subprocess.run([str(binary), "--abi"], capture_output=True,
                             text=True, timeout=30)
    assert printed.stdout.strip() == door.abi_marker


@pytest.mark.parametrize("instrument", sorted(frontdoor.FRONT_DOORS))
def test_a_built_front_door_fails_closed_on_an_unknown_subcommand(instrument):
    door = frontdoor.FRONT_DOORS[instrument]
    binary = door.find()
    if binary is None:
        pytest.skip(f"{door.name} is not built in this checkout")
    result = subprocess.run([str(binary), "definitely-not-a-subcommand"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "unknown subcommand" in result.stderr


# ---------------------------------------------------------------- sources

def _source_tree(tmp_path, times, quantity=None, units="dBZ"):
    geo = tmp_path / "grid.obspack"
    geo.write_bytes(geo_pack_bytes())
    packs = []
    for when in times:
        path = tmp_path / f"f{when.replace(':', '')}.obspack"
        path.write_bytes(grid_pack_bytes(valid_time=when, quantity=quantity))
        packs.append(path)
    return packs, geo


def test_a_gridded_source_indexes_its_packs_by_their_own_valid_time(tmp_path):
    packs, geo = _source_tree(
        tmp_path, ["2024-05-21T21:00:37", "2024-05-21T20:58:35",
                   "2024-05-21T21:02:40"])
    source = sources.MrmsCompositeSource(packs, geo)
    assert source.quantity() == sources.QUANTITY_COMPOSITE_REFLECTIVITY
    # Ascending, whatever order the paths arrived in.
    assert source.valid_times() == ("2024-05-21T20:58:35",
                                    "2024-05-21T21:00:37",
                                    "2024-05-21T21:02:40")


def test_two_packs_claiming_one_valid_time_are_refused(tmp_path):
    packs, geo = _source_tree(tmp_path, ["2024-05-21T21:00:37"])
    twin = tmp_path / "twin.obspack"
    twin.write_bytes(grid_pack_bytes(valid_time="2024-05-21T21:00:37"))
    with pytest.raises(ValueError, match="same valid time"):
        sources.MrmsCompositeSource([*packs, twin], geo)


def test_an_empty_source_is_refused_rather_than_scoring_nothing(tmp_path):
    geo = tmp_path / "grid.obspack"
    geo.write_bytes(geo_pack_bytes())
    with pytest.raises(ValueError, match="at least one pack"):
        sources.MrmsCompositeSource([], geo)


def test_a_pack_of_the_wrong_quantity_is_refused_by_its_own_metadata(tmp_path):
    packs, geo = _source_tree(
        tmp_path, ["2024-05-21T21:00:00"],
        quantity=sources.QUANTITY_PRECIPITATION_ACCUMULATION)
    with pytest.raises(ValueError, match="this source serves"):
        sources.MrmsCompositeSource(packs, geo)


def test_the_frame_search_refuses_a_distant_match(tmp_path):
    packs, geo = _source_tree(tmp_path, ["2024-05-21T21:00:37"])
    source = sources.MrmsCompositeSource(packs, geo, match_seconds=240)
    nearest = source._nearest(sources._parse_time("2024-05-21T21:00:00"))
    assert nearest.pack_path == packs[0]
    with pytest.raises(LookupError, match="refusing rather than reaching"):
        source._nearest(sources._parse_time("2024-05-21T21:10:00"))


def test_stage4_refuses_to_mix_accumulation_windows(tmp_path):
    geo = tmp_path / "grid.obspack"
    geo.write_bytes(geo_pack_bytes())
    hourly = tmp_path / "h.obspack"
    hourly.write_bytes(grid_pack_bytes(
        valid_time="2024-05-21T21:00:00",
        quantity=sources.QUANTITY_PRECIPITATION_ACCUMULATION,
        extra={"accumulation_hours": 1}))
    six = tmp_path / "s.obspack"
    six.write_bytes(grid_pack_bytes(
        valid_time="2024-05-21T18:00:00",
        quantity=sources.QUANTITY_PRECIPITATION_ACCUMULATION,
        extra={"accumulation_hours": 6}))
    with pytest.raises(ValueError, match="accumulation"):
        sources.Stage4PrecipSource([hourly, six], geo, accumulation_hours=1)


def test_verify_rehashes_the_named_object_and_catches_a_change(tmp_path):
    artifact = tmp_path / "obs.grib2.gz"
    artifact.write_bytes(b"the archive object as it arrived")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    packs, geo = _source_tree(tmp_path, ["2024-05-21T21:00:00"])
    source = sources.MrmsCompositeSource(packs, geo)

    class Provenance:
        uri = str(artifact)
        sha256 = digest

    assert source.verify(Provenance()) is True
    artifact.write_bytes(b"the archive object, quietly altered")
    assert source.verify(Provenance()) is False


def test_verify_raises_on_a_missing_object_rather_than_reporting_corruption(
        tmp_path):
    packs, geo = _source_tree(tmp_path, ["2024-05-21T21:00:00"])
    source = sources.MrmsCompositeSource(packs, geo)

    class Provenance:
        uri = str(tmp_path / "never-written.grib2.gz")
        sha256 = "b" * 64

    with pytest.raises(FileNotFoundError, match="not on disk"):
        source.verify(Provenance())


def _coverage_tree(tmp_path, frames, *, name="cov"):
    """Packs at given valid times, each carrying its own observed fraction.

    ``frames`` is ``[(valid_time, observed_fraction_or_None)]``; ``None``
    writes a pack with no ``sentinels`` block at all, which is what a product
    that does not record coverage looks like.
    """
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    geo = directory / "grid.obspack"
    geo.write_bytes(geo_pack_bytes())
    packs = []
    for index, (valid_time, fraction) in enumerate(frames):
        extra = ({} if fraction is None
                 else {"sentinels": {"observed_fraction": float(fraction)}})
        path = directory / f"frame{index}.obspack"
        path.write_bytes(grid_pack_bytes(valid_time=valid_time, extra=extra))
        packs.append(path)
    return packs, geo


def test_the_nearest_frame_below_the_coverage_floor_yields_to_a_covered_one(
        tmp_path):
    """The amendment's whole case: an ingest outage at the nearest frame.

    A 13-minute upstream outage put a 0.1586-covered frame nearest the hour
    with a 0.969-covered one 169 s later, inside the same registered
    tolerance. The registered floor selects the covered frame; without the
    floor the old rule stands, unchanged.
    """
    packs, geo = _coverage_tree(tmp_path, [
        ("2026-08-03T21:00:35", 0.1586),
        ("2026-08-03T21:02:49", 0.969),
        ("2026-08-03T20:57:00", 1.0)])
    target = sources._parse_time("2026-08-03T21:00:00")

    old_rule = sources.MrmsCompositeSource(packs, geo)
    amended = sources.MrmsCompositeSource(packs, geo,
                                          minimum_observed_fraction=0.9)

    assert old_rule._nearest(target).valid_time == \
        sources._parse_time("2026-08-03T21:00:35")
    chosen = amended._nearest(target)
    assert chosen.valid_time == sources._parse_time("2026-08-03T21:02:49")
    # Nearest-first, so the 169 s frame beats the 180 s one; and the answer
    # does not move between calls.
    assert amended._nearest(target).pack_path == chosen.pack_path


def test_the_selection_tie_inside_the_window_goes_to_the_earlier_frame(
        tmp_path):
    packs, geo = _coverage_tree(tmp_path, [
        ("2026-08-03T20:58:00", 1.0), ("2026-08-03T21:02:00", 1.0)])
    target = sources._parse_time("2026-08-03T21:00:00")
    source = sources.MrmsCompositeSource(packs, geo, match_seconds=240,
                                         minimum_observed_fraction=0.9)
    assert source._nearest(target).valid_time == \
        sources._parse_time("2026-08-03T20:58:00")


def test_the_coverage_floor_never_widens_the_matching_window(tmp_path):
    """A better-covered frame outside the tolerance is not a candidate."""
    packs, geo = _coverage_tree(tmp_path, [
        ("2026-08-03T21:00:35", 0.1586),
        ("2026-08-03T21:04:10", 0.95)])  # +250 s: outside +/-240 s
    target = sources._parse_time("2026-08-03T21:00:00")
    source = sources.MrmsCompositeSource(packs, geo, match_seconds=240,
                                         minimum_observed_fraction=0.9)

    with pytest.raises(LookupError) as refusal:
        source._nearest(target)

    assert "not widened" in str(refusal.value)
    assert [row["observed_fraction"] for row in refusal.value.candidates] == \
        [0.1586]


def test_a_full_coverage_frame_set_selects_exactly_as_it_did_before(tmp_path):
    """The no-op proof: where every frame is fully observed, nothing moves."""
    times = [f"2026-08-03T{hour:02d}:00:{seconds}"
             for hour, seconds in zip(range(12, 19),
                                      ("35", "39", "41", "33", "40", "36",
                                       "38"))]
    packs, geo = _coverage_tree(tmp_path, [(t, 1.0) for t in times])
    old_rule = sources.MrmsCompositeSource(packs, geo)
    amended = sources.MrmsCompositeSource(packs, geo,
                                          minimum_observed_fraction=0.9)

    for hour in range(12, 19):
        target = sources._parse_time(f"2026-08-03T{hour:02d}:00:00")
        assert amended._nearest(target).pack_path == \
            old_rule._nearest(target).pack_path


def test_an_unrecorded_observed_fraction_is_not_screened_on(tmp_path):
    """A product that records no coverage is selected on exactly as before."""
    packs, geo = _coverage_tree(tmp_path, [("2026-08-03T21:00:35", None),
                                           ("2026-08-03T21:02:49", None)])
    target = sources._parse_time("2026-08-03T21:00:00")
    source = sources.MrmsCompositeSource(packs, geo,
                                         minimum_observed_fraction=0.9)
    assert source._nearest(target).valid_time == \
        sources._parse_time("2026-08-03T21:00:35")


def test_a_coverage_floor_outside_the_unit_interval_is_refused(tmp_path):
    packs, geo = _coverage_tree(tmp_path, [("2026-08-03T21:00:35", 1.0)])
    with pytest.raises(ValueError, match="fraction in"):
        sources.MrmsCompositeSource(packs, geo, minimum_observed_fraction=1.5)


def test_a_relocated_archive_is_found_under_the_root_whatever_pulled_it(
        tmp_path):
    """The root override has to work in the direction it exists for.

    An archive pulled on one box and scored on another carries the *fetching*
    box's separators in its provenance, and the basename under the root is
    taken with the rule that reads both.  A POSIX ``Path`` does not split a
    Windows path at all, so this pin is what stops a Windows-pulled archive
    from being unscorable on a Linux node -- the one case the argument exists
    for.
    """
    artifact = tmp_path / "obs.grib2.gz"
    artifact.write_bytes(b"the archive object as it arrived")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    packs, geo = _source_tree(tmp_path, ["2024-05-21T21:00:00"])
    source = sources.MrmsCompositeSource(packs, geo)

    for recorded in (r"D:\obs\objects\mrms\20260803\obs.grib2.gz",
                     "/srv/obs/objects/mrms/20260803/obs.grib2.gz"):
        class Provenance:
            uri = recorded
            sha256 = digest

        with pytest.raises(FileNotFoundError, match="not on disk"):
            source.verify(Provenance())
        assert source.verify(Provenance(), root=tmp_path) is True

    artifact.write_bytes(b"the archive object, quietly altered")

    class Altered:
        uri = r"D:\obs\objects\mrms\20260803\obs.grib2.gz"
        sha256 = digest

    assert source.verify(Altered(), root=tmp_path) is False


def test_an_asos_record_of_the_wrong_schema_is_refused(tmp_path):
    path = tmp_path / "surface.json"
    path.write_text(json.dumps({"schema": "something.else.v1"}))
    with pytest.raises(ValueError, match="expected"):
        sources.AsosSurfaceSource(path)


# ------------------------------------------------- the seam, when present

def _contracts_or_skip():
    try:
        from gpuwm.verify.obs import contracts
    except ImportError:
        pytest.skip("the scoring lane's contracts module is not in this tree")
    return contracts


def test_a_field_built_from_a_pack_satisfies_the_scorers_contract(tmp_path):
    contracts = _contracts_or_skip()
    packs, geo = _source_tree(tmp_path, ["2024-05-21T21:00:37"])
    source = sources.MrmsCompositeSource(packs, geo)
    field = source.field("2024-05-21T21:00:00")
    assert isinstance(field, contracts.ObsGridField)
    assert field.quantity == "composite_reflectivity"
    assert field.units == contracts.GRID_UNITS["composite_reflectivity"]
    assert field.values.dtype == np.float64
    assert field.valid.dtype == np.bool_
    assert field.values.shape == field.valid.shape == field.latitude.shape
    # The contract's own geometry rules, restated by construction.
    assert np.all(field.longitude >= -180.0)
    assert np.all(field.longitude < 180.0)
    assert np.all(np.isfinite(field.values))


def test_the_sources_satisfy_the_protocols_the_scorer_checks(tmp_path):
    contracts = _contracts_or_skip()
    packs, geo = _source_tree(tmp_path, ["2024-05-21T21:00:37"])
    assert isinstance(sources.MrmsCompositeSource(packs, geo),
                      contracts.GriddedObsSource)


# --------------------------------------------- real bytes, when available

def _fixture_dir():
    named = os.environ.get("GPUWM_OBS_FIXTURES")
    if not named:
        pytest.skip("GPUWM_OBS_FIXTURES is not set; no archived bytes to read")
    directory = Path(named)
    if not directory.is_dir():
        pytest.skip(f"GPUWM_OBS_FIXTURES names no directory: {directory}")
    return directory


def test_the_mrms_front_door_decodes_archived_bytes_into_a_seam_field(tmp_path):
    """End to end on real archive bytes: decode, read back, check the seam.

    Skipped unless a checkout points at cached archive objects, because the
    battery's inputs are gigabytes and do not belong in the repository. The
    receipts that pin them live beside the cache.
    """

    directory = _fixture_dir()
    objects = sorted(directory.rglob("MRMS_*.grib2.gz"))
    if not objects:
        pytest.skip(f"no MRMS objects under {directory}")
    door = frontdoor.MRMS
    if door.find() is None:
        pytest.skip("rw_mrms is not built in this checkout")

    out = tmp_path / "field.obspack"
    record = door.run("decode", ["--file", str(objects[0]), "--out", str(out),
                                 "--bbox", "-100,37,-88,45"],
                      schema="gpuwm-obs.mrms-decode.v1")
    assert record["quantity"] == "composite_reflectivity"
    assert record["units"] == "dBZ"
    # The no-echo sentinel is an observation, not a gap: a case box over the
    # radar network must come back essentially fully observed, and a build
    # that masked -99 would report a fraction near a quarter instead.
    assert record["sentinels"]["observed_fraction"] > 0.9
    assert record["sentinels"]["no_echo_cells"] > 0

    pack = obspack.read_grid_pack(out)
    values = pack.array("values")
    valid = pack.array("valid").astype(bool)
    assert values.dtype == np.float64
    assert np.all(np.isfinite(values))
    assert values[valid].min() >= -40.0 and values[valid].max() <= 100.0
    verified = door.run("verify", ["--pack", str(out)],
                        schema="gpuwm-obs.mrms-verify.v1")
    assert verified["status"] == "PASS"


def test_the_stage4_front_door_decodes_archived_bytes(tmp_path):
    directory = _fixture_dir()
    objects = sorted(directory.rglob("ST4.*.01h.grib"))
    if not objects:
        pytest.skip(f"no Stage-IV objects under {directory}")
    door = frontdoor.STAGE4
    if door.find() is None:
        pytest.skip("rw_stage4 is not built in this checkout")

    out = tmp_path / "precip.obspack"
    record = door.run("decode", ["--file", str(objects[0]), "--out", str(out)],
                      schema="gpuwm-obs.stage4-decode.v1")
    assert record["units"] == "mm"
    assert record["accumulation_hours"] == 1
    assert record["grid"]["kind"] == "polar_stereographic"
    # The measured identity of every archived object: template 5.3 with the
    # missing cells in a Section-6 bitmap rather than in missing-value
    # management, which is the only reason the vendored decoder is sound here.
    assert record["packing"]["data_representation_template"] == 3
    assert record["packing"]["bitmap_present"] is True

    pack = obspack.read_grid_pack(out)
    values = pack.array("values")
    valid = pack.array("valid").astype(bool)
    assert np.all(np.isfinite(values))
    assert values[valid].min() >= 0.0 and values[valid].max() <= 2000.0
