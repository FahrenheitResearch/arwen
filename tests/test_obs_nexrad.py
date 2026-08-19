"""The ``rw_nexrad`` bridge, and what it refuses.

Three layers, each testable on its own terms:

* the resolution ladder and the record contract, which need no binary;
* the sweep-pack reader's refusals, which are pure Python over bytes this
  test builds;
* the binary's own refusal of a corrupt volume, which needs the built bin
  and is skipped without it;
* one live smoke over a single real volume, doubly gated on
  ``GPUWM_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess

import numpy as np
import pytest

from gpuwm import bridges
from gpuwm.obs import nexrad
from gpuwm.obs.radar_grid import read_radar_grid, write_radar_grid
from gpuwm.obs.superob import SuperobParams, merge_contributions, superob_volume
from gpuwm.obs.sweeps import (SWEEPS_SCHEMA, RadarSweepPackError,
                              read_sweep_pack)
from gpuwm.obs.sweeps import Censor, SWEEPS_SCHEMA_CENSOR
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

_MAGIC = b"GPWMRDR1"
_HEADER_BYTES = 64


def _pack_bytes(meta: dict, payload: bytes) -> bytes:
    encoded = json.dumps(meta).encode("utf-8")
    header = bytearray(_HEADER_BYTES)
    header[:8] = _MAGIC
    header[8:12] = struct.pack("<I", 1)
    header[12:16] = struct.pack("<I", len(encoded))
    header[16:24] = struct.pack("<Q", len(payload))
    return bytes(header) + encoded + payload


def _minimal_meta(payload: bytes, arrays: dict) -> dict:
    arrays = copy.deepcopy(arrays)      # callers mutate what they get back
    return {
        "schema": SWEEPS_SCHEMA,
        "status": "READY",
        "site": {"id": "KTLX", "name": "Oklahoma City, OK",
                 "lat_deg": 35.3331, "lon_deg": -97.2778, "alt_m": 370.0,
                 "source": "wx-radar-site-table"},
        "volume": {"file": "KTLX20260728_200316_V06", "bytes": 8102058,
                   "sha256": "0" * 64, "station_id": "KTLX",
                   "valid_time": "2026-07-28T20:03:16Z",
                   "volume_date": 20663, "volume_time_ms": 72196232,
                   "framing": {"magic": "AR2V0006", "block_count": 55,
                               "bzip2_block_count": 55, "bytes": 8102058}},
        "params": {"moments": ["REF"], "max_range_km": 250.0,
                   "max_elevation_deg": 20.0},
        "sweeps": [{
            "sweep_index": 0, "elevation_number": 1,
            "elevation_angle_deg": 0.5, "nyquist_velocity_ms": 32.0,
            "start_status": 3, "end_status": 2, "cut_sector": 0,
            "complete": True, "radial_count": 2,
            "azimuth_array": "a00000", "elevation_array": "a00001",
            "moments": [{"product": "REF", "unit": "dBZ", "gate_count": 3,
                         "first_gate_range_m": 2125.0, "gate_size_m": 250.0,
                         "array": "a00002"}]}],
        "arrays": arrays,
        "payload_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "dropped_sweeps": 0, "dropped_moments": 0, "trimmed_gates": 0,
    }


def _good_pack() -> bytes:
    azimuth = np.array([0.0, 1.0], dtype="<f4")
    elevation = np.array([0.5, 0.5], dtype="<f4")
    data = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]], dtype="<f4")
    payload = azimuth.tobytes() + elevation.tobytes() + data.tobytes()
    arrays = {
        "a00000": {"dtype": "<f4", "shape": [2], "offset": 0, "bytes": 8},
        "a00001": {"dtype": "<f4", "shape": [2], "offset": 8, "bytes": 8},
        "a00002": {"dtype": "<f4", "shape": [2, 3], "offset": 16,
                   "bytes": 24},
    }
    return _pack_bytes(_minimal_meta(payload, arrays), payload)


def _censored_pack(*, data=None, codes=None, schema=SWEEPS_SCHEMA_CENSOR,
                   attach=True) -> bytes:
    """A v2 pack: the same two radials, plus a ``|u1`` censor plane.

    Gate 0 of each radial is below threshold and gate 1 is range folded --
    both NaN in the moment plane, and told apart only by the censor plane.
    """

    azimuth = np.array([0.0, 1.0], dtype="<f4")
    elevation = np.array([0.5, 0.5], dtype="<f4")
    if data is None:
        data = np.array([[np.nan, np.nan, 30.0],
                         [np.nan, np.nan, 31.0]], dtype="<f4")
    if codes is None:
        codes = np.array([[Censor.BELOW_THRESHOLD, Censor.RANGE_FOLDED,
                           Censor.MEASURED],
                          [Censor.BELOW_THRESHOLD, Censor.RANGE_FOLDED,
                           Censor.MEASURED]], dtype="|u1")
    data = np.asarray(data, dtype="<f4")
    codes = np.asarray(codes, dtype="|u1")
    payload = (azimuth.tobytes() + elevation.tobytes() + data.tobytes()
               + codes.tobytes())
    arrays = {
        "a00000": {"dtype": "<f4", "shape": [2], "offset": 0, "bytes": 8},
        "a00001": {"dtype": "<f4", "shape": [2], "offset": 8, "bytes": 8},
        "a00002": {"dtype": "<f4", "shape": list(data.shape), "offset": 16,
                   "bytes": data.nbytes},
        "a00003": {"dtype": "|u1", "shape": list(codes.shape),
                   "offset": 16 + data.nbytes, "bytes": codes.nbytes},
    }
    meta = _minimal_meta(payload, arrays)
    meta["schema"] = schema
    meta["params"]["censor_flags"] = schema == SWEEPS_SCHEMA_CENSOR
    if attach:
        meta["sweeps"][0]["moments"][0]["censor_array"] = "a00003"
    return _pack_bytes(meta, payload)


# -- the censor plane: the v2 pack contract -------------------------------

def test_a_v1_pack_reads_with_no_censor_plane_at_all(tmp_path):
    """The do-no-harm side of the reader.

    ``None`` is not "all measured": it is "the reasons were never
    recorded", which is the only honest reading of a v1 pack.
    """

    path = tmp_path / "v1.pack"
    path.write_bytes(_good_pack())
    volume = read_sweep_pack(path)
    assert volume.pack_schema == SWEEPS_SCHEMA
    assert volume.sweeps[0].moments["REF"].censor is None
    assert volume.provenance()["pack_schema"] == SWEEPS_SCHEMA


def test_a_v2_pack_carries_the_reason_each_gate_is_not_a_number(tmp_path):
    path = tmp_path / "v2.pack"
    path.write_bytes(_censored_pack())
    volume = read_sweep_pack(path)
    assert volume.pack_schema == SWEEPS_SCHEMA_CENSOR
    assert volume.provenance()["pack_schema"] == SWEEPS_SCHEMA_CENSOR
    moment = volume.sweeps[0].moments["REF"]
    assert moment.censor is not None
    assert moment.censor.shape == moment.data.shape
    assert moment.censor[0, 0] == Censor.BELOW_THRESHOLD
    assert moment.censor[0, 1] == Censor.RANGE_FOLDED
    assert moment.censor[0, 2] == Censor.MEASURED
    # The two gates the old decoder could not tell apart are still the same
    # NaN in the moment plane.  The plane beside it is the whole difference.
    assert np.isnan(moment.data[0, 0]) and np.isnan(moment.data[0, 1])


def test_a_pack_whose_schema_and_arrays_disagree_is_refused(tmp_path):
    # v1 string, censor plane attached: a v1 consumer would read a file
    # making a claim its schema denies.
    path = tmp_path / "lying.pack"
    path.write_bytes(_censored_pack(schema=SWEEPS_SCHEMA))
    with pytest.raises(RadarSweepPackError, match="dtype"):
        read_sweep_pack(path)

    # v2 string, no censor plane: promises a distinction it cannot make.
    path = tmp_path / "empty.pack"
    path.write_bytes(_censored_pack(attach=False))
    with pytest.raises(RadarSweepPackError, match="missing its censor plane"):
        read_sweep_pack(path)


def test_a_censor_plane_that_contradicts_its_moment_plane_is_refused(tmp_path):
    """The failure that would fabricate observations, caught at the reader.

    A censor plane calling a NaN gate MEASURED, or calling a real number
    BELOW_THRESHOLD, is a plane describing a different volume.  Either
    direction would put a number into the clear-air count that the radar
    never reported, so neither is repaired -- the file is refused.
    """

    codes = np.array([[Censor.MEASURED, Censor.RANGE_FOLDED, Censor.MEASURED],
                      [Censor.BELOW_THRESHOLD, Censor.RANGE_FOLDED,
                       Censor.MEASURED]], dtype="|u1")
    path = tmp_path / "nan-called-measured.pack"
    path.write_bytes(_censored_pack(codes=codes))
    with pytest.raises(RadarSweepPackError, match="disagrees with"):
        read_sweep_pack(path)

    codes = np.array([[Censor.BELOW_THRESHOLD, Censor.RANGE_FOLDED,
                       Censor.BELOW_THRESHOLD],
                      [Censor.BELOW_THRESHOLD, Censor.RANGE_FOLDED,
                       Censor.MEASURED]], dtype="|u1")
    path = tmp_path / "number-called-clear.pack"
    path.write_bytes(_censored_pack(codes=codes))
    with pytest.raises(RadarSweepPackError, match="disagrees with"):
        read_sweep_pack(path)


def test_an_unknown_censor_code_is_refused_rather_than_ignored(tmp_path):
    """A code this build does not know might mean anything, including
    "range folded" under a future numbering.  Refuse it."""

    codes = np.array([[7, Censor.RANGE_FOLDED, Censor.MEASURED],
                      [Censor.BELOW_THRESHOLD, Censor.RANGE_FOLDED,
                       Censor.MEASURED]], dtype="|u1")
    path = tmp_path / "unknown-code.pack"
    path.write_bytes(_censored_pack(codes=codes))
    with pytest.raises(RadarSweepPackError, match="unknown codes"):
        read_sweep_pack(path)


# -- the bridge, without a binary -----------------------------------------

def test_resolution_ladder_is_deterministic_and_override_first(monkeypatch):
    monkeypatch.delenv(nexrad.NEXRAD_ENV, raising=False)
    candidates = nexrad.nexrad_candidates()
    # Five rungs since the platform wheel started carrying the binaries:
    # checkout release, checkout debug, <root>/libexec/bridges beside the
    # package, gpuwm/libexec/bridges INSIDE it, and ~/.gpuwm/bridges.
    assert len(candidates) == 5
    assert any(candidate.is_relative_to(bridges.packaged_bridge_dir())
               for candidate in candidates), (
        "the in-package rung is missing, so a wheel install cannot find "
        "the radar front door it ships")
    assert candidates[0].parts[-2:] == ("release",
                                        nexrad.executable_name("rw_nexrad"))
    monkeypatch.setenv(nexrad.NEXRAD_ENV, str(Path("nowhere") / "rw_nexrad"))
    assert nexrad.nexrad_candidates()[0] == Path("nowhere") / "rw_nexrad"


def test_an_override_naming_a_missing_file_is_a_hard_error(monkeypatch,
                                                           tmp_path):
    monkeypatch.setenv(nexrad.NEXRAD_ENV, str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match=nexrad.NEXRAD_ENV):
        nexrad.find_nexrad_bin()


def test_the_default_bucket_is_the_one_the_front_door_actually_reads():
    """The Python constant names the bucket a default command uses.

    It published ``noaa-nexrad-level2`` -- the archive of record, which
    answers anonymous ListObjectsV2 with 403 -- long after the Rust front
    door moved its own default to the Unidata mirror on a capability
    check.  The operational path was never wrong, because this wrapper
    passes ``--bucket`` only when a caller names one and Rust therefore
    chose the mirror; what was wrong was the statement, and an operator
    reading the Python surface was told the pipeline uses a bucket it
    cannot reach.
    """

    assert nexrad.DEFAULT_BUCKET == "unidata-nexrad-level2"
    assert nexrad.ARCHIVE_OF_RECORD_BUCKET == "noaa-nexrad-level2"
    assert nexrad.MIRROR_BUCKET == nexrad.DEFAULT_BUCKET
    assert nexrad.DEFAULT_BUCKET != nexrad.ARCHIVE_OF_RECORD_BUCKET


def test_the_wrapper_lets_the_front_door_choose_the_bucket():
    """Which is why the constant is a statement and not a parameter.

    Both directions: with no bucket named the wrapper emits no
    ``--bucket`` at all, so the default is Rust's; with one named it is
    passed through verbatim, so the archive of record stays selectable.
    """

    window = {"site": "KTLX", "start": "2013-05-20T19:50:00Z",
              "end": "2013-05-20T20:00:00Z", "limit": None}
    assert "--bucket" not in nexrad._window(bucket=None, **window)
    named = nexrad._window(bucket=nexrad.ARCHIVE_OF_RECORD_BUCKET, **window)
    assert named[named.index("--bucket") + 1] \
        == nexrad.ARCHIVE_OF_RECORD_BUCKET


def test_the_default_bucket_matches_the_binarys_own_declared_default():
    """Bound to the front door, not to a second copy of the string.

    ``rw_nexrad --help`` states which bucket it defaults to and which one
    is the archive of record.  Reading it here is what makes the constant
    above a fact about the pipeline rather than a comment that drifted:
    the two lanes' defaults cannot disagree again without this failing.
    """

    binary = _binary_or_skip()
    result = subprocess.run([str(binary), "--help"], capture_output=True,
                            text=True, errors="replace", timeout=30)
    assert result.returncode == 0, result.stderr
    declared = re.search(r"Default:\s*([A-Za-z0-9][A-Za-z0-9.\-]*)",
                         result.stdout)
    assert declared, f"--help states no default bucket:\n{result.stdout}"
    assert declared.group(1) == nexrad.DEFAULT_BUCKET
    assert nexrad.ARCHIVE_OF_RECORD_BUCKET in result.stdout


def test_the_live_route_is_a_second_key_space_not_a_second_mirror():
    """The chunk feed is not another ``--bucket`` value for the archive.

    Its keys are ``{SITE}/{VOLUME_ID}/{YYYYMMDD}-{HHMMSS}-{NNN}-{S|I|E}``,
    so a live subcommand pointed at the archive bucket would list a prefix
    that does not exist and report an empty sky.  Two constants, and they
    are required to differ.
    """

    assert nexrad.LIVE_DEFAULT_BUCKET == "unidata-nexrad-level2-chunks"
    assert nexrad.LIVE_DEFAULT_BUCKET != nexrad.DEFAULT_BUCKET
    assert nexrad.ARCHIVE_FEED != nexrad.LIVE_FEED


def test_the_live_wrapper_emits_no_flag_it_was_not_given():
    """Same rule as ``--bucket``: unset means the front door decides.

    A partial scan in particular must never be requested by omission --
    ``--allow-partial`` appears only when a caller asked for one.
    """

    bare = nexrad._live(site="KTLX", bucket=None, volumes=None,
                        volume_id=None, allow_partial=False, min_chunks=None)
    assert bare == ["--site", "KTLX"]
    full = nexrad._live(site="KTLX", bucket="some-chunks-mirror", volumes=3,
                        volume_id=571, allow_partial=True, min_chunks=7)
    assert full == ["--site", "KTLX", "--bucket", "some-chunks-mirror",
                    "--volumes", "3", "--volume-id", "571",
                    "--allow-partial", "--min-chunks", "7"]


def test_the_live_default_bucket_matches_the_binarys_own_declared_default():
    binary = _binary_or_skip()
    result = subprocess.run([str(binary), "--help"], capture_output=True,
                            text=True, errors="replace", timeout=30)
    assert result.returncode == 0, result.stderr
    declared = re.search(r"Live default:\s*([A-Za-z0-9][A-Za-z0-9.\-]*)",
                         result.stdout)
    assert declared, f"--help states no live default bucket:\n{result.stdout}"
    assert declared.group(1) == nexrad.LIVE_DEFAULT_BUCKET


def test_the_remedy_is_a_command_or_a_comment_on_every_line():
    for line in nexrad.nexrad_remedy().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        assert not stripped.startswith("then "), line


# -- the sweep pack reader's refusals -------------------------------------

def test_sweep_pack_round_trips_and_exposes_the_range_axis(tmp_path):
    path = tmp_path / "good.rdrpack"
    path.write_bytes(_good_pack())
    volume = read_sweep_pack(path)
    assert volume.site.id == "KTLX"
    assert volume.valid_time == "2026-07-28T20:03:16Z"
    assert len(volume.sweeps) == 1
    moment = volume.sweeps[0].moments["REF"]
    assert moment.data.shape == (2, 3)
    assert np.allclose(moment.slant_range_m(), [2125.0, 2375.0, 2625.0])
    assert volume.provenance()["pack_schema"] == SWEEPS_SCHEMA


@pytest.mark.parametrize("mutate,expected", [
    (lambda raw: raw[:32], "pack header alone"),
    (lambda raw: b"NOTAPACK" + raw[8:], "magic"),
    (lambda raw: raw[:8] + struct.pack("<I", 99) + raw[12:], "pack version"),
    (lambda raw: raw[:-1], "header declares"),
    (lambda raw: raw[:-1] + bytes([raw[-1] ^ 0xFF]), "payload hashes to"),
])
def test_sweep_pack_reader_refuses_damaged_packs(tmp_path, mutate, expected):
    path = tmp_path / "damaged.rdrpack"
    path.write_bytes(mutate(_good_pack()))
    with pytest.raises(RadarSweepPackError, match=expected):
        read_sweep_pack(path)


def test_sweep_pack_reader_refuses_a_wrong_schema_and_a_lying_array_index(
        tmp_path):
    azimuth = np.array([0.0, 1.0], dtype="<f4")
    payload = azimuth.tobytes()
    arrays = {"a00000": {"dtype": "<f4", "shape": [2], "offset": 0,
                         "bytes": 8}}
    meta = _minimal_meta(payload, arrays)
    meta["schema"] = "gpuwm-obs.radar-sweeps.v0"
    path = tmp_path / "wrong-schema.rdrpack"
    path.write_bytes(_pack_bytes(meta, payload))
    with pytest.raises(RadarSweepPackError, match="declares schema"):
        read_sweep_pack(path)

    meta = _minimal_meta(payload, arrays)
    meta["arrays"]["a00000"]["shape"] = [3]
    path = tmp_path / "lying-shape.rdrpack"
    path.write_bytes(_pack_bytes(meta, payload))
    with pytest.raises(RadarSweepPackError, match="declares shape"):
        read_sweep_pack(path)

    meta = _minimal_meta(payload, arrays)
    meta["arrays"]["a00000"]["offset"] = 4096
    path = tmp_path / "out-of-payload.rdrpack"
    path.write_bytes(_pack_bytes(meta, payload))
    with pytest.raises(RadarSweepPackError, match="spans past the end"):
        read_sweep_pack(path)


def test_sweep_pack_reader_refuses_a_pack_that_is_not_ready(tmp_path):
    payload = np.array([0.0, 1.0], dtype="<f4").tobytes()
    arrays = {"a00000": {"dtype": "<f4", "shape": [2], "offset": 0,
                         "bytes": 8}}
    meta = _minimal_meta(payload, arrays)
    meta["status"] = "PARTIAL"
    path = tmp_path / "not-ready.rdrpack"
    path.write_bytes(_pack_bytes(meta, payload))
    with pytest.raises(RadarSweepPackError, match="status"):
        read_sweep_pack(path)


# -- the binary's own refusals --------------------------------------------

def _binary_or_skip() -> Path:
    binary = nexrad.find_nexrad_bin()
    if binary is None:
        pytest.skip(f"rw_nexrad is not built: {nexrad.nexrad_remedy()}")
    return binary


def _archive2_shell(block_payload: bytes, declared: int | None = None
                    ) -> bytes:
    raw = bytearray(b"AR2V0006.")
    raw.extend(b"\0" * (24 - len(raw)))
    size = len(block_payload) if declared is None else declared
    raw.extend(struct.pack(">i", size))
    raw.extend(block_payload)
    return bytes(raw)


def test_the_binary_matches_the_abi_this_wrapper_was_written_against():
    binary = _binary_or_skip()
    ok, detail = nexrad.probe_nexrad_bin(binary)
    assert ok, detail


def test_the_binary_refuses_a_truncated_volume_and_writes_no_pack(tmp_path):
    binary = _binary_or_skip()
    volume = tmp_path / "KTLX20260728_200316_V06"
    # Declares 4096 bytes of LDM block, supplies a handful.
    volume.write_bytes(_archive2_shell(b"BZh9only-a-few", declared=4096))
    out = tmp_path / "truncated.rdrpack"
    with pytest.raises(RuntimeError, match="truncated Level-II volume"):
        nexrad.run_decode(binary, volume=volume, out=out)
    assert not out.exists()


def test_the_binary_refuses_a_file_that_is_not_archive_two(tmp_path):
    binary = _binary_or_skip()
    junk = tmp_path / "KTLX20260728_200316_V06"
    junk.write_bytes(b"\x00" * 4096)
    with pytest.raises(RuntimeError, match="not a Level-II volume"):
        nexrad.run_decode(binary, volume=junk, out=tmp_path / "junk.rdrpack")

    gz = tmp_path / "KTLX20130520_200356_V06"
    gz.write_bytes(b"\x1f\x8b" + b"\x00" * 4094)
    with pytest.raises(RuntimeError, match="gzip"):
        nexrad.run_decode(binary, volume=gz, out=tmp_path / "gz.rdrpack")


def test_the_binary_refuses_a_volume_that_is_neither_real_shape(tmp_path):
    """Archive-II magic, and bytes that are not either layout behind it.

    The archive holds exactly two shapes: an LDM table of bzip2 blocks
    (plain keys, roughly 2017 on) and a bare uncompressed message stream
    (the pre-2016 `.gz` keys, once expanded).  A file that carries the
    magic and then neither must be refused and must write no pack --
    whichever of the two diagnoses fits the bytes.
    """

    binary = _binary_or_skip()
    for index, payload in enumerate((b"plain-uncompressed-bytes",
                                     b"BZ",
                                     b"\x1f\x8bnot-bzip2-either")):
        volume = tmp_path / f"KTLX2026072{index}_200316_V06"
        volume.write_bytes(_archive2_shell(payload))
        out = tmp_path / f"plain{index}.rdrpack"
        with pytest.raises(RuntimeError,
                           match="truncated Level-II volume|no Message-31"):
            nexrad.run_decode(binary, volume=volume, out=out)
        assert not out.exists()


def test_the_binary_reads_a_gzipped_pre_2016_volume(tmp_path):
    """The `.gz` era must reach the same refusals, not a different door.

    A gzip-wrapped file is expanded before anything looks at it, so a
    wrapper around junk is refused for what it holds rather than for being
    wrapped -- and a wrapper that will not expand is refused as such.
    """

    import gzip

    binary = _binary_or_skip()
    wrapped = tmp_path / "KTLX20130520_195111_V06.gz"
    wrapped.write_bytes(gzip.compress(_archive2_shell(b"plain-bytes")))
    out = tmp_path / "wrapped.rdrpack"
    with pytest.raises(RuntimeError,
                       match="truncated Level-II volume|no Message-31"):
        nexrad.run_decode(binary, volume=wrapped, out=out)
    assert not out.exists()

    # A member cut short can still decode to a clean prefix without the
    # reader erroring, so the refusal may come from the inflate or from the
    # trailer's ISIZE.  Both are refusals; a prefix of a volume must never
    # be published as a volume.
    whole = gzip.compress(_archive2_shell(b"BZh9" + b"payload" * 400))
    for index, cut in enumerate((len(whole) // 2, len(whole) - 8)):
        broken = tmp_path / f"KTLX2013052{index}_195527_V06.gz"
        broken.write_bytes(whole[:cut])
        out = tmp_path / f"broken{index}.rdrpack"
        with pytest.raises(RuntimeError,
                           match="will not expand|incomplete"):
            nexrad.run_decode(binary, volume=broken, out=out)
        assert not out.exists()


def test_the_binary_serves_the_site_table_and_refuses_unknown_ids():
    binary = _binary_or_skip()
    record = nexrad.run_sites(binary)
    assert record["count"] >= 140
    one = nexrad.run_sites(binary, site="ktlx")
    assert one["count"] == 1
    assert one["sites"][0]["id"] == "KTLX"
    with pytest.raises(RuntimeError, match="not in the vendored"):
        nexrad.run_sites(binary, site="ZZZZ")


def test_the_binary_refuses_a_malformed_window():
    binary = _binary_or_skip()
    with pytest.raises(RuntimeError, match="four alphanumeric"):
        nexrad.run_list(binary, site="K", start="2026-07-28T20:00:00Z",
                        end="2026-07-28T20:05:00Z")
    with pytest.raises(RuntimeError, match="precedes"):
        nexrad.run_list(binary, site="KTLX", start="2026-07-28T20:05:00Z",
                        end="2026-07-28T20:00:00Z")


# -- one live volume, end to end ------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                    reason="live network smoke; set GPUWM_NETWORK_TESTS=1")
def test_one_real_volume_reaches_the_radar_grid_schema(tmp_path):
    """S3 -> volume -> pack -> superob -> ``gpuwm-obs.radar-grid.v1``.

    One volume, a few megabytes, from the mirror that still lists
    anonymously.  The window is deliberately wide and ``--limit 1`` keeps
    the download to a single object however busy the site was.
    """

    binary = _binary_or_skip()
    site = os.environ.get("GPUWM_RADAR_SMOKE_SITE", "KTLX")
    start = os.environ.get("GPUWM_RADAR_SMOKE_START")
    end = os.environ.get("GPUWM_RADAR_SMOKE_END")
    bucket = os.environ.get("GPUWM_RADAR_SMOKE_BUCKET", nexrad.MIRROR_BUCKET)
    if not (start and end):
        pytest.skip("set GPUWM_RADAR_SMOKE_START/END to a window the "
                    "rolling mirror still carries")

    listing = nexrad.run_list(binary, site=site, start=start, end=end,
                              bucket=bucket, limit=1)
    assert listing["status"] == "READY", listing
    assert listing["volumes"], "the window carries no volumes"
    assert listing["volumes"][0]["size_bytes"] < 40 * 1024 * 1024

    fetch = nexrad.run_fetch(binary, site=site, start=start, end=end,
                             bucket=bucket, limit=1, out=tmp_path / "volumes")
    assert len(fetch["files"]) == 1
    downloaded = Path(fetch["files"][0]["path"])
    assert downloaded.is_file()
    assert (hashlib.sha256(downloaded.read_bytes()).hexdigest()
            == fetch["files"][0]["sha256"])

    pack_path = tmp_path / "volume.rdrpack"
    decode = nexrad.run_decode(binary, volume=downloaded, out=pack_path,
                               moments=("REF", "VEL"), max_range_km=150.0,
                               max_elevation_deg=6.0)
    assert decode["sweeps"] > 0
    assert decode["gates"] > 0
    assert decode["volume"]["framing"]["bzip2_block_count"] > 0
    assert nexrad.run_verify(binary, pack=pack_path)["status"] == "PASS"

    volume = read_sweep_pack(pack_path)
    assert volume.station_id == site
    assert volume.volume_sha256 == fetch["files"][0]["sha256"]

    projection = LambertGrid(
        ref_lat=volume.site.lat_deg, ref_lon=volume.site.lon_deg,
        truelat1=volume.site.lat_deg - 2.0,
        truelat2=volume.site.lat_deg + 2.0,
        stand_lon=volume.site.lon_deg, dx=4000.0, dy=4000.0,
        e_we=81, e_sn=81)
    grid = TargetGrid.from_projection(
        projection, z_w=np.linspace(volume.site.alt_m,
                                    volume.site.alt_m + 15000.0, 16),
        name="smoke")
    params = SuperobParams(max_range_km=150.0, max_elevation_deg=6.0)
    contribution = superob_volume(volume, grid, params=params)
    observations = merge_contributions([contribution], grid, params=params)

    assert int(observations.z_mask.sum()) > 0, "a real volume gridded to nothing"
    assert observations.counts[0]["gates_considered"] > 0

    out = tmp_path / "radar-grid.nc"
    receipt = write_radar_grid(out, observations, grid,
                               valid_time=volume.valid_time, params=params)
    read = read_radar_grid(out,
                           expected_grid_identity=grid.identity_sha256())
    assert read["radars"][0]["id"] == site
    assert read["dims"]["radar"] == 1
    assert receipt["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert read["provenance"]["volumes"][0]["volume_sha256"] == \
        fetch["files"][0]["sha256"]
