"""The GFS matched pairs, and whether upstream still agrees.

``gfs_grib2_bridge`` accepts both published row orders (normalizing to
one) and both published packings: the NOMADS crop's DRT 5.0 and the raw
S3 objects' DRT 5.3, the latter certified by the SOILW missing-value
pair.  The bit-identity proofs live in Rust, where the decoder is
(``a_flipped_raw_s3_decode_is_bit_identical_to_the_nomads_crop`` and
``the_raw_53_bitmap_decode_matches_the_nomads_crop_cell_for_cell``);
what lives here is each pair's identity -- so a fixture cannot be
swapped for something that no longer demonstrates the point -- and one
live check that upstream has not changed its packing or row order since
the pairs were captured.

See ``tests/fixtures/gfs-scan-order/README.md`` for provenance.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpuwm import bridges

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gfs-scan-order"
RAW_S3 = FIXTURES / "s3-raw-tmp2m-20260729t18z-f000.grib2"
NOMADS_CROP = FIXTURES / "nomads-crop-20260729t18z-f000.grib2"
RAW_S3_SOILW = FIXTURES / "s3-raw-soilw-20260729t18z-f000.grib2"
NOMADS_CROP_SOILW = FIXTURES / "nomads-crop-soilw-20260729t18z-f000.grib2"

#: Each pair is a scientific artefact, not a convenience file: pin it.
FIXTURE_DIGESTS = {
    RAW_S3.name:
        "3cbf77deea57a0f1226c9bff5e3a8651b0e3a07152180c6ac89ea1eabb93bb45",
    NOMADS_CROP.name:
        "1a68737e6fb53256360e208aada933c5f1381b4968813cb519233b04168c6b6c",
    RAW_S3_SOILW.name:
        "c854e429091ee95b57878c9c74d6bf780e157c80617b0c0460d0a59aa781db5a",
    NOMADS_CROP_SOILW.name:
        "fe4397ef34206dddd2a7404d0e9274e4549de15577fb1674f83753e6dedfc72e",
}

#: Section 3, Grid Definition Template 3.0: the scanning-mode flags are
#: octet 72, i.e. offset 71 into the section.  `0x40` is +i/+j
#: (south-to-north rows); `0x00` is the WMO default +i/-j.
SCAN_MODE_OFFSET_IN_SECTION3 = 71
SOUTH_TO_NORTH = 0x40
NORTH_TO_SOUTH = 0x00


def _sections(payload: bytes):
    """Walk one GRIB2 message, yielding ``(number, bytes)`` per section.

    A deliberately small independent reader: the point of this module is
    to describe the fixtures without going through the decoder whose
    behaviour they are meant to pin.
    """

    assert payload[:4] == b"GRIB", "not a GRIB message"
    assert payload[7] == 2, "not GRIB edition 2"
    total = int.from_bytes(payload[8:16], "big")
    offset = 16
    while offset < total - 4:
        length = int.from_bytes(payload[offset:offset + 4], "big")
        number = payload[offset + 4]
        assert length >= 5, f"degenerate section at {offset}"
        yield number, payload[offset:offset + length]
        offset += length
    assert payload[total - 4:total] == b"7777", "missing 7777 terminator"


def _first_grid_section(path: Path) -> bytes:
    payload = path.read_bytes()
    for number, section in _sections(payload):
        if number == 3:
            return section
    raise AssertionError(f"{path.name} carries no Section 3")


@pytest.mark.parametrize("name", sorted(FIXTURE_DIGESTS))
def test_the_matched_pair_is_the_pair_that_was_certified(name):
    path = FIXTURES / name
    assert path.is_file(), f"missing fixture {name}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == FIXTURE_DIGESTS[name], (
        f"{name} is not the file the scan-order proof was written "
        "against; re-capture the pair and update both digests together")


def test_each_pair_really_carries_the_two_row_orders():
    """Otherwise the byte-identity tests prove nothing about the flip."""

    for raw_path, crop_path in ((RAW_S3, NOMADS_CROP),
                                (RAW_S3_SOILW, NOMADS_CROP_SOILW)):
        raw = _first_grid_section(raw_path)
        crop = _first_grid_section(crop_path)
        # Template 3.0 both, or the octet offsets below mean something
        # else.
        assert int.from_bytes(raw[12:14], "big") == 0
        assert int.from_bytes(crop[12:14], "big") == 0
        assert raw[SCAN_MODE_OFFSET_IN_SECTION3] == NORTH_TO_SOUTH
        assert crop[SCAN_MODE_OFFSET_IN_SECTION3] == SOUTH_TO_NORTH


def test_the_first_stored_row_runs_with_each_declared_scan():
    """`lat1` is the first stored row, so it moves with the scan flag."""

    def endpoints(section: bytes) -> tuple[float, float]:
        # GDT 3.0: la1 at offset 46, la2 at offset 55, both signed
        # micro-degrees.
        def micro(at: int) -> float:
            raw = int.from_bytes(section[at:at + 4], "big")
            sign = -1.0 if raw & 0x8000_0000 else 1.0
            return sign * (raw & 0x7FFF_FFFF) / 1_000_000.0
        return micro(46), micro(55)

    raw_lat1, raw_lat2 = endpoints(_first_grid_section(RAW_S3))
    crop_lat1, crop_lat2 = endpoints(_first_grid_section(NOMADS_CROP))
    # Global, north-first.
    assert (raw_lat1, raw_lat2) == (90.0, -90.0)
    # Cropped, south-first.
    assert (crop_lat1, crop_lat2) == (30.0, 40.0)


def _drt(path: Path) -> int:
    for number, section in _sections(path.read_bytes()):
        if number == 5:
            return int.from_bytes(section[9:11], "big")
    raise AssertionError(f"{path.name} carries no Section 5")


def test_the_raw_objects_are_complex_packed_and_the_crops_are_not():
    """The two packings the pairs certify, one per publisher form.

    Raw pgrb2.0p25 objects are DRT 5.3; the NOMADS crop re-encodes to
    5.0.  Accepting the row order was necessary but not sufficient for
    full-file GFS -- the 5.3 gate stayed shut until the SOILW pair
    supplied the complex-packing missing-value proof, and these facts
    are what make that pair the proof rather than a second copy of the
    first one.
    """

    assert _drt(RAW_S3) == 3
    assert _drt(NOMADS_CROP) == 0
    assert _drt(RAW_S3_SOILW) == 3
    assert _drt(NOMADS_CROP_SOILW) == 0


def test_the_soilw_pair_demonstrates_bitmap_missing_values():
    """Both SOILW forms carry a bitmap, and the raw record carries NO
    embedded missing-value management -- NCEP's missing cells travel in
    the bitmap, which is exactly the envelope the bridge admits."""

    def bitmap_indicator(path: Path) -> int:
        for number, section in _sections(path.read_bytes()):
            if number == 6:
                return section[5]
        raise AssertionError(f"{path.name} carries no Section 6")

    assert bitmap_indicator(RAW_S3_SOILW) == 0
    assert bitmap_indicator(NOMADS_CROP_SOILW) == 0
    # The TMP raw record has no bitmap (indicator 255): the two raw
    # fixtures cover both branches of the 5.3 decode.
    assert bitmap_indicator(RAW_S3) == 255

    for number, section in _sections(RAW_S3_SOILW.read_bytes()):
        if number == 5:
            # Template 5.3, octet 22 = group splitting method, octet 23
            # = missing value management (offsets 21 and 22).
            assert section[21] == 1, "general group splitting"
            assert section[22] == 0, (
                "the certified envelope carries no embedded missing-value "
                "management; if NCEP ever flips this octet the bridge must "
                "refuse, not decode")


BRIDGE = bridges.find_bridge("gfs_grib2_bridge")
needs_bridge = pytest.mark.skipif(
    BRIDGE is None,
    reason="gfs_grib2_bridge not built (cd tools/grib1_bridge && cargo "
           "build --release --locked --offline)")


@needs_bridge
def test_the_bridge_still_announces_its_series_contract():
    """The scan-order work must not have moved the CLI contract."""

    probe = subprocess.run([str(BRIDGE)], capture_output=True, text=True)
    assert probe.returncode != 0
    assert ("usage: gfs_grib2_bridge --series SERIES_TSV OUTPUT_DIR "
            "EXPECTED_CYCLE") in (probe.stdout + probe.stderr)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                    reason="live network smoke; set GPUWM_NETWORK_TESTS=1")
def test_upstream_still_publishes_the_two_forms_it_did(tmp_path):
    """Has NCEP changed row order or packing since the pair was captured?

    Fetches only the index of a recent cycle plus the CGI crop of one
    small box -- a few kilobytes -- and re-asserts the two facts the
    fixtures encode.  A failure here is upstream news, not a bug.
    """

    from datetime import datetime, timedelta, timezone
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    # The most recent synoptic cycle that has certainly finished.
    moment = datetime.now(timezone.utc) - timedelta(hours=8)
    cycle_hour = (moment.hour // 6) * 6
    date = f"{moment:%Y%m%d}"

    params = [
        ("file", f"gfs.t{cycle_hour:02d}z.pgrb2.0p25.f000"),
        ("subregion", ""), ("leftlon", "260"), ("rightlon", "270"),
        ("toplat", "40"), ("bottomlat", "30"),
        ("lev_2_m_above_ground", "on"), ("var_TMP", "on"),
        ("dir", f"/gfs.{date}/{cycle_hour:02d}/atmos"),
    ]
    url = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
           + urlencode(params))
    crop = tmp_path / "nomads-crop.grib2"
    with urlopen(Request(url, headers={"User-Agent": "gpuwm-fetch/1"}),
                 timeout=300) as response:
        crop.write_bytes(response.read())

    section = _first_grid_section(crop)
    assert section[SCAN_MODE_OFFSET_IN_SECTION3] == SOUTH_TO_NORTH, (
        "the NOMADS subregion crop no longer publishes south-to-north; "
        "the certified GFS form has changed")
    drt = next(int.from_bytes(body[9:11], "big")
               for number, body in _sections(crop.read_bytes())
               if number == 5)
    assert drt == 0, ("the NOMADS subregion crop no longer re-encodes to "
                      f"simple packing (DRT 5.{drt})")
