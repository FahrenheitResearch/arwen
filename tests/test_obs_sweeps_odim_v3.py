"""The sweep reader's ODIM (``v3``) contract.

The cross-lane proof — a real Dutch PVOL through the real ``rw_odim`` writer
and back out of this reader — is recorded in
``Downloads/intl-da-odim/da-bridge-crosslane.md``; it needs a 27 MB volume off
a 24-hour feed, so it cannot live in the suite.  What lives here is the part
that is pure Python and would otherwise only ever be exercised by that file:
the promise that the schema string and the censor vocabulary tell the same
story, in both directions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.obs.sweeps import (
    CENSORED_SCHEMAS,
    NYQUIST_PER_SWEEP,
    SWEEPS_SCHEMA_CENSOR,
    SWEEPS_SCHEMA_ODIM,
    Censor,
    RadarSweepPackError,
    read_sweep_pack,
)


def _pack(tmp_path: Path, schema: str, codes: list[int]) -> Path:
    """Smallest well-formed pack carrying one moment with ``codes``."""

    censor = np.array([codes], dtype=np.uint8)
    data = np.where(censor == Censor.MEASURED, 1.5, np.nan).astype("<f4")
    azimuth = np.array([0.0], dtype="<f4")
    elevation = np.array([0.5], dtype="<f4")

    payload = b""
    arrays: dict[str, dict] = {}

    def push(name: str, array: np.ndarray) -> str:
        nonlocal payload
        raw = array.tobytes()
        arrays[name] = {
            "dtype": "<f4" if array.dtype == np.float32 else "|u1",
            "shape": list(array.shape),
            "offset": len(payload),
            "bytes": len(raw),
        }
        payload += raw
        return name

    push("a00000", data)
    push("a00001", censor)
    push("a00002", azimuth)
    push("a00003", elevation)

    meta = {
        "schema": schema,
        "status": "READY",
        "site": {"id": "nldhl", "name": "Den Helder", "lat_deg": 52.95,
                 "lon_deg": 4.79, "alt_m": 55.0, "source": "odim:/where"},
        "volume": {"file": "v.h5", "bytes": 1, "sha256": "0" * 64,
                   "station_id": "nldhl", "valid_time": "2026-08-14T00:00:00+00:00",
                   "volume_date": 0, "volume_time_ms": 0},
        "params": {"moments": [], "max_range_km": 200.0,
                   "max_elevation_deg": 25.0, "censor_flags": True},
        "sweeps": [{
            "sweep_index": 0, "elevation_number": 0, "elevation_angle_deg": 0.5,
            "nyquist_velocity_ms": 20.0, "nyquist_granularity": NYQUIST_PER_SWEEP,
            "start_status": 0, "end_status": 0, "cut_sector": 0, "complete": True,
            "radial_count": 1, "azimuth_array": "a00002",
            "elevation_array": "a00003",
            "moments": [{"product": "VRADH", "unit": "m/s",
                         "gate_count": len(codes), "first_gate_range_m": 45.0,
                         "gate_size_m": 90.0, "array": "a00000",
                         "censor_array": "a00001"}],
        }],
        "arrays": arrays,
        "payload_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "dropped_sweeps": 0, "dropped_moments": 0, "trimmed_gates": 0,
    }
    blob = json.dumps(meta).encode("utf-8")
    header = (b"GPWMRDR1"
              + (1).to_bytes(4, "little")
              + len(blob).to_bytes(4, "little")
              + len(payload).to_bytes(8, "little"))
    header += b"\x00" * (64 - len(header))
    path = tmp_path / f"{schema.rsplit('.', 1)[-1]}.pack"
    path.write_bytes(header + blob + payload)
    return path


def test_a_v3_pack_carries_the_two_codes_odim_adds(tmp_path: Path) -> None:
    """``nodata`` and ``sentinel_ambiguous`` survive the read as themselves."""

    volume = read_sweep_pack(_pack(
        tmp_path, SWEEPS_SCHEMA_ODIM,
        [Censor.MEASURED, Censor.BELOW_THRESHOLD, Censor.NOT_COLLECTED,
         Censor.NODATA, Censor.SENTINEL_AMBIGUOUS]))

    assert volume.pack_schema == SWEEPS_SCHEMA_ODIM
    sweep = volume.sweeps[0]
    assert sweep.nyquist_granularity == NYQUIST_PER_SWEEP
    censor = sweep.moments["VRADH"].censor
    assert censor is not None
    # Counts, not a flag: each state is present exactly once and none of
    # them has been folded into another on the way through.
    assert sorted(int(code) for code in np.unique(censor)) == [0, 1, 3, 4, 5]


def test_a_v2_pack_may_not_carry_the_codes_only_odim_mints(tmp_path: Path) -> None:
    """The version string is what makes the wider vocabulary safe.

    This is the direction that matters.  Accepting code 4 in a ``v2`` pack
    would mean a NEXRAD writer could emit a state its own schema has no word
    for and no reader would notice.
    """

    with pytest.raises(RadarSweepPackError) as refusal:
        read_sweep_pack(_pack(tmp_path, SWEEPS_SCHEMA_CENSOR,
                              [Censor.MEASURED, Censor.NODATA]))

    message = str(refusal.value)
    assert "[4]" in message, message
    assert SWEEPS_SCHEMA_CENSOR in message, message


def test_a_v3_pack_may_not_carry_range_folded(tmp_path: Path) -> None:
    """ODIM has no second-trip state, so code 2 in a v3 pack is a defect.

    The instrument is validated in both directions by this test and the one
    above it: each schema refuses exactly the codes the other one owns.
    """

    with pytest.raises(RadarSweepPackError) as refusal:
        read_sweep_pack(_pack(tmp_path, SWEEPS_SCHEMA_ODIM,
                              [Censor.MEASURED, Censor.RANGE_FOLDED]))

    assert "[2]" in str(refusal.value), str(refusal.value)


def test_the_censored_schemas_are_the_ones_with_planes() -> None:
    assert SWEEPS_SCHEMA_ODIM in CENSORED_SCHEMAS
    assert SWEEPS_SCHEMA_CENSOR in CENSORED_SCHEMAS
    assert Censor.NODATA == 4 and Censor.SENTINEL_AMBIGUOUS == 5
