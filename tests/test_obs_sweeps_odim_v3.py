"""The sweep reader's ODIM (``v3``) contract.

The cross-lane proof — a real Dutch PVOL through the real ``rw_odim`` writer
and back out of this reader — is recorded in
the ODIM DA-bridge cross-lane record; it needs a 27 MB volume off
a 24-hour feed, so it cannot live in the suite.  What lives here is the part
that is pure Python and would otherwise only ever be exercised by that file:
the promise that the schema string and the censor vocabulary tell the same
story, in both directions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from test_obs_superob_odim import _write_pack

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
    return _write_pack(
        tmp_path / f"{schema.rsplit('.', 1)[-1]}.pack", schema,
        products={"VRADH": codes}, measured=1.5, unit="m/s")


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
