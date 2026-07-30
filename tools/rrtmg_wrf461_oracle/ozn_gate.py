"""Gate plumbing for the o3input=2 ozone pipeline port: fixture access.

The fixture is a single little-endian record file (lw_binio format)
written by ozn_extract (built from the unmodified WRF v4.6.1 sources by
ozn_build.sh).  All comparisons in tests/test_wrf_ozone.py are bitwise
(max_ulp 0).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from lw_fixtures import read_fixture  # noqa: E402,F401
from lw_gate import assert_ulp0, ulp_report  # noqa: E402,F401

#: Default fixture location (oracle scratch); override with env var.
DEFAULT_FIXDIR = os.environ.get(
    "GPUWM_WRF_OZONE_FIXTURES",
    os.path.expanduser(
        "~/.claude/jobs/fea7a141/tmp/rrtmg_integration/ozone/fixtures"),
)


def fixture_path(fixdir: str | None = None) -> str:
    return os.path.join(fixdir or DEFAULT_FIXDIR, "ozn_fixture.bin")


def have_fixture(fixdir: str | None = None) -> bool:
    return os.path.isfile(fixture_path(fixdir))


def load_fixture(fixdir: str | None = None) -> dict:
    path = fixture_path(fixdir)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"no ozone oracle fixture at {path}; run "
            "tools/rrtmg_wrf461_oracle/ozn_build.sh (see its header) or "
            "set GPUWM_WRF_OZONE_FIXTURES")
    return read_fixture(path)
