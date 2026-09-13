"""Explicit initial CWP uncertainty and observation timestamps.

The operator and the transport do not own these choices. See
``docs/local-da-automatic-cwp.md`` for the sources and limitations.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone


from gpuwm.obs.goes_cwp import CwpErrorModel

POLICY_ID = "regional-cwp-native-qc-v1"
# The 50 g m-2 scale is the lowest cloudy WP error in the 2019 study,
# section 2.2/Table 2, DOI:10.5194/gmd-12-3939-2019. The other
# choices are explicit provisional extensions, not coefficients from that
# study and not a measured ABI observation-error covariance.
DEFAULT_ERROR = CwpErrorModel(
    clear_g_m2=50.0, rel_liquid=0.5, floor_liquid_g_m2=50.0,
    rel_ice=1.0, floor_ice_g_m2=100.0,
    thin_inflation=1.5, thick_inflation=2.0,
)
ERROR_BASIS = {
    "reference": "doi:10.5194/gmd-12-3939-2019, section 2.2 and Table 2",
    "borrowed_scale_g_m2": 50.0,
    "not_transferred": "SatCORPS study is not an ABI validation; its operator differs",
    "choices": "clear 50; liquid max(0.5*CWP,50); ice max(CWP,100); thin x1.5; thick x2",
    "calibration": "UNCALIBRATED",
    "label": "literature-informed initial settings with provisional phase/quality extensions",
}


def default_errors() -> dict:
    return asdict(DEFAULT_ERROR)


def utc(value) -> datetime:
    """UTC with fractional seconds retained; automatic records must name a zone."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("satellite times must include a UTC offset")
    return value.astimezone(timezone.utc)


def stamp(value) -> str:
    return utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
