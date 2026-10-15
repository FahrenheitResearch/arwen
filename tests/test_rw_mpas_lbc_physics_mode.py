"""rw_mpas_lbc's physics mode: correct by default from 2.9.0, and the pin
that refuses a build from before the switch.

THE BREAKAGE THIS PREVENTS.  The boundary producer reads unknown flags
without complaint.  A build from before ``--physics-mode`` existed (2.8.9 and
the first 2.9.0 candidates) therefore takes ``--physics-mode correct``, writes
native case-9 boundary winds (IN4: on x1.2562 2023-03-31 12Z GFS, lbc_u
differs from the initial conditions' extrapolation on 9,075 of 422,400
edge-level words, up to 4.7 m/s, and lbc_w on 3,596 words) from a first-guess
file of any time in the hour (IN5), and a caller relying on the default gets
exact while 2.9.0 says correct.  The release handshake
(:func:`gpuwm.bridges.bridge_abi_matches`) is a static byte search for the
contract marker, so the marker has to reach the switch, spelled default
first, for a stale build to be refused instead of run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpuwm import bridges, mpas_mesh

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "tools" / "rustwx" / "crates" / "rw-mpas"
BIN = CRATE / "src" / "bin" / "rw_mpas_lbc.rs"
LBC = CRATE / "src" / "lbc" / "mod.rs"

#: The first usage line of the producer before the switch (2.9.0 at
#: 0bf505764): what a stale build carries.
STALE_FIRST_LINE = (
    "rw_mpas_lbc --grid INIT.nc --out-dir DIR "
    "--start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS "
    "--interval YYYY-MM-DD_HH:MM:SS=MET_PATH [--interval ...] "
    "--nfglevels N --fg-interval-seconds N --extrap-airtemp MODE "
    "--use-spechumd yes|no --theta-adv-order N --coef-3rd-order X "
    "--oned-underflow preserve|reproduce-ifx-ftz "
    "[--config-attrs JSON] [--receipt JSON]")
#: The same line with the switch but the old exact default (the 2.8.9
#: correct-mode lane's build, 5330a31fb/59fd7b4a3).
EXACT_DEFAULT_FIRST_LINE = STALE_FIRST_LINE.replace(
    "[--config-attrs JSON]",
    "[--physics-mode exact|correct] [--config-attrs JSON]")


def _crate_first_usage_line() -> str:
    if not BIN.is_file():  # pragma: no cover - wheel install
        pytest.skip("no checkout crate in this install")
    source = BIN.read_text(encoding="utf-8")
    match = re.search(r'pub const ABI_MARKER: &str = "(.*?)";', source, re.S)
    assert match, "rw_mpas_lbc.rs no longer declares ABI_MARKER"
    # Rust string continuation: a backslash before the newline drops the
    # newline and the next line's leading whitespace.
    literal = re.sub(r"\\\n\s*", "", match.group(1)).replace("\\n", "\n")
    return literal.split("\n")[0]


def test_the_two_python_spellings_of_the_lbc_marker_agree():
    assert (bridges.BRIDGE_ABI_MARKERS["rw_mpas_lbc"]
            == mpas_mesh.LBC_ABI_MARKER.encode("utf-8"))


def test_the_marker_is_the_crates_first_usage_line_through_the_switch():
    first = _crate_first_usage_line()
    assert first.startswith(mpas_mesh.LBC_ABI_MARKER), (first,
                                                        mpas_mesh.LBC_ABI_MARKER)
    assert mpas_mesh.LBC_ABI_MARKER.endswith("[--physics-mode correct|exact]")


def test_the_crate_defaults_to_correct():
    if not LBC.is_file():  # pragma: no cover - wheel install
        pytest.skip("no checkout crate in this install")
    binary = BIN.read_text(encoding="utf-8")
    assert '.unwrap_or("correct")' in binary
    assert '.unwrap_or("exact")' not in binary
    module = LBC.read_text(encoding="utf-8")
    assert re.search(r"#\[default\]\s*Correct,", module)


@pytest.mark.parametrize("first_line, speaks", [
    (STALE_FIRST_LINE, False),
    (EXACT_DEFAULT_FIRST_LINE, False),
    (None, True),
])
def test_the_handshake_refuses_a_build_from_before_the_correct_default(
        tmp_path, first_line, speaks):
    if first_line is None:
        first_line = _crate_first_usage_line()
    fake = tmp_path / "rw_mpas_lbc"
    fake.write_bytes(b"\x7fELF padding " + first_line.encode("utf-8")
                     + b"\nrw_mpas_lbc --source ROW ...\x00")
    ok, reason = bridges.bridge_abi_matches("rw_mpas_lbc", fake)
    assert ok is speaks, reason
