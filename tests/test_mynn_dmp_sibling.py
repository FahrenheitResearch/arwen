"""The DMP sibling preserves its historical source and tagged exports.

The ordinary mixing-length entry point now calls the rounded shared helper.
The sibling is still used only for its DMP exports. Its historical stripped
source digest remains fixed, and every byte outside that one entry point
must agree with the active source. Numerical DMP controls cover the actual
default and scalar-mixing paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_KDIR = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
_MARKER = "// MF-EXPORT"

#: Historical source identity retained independently of the active unit.
_FROZEN_SHA256 = (
    "b53ab90e634e61367afadfaa77667c8f2eb2430fc061ce9976509fe0e2f4490e"
)


def _read(name: str) -> str:
    return (_KDIR / name).read_text(encoding="utf-8")


def _historical_source():
    return "".join(line for line in _read("mynn_dmp_sibling.cu").splitlines(True)
                   if _MARKER not in line)


def _without_length_entry(source):
    start = source.index('extern "C" __global__\nvoid mynn_mixlength_default_columns(')
    end = source.index("\n}\n", start) + 3
    return source[:start].rstrip("\n") + "\n\n" + source[end:].lstrip("\n")


def test_historical_sibling_source_pin_unchanged():
    assert hashlib.sha256(_historical_source().encode()).hexdigest() == _FROZEN_SHA256


def test_only_the_ordinary_length_entry_differs_from_the_sibling():
    assert _without_length_entry(_read("mynn_pbl.cu")) == _without_length_entry(
        _historical_source())


def test_the_marker_is_actually_exercised():
    """A vacuous strip (no tagged lines) would mean the sibling exports
    nothing and the mixscalars lane silently reads garbage."""
    sibling = _read("mynn_dmp_sibling.cu")
    tagged = [line for line in sibling.splitlines() if _MARKER in line]
    assert len(tagged) >= 10, f"only {len(tagged)} tagged export lines"
    text = "\n".join(tagged)
    for needle in ("up_a_pre", "psig_w_o", "plume_active_o",
                   "limiter_adjustment_o"):
        assert needle in text, f"no tagged line exports {needle}"


def test_default_dmp_still_launches_the_original_module():
    """The DMP dispatch uses the sibling only for scalar-mixing exports."""
    core = (Path(__file__).resolve().parents[1] / "gpuwm" / "core"
            / "mynn_pbl_gpu.py").read_text(encoding="utf-8")
    assert 'get_kernel("mynn_pbl", "mynn_dmp_mf_columns")' in core
    assert 'get_kernel("mynn_dmp_sibling", "mynn_dmp_mf_columns")' in core
