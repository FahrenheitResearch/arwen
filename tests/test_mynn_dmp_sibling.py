"""The MYNN sibling DMP unit is the frozen unit plus tagged export lines.

The frozen-unit sibling pattern, third application (mf-close lane).
The W4 full
mass-flux admission needs four plume-edge terms that are register-local
in the frozen ``kernels/mynn_pbl.cu`` (byte pin ``b53ab90e...``,
``tests/test_mp8_frozen.py``): the PRE-limiter ``up_a``, ``psig_w``, the
``NUP2 > 0`` plume-active gate, and the heat-flux-limiter adjustment.
Editing the frozen unit is refused, so they are exported by a SIBLING
translation unit ``kernels/mynn_dmp_sibling.cu`` instead, dispatched only
under ``bl_mynn_mixscalars=1`` (``mynn_pbl_gpu.mynn_dmp_mf_cuda``); the
default path launches the frozen kernel exactly as before, so the
mixscalars=0 lane is bit-identical by construction.

The construction proof (the frozen-unit sibling pattern, adapted):
every addition is a COMPLETE LINE carrying the ``MF-EXPORT`` marker, and
stripping every marked line from the sibling must restore the frozen
source BYTE FOR BYTE.  Anything a line-strip cannot restore — an edited
expression, a changed constant, a "small fix" — fails as a byte diff.
The numerical gate (sibling classic outputs bit-identical to the frozen
kernel, exports bit-equal to the CPU reference, device flux chain
end-to-end) is ``tools/mynn_pbl_wrf461_oracle/probe_mynn_dmp_sibling_gpu
.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_KDIR = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
_MARKER = "// MF-EXPORT"

#: The frozen unit's byte pin, restated from tests/test_mp8_frozen.py so
#: this gate fails loudly on its own if the base ever moves.
_FROZEN_SHA256 = (
    "b53ab90e634e61367afadfaa77667c8f2eb2430fc061ce9976509fe0e2f4490e"
)


def _read(name: str) -> str:
    return (_KDIR / name).read_text(encoding="utf-8")


def test_frozen_unit_pin_unchanged():
    data = (_KDIR / "mynn_pbl.cu").read_bytes()
    assert hashlib.sha256(data).hexdigest() == _FROZEN_SHA256, (
        "kernels/mynn_pbl.cu no longer matches its byte pin -- the frozen "
        "unit was edited; the sibling proof below is void until the pin "
        "question is settled"
    )


def test_sibling_is_frozen_plus_tagged_lines_only():
    frozen = _read("mynn_pbl.cu")
    sibling = _read("mynn_dmp_sibling.cu")
    stripped = "".join(
        line for line in sibling.splitlines(keepends=True)
        if _MARKER not in line
    )
    assert stripped == frozen, (
        "mynn_dmp_sibling.cu is not the frozen mynn_pbl.cu plus "
        "MF-EXPORT lines: stripping the tagged lines does not "
        "restore the frozen bytes (an existing line was edited, or an "
        "addition is missing its marker)"
    )


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


def test_default_lane_still_launches_the_frozen_module():
    """The dispatch site may name the sibling module only on the
    mixscalars branch; the unconditional launch stays the frozen unit."""
    core = (Path(__file__).resolve().parents[1] / "gpuwm" / "core"
            / "mynn_pbl_gpu.py").read_text(encoding="utf-8")
    assert 'get_kernel("mynn_pbl", "mynn_dmp_mf_columns")' in core
    assert 'get_kernel("mynn_dmp_sibling", "mynn_dmp_mf_columns")' in core
