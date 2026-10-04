"""The DMP sibling preserves its historical source and tagged exports.

The ordinary mixing-length entry point now calls the rounded shared helper
with options 1 and 2, and initialization forwards the selected option.
The sibling is still used only for its DMP exports. Its stripped source
digest is re-pinned for the level-major layout, and every byte outside that
mixing-length changes must agree with the active source. Numerical DMP controls
cover the actual default and scalar-mixing paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_KDIR = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
_MARKER = "// MF-EXPORT"

#: Re-pinned for the level-major MYNN layout (speed lane, 2.8.1): the same
#: addressing-only edit as mynn_pbl.cu, whose FROZEN_MODULE_DIGESTS entry in
#: tests/test_mp8_frozen.py records the identity evidence.  The scalar-mixing
#: path it serves replayed bitwise identical on the RTX 4090 and RTX 5090.
#: Re-pinned for A146 (a98f2482e): the constant-divisor float divisions both
#: siblings share are spelled __fdiv_rn.  Previously 9c5c9543.
_FROZEN_SHA256 = (
    "06cada42b47a4f7874641216e491917eabb53185abd7cd0465e946978ba56c76"
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


def _without_local_length_option(source):
    start = source.index("// WRF v4.6.1 module_bl_mynn.F:2100-2232")
    end = source.index("// module_bl_mynn.F:1999-2098", start)
    source = source[:start] + source[end:]
    edits = (
        ("MynnColumn<real> dld, int nz, int mixlength)",
         "MynnColumn<real> dld, int nz)"),
        ("    if (mixlength == 2) {\n"
         "        mynn_mym_length_local_column(dz, zw, qke, dtv, edmf_w, edmf_a,\n"
         "            rmo, fltv, zi, psig_bl, el, qkw, qtke, nz);\n"
         "        return;\n    }\n", ""),
        ("int initialize_qke, int mixlength, int nz, int ncol)",
         "int initialize_qke, int nz, int ncol)"),
        ("el, qkw, qtke, thetaw, elblavg, dlu, dld, nz,\n"
         "                               mixlength);",
         "el, qkw, qtke, thetaw, elblavg, dlu, dld, nz);"),
    )
    for current, historical in edits:
        assert source.count(current) == 1, current
        source = source.replace(current, historical)
    return source


def test_only_mixing_length_entries_differ_from_the_sibling():
    # Do not let the new length option silently alter DMP scalar fluxes.
    assert _without_local_length_option(_without_length_entry(
        _read("mynn_pbl.cu"))) == _without_length_entry(
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
