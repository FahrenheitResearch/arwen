"""The Noah-MP CUDA gates' checks that need no device, where they can run.

These four assertions were written inside
``tests/test_noahmp_sflx_cuda.py`` and ``tests/test_noahmp_water_cuda.py``,
above each file's ``cp = pytest.importorskip("cupy")``, under a banner
saying "the checks that need no GPU".  They have never run.  A module-level
skip fires at IMPORT, so it takes the tests defined above it with it, and
``tests/conftest.py`` separately marks a module ``gpu`` in its entirety when
a cupy import appears at module scope.  Two independent reasons, same
outcome: collected zero, on every box, on every cut.

The WATER pair is the sharpest illustration -- its own module docstring
says of ``test_imported_sections_match_their_sources``, "That test needs no
GPU and runs everywhere."  It ran nowhere.

What they protect is real and is not device work:

* ``noahmp_sflx.cu``'s SC_/OU_ index blocks and the Python packing tables
  are two spellings of one layout.  A field added on one side alone shifts
  every column after it, which a bitwise gate then reports as physics.
* ``noahmp_sflx.cu`` covers ERROR and NOAHMP_SFLX's marshalling only, and
  must not quietly grow a call into another subsystem.
* ``noahmp_water.cu`` carries verbatim COPIES of ``noahmp_soilwater.cu``
  and ``noahmp_snow.cu``, because CuPy's RawModule compiles from a string
  with no include path.  If either source lane fixes an arithmetic site,
  this is what fails until the fix is carried across -- instead of the
  water file holding an older transcription its own device gate happens to
  accept.
* And the drift check's own falsifiability: a one-character change to a
  source must break the equality.
"""

from __future__ import annotations

import re

from noahmp_kernel_sources import (OUTPUTS, SCALARS, SFLX_KERNEL, code,
                                   read_kernel, snow_section, soil_section)


# ---------------------------------------------------------------------------
# NOAHMP_SFLX
# ---------------------------------------------------------------------------


def test_scalar_packing_matches_the_kernel():
    """The .cu's SC_*/OU_* blocks and the packing tables are one layout."""
    text = open(SFLX_KERNEL, encoding="ascii").read()
    for prefix, names, count_macro in (("SC_", SCALARS, "NSC"),
                                       ("OU_", OUTPUTS, "NOUT")):
        found = re.findall(rf"^#define {prefix}(\w+)\s+(\d+)$", text, re.M)
        assert [n.lower() for n, _ in found] == names, prefix
        assert [int(v) for _, v in found] == list(range(len(names))), prefix
        n = re.search(rf"^#define {count_macro}\s+(\d+)$", text, re.M)
        assert n and int(n.group(1)) == len(names), count_macro


def test_kernel_does_not_claim_the_column():
    """``noahmp_sflx.cu`` covers ERROR and the marshalling only.

    If it ever grows a call into another subsystem it must stop pretending
    otherwise -- and re-transcribing ENERGY or WATER there is the specific
    thing that lane must not do.
    """
    text = code(open(SFLX_KERNEL, encoding="ascii").read())
    entries = re.findall(r'extern "C" __global__\s*\nvoid (\w+)\(', text)
    assert entries == ["k_sflx_error", "k_sflx_marshal"], entries
    for banned in ("expf", "powf", "logf", "tanhf", "__expf", "exp2f"):
        assert banned not in text, (
            f"{banned} appeared: neither ERROR nor the marshalling evaluates "
            "a transcendental, which build_sflx_compose.sh proves with nm -u")


# ---------------------------------------------------------------------------
# WATER: the imported sections have not forked
# ---------------------------------------------------------------------------


def test_imported_sections_match_their_sources():
    """noahmp_water.cu's two copies must still equal what they came from."""
    water = read_kernel("noahmp_water.cu")
    for marker, source, derive in (
            ("noahmp_soilwater.cu", "noahmp_soilwater.cu", soil_section),
            ("noahmp_snow.cu (K_* -> SN_K_*)", "noahmp_snow.cu",
             snow_section)):
        begin = f"// >>> BEGIN imported section: {marker}"
        end = f"// <<< END imported section: {source}"
        i = water.index(begin) + len(begin) + 1
        j = water.index(end)
        got = water[i:j]
        want = derive(read_kernel(source))
        assert got == want, (
            f"{marker} section in noahmp_water.cu has drifted from "
            f"gpuwm/core/kernels/{source}")


def test_the_drift_check_can_fail():
    """A one-character change to a source must break the equality above."""
    s = snow_section(read_kernel("noahmp_snow.cu"))
    assert s != snow_section(read_kernel("noahmp_snow.cu").replace(
        "0x3CCCCCCDu, 0x3CCCCCCDu", "0x3CCCCCCEu, 0x3CCCCCCDu", 1))
