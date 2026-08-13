"""Shared source-scan pieces for the Noah-MP CUDA kernel gates.

WHY THIS MODULE EXISTS

``tests/test_noahmp_sflx_cuda.py`` and ``tests/test_noahmp_water_cuda.py``
each opened with two checks that read the ``.cu`` text and need no device at
all -- the SC_/OU_ layout agreement, the "this kernel has not grown a
column" claim, and the WATER file's own imported-section drift check, whose
docstring says in as many words "That test needs no GPU and runs
everywhere."

It did not.  Both files carried a module-level ``cp =
pytest.importorskip("cupy")`` PART WAY DOWN, under a "device" banner, and a
module-level skip is not local to the section it is written under: it fires
at import, so the CPU checks defined above it were skipped too.  On top of
that ``tests/conftest.py`` marks a module ``gpu`` in its entirety when a
cupy import appears at module scope, so under ``GPUWM_NO_LOCAL_GPU=1`` those
same checks were deselected a second time, for a second reason.  Four CPU
tests, on every box, on every cut, collected zero -- the same class as
``tests/test_first_products.py``'s line-568 skip.

The fix is structural rather than clever, because ``conftest``'s rule is
correct and must not be worked around: device tests live in a module that
imports cupy at module scope, source-scan tests live in a module that does
not.  This module holds what both halves read, so the split costs no
duplicated constant -- which matters especially for ``SCALARS`` and
``OUTPUTS``, whose entire purpose is to be ONE spelling of a layout.
"""

from __future__ import annotations

import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KDIR = os.path.join(_ROOT, "gpuwm", "core", "kernels")
SFLX_KERNEL = os.path.join(KDIR, "noahmp_sflx.cu")

#: Must match the SC_* block of noahmp_sflx.cu, in order.
SCALARS = ("swdown fsa fsr fira fsh fcev fgev fctr ssoil sav sag beg_wb "
           "canliq canice sneqv wa prcp ecan etran edir runsrf runsub dt "
           "qtldrn pah firr canhs irmirate irfirate acc_dwater acc_prcp "
           "acc_ecan acc_etran acc_edir").split()
#: Must match the OU_* block.
OUTPUTS = ("errwat acc_dwater acc_prcp acc_ecan acc_etran acc_edir errsw "
           "erreng end_wb").split()


def code(text: str) -> str:
    """The kernel with its prose removed.

    Every structural check is a scan for identifiers, and the CUDA files'
    headers *name* the identifiers they forbid in order to explain why they
    are forbidden.  Scanning the raw text would make the documentation trip
    its own gate, which is how a check ends up deleted instead of fixed.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", line)
                     for line in text.splitlines())


# ---------------------------------------------------------------------------
# WATER: the imported-section transform
# ---------------------------------------------------------------------------

SEP = "// " + "=" * 74
DASH = "// " + "-" * 74


def read_kernel(name: str) -> str:
    """Read a kernel source with LF line endings.

    The worktree may be checked out CRLF; the comparison is about the
    transcription, not about which platform wrote the file.
    """
    with open(os.path.join(KDIR, name), encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def soil_section(text: str) -> str:
    a = text.index("#define NSOIL 4")
    b = text.rindex(SEP, a, text.index("// Host-facing kernels."))
    s = text[a:b]
    return re.sub(r"^#define (IN_STRIDE|OUT_STRIDE)\s+\d+.*\n", "", s,
                  flags=re.M).rstrip() + "\n"


def snow_section(text: str) -> str:
    a = text.index("#define NSNOW 3")
    b = text.rindex(SEP, a,
                    text.index("// Entry points.  One thread per fixture case."))
    s = text[a:b]
    for start, end in (
            (DASH + "\n// glibc __exp2f_data",
             DASH + "\n// Every float32 constant"),
            (DASH + "\n// rounding-pinned primitives",
             DASH + "\n// glibc 2.39 expf"),
            (DASH + "\n// glibc 2.39 expf",
             DASH + "\n// column state, in WRF's index convention")):
        i = s.index(start)
        s = s[:i] + s[s.index(end, i):]
    s = re.sub(r"^#define (NSOIL|IN_STRIDE|OUT_STRIDE)\s+\d+.*\n", "", s,
               flags=re.M)
    names = sorted(set(re.findall(r"#define (K_[A-Z0-9_]+)", s)), key=len,
                   reverse=True)
    assert names, "no K_* macros found in the snow section"
    s = re.sub(r"\bC_F32\b", "C_SN_F32", s)
    for n in names:
        s = re.sub(r"\b%s\b" % n, "SN_" + n, s)
    return s.rstrip() + "\n"
