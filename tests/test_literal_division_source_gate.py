"""A146 gate, source layer: no float division by a constant outside __fdiv_rn.

The breakage this prevents: NVRTC's NVVM, for compute_100 and compute_120
(every Blackwell card: RTX 5090, 5070 Ti, PRO 4500, PRO 6000) under
``-ftz=true`` (which CuPy appends to every ``RawModule``), compiles
``x / C`` for a compile-time constant ``C`` as ``x * RN(1/C)``.  That is not
IEEE division: on an RTX 5070 Ti with NVRTC 13.4, exhaustively over every
float in [1, 2), ``x / 3.0f`` is one ULP off for 33 percent of inputs,
``x / 60.0f`` (the advection flux) for 65 percent, ``x / tgammaf(4.0f)`` for
33 percent, while compute_90 and older keep ``div.rn`` and round correctly.
``__fdiv_rn(x, C)`` is never rewritten and is the IEEE quotient on every
card, so it is the one spelling a kernel may use for such a division.

This file imports neither CuPy nor NVRTC, so it runs on every leg, the CPU
legs that set ``GPUWM_NO_LOCAL_GPU`` included (tools/battery/always_files.txt
lists it).  The compiler census, which sees divisors a source scan cannot,
is tests/test_literal_division_gate.py on the GPU shard.
"""
from __future__ import annotations

import textwrap

from tools.literal_division_scan import (
    inline_kernel_offenders,
    inline_kernel_strings,
    kernel_file_offenders,
    literal_divisions,
)


def test_kernel_sources_divide_by_float_constants_only_through_fdiv_rn():
    """A146: every -ftz=true kernel source spells a constant divisor as
    ``__fdiv_rn(x, C)``; a plain ``x / C`` is rounded one ULP off on
    Blackwell cards for a large share of inputs."""
    offenders = kernel_file_offenders()
    assert not offenders, (
        "A146: NVRTC compiles a float division by a compile-time constant "
        "as a multiply by the rounded reciprocal on compute_100/120 under "
        "-ftz=true, which is not IEEE on Blackwell cards; spell these as "
        "__fdiv_rn(x, C):\n  " + "\n  ".join(offenders))


def test_inline_python_kernels_divide_by_float_constants_only_through_fdiv_rn():
    """A146: the same rule for CUDA source held in gpuwm's Python modules
    (RawKernel/RawModule strings and ElementwiseKernel operations), which
    CuPy compiles with the same appended -ftz=true."""
    offenders = inline_kernel_offenders()
    assert not offenders, (
        "A146: spell these constant divisions as __fdiv_rn(x, C):\n  "
        + "\n  ".join(offenders))


def test_the_source_scan_catches_every_spelling_it_claims():
    """The scan is only a gate if it fires: each plain spelling below is
    refused and each allowed one is not."""
    header = {"THOMPSON_AA_D0C": 1.0e-6, "THOMPSON_AA_D0R": 50.0e-6}
    refused = [
        "y = x / 3.0f;", "y = a * b / 60.0f;", "y = x / G;", "y /= 12.0f;",
        "y = x / 1.718e-5f;", "#define KD 7.0f\ny = x / KD;",
        "const float c = 9.81f; y = x / c;", "y = (a - b) / 0.075f;",
        # The four sites the first A146 repair missed (review of 4810400f1):
        # a header constant, a parenthesized constant, and a C math
        # function of a literal, which compute_120 folds and compute_89
        # calls at run time.
        "lamc = (double)(cce2[nu] / THOMPSON_AA_D0C);",
        "lamc = (double)(cce2[nu] / (THOMPSON_AA_D0R * 2.0f));",
        "v = d * c * powf(x, e) * tgammaf(4.0f + e) / tgammaf(4.0f);",
        "v = d * c * powf(x, e) * tgammaf(5.0f + e) / tgammaf(5.0f);",
        # A constant numerator built from math calls is not folded before
        # the rewrite can see it: tgammaf(a) * powf(b, c) / tgammaf(d).
        "c1 = tgammaf(0.53f) * powf(0.2f, -1.0f / 3.0f) / tgammaf(0.2f);",
        "z = zm + (t - tm) / (-9.81f / 1004.5f);",
    ]
    allowed = [
        "y = __fdiv_rn(x, 3.0f);", "y = x / 2.0f;", "y = x / 0.5f;",
        "y = x / 3.0;", "y = x / dz;", "y = (1.0f / 6.0f) * q;",
        "c = -9.81f / 1004.5f;", "y = x / 3.0f; // comment",
        "// y = x / 3.0f;", "y = x / G_fn(z);", "y = x / (2.0 * 1.0e-12);",
        "y = x / (float)n;", "y = x / (2.0f * 4.0f);", "y = x / (dt * 2.0f);",
        "y = x / tgamma(4.0);", "y = x / tgammaf(e);",
        "y = __fdiv_rn(a * tgammaf(4.0f + e), tgammaf(4.0f));",
    ]
    for src in refused:
        assert literal_divisions(src, header, header), src
    for src in allowed:
        hits = literal_divisions(src, header, header)
        if src.endswith("// comment"):
            assert hits == [(1, "3.0f")], src
        else:
            assert not hits, (src, hits)


def test_the_inline_scan_reads_every_kernel_string_form():
    """A RawKernel body, an f-string body and an ElementwiseKernel
    operation each reach the scan; a docstring does not."""
    module = textwrap.dedent('''
        """Docs may say y = x / 3.0f; freely."""
        import cupy as cp
        BODY = r"""
        extern "C" __global__ void k(const float* x, float* y)
        { y[0] = x[0] / 3.0f; }
        """
        OP = cp.ElementwiseKernel("float32 x", "float32 y",
                                  "y = x / 60.0f", "op")
        TIER = 64
        FMT = f"""__device__ float h(float a) {{ return a / 12.0f; }}
        // {TIER}"""
    ''')
    divisors = sorted(token for _lineno, text in inline_kernel_strings(module)
                      for _line, token in literal_divisions(text))
    assert divisors == ["12.0f", "3.0f", "60.0f"], divisors
