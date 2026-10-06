"""The RK advection attribution's transient kernels still compile.

The breakage this prevents: ``tools/bigstep_wrf471_oracle/rk_advection_diagnostic.py``
splices kernel text (two ``zface_half`` calls and the open-boundary face
helpers) into the WRF-exact branch of ``flux_div_scalar``.  When a helper's
signature moves, the splice stops compiling and nothing else notices: the
tool runs only on a GPU, by hand, when someone reruns the attribution
recorded in ``rk_README.md``.  The vertical-order lane (286-vadv5) gave
``zface_half`` a trailing ``vorder`` and left this splice on the old
ten-argument call, so the ``wrf_flux_and_accumulation_order_no_fma`` variant
failed NVRTC (NVRTC_ERROR_COMPILATION) under the WRF-exact advection
selector, the only compile in which the spliced branch is live.

Two layers: an argument-count check that needs no CUDA at all (it runs on
the stage-1 CPU leg), and NVRTC compiles of every variant to PTX, which open
no device but import cupy, so they run on the GPU legs.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from gpuwm.core.kernels import module_source

TOOL = (Path(__file__).resolve().parents[1] / "tools" / "bigstep_wrf471_oracle"
        / "rk_advection_diagnostic.py")
#: The selector the attribution is run under (gpuwm.wrf_exact.STRICT_OPTIONS
#: with GPUWM_WRF_EXACT_ADVECTION=1): the splice lives in the
#: GPUWM_WRF_EXACT_C_ADVECTION branch, so only this compile parses it.
EXACT_ADVECTION = ("-DGPUWM_WRF_EXACT=1", "-DGPUWM_WRF_EXACT_C_ADVECTION=1")
_DEFINITION = re.compile(r"^real\s+(\w+)\s*\(", re.MULTILINE)


def _tool():
    spec = importlib.util.spec_from_file_location("rk_advection_diagnostic", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arguments(text: str, open_paren: int) -> list[str]:
    """Top-level comma-separated arguments of the call opening at ``open_paren``."""
    depth, start, out = 0, open_paren + 1, []
    for i in range(open_paren, len(text)):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                out.append(text[start:i])
                return [a.strip() for a in out if a.strip()]
        elif ch == "," and depth == 1:
            out.append(text[start:i])
            start = i + 1
    raise AssertionError(f"unbalanced call at {open_paren}")


def _parameter_counts(source: str) -> dict[str, int]:
    """``real name(...)`` device helpers of the module and their parameter counts."""
    counts = {}
    for match in _DEFINITION.finditer(source):
        counts[match.group(1)] = len(_arguments(source, match.end() - 1))
    return counts


def test_every_helper_call_the_splice_adds_matches_the_helper_definition():
    tool = _tool()
    baseline = module_source("advection")
    spliced = tool.wrf_scalar_accumulation(tool.wrf_flux_order(baseline))
    start = spliced.index("    if (open_x || open_y) {", spliced.index("void flux_div_scalar("))
    block = spliced[start:spliced.index("\n    real fx[2]", start)]
    counts = _parameter_counts(baseline)
    assert counts.get("zface_half") == 11, counts.get("zface_half")
    checked = 0
    for match in re.finditer(r"\b(\w+)\s*\(", block):
        name = match.group(1)
        if name not in counts:
            continue
        got = len(_arguments(block, match.end() - 1))
        assert got == counts[name], (name, got, counts[name], block[match.start():match.start() + 120])
        checked += 1
    assert checked >= 6  # two each of yface_cell_open, xface_cell_open, zface_half


def test_the_accumulation_splice_passes_the_kernel_vertical_order():
    spliced = _tool().variant_sources(module_source("advection"))[
        "wrf_flux_and_accumulation_order_no_fma"]
    assert "k + 1, j, i, nz, ny, nx, fnm, fnp, vorder);" in spliced


def _ptx(source, defines):
    from gpuwm.nvrtc_cache_key import compile_program
    blob = compile_program(source, tuple(_tool().VARIANT_OPTIONS) + defines
                           + ("-arch=compute_89",), name="advection.cu", target="ptx")
    return (blob.decode("utf-8") if isinstance(blob, bytes) else str(blob)).rstrip("\x00")


@pytest.mark.parametrize("defines", ((), EXACT_ADVECTION), ids=("default", "wrf_exact_advection"))
def test_every_rk_attribution_variant_compiles(defines):
    pytest.importorskip("cupy")
    variants = _tool().variant_sources(module_source("advection"))
    assert set(variants) == {"production", "no_fma", "wrf_flux_order_no_fma",
                             "wrf_flux_and_accumulation_order_no_fma"}
    for name, source in variants.items():
        if source is None:
            continue
        assert ".entry flux_div_scalar(" in _ptx(source, defines), name


def test_the_accumulation_splice_is_live_under_the_exact_selector():
    """The splice changes the kernel it means to change (it is dead text at
    the default compile, where its branch is preprocessed out)."""
    pytest.importorskip("cupy")
    variants = _tool().variant_sources(module_source("advection"))
    assert (_ptx(variants["wrf_flux_order_no_fma"], EXACT_ADVECTION)
            != _ptx(variants["wrf_flux_and_accumulation_order_no_fma"], EXACT_ADVECTION))
