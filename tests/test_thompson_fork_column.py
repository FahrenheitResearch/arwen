"""CPU gates on the fork Thompson column unit (thompson_version =
"wrf_39_noaa"): its constants are the fork's own words, it compiles without
FMA contraction or flush-to-zero, and the adapter routes the generation to
it.  The word-for-word comparison with the fork's Fortran is the GPU test
tests/test_thompson_fork_column_gpu.py.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "thompson_fork_oracle"
RECEIPT = TOOLS / "receipts" / "fork_constants.txt"
HEADER = ROOT / "gpuwm" / "core" / "kernels" / "thompson_fork_constants.cuh"


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_constants_header_is_the_dumped_fork_words():
    """thompson_fork_constants.cuh is make_constants_header.py's rendering
    of the committed dump, byte for byte: a hand edit, or a dump from
    another build, is refused."""
    tool = _tool("make_constants_header")
    import hashlib
    raw = RECEIPT.read_bytes()
    expected = tool.render(tool.parse(RECEIPT),
                           hashlib.sha256(raw).hexdigest())
    assert HEADER.read_text(encoding="utf-8") == expected


def test_the_dump_holds_the_forks_known_parameters():
    """Spot values the fork declares as PARAMETERs (module_mp_thompson.F
    :66-205), read back from the dump as exact words."""
    words = {}
    for line in RECEIPT.read_text().splitlines():
        name, index, kind, value = line.split()
        words[(name, int(index))] = (kind, value)

    def f32(name, index=1):
        kind, value = words[(name, index)]
        assert kind == "f"
        return struct.unpack("<f", struct.pack("<I", int(value, 16)))[0]

    import numpy as np
    assert f32("R1") == np.float32(1.0e-12)
    assert f32("D0s") == np.float32(200.0e-6)       # the fork's 200 microns
    assert f32("HGFR") == np.float32(235.16)
    assert f32("am_r") == np.float32(np.float32(3.1415926536)
                                     * np.float32(1000.0)) / np.float32(6.0)
    assert f32("r_g", 1) == np.float32(1.0e-5)       # 28-entry graupel axis
    assert words[("ntb_g", 1)] == ("i", "28")
    assert words[("ntb_s", 1)] == ("i", "28")


def test_the_unit_compiles_without_fma_or_flush_to_zero():
    """gfortran -O2 on x86-64 neither contracts a multiply-add nor flushes
    a subnormal; CuPy's RawModule appends -ftz=true, so the unit takes the
    direct NVRTC route with both switched off (gpuwm/core/thompson_fork.py
    MODULE_OPTIONS)."""
    from gpuwm.core import thompson_fork
    assert thompson_fork.MODULE_OPTIONS == (
        "-std=c++17", "--fmad=false", "--ftz=false")
    text = (ROOT / "gpuwm" / "core" / "thompson_fork.py").read_text(
        encoding="utf-8")
    assert "nvrtc_ptx_cache as compiler" in text
    assert '("-std=c++17", "--fmad=false", "--ftz=false")' in text
    assert "RawModule" not in text.replace("RawModule appends", "")


def test_the_arena_slots_are_the_kernels_own_count():
    """The launcher sizes the column arena from WS_FLOATS / WS_DOUBLES /
    WS_BOOLS; the kernel takes its slots in three places (the driver's
    arrays, then mp_thompson's or calc_refl10cm's, which reuse the same
    slots after it).  A slot taken past the count would write another
    thread's columns, so the count is read off the source here."""
    import re
    from gpuwm.core import thompson_fork as tf

    src = (ROOT / "gpuwm" / "core" / "kernels" / "thompson_fork.cu").read_text(
        encoding="utf-8")
    defines = {name: int(value) for name, value in re.findall(
        r"#define TF_WS_(FLOATS|DOUBLES|BOOLS) (\d+)", src)}
    assert defines == {"FLOATS": tf.WS_FLOATS, "DOUBLES": tf.WS_DOUBLES,
                       "BOOLS": tf.WS_BOOLS}

    def body(signature):
        start = src.index(signature)
        end = src.find("\n}\n", start)
        return src[start:end]

    driver = body('extern "C" __global__ void thompson_fork_column(')
    mp = body("__device__ bool tf_mp_thompson(")
    refl = body("__device__ void tf_refl10cm(")
    for kind, want in (("F", tf.WS_FLOATS), ("D", tf.WS_DOUBLES),
                       ("B", tf.WS_BOOLS)):
        take = f"ws.{kind}()"
        assert src.count(take) == (driver.count(take) + mp.count(take)
                                   + refl.count(take))
        assert driver.count(take) + max(mp.count(take),
                                        refl.count(take)) == want, kind
    assert "TF_KMAX" not in src
    assert tf.workspace_bytes(50, 64) == 51 * 64 * (79 * 4 + 7 * 8 + 5)
    assert tf.tile_threads(188, 10_000_000) == 188 * 4 * 64
    assert tf.tile_threads(188, 65) == 128
    with pytest.raises(ValueError):
        tf.workspace_bytes(1, 64)


def test_the_adapter_routes_the_fork_generation_to_the_column_unit(
        monkeypatch):
    pytest.importorskip("cupy")
    import gpuwm.core.microphysics_aerosol as adapter

    calls = []

    def fork_call(state, cfg, dt, table_root, *, refl_10cm_due):
        calls.append((cfg.thompson_version, dt, refl_10cm_due))
        return "fork"

    import gpuwm.wrf_exact as exact
    monkeypatch.setattr(exact, "ENABLED", True)
    monkeypatch.setattr(adapter, "_apply_thompson_fork_call", fork_call)
    state = SimpleNamespace(**{name: object() for name in (
        adapter.REQUIRED_STATE_FIELDS + adapter.REQUIRED_SURFACE_FIELDS)})
    cfg = SimpleNamespace(mp_physics=28, thompson_version="wrf_39_noaa",
                          thompson_fork_snow_fall="wrf_39_noaa")
    assert adapter._apply_thompson_aerosol(state, cfg, 20.0,
                                           refl_10cm_due=True) == "fork"
    assert calls == [("wrf_39_noaa", 20.0, True)]
