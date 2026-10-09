"""Classic Thompson (mp=8) reads rain collecting graupel from the one slab
WRF builds, never through WRF v4.6.1's idx_bg1=5 subscript.

Breakage prevented: WRF v4.6.1 allocates tcg_racg, tmr_racg, tcr_gacr,
tnr_racg and tnr_gacr with a graupel-density axis of extent 1 when the
scheme is not hail aware and then indexes that axis with 5
(module_mp_thompson.F:465, :607-615, :2527-2545), which lands four
rain-intercept bins further on and past the end of the arrays for the
largest rain intercepts.  thompson.cu reproduced that offset (+4*37*37
words) in all six of its reads and silently zeroed the rates where the
offset left the table, so mp=8 collected rain onto graupel with the rates
of the wrong drop sizes, while mp=28 read the slab.  These checks are pure
text over the exact source nvrtc and the host build compile; the numbers
are graded by tests/test_thompson_real_column_host_parity.py against WRF
compiled with only that index corrected.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from gpuwm.core.kernels import module_source

_FIXTURES = Path(__file__).parent / "fixtures"
#: WRF v4.6.1's active-collision columns, stock driver and source.
FIXTURE = _FIXTURES / "thompson-active-collision.json"
#: The same three columns through the same driver, with only WRF v4.6.1's
#: eight rain-graupel table reads changed to MIN(idx_bg(k),dimNRHG)
#: (tools/thompson_wrf461_oracle/active_collision_fixture.py
#: --corrected-racg).  Every column holds rain and graupel together, and
#: classic Thompson reads the one slab WRF builds (thompson_racg_index), a
#: declared divergence, so tests/test_thompson_active_collision.py grades
#: the adapter on this reference on the GPU shard; the unmodified answers
#: stay committed beside it as provenance.
FIXTURE_CORRECTED_RACG = _FIXTURES / "thompson-active-collision-corrected-racg.json"

#: The six kernels of thompson.cu that read the rain-graupel tables.
RACG_KERNELS = (
    "thompson_warm_frozen_source_network",
    "thompson_rain_graupel_collection",
    "thompson_cold_rain_snow_graupel_network",
    "thompson_cold_rain_source_network",
    "thompson_frozen_vapor_network",
    "thompson_frozen_vapor_cloud_network",
)
_TABLES = ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")
_SLAB_FIVE = re.compile(r"4\s*\*\s*37\s*\*\s*37|37\s*\*\s*37\s*\*\s*4")


def _kernel_bodies(text):
    """``{name: body}`` for every ``extern "C" __global__`` kernel."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(
        r'extern "C" __global__[^(]*?\b(\w+)\(', text)]
    bodies = {}
    for i, (start, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        bodies[name] = text[start:end]
    return bodies


def _code(text):
    """The source with its // comments removed (the comments that explain
    the divergence quote WRF's offset)."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def test_no_kernel_source_offsets_a_table_read_to_slab_five():
    for module in ("thompson", "thompson_aerosol_cold", "thompson_aerosol_warm"):
        text = _code(module_source(module))
        assert not _SLAB_FIVE.search(text), (
            f"{module}: a 4*37*37 slab offset is back in the kernel source")


def test_every_classic_rain_graupel_read_goes_through_the_one_slab_index():
    text = module_source("thompson")
    helper = text[text.index("thompson_racg_index("):]
    helper = helper[:helper.index("}")]
    # zero-based (g1, g, r1, r) over a (37, 37, 1, 37, 37) table: no
    # density term at all
    assert "(size_t)rain_intercept_bin" in helper
    assert "idx_bg" not in helper.split("{", 1)[1]
    bodies = _kernel_bodies(text)
    readers = sorted(name for name, body in bodies.items()
                     if re.search(r"\b(?:%s)\[" % "|".join(_TABLES), body))
    assert readers == sorted(RACG_KERNELS), readers
    for name in RACG_KERNELS:
        body = bodies[name]
        index_names = set(re.findall(
            r"\b(?:%s)\[(\w+)\]" % "|".join(_TABLES), body))
        assert index_names == {"table_idx"}, (name, index_names)
        racg_reads = [m.start() for m in re.finditer(
            r"\b(?:%s)\[table_idx\]" % "|".join(_TABLES), body)]
        for at in racg_reads:
            before = body[:at]
            last = before.rfind("const size_t table_idx")
            assert before[last:].startswith(
                "const size_t table_idx = thompson_racg_index("), (
                name, body[last:at][:200])


def test_corrected_reference_keeps_every_stock_input_and_changes_only_the_index():
    """Breakage prevented: a re-cut reference that drifted in its inputs,
    driver, flags or tables (or swapped in an easier column) would let the
    adapter gate pass against something other than the stock calls.  Kept
    in this cupy-free module so the CPU stage runs it: the conftest marks
    every item of a cupy-importing module ``gpu`` and skips it there."""
    stock = json.loads(FIXTURE.read_text(encoding="utf-8"))
    corrected = json.loads(FIXTURE_CORRECTED_RACG.read_text(encoding="utf-8"))
    variant = corrected["source_variant"]
    assert variant["oracle_source_variant"] == "corrected-racg"
    assert variant["replacement_count"] == 8
    assert variant["old"] == "idx_g1,idx_g,idx_bg(k),idx_r1,idx_r)"
    assert variant["new"] == "idx_g1,idx_g,MIN(idx_bg(k),dimNRHG),idx_r1,idx_r)"
    assert corrected["source_sha256"] == stock["source_sha256"]
    assert (corrected["support_sha256"]["module_mp_thompson.F"]
            == variant["corrected_source_sha256"])
    for key in ("reference_commit", "driver_parent_sha256", "compiler",
                "flags", "table_identity"):
        assert corrected[key] == stock[key], key
    assert [c["name"] for c in corrected["cases"]] == [
        c["name"] for c in stock["cases"]]
    for a, b in zip(stock["cases"], corrected["cases"], strict=True):
        assert a["dt_s"] == b["dt_s"]
        assert a["before"] == b["before"]
        # rain and graupel meet at every level, which is why the
        # reference had to be re-cut
        assert all(r["qr"] > 1e-12 and r["qg"] > 1e-12 for r in a["before"])
