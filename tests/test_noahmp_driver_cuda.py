"""Device acceptance gate for the Noah-MP driver cold-start kernels.

Same bar as the CPU side: bitwise identity with the unmodified WRF v4.6.1
driver over the committed oracle fixture.  Nothing is relaxed for the GPU --
the gate is ``max_ulp == 0`` and no tolerance is applied anywhere.

Three things this has to prove, not one:

1. ``noahmp_driver_snow_init`` and ``noahmp_driver_noahmp_init`` reproduce
   every emitted column of every case of ``noahmp-driver.csv`` bit for bit.
2. The one transcendental the group is built on -- glibc 2.39's ``powf``,
   which gfortran emits for ``**`` with a real exponent -- is bitwise right on
   the device over the sweep in ``glibc-libm-fp32.csv``, the live glibc
   symbol's own output.  A mis-folded ``__constant__`` entry could otherwise
   hide inside a branch no case selects.
3. The gate can fail.  A deliberately perturbed constant must be rejected, or
   the two checks above are unfalsifiable.

Everything compared here comes from the oracle, not from the CPU
transcription, so a shared mistake in the two ports cannot pass this file.
"""

from __future__ import annotations

import csv
import os
import struct
import sys
from collections import defaultdict

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gpuwm.core.noahmp_driver_gpu import (  # noqa: E402
    NI_IN, NI_IN_STRIDE, NI_IX, NI_IX_STRIDE, NI_OUT, NLAY_MAX, NSNOW,
    NSOIL_MAX, SI_IN, SI_IN_STRIDE, SI_IX, SI_IX_STRIDE, SI_OUT,
    driver_module, driver_source, run_noahmp_init, run_snow_init,
)

_DATA = os.path.join(_ROOT, "gpuwm", "data", "noahmp", "oracle")
FIXTURE = os.path.join(_DATA, "noahmp-driver.csv")
LIBM = os.path.join(_DATA, "glibc-libm-fp32.csv")


def _bits(x) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


@pytest.fixture(scope="module")
def table():
    out = defaultdict(dict)
    with open(FIXTURE, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["field"], int(row["index"]))
            value = (int(row["value"]) if row["dtype"] == "int"
                     else np.uint32(int(row["bits"], 16)).view(np.float32))
            out[(row["leaf"], row["case"], row["stage"])][key] = value
    return out


def _cases(table, leaf):
    return sorted(c for (lf, c, st) in table
                  if lf == leaf and st == "input" and c != "table_identity")


# ---------------------------------------------------------------------------
# SNOW_INIT
# ---------------------------------------------------------------------------

def _pack_snow_init(table, cases):
    x = np.zeros((len(cases), SI_IN_STRIDE), dtype=np.float32)
    ix = np.zeros((len(cases), SI_IX_STRIDE), dtype=np.int32)
    for i, case in enumerate(cases):
        inp = table[("snow_init", case, "input")]
        nsnow = inp[("nsnow", 0)]
        nsoil = inp[("nsoil", 0)]
        ix[i, SI_IX["nsnow"]] = nsnow
        ix[i, SI_IX["nsoil"]] = nsoil
        x[i, SI_IN["swe"]] = inp[("swe", 0)]
        x[i, SI_IN["tgxy"]] = inp[("tgxy", 0)]
        x[i, SI_IN["snodep"]] = inp[("snodep", 0)]
        for k in range(1, nsoil + 1):
            x[i, SI_IN["zsoil"] + k - 1] = inp[("zsoil", k)]
        for k in range(-nsnow + 1, nsoil + 1):
            x[i, SI_IN["zsnsoxy"] + k + nsnow - 1] = inp[("zsnsoxy", k)]
        for field in ("tsnoxy", "snicexy", "snliqxy"):
            for k in range(-nsnow + 1, 1):
                x[i, SI_IN[field] + k + nsnow - 1] = inp[(field, k)]
    return x, ix


def test_snow_init_matches_the_oracle_bitwise(table):
    cases = _cases(table, "snow_init")
    x, ix = _pack_snow_init(table, cases)
    y = cp.asnumpy(run_snow_init(x, ix))
    bad = []
    for i, case in enumerate(cases):
        inp = table[("snow_init", case, "input")]
        out = table[("snow_init", case, "output")]
        nsnow = inp[("nsnow", 0)]
        nsoil = inp[("nsoil", 0)]
        if int(y[i, SI_OUT["isnowxy"]]) != out[("isnowxy", 0)]:
            bad.append(f"{case} isnowxy {y[i, SI_OUT['isnowxy']]} != "
                       f"{out[('isnowxy', 0)]}")
        for k in range(-nsnow + 1, nsoil + 1):
            got = y[i, SI_OUT["zsnsoxy"] + k + nsnow - 1]
            if _bits(got) != _bits(out[("zsnsoxy", k)]):
                bad.append(f"{case} zsnsoxy[{k}] {_bits(got)} != "
                           f"{_bits(out[('zsnsoxy', k)])}")
        for field in ("tsnoxy", "snicexy", "snliqxy"):
            for k in range(-nsnow + 1, 1):
                got = y[i, SI_OUT[field] + k + nsnow - 1]
                if _bits(got) != _bits(out[(field, k)]):
                    bad.append(f"{case} {field}[{k}] {_bits(got)} != "
                               f"{_bits(out[(field, k)])}")
    assert not bad, "\n".join(bad[:40])


# ---------------------------------------------------------------------------
# NOAHMP_INIT
# ---------------------------------------------------------------------------

_SCALARS = (
    ("snow", "snow"), ("snowh", "snowh"), ("canwat", "canwat"),
    ("tvxy", "tv"), ("tgxy", "tg"), ("canicexy", "canice"),
    ("canliqxy", "canliq"), ("eahxy", "eah"), ("tahxy", "tah"),
    ("cmxy", "cm"), ("chxy", "ch"), ("fwetxy", "fwet"),
    ("sneqvoxy", "sneqvo"), ("alboldxy", "albold"), ("qsnowxy", "qsnow"),
    ("qrainxy", "qrain"), ("wslakexy", "wslake"), ("zwtxy", "zwt"),
    ("waxy", "wa"), ("wtxy", "wt"), ("lai", "lai"), ("xsaixy", "xsai"),
    ("lfmassxy", "lfmass"), ("rtmassxy", "rtmass"), ("stmassxy", "stmass"),
    ("woodxy", "wood"), ("stblcpxy", "stblcp"), ("fastcpxy", "fastcp"),
    ("grainxy", "grain"), ("gddxy", "gdd"), ("t2mvxy", "t2mv"),
    ("t2mbxy", "t2mb"), ("chstarxy", "chstar"), ("qtdrain", "qtdrain"),
)


def _pack_noahmp_init(table, cases):
    probe = table[("noahmp_init", "table_identity", "probe")]
    isice = probe[("isice_table", 0)]
    isurban = probe[("isurban_table", 0)]
    iswater = probe[("iswater_table", 0)]
    isbarren = probe[("isbarren_table", 0)]
    lcz_1 = probe[("lcz_1_table", 0)]

    x = np.zeros((len(cases), NI_IN_STRIDE), dtype=np.float32)
    ix = np.zeros((len(cases), NI_IX_STRIDE), dtype=np.int32)
    for i, case in enumerate(cases):
        inp = table[("noahmp_init", case, "input")]
        nsoil = inp[("nsoil", 0)]
        nsnow = inp[("nsnow", 0)]
        ix[i, NI_IX["nsoil"]] = nsoil
        ix[i, NI_IX["nsnow"]] = nsnow
        ix[i, NI_IX["fndsnowh"]] = inp[("fndsnowh", 0)]
        ix[i, NI_IX["vegtyp"]] = inp[("ivgtyp", 0)]
        ix[i, NI_IX["cropcat"]] = inp[("cropcat", 0)]
        ix[i, NI_IX["sf_urban_physics"]] = inp[("sf_urban_physics", 0)]
        ix[i, NI_IX["isice"]] = isice
        ix[i, NI_IX["isurban"]] = isurban
        ix[i, NI_IX["iswater"]] = iswater
        ix[i, NI_IX["isbarren"]] = isbarren
        # The eleven LCZ classes are consecutive in MPTABLE.TBL; only LCZ_1 is
        # emitted, and the fixture's lcz_point column is LCZ_1 itself, so the
        # remaining ten are filled with the consecutive values the table has.
        for n in range(11):
            ix[i, NI_IX["lcz"] + n] = lcz_1 + n
        for name, field in (("xice", "xice"), ("tsk", "tsk"), ("lai", "lai"),
                            ("bexp", "bexp_table"),
                            ("smcmax", "smcmax_table"),
                            ("psisat", "psisat_table"),
                            ("snow", "snow"), ("snowh", "snowh")):
            x[i, NI_IN[name]] = inp[(field, 0)]
        # SLA_TABLE is emitted only for a category inside 1..NVEG; a column
        # outside it is urban or LCZ and never reaches 2187.
        x[i, NI_IN["sla"]] = inp.get(("sla_table", 0), np.float32("nan"))
        x[i, NI_IN["sla_natural"]] = np.float32("nan")
        for k in range(1, nsoil + 1):
            x[i, NI_IN["dzs"] + k - 1] = inp[("dzs", k)]
            x[i, NI_IN["tslb"] + k - 1] = inp[("tslb", k)]
            x[i, NI_IN["smois"] + k - 1] = inp[("smois", k)]
        for k in range(-nsnow + 1, nsoil + 1):
            x[i, NI_IN["zsnsoxy"] + k + nsnow - 1] = inp[("zsnsoxy", k)]
        for name, field in (("tsnoxy", "tsnoxy"), ("snicexy", "snicexy"),
                            ("snliqxy", "snliqxy")):
            for k in range(-nsnow + 1, 1):
                x[i, NI_IN[name] + k + nsnow - 1] = inp[(field, k)]
    return x, ix


def _check_noahmp_init(table, cases, y):
    bad = []
    for i, case in enumerate(cases):
        inp = table[("noahmp_init", case, "input")]
        out = table[("noahmp_init", case, "output")]
        nsoil = inp[("nsoil", 0)]
        nsnow = inp[("nsnow", 0)]
        if int(y[i, NI_OUT["isnow"]]) != out[("isnowxy", 0)]:
            bad.append(f"{case} isnowxy")
        if int(y[i, NI_OUT["cropcat"]]) != out[("cropcat", 0)]:
            bad.append(f"{case} cropcat")
        for field, slot in _SCALARS:
            got = y[i, NI_OUT[slot]]
            if _bits(got) != _bits(out[(field, 0)]):
                bad.append(f"{case} {field} {_bits(got)} != "
                           f"{_bits(out[(field, 0)])}")
        for field, slot in (("tslb", "tslb"), ("smois", "smois"),
                            ("sh2o", "sh2o")):
            for k in range(1, nsoil + 1):
                got = y[i, NI_OUT[slot] + k - 1]
                if _bits(got) != _bits(out[(field, k)]):
                    bad.append(f"{case} {field}[{k}] {_bits(got)} != "
                               f"{_bits(out[(field, k)])}")
        for k in range(-nsnow + 1, nsoil + 1):
            got = y[i, NI_OUT["zsnso"] + k + nsnow - 1]
            if _bits(got) != _bits(out[("zsnsoxy", k)]):
                bad.append(f"{case} zsnsoxy[{k}]")
        for field, slot in (("tsnoxy", "tsno"), ("snicexy", "snice"),
                            ("snliqxy", "snliq")):
            for k in range(-nsnow + 1, 1):
                got = y[i, NI_OUT[slot] + k + nsnow - 1]
                if _bits(got) != _bits(out[(field, k)]):
                    bad.append(f"{case} {field}[{k}]")
    return bad


def test_noahmp_init_matches_the_oracle_bitwise(table):
    cases = _cases(table, "noahmp_init")
    x, ix = _pack_noahmp_init(table, cases)
    y = cp.asnumpy(run_noahmp_init(x, ix))
    bad = _check_noahmp_init(table, cases, y)
    assert not bad, "\n".join(bad[:40])


def test_cpu_and_device_agree_bit_for_bit(table):
    """The two ports must not merely both match; they must be the same bits.

    Cheap, and it catches a device answer that happens to match the oracle on
    the fixture while diverging from the CPU transcription elsewhere.
    """
    from gpuwm.core.noahmp import (DEFAULT_VEGETATION_DATASET,
                                   load_noahmp_parameters)
    from gpuwm.core.noahmp_driver import LandUseIdentity, snow_init_column

    _, parameters = load_noahmp_parameters().vegetation_groups(
        DEFAULT_VEGETATION_DATASET)
    assert parameters.scalar("ISICE") == table[
        ("noahmp_init", "table_identity", "probe")][("isice_table", 0)]

    cases = _cases(table, "snow_init")
    x, ix = _pack_snow_init(table, cases)
    y = cp.asnumpy(run_snow_init(x, ix))
    bad = []
    for i, case in enumerate(cases):
        inp = table[("snow_init", case, "input")]
        nsnow, nsoil = inp[("nsnow", 0)], inp[("nsoil", 0)]
        cpu = snow_init_column(
            nsnow, nsoil,
            np.array([inp[("zsoil", k)] for k in range(1, nsoil + 1)],
                     dtype=np.float32),
            inp[("swe", 0)], inp[("tgxy", 0)], inp[("snodep", 0)],
            np.array([inp[("zsnsoxy", k)]
                      for k in range(-nsnow + 1, nsoil + 1)], dtype=np.float32),
            np.array([inp[("tsnoxy", k)] for k in range(-nsnow + 1, 1)],
                     dtype=np.float32),
            np.array([inp[("snicexy", k)] for k in range(-nsnow + 1, 1)],
                     dtype=np.float32),
            np.array([inp[("snliqxy", k)] for k in range(-nsnow + 1, 1)],
                     dtype=np.float32))
        if int(y[i, SI_OUT["isnowxy"]]) != cpu.isnow:
            bad.append(f"{case} isnow")
        for k in range(-nsnow + 1, nsoil + 1):
            if _bits(y[i, SI_OUT["zsnsoxy"] + k + nsnow - 1]) != _bits(
                    cpu.zsnso[k + nsnow - 1]):
                bad.append(f"{case} zsnso[{k}]")
    assert not bad, "\n".join(bad[:40])


# ---------------------------------------------------------------------------
# The transcendental, and falsifiability
# ---------------------------------------------------------------------------

def test_device_powf_matches_glibc(table):
    """``r_pow`` on the device against the live glibc 2.39 symbol's output."""
    rows = []
    with open(LIBM, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["call"] == "powf":
                rows.append(row)
    assert rows, "glibc-libm-fp32.csv carries no powf sweep"

    src = driver_source() + """
extern "C" __global__ void _drv_powf_probe(const float *x, const float *y,
                                           float *z, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { z[i] = r_pow(x[i], y[i]); }
}
"""
    module = driver_module(source=src)
    xs = np.array([np.uint32(int(r["a"], 16)).view(np.float32)
                   for r in rows], dtype=np.float32)
    ys = np.array([np.uint32(int(r["b"], 16)).view(np.float32)
                   for r in rows], dtype=np.float32)
    want = np.array([np.uint32(int(r["result"], 16)).view(np.float32)
                     for r in rows], dtype=np.float32)
    dx, dy = cp.asarray(xs), cp.asarray(ys)
    dz = cp.zeros(xs.size, dtype=cp.float32)
    module.get_function("_drv_powf_probe")(
        ((xs.size + 63) // 64,), (64,), (dx, dy, dz, np.int32(xs.size)))
    got = cp.asnumpy(dz)
    bad = [i for i in range(xs.size)
           if _bits(got[i]) != _bits(want[i])]
    assert not bad, f"{len(bad)} of {xs.size} powf values differ"


def test_the_gate_can_fail(table):
    """Perturb one __constant__ bit pattern; the fixture must reject it."""
    src = driver_source()
    # ALBOLDXY is assigned unconditionally at 2140, so a one-ULP move of the
    # 0.65 constant must be visible on every single case.  Picking a constant
    # only one branch reads would make this probe weaker than it looks.
    needle = "0x3F266666u,  // [30] 0.65           2140"
    assert needle in src
    perturbed = src.replace(needle,
                            "0x3F266667u,  // perturbed by one ULP")
    module = driver_module(source=perturbed)
    cases = _cases(table, "noahmp_init")
    x, ix = _pack_noahmp_init(table, cases)
    y = cp.asnumpy(run_noahmp_init(x, ix, module=module))
    bad = _check_noahmp_init(table, cases, y)
    assert bad, "a one-ULP perturbation of the freeze threshold went unnoticed"
