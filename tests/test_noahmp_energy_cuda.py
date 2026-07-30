"""Device acceptance gate for the Noah-MP ENERGY assembly kernel.

Same bar as the CPU side: bitwise identity with the unmodified WRF v4.6.1
module over the committed oracle fixture.  Nothing is relaxed for the GPU.

Scope, so this is not read as more than it is.  ENERGY is a composition, and
``kernels/noahmp_energy.cu`` is the arithmetic ENERGY *itself* owns -- the
roughness geometry, FSNO, BTRAN, the psychrometric branches, the tile average,
EMISSI and TRAD.  The six subsystems it composes each have a device port in a
sibling ``.cu`` already, and running them inside this kernel is blocked by
three things that live in other lanes' files: three same-named copies of the
device glibc libm, ``TSNOSOI``/``PHASECHANGE`` exposed only as ``__global__``
entry points, and per-lane flat fixture packings in place of argument lists.
Their results are fed in from the same pinned fixture, so every number on both
sides of every comparison below still comes out of gfortran.

Two claims, not one:

1. ``noahmp_energy_assembly`` reproduces the ENERGY-owned outputs of all nine
   columns of ``noahmp-energy.csv`` bit for bit.
2. The device ``tanhf``/``expm1f`` transcriptions agree with the CPU ones
   statement for statement over a sweep far wider than ENERGY reaches.  Both
   were written from the same glibc 2.39 source but they are two separate
   pieces of code, and ``FSNO`` is the only ``TANH`` in Noah-MP -- a
   mis-folded constant would otherwise hide behind four snow columns.

The reference for (1) is the oracle CSV, never the CPU transcription.
"""

from __future__ import annotations

import csv
import os
import struct
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gpuwm.core.noahmp_energy_gpu import (                    # noqa: E402
    IN_SLOTS, N_IN, N_INT, N_OUT, OUT_SLOTS,
    evaluate_energy, evaluate_tanhf,
)

FIXTURE = os.path.join(_ROOT, "gpuwm", "data", "noahmp", "oracle",
                       "noahmp-energy.csv")
NSOIL = 4

CASES = (
    "veg_warm_day_dry", "veg_warm_night_rain", "snowpack_frozen_soil",
    "bare_thin_snow_melt", "veg_calm_desert_dry", "veg_deep_snow_saturated",
    "veg_subfreezing_canopy", "urban_snowfree", "veg_single_snow_layer",
)


def _from_bits(h):
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def _to_int(h):
    raw = int(h, 16)
    return raw - (1 << 32) if raw >= (1 << 31) else raw


def _load():
    table = {}
    with open(FIXTURE, newline="") as fh:
        for row in csv.DictReader(fh):
            case = table.setdefault(
                row["case"].strip(),
                {"opt": {}, "cfg": {}, "par": {}, "seed": {}, "in": {},
                 "out": {}, "undef": {}})
            case[row["role"].strip()][row["name"].strip()] = row["hex"].strip()
    return table


FX = _load()

#: Which fixture role each device input slot is read from.  ``out`` entries are
#: the pinned results of the subsystems this kernel does not re-run.
_SLOT_ROLE = {}
for _name in ("UU", "VV", "ELAI", "ESAI", "SNOWH", "SNEQV", "TG", "TV",
              "SFCPRS", "LWDN", "FVEG", "ZREF", "DT", "ACC_SSOIL",
              "PAHV", "PAHG", "PAHB"):
    _SLOT_ROLE[_name] = "in"
for _name in ("MFSNO", "SCFFAC", "Z0SNO", "Z0MVT", "HVT", "SNOW_EMIS"):
    _SLOT_ROLE[_name] = "par"
for _k in range(1, NSOIL + 1):
    _SLOT_ROLE[f"SH2O_{_k}"] = "in"
    _SLOT_ROLE[f"DZSNSO_{_k}"] = "in"
    _SLOT_ROLE[f"ZSOIL_{_k}"] = "in"
    _SLOT_ROLE[f"SMCWLT_{_k}"] = "par"
    _SLOT_ROLE[f"SMCREF_{_k}"] = "par"
for _name in ("IRC", "IRG", "IRB", "SHC", "SHG", "SHB", "EVC", "EVG", "EVB",
              "TR", "GHV", "GHB", "TGV", "TGB", "T2MV", "T2MB", "CHV", "CHB",
              "EAH", "QSFC", "Q2V", "Q2B"):
    _SLOT_ROLE[_name] = "out"
#: TV appears twice.  The psychrometric branch at :2211 reads the entry value;
#: TS = FVEG*TV + (1-FVEG)*TGB at :2298 reads what VEGE_FLUX wrote.  Feeding
#: one slot for both is how this kernel failed its first device run, on TS in
#: every vegetated column and nowhere else.
_SLOT_ROLE["TV_POST"] = "out"


def _pack(case):
    x = np.zeros(N_IN, dtype=np.float32)
    for i, slot in enumerate(IN_SLOTS):
        if slot == "EG":
            # parameters%EG(IST); IST is 1 everywhere in the admitted slice.
            assert _to_int(FX[case]["in"]["IST"]) == 1
            x[i] = _from_bits(FX[case]["par"]["EG_+1"])
            continue
        role = _SLOT_ROLE[slot]
        if slot == "TV_POST":
            x[i] = _from_bits(FX[case]["out"]["TV"])
            continue
        if "_" in slot and slot.rsplit("_", 1)[1].isdigit():
            base, idx = slot.rsplit("_", 1)
            key = f"{base}_{int(idx):+d}"
        else:
            key = slot
        x[i] = _from_bits(FX[case][role][key])
    ix = np.array([_to_int(FX[case]["par"]["NROOT"]),
                   _to_int(FX[case]["par"]["URBAN_FLAG"])], dtype=np.int32)
    return x, ix


def _expected(case, slot):
    """The pinned value for one device output slot, or 0.0 if WRF left it
    undefined -- which is the defined behaviour the ports are held to."""
    if "_" in slot and slot.rsplit("_", 1)[1].isdigit():
        base, idx = slot.rsplit("_", 1)
        key = f"{base}_{int(idx):+d}"
    else:
        key = slot
    out = FX[case]["out"]
    if key in out:
        # run_energy.F90 emits LOGICALs as integers, not as bit patterns, so
        # 00000001 there means 1 and not the denormal 1.4e-45.
        if key in ("FROZEN_CANOPY", "FROZEN_GROUND"):
            return float(_to_int(out[key]))
        return _from_bits(out[key])
    assert key in FX[case]["undef"], (case, key)
    return 0.0


def test_slot_tables_agree_with_the_kernel():
    assert len(IN_SLOTS) == N_IN
    assert len(OUT_SLOTS) == N_OUT
    assert N_INT == 2


def test_energy_assembly_bit_exact_on_device():
    xs, ixs = [], []
    for case in CASES:
        x, ix = _pack(case)
        xs.append(x)
        ixs.append(ix)
    got = evaluate_energy(np.stack(xs), np.stack(ixs))

    bad = []
    checked = 0
    for row, case in enumerate(CASES):
        for col, slot in enumerate(OUT_SLOTS):
            want = _expected(case, slot)
            g = np.float32(got[row, col])
            w = np.float32(want)
            checked += 1
            if g.tobytes() != w.tobytes():
                bad.append((case, slot, float(g), float(w)))
    assert not bad, bad[:12]
    assert checked == len(CASES) * N_OUT


def test_device_and_host_tanhf_agree_over_a_sweep():
    """The two transcriptions are separate code; make them measure each other.

    The sweep covers the whole tanhf argument range ENERGY can present, plus
    the branch boundaries of the glibc routine: |x| < 2**-55, |x| < 1,
    |x| >= 1, |x| >= 22, and expm1f's |x| > 0.5*ln2 and |x| > 1.5*ln2 arms.
    """
    from gpuwm.core.noahmp_libm import expm1f as host_expm1f
    from gpuwm.core.noahmp_libm import tanhf as host_tanhf

    rng = np.random.default_rng(20260725)
    samples = np.concatenate([
        np.float32([0.0, 1e-30, 5e-17, 0.25, 0.5, 0.6931472, 0.6931473,
                    0.99999994, 1.0, 1.0000001, 1.0397208, 1.75, 2.0,
                    8.999999, 9.0, 21.999998, 22.0, 22.000002, 30.0, 1e3]),
        np.float32(rng.uniform(1e-6, 1.0, 20000)),
        np.float32(rng.uniform(1.0, 22.0, 20000)),
        np.float32(rng.uniform(22.0, 90.0, 4000)),
        np.float32(np.exp(rng.uniform(np.log(1e-6), np.log(1e-1), 6000))),
    ])
    dev_tanh, dev_expm1 = evaluate_tanhf(samples)

    bad_t, bad_e = [], []
    for i, v in enumerate(samples):
        ht = np.float32(host_tanhf(v))
        if np.float32(dev_tanh[i]).tobytes() != ht.tobytes():
            bad_t.append((float(v), float(dev_tanh[i]), float(ht)))
        he = np.float32(host_expm1f(v))
        if np.float32(dev_expm1[i]).tobytes() != he.tobytes():
            bad_e.append((float(v), float(dev_expm1[i]), float(he)))
    assert not bad_t, bad_t[:8]
    assert not bad_e, bad_e[:8]
    assert samples.size >= 50000


def test_fsno_is_the_column_that_would_catch_a_wrong_tanh():
    """The device answer for FSNO has to come from glibc's tanhf, not a shim.

    Case 9 was given a snow density of exactly 250 kg/m3 so its FSNO argument
    lands just below 1.75, on an FP32 value where glibc's tanhf and an
    FP64-then-round shim differ.  This asserts the device lands on the glibc
    side of that split rather than merely agreeing with the fixture by luck.
    """
    case = "veg_single_snow_layer"
    x, ix = _pack(case)
    got = evaluate_energy(x[None, :], ix[None, :])
    want = np.float32(_expected(case, "FSNO"))
    assert np.float32(got[0, 0]).tobytes() == want.tobytes()

    # Reconstruct the exact FSNO argument this column presents, from the
    # fixture rather than from a remembered constant.
    import math

    from gpuwm.core.noahmp_libm import f32, powf

    snowh = _from_bits(FX[case]["in"]["SNOWH"])
    sneqv = _from_bits(FX[case]["in"]["SNEQV"])
    mfsno = _from_bits(FX[case]["par"]["MFSNO"])
    scffac = _from_bits(FX[case]["par"]["SCFFAC"])
    bdsno = f32(f32(sneqv) / f32(snowh))
    fmelt = powf(f32(bdsno / 100.0), mfsno)
    arg = np.float32(f32(f32(snowh) / f32(np.float32(scffac) * fmelt)))
    shim = np.float32(math.tanh(float(arg)))
    dev_tanh, _ = evaluate_tanhf(np.array([arg], dtype=np.float32))
    assert np.float32(dev_tanh[0]).tobytes() != shim.tobytes(), (
        f"the FSNO argument {float(arg)!r} no longer separates glibc tanhf "
        "from an FP64 shim; this has stopped being a control and the fixture "
        "case needs re-choosing")
