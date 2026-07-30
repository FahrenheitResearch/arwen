#!/usr/bin/env python3
"""max_ulp 0 gate for gpuwm/core/kernels/noahmp_vegeflux.cu against the oracle.

Runs on the rented RTX 5090 box.  Never run this on the developer machine --
this project's standing rule is that no GPU work touches it.

  ssh -p <port> root@<host>
  source /venv/main/bin/activate
  cd <worktree> && python tools/noahmp_wrf461_oracle/validate_vegeflux_cuda.py

Every constant table is uploaded from the host into ``__constant__`` memory
rather than compiled in as a literal array: ptxas 12.8's constant folder does
not honour round-to-nearest-even, so a literal FP table can have its
differences mis-folded at compile time, and ``__fsub_rn`` pins the hardware
rounding mode, not the compiler's folder.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

FIXTURE = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-vegeflux.csv"
KERNEL = REPO / "gpuwm" / "core" / "kernels" / "noahmp_vegeflux.cu"

NSNOW, NSOIL = 3, 4
VF_NLAYER = NSNOW + NSOIL          # indices -NSNOW+1 .. NSOIL == 7
VF_LOFF = NSNOW - 1                # layer k lives at slot k + VF_LOFF

# Slot orders; these mirror the enums in noahmp_vegeflux.cu exactly.
RAGRB_IN = ["VAI", "RHOAIR", "HG", "TAH", "ZPD", "Z0MG", "Z0HG", "HCAN", "UC",
            "Z0H", "FV", "CWP", "MPE", "TV", "MOZG", "FHG", "P_DLEAF"]
RAGRB_OUT = ["MOZG", "FHG", "RAMG", "RAHG", "RAWG", "RB"]

SFCDIF1_IN = ["SFCTMP", "RHOAIR", "H", "QAIR", "ZLVL", "ZPD", "Z0M", "Z0H",
              "UR", "MPE", "MOZ", "FM", "FH", "FM2", "FH2", "FV"]
SFCDIF1_OUT = ["MOZ", "FM", "FH", "FM2", "FH2", "CM", "CH", "FV", "CH2"]

STOMATA_IN = ["MPE", "APAR", "FOLN", "TV", "EI", "EA", "SFCTMP", "SFCPRS",
              "FVEG", "O2", "CO2", "IGS", "BTRAN", "RB"]
STOMATA_OUT = ["RS", "PSN"]

PARAM_SLOTS = ["DLEAF", "HVT", "CBIOM", "C3PSN", "KC25", "AKC", "KO25", "AKO",
               "AVCMX", "VCMX25", "BP", "MP", "QE25", "FOLNMX"]

VF_IN = ["DT", "SAV", "SAG", "LWDN", "UR", "UU", "VV", "SFCTMP", "QAIR",
         "EAIR", "RHOAIR", "SNOWH", "VAI", "GAMMAV", "GAMMAG", "FWET",
         "LAISUN", "LAISHA", "CWP", "ZLVL", "ZPD", "Z0M", "FVEG", "Z0MG",
         "EMV", "EMG", "CANLIQ", "CANICE", "RSURF", "LATHEAV", "PARSUN",
         "PARSHA", "IGS", "FOLN", "CO2AIR", "O2AIR", "BTRAN", "SFCPRS",
         "RHSUR", "PAHV", "PAHG", "EAH", "TAH", "TV", "TG", "CM", "CH",
         "QSFC", "PSFC", "FSR"]
VF_OUT = ["EAH", "TAH", "TV", "TG", "CM", "CH", "TAUXV", "TAUYV", "IRG", "IRC",
          "SHG", "SHC", "EVG", "EVC", "TR", "GH", "T2MV", "PSNSUN", "PSNSHA",
          "CANHS", "QSFC", "Q2V", "CAH2", "CHLEAF", "CHUC", "RSSUN", "RSSHA",
          "SAV", "SAG", "FSR"]


def d(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def fbits(h: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def ibits(h: str) -> int:
    v = int(h, 16)
    return v - (1 << 32) if v >> 31 else v


def load():
    table: dict = {}
    with FIXTURE.open(newline="") as fh:
        for row in csv.DictReader(fh):
            leaf = table.setdefault(row["leaf"], {})
            case = leaf.setdefault(row["case"], {"in": {}, "out": {}, "opt": {}})
            case[row["role"]][row["name"]] = row["hex"]
    return table


# --------------------------------------------------------------------------
# constant tables, byte-for-byte the same values the CPU transcription uses
# --------------------------------------------------------------------------
EXP2F_TAB = np.array([
    0x3FF0000000000000, 0x3FEFD9B0D3158574, 0x3FEFB5586CF9890F, 0x3FEF9301D0125B51,
    0x3FEF72B83C7D517B, 0x3FEF54873168B9AA, 0x3FEF387A6E756238, 0x3FEF1E9DF51FDEE1,
    0x3FEF06FE0A31B715, 0x3FEEF1A7373AA9CB, 0x3FEEDEA64C123422, 0x3FEECE086061892D,
    0x3FEEBFDAD5362A27, 0x3FEEB42B569D4F82, 0x3FEEAB07DD485429, 0x3FEEA47EB03A5585,
    0x3FEEA09E667F3BCD, 0x3FEE9F75E8EC5F74, 0x3FEEA11473EB0187, 0x3FEEA589994CCE13,
    0x3FEEACE5422AA0DB, 0x3FEEB737B0CDC5E5, 0x3FEEC49182A3F090, 0x3FEED503B23E255D,
    0x3FEEE89F995AD3AD, 0x3FEEFF76F2FB5E47, 0x3FEF199BDD85529C, 0x3FEF3720DCEF9069,
    0x3FEF5818DCFBA487, 0x3FEF7C97337B9B5F, 0x3FEFA4AFA2A490DA, 0x3FEFD0765B6E4540,
], dtype=np.uint64)

_N = 32
_SHIFT = d(0x4338000000000000)
EXP2F_POLY = np.array([d(0x3FAC6AF84B912394), d(0x3FCEBFCE50FAC4F3),
                       d(0x3FE62E42FF0C52D6)], dtype=np.float64)
EXP2F_POLY_SCALED = np.array([EXP2F_POLY[0] / _N / _N / _N,
                              EXP2F_POLY[1] / _N / _N,
                              EXP2F_POLY[2] / _N], dtype=np.float64)
EXP2F_SHIFT = np.float64(_SHIFT)
EXP2F_SHIFT_SCALED = np.float64(_SHIFT / _N)
EXP2F_INVLN2_SCALED = np.float64(d(0x3FF71547652B82FE) * _N)

LOGF_TAB = np.array([
    d(0x3FF661EC79F8F3BE), d(0xBFD57BF7808CAADE),
    d(0x3FF571ED4AAF883D), d(0xBFD2BEF0A7C06DDB),
    d(0x3FF49539F0F010B0), d(0xBFD01EAE7F513A67),
    d(0x3FF3C995B0B80385), d(0xBFCB31D8A68224E9),
    d(0x3FF30D190C8864A5), d(0xBFC6574F0AC07758),
    d(0x3FF25E227B0B8EA0), d(0xBFC1AA2BC79C8100),
    d(0x3FF1BB4A4A1A343F), d(0xBFBA4E76CE8C0E5E),
    d(0x3FF12358F08AE5BA), d(0xBFB1973C5A611CCC),
    d(0x3FF0953F419900A7), d(0xBFA252F438E10C1E),
    d(0x3FF0000000000000), d(0x0000000000000000),
    d(0x3FEE608CFD9A47AC), d(0x3FAAA5AA5DF25984),
    d(0x3FECA4B31F026AA0), d(0x3FBC5E53AA362EB4),
    d(0x3FEB2036576AFCE6), d(0x3FC526E57720DB08),
    d(0x3FE9C2D163A1AA2D), d(0x3FCBC2860D224770),
    d(0x3FE886E6037841ED), d(0x3FD1058BC8A07EE1),
    d(0x3FE767DCF5534862), d(0x3FD4043057B6EE09),
], dtype=np.float64)
LOGF_LN2 = np.float64(d(0x3FE62E42FEFA39EF))
LOGF_POLY = np.array([d(0xBFD00EA348B88334), d(0x3FD5575B0BE00B6A),
                      d(0xBFDFFFFEF20A4123)], dtype=np.float64)

POWF_TAB = np.array([
    d(0x3FF661EC79F8F3BE), d(0xBFDEFEC65B963019),
    d(0x3FF571ED4AAF883D), d(0xBFDB0B6832D4FCA4),
    d(0x3FF49539F0F010B0), d(0xBFD7418B0A1FB77B),
    d(0x3FF3C995B0B80385), d(0xBFD39DE91A6DCF7B),
    d(0x3FF30D190C8864A5), d(0xBFD01D9BF3F2B631),
    d(0x3FF25E227B0B8EA0), d(0xBFC97C1D1B3B7AF0),
    d(0x3FF1BB4A4A1A343F), d(0xBFC2F9E393AF3C9F),
    d(0x3FF12358F08AE5BA), d(0xBFB960CBBF788D5C),
    d(0x3FF0953F419900A7), d(0xBFAA6F9DB6475FCE),
    d(0x3FF0000000000000), d(0x0000000000000000),
    d(0x3FEE608CFD9A47AC), d(0x3FB338CA9F24F53D),
    d(0x3FECA4B31F026AA0), d(0x3FC476A9543891BA),
    d(0x3FEB2036576AFCE6), d(0x3FCE840B4AC4E4D2),
    d(0x3FE9C2D163A1AA2D), d(0x3FD40645F0C6651C),
    d(0x3FE886E6037841ED), d(0x3FD88E9C2C1B9FF8),
    d(0x3FE767DCF5534862), d(0x3FDCE0A44EB17BCC),
], dtype=np.float64)
POWF_POLY = np.array([d(0x3FD27616C9496E0B), d(0xBFD71969A075C67A),
                      d(0x3FDEC70A6CA7BADD), d(0xBFE7154748BEF6C8),
                      d(0x3FF71547652AB82B)], dtype=np.float64)


def f32(u: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", u), dtype=np.float32)[0]


ATANHI = np.array([f32(0x3EED6338), f32(0x3F490FDA), f32(0x3F7B985E),
                   f32(0x3FC90FDA)], dtype=np.float32)
ATANLO = np.array([f32(0x31AC3769), f32(0x33222168), f32(0x33140FB4),
                   f32(0x33A22168)], dtype=np.float32)
ATAN_AT = np.array([f32(0x3EAAAAAB), f32(0xBE4CCCCD), f32(0x3E124925),
                    f32(0xBDE38E38), f32(0x3DBA2E6E), f32(0xBD9D8795),
                    f32(0x3D886B35), f32(0xBD6EF16B), f32(0x3D4BDA59),
                    f32(0xBD15A221), f32(0x3C8569D7)], dtype=np.float32)

PHYS = np.array([9.80616, 5.67E-08, 0.40, 273.16, 1004.64, 4.188E06, 2.094E06,
                 1000.0, 917.0], dtype=np.float32)
ESAT_A = np.array([6.107799961, 4.436518521E-01, 1.428945805E-02, 2.650648471E-04,
                   3.031240396E-06, 2.034080948E-08, 6.136820929E-11], dtype=np.float32)
ESAT_B = np.array([6.109177956, 5.034698970E-01, 1.886013408E-02, 4.176223716E-04,
                   5.824720280E-06, 4.838803174E-08, 1.838826904E-10], dtype=np.float32)
ESAT_C = np.array([4.438099984E-01, 2.857002636E-02, 7.938054040E-04, 1.215215065E-05,
                   1.036561403E-07, 3.532421810E-10, -7.090244804E-13], dtype=np.float32)
ESAT_D = np.array([5.030305237E-01, 3.773255020E-02, 1.267995369E-03, 2.477563108E-05,
                   3.005693132E-07, 2.158542548E-09, 7.131097725E-12], dtype=np.float32)

CONSTANTS = {
    "c_exp2f_tab": EXP2F_TAB,
    "c_exp2f_poly": EXP2F_POLY,
    "c_exp2f_poly_scaled": EXP2F_POLY_SCALED,
    "c_exp2f_shift": np.array([EXP2F_SHIFT], dtype=np.float64),
    "c_exp2f_shift_scaled": np.array([EXP2F_SHIFT_SCALED], dtype=np.float64),
    "c_exp2f_invln2_scaled": np.array([EXP2F_INVLN2_SCALED], dtype=np.float64),
    "c_logf_tab": LOGF_TAB,
    "c_logf_ln2": np.array([LOGF_LN2], dtype=np.float64),
    "c_logf_poly": LOGF_POLY,
    "c_powf_tab": POWF_TAB,
    "c_powf_poly": POWF_POLY,
    "c_atanhi": ATANHI,
    "c_atanlo": ATANLO,
    "c_atan_aT": ATAN_AT,
    "c_phys": PHYS,
    "c_esat_a": ESAT_A,
    "c_esat_b": ESAT_B,
    "c_esat_c": ESAT_C,
    "c_esat_d": ESAT_D,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None, help="e.g. compute_120")
    ap.add_argument("--nc-devlibm", action="store_true",
                    help="negative control: build with -DUSE_DEVICE_LIBM so the "
                         "leaves call CUDA's device libm instead of the glibc "
                         "transcriptions.  Must FAIL.")
    ap.add_argument("--nc-cpair", action="store_true",
                    help="negative control: upload CPAIR perturbed by one ulp. "
                         "Must FAIL.")
    args = ap.parse_args()

    import cupy as cp

    src = KERNEL.read_text()
    opts = ["-std=c++14", "--fmad=false"]
    if args.nc_devlibm:
        opts.append("-DUSE_DEVICE_LIBM")
    if args.arch:
        opts.append(f"--gpu-architecture={args.arch}")
    mod = cp.RawModule(code=src, backend="nvrtc", options=tuple(opts))

    constants = dict(CONSTANTS)
    if args.nc_cpair:
        phys = PHYS.copy()
        u = struct.unpack("<I", struct.pack("<f", float(phys[4])))[0] + 1
        phys[4] = struct.unpack("<f", struct.pack("<I", u))[0]
        constants["c_phys"] = phys

    for name, arr in constants.items():
        ptr = mod.get_global(name)
        dst = cp.ndarray(arr.shape, dtype=arr.dtype, memptr=ptr)
        dst[...] = cp.asarray(arr)

    table = load()
    failures = 0
    total = 0

    def report(leaf, tags, got, want, names):
        nonlocal failures, total
        bad = 0
        for r, tag in enumerate(tags):
            for c, nm in enumerate(names):
                if nm not in want[r]:
                    continue
                total += 1
                g = int(struct.unpack("<I", struct.pack("<f", float(got[r, c])))[0])
                w = int(want[r][nm], 16)
                if g != w:
                    bad += 1
                    if bad <= 8:
                        print(f"  MISMATCH {leaf}/{tag}.{nm}: "
                              f"got 0x{g:08x} want 0x{w:08x}")
        failures += bad
        print(f"{leaf}: {len(tags)} cases -> {bad} mismatched values")

    # ---- ESAT ----------------------------------------------------------
    tags = sorted(table["esat"])
    tin = cp.asarray([fbits(table["esat"][t]["in"]["T"]) for t in tags],
                     dtype=cp.float32)
    out = cp.empty((len(tags), 4), dtype=cp.float32)
    mod.get_function("k_esat")((len(tags),), (1,), (tin, out, len(tags)))
    report("esat", tags, cp.asnumpy(out), [table["esat"][t]["out"] for t in tags],
           ["ESW", "ESI", "DESW", "DESI"])

    # ---- RAGRB ---------------------------------------------------------
    tags = sorted(table["ragrb"])
    a = np.array([[fbits(table["ragrb"][t]["in"][n]) for n in RAGRB_IN]
                  for t in tags], dtype=np.float32)
    it = np.array([ibits(table["ragrb"][t]["in"]["ITER"]) for t in tags],
                  dtype=np.int32)
    out = cp.empty((len(tags), len(RAGRB_OUT)), dtype=cp.float32)
    mod.get_function("k_ragrb")((len(tags),), (1,),
                                (cp.asarray(a), cp.asarray(it), out, len(tags)))
    report("ragrb", tags, cp.asnumpy(out),
           [table["ragrb"][t]["out"] for t in tags], RAGRB_OUT)

    # ---- SFCDIF1 -------------------------------------------------------
    tags = sorted(table["sfcdif1"])
    a = np.array([[fbits(table["sfcdif1"][t]["in"][n]) for n in SFCDIF1_IN]
                  for t in tags], dtype=np.float32)
    ii = np.array([[ibits(table["sfcdif1"][t]["in"]["ITER"]),
                    ibits(table["sfcdif1"][t]["in"]["MOZSGN"])] for t in tags],
                  dtype=np.int32)
    out = cp.empty((len(tags), len(SFCDIF1_OUT)), dtype=cp.float32)
    iout = cp.empty(len(tags), dtype=cp.int32)
    mod.get_function("k_sfcdif1")((len(tags),), (1,),
                                  (cp.asarray(a), cp.asarray(ii), out, iout,
                                   len(tags)))
    report("sfcdif1", tags, cp.asnumpy(out),
           [table["sfcdif1"][t]["out"] for t in tags], SFCDIF1_OUT)
    got_i = cp.asnumpy(iout)
    for r, t in enumerate(tags):
        want = ibits(table["sfcdif1"][t]["out"]["MOZSGN"])
        total += 1
        if int(got_i[r]) != want:
            failures += 1
            print(f"  MISMATCH sfcdif1/{t}.MOZSGN: got {got_i[r]} want {want}")

    # ---- STOMATA -------------------------------------------------------
    tags = sorted(table["stomata"])
    a = np.array([[fbits(table["stomata"][t]["in"][n]) for n in STOMATA_IN]
                  for t in tags], dtype=np.float32)
    pp = np.zeros((len(tags), len(PARAM_SLOTS)), dtype=np.float32)
    for r, t in enumerate(tags):
        for c, nm in enumerate(PARAM_SLOTS):
            key = "P_" + nm
            if key in table["stomata"][t]["in"]:
                pp[r, c] = fbits(table["stomata"][t]["in"][key])
    out = cp.empty((len(tags), 2), dtype=cp.float32)
    mod.get_function("k_stomata")((len(tags),), (1,),
                                  (cp.asarray(a), cp.asarray(pp), out, len(tags)))
    report("stomata", tags, cp.asnumpy(out),
           [table["stomata"][t]["out"] for t in tags], STOMATA_OUT)

    # ---- VEGE_FLUX -----------------------------------------------------
    tags = sorted(table["vegeflux"])
    a = np.array([[fbits(table["vegeflux"][t]["in"][n]) for n in VF_IN]
                  for t in tags], dtype=np.float32)
    pp = np.zeros((len(tags), len(PARAM_SLOTS)), dtype=np.float32)
    for r, t in enumerate(tags):
        for c, nm in enumerate(PARAM_SLOTS):
            pp[r, c] = fbits(table["vegeflux"][t]["in"]["P_" + nm])
    isnow = np.array([ibits(table["vegeflux"][t]["in"]["ISNOW"]) for t in tags],
                     dtype=np.int32)
    dz = np.zeros((len(tags), VF_NLAYER), dtype=np.float32)
    stc = np.zeros_like(dz)
    dfc = np.zeros_like(dz)
    for r, t in enumerate(tags):
        for k in range(-NSNOW + 1, NSOIL + 1):
            dz[r, k + VF_LOFF] = fbits(table["vegeflux"][t]["in"][f"DZSNSO_{k:+d}"])
            stc[r, k + VF_LOFF] = fbits(table["vegeflux"][t]["in"][f"STC_{k:+d}"])
            dfc[r, k + VF_LOFF] = fbits(table["vegeflux"][t]["in"][f"DF_{k:+d}"])
    out = cp.empty((len(tags), len(VF_OUT)), dtype=cp.float32)
    mod.get_function("k_vegeflux")((len(tags),), (1,),
                                   (cp.asarray(a), cp.asarray(pp),
                                    cp.asarray(isnow), cp.asarray(dz),
                                    cp.asarray(stc), cp.asarray(dfc), out,
                                    len(tags)))
    report("vegeflux", tags, cp.asnumpy(out),
           [table["vegeflux"][t]["out"] for t in tags], VF_OUT)

    print(f"\nvalues compared: {total}")
    print("RESULT:", "PASS (max_ulp 0)" if failures == 0
          else f"FAIL ({failures} mismatched values)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
