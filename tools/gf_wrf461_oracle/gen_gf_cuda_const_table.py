"""Emit the GF deep-kernel __constant__ float table.

Every word is computed from the same expression the CPU reference
(gpuwm/verify/gf_deep_ref.py / gf_deep_body.py / gf_ref.py) uses, so the
table IS the reference's constant set, not a retyping of it.  The paired
CUDA gate re-derives these words from the reference at test time and
compares them against a device dump of the table, which closes the
transcription-error class mechanically.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from gpuwm.verify import gf_deep_ref as R

F = np.float32


def w(v):
    return int(np.float32(v).view(np.uint32))


# name -> (value, provenance comment)
C = [
    # --- plain structural values -------------------------------------------
    ("ZERO", F(0.0), "0."),
    ("ONE", F(1.0), "1."),
    ("TWO", F(2.0), "2."),
    ("HALF", F(0.5), ".5"),
    ("NEG_HALF", F(-0.5), "-.5"),
    # --- the deep module's own parameters (module_cu_gf_deep.F:6-33) -------
    ("G", R.G, "g = 9.81"),
    ("CP", R.CP, "cp = 1004."),
    ("XLV", R.XLV, "xlv = 2.5e6"),
    ("RV", R.R_V, "r_v = 461."),
    ("C1", R.C1, "c1 = .001"),
    ("FRH_THRESH", R.FRH_THRESH, "frh_thresh = .9"),
    ("RH_THRESH", R.RH_THRESH, "rh_thresh = .97"),
    ("SIG_THRESH", F(F(F(1.0) - R.FRH_THRESH) * F(F(1.0) - R.FRH_THRESH)),
     "(1.-frh_thresh)**2, folded: constant-constant FP32 arithmetic must not reach ptxas"),
    ("BETAJB", R.BETAJB, "betajb = 1.5"),
    ("FLUXTUNE", R.FLUXTUNE, "fluxtune = 1.5"),
    ("PGCD", R.PGCD, "pgcd = 1."),
    ("TINY32", R._TINY32, "tiny(zws)"),
    # --- satvap (module_cu_gf_deep.F:3646-3668) ----------------------------
    ("T273_155", F(273.155), "273.155"),
    ("NEG20", F(-20.0), "-20."),
    ("T273_16", F(273.16), "273.16"),
    ("T373_16", F(373.16), "373.16"),
    ("SAT_A", F(-9.09718), "-9.09718"),
    ("SAT_B", F(3.56654), "3.56654"),
    ("SAT_C", F(0.876793), ".876793"),
    ("SAT_D", F(-7.90298), "-7.90298"),
    ("SAT_E", F(5.02808), "5.02808"),
    ("SAT_F", F(1.3816e-07), "1.3816E-07"),
    ("SAT_G", F(11.344), "11.344"),
    ("SAT_H", F(0.0081328), ".0081328"),
    ("SAT_I", F(-3.49149), "-3.49149"),
    ("TEN", F(10.0), "10."),
    ("LOG10", R._LOG10, "log(10.), front-end fold"),
    ("LOG6_OVER_LOG10", R._LOG6_OVER_LOG10, "log(6.1071)/log(10.), fold"),
    ("LOG1013_OVER_LOG10", R._LOG1013_OVER_LOG10, "log(1013.246)/log(10.), fold"),
    # --- cup_env / cup_env_clev --------------------------------------------
    ("P622", F(0.622), ".622"),
    ("E1M8", F(1.0e-8), "1.e-8"),
    ("E1M16", F(1.0e-16), "1.e-16"),
    ("LIT_G", F(9.81), "bare 9.81 literal in cup_env"),
    ("LIT_CP", F(1004.0), "bare 1004. literal in cup_env"),
    ("LIT_XLV", F(2.5e06), "bare 2.5e06 literal in cup_env"),
    ("XLV_OVER_CP", F(R.XLV / R.CP), "xlv/cp, front-end fold"),
    ("ONE_OVER_XLV", F(F(1.0) / R.XLV), "1./xlv, front-end fold"),
    # --- the body's trigger chain ------------------------------------------
    ("P608", F(0.608), ".608"),
    ("P41", F(0.41), ".41"),
    ("ONE_P2", F(1.2), "1.2"),
    ("P3333", F(0.3333), ".3333 (NOT 1/3)"),
    ("CAP_MAXS", F(75.0), "cap_maxs = 75."),
    ("TWENTY", F(20.0), "20."),
    ("TWENTYFIVE", F(25.0), "25."),
    ("SIXTEEN", F(16.0), "16."),
    ("ONE_P5", F(1.5), "1.5"),
    ("P0001", F(0.0001), ".0001 xland1 round"),
    ("ENTR_BASE", F(7.0e-5), "7.e-5"),
    ("ENTR_CSUM", F(3.0e-6), "3.e-6"),
    ("P2", F(0.2), ".2"),
    ("PI314", F(3.14), "3.14"),
    ("ZKBMAX", F(4000.0), "zkbmax = 4000."),
    ("Z_DETR", F(1000.0), "z_detr = 1000."),
    ("DEPTH_MIN", F(1000.0), "depth_min = 1000."),
    ("P1", F(0.1), ".1"),
    ("CAP_INC_DEEP", F(20.0), "cap_max_increment = 20."),
    ("P150", F(150.0), "150."),
    ("P200", F(200.0), "200."),
    ("ONE_P3", F(1.3), "1.3"),
    ("E1M9", F(1.0e-9), "1.e-9"),
    ("E1M6", F(1.0e-6), "1.e-6"),
    ("P9", F(0.9), ".9"),
    ("P4", F(0.4), ".4"),
    ("P013", F(0.013), ".013"),
    ("P8", F(0.8), ".8"),
    ("TWO_P5", F(2.5), "2.5"),
    ("FOUR", F(4.0), "4."),
    ("LAMBAU", F(2.0), "lambau = 2."),
    ("ZCUTDOWN", F(4000.0), "zcutdown = 4000."),
    ("P6", F(0.6), ".6"),
    ("FIVE", F(5.0), "5. (jmini floor is int)"),
    ("BETA_DN_A", F(0.05), ".05"),
    ("BETA_DN_B", F(0.0015), ".0015"),
    ("BETA_DN_MIN", F(0.02), ".02"),
    ("EDTMAX_LAND_A", F(0.4), ".4"),
    ("EDTMAX_LAND_B", F(0.015), ".015"),
    ("C0_UP", F(0.004), "c0 = .004"),
    ("T273_15", F(273.15), "273.15"),
    ("P07", F(0.07), ".07"),
    ("WMEAN", F(7.0), "wmean = 7."),
    ("TAU_A", F(1.0061), "1.0061"),
    ("TAU_B", F(1.23e-2), "1.23e-2"),
    ("THOUSAND", F(1000.0), "1000."),
    ("T_STAR", F(4.0), "t_star = 4."),
    ("MBDT", F(0.1), "mbdt = .1"),
    ("T190", F(190.0), "190."),
    ("HUNDRED", F(100.0), "100."),
    # --- cup_dd_edt --------------------------------------------------------
    ("E1P3", F(1.0e3), "1.e3"),
    ("PEF_A", F(1.591), "1.591"),
    ("PEF_B", F(0.639), ".639"),
    ("PEF_C", F(0.0953), ".0953"),
    ("PEF_D", F(0.00496), ".00496"),
    ("ZKBC_SCALE", F(3.281e-3), "3.281e-3"),
    ("PREZK0", F(0.02), ".02"),
    ("THREE", F(3.0), "3."),
    ("PZ_A", F(0.96729352), ".96729352"),
    ("PZ_B", F(-0.70034167), "-.70034167"),
    ("PZ_C", F(0.162179896), ".162179896"),
    ("PZ_D", F(-1.2569798e-2), "-1.2569798E-2"),
    ("PZ_E", F(4.2772e-4), "4.2772E-4"),
    ("PZ_F", F(-5.44e-6), "-5.44E-6"),
    ("PREZK25", F(2.4), "2.4"),
    ("EDT_EINC", F(0.2), ".2 einc factor"),
    ("EDTMIN", F(0.1), "edtmin = .1"),
    # --- cup_forcing_ens_3d ------------------------------------------------
    ("NEG_P01", F(-0.01), "-.01"),
    ("E1M2", F(1.0e-2), "1.e-2"),
    ("E1M5", F(1.0e-5), "1.e-5"),
    ("E1M3", F(1.0e-3), "1.e-3"),
    ("TWELVE", F(12.0), "closure_n = 12."),
    # --- neg_check ---------------------------------------------------------
    ("THRESH_DEEP", F(300.01), "300.01"),
    ("SECONDS_DAY", F(86400.0), "86400."),
    # --- get_inversion_layers ---------------------------------------------
    ("BIG1E9", F(1.0e9), "1.e9"),
    ("P100", F(100.0), "100. (800 hPa slot)"),
    ("P300", F(300.0), "300. (550 hPa slot)"),
    # --- gf_driver_prep-side (for later phases, harmless here) -------------
    ("QMIN", F(1.0e-08), "1.e-08 moisture floor"),
    ("TMIN", F(200.0), "200. temperature floor"),
    ("MB", F(0.01), "mb conversion"),
    ("CCN150", F(150.0), "ccn = 150."),
]

seen = {}
for name, val, _ in C:
    if name in seen:
        raise SystemExit(f"dup {name}")
    seen[name] = val

print("// Generated by tools/gf_wrf461_oracle/gen_gf_const_table.py from the")
print("// CPU reference's own constant set; the CUDA gate re-derives every")
print("// word from gpuwm.verify.gf_deep_ref at test time and compares it")
print("// against gf_deep_const_dump's output, so a wrong word here cannot")
print("// survive the suite.")
print(f"#define GF_NCONST {len(C)}")
print("__constant__ unsigned int GFC[GF_NCONST] = {")
for i, (name, val, com) in enumerate(C):
    print(f"    0x{w(val):08X}u,  // [{i:3d}] {name} = {com}")
print("};")
for i, (name, val, com) in enumerate(C):
    print(f"#define K_{name} __uint_as_float(GFC[{i}])")
