"""Reuse only the existing permissively licensed device libm implementations.

These slices contain Arm MIT and Sun fdlibm math, not Noah-MP physics. Their
notices are in gpuwm/core/kernels/LICENSE-third-party.txt and the root NOTICE.
The local CORE-MATH log10.cu retains its MIT notice and matches the newer
referee's correctly rounded log10f rather than the old Sun implementation.
"""
from pathlib import Path


def preamble(numerics):
    if numerics == "fast":
        return "\n".join(f"#define OWP_{a} {b}" for a,b in
                         (("POW","powf"),("EXP","expf"),("LOG","logf"),
                          ("LOG10","log10f"),("TANH","tanhf"),("ATAN","atanf"),
                          ("SIN","sinf"),("COS","cosf"),("ASIN","asinf")))+"\n"
    root = Path(__file__).resolve().parents[2]/"core/kernels"
    leaves = (root/"noahmp_leaves.cu").read_text()
    leaves = leaves[leaves.index("#define AD"):leaves.index("// module_sf_noahmplsm.F:204-220")]
    energy = (root/"noahmp_energy.cu").read_text()
    energy = energy[energy.index("__constant__ float NMPE_EXPM1F_Q"):energy.index("// Flat slot layout.")]
    log10 = Path(__file__).with_name("log10.cu").read_text()
    trig = (root/"glibc_trig_flt32.cuh").read_text()
    return "typedef float real;\n"+leaves+energy+log10+trig+"\n"+"\n".join(
        f"#define OWP_{a} {b}" for a,b in (("POW","r_pow"),("EXP","r_exp"),
            ("LOG","r_log"),("LOG10","owp_log10"),("TANH","nmpe_tanhf"),("ATAN","glibc_atanf"),
            ("SIN","glibc_sinf"),("COS","glibc_cosf"),("ASIN","glibc_asinf")))+"\n"
