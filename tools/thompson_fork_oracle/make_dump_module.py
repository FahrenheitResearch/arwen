#!/usr/bin/env python3
"""Write a copy of the HRRR v4.1.21 fork's module_mp_thompson.F with one
harness subroutine, ``fork_dump_constants``, added inside the module (before
``END MODULE``) so it can read the module's PRIVATE constants.  Nothing else
in the text changes; the subroutine only WRITEs.

It writes, one line per element, ``name index kind hexbits``, for every
derived constant thompson_init and radar_init leave in the two modules and
every PARAMETER the runtime code reads, so tools/thompson_fork_oracle/
make_constants_header.py can emit them as exact bit patterns for
gpuwm/core/kernels/thompson_fork.cu.

usage: make_dump_module.py PRISTINE.F OUTPUT.F
"""

from __future__ import annotations

import sys
from pathlib import Path

REAL_SCALARS = (
    # thompson_init outputs
    "mu_c", "Sc3", "D0i", "xm0s", "xm0g",
    "oig1", "oig2", "obmi", "ore1", "org1", "org2", "org3", "obmr",
    "oams", "obms", "ocms", "oge1", "ogg1", "ogg2", "ogg3", "oamg", "obmg",
    "ocmg",
    "t1_qr_qc", "t1_qr_qi", "t2_qr_qi", "t1_qg_qc", "t1_qs_qc", "t1_qs_qi",
    "t1_qr_ev", "t2_qr_ev", "t1_qs_sd", "t2_qs_sd", "t1_qg_sd", "t2_qg_sd",
    "t1_qs_me", "t2_qs_me", "t1_qg_me", "t2_qg_me",
    # PARAMETERs the runtime reads (their folded values)
    "T_0", "PI", "rho_w", "rho_s", "rho_g", "rho_i", "Nt_c", "Nt_c_max",
    "naIN0", "naIN1", "naCCN0", "naCCN1", "mu_r", "mu_g", "mu_i", "mu_s",
    "Kap0", "Kap1", "Lam0", "Lam1", "gonv_min", "gonv_max",
    "am_r", "bm_r", "am_s", "bm_s", "am_g", "bm_g", "am_i", "bm_i",
    "av_r", "bv_r", "fv_r", "av_s", "bv_s", "fv_s", "av_g", "bv_g",
    "av_i", "bv_i", "av_c", "bv_c", "C_cube", "C_sqrd",
    "Ef_si", "Ef_rs", "Ef_rg", "Ef_ri", "R1", "R2", "eps", "TNO", "ATO",
    "rho_not", "Sc", "HGFR", "Rv", "oRv", "R", "Cp", "R_uni", "ar_volume",
    "lsub", "lvap0", "lfus", "olfus", "xm0i", "D0c", "D0r", "D0s", "D0g",
    # radar module
    "xam_r", "xbm_r", "xmu_r", "xobmr", "xam_s", "xbm_s", "xmu_s", "xoams",
    "xobms", "xocms", "xam_g", "xbm_g", "xmu_g", "xoamg", "xobmg", "xocmg",
    "xorg2", "xosg2", "xogg2",
)
REAL_ARRAYS = (
    "cce", "ccg", "ocg1", "ocg2", "cie", "cig", "cre", "crg", "cse", "csg",
    "cge", "cgg", "r_c", "r_i", "r_r", "r_g", "r_s", "N0r_exp", "N0g_exp",
    "Nt_i", "ta_Na", "ta_Ww", "ta_Tk", "ta_Ra", "ta_Ka", "Nt_IN", "sa", "sb",
    "Tc", "xcre", "xcse", "xcge", "xcrg", "xcsg", "xcgg",
)
DOUBLE_SCALARS = ("k_b", "M_w", "M_a", "N_avo", "ma_w", "K_w", "PI5",
                  "lamda4", "lamda_radar", "melt_outside_s",
                  "melt_outside_g")
DOUBLE_ARRAYS = ("xDx", "Dc", "dtc", "Di", "dti", "Dr", "dtr", "Ds", "dts",
                 "Dg", "dtg", "t_Nc", "xxDx", "xxDs", "xdts", "xxDg",
                 "xdtg", "simpson")
INT_SCALARS = ("nic1", "nic2", "nii2", "nii3", "nir2", "nir3", "nis2",
               "nig2", "nig3", "niIN2", "nbins", "nbc", "nbi", "nbr", "nbs",
               "nbg", "ntb_c", "ntb_i", "ntb_r", "ntb_s", "ntb_g", "ntb_g1",
               "ntb_r1", "ntb_i1", "ntb_t", "ntb_arc", "ntb_arw", "ntb_art",
               "ntb_arr", "ntb_ark", "ntb_IN", "nrbins", "IFDRY")
COMPLEX_SCALARS = ("m_w_0", "m_i_0")


def subroutine() -> str:
    lines = [
        "!+---+ harness addition (tools/thompson_fork_oracle/"
        "make_dump_module.py): WRITE only.",
        "      subroutine fork_dump_constants(u)",
        "      implicit none",
        "      integer, intent(in):: u",
    ]
    for n in REAL_SCALARS:
        lines.append(f"      call fdr(u, '{n}', (/ REAL({n}) /))")
    for n in REAL_ARRAYS:
        lines.append(f"      call fdr(u, '{n}', RESHAPE(REAL({n}), "
                     f"(/ SIZE({n}) /)))")
    for n in DOUBLE_SCALARS:
        lines.append(f"      call fdd(u, '{n}', (/ DBLE({n}) /))")
    for n in DOUBLE_ARRAYS:
        lines.append(f"      call fdd(u, '{n}', RESHAPE({n}, "
                     f"(/ SIZE({n}) /)))")
    for n in INT_SCALARS:
        lines.append(f"      write(u, '(A,1X,I0,1X,A,1X,I0)') '{n}', 1, "
                     f"'i', {n}")
    for n in COMPLEX_SCALARS:
        lines.append(f"      call fdd(u, '{n}', (/ DBLE({n}), "
                     f"DIMAG({n}) /))")
    lines += [
        "      contains",
        "      subroutine fdr(u, name, a)",
        "      integer, intent(in):: u",
        "      character(len=*), intent(in):: name",
        "      real, intent(in):: a(:)",
        "      integer:: i",
        "      do i = 1, SIZE(a)",
        "         write(u, '(A,1X,I0,1X,A,1X,Z8.8)') name, i, 'f', "
        "TRANSFER(a(i), 0)",
        "      enddo",
        "      end subroutine fdr",
        "      subroutine fdd(u, name, a)",
        "      integer, intent(in):: u",
        "      character(len=*), intent(in):: name",
        "      double precision, intent(in):: a(:)",
        "      integer:: i",
        "      do i = 1, SIZE(a)",
        "         write(u, '(A,1X,I0,1X,A,1X,Z16.16)') name, i, 'd', "
        "TRANSFER(a(i), 0_8)",
        "      enddo",
        "      end subroutine fdd",
        "      end subroutine fork_dump_constants",
        "",
    ]
    return "\n".join(lines)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    text = Path(argv[1]).read_text(encoding="latin-1")
    anchor = "END MODULE module_mp_thompson\n"
    if text.count(anchor) != 1:
        raise SystemExit("END MODULE anchor not found exactly once")
    text = text.replace(anchor, subroutine() + anchor)
    Path(argv[2]).write_text(text, encoding="latin-1", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
