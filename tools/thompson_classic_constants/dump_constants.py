#!/usr/bin/env python3
"""Write a copy of WRF v4.6.1 module_mp_thompson.F with one appended
module procedure that prints every thompson_init constant classic
Thompson (mp_physics=8) reads, as exact hex words.

Only a CONTAINS-procedure is appended before END MODULE (module procedures
can read the module's PRIVATE state); no physics line changes.  The CUDA
unit thompson_classic.cu takes these words through
thompson_classic_constants.cuh (make_header.py), so no init constant is
re-derived on the GPU.

usage: dump_constants.py PRISTINE.F OUTPUT.F
"""
import sys

REAL_ARRAYS = ["cce", "ccg", "ocg1", "ocg2", "cie", "cig", "cre", "crg",
               "cse", "csg", "cge", "cgg", "oamg", "ocmg", "av_g", "bv_g"]
REAL_SCALARS = ["oig1", "oig2", "obmi", "ore1", "org1", "org2", "org3",
                "obmr", "oams", "obms", "ocms", "oge1", "ogg1", "ogg2",
                "ogg3", "obmg", "t1_qr_qc", "t1_qr_qi", "t2_qr_qi",
                "t1_qs_qc", "t1_qs_qi", "t1_qr_ev", "t2_qr_ev", "t1_qs_sd",
                "t2_qs_sd", "t1_qg_sd", "t1_qs_me", "t2_qs_me", "t1_qg_me",
                "mu_c", "Sc3", "D0i", "xm0s", "xm0g"]
DOUBLE_ARRAYS = ["Dc", "Di", "Dr", "Ds", "Dg", "t_Nc"]
INTS = ["nic1", "nic2", "nii2", "nii3", "nir2", "nir3", "nis2", "nig2",
        "nig3", "niIN2"]


def proc():
    lines = ["      subroutine arwen_dump_classic_constants()",
             "      implicit none", "      integer:: i_"]
    for n in REAL_ARRAYS:
        lines += [f"      do i_ = 1, size({n})",
                  f"         call arwen_wr('{n}', i_, reshape({n}, "
                  f"(/size({n})/)))",
                  "      enddo"]
    for n in REAL_SCALARS:
        lines.append(f"      write(*,'(A,1X,I0,1X,Z8.8)') 'R {n}', 1, "
                     f"transfer({n}, 0)")
    for n in DOUBLE_ARRAYS:
        lines += [f"      do i_ = 1, size({n})",
                  f"         write(*,'(A,1X,I0,1X,Z16.16)') 'D {n}', i_, "
                  f"transfer({n}(i_), 0_8)",
                  "      enddo"]
    for n in INTS:
        lines.append(f"      write(*,'(A,1X,I0,1X,I0)') 'I {n}', 1, {n}")
    lines += ["      contains",
              "      subroutine arwen_wr(name, i, a)",
              "      character(len=*), intent(in):: name",
              "      integer, intent(in):: i",
              "      real, intent(in):: a(:)",
              "      write(*,'(A,1X,A,1X,I0,1X,Z8.8)') 'R', name, i, "
              "transfer(a(i), 0)",
              "      end subroutine arwen_wr",
              "      end subroutine arwen_dump_classic_constants", ""]
    return "\n".join(lines)


def main(argv):
    src = open(argv[1], encoding="latin-1").read()
    anchor = "END MODULE module_mp_thompson"
    if src.count(anchor) != 1:
        raise SystemExit("END MODULE anchor not unique")
    out = src.replace(anchor, proc() + anchor)
    open(argv[2], "w", encoding="latin-1", newline="\n").write(out)


if __name__ == "__main__":
    main(sys.argv)
