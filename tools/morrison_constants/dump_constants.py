#!/usr/bin/env python3
"""Write a copy of WRF v4.6.1 module_mp_morr_two_moment.F with one appended
module procedure that prints every MORR_TWO_MOMENT_INIT word the Morrison
scheme reads, as exact hex words.

Only a CONTAINS-procedure is appended before END MODULE (module procedures
can read the module's PRIVATE state); no physics line changes.  The CUDA
unit morrison.cu takes these words through morrison_constants.cuh
(make_header.py), so no init constant is re-derived on the GPU.

usage: dump_constants.py PRISTINE.F OUTPUT.F
"""
import sys

REALS = [
    "NDCNST", "AI", "AC", "AS", "AR", "AG", "BI", "BC", "BS", "BR", "BG",
    "RHOSU", "RHOW", "RHOI", "RHOSN", "RHOG", "AIMM", "BIMM", "ECR", "DCS",
    "MI0", "MG0", "F1S", "F2S", "F1R", "F2R", "QSMALL", "CI", "DI", "CS",
    "DS", "CG", "DG", "EII", "ECI", "RIN", "CPW", "C1", "K1", "MW", "OSM",
    "VI", "EPSM", "RHOA", "MAP", "MA", "RR", "BACT", "RM1", "RM2",
    "NANEW1", "NANEW2", "SIG1", "SIG2", "F11", "F12", "F21", "F22",
    "MMULT", "LAMMAXI", "LAMMINI", "LAMMAXR", "LAMMINR", "LAMMAXS",
    "LAMMINS", "LAMMAXG", "LAMMING",
] + [f"CONS{i}" for i in range(1, 42)] + [
    # module_model_constants words the module uses (CP, G, R, RV, EP_2)
    # and its own PARAMETERs PI and xxx.
    "CP", "G", "R", "RV", "EP_2", "PI", "xxx",
]
INTS = ["IACT", "INUM", "ILIQ", "INUC", "IBASE", "ISUB", "IGRAUP", "IHAIL"]


def proc():
    lines = ["      subroutine arwen_dump_morr_constants(mode)",
             "      implicit none", "      integer, intent(in):: mode"]
    for n in REALS:
        lines.append(f"      write(*,'(A,1X,I0,1X,I0,1X,Z8.8)') 'R {n}', mode, "
                     f"1, transfer({n}, 0)")
    for n in INTS:
        lines.append(f"      write(*,'(A,1X,I0,1X,I0,1X,I0)') 'I {n}', mode, 1, "
                     f"{n}")
    lines += ["      end subroutine arwen_dump_morr_constants", ""]
    return "\n".join(lines)


def main(argv):
    src = open(argv[1], encoding="latin-1").read()
    anchor = "END MODULE module_mp_morr_two_moment"
    if src.count(anchor) != 1:
        raise SystemExit("END MODULE anchor not unique")
    out = src.replace(anchor, proc() + anchor)
    # The dump procedure must be reachable from the driver program.
    pub = "   PUBLIC  ::  POLYSVP\n"
    if out.count(pub) != 1:
        raise SystemExit("PUBLIC POLYSVP anchor not unique")
    out = out.replace(pub, pub + "   PUBLIC  ::  arwen_dump_morr_constants\n")
    open(argv[2], "w", encoding="latin-1", newline="\n").write(out)


if __name__ == "__main__":
    main(sys.argv)
