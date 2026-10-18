"""Column-oracle harness sources for SPP, derived from the established drivers.

Two Fortran programs are generated from harnesses already in the tree, by
exact-once anchor replacement (a moved anchor fails the build rather than
silently producing a different program):

* ``run_mynn_spp_columns.F90`` from
  ``tools/mynn_pbl_wrf461_oracle/run_driver_families.F90``: WRF v4.6.1's own
  ``mynn_bl_driver`` with ``spp_pbl = 1`` over 72 columns (six column
  families x three surface elevations x four pattern shapes), integrated for
  several 20 s steps with the families harness's between-step update.
* ``run_gf_spp_columns.F90`` from ``tools/gf_wrf461_oracle/run_cup_gf.F90``:
  GFDRV's column preparation, ``CUP_gf_sh`` and ``CUP_gf`` over the 216 GF
  fixture columns, with a different closure pattern on every column and
  member (216 distinct four-member patterns).  The pattern is clipped to
  [-1, 1] by GFDRV's own statements
  (module_cu_gf_wrfdrv.F:310-318) before it reaches ``rand_clos``; stock
  GFDRV and the stock cumulus driver never pass it there (see README.md),
  so the direct ``CUP_gf`` interface is the only WRF code that consumes it.

The reference physics modules are never edited.  Only the column drivers
change, and only where named below.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The SPP receipt was generated from the families driver as the exact chain
# carried it (8fa536308c5); the 2.9.1 line has since added the MYNN TKE-advection
# columns to tools/mynn_pbl_wrf461_oracle/run_driver_families.F90, so the SPP
# harness keeps that copy and its generated sources stay the receipted bytes.
MYNN_BASE = Path(__file__).resolve().parent / "mynn_driver_families_base.F90"
GF_BASE = ROOT / "tools" / "gf_wrf461_oracle" / "run_cup_gf.F90"

#: Six families x twelve variants.  Variant v: terrain = v % 3 (0 sea-level
#: pressure, 1 about 1.4 km, 2 about 3 km on a deep stretched grid whose top
#: passes 8 km so the diffusivity taper exp(-(zw-8000)/2000) is live), and
#: pattern = v // 3 (0 zero, 1 smooth +-0.3 = the WRF default amplitude,
#: 2 an edge table, 3 alternating +-0.9).
NFAM, NVAR = 6, 12
NCOL_MYNN = NFAM * NVAR
FAMILIES = ("convective_day", "stable_night", "stratocumulus",
            "shallow_cumulus", "cold_pool", "high_wind")
TERRAIN = ("sea_level", "elevated_1400m", "high_terrain_3000m")
PATTERN = ("zero", "smooth_default_amplitude", "edge_table", "alternating_0p9")

#: Per-level edge values for pattern 2: both signs of WRF's realistic range
#: (gridpt_stddev_spp_pbl 0.15 x cutoff 2.0 = 0.3), the -2/3 point where
#: 1 + 1.5 r crosses zero in the diffusivity, +-0.9, +-1 and both zeros.
MYNN_EDGE = ("-1.0", "-0.9", "-0.6666667", "-0.3", "-0.0", "0.0", "0.3",
             "0.6666667", "1.0")

#: Closure pattern table for GF: values beyond +-1 (GFDRV clips them), the
#: clip points themselves, one float32 step inside them, both zeros and the
#: realistic range (gridpt_stddev_spp_conv 0.3 x cutoff 3.0 = 0.9).
GF_TABLE = ("-1.7", "-1.0", "-0.99999994", "-0.9", "-0.75", "-0.5", "-0.3",
            "-0.1", "-0.0", "0.0", "0.05", "0.1", "0.2", "0.3", "0.45",
            "0.6", "0.75", "0.9", "0.99999994", "1.0", "1.0000001", "1.5",
            "2.5")


#: Multipliers of (case, dx index, shallow arm) for each closure member.  The
#: table length, 23, is prime, so these give 216 distinct four-member
#: patterns over the 216 columns in either pattern set.
GF_MULTIPLIERS = ((7, 3, 11), (5, 13, 2), (3, 17, 19), (11, 7, 5))


def _once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"SPP column harness anchor moved: {old[:70]!r}")
    return text.replace(old, new, 1)


def _all(text: str, old: str, new: str, count: int) -> str:
    if text.count(old) != count:
        raise ValueError(f"SPP column harness anchor count changed: {old!r}")
    return text.replace(old, new)


def mynn_source() -> str:
    text = MYNN_BASE.read_text(encoding="ascii")
    text = _all(text, "program run_mynn_bl_driver_families_oracle",
                "program run_mynn_spp_columns_oracle", 2)
    text = _once(text, "  integer, parameter :: ncol = 6, nz = 30",
                 f"  integer, parameter :: nfam = {NFAM}, nvar = {NVAR}\n"
                 "  integer, parameter :: ncol = nfam * nvar, nz = 30")
    text = _once(text, "  character(len=32), parameter :: names(ncol) =",
                 "  character(len=32), parameter :: names(nfam) =")
    edge = ", ".join(MYNN_EDGE)
    text = _once(text, "  integer :: c, k, s, unit\n",
                 "  integer :: c, k, s, unit\n"
                 "  integer :: fam, var, terr, pat\n"
                 "  character(len=48) :: label(ncol)\n"
                 f"  real, parameter :: spp_edge({len(MYNN_EDGE)}) = [{edge}]\n"
                 "  real :: ps_sea\n")
    text = _once(text, "  integer, parameter :: bl_mynn_mixqt = 0, spp_pbl = 0",
                 "  integer, parameter :: bl_mynn_mixqt = 0, spp_pbl = 1")
    text = _once(text, "  do c = 1, ncol\n    dz1 = 50.0\n",
                 "  do c = 1, ncol\n"
                 "    fam = mod(c - 1, nfam) + 1\n"
                 "    var = (c - 1) / nfam\n"
                 "    terr = mod(var, 3)\n"
                 "    pat = var / 3\n"
                 "    dz1 = 50.0\n")
    text = _once(text, "    select case (c)\n", "    select case (fam)\n")
    text = _once(text, "    end select\n    zw = 0.0\n",
                 "    end select\n"
                 "    ! Elevated and high-terrain copies of the family: the same\n"
                 "    ! potential-temperature profile under a lower surface pressure,\n"
                 "    ! with the skin temperature carried by the same Exner factor.\n"
                 "    ps_sea = ps(c)\n"
                 "    if (terr == 1) ps(c) = 0.85 * ps_sea\n"
                 "    if (terr == 2) then\n"
                 "      ps(c) = 0.70 * ps_sea\n"
                 "      dz1 = 150.0\n"
                 "      dzslope = 40.0\n"
                 "    end if\n"
                 "    ts(c) = ts(c) * (ps(c) / ps_sea) ** (287.0 / (7.0 * 287.0 / 2.0))\n"
                 "    write(label(c), '(A,\"_t\",I0,\"_p\",I0)') trim(names(fam)), terr, pat\n"
                 "    zw = 0.0\n")
    text = _all(text, "if (c == ", "if (fam == ", 6)
    text = _once(text, "      pattern_spp_pbl(c, k) = 0.0\n",
                 "      select case (pat)\n"
                 "      case (0)\n"
                 "        pattern_spp_pbl(c, k) = 0.0\n"
                 "      case (1)\n"
                 "        pattern_spp_pbl(c, k) = 0.3 * sin(0.31 * real(k) + 0.7 * real(c))\n"
                 "      case (2)\n"
                 f"        pattern_spp_pbl(c, k) = spp_edge(mod(c + k, {len(MYNN_EDGE)}) + 1)\n"
                 "      case default\n"
                 "        pattern_spp_pbl(c, k) = 0.9 * real(1 - 2 * mod(c + k, 2))\n"
                 "      end select\n")
    text = _once(text, "'pblh,kpbl,rmol,maxwidth,maxmf,ztop_plume,ktop_plume'",
                 "'pblh,kpbl,rmol,maxwidth,maxmf,ztop_plume,ktop_plume,rstoch'")
    text = _once(text, "            trim(names(c)), s, k, initflag",
                 "            trim(label(c)), s, k, initflag")
    text = _once(text, "        write(unit, '(\",\",I0)') ktop_plume(c)\n",
                 "        write(unit, '(\",\",I0,\",\",ES24.16E3)') ktop_plume(c), &\n"
                 "            pattern_spp_pbl(c, k)\n")
    return text


def gf_source() -> str:
    text = GF_BASE.read_text(encoding="ascii")
    text = _all(text, "program run_cup_gf_oracle", "program run_gf_spp_columns_oracle", 2)
    table = ", ".join(GF_TABLE)
    text = _once(text, "  integer :: spp0\n",
                 "  integer :: spp0, offset\n"
                 "  real :: rawp(4)\n"
                 f"  real, parameter :: spp_tab({len(GF_TABLE)}) = [{table}]\n"
                 "  ! Per-member multipliers of (case, dx index, shallow arm): each of\n"
                 "  ! the 216 columns gets a distinct four-member pattern.\n"
                 f"  integer, parameter :: mul_ic(4) = [{', '.join(str(m[0]) for m in GF_MULTIPLIERS)}]\n"
                 f"  integer, parameter :: mul_dx(4) = [{', '.join(str(m[1]) for m in GF_MULTIPLIERS)}]\n"
                 f"  integer, parameter :: mul_arm(4) = [{', '.join(str(m[2]) for m in GF_MULTIPLIERS)}]\n"
                 "  character(len=32) :: offset_arg\n")
    text = _once(text, "  call get_command_argument(3, cons_path)\n",
                 "  call get_command_argument(3, cons_path)\n"
                 "  call get_command_argument(4, offset_arg)\n"
                 "  offset = 0\n"
                 "  if (len_trim(offset_arg) > 0) read(offset_arg, *) offset\n")
    text = _once(text, "        call prepare_column(ic, dxv)\n",
                 "        call prepare_column(ic, dxv)\n"
                 "        ! A different closure pattern per column and member, clipped by\n"
                 "        ! GFDRV's own statements (module_cu_gf_wrfdrv.F:310-318).\n"
                 "        do n = 1, 4\n"
                 "          rawp(n) = spp_tab(mod(mul_ic(n) * ic + mul_dx(n) * idx &\n"
                 f"              + mul_arm(n) * iarm + offset, {len(GF_TABLE)}) + 1)\n"
                 "          rand_clos(its, n) = rawp(n)\n"
                 "          if (rawp(n) .le. -1.0) then\n"
                 "            rand_clos(its, n) = -1.0\n"
                 "          endif\n"
                 "          if (rawp(n) .ge.  1.0) then\n"
                 "            rand_clos(its, n) =  1.0\n"
                 "          endif\n"
                 "        end do\n")
    text = _once(text, "    'ierrs,kbcons,ktops,k22s,xmbs,prets,ierrc,ierrcs'",
                 "    'ierrs,kbcons,ktops,k22s,xmbs,prets,ierrc,ierrcs,spp1,spp2,spp3,spp4'")
    text = _once(text,
                 "        write(usfc, '(\",\",A,\",\",A)') trim(clean(ierrc(its))),              &\n"
                 "          trim(clean(ierrcs(its)))\n",
                 "        write(usfc, '(\",\",A,\",\",A,4(\",\",ES24.16E3))') trim(clean(ierrc(its))), &\n"
                 "          trim(clean(ierrcs(its))), rawp(1), rawp(2), rawp(3), rawp(4)\n")
    return text


def write_sources(directory: Path) -> dict[str, str]:
    directory = Path(directory)
    out = {"run_mynn_spp_columns.F90": mynn_source(),
           "run_gf_spp_columns.F90": gf_source()}
    for name, text in out.items():
        (directory / name).write_text(text, encoding="ascii")
    return out
