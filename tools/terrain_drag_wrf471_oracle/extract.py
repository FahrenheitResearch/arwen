"""Cut WRF v4.7.1's topo_wind static coefficients out of start_em.F verbatim.

    python extract.py <WRF_SOURCE_ROOT> <OUT_DIR>

WRF computes the topo_wind drag and 10 m blend coefficients (CTOPO, CTOPO2)
inline in dyn_em/start_em.F, inside start_domain_em, which cannot be compiled
on its own.  This copies the two blocks the port transcribes into one small
Fortran module, after checking the sha256 of the whole file and of each block,
so the oracle compiles WRF's own statements and nothing else:

* start_em.F:1539-1577, the terrain loop that also forms LAP_HGT (the
  laplacian of the terrain the topo_wind = 1 hill-top test reads);
* start_em.F:1579-1626, CTOPO and CTOPO2 for topo_wind = 1 (Jimenez and
  Dudhia 2012, from VAR_SSO and LAP_HGT) and topo_wind = 2 (from VAR).

The blocks are wrapped in one subroutine whose dummy arguments carry WRF's own
names (grid%..., config_flags%...), so the statements paste unmodified.

WRF v4.7.1 is tag v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186,
public domain.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"

START_EM = ("dyn_em/start_em.F",
            "4b6c98a06fb2b2c4472af73415099e311502747dc5543d9e9b1323497d83ddb4")

#: (name, first line, last line, sha256 of the lines, LF-joined).
BLOCKS = (
    ("slope_lap", 1539, 1577,
     "17f40c3cd24391edeb4c11dbe10cb81df63e0691ac460f5e6224f8783daaf94e"),
    ("ctopo", 1579, 1626,
     "df2ad5c6828888722c9356c1a51f596c63e67012c55aec1b2ae0c27af738fabf"),
)

_MODULE = """MODULE wrf_topo_wind_inline
  IMPLICIT NONE
  ! The fields start_em.F reads off grid% and config_flags%, under the
  ! names it reads them by, so its lines paste unmodified.
  TYPE grid_t
    REAL, ALLOCATABLE :: toposlpx(:,:), toposlpy(:,:), lap_hgt(:,:)
    REAL, ALLOCATABLE :: slope(:,:), slp_azi(:,:), ht(:,:)
    REAL, ALLOCATABLE :: msftx(:,:), msfty(:,:), sina(:,:), cosa(:,:)
    REAL, ALLOCATABLE :: ctopo(:,:), ctopo2(:,:), var_sso(:,:), var2d(:,:)
    REAL, ALLOCATABLE :: xland(:,:)
    REAL :: rdx, rdy
  END TYPE grid_t
  TYPE config_t
    LOGICAL :: periodic_x = .FALSE., periodic_y = .FALSE.
    INTEGER :: topo_wind = 0
  END TYPE config_t
CONTAINS
  SUBROUTINE topo_wind_static(grid, config_flags, ids, ide, jds, jde, &
                              its, ite, jts, jte)
    TYPE(grid_t), INTENT(INOUT) :: grid
    TYPE(config_t), INTENT(IN) :: config_flags
    INTEGER, INTENT(IN) :: ids, ide, jds, jde, its, ite, jts, jte
    INTEGER :: i, j, im1, ip1, jm1, jp1
    REAL :: hx, hy, pi
    REAL :: alpha, vfac
{slope_lap}{ctopo}  END SUBROUTINE topo_wind_static
END MODULE wrf_topo_wind_inline
"""


def _block(lines: list[bytes], name: str, first: int, last: int,
           digest: str) -> str:
    text = b"\n".join(lines[first - 1:last]) + b"\n"
    got = hashlib.sha256(text).hexdigest()
    if got != digest:
        raise SystemExit(f"start_em.F:{first}-{last} ({name}) is not the "
                         f"pinned block: sha256 {got}, expected {digest}")
    return text.decode("ascii")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit(__doc__.splitlines()[2].strip())
    root, out = Path(args[0]), Path(args[1])
    path, digest = START_EM
    raw = (root / path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != digest:
        raise SystemExit(f"{path} is not WRF v4.7.1 ({WRF_COMMIT}): sha256 "
                         f"{got}, expected {digest}")
    lines = raw.split(b"\n")
    blocks = {name: _block(lines, name, first, last, sha)
              for name, first, last, sha in BLOCKS}
    out.mkdir(parents=True, exist_ok=True)
    header = (f"! Extracted verbatim from WRF v4.7.1 ({WRF_COMMIT}) by "
              "tools/terrain_drag_wrf471_oracle/extract.py.\n"
              "! Do not edit: regenerate.\n")
    (out / "wrf_topo_wind_inline.F90").write_text(
        header + _MODULE.format(**blocks), encoding="ascii", newline="\n")
    print(f"extracted {len(BLOCKS)} blocks into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
