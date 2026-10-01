"""Cut WRF v4.7.1's slope-radiation and terrain-shading code out verbatim.

    python extract.py <WRF_SOURCE_ROOT> <OUT_DIR>

WRF's slope_rad / topo_shading code lives inside three large files that
cannot be compiled on their own (module_radiation_driver.F,
module_surface_driver.F, start_em.F).  This copies the exact source lines
the port transcribes into two small Fortran modules, after checking the
sha256 of each whole file and of each block, so the oracle compiles WRF's
own statements and nothing else:

* ``wrf_topo_blocks.F90``: toposhad_init and toposhad
  (module_radiation_driver.F), TOPO_RAD_ADJ_DRVR and TOPO_RAD_ADJ
  (module_surface_driver.F), whole subroutines, unmodified.
* ``wrf_topo_inline.F90``: the two INLINE blocks, wrapped in subroutines
  whose dummy arguments carry WRF's own names so the statements are pasted
  unmodified: start_em.F:1539-1577 (slope and slope azimuth from HGT) and
  module_radiation_driver.F:2894-2914 + 2918-2927 (the Ruiz-Arias diffuse
  split for shortwave schemes without their own, and diffuse_frac).

WRF v4.7.1 is tag v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186,
public domain.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"

FILES = {
    "phys/module_radiation_driver.F":
        "779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa",
    "phys/module_surface_driver.F":
        "a653965f9ffd1db5e4c42aa79e41b4d111755b6fca0f1a3b91f3a93455813ffd",
    "dyn_em/start_em.F":
        "4b6c98a06fb2b2c4472af73415099e311502747dc5543d9e9b1323497d83ddb4",
    "share/module_model_constants.F":
        "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
    "dyn_em/nest_init_utils.F":
        "07db287108ab031f89e8bb80d830871daf8ba962f60d394e1ff31c845c1b4a14",
}

#: (name, file, first line, last line, sha256 of the lines, LF-joined).
BLOCKS = (
    ("toposhad_init", "phys/module_radiation_driver.F", 4474, 4608,
     "0c08e0cab35e28122f6dbc1d5fc0ca6266d04a07ca8153651c9018669e0c12cc"),
    ("toposhad", "phys/module_radiation_driver.F", 4610, 4862,
     "3369bd08e314eb5ca79fffa6a737dd9b954f206529748f3277b8d56f42e5e9fb"),
    ("ruiz_arias", "phys/module_radiation_driver.F", 2894, 2914,
     "2df0b9d1a654d20e262075bfb5e47bc15ca92da300c3ccd33224ae8945a9c1a3"),
    ("diffuse_frac", "phys/module_radiation_driver.F", 2918, 2927,
     "937bfb9f7a249e1d826b0398948457459b06e345cfbf2d8d2b73ecc804fd9d21"),
    ("topo_rad_adj_drvr", "phys/module_surface_driver.F", 6977, 7044,
     "fb72d2f99a8a288928adf913658afbe62ddb28cc4f0d7dd3e0edab0a574f42d8"),
    ("topo_rad_adj", "phys/module_surface_driver.F", 7047, 7114,
     "86c1ff891e45263de793d7869215f479ab7add65b99b547f7f7860570610c535"),
    ("slope", "dyn_em/start_em.F", 1539, 1577,
     "17f40c3cd24391edeb4c11dbe10cb81df63e0691ac460f5e6224f8783daaf94e"),
    # smooth_cg_topo's operator (module_initialize_real.F:743 calls it on
    # d01 with toposoil as the coarse field).
    ("blend_terrain", "dyn_em/nest_init_utils.F", 712, 785,
     "879dfc94391640b3458cda9d247ef56f819a91cb8eb625fc8dd47367c2d305e3"),
)


def _block(root: Path, name: str, path: str, first: int, last: int,
           digest: str) -> str:
    lines = (root / path).read_bytes().split(b"\n")
    text = b"\n".join(lines[first - 1:last]) + b"\n"
    got = hashlib.sha256(text).hexdigest()
    if got != digest:
        raise SystemExit(f"{path}:{first}-{last} ({name}) is not the pinned "
                         f"block: sha256 {got}, expected {digest}")
    return text.decode("ascii")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit(__doc__.splitlines()[2].strip())
    root, out = Path(args[0]), Path(args[1])
    for path, digest in FILES.items():
        got = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{path} is not WRF v4.7.1 ({WRF_COMMIT}): "
                             f"sha256 {got}, expected {digest}")
    blocks = {name: _block(root, name, path, first, last, digest)
              for name, path, first, last, digest in BLOCKS}
    out.mkdir(parents=True, exist_ok=True)
    (out / "module_model_constants.F").write_bytes(
        (root / "share/module_model_constants.F").read_bytes())
    header = (f"! Extracted verbatim from WRF v4.7.1 ({WRF_COMMIT}) by "
              "tools/wrf_topo_radiation_v471_oracle/extract.py.\n"
              "! Do not edit: regenerate.\n")
    (out / "wrf_topo_blocks.F90").write_text(
        header + "MODULE wrf_topo_blocks\nCONTAINS\n"
        + blocks["toposhad_init"] + blocks["toposhad"]
        + blocks["topo_rad_adj_drvr"] + blocks["topo_rad_adj"]
        + "END MODULE wrf_topo_blocks\n", encoding="ascii", newline="\n")
    # blend_terrain reads spec_bdy_width and blend_width through the
    # namelist getters of module_configure; the stub answers what the
    # driver sets, nothing else.
    (out / "wrf_blend_terrain.F90").write_text(
        header + _CONFIGURE_STUB + blocks["blend_terrain"],
        encoding="ascii", newline="\n")
    (out / "wrf_topo_inline.F90").write_text(
        header + _INLINE.format(slope=blocks["slope"],
                                ruiz_arias=blocks["ruiz_arias"],
                                diffuse_frac=blocks["diffuse_frac"]),
        encoding="ascii", newline="\n")
    print(f"extracted {len(BLOCKS)} blocks into {out}")
    return 0


_CONFIGURE_STUB = """MODULE module_configure
  IMPLICIT NONE
  INTEGER :: stub_spec_bdy_width = 5, stub_blend_width = 5
CONTAINS
  SUBROUTINE nl_get_spec_bdy_width(id, value)
    INTEGER, INTENT(IN) :: id
    INTEGER, INTENT(OUT) :: value
    value = stub_spec_bdy_width
  END SUBROUTINE nl_get_spec_bdy_width
  SUBROUTINE nl_get_blend_width(id, value)
    INTEGER, INTENT(IN) :: id
    INTEGER, INTENT(OUT) :: value
    value = stub_blend_width
  END SUBROUTINE nl_get_blend_width
END MODULE module_configure
"""

_INLINE = """MODULE wrf_topo_inline
  IMPLICIT NONE
  ! The fields start_em.F reads off grid% and config_flags%, under the
  ! names it reads them by, so its lines paste unmodified.
  TYPE grid_t
    REAL, ALLOCATABLE :: toposlpx(:,:), toposlpy(:,:), lap_hgt(:,:)
    REAL, ALLOCATABLE :: slope(:,:), slp_azi(:,:), ht(:,:)
    REAL, ALLOCATABLE :: msftx(:,:), msfty(:,:), sina(:,:), cosa(:,:)
    REAL :: rdx, rdy
  END TYPE grid_t
  TYPE config_t
    LOGICAL :: periodic_x = .FALSE., periodic_y = .FALSE.
  END TYPE config_t
CONTAINS
  SUBROUTINE slope_block(grid, config_flags, ids, ide, jds, jde, &
                         its, ite, jts, jte)
    TYPE(grid_t), INTENT(INOUT) :: grid
    TYPE(config_t), INTENT(IN) :: config_flags
    INTEGER, INTENT(IN) :: ids, ide, jds, jde, its, ite, jts, jte
    INTEGER :: i, j, im1, ip1, jm1, jp1
    REAL :: hx, hy, pi
{slope}  END SUBROUTINE slope_block

  SUBROUTINE diffuse_block(coszen, swdown, ht, solcon, swddif, swddir, &
                           swddni, diffuse_frac, ruiz, its, ite, jts, jte)
    INTEGER, INTENT(IN) :: its, ite, jts, jte
    LOGICAL, INTENT(IN) :: ruiz
    REAL, INTENT(IN) :: solcon
    REAL, DIMENSION(its:ite, jts:jte), INTENT(IN) :: coszen, swdown, ht
    REAL, DIMENSION(its:ite, jts:jte), INTENT(INOUT) :: swddif, swddir, swddni
    REAL, DIMENSION(its:ite, jts:jte), INTENT(OUT) :: diffuse_frac
    INTEGER :: i, j
    REAL :: ioh, kt, airmass, kd
    IF (ruiz) THEN
{ruiz_arias}    ENDIF
{diffuse_frac}  END SUBROUTINE diffuse_block
END MODULE wrf_topo_inline
"""


if __name__ == "__main__":
    raise SystemExit(main())
