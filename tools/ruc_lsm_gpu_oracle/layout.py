"""The one byte layout both sides of the RUC LSM column oracle share.

``write_inputs`` serialises a :func:`columns.build_case` case (after the WOOF
runner has filled in its cold-start state) into ``inputs.bin``;
``fortran_driver`` emits the Fortran program that reads exactly that stream,
runs WRF's surface-driver RUC arm (module_surface_driver.F:3438-3593,
FRACTIONAL_SEAICE and isisfc as the run sets them) around the unmodified
``LSMRUC`` and ``SFCDIAGS_RUCLSM``, and writes ``outputs.bin`` in the order
``read_outputs`` expects.  Every array is little-endian float32 or int32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from columns import ATMOS, PROFILES, STATE_2D, STATIC_2D, STEP_2D

#: Fields the seam rewrites after the call, beyond STATE_2D.
SEAM_OUT = ("chs", "flhc", "flqc", "cpm", "cqs2", "chs2", "qgh", "vegfra")
#: Every 2-D word compared after each step.
OUT_2D = STATE_2D + SEAM_OUT
#: Cold-start (RUCLSMINIT) outputs.
INIT_2D = ("mavail", "znt")
INIT_PROFILES = ("sh2o", "smfr3d")
#: The single-count SFCEVP words the driver also writes.  module_sf_ruclsm.F
#: adds qfx*dt twice per land step (:1095 and :1116), a WRF defect WOOF does
#: not copy.  ``sfcevp_once_step`` is WRF's entry SFCEVP plus qfx*dt once (the
#: reference for a replayed step, which starts from WRF's state);
#: ``sfcevp_once_acc`` carries its own single-count accumulator from S0 (the
#: reference for a free-running comparison).  Both are the same float32
#: expression WRF evaluates, applied once.
SFCEVP_ONCE_STEP = "sfcevp_once_step"
SFCEVP_ONCE_ACC = "sfcevp_once_acc"

_HEADER_INTS = ("ncol", "nzs", "nlcat", "nscat", "nsteps", "mosaic_lu",
                "mosaic_soil", "lakemodel", "rdlai2d", "fractional_seaice",
                "usgs", "reserved")


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype="<f4"))


def write_inputs(path: Path, case: dict, s0: dict, raw: dict,
                 xice_threshold: float, seaice_albedo_default: float) -> None:
    """``s0``: WOOF field name -> numpy array, the state both sides start from.

    2-D arrays are (ncol,), profiles (nzs, ncol) in WOOF order.  C-order
    bytes of a (nzs, ncol) array are column-major (ncol, nzs) bytes, which is
    exactly the Fortran array the driver reads, so nothing is transposed.
    """
    ncol, nzs = case["ncol"], case["nzs"]
    mosaic = case["mosaic"]
    head = np.array([ncol, nzs, s0["landusef"].shape[0],
                     s0["soilctop"].shape[0], case["nsteps"], mosaic, mosaic,
                     case["lakemodel"], case["rdlai2d"],
                     case["fractional_seaice"], 0, 0], dtype="<i4")
    parts = [head.tobytes(),
             _f32([case["dt"], xice_threshold, seaice_albedo_default]).tobytes()]
    # RUCLSMINIT's inputs, as WOOF's cold start saw them.
    parts += [_f32(raw["tslb"]).tobytes(), _f32(raw["smois"]).tobytes(),
              np.asarray(raw["ivgtyp"], "<i4").tobytes(),
              np.asarray(raw["isltyp"], "<i4").tobytes(),
              _f32(raw["xice"]).tobytes()]
    for name in STATE_2D + STATIC_2D:
        parts.append(_f32(s0[name]).tobytes())
    for name in PROFILES:
        parts.append(_f32(s0[name]).tobytes())
    parts.append(np.asarray(s0["ivgtyp"], "<i4").tobytes())
    parts.append(np.asarray(s0["isltyp"], "<i4").tobytes())
    parts.append(_f32(s0["landusef"]).tobytes())
    parts.append(_f32(s0["soilctop"]).tobytes())
    for step in case["steps"]:
        for name in STEP_2D:
            parts.append(_f32(step["forcing"][name]).tobytes())
        for name in ATMOS:
            parts.append(_f32(step["atmos"][name]).tobytes())
    Path(path).write_bytes(b"".join(parts))


def read_outputs(path: Path, ncol: int, nzs: int, nsteps: int) -> dict:
    """``outputs.bin`` -> {'init': {...}, 'steps': [{...}, ...]}."""
    data = np.fromfile(path, dtype="<f4")
    at = 0

    def take(count):
        nonlocal at
        out = data[at:at + count]
        at += count
        return out

    init = {}
    for name in INIT_2D:
        init[name] = take(ncol).copy()
    for name in INIT_PROFILES:
        init[name] = take(ncol * nzs).reshape(nzs, ncol).copy()
    steps = []
    for _ in range(nsteps):
        out = {}
        for name in OUT_2D:
            out[name] = take(ncol).copy()
        for name in PROFILES:
            out[name] = take(ncol * nzs).reshape(nzs, ncol).copy()
        out[SFCEVP_ONCE_STEP] = take(ncol).copy()
        out[SFCEVP_ONCE_ACC] = take(ncol).copy()
        steps.append(out)
    if at != data.size:
        raise ValueError(f"{path}: {data.size - at} trailing words")
    return {"init": init, "steps": steps}


def fortran_driver() -> str:
    """The Fortran column driver, generated from the shared field lists."""
    two_d = STATE_2D + STATIC_2D + STEP_2D + ATMOS
    decl = [f"  real, allocatable :: {n}(:,:)" for n in two_d]
    decl += [f"  real, allocatable :: {n}(:,:,:)" for n in PROFILES]
    alloc = [f"  allocate({n}(ncol,1))" for n in two_d]
    alloc += [f"  allocate({n}(ncol,nzs,1))" for n in PROFILES]
    read_s0 = [f"  read(11) {n}" for n in STATE_2D + STATIC_2D + PROFILES]
    read_step = [f"    read(11) {n}" for n in STEP_2D + ATMOS]
    write_step = [f"    write(12) {n}" for n in STATE_2D + SEAM_OUT + PROFILES]
    nl = "\n"
    return f"""! Generated by tools/ruc_lsm_gpu_oracle/layout.py; do not edit.
! WRF v4.6.1 module_surface_driver.F:3438-3593 (CASE RUCLSMSCHEME) around the
! unmodified LSMRUC and SFCDIAGS_RUCLSM, one column per i, j=1.
program run_columns
  use module_model_constants, only: cp, rcp, g, xlv, stbolt, r_d
  use module_sf_ruclsm, only: lsmruc, ruclsminit
  use module_sf_sfcdiags_ruclsm, only: sfcdiags_ruclsm
  implicit none
  integer :: hdr(12), ncol, nzs, nlcat, nscat, nsteps, mosaic_lu, mosaic_soil
  integer :: lakemodel, irdlai2d, fractional_seaice, iusgs, iswater, isice
  integer :: i, k, step
  real :: rhdr(3), dt, xice_threshold, seaice_albedo_default
  logical :: rdlai2d, frpcpn, myj
  character(len=32) :: mminlu
  character(len=1024) :: inpath, outpath
  real, allocatable :: raw_tslb(:,:,:), raw_smois(:,:,:), raw_xice(:,:)
  integer, allocatable :: raw_ivgtyp(:,:), raw_isltyp(:,:)
  real, allocatable :: init_sh2o(:,:,:), init_smfr3d(:,:,:), init_mavail(:,:), init_znt(:,:)
  integer, allocatable :: ivgtyp(:,:), isltyp(:,:)
  real, allocatable :: landusef(:,:,:), soilctop(:,:,:)
  real, allocatable :: z3d(:,:,:), p8w(:,:,:), t3d(:,:,:), qv3d(:,:,:), qc3d(:,:,:), rho3d(:,:,:)
  real, allocatable :: pattern_spp_lsm(:,:,:), field_sf(:,:,:), cqs(:,:), sfcevp_in(:,:)
  real, allocatable :: sfcevp_once_step(:,:), sfcevp_once_acc(:,:)
  real, allocatable :: zs(:)
{nl.join(decl)}

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(unit=11, file=trim(inpath), access='stream', form='unformatted', status='old')
  read(11) hdr
  read(11) rhdr
  ncol=hdr(1); nzs=hdr(2); nlcat=hdr(3); nscat=hdr(4); nsteps=hdr(5)
  mosaic_lu=hdr(6); mosaic_soil=hdr(7); lakemodel=hdr(8); irdlai2d=hdr(9)
  fractional_seaice=hdr(10); iusgs=hdr(11)
  dt=rhdr(1); xice_threshold=rhdr(2); seaice_albedo_default=rhdr(3)
  rdlai2d = irdlai2d /= 0
  frpcpn = .true.
  myj = .false.
  if (iusgs /= 0) then
    mminlu='USGS'; iswater=16; isice=24
  else
    mminlu='MODIFIED_IGBP_MODIS_NOAH'; iswater=17; isice=15
  end if
  allocate(zs(nzs))
  if (nzs == 9) then
    zs = (/ 0.00 , 0.01 , 0.04 , 0.10 , 0.30, 0.60, 1.00 , 1.60, 3.00 /)
  else
    zs = (/ 0.00 , 0.05 , 0.20 , 0.40 , 1.60, 3.00 /)
  end if
  allocate(raw_tslb(ncol,nzs,1), raw_smois(ncol,nzs,1), raw_xice(ncol,1))
  allocate(raw_ivgtyp(ncol,1), raw_isltyp(ncol,1))
  allocate(init_sh2o(ncol,nzs,1), init_smfr3d(ncol,nzs,1), init_mavail(ncol,1), init_znt(ncol,1))
  allocate(ivgtyp(ncol,1), isltyp(ncol,1), landusef(ncol,nlcat,1), soilctop(ncol,nscat,1))
  allocate(z3d(ncol,nzs,1), p8w(ncol,nzs,1), t3d(ncol,nzs,1), qv3d(ncol,nzs,1))
  allocate(qc3d(ncol,nzs,1), rho3d(ncol,nzs,1))
  allocate(pattern_spp_lsm(ncol,nzs,1), field_sf(ncol,nzs,1), cqs(ncol,1))
  allocate(sfcevp_in(ncol,1), sfcevp_once_step(ncol,1), sfcevp_once_acc(ncol,1))
{nl.join(alloc)}
  pattern_spp_lsm = 0.0
  field_sf = 0.0

  ! ---- RUCLSMINIT on the cold-start inputs; it also reads the tables ----
  read(11) raw_tslb
  read(11) raw_smois
  read(11) raw_ivgtyp
  read(11) raw_isltyp
  read(11) raw_xice
  init_sh2o = 0.0
  init_smfr3d = 0.0
  init_mavail = 0.0
  init_znt = 0.0
  call ruclsminit(init_sh2o, init_smfr3d, raw_tslb, raw_smois, raw_isltyp, raw_ivgtyp, &
       mminlu, raw_xice, init_mavail, nzs, iswater, isice, init_znt, .false., .true., &
       1,ncol+1,1,2,1,nzs+1, 1,ncol,1,1,1,nzs, 1,ncol,1,1,1,nzs)
  open(unit=12, file=trim(outpath), access='stream', form='unformatted', status='replace')
  write(12) init_mavail
  write(12) init_znt
  write(12) init_sh2o
  write(12) init_smfr3d

  ! ---- the state both sides start from ----
{nl.join(read_s0)}
  read(11) ivgtyp
  read(11) isltyp
  read(11) landusef
  read(11) soilctop
  sfcevp_once_acc = sfcevp

  do step = 1, nsteps
{nl.join(read_step)}
    do k = 1, nzs
      z3d(:,k,1) = dz(:,1)
      p8w(:,k,1) = pressure(:,1)
      t3d(:,k,1) = temperature(:,1)
      qv3d(:,k,1) = qv(:,1)
      qc3d(:,k,1) = qc(:,1)
      rho3d(:,k,1) = rho(:,1)
    end do

    ! module_surface_driver.F:3453-3459
    do i = 1, ncol
      if ( ( xice(i,1) .ge. xice_threshold ) .and. ( xice(i,1) .le. 1. ) ) then
        albbck(i,1) = seaice_albedo_default
      end if
    end do
    ! :3460-3474 (FRACTIONAL_SEAICE == 1); isisfc is true (the fractional
    ! surface-layer wrapper supplies the open-water components)
    if ( fractional_seaice == 1 ) then
      do i = 1, ncol
        if ( ( xice(i,1) .ge. xice_threshold ) .and. ( xice(i,1) .le. 1 ) ) then
          albedo(i,1) = (albedo(i,1) - (1.-xice(i,1))*0.08) / xice(i,1)
          emiss(i,1)  = (emiss(i,1)  - (1.-xice(i,1))*0.98) / xice(i,1)
          tsk(i,1) = tsk_save(i,1)
        end if
      end do
    end if
    sfcevp_in = sfcevp

    call lsmruc(0, pattern_spp_lsm, field_sf, dt, step, nzs, &
         lakemodel, lakemask, surface_graupelncv, surface_snowncv, surface_rainncv, &
         zs, rainbl, snow, snowh, snowc, sr, frpcpn, &
         rhosnf, precipfr, &
         z3d, p8w, t3d, qv3d, qc3d, rho3d, &
         glw, gsw, emiss, chklowq, &
         chs, flqc, flhc, mavail, canwat, vegfra, albedo, znt, &
         z0, snoalb, albbck, lai, &
         mminlu, landusef, nlcat, mosaic_lu, &
         mosaic_soil, soilctop, nscat, &
         qsfc, qsg, qvg, qcg, dew, soilt1, tsnav, &
         tmn, ivgtyp, isltyp, xland, &
         iswater, isice, xice, xice_threshold, &
         cp, rcp, g, xlv, stbolt, &
         smois, sh2o, smstav, smstot, tslb, tsk, hfx, qfx, lh, &
         sfcrunoff, udrunoff, acrunoff, sfcexc, &
         sfcevp, grdflx, snowfallac, acsnow, acsnom, &
         smfr3d, keepfr3dflag, &
         myj, shdmin, shdmax, rdlai2d, &
         1,ncol+1,1,2,1,nzs+1, 1,ncol,1,1,1,nzs, 1,ncol,1,1,1,nzs)

    ! WRF counts qfx*dt into SFCEVP twice on a land step (:1095 and :1116);
    ! the single count, in the same float32 arithmetic, for the WOOF side.
    ! Only the land arm accumulates (:828 sends water to its own arm, :824
    ! skips a lake under lakemodel 1), so only the land arm counts once here.
    do i = 1, ncol
      if ( (xland(i,1)-1.5) < 0. .and. &
           .not. (lakemodel == 1 .and. lakemask(i,1) == 1.) ) then
        sfcevp_once_step(i,1) = sfcevp_in(i,1) + qfx(i,1) * dt
        sfcevp_once_acc(i,1) = sfcevp_once_acc(i,1) + qfx(i,1) * dt
      else
        sfcevp_once_step(i,1) = sfcevp(i,1)
      end if
    end do

    ! :3530-3577
    if ( fractional_seaice == 1 ) then
      do i = 1, ncol
        if ( ( xice(i,1) .ge. xice_threshold ) .and. ( xice(i,1) .le. 1.0 ) ) then
          albedo(i,1) = ( albedo(i,1) * xice(i,1) ) + ( (1.0-xice(i,1)) * 0.08  )
          emiss(i,1)  = ( emiss(i,1)  * xice(i,1) ) + ( (1.0-xice(i,1)) * 0.98  )
        end if
      end do
      do i = 1, ncol
        if ( ( xice(i,1) .ge. xice_threshold ) .and. ( xice(i,1) .le. 1.0 ) ) then
          flhc(i,1) = ( flhc(i,1) * xice(i,1) ) + ( (1.-xice(i,1)) * flhc_sea(i,1) )
          flqc(i,1) = ( flqc(i,1) * xice(i,1) ) + ( (1.-xice(i,1)) * flqc_sea(i,1) )
          cpm(i,1)  = ( cpm(i,1)  * xice(i,1) ) + ( (1.-xice(i,1)) * cpm_sea(i,1)  )
          cqs2(i,1) = ( cqs2(i,1) * xice(i,1) ) + ( (1.-xice(i,1)) * cqs2_sea(i,1) )
          chs2(i,1) = ( chs2(i,1) * xice(i,1) ) + ( (1.-xice(i,1)) * chs2_sea(i,1) )
          chs(i,1)  = ( chs(i,1)  * xice(i,1) ) + ( (1.-xice(i,1)) * chs_sea(i,1)  )
          qsfc(i,1) = ( qsfc(i,1) * xice(i,1) ) + ( (1.-xice(i,1)) * qsfc_sea(i,1) )
          qgh(i,1)  = ( qgh(i,1)  * xice(i,1) ) + ( (1.-xice(i,1)) * qgh_sea(i,1)  )
          hfx(i,1)  = ( hfx(i,1)  * xice(i,1) ) + ( (1.-xice(i,1)) * hfx_sea(i,1)  )
          qfx(i,1)  = ( qfx(i,1)  * xice(i,1) ) + ( (1.-xice(i,1)) * qfx_sea(i,1)  )
          lh(i,1)   = ( lh(i,1)   * xice(i,1) ) + ( (1.-xice(i,1)) * lh_sea(i,1)   )
          tsk_save(i,1)  = tsk(i,1)
          tsk(i,1)  = ( tsk(i,1)  * xice(i,1) ) + ( (1.-xice(i,1)) * tsk_sea(i,1)  )
        end if
      end do
    end if

    ! :3580-3585
    do i = 1, ncol
      cqs(i,1) = flqc(i,1)/(mavail(i,1)*rho3d(i,1,1))
      chs(i,1) = flhc(i,1)/(cpm(i,1)*rho3d(i,1,1) )
    end do

    ! :3587
    call sfcdiags_ruclsm(hfx, qfx, tsk, qsfc, cqs, cqs2, chs, chs2, t2, th2, q2, &
         t3d, qv3d, rho3d, p8w, psfc, snow, &
         cp, r_d, rcp, &
         1,ncol+1,1,2,1,nzs+1, 1,ncol,1,1,1,nzs, 1,ncol,1,1,1,nzs)

{nl.join(write_step)}
    write(12) sfcevp_once_step
    write(12) sfcevp_once_acc
  end do
  close(11)
  close(12)
end program run_columns
"""


if __name__ == "__main__":
    import sys
    Path(sys.argv[1]).write_text(fortran_driver())
