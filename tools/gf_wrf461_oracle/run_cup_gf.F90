program run_cup_gf_oracle
  ! Decompose the GFDRV fixture one stage deeper.
  !
  ! `run_cu_gf.F90` pins the WRF entry point, which is the right boundary for
  ! a fixture but a black box for a failing port: `ierr`, `ierrc`, `xmb`,
  ! `kbcon`, `k22`, `jmin`, `edt` and the ten-slot `forcing` diagnostic are
  ! all local to GFDRV and never reach a caller.  A port whose RTHCUTEN is
  ! wrong has no way to say whether it lost the parcel at `cup_kbcon`, at the
  ! closure, or in `neg_check`.
  !
  ! This program replicates GFDRV's own column preparation
  ! (module_cu_gf_wrfdrv.F:383-492) and then calls the public entry points
  ! directly -- `CUP_gf_sh`, `neg_check`, `cup_gf`, `neg_check` -- capturing
  ! every intermediate.  Both harnesses share `gf_cases.F90`, so their inputs
  ! are the same words.
  !
  ! The replication is not asserted, it is PROVEN per case: after the direct
  ! calls this program applies GFDRV's own output algebra (:724-840) to the
  ! captured tendencies and compares the result, bitwise, against a real
  ! GFDRV call on the same column.  `gf-stage-consistency.csv` records the
  ! differing-word count.  A nonzero count there means this decomposition has
  ! drifted from the driver and the stage numbers must not be trusted.
  !
  ! Constants: the preparation deliberately uses the GFS physcons values the
  ! driver USEs (con_g = 9.80665, con_cp = 1004.6, con_hvap = 2.5e6), NOT the
  ! solver's 9.81/1004.  The deep and shallow modules then use their own
  ! locals internally.  Both sets are live in the same call; nothing here
  ! harmonises them.
  use module_cu_gf_deep, only: cup_gf, neg_check
  use module_cu_gf_sh, only: cup_gf_sh
  use module_cu_gf_wrfdrv, only: gfdrv
  use gf_cases
  implicit none

  integer, parameter :: nz = gf_nz
  integer, parameter :: ncase = gf_ncase
  integer, parameter :: ndx = gf_ndx

  ! Direct-call geometry: one column, no halo contract (cup_gf has no write
  ! window of its own; the four-cell inset is GFDRV's).
  integer, parameter :: its = 1, ite = 1, itf = 1
  integer, parameter :: kts = 1, kte = nz, ktf = nz

  ! GFDRV cross-check tile: the four-cell inset the driver demands.
  integer, parameter :: gids = 1, gide = 10
  integer, parameter :: gits = gids + 4, gite = gide - 5
  integer, parameter :: gjds = 1, gjde = 10
  integer, parameter :: gjts = gjds + 4, gjte = gjde - 5
  integer, parameter :: gkds = 1, gkde = nz + 1

  ! GFS physcons, as USEd by module_cu_gf_wrfdrv.F:5-8.
  !
  ! These are DOUBLE PRECISION and that is load-bearing.  module_gfs_machine.F
  ! sets kind_phys = selected_real_kind(13,60), i.e. real(8), and
  ! module_gfs_physcons.F declares every constant real(kind=kind_phys).  So
  ! the driver's `g`, `cp`, `xlv` and `r_v` are real(8) parameters, and every
  ! expression they appear in -- omeg = -g*rho*w (:471), mconv += omeg*dq/g
  ! (:487), dhdt = cp*rthblten*pi + xlv*rqvblten (:416) -- is evaluated in
  ! double and rounded exactly once, on the store into the real(4) local.
  !
  ! A float32 port that spells these constants as float32 is off by an ULP in
  ! omeg and dhdt, and this fixture measured that: with float32 constants the
  ! decomposition disagreed with GFDRV on 10 of 216 rows, all of them columns
  ! sitting on a branch boundary where one ULP flips the answer.
  integer, parameter :: kind_phys = selected_real_kind(13, 60)
  real(kind=kind_phys), parameter :: con_g = 9.80665e+0_kind_phys
  real(kind=kind_phys), parameter :: con_cp = 1.0046e+3_kind_phys
  real(kind=kind_phys), parameter :: con_hvap = 2.5000e+6_kind_phys

  ! --- the prepared column (GFDRV's locals, by their WRF names) -------------
  real, dimension(its:ite, kts:kte) :: zo, t2d, q2d, po, p2d, us, vs, rhoi
  real, dimension(its:ite, kts:kte) :: tn, qo, tshall, qshall, dhdt, omeg
  real, dimension(its:ite, kts:kte) :: outt, outq, outqc, outu, outv
  real, dimension(its:ite, kts:kte) :: cupclw, cnvwt, zu, zd
  real, dimension(its:ite, kts:kte) :: outts, outqs, outqcs, outus, outvs
  real, dimension(its:ite, kts:kte) :: cupclws, cnvwts, zus
  real, dimension(its:ite, kts:kte) :: qcheck
  real, dimension(its:ite, kts:kte) :: omeg_in, q2d_in, qo_in
  ! Exner on the case column, saved by prepare_column so the GFDRV output
  ! algebra can be reconstructed without reading back the driver's own tile.
  real, dimension(kts:kte) :: pi_col
  real, dimension(its:ite) :: ter11, psur, xlandi, hfxi, qfxi, dxi, ccn
  real, dimension(its:ite) :: mconv, xmb, xmbs, xmbm, xmb_out, pret, prets
  real, dimension(its:ite) :: edt, mconv_in
  real, dimension(its:ite, 10) :: forcing
  real, dimension(its:ite, 4) :: rand_clos
  real, dimension(its:ite) :: rand_mom, rand_vmas
  integer, dimension(its:ite) :: kpbli, ierr, kbcon, ktop, k22, jmin, cactiv
  integer, dimension(its:ite) :: ierrs, kbcons, ktops, k22s
  character(len=50) :: ierrc(its:ite), ierrcs(its:ite)

  ! --- the driver cross-check ------------------------------------------------
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gu, gv, gw, gt, gq, gp
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gpi, grho, gdz, gp8w
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gthften, gqvften, gthraten
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gthblten, gqvblten
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gthcuten, gqvcuten
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gqccuten, gqicuten
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gdudt, gdvdt, gc1, gc2
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gpat, gfld
  real, dimension(gids:gide, gjds:gjde) :: graincv, gpratec, ghtop, ghbot
  real, dimension(gids:gide, gjds:gjde) :: ght, ghfx, gqfx, gxland, gxmbsh
  integer, dimension(gids:gide, gjds:gjde) :: gkpbl, gktopd, gk22s, gkbcs, gktps

  ! --- reconstructed driver outputs, from the direct calls ------------------
  real, dimension(kts:kte) :: r_thcuten, r_qvcuten, r_qccuten, r_qicuten
  real, dimension(kts:kte) :: r_dudt, r_dvdt
  real :: r_pratec, cuten, cutens

  real :: dtstep, dxv, tcrit
  integer :: ichoice, ichoice_s, ishallow
  integer :: ic, idx, iarm, k, n, ulev, usfc, ucon, ndiff
  integer :: spp0
  logical :: perx, pery
  character(len=1024) :: level_path, surface_path, cons_path

  call get_command_argument(1, level_path)
  call get_command_argument(2, surface_path)
  call get_command_argument(3, cons_path)
  if (len_trim(level_path) == 0 .or. len_trim(surface_path) == 0 .or. &
      len_trim(cons_path) == 0) then
    write(*, '(A)') 'usage: run_cup_gf LEVELS.csv SURFACE.csv CONSISTENCY.csv'
    error stop 2
  end if

  call gf_build_case_table()

  ! The values GFDRV hard-codes (module_cu_gf_wrfdrv.F:68-74, :257-262).
  tcrit = 258.
  ichoice = 0        ! clos_choice namelist default
  ichoice_s = 0      ! parameter, :71
  dtstep = 60.0
  spp0 = 0
  perx = .false.
  pery = .false.

  open(newunit=ulev, file=trim(level_path), status='replace', action='write')
  write(ulev, '(A)') 'case,idx,arm,k,' // &
    'zo,po,t2d,q2d,tn,qo,tshall,qshall,dhdt,us,vs,rhoi,omeg_in,' // &
    'zu,zd,cnvwt,cupclw,outt,outq,outqc,outu,outv,' // &
    'zus,cnvwts,cupclws,outts,outqs,outqcs,' // &
    'q2d_after,qo_after,omeg_after'
  open(newunit=usfc, file=trim(surface_path), status='replace', action='write')
  write(usfc, '(A)') 'case,idx,arm,dt,dx,ichoice,ishallow,' // &
    'ter11,psur,xlandi,hfxi,qfxi,kpbli,ccn,mconv_in,mconv_out,' // &
    'ierr,kbcon,ktop,k22,jmin,edt,xmb_out,pret,' // &
    'f1_capeGrell,f2_omega,f3_mconv,f4_aa0,f5_aa1,f6_xaa0,f7_xk,f8,f9,f10,' // &
    'ierrs,kbcons,ktops,k22s,xmbs,prets,ierrc,ierrcs'
  open(newunit=ucon, file=trim(cons_path), status='replace', action='write')
  write(ucon, '(A)') 'case,idx,arm,ndiff_words_vs_gfdrv,' // &
    'ktop_direct,ktop_driver,xmbs_direct,xmbs_driver,ierr_direct'

  do idx = 1, ndx
    dxv = gf_dxsweep(idx)
    do iarm = 1, 2
      ishallow = iarm - 1
      do ic = 1, ncase

        call fill_driver_tile(ic)
        call prepare_column(ic, dxv)
        omeg_in = omeg
        q2d_in = q2d
        qo_in = qo
        mconv_in = mconv

        ! ---- stage S: shallow -------------------------------------------
        ! GFDRV:504.  cutens starts at 1 and is forced to 0 when
        ! ishallow_g3 == 0 (:331), so the shallow arrays stay zero there.
        outts = 0.0; outqs = 0.0; outqcs = 0.0; outus = 0.0; outvs = 0.0
        cupclws = 0.0; cnvwts = 0.0; zus = 0.0
        xmbs = 0.0; prets = 0.0
        ierrs = 0; kbcons = 0; ktops = 0; k22s = 0
        ierrcs(:) = " "
        cutens = 1.0
        if (ishallow == 0) cutens = 0.0
        if (ishallow == 1) then
          call cup_gf_sh(zo, t2d, q2d, ter11, tshall, qshall, p2d, psur,   &
                         dhdt, kpbli, rhoi, hfxi, qfxi, xlandi, ichoice_s, &
                         tcrit, dtstep,                                    &
                         zus, xmbs, kbcons, ktops, k22s, ierrs, ierrcs,    &
                         outts, outqs, outqcs, cnvwts, prets, cupclws,     &
                         itf, ktf, its, ite, kts, kte, 0)
          if (xmbs(its) <= 0.0) cutens = 0.0
          call neg_check('shallow', 0, dtstep, q2d, outqs, outts, outus,   &
                         outvs, outqcs, prets, its, ite, kts, kte, itf, ktf)
        end if

        ! ---- stage D: deep ----------------------------------------------
        ! GFDRV:627.  imid = 0 and nranflag = 0 are literals in the driver;
        ! xmbm_in is the mid-level mass flux, which is identically zero
        ! because imid_gf is a parameter 0 (:69).
        outt = 0.0; outq = 0.0; outqc = 0.0; outu = 0.0; outv = 0.0
        cupclw = 0.0; cnvwt = 0.0; zu = 0.0; zd = 0.0
        edt = 0.0; xmb = 0.0; xmb_out = 0.0; xmbm = 0.0; pret = 0.0
        ierr = 0; kbcon = 0; ktop = 0; k22 = 0; jmin = 0
        forcing = 0.0
        ierrc(:) = " "
        call cup_gf(itf, ktf, its, ite, kts, kte,                          &
                    1,            & ! dicycle, parameter at :72
                    ichoice,                                               &
                    0,            & ! ipr
                    ccn, dtstep,                                           &
                    0,            & ! imid
                    kpbli, dhdt, xlandi,                                   &
                    zo, forcing, t2d, q2d, ter11, tn, qo, p2d, psur,       &
                    us, vs, rhoi, hfxi, qfxi, dxi, mconv, omeg,            &
                    cactiv, cnvwt, zu, zd, edt, xmb, xmbm, xmbs, pret,     &
                    outu, outv, outt, outq, outqc, kbcon, ktop, cupclw,    &
                    ierr, ierrc,                                           &
                    rand_mom, rand_vmas, rand_clos,                        &
                    0,            & ! nranflag
                    k22, jmin)
        do k = kts, ktf
          qcheck(its, k) = q2d(its, k) + outqs(its, k) * dtstep
        end do
        call neg_check('deep', 0, dtstep, qcheck, outq, outt, outu, outv,  &
                       outqc, pret, its, ite, kts, kte, itf, ktf)
        xmb_out(its) = xmb(its)

        ! ---- reconstruct GFDRV's outputs (:724-840) ----------------------
        cuten = 0.0
        if (pret(its) > 0.0) then
          cuten = 1.0
        else
          kbcon(its) = 0
          ktop(its) = 0
        end if
        do k = kts, ktf
          r_thcuten(k) = (cutens * outts(its, k) + cuten * outt(its, k))   &
                         / pi_col(k)
          r_qvcuten(k) = cuten * outq(its, k) + cutens * outqs(its, k)
          r_dudt(k) = outu(its, k) * cuten
          r_dvdt(k) = outv(its, k) * cuten
          if (t2d(its, k) < 258.) then
            r_qicuten(k) = outqcs(its, k) + outqc(its, k) * cuten
            r_qccuten(k) = 0.0
          else
            r_qicuten(k) = 0.0
            r_qccuten(k) = outqcs(its, k) + outqc(its, k) * cuten
          end if
        end do
        r_pratec = 0.0
        if (pret(its) > 0.0 .or. prets(its) > 0.0) then
          r_pratec = cuten * pret(its) + cutens * prets(its)
        end if

        ! ---- the driver, on the same column ------------------------------
        ! The tile is already built and untouched: GFDRV takes every state
        ! field INTENT(IN) and never writes the forcing arrays it declares
        ! INTENT(INOUT).
        call gfdrv(spp0, gpat, gfld, DT=dtstep, DX=dxv,                    &
                   rho=grho, RAINCV=graincv, PRATEC=gpratec,               &
                   U=gu, V=gv, t=gt, W=gw, q=gq, p=gp, pi=gpi,             &
                   dz8w=gdz, p8w=gp8w,                                     &
                   htop=ghtop, hbot=ghbot, ktop_deep=gktopd,               &
                   HT=ght, hfx=ghfx, qfx=gqfx, XLAND=gxland,               &
                   GDC=gc1, GDC2=gc2, kpbl=gkpbl,                          &
                   k22_shallow=gk22s, kbcon_shallow=gkbcs,                 &
                   ktop_shallow=gktps, xmb_shallow=gxmbsh,                 &
                   ichoice=ichoice, ishallow_g3=ishallow,                  &
                   ids=gids, ide=gide, jds=gjds, jde=gjde,                 &
                   kds=gkds, kde=gkde,                                     &
                   ims=gids, ime=gide, jms=gjds, jme=gjde,                 &
                   kms=gkds, kme=gkde,                                     &
                   its=gits, ite=gite, jts=gjts, jte=gjte,                 &
                   kts=kts, kte=kte,                                       &
                   periodic_x=perx, periodic_y=pery,                       &
                   RQVCUTEN=gqvcuten, RQCCUTEN=gqccuten,                   &
                   RQICUTEN=gqicuten,                                      &
                   RQVFTEN=gqvften, RTHFTEN=gthften,                       &
                   RTHCUTEN=gthcuten, RTHRATEN=gthraten,                   &
                   rqvblten=gqvblten, rthblten=gthblten,                   &
                   dudt_phy=gdudt, dvdt_phy=gdvdt)

        ndiff = 0
        do k = kts, ktf
          ndiff = ndiff                                                    &
            + bitdiff(r_thcuten(k), gthcuten(gits, k, gjts))               &
            + bitdiff(r_qvcuten(k), gqvcuten(gits, k, gjts))               &
            + bitdiff(r_qccuten(k), gqccuten(gits, k, gjts))               &
            + bitdiff(r_qicuten(k), gqicuten(gits, k, gjts))               &
            + bitdiff(r_dudt(k), gdudt(gits, k, gjts))                     &
            + bitdiff(r_dvdt(k), gdvdt(gits, k, gjts))
        end do
        ndiff = ndiff + bitdiff(r_pratec, gpratec(gits, gjts))
        write(ucon, '(I0,",",I0,",",I0,",",I0)', advance='no')            &
          ic, idx, ishallow, ndiff
        write(ucon, '(2(",",I0),2(",",ES24.16E3),",",I0)')                 &
          ktop(its), gktopd(gits, gjts), xmbs(its), gxmbsh(gits, gjts),    &
          ierr(its)

        ! ---- emit ---------------------------------------------------------
        do k = kts, ktf
          write(ulev, '(I0,",",I0,",",I0,",",I0)', advance='no')           &
            ic, idx, ishallow, k
          write(ulev, '(13(",",ES24.16E3))', advance='no')                 &
            zo(its, k), po(its, k), t2d(its, k), q2d_in(its, k),           &
            tn(its, k), qo_in(its, k), tshall(its, k), qshall(its, k),     &
            dhdt(its, k), us(its, k), vs(its, k), rhoi(its, k),            &
            omeg_in(its, k)
          write(ulev, '(18(",",ES24.16E3))')                               &
            zu(its, k), zd(its, k), cnvwt(its, k), cupclw(its, k),         &
            outt(its, k), outq(its, k), outqc(its, k), outu(its, k),       &
            outv(its, k),                                                  &
            zus(its, k), cnvwts(its, k), cupclws(its, k),                  &
            outts(its, k), outqs(its, k), outqcs(its, k),                  &
            q2d(its, k), qo(its, k), omeg(its, k)
        end do

        write(usfc, '(I0,",",I0,",",I0)', advance='no') ic, idx, ishallow
        write(usfc, '(2(",",ES24.16E3),2(",",I0))', advance='no')          &
          dtstep, dxv, ichoice, ishallow
        write(usfc, '(5(",",ES24.16E3),",",I0,3(",",ES24.16E3))',          &
          advance='no') ter11(its), psur(its), xlandi(its), hfxi(its),     &
          qfxi(its), kpbli(its), ccn(its), mconv_in(its), mconv(its)
        write(usfc, '(5(",",I0),3(",",ES24.16E3))', advance='no')          &
          ierr(its), kbcon(its), ktop(its), k22(its), jmin(its),           &
          edt(its), xmb_out(its), pret(its)
        write(usfc, '(10(",",ES24.16E3))', advance='no')                   &
          (forcing(its, n), n = 1, 10)
        write(usfc, '(4(",",I0),2(",",ES24.16E3))', advance='no')          &
          ierrs(its), kbcons(its), ktops(its), k22s(its), xmbs(its),       &
          prets(its)
        write(usfc, '(",",A,",",A)') trim(clean(ierrc(its))),              &
          trim(clean(ierrcs(its)))
      end do
    end do
  end do

  close(ulev)
  close(usfc)
  close(ucon)
  write(*, '(A)') 'gf stage oracle written'

contains

  integer function bitdiff(a, b)
    real, intent(in) :: a, b
    if (transfer(a, 0) == transfer(b, 0)) then
      bitdiff = 0
    else
      bitdiff = 1
    end if
  end function bitdiff

  ! ierrc carries free text; commas would break the CSV.  Nothing else about
  ! the string is touched -- the message is WRF's own diagnosis of why a
  ! column was rejected and is the single most useful thing a failing port
  ! can be compared against.
  function clean(s) result(o)
    character(len=*), intent(in) :: s
    character(len=50) :: o
    integer :: n
    o = s
    do n = 1, len(o)
      if (o(n:n) == ',') o(n:n) = ';'
    end do
  end function clean

  ! GFDRV's own column preparation, module_cu_gf_wrfdrv.F:383-492.
  !
  ! This reads the SAME array words the driver reads -- the tile that
  ! fill_driver_tile built -- rather than recomputing the sounding.  That is
  ! not fussiness: the decomposition is only a decomposition if both paths
  ! consume identical bytes, and recomputing an expression into a scalar
  ! instead of loading it from a real(4) array element is exactly the sort of
  ! difference that survives at -O0 and then gets amplified by a branch.
  subroutine prepare_column(ic, dxv)
    integer, intent(in) :: ic
    real, intent(in) :: dxv
    real :: dq
    integer :: k, i0, j0

    i0 = gits
    j0 = gjts

    ! per-column scalars (:353-357, :369, :384-395)
    dxi(its) = dxv
    psur(its) = gp8w(i0, 1, j0) * .01
    ter11(its) = max(0., ght(i0, j0))
    hfxi(its) = ghfx(i0, j0)
    qfxi(its) = gqfx(i0, j0)
    xlandi(its) = gxland(i0, j0)
    kpbli(its) = gkpbl(i0, j0)
    ccn(its) = 150.
    cactiv(its) = 0
    mconv(its) = 0.
    rand_mom(:) = 0.
    rand_vmas(:) = 0.
    rand_clos(:, :) = 0.

    ! the zo half-layer stack (:396-399)
    zo(its, kts) = ter11(its) + .5 * gdz(i0, 1, j0)
    do k = kts + 1, ktf
      zo(its, k) = zo(its, k - 1) + .5 * (gdz(i0, k - 1, j0) + gdz(i0, k, j0))
    end do

    do k = kts, ktf
      po(its, k) = gp(i0, k, j0) * .01
      pi_col(k) = gpi(i0, k, j0)
      p2d(its, k) = po(its, k)
      rhoi(its, k) = grho(i0, k, j0)
      us(its, k) = gu(i0, k, j0)
      vs(its, k) = gv(i0, k, j0)
      t2d(its, k) = gt(i0, k, j0)
      q2d(its, k) = gq(i0, k, j0)
      if (q2d(its, k) < 1.e-08) q2d(its, k) = 1.e-08

      tn(its, k) = t2d(its, k) + (gthften(i0, k, j0) + gthraten(i0, k, j0)   &
                                  + gthblten(i0, k, j0))                     &
                                 * gpi(i0, k, j0) * dtstep
      qo(its, k) = q2d(its, k) + (gqvften(i0, k, j0) + gqvblten(i0, k, j0))  &
                                 * dtstep
      tshall(its, k) = t2d(its, k) + gthblten(i0, k, j0) * gpi(i0, k, j0)    &
                                     * dtstep
      dhdt(its, k) = con_cp * gthblten(i0, k, j0) * gpi(i0, k, j0)           &
                     + con_hvap * gqvblten(i0, k, j0)
      qshall(its, k) = q2d(its, k) + gqvblten(i0, k, j0) * dtstep
      if (tn(its, k) < 200.) tn(its, k) = t2d(its, k)
      if (qo(its, k) < 1.e-08) qo(its, k) = 1.e-08

      ! omega, with the driver's real(8) con_g -- NOT the solver's 9.81 (:471)
      omeg(its, k) = -con_g * grho(i0, k, j0) * gw(i0, k, j0)
    end do

    ! mconv (:484-492), also with con_g
    do k = kts, ktf - 1
      dq = q2d(its, k + 1) - q2d(its, k)
      mconv(its) = mconv(its) + omeg(its, k) * dq / con_g
    end do
    if (mconv(its) < 0.) mconv(its) = 0.
  end subroutine prepare_column

  subroutine fill_driver_tile(ic)
    integer, intent(in) :: ic
    real :: zc(nz), dz(nz), tt(nz), qq(nz), pp(nz), ppw(nz + 1)
    real :: ppi(nz), rr(nz), uu(nz), vv(nz), ww(nz)
    integer :: ii, kk, jj

    gu = 0.0; gv = 0.0; gw = 0.0; gt = 0.0; gq = 0.0
    gp = 0.0; gpi = 1.0; grho = 0.0; gdz = 0.0; gp8w = 0.0
    gthften = 0.0; gqvften = 0.0; gthraten = 0.0
    gthblten = 0.0; gqvblten = 0.0
    gthcuten = 0.0; gqvcuten = 0.0; gqccuten = 0.0; gqicuten = 0.0
    gdudt = 0.0; gdvdt = 0.0; gc1 = 0.0; gc2 = 0.0
    gpat = 0.0; gfld = 0.0
    graincv = 0.0; gpratec = 0.0; ghtop = 0.0; ghbot = 0.0
    ght = 0.0; ghfx = 0.0; gqfx = 0.0; gxland = 1.0; gxmbsh = 0.0
    gkpbl = 1; gktopd = 0; gk22s = 0; gkbcs = 0; gktps = 0

    call gf_column(ic, zc, dz, tt, qq, pp, ppw, ppi, rr, uu, vv, ww)
    ii = gits
    do jj = gjds, gjde
      do kk = 1, nz
        gu(ii, kk, jj) = uu(kk); gv(ii, kk, jj) = vv(kk)
        gw(ii, kk, jj) = ww(kk); gt(ii, kk, jj) = tt(kk)
        gq(ii, kk, jj) = qq(kk); gp(ii, kk, jj) = pp(kk)
        gpi(ii, kk, jj) = ppi(kk); grho(ii, kk, jj) = rr(kk)
        gdz(ii, kk, jj) = dz(kk); gp8w(ii, kk, jj) = ppw(kk)
        gthften(ii, kk, jj) = c_thf(ic) * (1.0 - zc(kk) / 16000.0)
        gqvften(ii, kk, jj) = c_qvf(ic) * (1.0 - zc(kk) / 10000.0)
        gthraten(ii, kk, jj) = c_thrad(ic)
        if (kk <= c_kpbl(ic)) then
          gthblten(ii, kk, jj) = c_thbl(ic)
          gqvblten(ii, kk, jj) = c_qvbl(ic)
        end if
      end do
      gp8w(ii, nz + 1, jj) = ppw(nz + 1)
      gpi(ii, nz + 1, jj) = 1.0
      ght(ii, jj) = c_ht(ic); ghfx(ii, jj) = c_hfx(ic)
      gqfx(ii, jj) = c_qfx(ic); gxland(ii, jj) = c_xland(ic)
      gkpbl(ii, jj) = c_kpbl(ic)
    end do
  end subroutine fill_driver_tile

end program run_cup_gf_oracle
