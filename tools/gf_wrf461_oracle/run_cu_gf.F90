program run_cu_gf_oracle
  ! Drive the byte-unmodified WRF v4.6.1 Grell-Freitas cumulus scheme
  ! (cu_physics = 3) through its own WRF entry point `GFDRV` and dump every
  ! input and every output as float32 CSV.
  !
  ! The pinned boundary is the WRF driver, not the inner cloud model, for the
  ! same reason the Shin-Hong oracle pinned `shinhong` rather than
  ! `shinhong2d`: everything GFDRV does to build the column -- the mb
  ! conversion (module_cu_gf_wrfdrv.F:385,404), the 1.e-8 moisture floors
  ! (:411,:419), the `zo` half-layer stack (:396-399), the `omeg = -g*rho*w`
  ! identity (:471), the `mconv` vertical integral (:484-492), the dhdt
  ! boundary-layer forcing (:416), and the `cuten`/`cutens` gating and the
  ! `/pi` division that produce RTHCUTEN (:745-752) -- is inside the fixture.
  !
  ! `run_cup_gf.F90` beside this program decomposes the same cases one stage
  ! deeper, by replicating that preparation and calling `cup_gf` directly, so
  ! a failing port can be pointed at a stage instead of a black box.  The two
  ! share `gf_cases.F90` so their inputs cannot drift.
  !
  ! Three modules compile as they ship, with no stub, when WRF_CHEM = 0.
  ! `nm -u` on the objects is the receipt.
  !
  ! Two arms per case, because ishallow (namelist `ishallow`) is the one
  ! runtime switch that changes which schemes run at all:
  !   * arm 0: ishallow_g3 = 0 -- CUP_gf_sh is never called (:504) and
  !            cutens is forced to 0 (:331); deep convection only.
  !   * arm 1: ishallow_g3 = 1 -- CUP_gf_sh runs, neg_check('shallow') runs,
  !            and the shallow diagnostics are written (:713-723).
  !
  ! Column independence is measured, not assumed.  Every case is run twice:
  ! once packed into one gf_ncase-wide slab and once alone in a one-column
  ! tile.  gf-isolation.csv records the comparison.
  !
  ! Nothing here invents an expected value: every number in the CSVs is
  ! either an input this program constructed or a word the pinned scheme
  ! wrote.
  use module_cu_gf_wrfdrv, only: gfdrv
  use gf_cases
  implicit none

  integer, parameter :: nz = gf_nz
  integer, parameter :: ncase = gf_ncase
  integer, parameter :: ndx = gf_ndx
  integer, parameter :: narm = 2

  ! Tile geometry.  GFDRV computes its own write window as
  !     ibegc = max(its, ids+4),  iendc = min(ite, ide-5)
  !     jbegc = max(jts, jds+4),  jendc = min(jte, jde-5)
  ! for the non-periodic case (:264-277), and skips the whole j entirely when
  ! j < jbegc or j > jendc (:712).  A tile that does not respect that window
  ! silently returns all zeros, so the domain is sized to make every active
  ! column land inside it.
  integer, parameter :: ids = 1, ide = ncase + 9
  integer, parameter :: its = ids + 4, ite = ide - 5
  integer, parameter :: jds = 1, jde = 10
  integer, parameter :: jts = jds + 4, jte = jde - 5
  integer, parameter :: kds = 1, kde = nz + 1
  integer, parameter :: kts = 1, kte = nz
  integer, parameter :: ims = ids, ime = ide
  integer, parameter :: jms = jds, jme = jde
  integer, parameter :: kms = kds, kme = kde

  ! Solo tile: one active column, same halo contract.
  integer, parameter :: sids = 1, side = 10
  integer, parameter :: sits = sids + 4, site = side - 5

  ! --- slab state ------------------------------------------------------------
  real, dimension(ims:ime, kms:kme, jms:jme) :: u3, v3, w3, t3, q3, p3, pi3
  real, dimension(ims:ime, kms:kme, jms:jme) :: rho3, dz8w3, p8w3
  real, dimension(ims:ime, kms:kme, jms:jme) :: rthften, rqvften, rthraten
  real, dimension(ims:ime, kms:kme, jms:jme) :: rthblten, rqvblten
  real, dimension(ims:ime, kms:kme, jms:jme) :: rthcuten, rqvcuten
  real, dimension(ims:ime, kms:kme, jms:jme) :: rqccuten, rqicuten
  real, dimension(ims:ime, kms:kme, jms:jme) :: dudt_phy, dvdt_phy
  real, dimension(ims:ime, kms:kme, jms:jme) :: gdc, gdc2
  real, dimension(ims:ime, kms:kme, jms:jme) :: pattern_spp, field_spp
  real, dimension(ims:ime, jms:jme) :: raincv, pratec, htop, hbot
  real, dimension(ims:ime, jms:jme) :: ht2, hfx2, qfx2, xland2, xmbsh
  integer, dimension(ims:ime, jms:jme) :: kpbl2, ktopdeep, k22sh, kbconsh, ktopsh

  ! --- solo state ------------------------------------------------------------
  real, dimension(sids:side, kms:kme, jms:jme) :: su3, sv3, sw3, st3, sq3
  real, dimension(sids:side, kms:kme, jms:jme) :: sp3, spi3, srho3, sdz8w3, sp8w3
  real, dimension(sids:side, kms:kme, jms:jme) :: srthften, srqvften, srthraten
  real, dimension(sids:side, kms:kme, jms:jme) :: srthblten, srqvblten
  real, dimension(sids:side, kms:kme, jms:jme) :: srthcuten, srqvcuten
  real, dimension(sids:side, kms:kme, jms:jme) :: srqccuten, srqicuten
  real, dimension(sids:side, kms:kme, jms:jme) :: sdudt, sdvdt, sgdc, sgdc2
  real, dimension(sids:side, kms:kme, jms:jme) :: spattern, sfield
  real, dimension(sids:side, jms:jme) :: sraincv, spratec, shtop, shbot
  real, dimension(sids:side, jms:jme) :: sht2, shfx2, sqfx2, sxland2, sxmbsh
  integer, dimension(sids:side, jms:jme) :: skpbl2, sktopdeep, sk22sh
  integer, dimension(sids:side, jms:jme) :: skbconsh, sktopsh

  real :: dtstep, dxv
  integer :: ichoice, ishallow
  integer :: spp_conv
  logical :: periodic_x, periodic_y
  integer :: ic, idx, iarm, k, i, ulev, usfc, uiso
  integer :: ndiff_lev, ndiff_sfc
  character(len=1024) :: level_path, surface_path, iso_path

  call get_command_argument(1, level_path)
  call get_command_argument(2, surface_path)
  call get_command_argument(3, iso_path)
  if (len_trim(level_path) == 0 .or. len_trim(surface_path) == 0 .or. &
      len_trim(iso_path) == 0) then
    write(*, '(A)') 'usage: run_cu_gf LEVELS.csv SURFACE.csv ISOLATION.csv'
    error stop 2
  end if

  call gf_build_case_table()

  spp_conv = 0
  periodic_x = .false.
  periodic_y = .false.
  ! clos_choice.  The WRF namelist default is 0 = full 16-member ensemble
  ! average; the fixture uses the default so the whole ensemble is exercised.
  ichoice = 0
  dtstep = 60.0

  open(newunit=ulev, file=trim(level_path), status='replace', action='write')
  write(ulev, '(A)') 'case,idx,arm,k,' // &
    'u,v,w,t,qv,p,pi,rho,dz8w,p8w,' // &
    'rthften,rqvften,rthraten,rthblten,rqvblten,' // &
    'rthcuten,rqvcuten,rqccuten,rqicuten,dudt_phy,dvdt_phy,gdc,gdc2'
  open(newunit=usfc, file=trim(surface_path), status='replace', action='write')
  write(usfc, '(A)') 'case,idx,arm,nz,dt,dx,ichoice,ishallow,' // &
    'ht,hfx,qfx,xland,kpbl,psfc_p8w,' // &
    'raincv,pratec,htop,hbot,ktop_deep,' // &
    'k22_shallow,kbcon_shallow,ktop_shallow,xmb_shallow'
  open(newunit=uiso, file=trim(iso_path), status='replace', action='write')
  write(uiso, '(A)') 'case,idx,arm,ndiff_level_words,ndiff_surface_words'

  do idx = 1, ndx
    dxv = gf_dxsweep(idx)
    do iarm = 1, narm
      ishallow = iarm - 1

      ! ---- packed slab: every case in one GFDRV call ----------------------
      call fill_slab()
      call gfdrv(spp_conv, pattern_spp, field_spp,                          &
                 DT=dtstep, DX=dxv,                                         &
                 rho=rho3, RAINCV=raincv, PRATEC=pratec,                    &
                 U=u3, V=v3, t=t3, W=w3, q=q3, p=p3, pi=pi3,                &
                 dz8w=dz8w3, p8w=p8w3,                                      &
                 htop=htop, hbot=hbot, ktop_deep=ktopdeep,                  &
                 HT=ht2, hfx=hfx2, qfx=qfx2, XLAND=xland2,                  &
                 GDC=gdc, GDC2=gdc2, kpbl=kpbl2,                            &
                 k22_shallow=k22sh, kbcon_shallow=kbconsh,                  &
                 ktop_shallow=ktopsh, xmb_shallow=xmbsh,                    &
                 ichoice=ichoice, ishallow_g3=ishallow,                     &
                 ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,      &
                 ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,      &
                 its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte,      &
                 periodic_x=periodic_x, periodic_y=periodic_y,              &
                 RQVCUTEN=rqvcuten, RQCCUTEN=rqccuten, RQICUTEN=rqicuten,   &
                 RQVFTEN=rqvften, RTHFTEN=rthften,                          &
                 RTHCUTEN=rthcuten, RTHRATEN=rthraten,                      &
                 rqvblten=rqvblten, rthblten=rthblten,                      &
                 dudt_phy=dudt_phy, dvdt_phy=dvdt_phy)

      ! ---- emit the packed-slab answer ------------------------------------
      ! GFDRV takes every state field as INTENT(IN) and never writes the
      ! *FTEN / *BLTEN / RTHRATEN arrays it declares INTENT(INOUT) (it only
      ! reads them at :412-417), so the inputs printed here are the words the
      ! call actually saw.
      do ic = 1, ncase
        i = its + ic - 1
        do k = kts, kte
          write(ulev, '(I0,",",I0,",",I0,",",I0)', advance='no')            &
            ic, idx, ishallow, k
          write(ulev, '(23(",",ES24.16E3))')                                &
            u3(i, k, jts), v3(i, k, jts), w3(i, k, jts), t3(i, k, jts),     &
            q3(i, k, jts), p3(i, k, jts), pi3(i, k, jts),                   &
            rho3(i, k, jts), dz8w3(i, k, jts), p8w3(i, k, jts),             &
            rthften(i, k, jts), rqvften(i, k, jts), rthraten(i, k, jts),    &
            rthblten(i, k, jts), rqvblten(i, k, jts),                       &
            rthcuten(i, k, jts), rqvcuten(i, k, jts),                       &
            rqccuten(i, k, jts), rqicuten(i, k, jts),                       &
            dudt_phy(i, k, jts), dvdt_phy(i, k, jts),                       &
            gdc(i, k, jts), gdc2(i, k, jts)
        end do
        write(usfc, '(I0,",",I0,",",I0,",",I0)', advance='no')              &
          ic, idx, ishallow, nz
        write(usfc, '(",",ES24.16E3,",",ES24.16E3,",",I0,",",I0)',          &
              advance='no') dtstep, dxv, ichoice, ishallow
        write(usfc, '(4(",",ES24.16E3),",",I0,",",ES24.16E3)',              &
              advance='no') ht2(i, jts), hfx2(i, jts), qfx2(i, jts),        &
              xland2(i, jts), kpbl2(i, jts), p8w3(i, kts, jts)
        write(usfc, '(4(",",ES24.16E3),",",I0)', advance='no')              &
          raincv(i, jts), pratec(i, jts), htop(i, jts), hbot(i, jts),       &
          ktopdeep(i, jts)
        write(usfc, '(3(",",I0),",",ES24.16E3)')                            &
          k22sh(i, jts), kbconsh(i, jts), ktopsh(i, jts), xmbsh(i, jts)
      end do

      do ic = 1, ncase
        ! ---- solo tile: the same case, alone ------------------------------
        call fill_solo(ic)
        call gfdrv(spp_conv, spattern, sfield,                              &
                   DT=dtstep, DX=dxv,                                       &
                   rho=srho3, RAINCV=sraincv, PRATEC=spratec,               &
                   U=su3, V=sv3, t=st3, W=sw3, q=sq3, p=sp3, pi=spi3,       &
                   dz8w=sdz8w3, p8w=sp8w3,                                  &
                   htop=shtop, hbot=shbot, ktop_deep=sktopdeep,             &
                   HT=sht2, hfx=shfx2, qfx=sqfx2, XLAND=sxland2,            &
                   GDC=sgdc, GDC2=sgdc2, kpbl=skpbl2,                       &
                   k22_shallow=sk22sh, kbcon_shallow=skbconsh,              &
                   ktop_shallow=sktopsh, xmb_shallow=sxmbsh,                &
                   ichoice=ichoice, ishallow_g3=ishallow,                   &
                   ids=sids, ide=side, jds=jds, jde=jde, kds=kds, kde=kde,  &
                   ims=sids, ime=side, jms=jms, jme=jme, kms=kms, kme=kme,  &
                   its=sits, ite=site, jts=jts, jte=jte, kts=kts, kte=kte,  &
                   periodic_x=periodic_x, periodic_y=periodic_y,            &
                   RQVCUTEN=srqvcuten, RQCCUTEN=srqccuten,                  &
                   RQICUTEN=srqicuten,                                      &
                   RQVFTEN=srqvften, RTHFTEN=srthften,                      &
                   RTHCUTEN=srthcuten, RTHRATEN=srthraten,                  &
                   rqvblten=srqvblten, rthblten=srthblten,                  &
                   dudt_phy=sdudt, dvdt_phy=sdvdt)

        i = its + ic - 1
        ndiff_lev = 0
        do k = kts, kte
          ndiff_lev = ndiff_lev                                            &
            + bitdiff(rthcuten(i, k, jts), srthcuten(sits, k, jts))        &
            + bitdiff(rqvcuten(i, k, jts), srqvcuten(sits, k, jts))        &
            + bitdiff(rqccuten(i, k, jts), srqccuten(sits, k, jts))        &
            + bitdiff(rqicuten(i, k, jts), srqicuten(sits, k, jts))        &
            + bitdiff(dudt_phy(i, k, jts), sdudt(sits, k, jts))            &
            + bitdiff(dvdt_phy(i, k, jts), sdvdt(sits, k, jts))            &
            + bitdiff(gdc(i, k, jts), sgdc(sits, k, jts))                  &
            + bitdiff(gdc2(i, k, jts), sgdc2(sits, k, jts))
        end do
        ndiff_sfc = bitdiff(raincv(i, jts), sraincv(sits, jts))            &
          + bitdiff(pratec(i, jts), spratec(sits, jts))                    &
          + bitdiff(htop(i, jts), shtop(sits, jts))                        &
          + bitdiff(hbot(i, jts), shbot(sits, jts))                        &
          + bitdiff(xmbsh(i, jts), sxmbsh(sits, jts))                      &
          + intdiff(ktopdeep(i, jts), sktopdeep(sits, jts))                &
          + intdiff(k22sh(i, jts), sk22sh(sits, jts))                      &
          + intdiff(kbconsh(i, jts), skbconsh(sits, jts))                  &
          + intdiff(ktopsh(i, jts), sktopsh(sits, jts))
        write(uiso, '(I0,",",I0,",",I0,",",I0,",",I0)')                    &
          ic, idx, ishallow, ndiff_lev, ndiff_sfc
      end do
    end do
  end do

  close(ulev)
  close(usfc)
  close(uiso)
  write(*, '(A)') 'gf oracle written'

contains

  ! Bitwise inequality of two float32 words, NaN-aware: two NaNs with the same
  ! payload compare equal here, a NaN against a number does not.
  integer function bitdiff(a, b)
    real, intent(in) :: a, b
    if (transfer(a, 0) == transfer(b, 0)) then
      bitdiff = 0
    else
      bitdiff = 1
    end if
  end function bitdiff

  integer function intdiff(a, b)
    integer, intent(in) :: a, b
    if (a == b) then
      intdiff = 0
    else
      intdiff = 1
    end if
  end function intdiff

  subroutine fill_slab()
    real :: zc(nz), dz(nz), tt(nz), qq(nz), pp(nz), ppw(nz + 1)
    real :: ppi(nz), rr(nz), uu(nz), vv(nz), ww(nz)
    integer :: n, ii, kk, jj

    u3 = 0.0; v3 = 0.0; w3 = 0.0; t3 = 0.0; q3 = 0.0
    p3 = 0.0; pi3 = 1.0; rho3 = 0.0; dz8w3 = 0.0; p8w3 = 0.0
    rthften = 0.0; rqvften = 0.0; rthraten = 0.0
    rthblten = 0.0; rqvblten = 0.0
    rthcuten = 0.0; rqvcuten = 0.0; rqccuten = 0.0; rqicuten = 0.0
    dudt_phy = 0.0; dvdt_phy = 0.0; gdc = 0.0; gdc2 = 0.0
    pattern_spp = 0.0; field_spp = 0.0
    raincv = 0.0; pratec = 0.0; htop = 0.0; hbot = 0.0
    ht2 = 0.0; hfx2 = 0.0; qfx2 = 0.0; xland2 = 1.0; xmbsh = 0.0
    kpbl2 = 1; ktopdeep = 0; k22sh = 0; kbconsh = 0; ktopsh = 0

    do jj = jms, jme
      do n = 1, ncase
        ii = its + n - 1
        call gf_column(n, zc, dz, tt, qq, pp, ppw, ppi, rr, uu, vv, ww)
        do kk = 1, nz
          u3(ii, kk, jj) = uu(kk); v3(ii, kk, jj) = vv(kk)
          w3(ii, kk, jj) = ww(kk); t3(ii, kk, jj) = tt(kk)
          q3(ii, kk, jj) = qq(kk); p3(ii, kk, jj) = pp(kk)
          pi3(ii, kk, jj) = ppi(kk); rho3(ii, kk, jj) = rr(kk)
          dz8w3(ii, kk, jj) = dz(kk); p8w3(ii, kk, jj) = ppw(kk)
          rthften(ii, kk, jj) = c_thf(n) * (1.0 - zc(kk) / 16000.0)
          rqvften(ii, kk, jj) = c_qvf(n) * (1.0 - zc(kk) / 10000.0)
          rthraten(ii, kk, jj) = c_thrad(n)
          if (kk <= c_kpbl(n)) then
            rthblten(ii, kk, jj) = c_thbl(n)
            rqvblten(ii, kk, jj) = c_qvbl(n)
          end if
        end do
        p8w3(ii, nz + 1, jj) = ppw(nz + 1)
        pi3(ii, nz + 1, jj) = 1.0
        ht2(ii, jj) = c_ht(n); hfx2(ii, jj) = c_hfx(n)
        qfx2(ii, jj) = c_qfx(n); xland2(ii, jj) = c_xland(n)
        kpbl2(ii, jj) = c_kpbl(n)
      end do
    end do
  end subroutine fill_slab

  subroutine fill_solo(n)
    integer, intent(in) :: n
    real :: zc(nz), dz(nz), tt(nz), qq(nz), pp(nz), ppw(nz + 1)
    real :: ppi(nz), rr(nz), uu(nz), vv(nz), ww(nz)
    integer :: ii, kk, jj

    su3 = 0.0; sv3 = 0.0; sw3 = 0.0; st3 = 0.0; sq3 = 0.0
    sp3 = 0.0; spi3 = 1.0; srho3 = 0.0; sdz8w3 = 0.0; sp8w3 = 0.0
    srthften = 0.0; srqvften = 0.0; srthraten = 0.0
    srthblten = 0.0; srqvblten = 0.0
    srthcuten = 0.0; srqvcuten = 0.0; srqccuten = 0.0; srqicuten = 0.0
    sdudt = 0.0; sdvdt = 0.0; sgdc = 0.0; sgdc2 = 0.0
    spattern = 0.0; sfield = 0.0
    sraincv = 0.0; spratec = 0.0; shtop = 0.0; shbot = 0.0
    sht2 = 0.0; shfx2 = 0.0; sqfx2 = 0.0; sxland2 = 1.0; sxmbsh = 0.0
    skpbl2 = 1; sktopdeep = 0; sk22sh = 0; skbconsh = 0; sktopsh = 0

    call gf_column(n, zc, dz, tt, qq, pp, ppw, ppi, rr, uu, vv, ww)
    do jj = jms, jme
      ii = sits
      do kk = 1, nz
        su3(ii, kk, jj) = uu(kk); sv3(ii, kk, jj) = vv(kk)
        sw3(ii, kk, jj) = ww(kk); st3(ii, kk, jj) = tt(kk)
        sq3(ii, kk, jj) = qq(kk); sp3(ii, kk, jj) = pp(kk)
        spi3(ii, kk, jj) = ppi(kk); srho3(ii, kk, jj) = rr(kk)
        sdz8w3(ii, kk, jj) = dz(kk); sp8w3(ii, kk, jj) = ppw(kk)
        srthften(ii, kk, jj) = c_thf(n) * (1.0 - zc(kk) / 16000.0)
        srqvften(ii, kk, jj) = c_qvf(n) * (1.0 - zc(kk) / 10000.0)
        srthraten(ii, kk, jj) = c_thrad(n)
        if (kk <= c_kpbl(n)) then
          srthblten(ii, kk, jj) = c_thbl(n)
          srqvblten(ii, kk, jj) = c_qvbl(n)
        end if
      end do
      sp8w3(ii, nz + 1, jj) = ppw(nz + 1)
      spi3(ii, nz + 1, jj) = 1.0
      sht2(ii, jj) = c_ht(n); shfx2(ii, jj) = c_hfx(n)
      sqfx2(ii, jj) = c_qfx(n); sxland2(ii, jj) = c_xland(n)
      skpbl2(ii, jj) = c_kpbl(n)
    end do
  end subroutine fill_solo

end program run_cu_gf_oracle
