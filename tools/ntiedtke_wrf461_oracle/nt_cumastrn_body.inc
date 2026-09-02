! Statement-order replication of cumastrn's body, cu_ntiedtke.F90:460-1085.
!
! Included into run_nt_cumastrn.F90 as a CONTAINED subroutine, so it sees
! the host's variables directly and needs no argument list -- which keeps
! the transcription readable next to the Fortran it mirrors.
!
! Every callee is the REAL globalized procedure.  Nothing here transcribes
! physics except cumastrn's own arithmetic, which is what the closure at
! :620-760 is.  The section numbers in the comments are the reference's.
!
! ldcum / loddraf / llo2 are LOGICAL(4) in the reference and integer here,
! because the bind(C) interfaces to the private routines take them that way
! (bind(C) would map Fortran LOGICAL to _Bool, which is not the callee's
! ABI).  0/1 is exactly how gfortran stores LOGICAL(4).

subroutine replicate_cumastrn()
  logical :: llo1
  integer :: jl2, jk2

  !*    1.  specify constants and parameters  (:466-469)
  zcons  = 1./(g_l*delt)
  zcons2 = 3./(g_l*delt)

  !*    2.  initialize values at vertical grid points in 'cuini'  (:474)
  do jl = 1, nc
    do jk = 1, nz
      zmfus(jl,jk)=0.; zmfds(jl,jk)=0.; zmfuq(jl,jk)=0.; zmfdq(jl,jk)=0.
      zdmfup(jl,jk)=0.; zdmfdp(jl,jk)=0.; zlude(jl,jk)=0.; zdpmel(jl,jk)=0.
      zmfu(jl,jk)=0.; zmfd(jl,jk)=0.; zlu(jl,jk)=0.
    end do
  end do
  call x_cuinin(nc,nz,nz1,nz-1,ztp1,zqp1,zqsat,pum1,pvm1,pverv,pgeo, &
       prsi,pgeoh,ztenh,zqenh,zqsenh,ilwmin,ztu,zqu,ztd,zqd,zuu,zvu, &
       zud,zvd,zmfu,zmfd,zmfus,zmfds,zmfuq,zmfdq,zdmfup,zdmfdp,      &
       zdpmel,zlu,zlude,ilab)

  !*    3.0 cloud base calculations -- (a) cutypen  (:490)
  call x_cutypen(nc,nz,nz1,nz-1,zqp1,ztenh,zqenh,zqsenh,pgeoh,prsi,  &
       phhfl,pqhfl,pgeo,zqsat,prsl,ztp1,slimsk,ztu,zqu,ilab,ldcum,   &
       kcbot,ictop0,ktype,wbase,zlu,kdpl)

  !*        (b) first guess mass flux at cloud base  (:500-541)
  do jl = 1, nc
    zdhpbl(jl)=0.0; upbl(jl)=0.0; idtop(jl)=0
  end do
  do jk = 2, nz
    do jl = 1, nc
      if (jk >= kcbot(jl) .and. ldcum(jl) /= 0) then
        zdhpbl(jl) = zdhpbl(jl) + (alv_l*pqte(jl,jk)+cpd_l*ptte(jl,jk)) &
                     *(prsi(jl,jk+1)-prsi(jl,jk))
        if (slimsk(jl) == 0) then
          wspeed = sqrt(pum1(jl,jk)**2 + pvm1(jl,jk)**2)
          upbl(jl) = upbl(jl) + wspeed*(prsi(jl,jk+1)-prsi(jl,jk))
        end if
      end if
    end do
  end do
  do jl = 1, nc
    if (ldcum(jl) /= 0) then
      ikb = kcbot(jl)
      zmfmax = (prsi(jl,ikb)-prsi(jl,ikb-1))*zcons2
      if (ktype(jl) == 1) then
        zmfub(jl) = 0.1*zmfmax
      else if (ktype(jl) == 2) then
        zqumqe = zqu(jl,ikb)+zlu(jl,ikb)-zqenh(jl,ikb)
        zdqmin = max(0.01*zqenh(jl,ikb),1.e-10)
        zdh = cpd_l*(ztu(jl,ikb)-ztenh(jl,ikb)) + alv_l*zqumqe
        zdh = g_l*max(zdh,1.e5*zdqmin)
        if (zdhpbl(jl) > 0.) then
          zmfub(jl) = zdhpbl(jl)/zdh
          zmfub(jl) = min(zmfub(jl),zmfmax)
        else
          zmfub(jl) = 0.1*zmfmax
          ldcum(jl) = 0
        end if
      end if
    else
      zmfub(jl) = 0.
    end if
  end do

  ! ---- CAPTURE: cuascn's inputs, at the real call site ----------------
  do n = 1, nc
    do k = 1, nz
      write(62,'(i0,",",f0.1,",",i0,13(",",a),",",i0)') n, dxv, k,      &
           hexw(ztenh(n,k)), hexw(zqenh(n,k)), hexw(zqsenh(n,k)),       &
           hexw(ztu(n,k)),   hexw(zqu(n,k)),   hexw(zlu(n,k)),          &
           hexw(zmfu(n,k)),  hexw(zmfub(n)),   hexw(zqsat(n,k)),        &
           hexw(prsl(n,k)),  hexw(prsi(n,k)),  hexw(pgeoh(n,k)),        &
           hexw(pgeo(n,k)),  ilab(n,k)
    end do
  end do

  ! ---- CAPTURE: the rest of what cuascn reads ------------------------
  ! Everything cuascn touches that the first capture did not carry.  It is
  ! cheaper to capture the whole argument list than to reason about which
  ! members are live -- reconstructing them is what failed five times.
  do n = 1, nc
    write(69,'(i0,",",f0.1,8(",",i0),",",a)') n, dxv, ilwmin(n),   &
         ictop0(n), kdpl(n), ldcum(n), ktype(n), kcbot(n), kctop(n), &
         slimsk(n), hexw(wbase(n))
    do k = 1, nz
      write(70,'(i0,",",f0.1,",",i0,12(",",a))') n, dxv, k,        &
           hexw(pum1(n,k)),  hexw(pvm1(n,k)),  hexw(ztp1(n,k)),    &
           hexw(zqp1(n,k)),  hexw(pqte(n,k)),  hexw(pverv(n,k)),   &
           hexw(zuu(n,k)),   hexw(zvu(n,k)),   hexw(zmfus(n,k)),   &
           hexw(zmfuq(n,k)), hexw(zmful(n,k)), hexw(zlude(n,k))
    end do
  end do

  !*    4.  cloud ascent for entraining plume  (:547)
  icum = 0
  call x_cuascn(nc,nz,nz1,nz-1,ztenh,zqenh,pum1,pvm1,ztp1,zqp1,zqsat,  &
       pgeo,pgeoh,prsl,prsi,pqte,pverv,ilwmin,ldcum,zhcbase,ktype,     &
       ilab,ztu,zqu,zlu,zuu,zvu,zmfu,zmfub,zmfus,zmfuq,zmful,zlude,    &
       zdmfup,kcbot,kctop,ictop0,icum,delt,zqsenh,zlglac,slimsk,wup,   &
       wbase,kdpl,pmfude_rate)

  ! ---- CAPTURE: cuascn's outputs -------------------------------------
  do n = 1, nc
    write(64,'(i0,",",f0.1,6(",",i0),2(",",a))') n, dxv, ldcum(n),      &
         ktype(n), kcbot(n), kctop(n), ictop0(n), kdpl(n),              &
         hexw(wup(n)), hexw(wbase(n))
    do k = 1, nz
      ! ztenh/zqenh are OUTPUTS here as well as inputs: :2118-2119 rewrites
      ! both on the negative-buoyancy branch.  Without them in this file the
      ! rewrite is transcribed but never graded.
      write(63,'(i0,",",f0.1,",",i0,13(",",a),",",i0)') n, dxv, k,      &
           hexw(ztu(n,k)),   hexw(zqu(n,k)),   hexw(zlu(n,k)),          &
           hexw(zmfu(n,k)),  hexw(zmfus(n,k)), hexw(zmfuq(n,k)),        &
           hexw(zmful(n,k)), hexw(zlude(n,k)), hexw(zdmfup(n,k)),       &
           hexw(zlglac(n,k)), hexw(pmfude_rate(n,k)),                   &
           hexw(ztenh(n,k)), hexw(zqenh(n,k)), ilab(n,k)
    end do
  end do

  !*    (b) check cloud depth, precipitation rate  (:562-578)
  do jl = 1, nc
    if (ldcum(jl) /= 0) then
      ikb = kcbot(jl)
      itopm2 = kctop(jl)
      zpbmpt = prsi(jl,ikb) - prsi(jl,itopm2)
      if (ktype(jl) == 1 .and. zpbmpt <  2.0e4) ktype(jl) = 2
      if (ktype(jl) == 2 .and. zpbmpt >= 2.0e4) ktype(jl) = 1
      ictop0(jl) = kctop(jl)
    end if
    zrfl(jl) = zdmfup(jl,1)
  end do
  do jk = 2, nz
    do jl = 1, nc
      zrfl(jl) = zrfl(jl) + zdmfup(jl,jk)
    end do
  end do
  do jk = 1, nz
    do jl = 1, nc
      zmfd(jl,jk)=0.; zmfds(jl,jk)=0.; zmfdq(jl,jk)=0.
      zdmfdp(jl,jk)=0.; zdpmel(jl,jk)=0.
    end do
  end do

  ! ---- CAPTURE: cudlfsn's inputs, at its real call site ---------------
  ! The whole argument list, not the members judged live.  ktype and ictop0
  ! here are POST-FLIP -- :566-568 changes ktype between cuascn and this
  ! point, which is the bug that cost two rounds on the closure.  prfl is an
  ! updated parameter, so it is captured on both sides.
  do n = 1, nc
    write(71,'(i0,",",f0.1,5(",",i0),2(",",a))') n, dxv, ldcum(n),      &
         ktype(n), kcbot(n), kctop(n), slimsk(n),                       &
         hexw(zmfub(n)), hexw(zrfl(n))
    do k = 1, nz
      ! ztd..zdmfdp are CLASS-2 dummies of cudlfsn -- nt-aliasing-audit.txt
      ! lists all six as "writes only at" one line each, so every slot the
      ! routine does not reach keeps the caller's value.  They are INPUTS
      ! as much as outputs and are captured as such.
      write(72,'(i0,",",f0.1,",",i0,20(",",a))') n, dxv, k,             &
           hexw(ztenh(n,k)), hexw(zqenh(n,k)), hexw(pum1(n,k)),         &
           hexw(pvm1(n,k)),  hexw(ztp1(n,k)),  hexw(zqsat(n,k)),        &
           hexw(pgeo(n,k)),  hexw(pgeoh(n,k)), hexw(prsi(n,k)),         &
           hexw(ztu(n,k)),   hexw(zqu(n,k)),   hexw(zlu(n,k)),          &
           hexw(zuu(n,k)),   hexw(zvu(n,k)),                            &
           hexw(ztd(n,k)),   hexw(zqd(n,k)),   hexw(zmfd(n,k)),         &
           hexw(zmfds(n,k)), hexw(zmfdq(n,k)), hexw(zdmfdp(n,k))
    end do
  end do

  !*    6.0 downdraft calculations  (lmfdd is .true.)  (:592-613)
  call x_cudlfsn(nc,nz,kcbot,kctop,slimsk,ldcum,ztenh,zqenh,pum1,pvm1, &
       ztp1,zqsat,pgeo,pgeoh,prsi,ztu,zqu,zlu,zuu,zvu,zmfub,zrfl,      &
       ztd,zqd,zud,zvd,zmfd,zmfds,zmfdq,zdmfdp,idtop,loddraf)
  ! ---- CAPTURE: cudlfsn's outputs, BEFORE cuddrafn runs ---------------
  ! Graded standalone.  The pre-closure capture further down is the state
  ! after BOTH downdraft routines, and cannot separate them.
  do n = 1, nc
    write(73,'(i0,",",f0.1,2(",",i0),",",a)') n, dxv, idtop(n),        &
         loddraf(n), hexw(zrfl(n))
    do k = 1, nz
      write(74,'(i0,",",f0.1,",",i0,8(",",a))') n, dxv, k,             &
           hexw(ztd(n,k)),   hexw(zqd(n,k)),   hexw(zud(n,k)),        &
           hexw(zvd(n,k)),   hexw(zmfd(n,k)),  hexw(zmfds(n,k)),      &
           hexw(zmfdq(n,k)), hexw(zdmfdp(n,k))
    end do
  end do

  ! ---- CAPTURE: cuddrafn's inputs, at its real call site --------------
  ! Its six class-1 dummies (prfl, ptd, pqd, pmfd, pmfds, pmfdq) are
  ! cudlfsn's outputs and nothing runs between the two calls, so the file
  ! above would serve -- but stitching one routine's exit capture into
  ! another's entry is the reconstruction this port keeps being burned by.
  ! Captured here instead.
  !
  ! paph_sfc IS NOT OPTIONAL.  cuddrafn reads paph(klev+1) three times
  ! (:2618, :2648, :2649) -- the surface interface that cuascn never touches
  ! and whose slot the cuascn fixture deliberately poisons with NaN.
  do n = 1, nc
    write(75,'(i0,",",f0.1,",",i0,2(",",a))') n, dxv, loddraf(n),        &
         hexw(zrfl(n)), hexw(prsi(n,nz+1))
    do k = 1, nz
      write(76,'(i0,",",f0.1,",",i0,12(",",a))') n, dxv, k,             &
           hexw(ztenh(n,k)), hexw(zqenh(n,k)), hexw(pgeo(n,k)),         &
           hexw(pgeoh(n,k)), hexw(prsi(n,k)),  hexw(zmfu(n,k)),         &
           hexw(ztd(n,k)),   hexw(zqd(n,k)),   hexw(zmfd(n,k)),         &
           hexw(zmfds(n,k)), hexw(zmfdq(n,k)), hexw(zdmfdp(n,k))
    end do
  end do

  call x_cuddrafn(nc,nz,loddraf,ztenh,zqenh,pum1,pvm1,pgeo,pgeoh,prsi, &
       zrfl,ztd,zqd,zud,zvd,zmfu,zmfd,zmfds,zmfdq,zdmfdp,pmfdde_rate)

  ! cuddrafn's updated prfl.  Its level outputs are the pre-closure capture
  ! immediately below -- nothing runs between.
  do n = 1, nc
    write(77,'(i0,",",f0.1,",",a)') n, dxv, hexw(zrfl(n))
  end do

  ! ---- CAPTURE: the PRE-CLOSURE state, at the closure's own entry ------
  ! ztenh/zqenh are captured HERE and not at cuascn's entry because CUASCN
  ! MODIFIES THEM (:2119-2120, the negative-buoyancy correction rewrites
  ! them as 0.5*(pten(jk)+pten(jk-1))).  The closure reads the modified
  ! values.  Capturing the pre-cuascn ones and calling them the closure's
  ! inputs is exactly the reconstruction mistake this architecture exists
  ! to remove.
  do n = 1, nc
    ! zdhpbl is captured HERE, not recomputed downstream: the closure's
    ! shallow arm (:710) reads the value cumastrn built at :506-517 using
    ! CUTYPEN's kcbot and ldcum, and cuascn has since changed both.
    ! Recomputing it from closure-time state is wrong on every shallow
    ! column -- measured, before this was captured.
    ! ktype is captured HERE and not at cuascn's exit: :566-568 FLIPS it
    ! between the two points -- deep becomes shallow when the cloud is
    ! thinner than zdnoprc and shallow becomes deep when it is not.  The
    ! closure selects its arm on the flipped value.  Grading against the
    ! post-cuascn ktype ran the shallow arm on six columns the reference
    ! had already promoted to deep, which is what the 15.8x was.
    write(67,'(i0,",",f0.1,3(",",i0),2(",",a))') n, dxv, loddraf(n), &
         idtop(n), ktype(n), hexw(upbl(n)), hexw(zdhpbl(n))
    do k = 1, nz
      write(66,'(i0,",",f0.1,",",i0,9(",",a))') n, dxv, k,        &
           hexw(zmfd(n,k)),  hexw(zmfds(n,k)), hexw(zmfdq(n,k)),  &
           hexw(zdmfdp(n,k)), hexw(ztd(n,k)),  hexw(zqd(n,k)),    &
           hexw(pmfdde_rate(n,k)),                                &
           hexw(ztenh(n,k)), hexw(zqenh(n,k))
    end do
  end do

  !*    6.1 CAPE closure for deep convection  (:620-728)
  do jl = 1, nc
    if (ldcum(jl) /= 0 .and. ktype(jl) == 1) then
      ikb = kcbot(jl); ikt = kctop(jl)
      zheat(jl)=0.0; zcape(jl)=0.0; zcape1(jl)=0.0; zcape2(jl)=0.0
      zmfub1(jl)=zmfub(jl)
      ztauc(jl) = (pgeoh(jl,ikt)-pgeoh(jl,ikb))/((2.+min(15.0,wup(jl)))*g_l)
      if (slimsk(jl) == 0) then
        upbl(jl) = 2.+ upbl(jl)/(prsi(jl,nz1)-prsi(jl,ikb))
        ztaubl(jl) = (pgeoh(jl,ikb)-pgeoh(jl,nz1))/(g_l*upbl(jl))
        ztaubl(jl) = min(300., ztaubl(jl))
      else
        ztaubl(jl) = ztauc(jl)
      end if
    end if
  end do
  do jk = 1, nz
    do jl = 1, nc
      llo1 = ldcum(jl) /= 0 .and. ktype(jl) == 1
      if (llo1 .and. jk <= kcbot(jl) .and. jk > kctop(jl)) then
        ikb = kcbot(jl)
        zdz = pgeo(jl,jk-1)-pgeo(jl,jk)
        zdp = prsl(jl,jk)-prsl(jl,jk-1)
        zheat(jl) = zheat(jl) + ((ztp1(jl,jk-1)-ztp1(jl,jk)+zdz*rcpd_l) / &
                    ztenh(jl,jk)+vtmpc1_l*(zqp1(jl,jk-1)-zqp1(jl,jk))) *  &
                    (g_l*(zmfu(jl,jk)+zmfd(jl,jk)))
        zcape1(jl) = zcape1(jl) + ((ztu(jl,jk)-ztenh(jl,jk))/ztenh(jl,jk) &
                    + vtmpc1_l*(zqu(jl,jk)-zqenh(jl,jk))-zlu(jl,jk))*zdp
      end if
      if (llo1 .and. jk >= kcbot(jl)) then
      if ((prsi(jl,nz1)-prsi(jl,kdpl(jl))) < 50.e2) then
        zdp = prsi(jl,jk+1)-prsi(jl,jk)
        zcape2(jl) = zcape2(jl) + ztaubl(jl)* &
                   ((1.+vtmpc1_l*zqp1(jl,jk))*ptte(jl,jk) &
                    + vtmpc1_l*ztp1(jl,jk)*pqte(jl,jk))*zdp
      end if
      end if
    end do
  end do
  do jl = 1, nc
    if (ldcum(jl) /= 0 .and. ktype(jl) == 1) then
      ikb = kcbot(jl); ikt = kctop(jl)
      ztauc(jl) = max(delt,ztauc(jl))
      ztauc(jl) = max(360.,ztauc(jl))
      ztauc(jl) = min(10800.,ztauc(jl))
      ztau = ztauc(jl)*scale_fac(jl)
      ! nonequil is .true.
      zcape2(jl) = max(0.,zcape2(jl))
      zcape(jl)  = max(0.,min(zcape1(jl)-zcape2(jl),5000.))
      zheat(jl)  = max(1.e-4,zheat(jl))
      zmfub1(jl) = (zcape(jl)*zmfub(jl))/(zheat(jl)*ztau)
      zmfub1(jl) = max(zmfub1(jl),0.001)
      zmfmax = (prsi(jl,ikb)-prsi(jl,ikb-1))*zcons2
      zmfub1(jl) = min(zmfub1(jl),zmfmax)
    end if
  end do
  !*  6.2 ktype = 2  (:696-720)
  do jl = 1, nc
    if (ldcum(jl) /= 0 .and. ktype(jl) == 2) then
      ikb = kcbot(jl)
      if (zmfd(jl,ikb) < 0.0 .and. loddraf(jl) /= 0) then
        zeps = -zmfd(jl,ikb)/max(zmfub(jl),1.e-10)
      else
        zeps = 0.
      end if
      zqumqe = zqu(jl,ikb)+zlu(jl,ikb)-zeps*zqd(jl,ikb) &
               -(1.-zeps)*zqenh(jl,ikb)
      zdqmin = max(0.01*zqenh(jl,ikb),1.e-10)
      zmfmax = (prsi(jl,ikb)-prsi(jl,ikb-1))*zcons2
      zdh = cpd_l*(ztu(jl,ikb)-zeps*ztd(jl,ikb)-(1.-zeps)*ztenh(jl,ikb)) &
            + alv_l*zqumqe
      zdh = g_l*max(zdh,1.e5*zdqmin)
      if (zdhpbl(jl) > 0.) then
        zmfub1(jl) = zdhpbl(jl)/zdh
      else
        zmfub1(jl) = zmfub(jl)
      end if
      ! CAPTURE the shallow arm's own intermediates.  The mirror
      ! disagreed here by 15.8x and two rounds of reasoning about which
      ! input was stale did not settle it; capturing the terms localises
      ! it to one of them in a single run.
      write(68,'(i0,",",f0.1,8(",",a))') jl, dxv, hexw(zeps), &
           hexw(zqumqe), hexw(zdqmin), hexw(zdh), hexw(zmfmax), &
           hexw(zqu(jl,ikb)), hexw(zlu(jl,ikb)), hexw(zqenh(jl,ikb))
      zmfub1(jl) = zmfub1(jl)/scale_fac2(jl)
      zmfub1(jl) = min(zmfub1(jl),zmfmax)
    end if
    !*  6.3 mid-level
    if (ldcum(jl) /= 0 .and. ktype(jl) == 3) zmfub1(jl) = zmfub(jl)
  end do
  !*  6.4 scale the downdraft mass flux  (:726-740)
  do jk = 1, nz
    do jl = 1, nc
      if (ldcum(jl) /= 0) then
        zfac = zmfub1(jl)/max(zmfub(jl),1.e-10)
        zmfd(jl,jk)=zmfd(jl,jk)*zfac
        zmfds(jl,jk)=zmfds(jl,jk)*zfac
        zmfdq(jl,jk)=zmfdq(jl,jk)*zfac
        zdmfdp(jl,jk)=zdmfdp(jl,jk)*zfac
        pmfdde_rate(jl,jk)=pmfdde_rate(jl,jk)*zfac
      end if
    end do
  end do

  ! ---- CAPTURE: the closure ------------------------------------------
  do n = 1, nc
    write(65,'(i0,",",f0.1,13(",",a))') n, dxv, &
         hexw(zheat(n)), hexw(zcape(n)), hexw(zcape1(n)), hexw(zcape2(n)), &
         hexw(ztauc(n)), hexw(ztaubl(n)), &
         hexw(max(delt,min(10800.0_kind_phys,ztauc(n)))*scale_fac(n)), &
         hexw(zmfub(n)), hexw(zmfub1(n)), hexw(zmfs(n)), hexw(upbl(n)), &
         hexw(scale_fac(n)), hexw(scale_fac2(n))
  end do

  ! ---- CAPTURE: the 6.5/6.6/6.7 block's entry  (:743-819) -------------
  ! CAPTURE FIRST, per the inverted default.  Its OUTPUTS are cuflxn's
  ! captured entry state, so only the entry side is new -- and every array
  ! it scales is one the closure or cuascn produced, which is exactly the
  ! "a neighbour's capture will do" shape that has been wrong six times.
  do n = 1, nc
    write(91,'(i0,",",f0.1,4(",",i0),2(",",a))') n, dxv, ldcum(n),      &
         ktype(n), kcbot(n), idtop(n), hexw(zmfub1(n)), hexw(zmfub(n))
    do k = 1, nz
      write(92,'(i0,",",f0.1,",",i0,13(",",a))') n, dxv, k,              &
           hexw(zmfu(n,k)),   hexw(zmfus(n,k)), hexw(zmfuq(n,k)),       &
           hexw(zmful(n,k)),  hexw(zdmfup(n,k)),hexw(zlude(n,k)),       &
           hexw(pmfude_rate(n,k)), hexw(prsi(n,k)),                    &
           hexw(zmfd(n,k)),   hexw(zmfds(n,k)), hexw(zmfdq(n,k)),       &
           hexw(zdmfdp(n,k)), hexw(pmfdde_rate(n,k))
    end do
  end do

  !*  6.5 scale the updraft mass flux  (:743-773)
  do jl = 1, nc
    if (ldcum(jl) /= 0) zmfs(jl) = zmfub1(jl)/max(1.e-10,zmfub(jl))
  end do
  do jk = 2, nz
    do jl = 1, nc
      if (ldcum(jl) /= 0 .and. jk >= kctop(jl)-1) then
        ikb = kcbot(jl)
        if (jk > ikb) then
          zdz = ((prsi(jl,nz1)-prsi(jl,jk))/(prsi(jl,nz1)-prsi(jl,ikb)))
          zmfu(jl,jk) = zmfu(jl,ikb)*zdz
        end if
        zmfmax = (prsi(jl,jk)-prsi(jl,jk-1))*zcons2
        if (zmfu(jl,jk)*zmfs(jl) > zmfmax) then
          zmfs(jl) = min(zmfs(jl),zmfmax/zmfu(jl,jk))
        end if
      end if
    end do
  end do
  do jk = 2, nz
    do jl = 1, nc
      if (ldcum(jl) /= 0 .and. jk <= kcbot(jl) .and. jk >= kctop(jl)-1) then
        zmfu(jl,jk)=zmfu(jl,jk)*zmfs(jl)
        zmfus(jl,jk)=zmfus(jl,jk)*zmfs(jl)
        zmfuq(jl,jk)=zmfuq(jl,jk)*zmfs(jl)
        zmful(jl,jk)=zmful(jl,jk)*zmfs(jl)
        zdmfup(jl,jk)=zdmfup(jl,jk)*zmfs(jl)
        zlude(jl,jk)=zlude(jl,jk)*zmfs(jl)
        pmfude_rate(jl,jk)=pmfude_rate(jl,jk)*zmfs(jl)
      end if
    end do
  end do

  !*  6.6 ktype = 2 with kcbot == kctop is not allowed  (:776-784)
  do jl = 1, nc
    if (ktype(jl) == 2 .and. kcbot(jl) == kctop(jl) .and. &
        kcbot(jl) >= nz-1) then
      ldcum(jl) = 0
      ktype(jl) = 0
    end if
  end do
  ! lmfscv and lmfpen are both .true. parameters, so the block at :786-802
  ! is dead and llo2 is never set.  Transcribed as the initialisation the
  ! reference leaves it at.
  do jl = 1, nc
    llo2(jl) = 0
  end do

  !*  6.7 downdraft mass fluxes to zero above cloud top  (:805-820)
  do jl = 1, nc
    if (loddraf(jl) /= 0 .and. idtop(jl) <= kctop(jl)) then
      idtop(jl) = kctop(jl) + 1
    end if
  end do
  do jk = 2, nz
    do jl = 1, nc
      if (loddraf(jl) /= 0) then
        if (jk < idtop(jl)) then
          zmfd(jl,jk)=0.; zmfds(jl,jk)=0.; zmfdq(jl,jk)=0.
          pmfdde_rate(jl,jk)=0.; zdmfdp(jl,jk)=0.
        else if (jk == idtop(jl)) then
          pmfdde_rate(jl,jk)=0.
        end if
      end if
    end do
  end do

  ! ---- CAPTURE: cuflxn's inputs, at its real call site ----------------
  ! cuflxn has FOUR class-1 dummies (lddraf, ktype, pmfu, pmfd) and
  ! rewrites pmfus/pmfuq/pmfds/pmfdq/plglac/pqsen/pdmfup/pdmfdp IN PLACE
  ! off their incoming values, so almost the whole argument list is
  ! load-bearing on entry.  Captured whole rather than reasoned about.
  !
  ! paph carries klev+1 here: :2921, :2928 and many later lines read the
  ! surface interface, and pmflxr/pmflxs are klev+1 arrays in their own
  ! right (:2920-2923 writes their klev+1 slot).
  do n = 1, nc
    write(78,'(i0,",",f0.1,6(",",i0),",",a)') n, dxv, ldcum(n),          &
         loddraf(n), ktype(n), kcbot(n), kctop(n), idtop(n),             &
         hexw(prsi(n,nz+1))
    do k = 1, nz
      ! zlglac and zlude are a THIRD hazardous class the aliasing audit
      ! does not detect: intent(inout) whose FIRST use is a self-referential
      ! write.  :2887 is  plglac = pmfu*plglac  -- the RHS reads the
      ! incoming value, but "first use" classifies the statement as a
      ! write, so class 1 misses it, and class 2 filters to intent(out),
      ! so that misses it too.  Found by a failing parity test, not by the
      ! audit; the audit is extended to catch the shape.
      write(79,'(i0,",",f0.1,",",i0,20(",",a))') n, dxv, k,              &
           hexw(ztp1(n,k)),  hexw(zqp1(n,k)),  hexw(zqsat(n,k)),         &
           hexw(ztenh(n,k)), hexw(zqenh(n,k)), hexw(prsi(n,k)),          &
           hexw(prsl(n,k)),  hexw(pgeoh(n,k)), hexw(zmfu(n,k)),          &
           hexw(zmfd(n,k)),  hexw(zmfus(n,k)), hexw(zmfds(n,k)),         &
           hexw(zmfuq(n,k)), hexw(zmfdq(n,k)), hexw(zmful(n,k)),         &
           hexw(zdmfup(n,k)), hexw(zdmfdp(n,k)),                         &
           hexw(zlglac(n,k)), hexw(zlude(n,k)), hexw(pmfdde_rate(n,k))
    end do
  end do

  !*  7.0 final convective fluxes  (:822)
  call x_cuflxn(nc,nz,delt,ztp1,zqp1,zqsat,ztenh,zqenh,prsi,prsl,pgeoh, &
       slimsk,ldcum,kcbot,kctop,idtop,itopm2,ktype,loddraf,zmfu,zmfd,   &
       zmfus,zmfds,zmfuq,zmfdq,zmful,zlude,zdmfup,zdmfdp,zdpmel,zlglac, &
       prain,pmfdde_rate,pmflxr,pmflxs)

  ! ---- CAPTURE: cuflxn's outputs -------------------------------------
  ! pmflxr/pmflxs carry klev+1 entries; the surface slot is the scheme's
  ! actual surface rain and snow flux, so it is captured on the surface
  ! file rather than dropped off the end of the level file.
  do n = 1, nc
    write(80,'(i0,",",f0.1,",",i0,3(",",a))') n, dxv, idtop(n),          &
         hexw(prain(n)), hexw(pmflxr(n,nz+1)), hexw(pmflxs(n,nz+1))
    do k = 1, nz
      ! pqsen and plude are rewritten by cuflxn too -- :2984 and the else
      ! branch at :2915.  Without them here they would be transcribed and
      ! ungraded, which is the ptenh/pqenh mistake from the cuascn slice.
      write(81,'(i0,",",f0.1,",",i0,17(",",a))') n, dxv, k,              &
           hexw(zmfu(n,k)),   hexw(zmfd(n,k)),  hexw(zmfus(n,k)),        &
           hexw(zmfds(n,k)),  hexw(zmfuq(n,k)), hexw(zmfdq(n,k)),        &
           hexw(zmful(n,k)),  hexw(zlglac(n,k)),hexw(zdmfup(n,k)),       &
           hexw(zdmfdp(n,k)), hexw(zdpmel(n,k)),                         &
           hexw(pmflxr(n,k)), hexw(pmflxs(n,k)),                         &
           hexw(zqsat(n,k)),  hexw(zlude(n,k)), hexw(pmfdde_rate(n,k)), &
           hexw(pmfude_rate(n,k))
    end do
  end do

  ! some adjustments needed  (:833-915)
  do jl = 1, nc
    zmfs(jl) = 1.
    zmfuub(jl) = 0.
  end do
  do jk = 2, nz
    do jl = 1, nc
      if (loddraf(jl) /= 0 .and. jk >= idtop(jl)-1) then
        zmfmax = zmfu(jl,jk)*0.98
        if (zmfd(jl,jk)+zmfmax+1.e-15 < 0.) then
          zmfs(jl) = min(zmfs(jl),-zmfmax/zmfd(jl,jk))
        end if
      end if
    end do
  end do
  do jk = 2, nz
    do jl = 1, nc
      if (zmfs(jl) < 1. .and. jk >= idtop(jl)-1) then
        zmfd(jl,jk)=zmfd(jl,jk)*zmfs(jl)
        zmfds(jl,jk)=zmfds(jl,jk)*zmfs(jl)
        zmfdq(jl,jk)=zmfdq(jl,jk)*zmfs(jl)
        pmfdde_rate(jl,jk)=pmfdde_rate(jl,jk)*zmfs(jl)
        zmfuub(jl) = zmfuub(jl) - (1.-zmfs(jl))*zdmfdp(jl,jk)
        pmflxr(jl,jk+1) = pmflxr(jl,jk+1) + zmfuub(jl)
        zdmfdp(jl,jk)=zdmfdp(jl,jk)*zmfs(jl)
      end if
    end do
  end do
  do jk = 2, nz-1
    do jl = 1, nc
      if (loddraf(jl) /= 0 .and. jk >= idtop(jl)-1) then
        zerate = -zmfd(jl,jk) + zmfd(jl,jk-1) + pmfdde_rate(jl,jk)
        if (zerate < 0.) pmfdde_rate(jl,jk) = pmfdde_rate(jl,jk) - zerate
      end if
      if (ldcum(jl) /= 0 .and. jk >= kctop(jl)-1) then
        zerate = zmfu(jl,jk) - zmfu(jl,jk+1) + pmfude_rate(jl,jk)
        if (zerate < 0.) pmfude_rate(jl,jk) = pmfude_rate(jl,jk) - zerate
        zdmfup(jl,jk) = pmflxr(jl,jk+1) + pmflxs(jl,jk+1) - &
                        pmflxr(jl,jk) - pmflxs(jl,jk)
        zdmfdp(jl,jk) = 0.
      end if
    end do
  end do
  do jl = 1, nc
    if (loddraf(jl) /= 0) then
      jk = idtop(jl)
      ik = min(jk+1,nz)
      if (zmfdq(jl,jk) < 0.3*zmfdq(jl,ik)) zmfdq(jl,jk) = 0.3*zmfdq(jl,ik)
    end if
  end do
  do jk = 2, nz
    do jl = 1, nc
      if (ldcum(jl) /= 0 .and. jk >= kctop(jl)-1 .and. jk < kcbot(jl)) then
        zdz = delt*g_l/(prsi(jl,jk+1)-prsi(jl,jk))
        zmfa = zmfuq(jl,jk+1) + zmfdq(jl,jk+1) - zmfuq(jl,jk) - zmfdq(jl,jk) &
               + zmful(jl,jk+1) - zmful(jl,jk) + zdmfup(jl,jk)
        zmfa = (zmfa-zlude(jl,jk))*zdz
        if (zqp1(jl,jk)+zmfa < 0.) then
          zlude(jl,jk) = zlude(jl,jk) + 2.*(zqp1(jl,jk)+zmfa)/zdz
        end if
        if (zlude(jl,jk) < 0.) zlude(jl,jk) = 0.
      end if
      if (ldcum(jl) == 0) pmfude_rate(jl,jk) = 0.
      if (abs(zmfd(jl,jk-1)) < 1.0e-20) pmfdde_rate(jl,jk) = 0.
    end do
  end do
  do jl = 1, nc
    prsfc(jl) = pmflxr(jl,nz1)
    pssfc(jl) = pmflxs(jl,nz1)
  end do

  !*  8.0 tendencies for t and q  (:922)
  ! ---- CAPTURE: the adjustments block's exit  (:833-919) --------------
  ! Its INPUTS are cuflxn's outputs, already captured, so only the exit
  ! state is new -- and only the members cudtdqn's capture does not carry:
  ! pmfdde_rate, pmfude_rate, pmflxr/pmflxs and the surface fluxes.
  do n = 1, nc
    write(88,'(i0,",",f0.1,2(",",a))') n, dxv,                          &
         hexw(pmflxr(n,nz+1)), hexw(pmflxs(n,nz+1))
    do k = 1, nz
      write(89,'(i0,",",f0.1,",",i0,7(",",a))') n, dxv, k,              &
           hexw(zmfd(n,k)),   hexw(zmfds(n,k)), hexw(zmfdq(n,k)),       &
           hexw(zdmfdp(n,k)), hexw(zdmfup(n,k)),                        &
           hexw(pmfdde_rate(n,k)), hexw(pmfude_rate(n,k))
    end do
  end do

  ! ---- CAPTURE: cudtdqn's inputs, at its real call site ---------------
  ! ptte and pqte are SELF-REFERENTIAL accumulators (:3140-3141,
  ! ptent = ptent + zdtdt).  The audit's third report flagged both BEFORE
  ! this routine was ported -- the first time that tooling got ahead of a
  ! failure rather than behind one -- so their entry values are captured
  ! rather than discovered wrong.  pcte is conditionally written and is
  ! captured for the same reason.
  do n = 1, nc
    write(82,'(i0,",",f0.1,3(",",i0),",",a)') n, dxv, ldcum(n),         &
         kctop(n), itopm2, hexw(prsi(n,nz+1))
    do k = 1, nz
      write(83,'(i0,",",f0.1,",",i0,17(",",a))') n, dxv, k,             &
           hexw(prsi(n,k)),  hexw(ztp1(n,k)),  hexw(zlglac(n,k)),       &
           hexw(zlude(n,k)), hexw(zmfus(n,k)), hexw(zmfds(n,k)),        &
           hexw(zmfuq(n,k)), hexw(zmfdq(n,k)), hexw(zmful(n,k)),        &
           hexw(zdmfup(n,k)),hexw(zdmfdp(n,k)),hexw(zdpmel(n,k)),       &
           hexw(ptte(n,k)),  hexw(pqte(n,k)),  hexw(pcte(n,k)),         &
           hexw(zmfu(n,k)),  hexw(zmfd(n,k))
    end do
  end do

  call x_cudtdqn(nc,nz,itopm2,kctop,idtop,ldcum,loddraf,delt,prsi,pgeoh, &
       pgeo,ztp1,ztenh,zqp1,zqenh,zqsat,zlglac,zlude,zmfu,zmfd,zmfus,    &
       zmfds,zmfuq,zmfdq,zmful,zdmfup,zdmfdp,zdpmel,ptte,pqte,pcte)

  ! ---- CAPTURE: cudtdqn's outputs ------------------------------------
  do n = 1, nc
    do k = 1, nz
      write(84,'(i0,",",f0.1,",",i0,3(",",a))') n, dxv, k,              &
           hexw(ptte(n,k)), hexw(pqte(n,k)), hexw(pcte(n,k))
    end do
  end do

  ! ---- CAPTURE: the momentum-profile block's entry  (:927-995) --------
  ! CAPTURE FIRST.  Its OUTPUT (zuu/zvu/zud/zvd) is cududvn's captured
  ! entry at :1025, but its INPUT is not any existing capture's exit --
  ! zuu/zvu arrive from cuinin, 450 lines and eight stages earlier, and
  ! assuming that survives is instance 8 exactly.
  do n = 1, nc
    write(93,'(i0,",",f0.1,6(",",i0))') n, dxv, ldcum(n), ktype(n),     &
         kcbot(n), kctop(n), kdpl(n), idtop(n)
    do k = 1, nz
      write(94,'(i0,",",f0.1,",",i0,9(",",a))') n, dxv, k,              &
           hexw(zmfu(n,k)),  hexw(zmfd(n,k)),  hexw(pum1(n,k)),         &
           hexw(pvm1(n,k)),  hexw(zuu(n,k)),   hexw(zvu(n,k)),          &
           hexw(zud(n,k)),   hexw(zvd(n,k)),   hexw(pmfude_rate(n,k))
    end do
  end do

  !*  9.0 tendencies for u and v  (lmfdudv is .true.)  (:927-1057)
  do jk = nz-1, 2, -1
    ik = jk + 1
    do jl = 1, nc
      if (ldcum(jl) /= 0) then
        if (jk == kcbot(jl) .and. ktype(jl) < 3) then
          ikb = kdpl(jl)
          zuu(jl,jk) = pum1(jl,ikb-1)
          zvu(jl,jk) = pvm1(jl,ikb-1)
        else if (jk == kcbot(jl) .and. ktype(jl) == 3) then
          zuu(jl,jk) = pum1(jl,jk-1)
          zvu(jl,jk) = pvm1(jl,jk-1)
        end if
        if (jk < kcbot(jl) .and. jk >= kctop(jl)) then
          ! momtrans is 2, a PARAMETER, so the if(momtrans == 1) arm
          ! at 943-955 is dead and this pressure-gradient else arm is
          ! the live one.  The range is named so the omission cannot
          ! read as a transcription slip.
          zdz = -pgcoef_l*0.5*(zmfu(jl,ik)*(pum1(jl,ik)-pum1(jl,jk)) + &
                zmfu(jl,jk)*(pum1(jl,jk)-pum1(jl,jk-1)))
          zdp = -pgcoef_l*0.5*(zmfu(jl,ik)*(pvm1(jl,ik)-pvm1(jl,jk)) + &
                zmfu(jl,jk)*(pvm1(jl,jk)-pvm1(jl,jk-1)))
          zerate = zmfu(jl,jk) - zmfu(jl,ik) + pmfude_rate(jl,jk)
          zmfa = 1./max(1.e-10,zmfu(jl,jk))
          zuu(jl,jk) = (zuu(jl,ik)*zmfu(jl,ik) + zerate*pum1(jl,jk) - &
                        pmfude_rate(jl,jk)*zuu(jl,ik) + zdz)*zmfa
          zvu(jl,jk) = (zvu(jl,ik)*zmfu(jl,ik) + zerate*pvm1(jl,jk) - &
                        pmfude_rate(jl,jk)*zvu(jl,ik) + zdp)*zmfa
        end if
      end if
    end do
  end do
  do jk = 3, nz
    ik = jk - 1
    do jl = 1, nc
      if (ldcum(jl) /= 0) then
        if (jk == idtop(jl)) then
          zud(jl,jk) = 0.5*(zuu(jl,jk)+pum1(jl,ik))
          zvd(jl,jk) = 0.5*(zvu(jl,jk)+pvm1(jl,ik))
        else if (jk > idtop(jl)) then
          zerate = -zmfd(jl,jk) + zmfd(jl,ik) + pmfdde_rate(jl,jk)
          zmfa = 1./min(-1.e-10,zmfd(jl,jk))
          zud(jl,jk) = (zud(jl,ik)*zmfd(jl,ik) - zerate*pum1(jl,ik) + &
                        pmfdde_rate(jl,jk)*zud(jl,ik))*zmfa
          zvd(jl,jk) = (zvd(jl,ik)*zmfd(jl,ik) - zerate*pvm1(jl,ik) + &
                        pmfdde_rate(jl,jk)*zvd(jl,ik))*zmfa
        end if
      end if
    end do
  end do
  ! ---- CAPTURE: the MOMENTUM rescale's inputs  (:996-1016) ------------
  ! CAPTURE FIRST, per the inverted default in docs/ntiedtke/STANDING-RULES.md.
  ! zmfu/zmfd here MIGHT be the adjust block's outputs unchanged -- :927-995
  ! runs in between -- and deciding that by reading rather than recording is
  ! exactly what was wrong six times.  One write statement settles it.
  !
  ! This block is the ONLY consumer of zcons (1/(g*dt)) in the whole
  ! scheme.  Every other mass-flux cap uses zcons2 (3/(g*dt)), one
  ! character away.  See tests/test_ntiedtke_constant_family.py.
  do n = 1, nc
    do k = 1, nz
      write(90,'(i0,",",f0.1,",",i0,3(",",a))') n, dxv, k,              &
           hexw(zmfu(n,k)), hexw(zmfd(n,k)), hexw(prsi(n,k))
    end do
  end do

  do jl = 1, nc
    zmfs(jl) = 1.
  end do
  do jk = 2, nz
    do jl = 1, nc
      if (ldcum(jl) /= 0 .and. jk >= kctop(jl)-1) then
        zmfmax = (prsi(jl,jk)-prsi(jl,jk-1))*zcons
        if (zmfu(jl,jk) > zmfmax .and. jk >= kctop(jl)) then
          zmfs(jl) = min(zmfs(jl),zmfmax/zmfu(jl,jk))
        end if
      end if
    end do
  end do
  do jk = 1, nz
    do jl = 1, nc
      zmfuus(jl,jk) = zmfu(jl,jk)
      zmfdus(jl,jk) = zmfd(jl,jk)
      if (ldcum(jl) /= 0 .and. jk >= kctop(jl)-1) then
        zmfuus(jl,jk) = zmfu(jl,jk)*zmfs(jl)
        zmfdus(jl,jk) = zmfd(jl,jk)*zmfs(jl)
      end if
    end do
  end do
  do jk = 1, nz
    do jl = 1, nc
      ztenu(jl,jk) = pvom(jl,jk)
      ztenv(jl,jk) = pvol(jl,jk)
    end do
  end do
  ! ---- CAPTURE: cududvn's inputs, at its real call site ---------------
  ! pvom/pvol are self-referential accumulators (:3245-3246), flagged by the
  ! audit's third report before this routine was ported.  Unlike ptte/pqte
  ! they are seeded ZERO by cu_ntiedtke_run:258-259, so accumulate and
  ! replace coincide here -- captured anyway, because "measured zero" and
  ! "assumed zero" are the distinction this port keeps paying for.
  !
  ! zmfuus/zmfdus are the SCALED updraft/downdraft mass fluxes, not zmfu and
  ! zmfd: :833-915 rescales them, so feeding the unscaled pair is wrong.
  do n = 1, nc
    write(85,'(i0,",",f0.1,4(",",i0),",",a)') n, dxv, ldcum(n), ktype(n), &
         kcbot(n), itopm2, hexw(prsi(n,nz+1))
    do k = 1, nz
      write(86,'(i0,",",f0.1,",",i0,11(",",a))') n, dxv, k,              &
           hexw(prsi(n,k)),   hexw(pum1(n,k)), hexw(pvm1(n,k)),          &
           hexw(zmfuus(n,k)), hexw(zmfdus(n,k)),                         &
           hexw(zuu(n,k)),    hexw(zud(n,k)),  hexw(zvu(n,k)),           &
           hexw(zvd(n,k)),    hexw(pvom(n,k)), hexw(pvol(n,k))
    end do
  end do

  call x_cududvn(nc,nz,itopm2,ktype,kcbot,kctop,ldcum,delt,prsi,pum1, &
       pvm1,zmfuus,zmfdus,zuu,zud,zvu,zvd,pvom,pvol)
  ! ---- CAPTURE: cududvn's outputs ------------------------------------
  do n = 1, nc
    do k = 1, nz
      write(87,'(i0,",",f0.1,",",i0,2(",",a))') n, dxv, k,               &
           hexw(pvom(n,k)), hexw(pvol(n,k))
    end do
  end do

  ! ---- CAPTURE: the KE-dissipation block's entry  (:1030-1056) --------
  ! CAPTURE FIRST.  ztenu/ztenv are pvom/pvol BEFORE cududvn and pvom/pvol
  ! are after, so cududvn's own in/out captures would serve -- which is the
  ! shape that has been wrong seven times, so they are recorded here.
  do n = 1, nc
    do k = 1, nz
      write(95,'(i0,",",f0.1,",",i0,7(",",a))') n, dxv, k,              &
           hexw(ztenu(n,k)), hexw(ztenv(n,k)), hexw(pvom(n,k)),         &
           hexw(pvol(n,k)),  hexw(pum1(n,k)),  hexw(pvm1(n,k)),         &
           hexw(ptte(n,k))
    end do
  end do

  do jl = 1, nc
    zsum12(jl) = 0.
    zsum22(jl) = 0.
  end do
  do jk = 1, nz
    do jl = 1, nc
      zuv2(jl,jk) = 0.
      if (ldcum(jl) /= 0 .and. jk >= kctop(jl)-1) then
        zdz = (prsi(jl,jk+1)-prsi(jl,jk))
        zduten = pvom(jl,jk) - ztenu(jl,jk)
        zdvten = pvol(jl,jk) - ztenv(jl,jk)
        zuv2(jl,jk) = sqrt(zduten**2+zdvten**2)
        zsum22(jl) = zsum22(jl) + zuv2(jl,jk)*zdz
        zsum12(jl) = zsum12(jl) - &
          (pum1(jl,jk)*zduten+pvm1(jl,jk)*zdvten)*zdz
      end if
    end do
  end do
  do jk = 1, nz
    do jl = 1, nc
      if (ldcum(jl) /= 0 .and. jk >= kctop(jl)-1) then
        ztdis = rcpd_l*zsum12(jl)*zuv2(jl,jk)/max(1.e-15,zsum22(jl))
        ptte(jl,jk) = ptte(jl,jk) + ztdis
      end if
    end do
  end do

  ! ---- CAPTURE: the KE-dissipation block's exit -----------------------
  do n = 1, nc
    do k = 1, nz
      write(96,'(i0,",",f0.1,",",i0,",",a)') n, dxv, k, hexw(ptte(n,k))
    end do
  end do

  ! Section 10 (:1060-1082) is guarded by (.not. lmfscv .or. .not. lmfpen).
  ! Both are .true. parameters, so it is dead.  Transcribed as a no-op so a
  ! future WRF that flips either one breaks the parity gate rather than
  ! silently changing the answer.
  ! (llo2 is all zero here by construction, so the loop body never runs.)

end subroutine replicate_cumastrn
