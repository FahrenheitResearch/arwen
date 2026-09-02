program run_cu_ntiedtke
  ! Stage A of the New Tiedtke oracle: pin cu_ntiedtke_driver, the WRF entry
  ! point (module_cu_ntiedtke.F:158-...), over the case x dx fixture.
  !
  ! This is the RIGHT fixture boundary -- it is what module_cumulus_driver.F
  ! :1384-1407 actually calls -- and it is a black box for a failing port.
  ! ktype, ldcum, kcbot, kctop, kdpl, zmfub, zmfub1, ztauc, ztau, zcape1,
  ! zcape2, zheat and wup are all cumastrn locals and none of them leave the
  ! driver.  A port whose RTHCUTEN is wrong cannot tell from this file alone
  ! whether it lost the parcel in cutypen's trigger, in the closure, or in
  ! cuflxn.  That is what the later decomposition harness is for; this one
  ! establishes the answer the decomposition has to reproduce.
  !
  ! Writes three files:
  !   nt-levels.csv     every per-level input the driver saw and every
  !                     per-level output it wrote
  !   nt-surface.csv    the scalars, plus RAINCV/PRATEC/CU_ACT_FLAG and the
  !                     two scale factors as a function of dx
  !   nt-isolation.csv  output words that differ bitwise between the packed
  !                     18-column slab and the same column run alone
  !
  ! Every float is written as its raw IEEE-754 word in hex.  A decimal
  ! rendering is a lossy view of the thing being pinned, and the bar here is
  ! max_ulp == 0.
  use iso_fortran_env, only: int32
  use ccpp_kind_types, only: kind_phys
  use module_cu_ntiedtke, only: cu_ntiedtke_driver
  use nt_cases
  implicit none

  integer, parameter :: nz  = nt_nz
  integer, parameter :: nc  = nt_ncase
  integer, parameter :: ndx = nt_ndx

  ! One j row, nc columns in i, nz half levels.
  integer, parameter :: ids = 1, ide = nc + 1, jds = 1, jde = 2
  integer, parameter :: kds = 1, kde = nz + 1
  integer, parameter :: ims = 1, ime = nc,     jms = 1, jme = 1
  integer, parameter :: kms = 1, kme = nz + 1
  integer, parameter :: its = 1, ite = nc,     jts = 1, jte = 1
  integer, parameter :: kts = 1, kte = nz

  real(kind=kind_phys), dimension(ims:ime, kms:kme, jms:jme) :: &
       u3d, v3d, w, t3d, qv3d, qc3d, qi3d, pi3d, rho3d, &
       dz8w, pcps, p8w, qvften, thften, &
       rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten
  real(kind=kind_phys), dimension(ims:ime, jms:jme) :: &
       raincv, pratec, qfx, hfx, xland, dx
  logical, dimension(ims:ime, jms:jme) :: cu_act_flag

  ! single-column mirrors, for the isolation arm
  real(kind=kind_phys), dimension(1, kms:kme, 1) :: &
       s_u3d, s_v3d, s_w, s_t3d, s_qv3d, s_qc3d, s_qi3d, s_pi3d, s_rho3d, &
       s_dz8w, s_pcps, s_p8w, s_qvften, s_thften, &
       s_rthcuten, s_rqvcuten, s_rqccuten, s_rqicuten, s_rucuten, s_rvcuten
  real(kind=kind_phys), dimension(1, 1) :: &
       s_raincv, s_pratec, s_qfx, s_hfx, s_xland, s_dx
  logical, dimension(1, 1) :: s_cu_act_flag

  ! WRF-order column scratch
  real, dimension(nz)     :: b_t, b_qv, b_qc, b_qi, b_u, b_v
  real, dimension(nz)     :: b_pcps, b_dz, b_rho, b_pi, b_qvften, b_thften
  real, dimension(nz + 1) :: b_p8w, b_w
  real :: b_xland, b_hfx, b_qfx

  real(kind=kind_phys), parameter :: dt = 60.0_kind_phys
  integer, parameter :: stepcu = 1
  integer, parameter :: itimestep = 2      ! > 1 so qvften/thften are READ

  ! THE f_* FLAGS ARE NOT OPTIONAL IN PRACTICE, whatever the declaration
  ! says.  module_cu_ntiedtke.F:253-254 and :263-264 read
  !
  !     if(present(rqccuten))then
  !        if(f_qc) then
  !
  ! -- guarding on rqccuten and then dereferencing f_qc with no present()
  ! check of its own.  Omit f_qc while passing rqccuten and the driver
  ! segfaults (measured here before these were added).  It never fires in
  ! WRF because module_cumulus_driver.F:1402-1403 always passes all five,
  ! so this is a latent bug inside the pinned boundary, not a live one.
  ! The fixture reproduces WRF's call site rather than the declaration:
  ! all five true, which is what an mp_physics = 8 (Thompson) run has.
  logical, parameter :: p_f_qv = .true., p_f_qc = .true., p_f_qr = .true.
  logical, parameter :: p_f_qi = .true., p_f_qs = .true.

  character(len=256) :: errmsg
  integer :: errflg
  integer :: n, m, k, ndiff
  real :: sf, sf2, dxv

  ! ---- the precision gate -------------------------------------------------
  ! kind_phys is single on any default WRF build, and the whole float32 port
  ! rests on that.  v4.6.1's phys/ccpp_kind_types.F gates on
  ! `#if ( RWORDSIZE == 4 )`, and cpp evaluates an UNDEFINED identifier as 0,
  ! so a build that forgets -DRWORDSIZE=4 silently takes selected_real_kind(12)
  ! and generates a DOUBLE-precision oracle that compiles clean and looks
  ! plausible.  build.sh passes the define; this refuses to run without it.
  if (kind(1.0_kind_phys) /= 4) then
     write(*, '(a,i0,a)') 'FATAL: kind_phys is ', kind(1.0_kind_phys), &
          ' bytes, not 4.  Rebuild with -DRWORDSIZE=4 (v4.6.1 spelling).'
     stop 3
  end if

  call nt_build_case_table()

  open(unit=11, file='nt-levels.csv',    status='replace', action='write')
  open(unit=12, file='nt-surface.csv',   status='replace', action='write')
  open(unit=13, file='nt-isolation.csv', status='replace', action='write')

  write(11, '(a)') 'case,dx,k,t3d,qv3d,qc3d,qi3d,u3d,v3d,pcps,p8w,dz8w,' // &
       'rho3d,pi3d,w,qvften,thften,rthcuten,rqvcuten,rqccuten,rqicuten,' // &
       'rucuten,rvcuten'
  write(12, '(a)') 'case,dx,scale_fac,scale_fac2,xland,hfx,qfx,psfc,' // &
       'raincv,pratec,cu_act_flag'
  write(13, '(a)') 'case,dx,differing_words'

  do m = 1, ndx
    dxv = nt_dxsweep(m)

    ! scale_fac / scale_fac2 recomputed here exactly as cu_ntiedtke.F90
    ! :230-238 forms them, so the fixture records the dx dependence itself
    ! and a port can be graded on the factor before it is graded on any
    ! physics that consumes it.
    if (dxv < 15000.0) then
      sf  = (1.06133 + log(15000.0 / dxv)) ** 3
      sf2 = sf ** 0.5
    else
      sf  = 1.0 + 1.33e-5 * dxv
      sf2 = 1.0
    end if

    ! ---- pack the slab --------------------------------------------------
    do n = 1, nc
      call nt_build_column(n, nz, b_t, b_qv, b_qc, b_qi, b_u, b_v, &
                           b_pcps, b_p8w, b_dz, b_rho, b_pi, b_w, &
                           b_qvften, b_thften, b_xland, b_hfx, b_qfx)
      do k = 1, nz
        t3d(n,k,1)    = b_t(k)
        qv3d(n,k,1)   = b_qv(k)
        qc3d(n,k,1)   = b_qc(k)
        qi3d(n,k,1)   = b_qi(k)
        u3d(n,k,1)    = b_u(k)
        v3d(n,k,1)    = b_v(k)
        pcps(n,k,1)   = b_pcps(k)
        dz8w(n,k,1)   = b_dz(k)
        rho3d(n,k,1)  = b_rho(k)
        pi3d(n,k,1)   = b_pi(k)
        qvften(n,k,1) = b_qvften(k)
        thften(n,k,1) = b_thften(k)
      end do
      do k = 1, nz + 1
        p8w(n,k,1) = b_p8w(k)
        w(n,k,1)   = b_w(k)
      end do
      xland(n,1) = b_xland
      hfx(n,1)   = b_hfx
      qfx(n,1)   = b_qfx
      dx(n,1)    = dxv
    end do

    raincv = 0.0;  pratec = 0.0;  cu_act_flag = .false.
    rthcuten = 0.0; rqvcuten = 0.0; rqccuten = 0.0
    rqicuten = 0.0; rucuten  = 0.0; rvcuten  = 0.0

    call cu_ntiedtke_driver( &
         dt=dt, itimestep=itimestep, stepcu=stepcu, hfx=hfx, &
         raincv=raincv, pratec=pratec, qfx=qfx, &
         u3d=u3d, v3d=v3d, w=w, t3d=t3d, pi3d=pi3d, rho3d=rho3d, &
         qv3d=qv3d, qc3d=qc3d, qi3d=qi3d, &
         dz8w=dz8w, pcps=pcps, p8w=p8w, xland=xland, dx=dx, &
         cu_act_flag=cu_act_flag, &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
         ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, &
         qvften=qvften, thften=thften, &
         f_qv=p_f_qv, f_qc=p_f_qc, f_qr=p_f_qr, &
         f_qi=p_f_qi, f_qs=p_f_qs, &
         rthcuten=rthcuten, rqvcuten=rqvcuten, &
         rqccuten=rqccuten, rqicuten=rqicuten, &
         rucuten=rucuten, rvcuten=rvcuten, &
         grav=real(nt_g,kind_phys), xlf=real(nt_xlf,kind_phys), &
         xls=real(nt_xls,kind_phys), xlv=real(nt_xlv,kind_phys), &
         rd=real(nt_rd,kind_phys), rv=real(nt_rv,kind_phys), &
         cp=real(nt_cp,kind_phys), &
         errmsg=errmsg, errflg=errflg)

    if (errflg /= 0) then
      write(*,'(a,i0,a,a)') 'driver errflg=', errflg, ' : ', trim(errmsg)
      stop 4
    end if

    ! ---- record ---------------------------------------------------------
    do n = 1, nc
      do k = 1, nz
        write(11,'(i0,",",f0.1,",",i0,20(",",a))') n, dxv, k, &
             hexw(t3d(n,k,1)),    hexw(qv3d(n,k,1)),  hexw(qc3d(n,k,1)),  &
             hexw(qi3d(n,k,1)),   hexw(u3d(n,k,1)),   hexw(v3d(n,k,1)),   &
             hexw(pcps(n,k,1)),   hexw(p8w(n,k,1)),   hexw(dz8w(n,k,1)),  &
             hexw(rho3d(n,k,1)),  hexw(pi3d(n,k,1)),  hexw(w(n,k,1)),     &
             hexw(qvften(n,k,1)), hexw(thften(n,k,1)),                    &
             hexw(rthcuten(n,k,1)), hexw(rqvcuten(n,k,1)),                &
             hexw(rqccuten(n,k,1)), hexw(rqicuten(n,k,1)),                &
             hexw(rucuten(n,k,1)),  hexw(rvcuten(n,k,1))
      end do
      write(12,'(i0,",",f0.1,8(",",a),",",l1)') n, dxv, &
           hexw(real(sf,kind_phys)), hexw(real(sf2,kind_phys)), &
           hexw(xland(n,1)), hexw(hfx(n,1)), hexw(qfx(n,1)), &
           hexw(p8w(n,1,1)), hexw(raincv(n,1)), hexw(pratec(n,1)), &
           cu_act_flag(n,1)
    end do

    ! ---- isolation: the same column, alone in its own tile --------------
    ! Grell-Freitas' fixture found real packing sensitivity here; New
    ! Tiedtke has no horizontal coupling at all (no jl+/-1 access, no
    ! reduction over the jl dimension anywhere in cu_ntiedtke.F90), so the
    ! expected value is 0 differing words on every row.  That is a CLAIM,
    ! and this is the measurement that either backs it or retires it.
    do n = 1, nc
      call nt_build_column(n, nz, b_t, b_qv, b_qc, b_qi, b_u, b_v, &
                           b_pcps, b_p8w, b_dz, b_rho, b_pi, b_w, &
                           b_qvften, b_thften, b_xland, b_hfx, b_qfx)
      do k = 1, nz
        s_t3d(1,k,1)=b_t(k);      s_qv3d(1,k,1)=b_qv(k)
        s_qc3d(1,k,1)=b_qc(k);    s_qi3d(1,k,1)=b_qi(k)
        s_u3d(1,k,1)=b_u(k);      s_v3d(1,k,1)=b_v(k)
        s_pcps(1,k,1)=b_pcps(k);  s_dz8w(1,k,1)=b_dz(k)
        s_rho3d(1,k,1)=b_rho(k);  s_pi3d(1,k,1)=b_pi(k)
        s_qvften(1,k,1)=b_qvften(k); s_thften(1,k,1)=b_thften(k)
      end do
      do k = 1, nz + 1
        s_p8w(1,k,1)=b_p8w(k);    s_w(1,k,1)=b_w(k)
      end do
      s_xland(1,1)=b_xland; s_hfx(1,1)=b_hfx; s_qfx(1,1)=b_qfx
      s_dx(1,1)=dxv
      s_raincv=0.0; s_pratec=0.0; s_cu_act_flag=.false.
      s_rthcuten=0.0; s_rqvcuten=0.0; s_rqccuten=0.0
      s_rqicuten=0.0; s_rucuten=0.0;  s_rvcuten=0.0

      call cu_ntiedtke_driver( &
           dt=dt, itimestep=itimestep, stepcu=stepcu, hfx=s_hfx, &
           raincv=s_raincv, pratec=s_pratec, qfx=s_qfx, &
           u3d=s_u3d, v3d=s_v3d, w=s_w, t3d=s_t3d, pi3d=s_pi3d, &
           rho3d=s_rho3d, qv3d=s_qv3d, qc3d=s_qc3d, qi3d=s_qi3d, &
           dz8w=s_dz8w, pcps=s_pcps, p8w=s_p8w, xland=s_xland, dx=s_dx, &
           cu_act_flag=s_cu_act_flag, &
           ids=1, ide=2, jds=1, jde=2, kds=kds, kde=kde, &
           ims=1, ime=1, jms=1, jme=1, kms=kms, kme=kme, &
           its=1, ite=1, jts=1, jte=1, kts=kts, kte=kte, &
           qvften=s_qvften, thften=s_thften, &
           f_qv=p_f_qv, f_qc=p_f_qc, f_qr=p_f_qr, &
           f_qi=p_f_qi, f_qs=p_f_qs, &
           rthcuten=s_rthcuten, rqvcuten=s_rqvcuten, &
           rqccuten=s_rqccuten, rqicuten=s_rqicuten, &
           rucuten=s_rucuten, rvcuten=s_rvcuten, &
           grav=real(nt_g,kind_phys), xlf=real(nt_xlf,kind_phys), &
           xls=real(nt_xls,kind_phys), xlv=real(nt_xlv,kind_phys), &
           rd=real(nt_rd,kind_phys), rv=real(nt_rv,kind_phys), &
           cp=real(nt_cp,kind_phys), &
           errmsg=errmsg, errflg=errflg)

      ndiff = 0
      do k = 1, nz
        if (wne(s_rthcuten(1,k,1), rthcuten(n,k,1))) ndiff = ndiff + 1
        if (wne(s_rqvcuten(1,k,1), rqvcuten(n,k,1))) ndiff = ndiff + 1
        if (wne(s_rqccuten(1,k,1), rqccuten(n,k,1))) ndiff = ndiff + 1
        if (wne(s_rqicuten(1,k,1), rqicuten(n,k,1))) ndiff = ndiff + 1
        if (wne(s_rucuten(1,k,1),  rucuten(n,k,1)))  ndiff = ndiff + 1
        if (wne(s_rvcuten(1,k,1),  rvcuten(n,k,1)))  ndiff = ndiff + 1
      end do
      if (wne(s_raincv(1,1), raincv(n,1))) ndiff = ndiff + 1
      if (wne(s_pratec(1,1), pratec(n,1))) ndiff = ndiff + 1
      write(13,'(i0,",",f0.1,",",i0)') n, dxv, ndiff
    end do
  end do

  close(11); close(12); close(13)
  write(*,'(a)') 'run_cu_ntiedtke OK'

end program run_cu_ntiedtke
