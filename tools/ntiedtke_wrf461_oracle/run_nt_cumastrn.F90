program run_nt_cumastrn
  ! THE CAPTURE ARCHITECTURE.  A statement-order replication of cumastrn's
  ! body (:460-1085) that calls the REAL private routines at every step,
  ! captures state at every call boundary, and proves itself against a real
  ! cu_ntiedtke_run call.
  !
  ! ===================================================================
  ! WHY THIS EXISTS
  ! ===================================================================
  ! Every harness before this one SYNTHESISED the inputs of the routine it
  ! graded -- reconstructing what cumastrn would have passed.  That was
  ! wrong three times in five slices:
  !
  !   cutypen : passed FRESH arrays where cumastrn passes live ones
  !   slice 4a: skipped cumastrn:500-541, so pmfub was 0
  !   slice 4b: never captured paph at the surface interface
  !
  ! Each time the oracle and the NumPy mirror agreed, because they agreed
  ! with each other about a state WRF never visits.  max_ulp == 0 is
  ! structurally blind to it: there is nothing for the reference to
  ! disagree with.  Three different mistakes, one root cause.
  !
  ! Capturing at the real call site removes the class.  Interposition would
  ! have been the cheap way -- probe_interposition.sh tested it and it is
  ! DEAD: gfortran binds cumastrn -> cuentrn directly and never goes
  ! through the PLT, so LD_PRELOAD never sees the call.  (-fPIC was
  ! measured bit-identical, so that half was fine; the binding is the
  ! blocker.)  So the fallback is replication, which is what
  ! run_gf_stages.F90 already does for Grell-Freitas.
  !
  ! ===================================================================
  ! WHAT THE SELF-PROOF DOES AND DOES NOT ESTABLISH
  ! ===================================================================
  ! The proof is: run this replication, apply cu_ntiedtke_run's own
  ! post-processing, and compare against a real cu_ntiedtke_run call on the
  ! same column.  Zero differing words is the only accepted value.
  !
  ! That proves the replication CONVERGES TO THE SAME ANSWER.  It does NOT,
  ! by itself, prove that every intermediate capture is what cumastrn held
  ! -- a replication could differ internally and still land in the same
  ! place.  What carries the intermediate correctness is (a) statement-order
  ! fidelity to :460-1085 and (b) every callee being the real globalized
  ! procedure, not a transcription.  That is a strong argument.  It is an
  ! ARGUMENT, not a measurement, and with interposition dead no measurement
  ! is available.  Read the captures accordingly.
  use iso_fortran_env, only: int32
  use ccpp_kind_types, only: kind_phys
  use cu_ntiedtke, only: cu_ntiedtke_run, cu_ntiedtke_init
  use nt_cases
  implicit none

  ! ---- the real private routines, by mangled symbol -------------------
  ! build.sh globalizes these with objcopy --globalize-symbol, which flips
  ! a binding bit and leaves .text byte-identical (asserted every build).
  ! bind(C) names the symbol and nothing else; every dummy is explicit
  ! shape, so the ABI is the one gfortran already emits.  LOGICAL(4) is
  ! declared integer(4) here: bind(C) would map Fortran LOGICAL to _Bool,
  ! which is not the callee's ABI, while LOGICAL(4) is stored as int32 0/1.
  ! Every x_* interface lives in nt_cases, declared once.  Two copies
  ! of x_cuinin and x_cutypen used to sit in this directory with
  ! different line breaks and nothing comparing them.

  integer, parameter :: nz = nt_nz, nc = nt_ncase, ndx = nt_ndx
  integer, parameter :: nz1 = nz + 1

  ! ---- cu_ntiedtke_run's own state (its :240-320) ---------------------
  real(kind=kind_phys), dimension(nc,nz) :: pum1,pvm1,ztt,ptte,pqte
  real(kind=kind_phys), dimension(nc,nz) :: pvom,pvol,pverv,pgeo,zqq,pcte
  real(kind=kind_phys), dimension(nc,nz) :: ztp1,zqp1,ztu,zqu,zlu,zlude
  real(kind=kind_phys), dimension(nc,nz) :: zmfu,zmfd,zqsat
  real(kind=kind_phys), dimension(nc,nz1) :: pgeoh
  real(kind=kind_phys), dimension(nc) :: pqhfl,prsfc,pssfc,phhfl,zrain
  integer, dimension(nc) :: icbot,ictop,ktype
  integer, dimension(nc) :: locum

  ! ---- cumastrn's locals ----------------------------------------------
  integer, dimension(nc) :: kdpl,idtop,ictop0,ilwmin,kcbot,kctop,loddraf
  integer, dimension(nc) :: ldcum, llo2
  integer, dimension(nc,nz) :: ilab
  real(kind=kind_phys), dimension(nc) :: zmfs,zsfl,zcape,zcape1,zcape2
  real(kind=kind_phys), dimension(nc) :: ztauc,ztaubl,zheat,wup,zdqcv
  real(kind=kind_phys), dimension(nc) :: wbase,zmfuub,upbl,zhcbase,zmfub
  real(kind=kind_phys), dimension(nc) :: zmfub1,zdhpbl,zmfuvb,zsum12,zsum22
  real(kind=kind_phys), dimension(nc) :: zrfl,prain
  real(kind=kind_phys), dimension(nc,nz) :: pmfude_rate,pmfdde_rate,zdpmel
  real(kind=kind_phys), dimension(nc,nz) :: zmfuus,zmfdus,zuv2,ztenu,ztenv
  real(kind=kind_phys), dimension(nc,nz) :: ztenh,zqenh,zqsenh,ztd,zqd
  real(kind=kind_phys), dimension(nc,nz) :: zmfus,zmfds,zmfuq,zmfdq
  real(kind=kind_phys), dimension(nc,nz) :: zdmfup,zdmfdp,zmful
  real(kind=kind_phys), dimension(nc,nz) :: zuu,zvu,zud,zvd,zlglac
  real(kind=kind_phys), dimension(nc,nz1) :: pmflxr,pmflxs

  ! ---- the prep arrays (scheme order) ---------------------------------
  real(kind=kind_phys), dimension(nc,nz) :: prsl,ghtl,omg
  real(kind=kind_phys), dimension(nc,nz) :: uf,vf,tf,qvf,qcf,qif
  real(kind=kind_phys), dimension(nc,nz) :: qvftenz,thftenz
  real(kind=kind_phys), dimension(nc,nz1) :: prsi,ghti
  ! WRF-order staging for the REAL cu_ntiedtke_pre_run.  zi/zl/dotv were
  ! pre_run's LOCALS, hand-copied here; they belong to the routine.
  real(kind=kind_phys), dimension(nc)     :: s_xland
  real(kind=kind_phys), dimension(nc,nz)  :: s_t,s_qv,s_qc,s_qi,s_u,s_v
  real(kind=kind_phys), dimension(nc,nz)  :: s_pres,s_dz,s_rho
  real(kind=kind_phys), dimension(nc,nz)  :: s_qvften,s_thften
  real(kind=kind_phys), dimension(nc,nz1) :: s_presi,s_w
  integer :: s_im,s_kx,s_kx1,s_errflg,jj
  character(kind=c_char) :: s_errmsg(256)
  character(len=256) :: s_errmsg_s
  integer, dimension(nc) :: slimsk
  real(kind=kind_phys), dimension(nc) :: dx_hv,hfx_hv,qfx_hv,rn

  ! ---- the control: a real cu_ntiedtke_run ----------------------------
  real(kind=kind_phys), dimension(nc,nz) :: c_pu,c_pv,c_pt,c_pqv,c_pqc,c_pqi
  real(kind=kind_phys), dimension(nc) :: c_rn
  ! and our own post-processed answer
  real(kind=kind_phys), dimension(nc,nz) :: r_pt,r_pqv,r_pqc,r_pqi,r_pu,r_pv
  real(kind=kind_phys), dimension(nc) :: r_rn

  real, dimension(nz) :: b_t,b_qv,b_qc,b_qi,b_u,b_v
  real, dimension(nz) :: b_pcps,b_dz,b_rho,b_pi,b_qvften,b_thften
  real, dimension(nz1) :: b_p8w,b_w
  real :: b_xland,b_hfx,b_qfx,dxv

  real(kind=kind_phys), parameter :: dtc = 60.0_kind_phys
  integer, parameter :: stepcu = 1, itimestep = 2
  real(kind=kind_phys) :: delt,tt,zew,zqs,zcor,vtmpc1_l,rcpd_l,zrg_l
  real(kind=kind_phys) :: zcons,zcons2,zmfmax,zqumqe,zdqmin,zdh,zpbmpt
  real(kind=kind_phys) :: zdz,zdp,wspeed,zfac,zeps,ztau,zerate,zmfa
  real(kind=kind_phys) :: zduten,zdvten,ztdis,alv_l,cpd_l,g_l,rd_l
  real(kind=kind_phys) :: scale_fac(nc), scale_fac2(nc), dxref
  ! cu_ntiedtke_common:35 -- a parameter, so a literal here.
  real(kind=kind_phys), parameter :: pgcoef_l = 0.7
  character(len=256) :: errmsg
  integer :: errflg,n,m,k,pp,zz,ik,ikb,ikt,itopm2,icum,jk,jl,total_bad,nd

  if (kind(1.0_kind_phys) /= 4) stop 3
  call nt_build_case_table()
  total_bad = 0

  open(unit=61,file='nt-cumastrn-consistency.csv',status='replace')
  open(unit=62,file='nt-cuascn-in-levels.csv',status='replace')
  open(unit=63,file='nt-cuascn-out-levels.csv',status='replace')
  open(unit=64,file='nt-cuascn-surface.csv',status='replace')
  open(unit=65,file='nt-closure-surface.csv',status='replace')
  ! The closure reads downdraft state -- zmfd in zheat, ztd/zqd in the
  ! ktype = 2 arm, loddraf in the zeps guard -- so it is captured after
  ! cuddrafn, at the point the closure actually sees it.
  open(unit=66,file='nt-downdraft-levels.csv',status='replace')
  open(unit=67,file='nt-downdraft-surface.csv',status='replace')
  open(unit=68,file='nt-shallow-arm.csv',status='replace')
  ! cuascn's remaining inputs, captured at its own call site.
  open(unit=69,file='nt-cuascn-in2-surface.csv',status='replace')
  open(unit=70,file='nt-cuascn-in2-levels.csv',status='replace')
  ! cudlfsn, captured and graded on its own.
  open(unit=71,file='nt-cudlfsn-in-surface.csv',status='replace')
  open(unit=72,file='nt-cudlfsn-in-levels.csv',status='replace')
  open(unit=73,file='nt-cudlfsn-out-surface.csv',status='replace')
  open(unit=74,file='nt-cudlfsn-out-levels.csv',status='replace')
  ! cuddrafn, captured and graded on its own.
  open(unit=75,file='nt-cuddrafn-in-surface.csv',status='replace')
  open(unit=76,file='nt-cuddrafn-in-levels.csv',status='replace')
  open(unit=77,file='nt-cuddrafn-out-surface.csv',status='replace')
  ! cuflxn, captured and graded on its own.
  open(unit=78,file='nt-cuflxn-in-surface.csv',status='replace')
  open(unit=79,file='nt-cuflxn-in-levels.csv',status='replace')
  open(unit=80,file='nt-cuflxn-out-surface.csv',status='replace')
  open(unit=81,file='nt-cuflxn-out-levels.csv',status='replace')
  ! cudtdqn, captured and graded on its own.
  open(unit=82,file='nt-cudtdqn-in-surface.csv',status='replace')
  open(unit=83,file='nt-cudtdqn-in-levels.csv',status='replace')
  open(unit=84,file='nt-cudtdqn-out-levels.csv',status='replace')
  ! cududvn, captured and graded on its own.
  open(unit=85,file='nt-cududvn-in-surface.csv',status='replace')
  open(unit=86,file='nt-cududvn-in-levels.csv',status='replace')
  open(unit=87,file='nt-cududvn-out-levels.csv',status='replace')
  ! cumastrn:833-919, the adjustments block.
  open(unit=88,file='nt-adjust-out-surface.csv',status='replace')
  open(unit=89,file='nt-adjust-out-levels.csv',status='replace')
  ! cumastrn:996-1016, the momentum mass-flux rescale.
  open(unit=90,file='nt-mrescale-in-levels.csv',status='replace')
  ! cumastrn:743-819, the updraft rescale and the two cleanups.
  open(unit=91,file='nt-uscale-in-surface.csv',status='replace')
  open(unit=92,file='nt-uscale-in-levels.csv',status='replace')
  ! cumastrn:927-995, the updraft/downdraft momentum profiles.
  open(unit=93,file='nt-mprofile-in-surface.csv',status='replace')
  open(unit=94,file='nt-mprofile-in-levels.csv',status='replace')
  ! cumastrn:1030-1056, the KE dissipation.
  open(unit=95,file='nt-kedis-in-levels.csv',status='replace')
  open(unit=96,file='nt-kedis-out-levels.csv',status='replace')
  ! cu_ntiedtke_run's post-conversion, :278-320.  THE MISSING LINK: it
  ! turns cumastrn's tendencies into the state cu_ntiedtke_post_run
  ! differences, so without it the chain from the last cumastrn stage to
  ! the eight graded fields has a hole in it.  Captured at its own
  ! boundary -- immediately after cumastrn returns and before the block
  ! runs -- because zqp1 is UPDATED IN PLACE and a capture taken after is
  ! the answer, not the input.
  open(unit=97,file='nt-postconv-in-levels.csv',   status='replace')
  open(unit=98,file='nt-postconv-in-surface.csv',  status='replace')
  open(unit=99,file='nt-postconv-out-levels.csv',  status='replace')
  open(unit=100,file='nt-postconv-out-surface.csv',status='replace')
  write(97,'(a)') 'case,dx,k,pcte,ztp1,ptte,ztt,pqte,zqq,zqp1,qcf,qif,' // &
       'uf,vf,pvom,pvol'
  write(98,'(a)') 'case,dx,prsfc,pssfc,delt'
  write(99,'(a)') 'case,dx,k,pqc,pqi,pt,pqv,pu,pv'
  write(100,'(a)') 'case,dx,zprecc'
  write(61,'(a)') 'case,dx,differing_words'
  ! 13 hex fields then one integer -- pgeo was missing from this header,
  ! which silently shifted every name after it and put pgeo's word in
  ! the klab column.  Caught by re-grading the reconstructions against
  ! this capture; nothing else would have noticed.
  write(62,'(a)') 'case,dx,k,ptenh,pqenh,pqsenh,ptu,pqu,plu,pmfu,pmfub_s,'// &
       'pqsen,pap,paph,pgeoh,pgeo,klab'
  write(63,'(a)') 'case,dx,k,ptu,pqu,plu,pmfu,pmfus,pmfuq,pmful,plude,'// &
       'pdmfup,plglac,pmfude_rate,ptenh_out,pqenh_out,klab'
  write(64,'(a)') 'case,dx,ldcum,ktype,kcbot,kctop,kctop0,kdpl,wup,wbase'
  write(65,'(a)') 'case,dx,zheat,zcape,zcape1,zcape2,ztauc,ztaubl,ztau,'// &
       'zmfub,zmfub1,zmfs,upbl,scale_fac,scale_fac2'
  write(66,'(a)') 'case,dx,k,zmfd,zmfds,zmfdq,zdmfdp,ztd,zqd,'// &
       'pmfdde_rate,ztenh,zqenh'
  write(67,'(a)') 'case,dx,loddraf,idtop,ktype_closure,upbl_pre,zdhpbl'
  write(68,'(a)') 'case,dx,zeps,zqumqe,zdqmin,zdh,zmfmax,pqu_ikb,'// &
       'plu_ikb,pqenh_ikb'
  write(69,'(a)') 'case,dx,klwmin,kctop0,kdpl,ldcum,ktype,'// &
       'kcbot,kctop,lndj,wbase'
  write(70,'(a)') 'case,dx,k,puen,pven,pten,pqen,pqte,pverv,puu,pvu,'// &
       'pmfus,pmfuq,pmful,plude'
  write(71,'(a)') 'case,dx,ldcum,ktype,kcbot,kctop,lndj,pmfub,prfl'
  write(72,'(a)') 'case,dx,k,ptenh,pqenh,puen,pven,pten,pqsen,pgeo,'// &
       'pgeoh,paph,ptu,pqu,plu,puu,pvu,ptd_in,pqd_in,pmfd_in,'//        &
       'pmfds_in,pmfdq_in,pdmfdp_in'
  write(73,'(a)') 'case,dx,kdtop,lddraf,prfl_out'
  write(74,'(a)') 'case,dx,k,ptd,pqd,pud,pvd,pmfd,pmfds,pmfdq,pdmfdp'
  write(75,'(a)') 'case,dx,lddraf,prfl,paph_sfc'
  write(76,'(a)') 'case,dx,k,ptenh,pqenh,pgeo,pgeoh,paph,pmfu,'// &
       'ptd,pqd,pmfd,pmfds,pmfdq,pdmfdp'
  write(77,'(a)') 'case,dx,prfl_out'
  write(78,'(a)') 'case,dx,ldcum,lddraf,ktype,kcbot,kctop,kdtop,'// &
       'paph_sfc'
  write(79,'(a)') 'case,dx,k,pten,pqen,pqsen,ptenh,pqenh,paph,'// &
       'pap,pgeoh,pmfu,pmfd,pmfus,pmfds,pmfuq,pmfdq,pmful,'//     &
       'pdmfup,pdmfdp,plglac_in,plude_in,pmfdde_rate_in'
  write(80,'(a)') 'case,dx,kdtop_out,prain,pmflxr_sfc,pmflxs_sfc'
  write(81,'(a)') 'case,dx,k,pmfu,pmfd,pmfus,pmfds,pmfuq,pmfdq,'// &
       'pmful,plglac,pdmfup,pdmfdp,pdpmel,pmflxr,pmflxs,'// &
       'pqsen,plude,pmfdde_rate,pmfude_rate'
  write(82,'(a)') 'case,dx,ldcum,kctop,ktopm2,paph_sfc'
  write(83,'(a)') 'case,dx,k,paph,pten,plglac,plude,pmfus,pmfds,'// &
       'pmfuq,pmfdq,pmful,pdmfup,pdmfdp,pdpmel,ptent_in,'//        &
       'ptenq_in,pcte_in,pmfu,pmfd'
  write(84,'(a)') 'case,dx,k,ptent,ptenq,pcte'
  write(85,'(a)') 'case,dx,ldcum,ktype,kcbot,ktopm2,paph_sfc'
  write(86,'(a)') 'case,dx,k,paph,puen,pven,pmfu,pmfd,puu,pud,'// &
       'pvu,pvd,ptenu_in,ptenv_in'
  write(87,'(a)') 'case,dx,k,ptenu,ptenv'
  write(88,'(a)') 'case,dx,prsfc,pssfc'
  write(89,'(a)') 'case,dx,k,pmfd,pmfds,pmfdq,pdmfdp,pdmfup,'// &
       'pmfdde_rate,pmfude_rate'
  write(90,'(a)') 'case,dx,k,pmfu,pmfd,paph'
  write(91,'(a)') 'case,dx,ldcum,ktype,kcbot,kdtop,zmfub1,zmfub'
  write(92,'(a)') 'case,dx,k,pmfu,pmfus,pmfuq,pmful,pdmfup,'// &
       'plude,pmfude_rate,paph,pmfd,pmfds,pmfdq,pdmfdp,pmfdde_rate'
  write(93,'(a)') 'case,dx,ldcum,ktype,kcbot,kctop,kdpl,kdtop'
  write(94,'(a)') 'case,dx,k,pmfu,pmfd,puen,pven,puu,pvu,pud,'// &
       'pvd,pmfude_rate'
  write(95,'(a)') 'case,dx,k,ztenu,ztenv,pvom,pvol,puen,pven,ptte'
  write(96,'(a)') 'case,dx,k,ptte'

  call cu_ntiedtke_init(real(nt_cp,kind_phys),real(nt_rd,kind_phys), &
       real(nt_rv,kind_phys),real(nt_xlv,kind_phys), &
       real(nt_xls,kind_phys),real(nt_xlf,kind_phys), &
       real(nt_g,kind_phys),errmsg,errflg)
  cpd_l = real(nt_cp,kind_phys);  g_l = real(nt_g,kind_phys)
  rd_l  = real(nt_rd,kind_phys);  alv_l = real(nt_xlv,kind_phys)
  vtmpc1_l = real(nt_rv,kind_phys)/rd_l - 1.0
  rcpd_l = 1.0/cpd_l
  zrg_l  = 1.0/g_l

  do m = 1, ndx
    dxv = nt_dxsweep(m)
    delt = dtc*stepcu

    ! ================= prep: the REAL cu_ntiedtke_pre_run ================
    ! This was the SECOND hand-written copy of the prep transcription (the
    ! third was in run_nt_cuinin.F90), and its header said "proven by
    ! run_nt_prep.F90".  run_nt_prep proves its OWN copy.  Nothing checked
    ! that the copies still said the same thing, which is resolution by
    ! apparent identity -- the port's own recurring failure, in its oracle.
    !
    ! Measured before removing it: all 52 recorded CSVs are byte-identical
    ! either way, so the copies did agree.  The exposure was real and the
    ! damage was nil, and that is worth stating as a result rather than
    ! quietly deleting the code.
    do n = 1, nc
      call nt_build_column(n,nz,b_t,b_qv,b_qc,b_qi,b_u,b_v,b_pcps,b_p8w, &
                           b_dz,b_rho,b_pi,b_w,b_qvften,b_thften,        &
                           b_xland,b_hfx,b_qfx)
      s_xland(n)=b_xland
      dx_hv(n)=dxv; hfx_hv(n)=b_hfx; qfx_hv(n)=b_qfx
      do k=1,nz
        s_t(n,k)=b_t(k);   s_qv(n,k)=b_qv(k); s_qc(n,k)=b_qc(k)
        s_qi(n,k)=b_qi(k); s_u(n,k)=b_u(k);   s_v(n,k)=b_v(k)
        s_pres(n,k)=b_pcps(k); s_dz(n,k)=b_dz(k); s_rho(n,k)=b_rho(k)
        s_qvften(n,k)=b_qvften(k); s_thften(n,k)=b_thften(k)
      end do
      do k=1,nz1
        s_presi(n,k)=b_p8w(k); s_w(n,k)=b_w(k)
      end do
    end do

    s_errmsg='x'; s_errflg=-1; s_im=-1; s_kx=-1; s_kx1=-1
    call x_pre_run(1,nc,1,nz,s_im,s_kx,s_kx1,itimestep,stepcu,dtc,g_l, &
         s_xland,s_dz,s_pres,s_presi,s_t,s_rho,s_qv,s_qc,s_qi,s_u,s_v, &
         s_w,s_qvften,s_thften,qvftenz,thftenz,slimsk,delt,prsl,ghtl,  &
         tf,qvf,qcf,qif,uf,vf,prsi,ghti,omg,s_errmsg,s_errflg,         &
         int(len(s_errmsg_s),kind=c_size_t))
    do jj = 1, len(s_errmsg_s)
       s_errmsg_s(jj:jj) = s_errmsg(jj)
    end do
    if (s_errflg /= 0 .or. &
        trim(s_errmsg_s) /= 'cu_ntiedtke_pre_run OK') then
       write(*,'(a)') 'FATAL: cu_ntiedtke_pre_run reported: ' // &
            trim(s_errmsg_s)
       stop 9
    end if

    include 'nt_run_conversion.inc'

    call replicate_cumastrn()

    ! ---- capture at the post-conversion's OWN boundary ----------------
    do n=1,nc
      do k=1,nz
        write(97,'(i0,",",f0.1,",",i0,13(",",a))') n, dxv, k, &
             hexw(pcte(n,k)), hexw(ztp1(n,k)), hexw(ptte(n,k)), &
             hexw(ztt(n,k)),  hexw(pqte(n,k)), hexw(zqq(n,k)),  &
             hexw(zqp1(n,k)), hexw(qcf(n,k)),  hexw(qif(n,k)),  &
             hexw(uf(n,k)),   hexw(vf(n,k)),   hexw(pvom(n,k)), &
             hexw(pvol(n,k))
      end do
      write(98,'(i0,",",f0.1,3(",",a))') n, dxv, &
           hexw(prsfc(n)), hexw(pssfc(n)), hexw(delt)
    end do

    ! ---- cu_ntiedtke_run's post-conversion (:278-320) -----------------
    do k=1,nz
      do n=1,nc
        if (pcte(n,k) > 0.) then
          r_pqc(n,k) = qcf(n,k) + nt_foealfa(ztp1(n,k))*pcte(n,k)*delt
          r_pqi(n,k) = qif(n,k) + (1.0-nt_foealfa(ztp1(n,k)))*pcte(n,k)*delt
        else
          r_pqc(n,k) = qcf(n,k);  r_pqi(n,k) = qif(n,k)
        end if
      end do
    end do
    do k=1,nz
      do n=1,nc
        r_pt(n,k)  = ztp1(n,k)+(ptte(n,k)-ztt(n,k))*delt
        zqp1(n,k)  = zqp1(n,k)+(pqte(n,k)-zqq(n,k))*delt
        r_pqv(n,k) = zqp1(n,k)/(1.0-zqp1(n,k))
        r_pu(n,k)  = uf(n,k)+pvom(n,k)*delt
        r_pv(n,k)  = vf(n,k)+pvol(n,k)*delt
      end do
    end do
    do n=1,nc
      r_rn(n) = amax1(0.0,(prsfc(n)+pssfc(n))*delt)
    end do

    ! ---- and what the block produced ----------------------------------
    do n=1,nc
      do k=1,nz
        write(99,'(i0,",",f0.1,",",i0,6(",",a))') n, dxv, k, &
             hexw(r_pqc(n,k)), hexw(r_pqi(n,k)), hexw(r_pt(n,k)), &
             hexw(r_pqv(n,k)), hexw(r_pu(n,k)),  hexw(r_pv(n,k))
      end do
      write(100,'(i0,",",f0.1,",",a)') n, dxv, hexw(r_rn(n))
    end do

    ! ================= the control =====================================
    do n=1,nc
      do k=1,nz
        c_pt(n,k)=tf(n,k); c_pqv(n,k)=qvf(n,k); c_pqc(n,k)=qcf(n,k)
        c_pqi(n,k)=qif(n,k); c_pu(n,k)=uf(n,k); c_pv(n,k)=vf(n,k)
      end do
    end do
    call cu_ntiedtke_run(pu=c_pu,pv=c_pv,pt=c_pt,pqv=c_pqv,pqc=c_pqc, &
         pqi=c_pqi,pqvf=qvftenz,ptf=thftenz,poz=ghtl,pzz=ghti,pomg=omg, &
         pap=prsl,paph=prsi,evap=qfx_hv,hfx=hfx_hv,zprecc=c_rn, &
         lndj=slimsk,lq=nc,km=nz,km1=nz1,dt=delt,dx=dx_hv, &
         errmsg=errmsg,errflg=errflg)

    ! ================= the proof =======================================
    do n=1,nc
      nd = 0
      do k=1,nz
        if (wne(r_pt(n,k), c_pt(n,k)))   nd=nd+1
        if (wne(r_pqv(n,k),c_pqv(n,k)))  nd=nd+1
        if (wne(r_pqc(n,k),c_pqc(n,k)))  nd=nd+1
        if (wne(r_pqi(n,k),c_pqi(n,k)))  nd=nd+1
        if (wne(r_pu(n,k), c_pu(n,k)))   nd=nd+1
        if (wne(r_pv(n,k), c_pv(n,k)))   nd=nd+1
      end do
      if (wne(r_rn(n), c_rn(n))) nd=nd+1
      total_bad = total_bad + nd
      write(61,'(i0,",",f0.1,",",i0)') n, dxv, nd
    end do
  end do

  close(61); close(62); close(63); close(64); close(65)
  close(66); close(67); close(68); close(69); close(70)
  close(71); close(72); close(73); close(74)
  close(75); close(76); close(77)
  close(78); close(79); close(80); close(81)
  close(82); close(83); close(84)
  close(85); close(86); close(87)
  close(88); close(89); close(90); close(91); close(92)
  close(93); close(94); close(95); close(96)
  close(97); close(98); close(99); close(100)
  if (total_bad /= 0) then
    write(*,'(a,i0,a)') 'FATAL: the cumastrn replication differs in ', &
        total_bad, ' words; the captures are not cumastrn''s state.'
    stop 7
  end if
  write(*,'(a)') 'run_nt_cumastrn OK -- 0 differing words'

contains

  include 'nt_cumastrn_body.inc'




end program run_nt_cumastrn
