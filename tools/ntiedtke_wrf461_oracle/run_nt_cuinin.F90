program run_nt_cuinin
  ! Slice 2: cu_ntiedtke_run's variable conversion, and cuinin.
  !
  ! ===================================================================
  ! HOW THIS REACHES A PRIVATE ROUTINE WITHOUT EDITING PINNED SOURCE
  ! ===================================================================
  ! cumastrn, cuinin and everything under them are PRIVATE to module
  ! cu_ntiedtke -- only cu_ntiedtke_run / _init / _finalize are public --
  ! and gfortran gives private module procedures LOCAL symbol binding
  ! (nm shows `t`, not `T`), so a plain external declaration will not link.
  !
  ! The route is NOT to make them public.  Editing cu_ntiedtke.F90 would
  ! break the sha256 pin the whole oracle rests on, and a fixture built
  ! from modified source is not a fixture.
  !
  ! Instead build.sh runs `objcopy --globalize-symbol` on the COMPILED
  ! OBJECT.  That flips a binding bit in the ELF symbol table and touches
  ! no instruction: MEASURED, the .text section is byte-identical before
  ! and after (146,406 B, sha256 7ec70a39...), and 377 bytes differ in the
  ! whole object, all of them symbol-table.  build.sh asserts that identity
  ! every run rather than trusting this comment.  The source stays pinned,
  ! the code generation stays untouched, and only linkage changes.
  !
  ! The interfaces below use bind(C) purely to NAME the mangled symbol.
  ! Both sides are gfortran, every dummy is explicit-shape, and non-VALUE
  ! dummies pass by reference under either convention, so the ABI is the
  ! one gfortran already emits.  Assumed-shape dummies would NOT be safe
  ! this way (they pass descriptors) -- cu_ntiedtke_run has them, which is
  ! why the public interface is USEd normally and only the private
  ! explicit-shape routines are reached by symbol.
  !
  ! ===================================================================
  ! WHY cuinin NEEDS NO TRIGGER VISIBILITY
  ! ===================================================================
  ! cuinin takes no ldcum, no ktype and no ierr, and its only flag is
  ! loflag(jl) = .true. set unconditionally for every column.  It runs at
  ! cumastrn:474, BEFORE cutypen at :490 decides the convection type.  So
  ! it is column-universal exactly the way the prep is, and it can be
  ! graded against the existing 108-column fixture without the decomposition
  ! harness that cutypen and the closure will genuinely need.
  !
  ! cumastrn passes its own dummies straight into cuinin (:474-481), so
  ! cuinin's inputs ARE cu_ntiedtke_run's conversion outputs.  That is why
  ! this one program captures both.
  use iso_fortran_env, only: int32
  use iso_c_binding, only: c_int
  use ccpp_kind_types, only: kind_phys
  use cu_ntiedtke, only: cu_ntiedtke_run, cu_ntiedtke_init
  use nt_cases
  implicit none

  ! Every x_* interface lives in nt_cases, declared once.  Two copies
  ! of x_cuinin and x_cutypen used to sit in this directory with
  ! different line breaks and nothing comparing them.

  integer, parameter :: nz = nt_nz, nc = nt_ncase, ndx = nt_ndx

  real(kind=kind_phys), dimension(nc, nz) :: &
       pum1, pvm1, ztp1, zqp1, zqsat, pgeo, pverv, ptte, pqte
  real(kind=kind_phys), dimension(nc, nz + 1) :: pgeoh
  ! cuinin outputs
  real(kind=kind_phys), dimension(nc, nz) :: &
       ztenh, zqenh, zqsenh, ptu, pqu, ztd, zqd, zuu, zvu, zud, zvd, &
       pmfu, pmfd, zmfus, zmfds, zmfuq, zmfdq, zdmfup, zdmfdp, zdpmel, &
       plu, plude
  integer, dimension(nc) :: ilwmin
  integer, dimension(nc, nz) :: ilab

  ! cutypen outputs.  cutu/cuqu/culu/culab are intent(out) THERE, so
  ! cutypen does not consume cuinin's ptu/pqu/plu/ilab -- it rebuilds them.
  integer, dimension(nc) :: t_ldcum, t_cubot, t_cutop, t_ktype, t_kdpl
  real(kind=kind_phys), dimension(nc) :: t_wbase
  real(kind=kind_phys), dimension(nc, nz) :: t_cutu, t_cuqu, t_culu
  integer, dimension(nc, nz) :: t_culab
  real(kind=kind_phys), dimension(nc, nz) :: c_ptu, c_pqu, c_plu
  integer, dimension(nc, nz) :: c_ilab

  ! cuascn's pre-state (its :1903-1952), and the two callees' outputs.
  integer, dimension(nc) :: a_ldcum, a_ktype, a_kcbot, a_kctop, a_kctop0
  integer, dimension(nc, nz) :: a_klab
  real(kind=kind_phys), dimension(nc) :: a_pmfub, a_dmfen, a_dmfde
  real(kind=kind_phys), dimension(nc, nz) :: a_plu, a_pmfu, a_pmfus
  real(kind=kind_phys), dimension(nc, nz) :: a_pmfuq, a_pmful, a_plude
  real(kind=kind_phys), dimension(nc, nz) :: a_plglac, a_pdmfup, a_zlrain
  real(kind=kind_phys), dimension(nc, nz) :: a_zbuo, a_kup, a_pdmfen
  real(kind=kind_phys), dimension(nc, nz) :: a_rate, a_ptu, a_pqu
  real(kind=kind_phys), dimension(nc, nz, nz) :: a_en, a_de
  integer :: ik
  ! cumastrn:500-541 -- the first-guess cloud-base mass flux.
  real(kind=kind_phys), dimension(nc) :: m_zdhpbl, m_upbl, m_zmfub
  integer, dimension(nc) :: m_ldcum
  real(kind=kind_phys) :: zcons2_l, zmfmax_l, zqumqe_l, zdqmin_l, zdh_l
  real(kind=kind_phys) :: wspeed_l

  ! the prep arrays (scheme order), rebuilt here the same way run_nt_prep does
  real(kind=kind_phys), dimension(nc, nz) :: prsl, ghtl, omg
  real(kind=kind_phys), dimension(nc, nz) :: uf, vf, tf, qvf, qcf, qif
  real(kind=kind_phys), dimension(nc, nz) :: qvftenz, thftenz
  real(kind=kind_phys), dimension(nc, nz + 1) :: prsi, ghti
  integer, dimension(nc) :: slimsk

  ! WRF-order staging for the REAL cu_ntiedtke_pre_run.  zi/zl/dotv used to
  ! live here; they were pre_run's LOCALS, hand-copied into this program.
  real(kind=kind_phys), dimension(nc)         :: g_xland
  real(kind=kind_phys), dimension(nc, nz)     :: g_t, g_qv, g_qc, g_qi
  real(kind=kind_phys), dimension(nc, nz)     :: g_u, g_v, g_pres, g_dz
  real(kind=kind_phys), dimension(nc, nz)     :: g_rho, g_qvften, g_thften
  real(kind=kind_phys), dimension(nc, nz + 1) :: g_presi, g_w
  integer :: g_im, g_kx, g_kx1, g_errflg, jj
  character(kind=c_char) :: g_errmsg(256)
  character(len=256) :: g_errmsg_s
  real(kind=kind_phys), dimension(nc) :: dx_hv, hfx_hv, qfx_hv

  real, dimension(nz) :: b_t, b_qv, b_qc, b_qi, b_u, b_v
  real, dimension(nz) :: b_pcps, b_dz, b_rho, b_pi, b_qvften, b_thften
  real, dimension(nz + 1) :: b_p8w, b_w
  real :: b_xland, b_hfx, b_qfx, dxv

  real(kind=kind_phys), parameter :: dtc = 60.0_kind_phys
  integer, parameter :: stepcu = 1, itimestep = 2
  real(kind=kind_phys) :: delt, tt, zew, zqs, zcor, vtmpc1_l

  ! Named in nt_run_conversion.inc and NOT READ here.  See the comment at
  ! the head of that file: the alternative to carrying them is splitting a
  ! transcription across two files, which would reorder statements that are
  ! one do-loop in the source.
  integer, parameter :: nz1 = nz + 1
  real(kind=kind_phys), parameter :: g_l = real(nt_g, kind_phys)
  real(kind=kind_phys) :: dxref
  real(kind=kind_phys), dimension(nc) :: scale_fac, scale_fac2, zrain
  real(kind=kind_phys), dimension(nc) :: prsfc, pssfc, pqhfl, phhfl
  integer, dimension(nc) :: locum
  real(kind=kind_phys), dimension(nc, nz) :: pcte, pvom, pvol, zqq, ztt
  character(len=256) :: errmsg
  integer :: errflg, n, m, k, pp, zz

  if (kind(1.0_kind_phys) /= 4) stop 3
  call nt_build_case_table()

  open(unit=31, file='nt-conv-levels.csv',   status='replace')
  open(unit=32, file='nt-cuinin-levels.csv', status='replace')
  open(unit=33, file='nt-cuinin-surface.csv', status='replace')
  open(unit=34, file='nt-cutypen-levels.csv',  status='replace')
  open(unit=35, file='nt-cutypen-surface.csv', status='replace')
  open(unit=36, file='nt-midlevel-levels.csv',  status='replace')
  open(unit=37, file='nt-midlevel-surface.csv', status='replace')
  open(unit=38, file='nt-cuentrn.csv',          status='replace')
  open(unit=39, file='nt-mfub-surface.csv',     status='replace')
  write(31,'(a)') 'case,dx,k,ztp1,zqp1,zqsat,pgeo,pgeoh,pverv,ptte,pqte'
  write(32,'(a)') 'case,dx,k,ptenh,pqenh,pqsenh,ptu,pqu,ptd,pqd,puu,' // &
       'pvu,pud,pvd,plu,klab'
  write(33,'(a)') 'case,dx,klwmin'
  write(34,'(a)') 'case,dx,k,cutu,cuqu,culu,culab'
  write(35,'(a)') 'case,dx,ldcum,ktype,cubot,cutop,kdpl,wbase'
  write(36,'(a)') 'case,dx,k,ptu,pqu,plu,pmfu,pmfus,pmfuq,pmful,' // &
       'pdmfup,plrain,klab'
  write(37,'(a)') 'case,dx,ktype,kcbot,pmfub'
  write(38,'(a)') 'case,dx,kk,pdmfen,pdmfde'
  write(39,'(a)') 'case,dx,ldcum,zdhpbl,upbl,zmfub'

  call cu_ntiedtke_init(real(nt_cp,kind_phys), real(nt_rd,kind_phys), &
                        real(nt_rv,kind_phys), real(nt_xlv,kind_phys), &
                        real(nt_xls,kind_phys), real(nt_xlf,kind_phys), &
                        real(nt_g,kind_phys), errmsg, errflg)
  vtmpc1_l = real(nt_rv,kind_phys)/real(nt_rd,kind_phys) - 1.0

  do m = 1, ndx
    dxv = nt_dxsweep(m)
    delt = dtc * stepcu

    ! ---- prep: the REAL cu_ntiedtke_pre_run =============================
    ! This was a hand-written THIRD copy of the prep transcription, and it
    ! carried the comment "run_nt_prep.F90 proves this replication exact".
    ! IT DID NOT.  run_nt_prep proves its OWN copy, and this one was not the
    ! same text: it dropped pre_run's `if (itimestep == 1)` branch entirely
    ! -- itimestep was declared in this program and never read -- and it ran
    ! the loop nest per column where the routine runs it per level.  It
    ! agreed only because itimestep happens to be 2 in this fixture, which
    ! is a property of the fixture and not of the transcription.
    !
    ! That is the port's own recurring failure appearing in its oracle:
    ! resolution by APPARENT IDENTITY rather than by provenance.  Every
    ! CSV this program records was recorded on a state nothing graded.
    ! build.sh globalizes cu_ntiedtke_pre_run, so it is now CALLED.
    do n = 1, nc
      call nt_build_column(n, nz, b_t, b_qv, b_qc, b_qi, b_u, b_v, &
                           b_pcps, b_p8w, b_dz, b_rho, b_pi, b_w, &
                           b_qvften, b_thften, b_xland, b_hfx, b_qfx)
      g_xland(n) = b_xland
      dx_hv(n) = dxv;  hfx_hv(n) = b_hfx;  qfx_hv(n) = b_qfx
      do k = 1, nz
        g_t(n,k)   = b_t(k);     g_qv(n,k)  = b_qv(k)
        g_qc(n,k)  = b_qc(k);    g_qi(n,k)  = b_qi(k)
        g_u(n,k)   = b_u(k);     g_v(n,k)   = b_v(k)
        g_pres(n,k) = b_pcps(k); g_dz(n,k)  = b_dz(k)
        g_rho(n,k) = b_rho(k)
        g_qvften(n,k) = b_qvften(k);  g_thften(n,k) = b_thften(k)
      end do
      do k = 1, nz + 1
        g_presi(n,k) = b_p8w(k);  g_w(n,k) = b_w(k)
      end do
    end do

    g_errmsg = 'x';  g_errflg = -1
    g_im = -1;  g_kx = -1;  g_kx1 = -1
    call x_pre_run(1, nc, 1, nz, g_im, g_kx, g_kx1, itimestep, stepcu, &
         dtc, real(nt_g,kind_phys), g_xland, g_dz, g_pres, g_presi, &
         g_t, g_rho, g_qv, g_qc, g_qi, g_u, g_v, g_w, g_qvften, g_thften, &
         qvftenz, thftenz, slimsk, delt, prsl, ghtl, tf, qvf, qcf, qif, &
         uf, vf, prsi, ghti, omg, g_errmsg, g_errflg, &
         int(len(g_errmsg_s), kind=c_size_t))
    do jj = 1, len(g_errmsg_s)
       g_errmsg_s(jj:jj) = g_errmsg(jj)
    end do
    if (g_errflg /= 0 .or. &
        trim(g_errmsg_s) /= 'cu_ntiedtke_pre_run OK') then
       write(*,'(a)') 'FATAL: cu_ntiedtke_pre_run reported: ' // &
            trim(g_errmsg_s)
       stop 9
    end if

    include 'nt_run_conversion.inc'

    ! ---- cuinin, through the globalized symbol --------------------------
    pmfu = 0.; pmfd = 0.; zmfus = 0.; zmfds = 0.; zmfuq = 0.; zmfdq = 0.
    zdmfup = 0.; zdmfdp = 0.; plude = 0.; zdpmel = 0.
    call x_cuinin(nc, nz, nz+1, nz-1, ztp1, zqp1, zqsat, pum1, pvm1, &
         pverv, pgeo, prsi, pgeoh, ztenh, zqenh, zqsenh, ilwmin, &
         ptu, pqu, ztd, zqd, zuu, zvu, zud, zvd, pmfu, pmfd, zmfus, &
         zmfds, zmfuq, zmfdq, zdmfup, zdmfdp, zdpmel, plu, plude, ilab)

    ! ---- cutypen, through the globalized symbol -------------------------
    ! Its inputs are the conversion outputs plus cuinin's ptenh/pqenh/
    ! pqsenh, all already captured, so this needs nothing new.
    ! Snapshot cuinin's outputs BEFORE cutypen overwrites them in place --
    ! the cuinin CSV must record cuinin's answer, not cutypen's.
    c_ptu = ptu;  c_pqu = pqu;  c_plu = plu;  c_ilab = ilab
    t_cutu = ptu;  t_cuqu = pqu;  t_culu = plu;  t_culab = ilab
    call x_cutypen(nc, nz, nz+1, nz-1, zqp1, ztenh, zqenh, zqsenh, &
         pgeoh, prsi, hfx_hv, qfx_hv, pgeo, zqsat, prsl, ztp1, slimsk, &
         t_cutu, t_cuqu, t_culab, t_ldcum, t_cubot, t_cutop, t_ktype, &
         t_wbase, t_culu, t_kdpl)

    ! ============ cumastrn:500-541, the first-guess mass flux ===========
    ! This runs BETWEEN cutypen and cuascn, and it is not optional: it
    ! produces pmfub, which cuascn READS (:1949-1952, :1992), and it can
    ! flip ldcum to false for a ktype = 2 column whose PBL moist static
    ! energy budget is non-positive (:536).  A fixture that skips it hands
    ! cuascn pmfub = 0 and a stale ldcum, and every mass-flux quantity
    ! downstream is then structurally zero -- green, and meaningless.
    zcons2_l = 3./(real(nt_g,kind_phys)*delt)
    do n = 1, nc
      m_zdhpbl(n) = 0.0
      m_upbl(n)   = 0.0
      m_ldcum(n)  = t_ldcum(n)
      m_zmfub(n)  = 0.0
    end do
    do k = 2, nz
      do n = 1, nc
        if (k >= t_cubot(n) .and. m_ldcum(n) /= 0) then
          m_zdhpbl(n) = m_zdhpbl(n) &
              + (real(nt_xlv,kind_phys)*pqte(n,k) &
                 + real(nt_cp,kind_phys)*ptte(n,k)) &
                * (prsi(n,k+1) - prsi(n,k))
          if (slimsk(n) == 0) then
            wspeed_l = sqrt(pum1(n,k)**2 + pvm1(n,k)**2)
            m_upbl(n) = m_upbl(n) + wspeed_l*(prsi(n,k+1) - prsi(n,k))
          end if
        end if
      end do
    end do
    do n = 1, nc
      if (m_ldcum(n) /= 0) then
        ik = t_cubot(n)
        zmfmax_l = (prsi(n,ik) - prsi(n,ik-1))*zcons2_l
        if (t_ktype(n) == 1) then
          m_zmfub(n) = 0.1*zmfmax_l
        else if (t_ktype(n) == 2) then
          zqumqe_l = t_cuqu(n,ik) + t_culu(n,ik) - zqenh(n,ik)
          zdqmin_l = max(0.01*zqenh(n,ik), 1.e-10)
          zdh_l = real(nt_cp,kind_phys)*(t_cutu(n,ik) - ztenh(n,ik)) &
                  + real(nt_xlv,kind_phys)*zqumqe_l
          zdh_l = real(nt_g,kind_phys)*max(zdh_l, 1.e5*zdqmin_l)
          if (m_zdhpbl(n) > 0.) then
            m_zmfub(n) = m_zdhpbl(n)/zdh_l
            m_zmfub(n) = min(m_zmfub(n), zmfmax_l)
          else
            m_zmfub(n) = 0.1*zmfmax_l
            m_ldcum(n) = 0
          end if
        end if
      else
        m_zmfub(n) = 0.
      end if
    end do

    ! ================= cuascn's pre-state, then the two callees ==========
    ! Transcribed from cuascn:1903-1952.  It is cuascn's own prologue, so
    ! it is NOT proven here -- sub-stage 2 proves it when the full cuascn
    ! runs.  What IS graded here is cubasmcn and cuentrn given this state.
    !
    ! Note what it does to a rejected column: kcbot = -1, so the
    ! `if(jk.ne.kcbot)` test zeroes plu at EVERY level, and klab is zeroed
    ! outright.  That is why cubasmcn's `klab(kk+1) == 0` gate is satisfied
    ! on every rejected column, and why ktype = 3 is reachable at all.
    do n = 1, nc
      a_ldcum(n) = m_ldcum(n);  a_ktype(n) = t_ktype(n)
      a_kcbot(n) = t_cubot(n);  a_kctop0(n) = 0
      a_pmfub(n) = m_zmfub(n)
      do k = 1, nz
        a_ptu(n,k) = t_cutu(n,k);  a_pqu(n,k) = t_cuqu(n,k)
        a_plu(n,k) = t_culu(n,k);  a_klab(n,k) = t_culab(n,k)
      end do
      if (a_ldcum(n) == 0) then
        a_ktype(n) = 0
        a_kcbot(n) = -1
        a_pmfub(n) = 0.
        a_pqu(n,nz) = 0.
      end if
      do k = 1, nz
        if (k /= a_kcbot(n)) a_plu(n,k) = 0.
        a_pmfu(n,k)=0.;  a_pmfus(n,k)=0.;  a_pmfuq(n,k)=0.
        a_pmful(n,k)=0.; a_plude(n,k)=0.;  a_plglac(n,k)=0.
        a_pdmfup(n,k)=0.; a_zlrain(n,k)=0.; a_zbuo(n,k)=0.
        a_kup(n,k)=0.;   a_pdmfen(n,k)=0.; a_rate(n,k)=0.
        if (a_ldcum(n) == 0 .or. a_ktype(n) == 3) a_klab(n,k) = 0
        if (a_ldcum(n) == 0 .and. prsi(n,k) < 4.e4) a_kctop0(n) = k
      end do
      if (a_ktype(n) == 3) a_ldcum(n) = 0
      a_kctop(n) = a_kcbot(n)
      if (a_ldcum(n) /= 0) then
        ik = a_kcbot(n)
        a_kup(n,ik)   = 0.5*t_wbase(n)**2
        a_pmfu(n,ik)  = a_pmfub(n)
        a_pmfus(n,ik) = a_pmfub(n)*(real(nt_cp,kind_phys)*a_ptu(n,ik) &
                                    + pgeoh(n,ik))
        a_pmfuq(n,ik) = a_pmfub(n)*a_pqu(n,ik)
        a_pmful(n,ik) = a_pmfub(n)*a_plu(n,ik)
      end if
    end do

    ! cuascn calls cubasmcn once per level, jk = klevm1 .. 3 (:1959-1974).
    ! cuentrn is called on the same levels (:2008).
    do ik = nz-1, 3, -1
      call x_cubasmcn(nc, nz, nz-1, ik, ztp1, zqp1, zqsat, pum1, pvm1, &
           pverv, pgeo, pgeoh, a_ldcum, a_ktype, a_klab, a_zlrain,   &
           a_pmfu, a_pmfub, a_kcbot, a_ptu, a_pqu, a_plu, pum1, pvm1,  &
           a_pmfus, a_pmfuq, a_pmful, a_pdmfup)
      call x_cuentrn(nc, nz, ik, a_kcbot, a_ldcum, 1, pgeoh, a_pmfu, &
           a_dmfen, a_dmfde)
      do n = 1, nc
        a_en(n,ik,1) = a_dmfen(n)
        a_de(n,ik,1) = a_dmfde(n)
      end do
    end do

    do n = 1, nc
      do k = 1, nz
        write(31,'(i0,",",f0.1,",",i0,8(",",a))') n, dxv, k, &
             hexw(ztp1(n,k)), hexw(zqp1(n,k)), hexw(zqsat(n,k)), &
             hexw(pgeo(n,k)), hexw(pgeoh(n,k)), hexw(pverv(n,k)), &
             hexw(ptte(n,k)), hexw(pqte(n,k))
        write(32,'(i0,",",f0.1,",",i0,12(",",a),",",i0)') n, dxv, k, &
             hexw(ztenh(n,k)), hexw(zqenh(n,k)), hexw(zqsenh(n,k)), &
             hexw(c_ptu(n,k)), hexw(c_pqu(n,k)), hexw(ztd(n,k)),    &
             hexw(zqd(n,k)),   hexw(zuu(n,k)),   hexw(zvu(n,k)),    &
             hexw(zud(n,k)),   hexw(zvd(n,k)),   hexw(c_plu(n,k)),  &
             c_ilab(n,k)
      end do
      write(33,'(i0,",",f0.1,",",i0)') n, dxv, ilwmin(n)
      write(35,'(i0,",",f0.1,5(",",i0),",",a)') n, dxv, t_ldcum(n), &
           t_ktype(n), t_cubot(n), t_cutop(n), t_kdpl(n), hexw(t_wbase(n))
      do k = 1, nz
        write(34,'(i0,",",f0.1,",",i0,3(",",a),",",i0)') n, dxv, k, &
             hexw(t_cutu(n,k)), hexw(t_cuqu(n,k)), hexw(t_culu(n,k)), &
             t_culab(n,k)
      end do
      write(39,'(i0,",",f0.1,",",i0,3(",",a))') n, dxv, m_ldcum(n), &
           hexw(m_zdhpbl(n)), hexw(m_upbl(n)), hexw(m_zmfub(n))
      write(37,'(i0,",",f0.1,2(",",i0),",",a)') n, dxv, a_ktype(n), &
           a_kcbot(n), hexw(a_pmfub(n))
      do k = 1, nz
        write(36,'(i0,",",f0.1,",",i0,9(",",a),",",i0)') n, dxv, k, &
             hexw(a_ptu(n,k)),   hexw(a_pqu(n,k)),   hexw(a_plu(n,k)),  &
             hexw(a_pmfu(n,k)),  hexw(a_pmfus(n,k)), hexw(a_pmfuq(n,k)),&
             hexw(a_pmful(n,k)), hexw(a_pdmfup(n,k)),                   &
             hexw(a_zlrain(n,k)), a_klab(n,k)
      end do
      do k = 3, nz-1
        write(38,'(i0,",",f0.1,",",i0,2(",",a))') n, dxv, k, &
             hexw(a_en(n,k,1)), hexw(a_de(n,k,1))
      end do
    end do
  end do

  close(31); close(32); close(33); close(34); close(35)
  close(36); close(37); close(38); close(39)
  write(*,'(a)') 'run_nt_cuinin OK (cuinin + cutypen + cubasmcn + cuentrn)'

end program run_nt_cuinin
