program run_nt_prep
  ! Prep-stage fixture: capture the arrays cu_ntiedtke_pre_run builds, which
  ! are exactly cu_ntiedtke_run's arguments.
  !
  ! WHY THIS BOUNDARY.  `cumastrn` and every routine under it is PRIVATE to
  ! module cu_ntiedtke (only cu_ntiedtke_run / _init / _finalize are public),
  ! so the scheme's internals cannot be reached without editing pinned source.
  ! But the prep splits at a PUBLIC boundary: cu_ntiedtke_pre_run's outputs
  ! ARE cu_ntiedtke_run's dummy arguments (module_cu_ntiedtke.F:205-226).  So
  ! this program replicates pre_run statement by statement, calls the real
  ! public cu_ntiedtke_run with the replicated arrays, replicates post_run,
  ! and compares the result bitwise against a real cu_ntiedtke_driver call on
  ! the same column.
  !
  ! It therefore PROVES its own fidelity rather than asserting it, which is
  ! the same structure run_cup_gf.F90 uses for Grell-Freitas.  Zero differing
  ! words is the only accepted value and build.sh fails otherwise: if the
  ! replication were wrong the captured prep arrays would be a fiction, and a
  ! port graded against them would be graded against the wrong numbers.
  !
  ! THE FLIP IS THE POINT.  pre_run reverses the vertical: WRF's k = kts
  ! (surface) becomes zz = kte.  Every ArWen cumulus kernel to date is
  ! bottom-up and this scheme is top-down, so this is the one structural
  ! inversion in the port, and its failure mode is silent -- an upside-down
  ! column produces plausible-looking finite numbers, not a crash.  Grading
  ! it first is worth far more than its line count.
  use iso_fortran_env, only: int32
  use iso_c_binding, only: c_char, c_size_t
  use ccpp_kind_types, only: kind_phys
  use cu_ntiedtke, only: cu_ntiedtke_run, cu_ntiedtke_init
  use module_cu_ntiedtke, only: cu_ntiedtke_driver
  use nt_cases
  implicit none

  ! x_pre_run and x_post_run come from nt_cases, declared once for
  ! every harness (see the comment there).

  integer, parameter :: nz = nt_nz, nc = nt_ncase, ndx = nt_ndx
  integer, parameter :: kts = 1, kte = nz, its = 1, ite = nc

  ! --- driver-shaped arrays, for the control call -------------------------
  real(kind=kind_phys), dimension(nc, nz + 1, 1) :: &
       u3d, v3d, w, t3d, qv3d, qc3d, qi3d, pi3d, rho3d, &
       dz8w, pcps, p8w, qvften, thften, &
       rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten
  real(kind=kind_phys), dimension(nc, 1) :: raincv, pratec, qfx, hfx, xland, dx
  logical, dimension(nc, 1) :: cu_act_flag

  ! --- the replication's own state ----------------------------------------
  integer, dimension(nc) :: slimsk
  real(kind=kind_phys), dimension(nc) :: rn, dx_hv, hfx_hv, qfx_hv
  real(kind=kind_phys), dimension(nc, nz)     :: prsl, omg, ghtl
  real(kind=kind_phys), dimension(nc, nz)     :: uf, vf, tf, qvf, qcf, qif
  real(kind=kind_phys), dimension(nc, nz)     :: qvftenz, thftenz

  ! --- what the REAL cu_ntiedtke_pre_run produces, for the direct proof ---
  ! Until cu_ntiedtke_pre_run was globalized (build.sh's globalize_objects)
  ! it was PRIVATE and unlinkable, so this program could only replicate it
  ! and prove the replication by CONVERGENCE -- the whole pipeline landing
  ! on the driver's answer.  That is the limit docs/ntiedtke/PORT-RECORD.md section 8
  ! states: converging does not prove any intermediate.
  !
  ! The flip is the one structural inversion in the port and its failure
  ! mode is silent, so proving it only at the far end was the weakest
  ! evidence for the thing most able to be quietly wrong.  These arrays
  ! take the real routine's output and every field is compared bitwise.
  integer, dimension(nc) :: r_slimsk
  real(kind=kind_phys) :: r_delt
  integer :: r_im, r_kx, r_kx1
  real(kind=kind_phys), dimension(nc, nz)     :: r_prsl, r_omg, r_ghtl
  real(kind=kind_phys), dimension(nc, nz)     :: r_uf, r_vf, r_tf
  real(kind=kind_phys), dimension(nc, nz)     :: r_qvf, r_qcf, r_qif
  real(kind=kind_phys), dimension(nc, nz)     :: r_qvftenz, r_thftenz
  real(kind=kind_phys), dimension(nc, nz + 1) :: r_prsi, r_ghti
  ! errmsg as a c_char ARRAY, not a character scalar: the dummy in a
  ! BIND(C) interface must be an array of single characters, and building
  ! the string back by hand is unambiguous where sequence-associating a
  ! character scalar into it would depend on how gfortran chooses to read
  ! the standard.
  character(kind=c_char) :: r_errmsg(256)
  character(len=256) :: r_errmsg_s
  integer :: r_errflg, r_bad, jj
  ! Per-field counts, so a failure NAMES the field.  That is the whole
  ! difference between this gate and the convergence gate below it: the
  ! convergence gate says the pipeline disagrees with the driver, which is
  ! true of every one of the fifteen fields at once and localises nothing.
  ! post_run's outputs from the REAL routine, for the same direct grade.
  real(kind=kind_phys), dimension(nc, nz) :: &
       p_rthcuten, p_rqvcuten, p_rqccuten, p_rqicuten, p_rucuten, p_rvcuten
  real(kind=kind_phys), dimension(nc) :: p_raincv, p_pratec
  integer :: p_bad, p_fieldbad(8)
  character(len=8), parameter :: p_fieldname(8) = [character(len=8) :: &
       'raincv', 'pratec', 'rthcuten', 'rqvcuten', 'rqccuten', 'rqicuten', &
       'rucuten', 'rvcuten']

  integer, parameter :: nfield = 15
  integer :: r_fieldbad(nfield)
  character(len=8), parameter :: r_fieldname(nfield) = [character(len=8) :: &
       'delt', 'slimsk', 'prsl', 'ghtl', 'omg', 'tf', 'qvf', 'qcf', &
       'qif', 'uf', 'vf', 'qvftenz', 'thftenz', 'prsi', 'ghti']
  real(kind=kind_phys), dimension(nc, nz + 1) :: prsi, ghti
  real(kind=kind_phys), dimension(nc, nz)     :: zl, dot
  real(kind=kind_phys), dimension(nc, nz + 1) :: zi
  real(kind=kind_phys), dimension(nc, nz) :: &
       r_rthcuten, r_rqvcuten, r_rqccuten, r_rqicuten, r_rucuten, r_rvcuten
  real(kind=kind_phys), dimension(nc) :: r_raincv, r_pratec

  real, dimension(nz)     :: b_t, b_qv, b_qc, b_qi, b_u, b_v
  real, dimension(nz)     :: b_pcps, b_dz, b_rho, b_pi, b_qvften, b_thften
  real, dimension(nz + 1) :: b_p8w, b_w
  real :: b_xland, b_hfx, b_qfx

  real(kind=kind_phys), parameter :: dtc = 60.0_kind_phys
  integer, parameter :: stepcu = 1, itimestep = 2
  logical, parameter :: p_f_qv=.true., p_f_qc=.true., p_f_qr=.true.
  logical, parameter :: p_f_qi=.true., p_f_qs=.true.

  real(kind=kind_phys) :: delt, rdelt
  character(len=256) :: errmsg
  integer :: errflg, n, m, k, kk, pp, zz, ndiff, total_bad
  real :: dxv

  if (kind(1.0_kind_phys) /= 4) then
     write(*,'(a)') 'FATAL: kind_phys is not 4 bytes.  Need -DRWORDSIZE=4.'
     stop 3
  end if

  call nt_build_case_table()
  total_bad = 0
  r_bad = 0
  r_fieldbad = 0
  p_bad = 0
  p_fieldbad = 0

  open(unit=21, file='nt-prep-levels.csv',      status='replace')
  open(unit=22, file='nt-prep-surface.csv',     status='replace')
  open(unit=23, file='nt-prep-consistency.csv', status='replace')
  ! The prep fixture carries its own INPUTS as well as its outputs, so a
  ! port can be graded without rebuilding nt_cases in another language.
  ! Rows run k = 1..nz+1: p8w and w are interface arrays and the (nz+1)th
  ! entry is real data the prep reads (dot[nz] needs w(nz+1)); the
  ! half-level columns are written as zero there and must not be read.
  open(unit=24, file='nt-prep-input.csv', status='replace')
  open(unit=25, file='nt-post-in-levels.csv',   status='replace')
  open(unit=26, file='nt-post-in-surface.csv',  status='replace')
  open(unit=27, file='nt-post-out-levels.csv',  status='replace')
  open(unit=28, file='nt-post-out-surface.csv', status='replace')

  write(21,'(a)') 'case,dx,k,prsl,ghtl,omg,tf,qvf,qcf,qif,uf,vf,' // &
       'qvftenz,thftenz,prsi,ghti'
  write(22,'(a)') 'case,dx,slimsk,dx_hv,hfx_hv,qfx_hv,delt'
  write(23,'(a)') 'case,dx,differing_words'
  ! post_run reads from TWO conventions at once and the row carries both.
  ! exner/qv/qc/qi/t/u/v are WRF order (k = 1 is the SURFACE) -- they are
  ! the driver's untouched inputs, the reference state the tendency is
  ! measured against.  tf/qvf/qcf/qif/uf/vf are SCHEME order (k = 1 is the
  ! model TOP) and carry cu_ntiedtke_run's answer.  post_run pairs them by
  ! flipping (`tf(i,zz)`, zz = kte - pp), so each array is recorded at its
  ! OWN index here and the flip stays the port's job, exactly as it is the
  ! routine's job.  Pairing them in the fixture would hide a flip error in
  ! the one capture built to expose it.
  write(25,'(a)') 'case,dx,k,exner,qv,qc,qi,t,u,v,tf,qvf,qcf,qif,uf,vf'
  write(26,'(a)') 'case,dx,stepcu,dt,rn'
  write(27,'(a)') 'case,dx,k,rthcuten,rqvcuten,rqccuten,rqicuten,' // &
       'rucuten,rvcuten'
  write(28,'(a)') 'case,dx,raincv,pratec'
  write(24,'(a)') 'case,dx,k,t3d,qv3d,qc3d,qi3d,u3d,v3d,pcps,dz8w,' // &
       'rho3d,pi3d,qvften,thften,p8w,w'

  do m = 1, ndx
    dxv = nt_dxsweep(m)

    do n = 1, nc
      call nt_build_column(n, nz, b_t, b_qv, b_qc, b_qi, b_u, b_v, &
                           b_pcps, b_p8w, b_dz, b_rho, b_pi, b_w, &
                           b_qvften, b_thften, b_xland, b_hfx, b_qfx)
      do k = 1, nz
        t3d(n,k,1)=b_t(k);    qv3d(n,k,1)=b_qv(k);  qc3d(n,k,1)=b_qc(k)
        qi3d(n,k,1)=b_qi(k);  u3d(n,k,1)=b_u(k);    v3d(n,k,1)=b_v(k)
        pcps(n,k,1)=b_pcps(k);dz8w(n,k,1)=b_dz(k);  rho3d(n,k,1)=b_rho(k)
        pi3d(n,k,1)=b_pi(k)
        qvften(n,k,1)=b_qvften(k); thften(n,k,1)=b_thften(k)
      end do
      do k = 1, nz + 1
        p8w(n,k,1)=b_p8w(k);  w(n,k,1)=b_w(k)
      end do
      xland(n,1)=b_xland; hfx(n,1)=b_hfx; qfx(n,1)=b_qfx; dx(n,1)=dxv
    end do

    ! ================= the control: the real driver ======================
    raincv=0.; pratec=0.; cu_act_flag=.false.
    rthcuten=0.; rqvcuten=0.; rqccuten=0.; rqicuten=0.; rucuten=0.; rvcuten=0.
    call cu_ntiedtke_driver( &
         dt=dtc, itimestep=itimestep, stepcu=stepcu, hfx=hfx, &
         raincv=raincv, pratec=pratec, qfx=qfx, &
         u3d=u3d, v3d=v3d, w=w, t3d=t3d, pi3d=pi3d, rho3d=rho3d, &
         qv3d=qv3d, qc3d=qc3d, qi3d=qi3d, &
         dz8w=dz8w, pcps=pcps, p8w=p8w, xland=xland, dx=dx, &
         cu_act_flag=cu_act_flag, &
         ids=1, ide=nc+1, jds=1, jde=2, kds=1, kde=nz+1, &
         ims=1, ime=nc, jms=1, jme=1, kms=1, kme=nz+1, &
         its=its, ite=ite, jts=1, jte=1, kts=kts, kte=kte, &
         qvften=qvften, thften=thften, &
         f_qv=p_f_qv, f_qc=p_f_qc, f_qr=p_f_qr, f_qi=p_f_qi, f_qs=p_f_qs, &
         rthcuten=rthcuten, rqvcuten=rqvcuten, &
         rqccuten=rqccuten, rqicuten=rqicuten, &
         rucuten=rucuten, rvcuten=rvcuten, &
         grav=real(nt_g,kind_phys), xlf=real(nt_xlf,kind_phys), &
         xls=real(nt_xls,kind_phys), xlv=real(nt_xlv,kind_phys), &
         rd=real(nt_rd,kind_phys), rv=real(nt_rv,kind_phys), &
         cp=real(nt_cp,kind_phys), errmsg=errmsg, errflg=errflg)

    ! ============ the replication: pre_run, verbatim =====================
    ! module_cu_ntiedtke.F:391-455.  Every statement in source order; the
    ! whole value of this file is that it is a transcription and not a
    ! paraphrase.
    call cu_ntiedtke_init(real(nt_cp,kind_phys), real(nt_rd,kind_phys), &
                          real(nt_rv,kind_phys), real(nt_xlv,kind_phys), &
                          real(nt_xls,kind_phys), real(nt_xlf,kind_phys), &
                          real(nt_g,kind_phys), errmsg, errflg)

    delt = dtc * stepcu
    do n = its, ite
       slimsk(n) = (abs(xland(n,1) - 2.))
       dx_hv(n)  = dx(n,1)
       hfx_hv(n) = hfx(n,1)
       qfx_hv(n) = qfx(n,1)
    end do

    do n = its, ite
       zi(n, kts) = 0.
    end do
    do k = kts, kte
       do n = its, ite
          zi(n, k+1) = zi(n, k) + dz8w(n, k, 1)
       end do
    end do
    do k = kts, kte
       do n = its, ite
          zl(n, k)  = 0.5 * (zi(n, k) + zi(n, k+1))
          dot(n, k) = -0.5 * real(nt_g, kind_phys) * rho3d(n, k, 1) &
                      * (w(n, k, 1) + w(n, k+1, 1))
       end do
    end do

    pp = 0
    do k = kts, kte + 1
       zz = kte + 1 - pp
       do n = its, ite
          ghti(n, zz) = zi(n, k)
          prsi(n, zz) = p8w(n, k, 1)
       end do
       pp = pp + 1
    end do
    pp = 0
    do k = kts, kte
       zz = kte - pp
       do n = its, ite
          ghtl(n, zz) = zl(n, k)
          omg(n, zz)  = dot(n, k)
          prsl(n, zz) = pcps(n, k, 1)
       end do
       pp = pp + 1
    end do
    pp = 0
    do k = kts, kte
       zz = kte - pp
       do n = its, ite
          tf(n, zz)  = t3d(n, k, 1)
          qvf(n, zz) = qv3d(n, k, 1)
          qcf(n, zz) = qc3d(n, k, 1)
          qif(n, zz) = qi3d(n, k, 1)
          uf(n, zz)  = u3d(n, k, 1)
          vf(n, zz)  = v3d(n, k, 1)
       end do
       pp = pp + 1
    end do
    if (itimestep == 1) then
       qvftenz = 0.;  thftenz = 0.
    else
       pp = 0
       do k = kts, kte
          zz = kte - pp
          do n = its, ite
             qvftenz(n, zz) = qvften(n, k, 1)
             thftenz(n, zz) = thften(n, k, 1)
          end do
          pp = pp + 1
       end do
    end if

    ! ======== the direct proof: the REAL cu_ntiedtke_pre_run =============
    ! Everything above this line is a transcription.  Until build.sh
    ! globalized the symbol, a transcription was all that was AVAILABLE --
    ! cu_ntiedtke_pre_run is private, so the replication could only be
    ! proved by CONVERGENCE, the whole pipeline landing on the driver's
    ! answer 200 lines further down.  docs/ntiedtke/PORT-RECORD.md section 8 states
    ! what that is worth: converging proves no intermediate.
    !
    ! That was the weakest evidence in the port attached to the thing most
    ! able to be quietly wrong.  Section 2: the flip's failure mode is
    ! SILENT -- an upside-down column is finite, plausible and entirely
    ! wrong -- and it is the one structural inversion the whole port rests
    ! on.  Below, every field the replication produces is compared bitwise
    ! against the real routine's own output, on every case and every dx.
    !
    ! The poison values matter: a field the real routine never writes stays
    ! at -1 and is caught, so "agrees" cannot be produced by absence.
    r_im = -1; r_kx = -1; r_kx1 = -1
    r_delt = -1.0_kind_phys
    r_slimsk = -1
    r_prsl = -1.; r_ghtl = -1.; r_omg = -1.
    r_tf = -1.; r_qvf = -1.; r_qcf = -1.; r_qif = -1.
    r_uf = -1.; r_vf = -1.; r_qvftenz = -1.; r_thftenz = -1.
    r_prsi = -1.; r_ghti = -1.
    r_errmsg = 'x'
    r_errflg = -1

    call x_pre_run(its, ite, kts, kte, r_im, r_kx, r_kx1, itimestep, &
         stepcu, dtc, real(nt_g,kind_phys), xland, dz8w, pcps, p8w, &
         t3d, rho3d, qv3d, qc3d, qi3d, u3d, v3d, w, qvften, thften, &
         r_qvftenz, r_thftenz, r_slimsk, r_delt, r_prsl, r_ghtl, &
         r_tf, r_qvf, r_qcf, r_qif, r_uf, r_vf, r_prsi, r_ghti, &
         r_omg, r_errmsg, r_errflg, int(len(r_errmsg_s), kind=c_size_t))

    do jj = 1, len(r_errmsg_s)
       r_errmsg_s(jj:jj) = r_errmsg(jj)
    end do
    ! THE HIDDEN-LENGTH ABI, CHECKED.  errmsg is character(len=*) in the
    ! callee, so gfortran passes its length as a hidden trailing size_t,
    ! which BIND(C) cannot express and which is therefore the ONE thing in
    ! this interface that is assumed rather than transcribed.  pre_run ends
    ! by writing a fixed string, so a wrong length shows up here instead of
    ! silently corrupting a stack slot.
    if (r_errflg /= 0 .or. &
        trim(r_errmsg_s) /= 'cu_ntiedtke_pre_run OK') then
       write(*,'(a)') 'FATAL: the real cu_ntiedtke_pre_run reported: ' // &
            trim(r_errmsg_s)
       stop 9
    end if
    if (r_im /= nc .or. r_kx /= nz .or. r_kx1 /= nz + 1) then
       write(*,'(a,3(1x,i0))') 'FATAL: pre_run disagrees on shape:', &
            r_im, r_kx, r_kx1
       stop 9
    end if

    if (wne(r_delt, delt)) r_fieldbad(1) = r_fieldbad(1) + 1
    do n = 1, nc
       if (r_slimsk(n) /= slimsk(n)) r_fieldbad(2) = r_fieldbad(2) + 1
       do k = 1, nz
          if (wne(r_prsl(n,k),    prsl(n,k)))    r_fieldbad(3)  = r_fieldbad(3)  + 1
          if (wne(r_ghtl(n,k),    ghtl(n,k)))    r_fieldbad(4)  = r_fieldbad(4)  + 1
          if (wne(r_omg(n,k),     omg(n,k)))     r_fieldbad(5)  = r_fieldbad(5)  + 1
          if (wne(r_tf(n,k),      tf(n,k)))      r_fieldbad(6)  = r_fieldbad(6)  + 1
          if (wne(r_qvf(n,k),     qvf(n,k)))     r_fieldbad(7)  = r_fieldbad(7)  + 1
          if (wne(r_qcf(n,k),     qcf(n,k)))     r_fieldbad(8)  = r_fieldbad(8)  + 1
          if (wne(r_qif(n,k),     qif(n,k)))     r_fieldbad(9)  = r_fieldbad(9)  + 1
          if (wne(r_uf(n,k),      uf(n,k)))      r_fieldbad(10) = r_fieldbad(10) + 1
          if (wne(r_vf(n,k),      vf(n,k)))      r_fieldbad(11) = r_fieldbad(11) + 1
          if (wne(r_qvftenz(n,k), qvftenz(n,k))) r_fieldbad(12) = r_fieldbad(12) + 1
          if (wne(r_thftenz(n,k), thftenz(n,k))) r_fieldbad(13) = r_fieldbad(13) + 1
       end do
       do k = 1, nz + 1
          if (wne(r_prsi(n,k), prsi(n,k))) r_fieldbad(14) = r_fieldbad(14) + 1
          if (wne(r_ghti(n,k), ghti(n,k))) r_fieldbad(15) = r_fieldbad(15) + 1
       end do
    end do
    r_bad = sum(r_fieldbad)

    ! CHECKED HERE, not at the end of the program, and deliberately BEFORE
    ! the convergence gate.  A failure of the replication reaches both, and
    ! the direct gate is the one that says which field -- reporting the
    ! composite first would bury it.
    if (r_bad /= 0) then
       write(*,'(a)') 'FATAL: the pre_run replication differs from the ' // &
            'real cu_ntiedtke_pre_run.  Differing words by field:'
       do jj = 1, nfield
          if (r_fieldbad(jj) /= 0) &
               write(*,'(a,a,a,i0)') '  ', r_fieldname(jj), ': ', &
               r_fieldbad(jj)
       end do
       stop 9
    end if

    ! ---- capture, BEFORE cu_ntiedtke_run overwrites the state ------------
    ! pu/pv/pt/pqv/pqc/pqi are intent(inout): the scheme updates them in
    ! place, so a capture taken after the call is the ANSWER, not the input.
    do n = 1, nc
      do k = 1, nz
        write(21,'(i0,",",f0.1,",",i0,13(",",a))') n, dxv, k, &
             hexw(prsl(n,k)), hexw(ghtl(n,k)), hexw(omg(n,k)), &
             hexw(tf(n,k)),   hexw(qvf(n,k)),  hexw(qcf(n,k)), &
             hexw(qif(n,k)),  hexw(uf(n,k)),   hexw(vf(n,k)),  &
             hexw(qvftenz(n,k)), hexw(thftenz(n,k)), &
             hexw(prsi(n,k)), hexw(ghti(n,k))
      end do
      write(22,'(i0,",",f0.1,",",i0,4(",",a))') n, dxv, slimsk(n), &
           hexw(dx_hv(n)), hexw(hfx_hv(n)), hexw(qfx_hv(n)), hexw(delt)
      do k = 1, nz + 1
        if (k <= nz) then
          write(24,'(i0,",",f0.1,",",i0,14(",",a))') n, dxv, k, &
               hexw(t3d(n,k,1)),  hexw(qv3d(n,k,1)), hexw(qc3d(n,k,1)), &
               hexw(qi3d(n,k,1)), hexw(u3d(n,k,1)),  hexw(v3d(n,k,1)),  &
               hexw(pcps(n,k,1)), hexw(dz8w(n,k,1)), hexw(rho3d(n,k,1)),&
               hexw(pi3d(n,k,1)), hexw(qvften(n,k,1)), &
               hexw(thften(n,k,1)), hexw(p8w(n,k,1)), hexw(w(n,k,1))
        else
          write(24,'(i0,",",f0.1,",",i0,14(",",a))') n, dxv, k, &
               hexw(0.0_kind_phys), hexw(0.0_kind_phys), &
               hexw(0.0_kind_phys), hexw(0.0_kind_phys), &
               hexw(0.0_kind_phys), hexw(0.0_kind_phys), &
               hexw(0.0_kind_phys), hexw(0.0_kind_phys), &
               hexw(0.0_kind_phys), hexw(0.0_kind_phys), &
               hexw(0.0_kind_phys), hexw(0.0_kind_phys), &
               hexw(p8w(n,k,1)), hexw(w(n,k,1))
        end if
      end do
    end do

    ! ============ the real public entry point ============================
    call cu_ntiedtke_run( &
         pu=uf, pv=vf, pt=tf, pqv=qvf, pqc=qcf, pqi=qif, &
         pqvf=qvftenz, ptf=thftenz, poz=ghtl, pzz=ghti, pomg=omg, &
         pap=prsl, paph=prsi, evap=qfx_hv, hfx=hfx_hv, zprecc=rn, &
         lndj=slimsk, lq=nc, km=nz, km1=nz+1, dt=delt, dx=dx_hv, &
         errmsg=errmsg, errflg=errflg)

    ! ============ post_run, verbatim (module_cu_ntiedtke.F:504-524) ======
    rdelt = 1. / delt
    do n = its, ite
       r_raincv(n) = rn(n) / stepcu
       r_pratec(n) = rn(n) / (stepcu * dtc)
    end do
    pp = 0
    do k = kts, kte
       zz = kte - pp
       do n = its, ite
          r_rthcuten(n,k) = (tf(n,zz)-t3d(n,k,1))/pi3d(n,k,1)*rdelt
          r_rqvcuten(n,k) = (qvf(n,zz)-qv3d(n,k,1))*rdelt
          r_rqccuten(n,k) = (qcf(n,zz)-qc3d(n,k,1))*rdelt
          r_rqicuten(n,k) = (qif(n,zz)-qi3d(n,k,1))*rdelt
          r_rucuten(n,k)  = (uf(n,zz)-u3d(n,k,1))*rdelt
          r_rvcuten(n,k)  = (vf(n,zz)-v3d(n,k,1))*rdelt
       end do
       pp = pp + 1
    end do

    ! ======== the direct proof: the REAL cu_ntiedtke_post_run ============
    ! post_run forms all eight tendency fields of nt-levels.csv, and until
    ! the globalize loop took it, the twenty lines above were the only
    ! statement of what it does.  Nobody CHOSE that -- the symbol was
    ! private, so replication was the only option and the convergence
    ! caveat rode along unremarked.
    ! ---- capture at post_run's OWN entry, before it runs ---------------
    ! Every value at the point THIS routine reads it.  t3d/qv3d/... are
    ! also in nt-prep-input.csv and are provably untouched by the scheme,
    ! and that is precisely the "a neighbour's capture will do" reasoning
    ! that has been wrong eight times.  Recording them again costs six
    ! columns and removes the argument.
    do n = 1, nc
      do k = 1, nz
        write(25,'(i0,",",f0.1,",",i0,13(",",a))') n, dxv, k, &
             hexw(pi3d(n,k,1)), hexw(qv3d(n,k,1)), hexw(qc3d(n,k,1)), &
             hexw(qi3d(n,k,1)), hexw(t3d(n,k,1)),  hexw(u3d(n,k,1)),  &
             hexw(v3d(n,k,1)), &
             hexw(tf(n,k)),  hexw(qvf(n,k)), hexw(qcf(n,k)), &
             hexw(qif(n,k)), hexw(uf(n,k)),  hexw(vf(n,k))
      end do
      write(26,'(i0,",",f0.1,",",i0,2(",",a))') n, dxv, stepcu, &
           hexw(dtc), hexw(rn(n))
    end do

    p_raincv = -1.; p_pratec = -1.
    p_rthcuten = -1.; p_rqvcuten = -1.; p_rqccuten = -1.
    p_rqicuten = -1.; p_rucuten = -1.; p_rvcuten = -1.
    r_errmsg = 'x'
    r_errflg = -1
    call x_post_run(its, ite, kts, kte, stepcu, dtc, pi3d, qv3d, qc3d, &
         qi3d, t3d, u3d, v3d, qvf, qcf, qif, tf, uf, vf, rn, &
         p_raincv, p_pratec, p_rthcuten, p_rqvcuten, p_rqccuten, &
         p_rqicuten, p_rucuten, p_rvcuten, r_errmsg, r_errflg, &
         int(len(r_errmsg_s), kind=c_size_t))
    do jj = 1, len(r_errmsg_s)
       r_errmsg_s(jj:jj) = r_errmsg(jj)
    end do
    ! Note the string: post_run signs itself 'cu_ntiedtke_timestep_final',
    ! not 'cu_ntiedtke_post_run' (:526).  Transcribed from the routine, not
    ! guessed from its name -- guessing here would have produced a gate
    ! that fails on a correct call.
    if (r_errflg /= 0 .or. &
        trim(r_errmsg_s) /= 'cu_ntiedtke_timestep_final OK') then
       write(*,'(a)') 'FATAL: the real cu_ntiedtke_post_run reported: ' // &
            trim(r_errmsg_s)
       stop 9
    end if

    ! ---- and post_run's own outputs, from the REAL routine -------------
    do n = 1, nc
      do k = 1, nz
        write(27,'(i0,",",f0.1,",",i0,6(",",a))') n, dxv, k, &
             hexw(p_rthcuten(n,k)), hexw(p_rqvcuten(n,k)), &
             hexw(p_rqccuten(n,k)), hexw(p_rqicuten(n,k)), &
             hexw(p_rucuten(n,k)),  hexw(p_rvcuten(n,k))
      end do
      write(28,'(i0,",",f0.1,2(",",a))') n, dxv, &
           hexw(p_raincv(n)), hexw(p_pratec(n))
    end do

    do n = 1, nc
       if (wne(p_raincv(n), r_raincv(n))) p_fieldbad(1) = p_fieldbad(1) + 1
       if (wne(p_pratec(n), r_pratec(n))) p_fieldbad(2) = p_fieldbad(2) + 1
       do k = 1, nz
          if (wne(p_rthcuten(n,k), r_rthcuten(n,k))) p_fieldbad(3) = p_fieldbad(3) + 1
          if (wne(p_rqvcuten(n,k), r_rqvcuten(n,k))) p_fieldbad(4) = p_fieldbad(4) + 1
          if (wne(p_rqccuten(n,k), r_rqccuten(n,k))) p_fieldbad(5) = p_fieldbad(5) + 1
          if (wne(p_rqicuten(n,k), r_rqicuten(n,k))) p_fieldbad(6) = p_fieldbad(6) + 1
          if (wne(p_rucuten(n,k),  r_rucuten(n,k)))  p_fieldbad(7) = p_fieldbad(7) + 1
          if (wne(p_rvcuten(n,k),  r_rvcuten(n,k)))  p_fieldbad(8) = p_fieldbad(8) + 1
       end do
    end do
    p_bad = sum(p_fieldbad)
    if (p_bad /= 0) then
       write(*,'(a)') 'FATAL: the post_run replication differs from the ' // &
            'real cu_ntiedtke_post_run.  Differing words by field:'
       do jj = 1, size(p_fieldbad)
          if (p_fieldbad(jj) /= 0) &
               write(*,'(a,a,a,i0)') '  ', p_fieldname(jj), ': ', &
               p_fieldbad(jj)
       end do
       stop 9
    end if

    ! ============ the proof ==============================================
    do n = 1, nc
      ndiff = 0
      do k = 1, nz
        if (wne(r_rthcuten(n,k), rthcuten(n,k,1))) ndiff = ndiff + 1
        if (wne(r_rqvcuten(n,k), rqvcuten(n,k,1))) ndiff = ndiff + 1
        if (wne(r_rqccuten(n,k), rqccuten(n,k,1))) ndiff = ndiff + 1
        if (wne(r_rqicuten(n,k), rqicuten(n,k,1))) ndiff = ndiff + 1
        if (wne(r_rucuten(n,k),  rucuten(n,k,1)))  ndiff = ndiff + 1
        if (wne(r_rvcuten(n,k),  rvcuten(n,k,1)))  ndiff = ndiff + 1
      end do
      if (wne(r_raincv(n), raincv(n,1))) ndiff = ndiff + 1
      if (wne(r_pratec(n), pratec(n,1))) ndiff = ndiff + 1
      total_bad = total_bad + ndiff
      write(23,'(i0,",",f0.1,",",i0)') n, dxv, ndiff
    end do
  end do

  close(21); close(22); close(23); close(24)
  close(25); close(26); close(27); close(28)
  if (total_bad /= 0) then
     write(*,'(a,i0,a)') 'FATAL: the prep replication differs from the ', &
          total_bad, ' driver words; the capture is not a decomposition.'
     stop 7
  end if
  write(*,'(a)') 'run_nt_prep OK -- 0 differing words; pre_run AND ' // &
       'post_run are graded DIRECTLY against the real routines'

end program run_nt_prep
