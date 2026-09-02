program run_nt_live
  ! THE LIVE-COLUMN ARM: drive the byte-unmodified WRF cu_ntiedtke_driver
  ! over columns taken from a RUNNING ArWen forecast, not from the analytic
  ! fixture.
  !
  ! WHY THIS EXISTS.  run_cu_ntiedtke.F90 pins the same driver over 18
  ! SYNTHESISED columns -- a lapse-rate troposphere with a three-point RH
  ! profile, at nz = 49 -- and the port grades against it at max_ulp == 0.
  ! The live run is nz = 61 with a near-saturated hurricane core.  That
  ! generator cannot produce this column, and one branch is already known
  ! dead under it: the cumastrn:566 deep->shallow demotion never fires,
  ! because the generated RH profile is monotone.  So max_ulp == 0 over
  ! that fixture is a receipt about soundings that exclude the case that
  ! matters, and this program is how the case that matters gets graded.
  !
  ! THE CONTROL.  Comparing ArWen's state at forecast hour N against WRF's
  ! state at forecast hour N is confounded -- the two runs have diverged by
  ! then, so their inputs legitimately differ and neither agreement nor
  ! disagreement carries information.  This holds the INPUT fixed and
  ! varies only the implementation: same IEEE-754 words in, both codes,
  ! one diff.
  !
  !   run_nt_live <prefix>
  !
  ! reads  <prefix>-meta.txt, <prefix>-lev.csv, <prefix>-iface.csv,
  !        <prefix>-sfc.csv
  ! writes <prefix>-wrf-lev.csv, <prefix>-wrf-sfc.csv
  !
  ! Every float crosses in and out as its raw IEEE-754 word in hex, the
  ! same convention nt_cases.F90's hexw uses, because a decimal rendering
  ! is a lossy view of the thing being compared.
  !
  ! EVERYTHING IS IN WRF ORDER -- k = 1 is the surface.  cu_ntiedtke_pre_run
  ! flips it for the scheme, exactly as it does for the live adapter, so a
  ! dump written upside down fails here rather than producing a plausible
  ! inverted answer.  The producing probe records surface_first in the meta
  ! file and this program refuses a dump that says otherwise.
  use iso_fortran_env, only: int32
  use ccpp_kind_types, only: kind_phys
  use module_cu_ntiedtke, only: cu_ntiedtke_driver
  use nt_cases, only: hexw, nt_g, nt_rd, nt_rv, nt_cp, nt_xlv, nt_xlf, nt_xls
  implicit none

  integer, parameter :: NLEV_IN = 12, NIFACE_IN = 2, NSFC_IN = 3
  integer, parameter :: NLEV_OUT = 6

  character(len=512) :: prefix, path
  character(len=4096) :: line
  character(len=64) :: key, val
  character(len=16) :: fld(0:32)

  integer :: nc, nz, nz1, stepcu, itimestep, surface_first
  real(kind=kind_phys) :: dt

  real(kind=kind_phys), allocatable, dimension(:,:,:) :: &
       u3d, v3d, w, t3d, qv3d, qc3d, qi3d, pi3d, rho3d, &
       dz8w, pcps, p8w, qvften, thften, &
       rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten
  real(kind=kind_phys), allocatable, dimension(:,:) :: &
       raincv, pratec, qfx, hfx, xland, dx
  logical, allocatable, dimension(:,:) :: cu_act_flag

  ! module_cu_ntiedtke.F:253-254 and :263-264 guard on the array being
  ! present and then dereference f_qc with no present() check of its own,
  ! so omitting a flag while passing its array segfaults.  All five true is
  ! what module_cumulus_driver.F:1402-1403 hands an mp_physics = 8 run.
  logical, parameter :: p_f_qv = .true., p_f_qc = .true., p_f_qr = .true.
  logical, parameter :: p_f_qi = .true., p_f_qs = .true.

  character(len=256) :: errmsg
  integer :: errflg, i, k, c, u, nf, ios

  if (command_argument_count() /= 1) then
     write(*,'(a)') 'usage: run_nt_live <prefix>'
     stop 2
  end if
  call get_command_argument(1, prefix)

  ! kind_phys must be single.  Same guard the other harnesses carry: a build
  ! that misses -DRWORDSIZE=4 takes ccpp_kind_types.F's DOUBLE branch,
  ! compiles clean, links clean, and writes a plausible double-precision
  ! answer against which a correct float32 port fails every bitwise gate.
  if (kind(1.0_kind_phys) /= 4) then
     write(*,'(a)') 'FATAL: kind_phys is not single precision'
     stop 3
  end if

  ! ---- meta -----------------------------------------------------------
  nc = -1; nz = -1; stepcu = -1; itimestep = -1; surface_first = -1
  dt = -1.0_kind_phys
  path = trim(prefix)//'-meta.txt'
  open(newunit=u, file=trim(path), status='old', action='read')
  do
     read(u,'(a)',iostat=ios) line
     if (ios /= 0) exit
     read(line,*,iostat=ios) key, val
     if (ios /= 0) cycle
     select case (trim(key))
     case ('ncol_selected');  read(val,*) nc
     case ('nz');             read(val,*) nz
     case ('scheme_dt');      read(val,*) dt
     case ('stepcu');         read(val,*) stepcu
     case ('itimestep');      read(val,*) itimestep
     case ('surface_first');  read(val,*) surface_first
     end select
  end do
  close(u)

  if (nc <= 0 .or. nz <= 0 .or. stepcu <= 0 .or. itimestep <= 0 &
       .or. dt <= 0.0_kind_phys) then
     write(*,'(a)') 'FATAL: meta file is incomplete'
     stop 4
  end if
  ! THE ORIENTATION GATE.  Not a comment: a dump written top-down would
  ! otherwise run cleanly and disagree spectacularly, and that failure
  ! would read as a port defect.
  if (surface_first /= 1) then
     write(*,'(a,i0)') 'FATAL: dump is not surface-first, surface_first=', &
          surface_first
     stop 5
  end if
  nz1 = nz + 1

  allocate(u3d(nc,nz1,1), v3d(nc,nz1,1), w(nc,nz1,1), t3d(nc,nz1,1), &
       qv3d(nc,nz1,1), qc3d(nc,nz1,1), qi3d(nc,nz1,1), pi3d(nc,nz1,1), &
       rho3d(nc,nz1,1), dz8w(nc,nz1,1), pcps(nc,nz1,1), p8w(nc,nz1,1), &
       qvften(nc,nz1,1), thften(nc,nz1,1), &
       rthcuten(nc,nz1,1), rqvcuten(nc,nz1,1), rqccuten(nc,nz1,1), &
       rqicuten(nc,nz1,1), rucuten(nc,nz1,1), rvcuten(nc,nz1,1))
  allocate(raincv(nc,1), pratec(nc,1), qfx(nc,1), hfx(nc,1), xland(nc,1), &
       dx(nc,1), cu_act_flag(nc,1))

  ! Zeroed whole, including the kme slot the driver never writes, so an
  ! unread word is a deterministic 0 rather than whatever the allocator
  ! handed back.
  u3d = 0.0; v3d = 0.0; w = 0.0; t3d = 0.0; qv3d = 0.0; qc3d = 0.0
  qi3d = 0.0; pi3d = 0.0; rho3d = 0.0; dz8w = 0.0; pcps = 0.0; p8w = 0.0
  qvften = 0.0; thften = 0.0
  rthcuten = 0.0; rqvcuten = 0.0; rqccuten = 0.0; rqicuten = 0.0
  rucuten = 0.0; rvcuten = 0.0
  raincv = 0.0; pratec = 0.0; cu_act_flag = .false.

  ! ---- level inputs ---------------------------------------------------
  ! col,k,t3d,qv3d,qc3d,qi3d,u3d,v3d,pcps,dz8w,rho3d,exner,qvften,thften
  path = trim(prefix)//'-lev.csv'
  open(newunit=u, file=trim(path), status='old', action='read')
  read(u,'(a)') line
  ! THE HEADER IS CHECKED, NOT SKIPPED.  This program hardcodes field
  ! positions while the producing probe writes by name, so a reordered
  ! column would be read as a different variable -- qi where qc belongs --
  ! and would surface as a spectacular fake port defect rather than as the
  ! plumbing error it is.  The two ends agree by assertion or not at all.
  call want_header(line, &
       'col,k,t3d,qv3d,qc3d,qi3d,u3d,v3d,pcps,dz8w,rho3d,exner,' // &
       'qvften,thften')
  do
     read(u,'(a)',iostat=ios) line
     if (ios /= 0) exit
     call split(line, fld, nf)
     if (nf /= NLEV_IN + 2) then
        write(*,'(a,i0)') 'FATAL: lev row has fields: ', nf
        stop 6
     end if
     read(fld(0),*) c
     read(fld(1),*) k
     t3d(c,k,1)    = unhex(fld(2))
     qv3d(c,k,1)   = unhex(fld(3))
     qc3d(c,k,1)   = unhex(fld(4))
     qi3d(c,k,1)   = unhex(fld(5))
     u3d(c,k,1)    = unhex(fld(6))
     v3d(c,k,1)    = unhex(fld(7))
     pcps(c,k,1)   = unhex(fld(8))
     dz8w(c,k,1)   = unhex(fld(9))
     rho3d(c,k,1)  = unhex(fld(10))
     pi3d(c,k,1)   = unhex(fld(11))
     qvften(c,k,1) = unhex(fld(12))
     thften(c,k,1) = unhex(fld(13))
  end do
  close(u)

  ! ---- interface inputs -----------------------------------------------
  path = trim(prefix)//'-iface.csv'
  open(newunit=u, file=trim(path), status='old', action='read')
  read(u,'(a)') line
  call want_header(line, 'col,k,p8w,w')
  do
     read(u,'(a)',iostat=ios) line
     if (ios /= 0) exit
     call split(line, fld, nf)
     if (nf /= NIFACE_IN + 2) then
        write(*,'(a,i0)') 'FATAL: iface row has fields: ', nf
        stop 6
     end if
     read(fld(0),*) c
     read(fld(1),*) k
     p8w(c,k,1) = unhex(fld(2))
     w(c,k,1)   = unhex(fld(3))
  end do
  close(u)

  ! ---- surface inputs --------------------------------------------------
  path = trim(prefix)//'-sfc.csv'
  open(newunit=u, file=trim(path), status='old', action='read')
  read(u,'(a)') line
  call want_header(line, 'col,gridcol,xland,hfx,qfx,dx')
  do
     read(u,'(a)',iostat=ios) line
     if (ios /= 0) exit
     call split(line, fld, nf)
     if (nf /= NSFC_IN + 3) then
        write(*,'(a,i0)') 'FATAL: sfc row has fields: ', nf
        stop 6
     end if
     read(fld(0),*) c
     xland(c,1) = unhex(fld(2))
     hfx(c,1)   = unhex(fld(3))
     qfx(c,1)   = unhex(fld(4))
     dx(c,1)    = unhex(fld(5))
  end do
  close(u)

  ! ---- echo what was parsed -------------------------------------------
  ! THE READER'S OWN CONTROL, written BEFORE the driver runs.  The
  ! comparator requires this file to be byte-identical to the dump it was
  ! given.  Without it, a lossy hex path or a dropped row would present as
  ! a disagreement between ArWen and WRF -- which is precisely the claim
  ! this program exists to make, so it must not be able to manufacture one.
  path = trim(prefix)//'-echo-lev.csv'
  open(newunit=u, file=trim(path), status='replace', action='write')
  write(u,'(a)') 'col,k,t3d,qv3d,qc3d,qi3d,u3d,v3d,pcps,dz8w,rho3d,' // &
       'exner,qvften,thften'
  do c = 1, nc
     do k = 1, nz
        write(u,'(i0,",",i0,12(",",a))') c, k, &
             hexw(t3d(c,k,1)),   hexw(qv3d(c,k,1)), hexw(qc3d(c,k,1)), &
             hexw(qi3d(c,k,1)),  hexw(u3d(c,k,1)),  hexw(v3d(c,k,1)),  &
             hexw(pcps(c,k,1)),  hexw(dz8w(c,k,1)), hexw(rho3d(c,k,1)), &
             hexw(pi3d(c,k,1)),  hexw(qvften(c,k,1)), hexw(thften(c,k,1))
     end do
  end do
  close(u)

  path = trim(prefix)//'-echo-iface.csv'
  open(newunit=u, file=trim(path), status='replace', action='write')
  write(u,'(a)') 'col,k,p8w,w'
  do c = 1, nc
     do k = 1, nz1
        write(u,'(i0,",",i0,2(",",a))') c, k, &
             hexw(p8w(c,k,1)), hexw(w(c,k,1))
     end do
  end do
  close(u)

  ! ---- the reference --------------------------------------------------
  ! Constants come from nt_cases so this harness cannot drift from the one
  ! the shipped oracle was built with.  ArWen's adapter uses the same set.
  errflg = 0
  call cu_ntiedtke_driver( &
       dt=dt, itimestep=itimestep, stepcu=stepcu, hfx=hfx, &
       raincv=raincv, pratec=pratec, qfx=qfx, &
       u3d=u3d, v3d=v3d, w=w, t3d=t3d, pi3d=pi3d, rho3d=rho3d, &
       qv3d=qv3d, qc3d=qc3d, qi3d=qi3d, &
       dz8w=dz8w, pcps=pcps, p8w=p8w, xland=xland, dx=dx, &
       cu_act_flag=cu_act_flag, &
       ids=1, ide=nc+1, jds=1, jde=2, kds=1, kde=nz1, &
       ims=1, ime=nc, jms=1, jme=1, kms=1, kme=nz1, &
       its=1, ite=nc, jts=1, jte=1, kts=1, kte=nz, &
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
     stop 7
  end if

  ! ---- record ----------------------------------------------------------
  path = trim(prefix)//'-wrf-lev.csv'
  open(newunit=u, file=trim(path), status='replace', action='write')
  write(u,'(a)') 'col,k,rthcuten,rqvcuten,rqccuten,rqicuten,rucuten,rvcuten'
  do c = 1, nc
     do k = 1, nz
        write(u,'(i0,",",i0,6(",",a))') c, k, &
             hexw(rthcuten(c,k,1)), hexw(rqvcuten(c,k,1)), &
             hexw(rqccuten(c,k,1)), hexw(rqicuten(c,k,1)), &
             hexw(rucuten(c,k,1)),  hexw(rvcuten(c,k,1))
     end do
  end do
  close(u)

  path = trim(prefix)//'-wrf-sfc.csv'
  open(newunit=u, file=trim(path), status='replace', action='write')
  write(u,'(a)') 'col,raincv,pratec,cu_act_flag'
  do c = 1, nc
     write(u,'(i0,2(",",a),",",l1)') c, &
          hexw(raincv(c,1)), hexw(pratec(c,1)), cu_act_flag(c,1)
  end do
  close(u)

  write(*,'(a,i0,a,i0,a)') 'run_nt_live: ', nc, ' columns x ', nz, ' levels'

contains

  ! Refuse a dump whose header is not the one this program's hardcoded
  ! field positions were written against.
  subroutine want_header(got, expect)
    character(len=*), intent(in) :: got, expect
    if (trim(adjustl(got)) /= trim(expect)) then
       write(*,'(a)') 'FATAL: unexpected CSV header'
       write(*,'(a,a)') '  expected: ', trim(expect)
       write(*,'(a,a)') '  got     : ', trim(adjustl(got))
       stop 10
    end if
  end subroutine want_header

  ! Split a comma-separated line into fields 0..nf-1.  Fields here are
  ! either short integers or 8-digit hex words, so 16 characters is ample
  ! and an over-long field is a corrupt dump rather than something to
  ! silently truncate.
  subroutine split(s, f, n)
    character(len=*), intent(in) :: s
    character(len=16), intent(out) :: f(0:)
    integer, intent(out) :: n
    integer :: p, q, L
    L = len_trim(s)
    n = 0
    p = 1
    do while (p <= L)
       q = index(s(p:L), ',')
       if (q == 0) then
          q = L + 1
       else
          q = p + q - 1
       end if
       if (q - p > len(f)) then
          write(*,'(a)') 'FATAL: over-long CSV field'
          stop 8
       end if
       f(n) = s(p:q-1)
       n = n + 1
       p = q + 1
    end do
    ! A trailing comma would mean an empty final field; the writer never
    ! emits one, and treating it as a field would shift every index.
  end subroutine split

  ! The inverse of nt_cases' hexw.  gfortran's Z edit descriptor reads
  ! either case, so the writer's choice of lowercase does not matter here.
  real(kind=kind_phys) function unhex(s)
    character(len=*), intent(in) :: s
    integer(int32) :: wd
    integer :: ios2
    read(s,'(z8)',iostat=ios2) wd
    if (ios2 /= 0) then
       write(*,'(a,a)') 'FATAL: not an 8-digit hex word: ', trim(s)
       stop 9
    end if
    unhex = real(transfer(wd, 1.0_4), kind_phys)
  end function unhex

end program run_nt_live
